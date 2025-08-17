#!/usr/bin/env python
# train_models.py - Script to train recommendation system models
#!/usr/bin/env python
# train_models.py - Script to train recommendation system models

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix
from collections import Counter

def main():
    # --- Path Configuration ---
    PROJECT_ROOT = Path(__file__).parent.parent
    ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
    DATA_DIR = PROJECT_ROOT / "data/processed"
    
    # Input paths
    DATA_FILTERED_PATH = DATA_DIR / "df_filtered.pkl"
    ENCODED_ITEM_PROPS_PATH = DATA_DIR / "item_properties_encoded.pkl"
    
    # Output directories
    EMBEDDINGS_DIR = ARTIFACTS_DIR / "embeddings"
    MODELS_DIR = ARTIFACTS_DIR / "models"
    INDICES_DIR = ARTIFACTS_DIR / "indices"
    
    # Create directories if they don't exist
    for directory in [EMBEDDINGS_DIR, MODELS_DIR, INDICES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # --- Data Loading ---
    print("Loading data...")
    df_filtered = pd.read_pickle(DATA_FILTERED_PATH)
    encoded_item_props = pd.read_pickle(ENCODED_ITEM_PROPS_PATH)
    print(f"Loaded filtered data: {df_filtered.shape}")
    print(f"Loaded encoded item properties: {encoded_item_props.shape}")

    # --- 1. Popularity Model ---
    print("\nBuilding popularity model...")
    popularity = df_filtered['itemid'].value_counts()
    popularity_df = popularity.reset_index()
    popularity_df.columns = ['itemid', 'count']
    popularity_df.to_parquet(MODELS_DIR / "popularity_model.parquet")
    print("Saved popularity model.")

    # --- 2. Content-Based Filtering ---
    print("\nBuilding content-based filtering model...")
    
    # Clean and prepare item properties
    bool_cols = encoded_item_props.select_dtypes(include=['bool']).columns
    encoded_item_props[bool_cols] = encoded_item_props[bool_cols].astype(int)
    
    obj_cols = encoded_item_props.select_dtypes(include=['object']).columns
    encoded_item_props[obj_cols] = encoded_item_props[obj_cols].apply(pd.to_numeric, errors='coerce')
    encoded_item_props = encoded_item_props.fillna(0).select_dtypes(include=[np.number])
    
    print(f"Cleaned encoded_item_props shape: {encoded_item_props.shape}")

    # Create subset of top popular items
    top_items = popularity_df['itemid'].head(10000).tolist()
    subset_items = [i for i in top_items if i in encoded_item_props.index]
    encoded_subset = encoded_item_props.loc[subset_items]
    print(f"Computing similarity on subset: {encoded_subset.shape}")

    # Compute and save similarity matrix
    item_features = encoded_subset.values
    item_sim_matrix = cosine_similarity(item_features)
    
    np.save(EMBEDDINGS_DIR / "item_similarity_sub.npy", item_sim_matrix)
    encoded_subset.index.to_series().to_csv(
        EMBEDDINGS_DIR / "item_similarity_sub_items.csv", 
        index=False, 
        header=False
    )
    print("Saved item similarity matrix and subset items.")

    # --- 3. KNN Collaborative Filtering ---
    print("\nBuilding KNN collaborative filtering model...")
    
    # Prepare sparse matrix
    user_cat = df_filtered['visitorid'].astype('category')
    item_cat = df_filtered['itemid'].astype('category')
    
    user_codes = user_cat.cat.codes.values
    item_codes = item_cat.cat.codes.values
    
    pair_counts = Counter(zip(user_codes, item_codes))
    rows, cols, data = zip(*[(u, i, c) for (u, i), c in pair_counts.items()])
    
    user_item_sparse = coo_matrix(
        (data, (rows, cols)),
        shape=(user_cat.cat.categories.size, item_cat.cat.categories.size)
    )
    
    print(f"Sparse user-item matrix shape: {user_item_sparse.shape}, nnz={user_item_sparse.nnz}")

    # Train and save KNN model
    knn_model = NearestNeighbors(
        metric='cosine', 
        algorithm='brute', 
        n_neighbors=11, 
        n_jobs=-1
    )
    knn_model.fit(user_item_sparse.T)
    
    with open(MODELS_DIR / "knn_model.pkl", "wb") as f:
        pickle.dump(knn_model, f)
    print("Saved KNN model.")

    print("\nAll models trained and saved successfully!")

if __name__ == "__main__":
    main()
    
    
# Run this script to train the models: python src/train_models.py
# This script trains three recommendation models: popularity, content-based filtering, and KNN collaborative filtering.
# It prepares the data, computes item similarities, and saves the models and embeddings in the specified directories.
# Ensure the necessary directories exist before running this script.
# The script also handles data loading, cleaning, and preprocessing to ensure the models are trained on clean data.
# The trained models are saved in the artifacts directory for later use in recommendations.