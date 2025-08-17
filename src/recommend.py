import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
from src.config import (
    POPULARITY_MODEL_PATH,
    KNN_MODEL_PATH,
    CB_SIM_MATRIX_PATH,
    DEFAULT_NUM_RECS
)

class Recommender:
    def __init__(self):
        self.popular_items = []
        self.cb_sim_matrix = None
        self.cb_items = []
        self.knn_model = None
        self.knn_items = []
        self._load_models()
        self._validate_items()
        self._print_item_stats()
    
    def _load_models(self):
        """Load all recommendation models with validation"""
        print("\n===== LOADING MODELS =====")
        try:
            # Load popularity model
            if POPULARITY_MODEL_PATH.exists():
                pop_df = pd.read_parquet(POPULARITY_MODEL_PATH)
                self.popular_items = pop_df['itemid'].astype(str).tolist()
                print(f"✓ Loaded {len(self.popular_items)} popular items")
            
            # Load KNN model
            if KNN_MODEL_PATH.exists():
                with open(KNN_MODEL_PATH, "rb") as f:
                    self.knn_model = pickle.load(f)
                self.knn_items = self.popular_items[:10000] if self.popular_items else []
                print(f"✓ Loaded KNN model with {len(self.knn_items)} items")
            
            # Load content-based model
            if CB_SIM_MATRIX_PATH.exists():
                self.cb_sim_matrix = np.load(CB_SIM_MATRIX_PATH)
                self.cb_items = self.popular_items[:self.cb_sim_matrix.shape[0]] if self.popular_items else []
                print(f"✓ Loaded CB model with {len(self.cb_items)} items")
            
        except Exception as e:
            print(f"! Loading error: {e}")

    def _validate_items(self):
        """Ensure all item lists contain strings and are unique"""
        self.popular_items = list(dict.fromkeys(str(item) for item in self.popular_items))
        self.knn_items = list(dict.fromkeys(str(item) for item in self.knn_items))
        self.cb_items = list(dict.fromkeys(str(item) for item in self.cb_items))

    def _print_item_stats(self):
        """Print detailed item information for debugging"""
        print("\n===== ITEM STATISTICS =====")
        print(f"Total unique items: {len(self.get_all_items())}")
        print(f"Popular items: {len(self.popular_items)}")
        print(f"KNN items: {len(self.knn_items)}")
        print(f"Content-Based items: {len(self.cb_items)}")
        
        # Check overlap between models
        cb_knn_overlap = len(set(self.cb_items) & set(self.knn_items))
        print(f"\nItems in both CB and KNN: {cb_knn_overlap}")
        
        # Print some sample items
        print("\nSample popular items:", self.popular_items[:3])
        print("Sample KNN items:", self.knn_items[:3])
        print("Sample CB items:", self.cb_items[:3])

    def get_all_items(self):
        """Get all available item IDs with validation"""
        items = set()
        items.update(self.popular_items)
        items.update(self.knn_items)
        items.update(self.cb_items)
        return sorted(items)

    def get_verified_examples(self, count=3):
        """Get example items that exist in all models"""
        examples = []
        # Find intersection of items from all models
        common_items = set(self.popular_items) & set(self.knn_items) & set(self.cb_items)
        
        for item in common_items:
            if len(examples) >= count:
                break
            examples.append(f"{item} - Product {item}")
        
        return examples

    def recommend_content_based(self, item_id, top_n=10):
        """Content-based recommendations with validation"""
        try:
            if not self.cb_sim_matrix or not self.cb_items:
                return []
            
            item_id = str(item_id)
            if item_id not in self.cb_items:
                return []
                
            item_idx = self.cb_items.index(item_id)
            sim_scores = self.cb_sim_matrix[item_idx]
            
            # Get top similar items with minimum similarity threshold
            similar_indices = [i for i in np.argsort(-sim_scores) 
                             if sim_scores[i] > 0.3][1:top_n+1]
            return [self.cb_items[i] for i in similar_indices]
            
        except Exception as e:
            print(f"CB recommendation error: {e}")
            return []

    def recommend_knn(self, item_id, top_n=10):
        """KNN recommendations with validation"""
        try:
            if not self.knn_model or not self.knn_items:
                return []
                
            item_id = str(item_id)
            if item_id not in self.knn_items:
                return []
                
            item_idx = self.knn_items.index(item_id)
            distances, indices = self.knn_model.kneighbors(
                [self.knn_model._fit_X[item_idx]], 
                n_neighbors=top_n+1
            )
            
            # Filter by maximum distance threshold
            return [self.knn_items[i] for i in indices.flatten()[1:] 
                   if distances.flatten()[i] < 0.7]
            
        except Exception as e:
            print(f"KNN recommendation error: {e}")
            return []

    def recommend_popularity(self, top_n=10):
        """Popularity fallback recommendations"""
        return self.popular_items[:top_n]