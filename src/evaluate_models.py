import os
from pathlib import Path
import math
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors

# === Paths (relative to notebooks/) ===

# Get the absolute path to the project root (where README.md is located)
PROJECT_ROOT = Path(__file__).parent.parent

# Update all paths to use absolute paths
DATA_FILTERED_PATH = PROJECT_ROOT / "data/processed/df_filtered.pkl"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
EMBEDDINGS_DIR = ARTIFACTS_DIR / "embeddings"

POPULARITY_PATH = MODELS_DIR / "popularity_model.parquet"
KNN_MODEL_PATH = MODELS_DIR / "knn_model.pkl"
CB_SIM_PATH = EMBEDDINGS_DIR / "item_similarity_sub.npy"
CB_ITEMS_PATH = EMBEDDINGS_DIR / "item_similarity_sub_items.csv"
def main():
    # === Load Data and Models ===
    print("Loading filtered interaction data...")
    df_filtered = pd.read_pickle(DATA_FILTERED_PATH)
    print(f"Loaded filtered data with shape: {df_filtered.shape}")

    print("Loading popularity model...")
    popularity_df = pd.read_parquet(POPULARITY_PATH).sort_values("count", ascending=False)
    popular_items = popularity_df["itemid"].tolist()

    print("Loading content-based similarity subset...")
    item_sim_matrix = np.load(CB_SIM_PATH, mmap_mode="r")
    subset_items = pd.read_csv(CB_ITEMS_PATH, header=None).iloc[:, 0].tolist()
    subset_item_to_idx = {item: idx for idx, item in enumerate(subset_items)}
    print(f"Loaded {len(subset_items)} items in content-based similarity subset.")

    print("Loading KNN collaborative filtering model...")
    with open(KNN_MODEL_PATH, "rb") as f:
        knn_model: NearestNeighbors = pickle.load(f)
    print("KNN model loaded.")

    # === Train/Test Split (user-wise) ===
    def train_test_split_userwise(df, test_ratio=0.2, seed=42):
        rng = np.random.default_rng(seed)
        train_parts, test_parts = [], []
        for user_id, group in df.groupby("visitorid"):
            if len(group) < 2:
                train_parts.append(group)
                continue
            test_size = max(1, int(len(group) * test_ratio))
            test_indices = rng.choice(group.index.values, size=test_size, replace=False)
            test_parts.append(group.loc[test_indices])
            train_parts.append(group.drop(test_indices))
        return pd.concat(train_parts), pd.concat(test_parts)

    train_df, test_df = train_test_split_userwise(df_filtered, test_ratio=0.2, seed=42)
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # === Prepare Ground Truth for Evaluation ===
    test_user_items = defaultdict(set)
    for row in test_df.itertuples(index=False):
        test_user_items[row.visitorid].add(row.itemid)

    # === Build Sparse User-Item Interaction Matrix for TRAIN ===
    def build_sparse_user_item(df):
        users_cat = df["visitorid"].astype("category")
        items_cat = df["itemid"].astype("category")
        u_codes = users_cat.cat.codes
        i_codes = items_cat.cat.codes
        counts = (
            pd.DataFrame({"u": u_codes, "i": i_codes})
            .value_counts()
            .reset_index(name="c")
        )
        n_users = users_cat.cat.categories.size
        n_items = items_cat.cat.categories.size

        mat = coo_matrix(
            (counts["c"].values, (counts["u"].values, counts["i"].values)),
            shape=(n_users, n_items),
        ).tocsr()

        user_idx_to_id = users_cat.cat.categories
        item_idx_to_id = items_cat.cat.categories
        user_id_to_idx = {uid: i for i, uid in enumerate(user_idx_to_id)}
        item_id_to_idx = {iid: i for i, iid in enumerate(item_idx_to_id)}
        return mat, user_idx_to_id, item_idx_to_id, user_id_to_idx, item_id_to_idx

    print("Building sparse user-item matrix from TRAIN data...")
    user_item_csr, u_idx2id, i_idx2id, u_id2idx, i_id2idx = build_sparse_user_item(train_df)
    print(f"Sparse TRAIN matrix shape: users={user_item_csr.shape[0]}, items={user_item_csr.shape[1]}, non-zeros={user_item_csr.nnz}")

    # === Check KNN model compatibility ===
    expected_features = getattr(knn_model, "n_features_in_", None)
    if expected_features is not None and expected_features != user_item_csr.shape[0]:
        print(f"WARNING: KNN model expects {expected_features} features, but user-item matrix has {user_item_csr.shape[0]} users.")
        print("Skipping KNN evaluation due to incompatibility.")
        knn_model = None

    # === Recommendation Functions ===
    def recommend_popularity(_seed_item, top_n=10):
        # Simply return top popular items regardless of seed item
        return popular_items[:top_n]

    def recommend_content_based(seed_item, top_n=10):
        idx = subset_item_to_idx.get(seed_item)
        if idx is None:
            return []
        sims = item_sim_matrix[idx]
        # Get indices of top similar items (excluding seed itself)
        top_indices = np.argpartition(-sims, range(top_n + 1))[:top_n + 1]
        top_indices = [i for i in top_indices if i != idx][:top_n]
        top_indices = sorted(top_indices, key=lambda i: sims[i], reverse=True)[:top_n]
        return [subset_items[i] for i in top_indices]

    def recommend_knn(seed_item, top_n=10):
        if knn_model is None:
            return []
        item_idx = i_id2idx.get(seed_item)
        if item_idx is None:
            return []
        # Extract the item vector (column) as dense row vector shape (1, num_users)
        vec = user_item_csr[:, item_idx].toarray().reshape(1, -1)
        # Verify dimensions match knn_model input feature count
        expected_dim = getattr(knn_model, "n_features_in_", None)
        if expected_dim is not None and expected_dim != vec.shape[1]:
            print(f"WARNING: KNN query vector dimension {vec.shape[1]} does not match model expected {expected_dim}.")
            return []
        distances, indices = knn_model.kneighbors(vec, n_neighbors=top_n + 1)
        rec_idxs = [j for j in indices.flatten() if j != item_idx][:top_n]
        # Defensive: clip indices to valid range
        max_idx = len(i_idx2id) - 1
        rec_idxs = [min(max(0, idx), max_idx) for idx in rec_idxs]
        return [i_idx2id[idx] for idx in rec_idxs]

    # === Evaluation Metrics ===
    def precision_at_k(recommended, relevant, k):
        if k == 0:
            return 0.0
        return len(set(recommended[:k]) & relevant) / k

    def recall_at_k(recommended, relevant, k):
        if not relevant:
            return 0.0
        return len(set(recommended[:k]) & relevant) / len(relevant)

    def average_precision(recommended, relevant, k):
        if not relevant:
            return 0.0
        ap = 0.0
        hits = 0
        for rank, item in enumerate(recommended[:k], start=1):
            if item in relevant:
                hits += 1
                ap += hits / rank
        return ap / min(len(relevant), k)

    def ndcg_at_k(recommended, relevant, k):
        if not relevant:
            return 0.0
        dcg = 0.0
        for rank, item in enumerate(recommended[:k], start=1):
            if item in relevant:
                dcg += 1.0 / math.log2(rank + 1)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
        return dcg / idcg if idcg > 0 else 0.0

    # === Evaluation Loop ===
    def evaluate(recommender_fn, users, k=10, seed=42, max_users=None, require_seed_in_domain=False):
        rng = np.random.default_rng(seed)
        if max_users is not None and len(users) > max_users:
            users = rng.choice(users, size=max_users, replace=False)

        precisions, recalls, maps, ndcgs = [], [], [], []
        train_user_items = train_df.groupby("visitorid")["itemid"].apply(set).to_dict()

        for user in users:
            if user not in test_user_items:
                continue
            relevant = test_user_items[user]
            if not relevant:
                continue
            seed_pool = train_user_items.get(user, set())
            if not seed_pool:
                continue

            # Pick a seed item from user's training history
            if require_seed_in_domain:
                # Restrict seed to items in content-based similarity subset
                candidates = list(seed_pool & set(subset_items))
                if not candidates:
                    continue
                seed_item = rng.choice(candidates)
            else:
                seed_item = rng.choice(list(seed_pool))

            recommended = recommender_fn(seed_item, top_n=k)
            if not recommended:
                continue

            precisions.append(precision_at_k(recommended, relevant, k))
            recalls.append(recall_at_k(recommended, relevant, k))
            maps.append(average_precision(recommended, relevant, k))
            ndcgs.append(ndcg_at_k(recommended, relevant, k))

        return {
            "Precision@K": float(np.mean(precisions)) if precisions else 0.0,
            "Recall@K": float(np.mean(recalls)) if recalls else 0.0,
            "MAP": float(np.mean(maps)) if maps else 0.0,
            "NDCG": float(np.mean(ndcgs)) if ndcgs else 0.0,
            "UsersEvaluated": len(precisions),
        }

    # === Run Evaluation ===
    all_train_users = train_df["visitorid"].unique()
    K = 10
    MAX_EVAL_USERS = 10000

    print(f"\nEvaluating Popularity-based recommender on up to {MAX_EVAL_USERS} users...")
    pop_metrics = evaluate(recommend_popularity, all_train_users, k=K, max_users=MAX_EVAL_USERS)

    print(f"Evaluating Content-Based recommender on up to {MAX_EVAL_USERS} users...")
    cb_metrics = evaluate(recommend_content_based, all_train_users, k=K, max_users=MAX_EVAL_USERS, require_seed_in_domain=True)

    print(f"Evaluating KNN Collaborative Filtering recommender on up to {MAX_EVAL_USERS} users...")
    knn_metrics = {}
    if knn_model is not None:
        knn_metrics = evaluate(recommend_knn, all_train_users, k=K, max_users=MAX_EVAL_USERS)
    else:
        print("KNN model unavailable or incompatible, skipping evaluation.")

    # === Summarize Results ===
    results_df = pd.DataFrame([
        {"Model": "Popularity", **pop_metrics},
        {"Model": "Content-Based", **cb_metrics},
    ])
    if knn_metrics:
        results_df = pd.concat([results_df, pd.DataFrame([{"Model": "KNN Collaborative Filtering", **knn_metrics}])], ignore_index=True)

    print("\nEvaluation Results (Top-K = {}):".format(K))
    print(results_df)

    # === Select Best Model ===
    best_model_row = results_df.loc[results_df["MAP"].idxmax()]
    print(f"\nBest performing model based on MAP: {best_model_row['Model']}")
    print(best_model_row)

    # === Save evaluation results to CSV ===
    eval_results_path = os.path.join(MODELS_DIR, "model_evaluation_results.csv")
    results_df.to_csv(eval_results_path, index=False)
    print(f"Saved evaluation results to {eval_results_path}")

if __name__ == "__main__":
    main()
    
# Run the script to evaluate models: python src/evaluate_models.py
# This script evaluates different recommendation models (popularity, content-based, KNN collaborative filtering)
# on a filtered dataset of user interactions. It computes precision, recall, MAP, and NDCG metrics
# for each model and summarizes the results. The best performing model is selected based on MAP score.
# Ensure the necessary directories and files exist before running this script.
# The script also saves the evaluation results to a CSV file for further analysis.
