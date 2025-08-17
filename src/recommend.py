import pickle
import numpy as np
from pathlib import Path
from config import SAFE_RECSYS_PATH, DEFAULT_NUM_RECS

# ---------- Load safe pickle ----------
if not SAFE_RECSYS_PATH.exists():
    raise FileNotFoundError(f"Safe pickle not found at {SAFE_RECSYS_PATH}")

with open(SAFE_RECSYS_PATH, "rb") as f:
    safe_data = pickle.load(f)

# Content-based
cb_data = safe_data.get("content_based", {})
cb_sim_matrix = cb_data.get("sim_matrix")
cb_items = cb_data.get("items")

# KNN
knn_data = safe_data.get("knn_sparse", {})
knn_model = knn_data.get("model")
knn_items = knn_data.get("items")


# ---------- Recommendation Functions ----------
def recommend_content_based(seed_item_id: str, top_n: int = DEFAULT_NUM_RECS):
    """Content-based recommendation."""
    if cb_sim_matrix is None or cb_items is None:
        return []

    try:
        idx = np.where(cb_items == seed_item_id)[0][0]
    except IndexError:
        return []

    sims = cb_sim_matrix[idx]
    similar_indices = np.argsort(sims)[::-1]
    similar_indices = similar_indices[similar_indices != idx]
    top_indices = similar_indices[:top_n]

    return cb_items[top_indices].tolist()


def recommend_knn(seed_item_id: str, top_n: int = DEFAULT_NUM_RECS):
    """KNN recommendation."""
    if knn_model is None or knn_items is None:
        return []

    try:
        idx = np.where(knn_items == seed_item_id)[0][0]
    except IndexError:
        return []

    distances, indices = knn_model.kneighbors([knn_model._fit_X[idx]], n_neighbors=top_n + 1)
    indices = indices.flatten()
    indices = indices[indices != idx]
    top_indices = indices[:top_n]

    return knn_items[top_indices].tolist()


def get_all_items():
    """Return all available item IDs."""
    if cb_items is not None:
        return cb_items.tolist()
    elif knn_items is not None:
        return knn_items.tolist()
    else:
        return []
