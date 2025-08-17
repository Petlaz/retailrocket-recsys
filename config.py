from pathlib import Path

# Base directory (project root)
BASE_DIR = Path(__file__).resolve().parent.parent

# Artifacts and models directories
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
EMBEDDINGS_DIR = ARTIFACTS_DIR / "embeddings"
DATA_DIR = ARTIFACTS_DIR / "data"

# Safe recommender pickle
SAFE_RECSYS_PATH = MODELS_DIR / "safe_recommender.pkl"

# Content-based
CB_SIM_MATRIX_PATH = EMBEDDINGS_DIR / "item_similarity_sub.npy"
CB_ITEMS_PATH = EMBEDDINGS_DIR / "item_similarity_sub_items.csv"

# KNN
KNN_MODEL_PATH = MODELS_DIR / "knn_model.pkl"
KNN_ITEM_CATEGORIES_PATH = EMBEDDINGS_DIR / "knn_item_categories.npy"
KNN_USER_ITEM_SHAPE_PATH = EMBEDDINGS_DIR / "knn_user_item_shape.npy"

# Popularity model
POPULARITY_MODEL_PATH = MODELS_DIR / "popularity_model.parquet"

# Gradio defaults
DEFAULT_NUM_RECS = 10
