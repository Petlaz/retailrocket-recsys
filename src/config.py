from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
EMBEDDINGS_DIR = ARTIFACTS_DIR / "embeddings"

# Model files
POPULARITY_MODEL_PATH = MODELS_DIR / "popularity_model.parquet"
KNN_MODEL_PATH = MODELS_DIR / "knn_model.pkl"
CB_SIM_MATRIX_PATH = EMBEDDINGS_DIR / "item_similarity_sub.npy"

# Remove CB_ITEMS_PATH since it doesn't exist in your project
DEFAULT_NUM_RECS = 10