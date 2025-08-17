# 📦 RetailRocket Recommender System – Realistic e-commerce recommendations using popularity, content, and KNN collaborative filtering.

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Build](https://img.shields.io/badge/build-passing-brightgreen)](#)
[![Coverage](https://img.shields.io/badge/coverage-95%25-yellow)](#)

Project Overview

This project implements and evaluates recommender system models on the RetailRocket dataset.  
We explore popularity-based, content-based, and collaborative filtering methods, with a focus on optimizing a KNN-based collaborative filtering model.

<details>
<summary>📁 Project Structure</summary>

```text
retailrocket-recsys/
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 .gitignore
├── 📁 data/
│   ├── 📁 raw/
│   │   ├── category_tree.csv
│   │   ├── ecommerce-dataset.zip
│   │   ├── item_properties_part1.csv
│   │   ├── item_properties_part2.csv
│   │   └── retailrocket_events.csv.gz
│   └── 📁 processed/
│       ├── df_filtered.pkl
│       ├── df_preprocessed.parquet
│       ├── events.parquet
│       ├── item_properties_encoded.pkl
│       ├── item_properties_wide.parquet
│       ├── reconstructed_events.parquet
│       └── session_lengths.csv
├── 📁 artifacts/
│   ├── 📁 embeddings/
│   │   ├── item_similarity_sub.npy
│   │   └── item_similarity_sub_items.csv
│   ├── 📁 indices/
│   └── 📁 models/
│       ├── knn_model.pkl
│       ├── knn_model_tuned.pkl
│       ├── popularity_model.parquet
│       ├── recommender_functions.pkl
│       └── model_evaluation_results.csv
├── 📁 notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_deep_eda.ipynb
│   ├── 03_data_preprocessing_&_feature_eng.ipynb
│   ├── 04_model_building.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── Documentation.ipynb
├── 📁 src/
│   ├── __init__.py
│   ├── prepare_data.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── recommend.py
└── 📄 setup.sh

</details>


Dataset

RetailRocket e-commerce user interaction logs including clicks, cart additions, and orders.

Data Processing

- Filtered and preprocessed data stored in `data/processed/`
- User-wise train/test split to prevent leakage
- User-item interaction matrix creation for collaborative filtering

Models

- **Popularity:** Simple baseline recommending most popular items
- **Content-Based:** Item similarity based on precomputed embeddings
- **Collaborative Filtering:** KNN on user-item sparse matrix, tuned for best performance

Hyperparameter Tuning

- Grid search over `n_neighbors`, `algorithm`, and `metric` parameters for KNN
- Best model selected based on Mean Average Precision (MAP)

Evaluation Results

| Model                      | Precision@10 | Recall@10 | MAP    | NDCG   |
|----------------------------|--------------|-----------|--------|--------|
| Popularity Baseline         | 0.0015       | 0.0082    | 0.0037 | 0.0053 |
| Content-Based Filtering     | 0.0002       | 0.0008    | 0.0003 | 0.0005 |
| KNN Collaborative Filtering| **0.0217**   | **0.1452**|**0.0720**|**0.0950**|

Usage Instructions

Local

1. Clone this repository  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
3. Run notebooks from Step 1 through Step 5 sequentially to reproduce results

Future Work

* Real-time user profile updates

* Neural recommendation models

* Cold-start strategies

* Session-based recommendation approaches

Contact

Peter Obi
Email: peter@example.com
GitHub: github.com/peterobi
