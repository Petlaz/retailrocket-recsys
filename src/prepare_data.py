import pandas as pd
import os

RAW_PATH = "data/raw/retailrocket_events.csv.gz"
PROCESSED_PATH = "data/processed/events.parquet"

def prepare_data():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Raw dataset not found at {RAW_PATH}. Please download first.")

    print("Reading raw CSV...")
    df = pd.read_csv(RAW_PATH)

    print("Initial shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # Reduce memory usage
    if 'category' in df.columns:
        df['category'] = df['category'].astype('category')
    if 'visitorid' in df.columns:
        df['visitorid'] = df['visitorid'].astype('int32')
    if 'itemid' in df.columns:
        df['itemid'] = df['itemid'].astype('int32')

    print("Saving to Parquet...")
    df.to_parquet(PROCESSED_PATH, compression="snappy")
    print(f"Processed data saved to {PROCESSED_PATH}")

if __name__ == "__main__":
    prepare_data()
    
# Run this script to prepare the data: python src/prepare_data.py
# This script prepares the raw retail rocket events data for further analysis.
# It reads the raw CSV file, reduces memory usage by converting data types,
# and saves the processed data in Parquet format for efficient storage and access.
# Ensure the necessary directories exist before running this script.