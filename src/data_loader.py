"""Dataset loading and initial data quality checks."""

import pandas as pd 

from src.config import RAW_DATA_PATH, TEXT_COL, LABEL_COL, RANDOM_SEED  


def load_raw_data():
    """Load the original dataset"""
    df = pd.read_csv(RAW_DATA_PATH, encoding="utf-8")  # Read the csv as stored
    print(f"Loaded {len(df)} rows and {df.shape[1]} columns from {RAW_DATA_PATH.name}")
    return df  


def inspect_data(df):
    """Inspection to see structure, missing values, labels, duplicates."""

    print("\n--- STRUCTURE ---")
    
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),  # Data type
        "non_null": df.notna().sum(),    # Number of non-empty cells
        "missing": df.isna().sum(),      # Number of empty cells
        "unique": df.nunique(),          # Number of distinct values
    })
    print(summary.to_string()) 

    print("\n--- CLASS DISTRIBUTION ---")
    counts = df[LABEL_COL].value_counts()                             # Class frequency
    percent = df[LABEL_COL].value_counts(normalize=True) * 100        # Class frequency as percentage
    print(pd.DataFrame({"count": counts, "percent": percent.round(1)}).to_string())

    print("\n--- DUPLICATES ---")
    duplicate_texts = df[TEXT_COL].duplicated().sum()                        # Dublicates
    duplicate_pairs = df.duplicated(subset=[TEXT_COL, LABEL_COL]).sum()      # Identical text with the same classification
    print(f"Repeated tweet texts      : {duplicate_texts}")
    print(f"Repeated text-label pairs : {duplicate_pairs}")

    print("\n--- SAMPLE TWEETS ---")
    
    print(df[[TEXT_COL, LABEL_COL]].sample(5, random_state=RANDOM_SEED).to_string())
