"""Central configuration: file paths, random seed and dataset column names."""

from pathlib import Path  


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Original dataset
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Tweets.csv"


FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Single random seed for reproducibility
RANDOM_SEED = 42


TEXT_COL = "text"

LABEL_COL = "airline_sentiment"

AIRLINE_COL = "airline"
