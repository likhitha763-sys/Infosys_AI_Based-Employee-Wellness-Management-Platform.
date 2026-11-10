"""Central settings for the Milestone 2 experiments.

Everything that affects results lives here (the random seed, folders, dataset
sources, and model names), so anyone can reproduce an experiment from this file.
Values marked "set in Step N" are filled in when that step is done.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_DIR = PROJECT_ROOT / "ml"

# Folders. Data and models are not committed to Git (they are large, and
# datasets have their own licenses). Small result files are committed.
DATA_DIR = ML_DIR / "data"
RAW_DIR = DATA_DIR / "raw"  # datasets exactly as downloaded
PROCESSED_DIR = DATA_DIR / "processed"  # cleaned and split data
MODELS_DIR = ML_DIR / "models"  # trained checkpoints
RESULTS_DIR = ML_DIR / "results"  # metrics tables, reports, environment info

# One seed for everything, so splits and training can be repeated.
SEED = 42

# The two transformer models compared in this milestone. Both are "cased"
# (they keep capital letters), as in the original GoEmotions baseline.
MODELS = {
    "distilbert": "distilbert-base-cased",
    "bert": "bert-base-cased",
}

# Datasets. Each entry records where the data comes from and under what
# terms, so the project report can cite them accurately.
GOEMOTIONS = {
    "name": "GoEmotions",
    "repo_id": "google-research-datasets/go_emotions",
    "subset": "simplified",  # the version with official train/validation/test splits
    "license": "Apache 2.0, as stated on the dataset card",
    "revision": None,  # the exact dataset version is pinned in Step 3
}

SEMEVAL = {
    "name": "SemEval-2018 Task 1: Affect in Tweets",
    "source": None,  # set in Step 4 (official source; read the terms of use first)
    "license": None,  # set in Step 4
}

ISEAR = {
    "name": "ISEAR",
    "source": None,  # set in Step 5 (legitimate source; read the terms of use first)
    "license": None,  # set in Step 5
}


def ensure_directories():
    """Create the folders above if they do not exist yet."""
    for folder in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, RESULTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)