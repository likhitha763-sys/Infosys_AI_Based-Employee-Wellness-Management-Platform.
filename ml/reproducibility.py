"""Reproducibility helpers: fixing random seeds and recording the environment.

Run from the project root to save a report of your Python and library versions:
    python -m ml.reproducibility
"""

import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from importlib import metadata

import numpy as np
import torch

from ml import config

KEY_PACKAGES = (
    "torch", "transformers", "numpy", "pandas", "pyarrow", "scikit-learn",
    "huggingface-hub", "vaderSentiment", "fastapi", "streamlit", "sqlalchemy",
)


def set_seed(seed=config.SEED):
    """Fix the random number generators used by Python, NumPy, and PyTorch.

    Note: results can still differ very slightly between computers or between
    CPU and GPU, because of how they add up numbers. We report the settings
    used rather than promising bit-for-bit identical results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def package_versions():
    """The installed version of each key package ('not installed' if missing)."""
    versions = {}
    for name in KEY_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def environment_report():
    """A dictionary describing the computer and software used for an experiment."""
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": config.SEED,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_logical_cores": os.cpu_count(),
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cpu_threads": torch.get_num_threads(),
        "packages": package_versions(),
    }


def save_environment_report(path=None):
    """Write the environment report as JSON and return the file path."""
    config.ensure_directories()
    path = path or (config.RESULTS_DIR / "environment.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(environment_report(), handle, indent=2)
    return path


if __name__ == "__main__":
    saved = save_environment_report()
    print(json.dumps(environment_report(), indent=2))
    print(f"\nSaved to {saved}")