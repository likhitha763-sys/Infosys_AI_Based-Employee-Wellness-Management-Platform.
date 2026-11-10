"""Checks for the Milestone 2 setup (settings, seeds, environment, libraries).

Run from the project root:
    python -m ml.tests.check_setup
"""

import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from ml import config, reproducibility

passed = 0
failed = 0


def check(description, condition):
    """Print PASS or FAIL for one check and keep count."""
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {description}")
    else:
        failed += 1
        print(f"  FAIL  {description}")


def close(value, expected, tolerance=1e-3):
    return abs(value - expected) < tolerance


print("1. Settings")
check("the seed is a whole number", isinstance(config.SEED, int))
config.ensure_directories()
check(
    "the data, models, and results folders exist",
    all(folder.is_dir() for folder in (config.RAW_DIR, config.PROCESSED_DIR, config.MODELS_DIR, config.RESULTS_DIR)),
)
check(
    "BERT and DistilBERT are both named, and are different models",
    config.MODELS["bert"] and config.MODELS["distilbert"] and config.MODELS["bert"] != config.MODELS["distilbert"],
)
check("GoEmotions records its source and license", "go_emotions" in config.GOEMOTIONS["repo_id"] and "Apache" in config.GOEMOTIONS["license"])

print("\n2. Random seeds")


def draw():
    """One random number from each generator."""
    return random.random(), float(np.random.rand()), float(torch.rand(1))


reproducibility.set_seed(123)
first = draw()
reproducibility.set_seed(123)
second = draw()
reproducibility.set_seed(124)
third = draw()
check("the same seed gives the same Python random numbers", first[0] == second[0])
check("the same seed gives the same NumPy random numbers", first[1] == second[1])
check("the same seed gives the same PyTorch random numbers", first[2] == second[2])
check("a different seed gives different numbers", first != third)

print("\n3. Environment report")
report = reproducibility.environment_report()
check(
    "the report has the expected fields",
    {"created_utc", "seed", "python", "platform", "torch_cuda_available", "packages"} <= set(report),
)
check(
    "torch, transformers, and scikit-learn are installed",
    all(report["packages"][name] != "not installed" for name in ("torch", "transformers", "scikit-learn", "pyarrow")),
)
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
    saved = reproducibility.save_environment_report(Path(folder) / "environment.json")
    reloaded = json.loads(Path(saved).read_text(encoding="utf-8"))
    check("the report is saved as JSON and can be read back", reloaded["seed"] == config.SEED and "packages" in reloaded)

print("\n4. Libraries (metrics worked out by hand)")
# Two emotions, three texts. Emotion A: 1 hit and 1 miss. Emotion B: 2 hits.
true = np.array([[1, 0], [1, 1], [0, 1]])
predicted = np.array([[1, 0], [0, 1], [0, 1]])
check("macro F1 is (2/3 + 1) / 2 = 0.833", close(f1_score(true, predicted, average="macro", zero_division=0), 0.8333))
check("micro F1 is 6/7 = 0.857", close(f1_score(true, predicted, average="micro", zero_division=0), 0.8571))
check(
    "macro precision is 1.0 and macro recall is (0.5 + 1) / 2 = 0.75",
    close(precision_score(true, predicted, average="macro", zero_division=0), 1.0)
    and close(recall_score(true, predicted, average="macro", zero_division=0), 0.75),
)
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
    table = pd.DataFrame({"text": ["a", "b"], "labels": [[0, 1], [2]]})
    table.to_parquet(Path(folder) / "test.parquet")
    back = pd.read_parquet(Path(folder) / "test.parquet")
    check("a table with list columns survives a parquet round trip", list(back["text"]) == ["a", "b"] and [list(x) for x in back["labels"]] == [[0, 1], [2]])

print("\n" + "=" * 50)
print(f"RESULT: {passed} passed, {failed} failed")
print("=" * 50)
sys.exit(1 if failed else 0)