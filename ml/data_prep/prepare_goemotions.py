"""Prepare GoEmotions for training and evaluation (Milestone 2, Step 4).

Run from the project root:
    python -m ml.data_prep.prepare_goemotions

What it does:
  1. Loads the official train/validation/test splits (the pinned version).
  2. Makes "clean" validation and test sets: rows whose text also appears in the
     training split are removed, so no evaluation text was seen in training.
  3. Saves everything in ml/data/processed/goemotions.
  4. Computes loss weights for rare labels (from the training split only).
  5. Writes a small report (committed to Git) describing exactly what was done.

The text itself is not changed (no lowercasing, no stop-word removal): the
transformer models tokenize the raw text themselves.
"""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ml import config
from ml.data_prep import goemotions as go
from ml.data_prep import labels as lab

PROCESSED_FOLDER = config.PROCESSED_DIR / "goemotions"
REPORT_PATH = config.RESULTS_DIR / "goemotions_processing_report.json"
FEW_POSITIVES = 20  # fewer positives than this make a label's scores very noisy


def clean_eval_split(eval_frame, train_frame):
    """Remove evaluation rows whose text also appears in training.

    Returns (cleaned table, number of rows removed). The inputs are not changed.
    """
    train_texts = {go.normalize_text(text) for text in train_frame["text"]}
    keep = [go.normalize_text(text) not in train_texts for text in eval_frame["text"]]
    cleaned = eval_frame[keep].reset_index(drop=True)
    return cleaned, len(eval_frame) - len(cleaned)


def compute_pos_weights(counts, total):
    """Loss weights for each label: (examples without it) / (examples with it).

    'sqrt' is the square root of that, which keeps training steadier.
    A label with no positive examples gets weight 1 and is listed separately.
    """
    raw, tempered, zero_positive = [], [], []
    for index, count in enumerate(counts):
        if count > 0:
            value = (total - count) / count
        else:
            value = 1.0
            zero_positive.append(index)
        raw.append(float(value))
        tempered.append(float(np.sqrt(value)))
    return {"raw": raw, "sqrt": tempered, "zero_positive_labels": zero_positive}


def save_split(name, frame):
    """Write one split to a parquet file and return its path."""
    PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_FOLDER / f"{name}.parquet"
    frame.to_parquet(path, index=False)
    return path


def load_processed(name):
    """Read one prepared split (columns: text, labels as a list of numbers, id)."""
    frame = pd.read_parquet(PROCESSED_FOLDER / f"{name}.parquet")
    frame["labels"] = frame["labels"].apply(lambda values: [int(value) for value in values])
    return frame


def build():
    """Prepare all splits and write the report. Returns (report, splits)."""
    config.ensure_directories()
    manifest = go.ensure_downloaded()
    problems = go.verify_files(manifest)
    if problems:
        raise RuntimeError(f"The downloaded files have problems: {problems}")

    official = go.load_all_splits(manifest)
    train = official["train"]
    splits = {"train": train, "validation": official["validation"], "test": official["test"]}
    removed = {}
    for name in ("validation", "test"):
        splits[f"{name}_clean"], removed[name] = clean_eval_split(official[name], train)

    files = {}
    for name, frame in splits.items():
        path = save_split(name, frame)
        files[name] = {
            "rows": len(frame),
            "file": path.relative_to(config.PROJECT_ROOT).as_posix(),
            "sha256": go.sha256_of(path),
        }

    positives = {
        name: [int(value) for value in go.label_counts(splits[name])]
        for name in ("train", "validation_clean", "test_clean")
    }
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_revision": manifest["revision"],
        "text_processing": "none (raw text, case preserved)",
        "splits": files,
        "removed": removed,
        "label_names": list(lab.LABELS),
        "positives": positives,
        "pos_weights": compute_pos_weights(positives["train"], len(train)),
        "views": {
            "ekman_groups": list(lab.EKMAN_GROUPS),
            "headline_emotions": list(lab.HEADLINE_EMOTIONS),
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report, splits


def main():
    report, splits = build()
    print("Source version:", report["source_revision"])

    print("\n1. Clean evaluation sets (rows whose text also appears in train are removed)")
    for name in ("validation", "test"):
        official_rows = report["splits"][name]["rows"]
        removed = report["removed"][name]
        clean_rows = report["splits"][f"{name}_clean"]["rows"]
        print(f"  {name:<11} official {official_rows:>5} | removed {removed:>4} ({100 * removed / official_rows:.1f}%) | clean {clean_rows:>5}")

    print("\n2. Saved files")
    for name, info in report["splits"].items():
        print(f"  {name:<17} {info['rows']:>6} rows  {info['file']}")

    print("\n3. Positive examples per label, and the planned loss weights")
    weights = report["pos_weights"]
    rows = []
    for index, label in enumerate(lab.LABELS):
        validation = report["positives"]["validation_clean"][index]
        test = report["positives"]["test_clean"][index]
        rows.append(
            {
                "label": label,
                "train": report["positives"]["train"][index],
                "val_clean": validation,
                "test_clean": test,
                "weight_raw": round(weights["raw"][index], 1),
                "weight_sqrt": round(weights["sqrt"][index], 1),
                "few": "*" if min(validation, test) < FEW_POSITIVES else "",
            }
        )
    print(pd.DataFrame(rows).sort_values("train", ascending=False).to_string(index=False))
    print(f"\n  * = fewer than {FEW_POSITIVES} positive examples in the clean validation or test set,")
    print("      so this label's scores will be very noisy.")

    print("\n4. Examples per evaluation view")
    for name in ("train", "validation_clean", "test_clean"):
        matrix = lab.multi_hot(splits[name]["labels"])
        ekman = dict(zip(lab.EKMAN_GROUPS, (int(v) for v in lab.to_ekman_view(matrix).sum(axis=0))))
        headline = dict(zip(lab.HEADLINE_EMOTIONS, (int(v) for v in lab.to_headline_view(matrix).sum(axis=0))))
        print(f"  {name}\n    Ekman:    {ekman}\n    headline: {headline}")

    print(f"\nSaved the report to {REPORT_PATH}")
    print("Next: create and commit ml/results/evaluation_plan.md BEFORE any training (see the step instructions).")


if __name__ == "__main__":
    main()