"""Download and inspect GoEmotions (Milestone 2, Step 3).

This only looks at the data. It does not train anything or choose categories.
Run from the project root:
    python -m ml.inspect_goemotions
"""

import json
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # comments can contain emojis

from ml import config, reproducibility  # noqa: E402
from ml.data_prep import goemotions as go  # noqa: E402

LABELS = list(config.GOEMOTIONS_LABELS)
RARE_PERCENT = 1.0  # a label in under 1% of training examples counts as rare


def heading(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


reproducibility.set_seed()
config.ensure_directories()

print("Preparing the dataset (the first run downloads about 3.5 MB)...")
manifest = go.ensure_downloaded()
problems = go.verify_files(manifest)
if problems:
    print("PROBLEM with the downloaded files:", problems)
    sys.exit(1)
frames = go.load_all_splits(manifest)
train = frames["train"]
stats = {"revision": manifest["revision"], "rows": {split: len(frame) for split, frame in frames.items()}}

heading("1. Dataset version")
print(f"Source:   {manifest['repo_id']} ({manifest['subset']})")
print(f"Revision: {manifest['revision']}")
print(f"License:  {manifest['license']}")
for split in go.SPLITS:
    print(f"  {split:<11} {len(frames[split]):>6} examples  ({manifest['files'][split]['bytes']:,} bytes)")

heading("2. Label names")
in_data = manifest.get("label_names_from_data")
if in_data is None:
    print("The files do not list label names, so the list in ml/config.py is used.")
elif in_data == LABELS:
    print(f"The label names stored in the data match ml/config.py ({len(LABELS)} labels).")
else:
    print("WARNING: the label names in the data differ from ml/config.py:", in_data)

heading("3. Examples per label (an example can have several labels)")
counts = {split: go.label_counts(frame) for split, frame in frames.items()}
table = pd.DataFrame(
    {
        "label": LABELS,
        "train": counts["train"],
        "train_%": np.round(100 * counts["train"] / len(train), 1),
        "validation": counts["validation"],
        "test": counts["test"],
    }
).sort_values("train", ascending=False)
print(table.to_string(index=False))
table.to_csv(config.RESULTS_DIR / "goemotions_label_counts.csv", index=False)

heading("4. How many labels does each example have?")
for split, frame in frames.items():
    sizes = pd.Series(go.labels_per_example(frame))
    shares = sizes.clip(upper=4).value_counts(normalize=True).sort_index()
    parts = ", ".join(f"{'4+' if size == 4 else size} label(s): {100 * share:.1f}%" for size, share in shares.items())
    print(f"{split:<11} average {sizes.mean():.2f} | {parts}")
    stats[f"mean_labels_{split}"] = round(float(sizes.mean()), 3)

heading("5. Class imbalance (training split)")
per_label = pd.Series(counts["train"], index=LABELS)
print(f"Most frequent: {per_label.idxmax()} ({per_label.max()} examples)")
print(f"Rarest:        {per_label.idxmin()} ({per_label.min()} examples)")
if per_label.min() > 0:
    print(f"Ratio most/least frequent: {per_label.max() / per_label.min():.1f}")
rare = [name for name, count in per_label.items() if 100 * count / len(train) < RARE_PERCENT]
print(f"Rare labels (under {RARE_PERCENT}% of training examples): {rare if rare else 'none'}")
weights = ((len(train) - per_label) / per_label.replace(0, np.nan)).sort_values(ascending=False)
print("Planned loss weight for each label = (examples without it) / (examples with it)")
print("  highest:", ", ".join(f"{name} {value:.1f}" for name, value in weights.head(3).items()))
print("  lowest: ", ", ".join(f"{name} {value:.1f}" for name, value in weights.tail(3).items()))
stats["rare_labels"] = rare

heading("6. Text length (words, training split)")
words = train["text"].str.split().str.len()
print(f"median {words.median():.0f} | 90th percentile {words.quantile(0.9):.0f} | 99th percentile {words.quantile(0.99):.0f} | longest {words.max()}")

heading("7. Leakage checks")
all_ids = pd.concat([frame["id"] for frame in frames.values()])
print(f"Comment IDs that appear more than once (any split): {int(all_ids.duplicated().sum())}")
overlaps = go.find_overlaps(frames)
for pair, number in overlaps.items():
    print(f"Distinct texts shared by {pair}: {number}")
for split in ("validation", "test"):
    found = go.rows_found_in(frames[split], train)
    print(f"{split}: {found} of {len(frames[split])} examples ({100 * found / len(frames[split]):.1f}%) have a text that also appears in train")
    stats[f"{split}_rows_also_in_train"] = found
for split, frame in frames.items():
    print(f"{split}: {go.duplicate_rows(frame)} rows repeat another row's text; {go.conflicting_duplicates(frame)} repeated texts have different labels")
stats["shared_texts"] = overlaps
print("Note: the simplified data has no author or thread information, so same-author leakage cannot be checked.")

heading("8. Random training examples (seeded)")
rng = np.random.default_rng(config.SEED)
multi = train[[len(labels) >= 2 for labels in train["labels"]]]
chosen = list(train.iloc[rng.choice(len(train), size=4, replace=False)].itertuples()) + list(
    multi.iloc[rng.choice(len(multi), size=2, replace=False)].itertuples()
)
for row in chosen:
    print(f"- {row.text[:150]!r}\n    labels: {[LABELS[label] for label in row.labels]}")

(config.RESULTS_DIR / "goemotions_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
print(f"\nSaved: {config.RESULTS_DIR / 'goemotions_manifest.json'}, goemotions_label_counts.csv, goemotions_stats.json")