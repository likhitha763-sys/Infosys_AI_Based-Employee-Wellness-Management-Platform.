"""Checks for the GoEmotions data module and the real downloaded data.

Run the inspection script first (it downloads the data):
    python -m ml.inspect_goemotions
Then, from the project root:
    python -m ml.tests.check_goemotions
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd

from ml import config
from ml.data_prep import goemotions as go

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


def raises(error_type, function, *arguments):
    """True if calling the function raises the given error type."""
    try:
        function(*arguments)
    except error_type:
        return True
    except Exception:
        return False
    return False


print("1. The functions, on small made-up data")
small = pd.DataFrame({"text": ["a", "b", "c"], "labels": [[0], [0, 2], [27]], "id": ["x", "y", "z"]})
counts = go.label_counts(small)
check("label counts are correct", counts[0] == 2 and counts[2] == 1 and counts[27] == 1 and counts.sum() == 4)
check("labels per example are correct", go.labels_per_example(small) == [1, 2, 1])
check("text normalization ignores case and extra spaces", go.normalize_text("  Hello   WORLD \n") == "hello world")
overlaps = go.find_overlaps(
    {
        "train": pd.DataFrame({"text": ["Hello there", "Other"]}),
        "validation": pd.DataFrame({"text": ["hello   THERE", "New"]}),
        "test": pd.DataFrame({"text": ["Brand new"]}),
    }
)
check("shared texts between splits are found", overlaps == {"train&validation": 1, "train&test": 0, "validation&test": 0})
names = ["simplified/train-00000-of-00001.parquet", "simplified/validation-00000-of-00001.parquet", "simplified/test-00000-of-00001.parquet", "raw/train.parquet"]
check("the right file is picked for each split", go.find_split_files(names)["validation"] == names[1] and len(go.find_split_files(names)) == 3)
check("a missing split file is reported", raises(RuntimeError, go.find_split_files, names[:2]))
fake_metadata = {"info": {"features": {"text": {"dtype": "string"}, "labels": {"feature": {"names": ["a", "b"], "_type": "ClassLabel"}, "_type": "Sequence"}}}}
check("label names are found inside dataset metadata", go.find_label_names(fake_metadata) == ["a", "b"])
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
    sample = Path(folder) / "sample.txt"
    sample.write_bytes(b"abc")
    check("the checksum of 'abc' is the known SHA-256 value", go.sha256_of(sample) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
grouped = [label for members in config.EKMAN_MAPPING.values() for label in members]
check(
    "the Ekman grouping covers each of the 27 emotions exactly once",
    sorted(grouped) == sorted(label for label in config.GOEMOTIONS_LABELS if label != "neutral"),
)

print("\n2. The real downloaded data")
if not go.MANIFEST_PATH.exists():
    check("the dataset has been downloaded (run: python -m ml.inspect_goemotions)", False)
else:
    manifest = go.load_manifest()
    frames = go.load_all_splits(manifest)
    check("the manifest lists all three splits and a 40-character revision", set(manifest["files"]) == set(go.SPLITS) and len(manifest["revision"]) == 40)
    check(
        "the split sizes match the dataset card (43,410 / 5,426 / 5,427)",
        {split: len(frame) for split, frame in frames.items()} == {"train": 43410, "validation": 5426, "test": 5427},
    )
    check("every file matches its recorded checksum", go.verify_files(manifest) == [])
    all_labels = [label for frame in frames.values() for labels in frame["labels"] for label in labels]
    check(
        "all label numbers are between 0 and 27, and no text is missing",
        min(all_labels) >= 0 and max(all_labels) <= 27 and all(frame["text"].notna().all() for frame in frames.values()),
    )
    in_data = manifest.get("label_names_from_data")
    check("label names stored in the data (if any) match the list in config.py", in_data is None or in_data == list(config.GOEMOTIONS_LABELS))
    thankful = frames["train"][frames["train"]["text"].str.lower().str.contains("thank")]
    best = config.GOEMOTIONS_LABELS[go.label_counts(thankful).argmax()]
    check(f"texts containing 'thank' are mostly labeled gratitude (found: {best})", best == "gratitude")
    ids = [set(frame["id"]) for frame in frames.values()]
    check(
        "no comment ID appears twice, in or across splits",
        sum(len(group) for group in ids) == len(set().union(*ids)) and all(frame["id"].is_unique for frame in frames.values()),
    )

print("\n" + "=" * 50)
print(f"RESULT: {passed} passed, {failed} failed")
print("=" * 50)
sys.exit(1 if failed else 0)