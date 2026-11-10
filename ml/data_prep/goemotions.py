"""Downloading, loading, and checking GoEmotions (the official 'simplified' version).

The data is read straight from its files (parquet) with pandas. The exact
dataset version (a commit hash on Hugging Face) and a checksum of every file
are saved in a manifest, so the same data can be downloaded and verified again.
"""

import hashlib
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from ml import config

SPLITS = ("train", "validation", "test")
DATA_FOLDER = config.RAW_DIR / "goemotions"
MANIFEST_PATH = config.RESULTS_DIR / "goemotions_manifest.json"


# ---------------------------------------------------------------------------
# Finding and downloading the files
# ---------------------------------------------------------------------------


def sha256_of(path):
    """The SHA-256 checksum of a file (a fingerprint of its exact contents)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_split_files(repo_files, subset=config.GOEMOTIONS["subset"]):
    """Pick the parquet file of each split from the list of files in the repository."""
    chosen = {}
    for split in SPLITS:
        matches = sorted(
            name for name in repo_files
            if name.startswith(f"{subset}/{split}-") and name.endswith(".parquet")
        )
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one '{subset}' parquet file for the {split} split, found {len(matches)}."
            )
        chosen[split] = matches[0]
    return chosen


def find_label_names(node):
    """Search nested metadata for the list of label names (None if there is none)."""
    if isinstance(node, dict):
        names = node.get("names")
        if isinstance(names, list) and all(isinstance(name, str) for name in names):
            return names
        children = node.values()
    elif isinstance(node, list):
        children = node
    else:
        return None
    for child in children:
        found = find_label_names(child)
        if found:
            return found
    return None


def read_label_names(parquet_path):
    """The label names stored inside a parquet file, or None if it has none."""
    metadata = pq.read_schema(parquet_path).metadata or {}
    raw = metadata.get(b"huggingface")
    if not raw:
        return None
    try:
        return find_label_names(json.loads(raw))
    except ValueError:
        return None


def download_goemotions(revision=None):
    """Download the three split files and write the manifest. Returns the manifest.

    revision=None downloads the latest version; pass a commit hash from an
    earlier manifest to get exactly the same data again.
    """
    repo_id = config.GOEMOTIONS["repo_id"]
    api = HfApi()
    sha = api.dataset_info(repo_id, revision=revision).sha
    split_files = find_split_files(api.list_repo_files(repo_id, repo_type="dataset", revision=sha))

    DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    files, rows, label_names = {}, {}, None
    for split, remote_name in split_files.items():
        local = Path(
            hf_hub_download(
                repo_id=repo_id, filename=remote_name, repo_type="dataset",
                revision=sha, local_dir=DATA_FOLDER,
            )
        ).resolve()
        files[split] = {
            "remote_name": remote_name,
            "local_path": local.relative_to(config.PROJECT_ROOT).as_posix(),
            "bytes": local.stat().st_size,
            "sha256": sha256_of(local),
        }
        rows[split] = pq.ParquetFile(local).metadata.num_rows
        label_names = label_names or read_label_names(local)

    manifest = {
        "dataset": config.GOEMOTIONS["name"],
        "repo_id": repo_id,
        "subset": config.GOEMOTIONS["subset"],
        "revision": sha,
        "license": config.GOEMOTIONS["license"],
        "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label_names_from_data": label_names,
        "files": files,
        "rows": rows,
    }
    config.ensure_directories()
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_manifest():
    """Read the saved manifest."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("No manifest yet. Run: python -m ml.inspect_goemotions")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def ensure_downloaded():
    """Make sure the data is on disk, using the pinned version if a manifest exists."""
    if MANIFEST_PATH.exists():
        manifest = load_manifest()
        if all((config.PROJECT_ROOT / info["local_path"]).exists() for info in manifest["files"].values()):
            return manifest
        return download_goemotions(revision=manifest["revision"])
    return download_goemotions()


def verify_files(manifest):
    """Compare each file with its recorded checksum. Returns a list of problems."""
    problems = []
    for split, info in manifest["files"].items():
        path = config.PROJECT_ROOT / info["local_path"]
        if not path.exists():
            problems.append(f"{split}: the file is missing")
        elif sha256_of(path) != info["sha256"]:
            problems.append(f"{split}: the checksum does not match the manifest")
    return problems


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_split(split, manifest=None):
    """One split as a table with the columns: text, labels (a list of numbers), id."""
    manifest = manifest or load_manifest()
    frame = pd.read_parquet(config.PROJECT_ROOT / manifest["files"][split]["local_path"])
    frame["labels"] = frame["labels"].apply(lambda values: [int(value) for value in values])
    return frame


def load_all_splits(manifest=None):
    """All three splits as a dictionary."""
    manifest = manifest or load_manifest()
    return {split: load_split(split, manifest) for split in SPLITS}


# ---------------------------------------------------------------------------
# Statistics and leakage checks
# ---------------------------------------------------------------------------


def label_counts(frame, number_of_labels=len(config.GOEMOTIONS_LABELS)):
    """How many examples carry each label (an example can carry several)."""
    counts = np.zeros(number_of_labels, dtype=int)
    for labels in frame["labels"]:
        for label in labels:
            counts[label] += 1
    return counts


def labels_per_example(frame):
    """The number of labels on each example."""
    return [len(labels) for labels in frame["labels"]]


def normalize_text(text):
    """Lowercase and tidy spaces, so near-identical texts compare as equal."""
    return " ".join(str(text).lower().split())


def find_overlaps(frames):
    """How many distinct (normalized) texts appear in more than one split."""
    texts = {split: {normalize_text(text) for text in frame["text"]} for split, frame in frames.items()}
    present = [split for split in SPLITS if split in frames]
    return {f"{first}&{second}": len(texts[first] & texts[second]) for first, second in combinations(present, 2)}


def rows_found_in(frame, other):
    """How many rows of `frame` have a text that also appears in `other`."""
    other_texts = {normalize_text(text) for text in other["text"]}
    return int(sum(normalize_text(text) in other_texts for text in frame["text"]))


def duplicate_rows(frame):
    """How many rows share their (normalized) text with at least one other row."""
    counts = frame["text"].map(normalize_text).value_counts()
    return int(counts[counts > 1].sum())


def conflicting_duplicates(frame):
    """How many repeated texts have different label sets on different rows."""
    seen = {}
    for text, labels in zip(frame["text"].map(normalize_text), frame["labels"]):
        seen.setdefault(text, set()).add(tuple(sorted(labels)))
    return sum(1 for label_sets in seen.values() if len(label_sets) > 1)