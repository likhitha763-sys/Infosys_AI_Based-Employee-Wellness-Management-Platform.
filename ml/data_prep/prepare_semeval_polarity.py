"""Prepare the SemEval-2018 Task 1 V-oc binary polarity subset.

Rule: valence -3..-1 -> negative (0), valence +1..+3 -> positive (1),
valence 0 (neutral or mixed) is excluded.
"""
from __future__ import annotations

import csv
import html
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = (
    ROOT / "ml" / "data" / "raw" / "semeval2018_voc" / "extracted"
    / "SemEval2018-Task1-all-data" / "English" / "V-oc"
)
OUT_DIR = ROOT / "ml" / "data" / "processed" / "semeval2018_voc_binary"
REPORT_PATH = ROOT / "ml" / "results" / "semeval_polarity_processing_report.json"

RAW_FILES = {
    "train": "2018-Valence-oc-En-train.txt",
    "validation": "2018-Valence-oc-En-dev.txt",
    "test": "2018-Valence-oc-En-test-gold.txt",
}
LABEL_TO_ID = {"negative": 0, "positive": 1}
EXPECTED_COLUMNS = ["ID", "Tweet", "Affect Dimension", "Intensity Class"]


def clean_text(text: str) -> str:
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def read_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        quoting=csv.QUOTE_NONE,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
    )
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}; found {list(df.columns)}")
    return df


def process_split(name: str, path: Path) -> tuple[pd.DataFrame, dict]:
    raw = read_raw(path)
    stats = {"file": path.name, "raw_rows": int(len(raw))}

    df = raw.copy()
    stats["duplicate_ids_dropped"] = int(df["ID"].duplicated().sum())
    df = df.drop_duplicates(subset="ID")

    df["text_original"] = df["Tweet"]
    df["text"] = df["Tweet"].map(clean_text)
    stats["empty_text_dropped"] = int((df["text"] == "").sum())
    df = df[df["text"] != ""]

    parsed = df["Intensity Class"].str.extract(r"([+-]?\d+)")[0]
    if parsed.isna().any():
        raise ValueError(f"{path.name}: some Intensity Class values have no number")
    df["valence"] = parsed.astype(int)
    if not df["valence"].between(-3, 3).all():
        raise ValueError(f"{path.name}: valence outside -3..3")

    stats["valence_counts"] = {
        str(k): int(v) for k, v in df["valence"].value_counts().sort_index().items()
    }

    stats["neutral_excluded"] = int((df["valence"] == 0).sum())
    df = df[df["valence"] != 0].copy()
    df["label"] = df["valence"].map(lambda v: "positive" if v > 0 else "negative")
    df["label_id"] = df["label"].map(LABEL_TO_ID)

    # Same text with conflicting labels: drop every copy. Then drop plain duplicates.
    key = df["text"].str.lower()
    conflict = df.groupby(key)["label"].transform("nunique") > 1
    stats["conflicting_duplicate_rows_dropped"] = int(conflict.sum())
    df = df[~conflict]
    stats["duplicate_text_rows_dropped"] = int(df["text"].str.lower().duplicated().sum())
    df = df.drop_duplicates(subset="text", keep="first")

    cols = ["ID", "text", "text_original", "valence", "label", "label_id"]
    df = df[cols].rename(columns={"ID": "id"}).reset_index(drop=True)
    return df, stats


def remove_overlap(later: pd.DataFrame, earlier: list[pd.DataFrame]) -> tuple[pd.DataFrame, int]:
    seen_text = set()
    seen_id = set()
    for e in earlier:
        seen_text |= set(e["text"].str.lower())
        seen_id |= set(e["id"])
    bad = later["text"].str.lower().isin(seen_text) | later["id"].isin(seen_id)
    return later[~bad].reset_index(drop=True), int(bad.sum())


def main() -> None:
    splits, report = {}, {"splits": {}}
    for name, fname in RAW_FILES.items():
        path = RAW_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing raw file: {path}")
        splits[name], report["splits"][name] = process_split(name, path)

    splits["validation"], n_val = remove_overlap(splits["validation"], [splits["train"]])
    splits["test"], n_test = remove_overlap(splits["test"], [splits["train"], splits["validation"]])
    report["splits"]["validation"]["overlap_with_earlier_splits_dropped"] = n_val
    report["splits"]["test"]["overlap_with_earlier_splits_dropped"] = n_test

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df.to_parquet(OUT_DIR / f"{name}.parquet", index=False)
        report["splits"][name]["final_rows"] = int(len(df))
        report["splits"][name]["final_label_counts"] = {
            k: int(v) for k, v in df["label"].value_counts().items()
        }

    report["rule"] = "valence<0 -> negative(0); valence>0 -> positive(1); valence==0 excluded"
    report["label_to_id"] = LABEL_TO_ID
    report["text_cleaning"] = "html.unescape + whitespace collapse; mentions/hashtags/emojis kept"
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("SemEval-2018 V-oc binary polarity subset")
    for name, df in splits.items():
        s = report["splits"][name]
        print(
            f"  {name:<10} raw={s['raw_rows']:>5}  neutral_excluded={s['neutral_excluded']:>4}"
            f"  final={s['final_rows']:>5}  {s['final_label_counts']}"
        )
    print(f"Saved to: {OUT_DIR}")
    print(f"Report:   {REPORT_PATH}")


if __name__ == "__main__":
    main()