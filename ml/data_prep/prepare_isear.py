"""Prepare ISEAR as a held-out evaluation set (never used for training)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = ROOT / "ml" / "data" / "raw" / "isear" / "isear-train.csv"
OUT_DIR = ROOT / "ml" / "data" / "processed" / "isear"
OUT_FILE = OUT_DIR / "isear_clean.parquet"
REPORT_PATH = ROOT / "ml" / "results" / "isear_processing_report.json"

LEAKAGE_FILES = {
    "goemotions_train": ROOT / "ml" / "data" / "processed" / "goemotions" / "train.parquet",
    "semeval_train": ROOT / "ml" / "data" / "processed" / "semeval2018_voc_binary" / "train.parquet",
}

ISEAR_LABELS = ["joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"]
LABEL_MAP = {
    "joy": "joy",
    "fear": "fear",
    "anger": "anger",
    "sadness": "sadness",
    "disgust": "disgust",
}
UNMAPPED_LABELS = ["shame", "guilt"]


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def is_placeholder(text: str) -> bool:
    return text.replace(" ", "").lower() == "[noresponse.]"


def load_texts(path: Path):
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "text" not in df.columns:
        raise ValueError(f"{path.name}: no 'text' column; columns are {list(df.columns)}")
    return set(df["text"].astype(str).str.lower().map(clean_text))


def main() -> None:
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Missing raw file: {RAW_FILE}")

    df = pd.read_csv(
        RAW_FILE, header=None, names=["emotion", "text"],
        dtype=str, keep_default_na=False, encoding="utf-8-sig",
    )
    report = {"source_file": RAW_FILE.name, "raw_rows": int(len(df))}

    df["emotion"] = df["emotion"].str.strip().str.lower()
    unknown = sorted(set(df["emotion"]) - set(ISEAR_LABELS))
    if unknown:
        raise ValueError(f"Unexpected emotion labels: {unknown}")

    df["text"] = df["text"].map(clean_text)

    placeholder = df["text"].map(is_placeholder)
    report["no_response_dropped"] = int(placeholder.sum())
    df = df[~placeholder]

    empty = df["text"] == ""
    report["empty_text_dropped"] = int(empty.sum())
    df = df[~empty]

    key = df["text"].str.lower()
    conflict = df.groupby(key)["emotion"].transform("nunique") > 1
    report["conflicting_duplicate_rows_dropped"] = int(conflict.sum())
    df = df[~conflict]

    dup = df["text"].str.lower().duplicated()
    report["duplicate_text_rows_dropped"] = int(dup.sum())
    df = df[~dup]

    report["leakage_removed"] = {}
    for name, path in LEAKAGE_FILES.items():
        texts = load_texts(path)
        if texts is None:
            report["leakage_removed"][name] = "file not found, not checked"
            continue
        hit = df["text"].str.lower().isin(texts)
        report["leakage_removed"][name] = int(hit.sum())
        df = df[~hit]

    df = df.reset_index(drop=True)
    df.insert(0, "id", [f"isear-{i:05d}" for i in range(len(df))])
    df = df.rename(columns={"emotion": "emotion_original"})
    df["mapped_label"] = df["emotion_original"].map(LABEL_MAP)
    df["in_main_eval"] = df["mapped_label"].notna()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_FILE, index=False)

    report["final_rows"] = int(len(df))
    report["final_emotion_counts"] = {k: int(v) for k, v in df["emotion_original"].value_counts().items()}
    report["main_eval_rows"] = int(df["in_main_eval"].sum())
    report["unmapped_rows"] = int((~df["in_main_eval"]).sum())
    report["rows_under_3_words_kept"] = int((df["text"].str.split().str.len() < 3).sum())
    report["label_map"] = LABEL_MAP
    report["unmapped_labels"] = UNMAPPED_LABELS
    report["source_note"] = (
        "Hugging Face community mirror TahaRasouli/ISEAR; contains only isear-train.csv "
        "(a subset of the full ~7.6k ISEAR). Used for held-out evaluation only."
    )
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("ISEAR held-out set")
    for k in ["raw_rows", "no_response_dropped", "empty_text_dropped",
              "conflicting_duplicate_rows_dropped", "duplicate_text_rows_dropped"]:
        print(f"  {k}: {report[k]}")
    print("  leakage_removed:", report["leakage_removed"])
    print(f"  final_rows: {report['final_rows']}  (main eval: {report['main_eval_rows']}, unmapped: {report['unmapped_rows']})")
    print("  emotion counts:", report["final_emotion_counts"])
    print(f"Saved: {OUT_FILE}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()