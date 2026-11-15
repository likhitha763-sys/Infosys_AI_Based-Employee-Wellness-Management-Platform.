"""Checks for the ISEAR held-out set."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_FILE = ROOT / "ml" / "data" / "processed" / "isear" / "isear_clean.parquet"
REPORT = ROOT / "ml" / "results" / "isear_processing_report.json"
GOEMO_TRAIN = ROOT / "ml" / "data" / "processed" / "goemotions" / "train.parquet"

failures = []


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


check(OUT_FILE.exists(), "isear_clean.parquet exists")
if OUT_FILE.exists():
    df = pd.read_parquet(OUT_FILE)
    check(len(df) > 1000, f"has a reasonable number of rows ({len(df)})")
    check(not df["id"].duplicated().any(), "no duplicate ids")
    check(not df["text"].str.lower().duplicated().any(), "no duplicate texts")
    check((df["text"].str.strip() != "").all(), "no empty text")
    check(not df["text"].str.replace(" ", "").str.lower().eq("[noresponse.]").any(), "no '[ No response.]' rows")
    check(set(df["emotion_original"]) == {"joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"},
          "all 7 ISEAR emotions present")
    unm = df[df["emotion_original"].isin(["shame", "guilt"])]
    check((~unm["in_main_eval"]).all() and unm["mapped_label"].isna().all(),
          "shame and guilt are unmapped and excluded from the main eval")
    mapped = df[df["in_main_eval"]]
    check((mapped["mapped_label"] == mapped["emotion_original"]).all(),
          "mapped labels equal the original for the 5 direct matches")
    if GOEMO_TRAIN.exists():
        g = pd.read_parquet(GOEMO_TRAIN)
        if "text" in g.columns:
            overlap = set(df["text"].str.lower()) & set(g["text"].astype(str).str.lower().str.strip())
            check(len(overlap) == 0, "no overlap with GoEmotions train text")

check(REPORT.exists(), "processing report exists")
if REPORT.exists():
    json.loads(REPORT.read_text(encoding="utf-8"))
    check(True, "processing report is valid JSON")

print()
print("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED")
raise SystemExit(1 if failures else 0)