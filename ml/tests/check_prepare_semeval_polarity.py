"""Checks for the SemEval V-oc binary polarity subset."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "ml" / "data" / "processed" / "semeval2018_voc_binary"
REPORT = ROOT / "ml" / "results" / "semeval_polarity_processing_report.json"

failures = []


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


splits = {}
for name in ["train", "validation", "test"]:
    p = OUT_DIR / f"{name}.parquet"
    check(p.exists(), f"{name}.parquet exists")
    if p.exists():
        splits[name] = pd.read_parquet(p)

for name, df in splits.items():
    check(len(df) > 0, f"{name}: not empty ({len(df)} rows)")
    check(set(df["label_id"].unique()) <= {0, 1}, f"{name}: labels are only 0/1")
    check((df["valence"] != 0).all(), f"{name}: no neutral tweets")
    check(((df["valence"] > 0) == (df["label_id"] == 1)).all(), f"{name}: label matches valence sign")
    check((df["text"].str.strip() != "").all(), f"{name}: no empty text")
    check(not df["id"].duplicated().any(), f"{name}: no duplicate ids")
    check(df["label_id"].nunique() == 2, f"{name}: has both classes")

if len(splits) == 3:
    names = list(splits)
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = splits[names[i]], splits[names[j]]
            check(not (set(a["id"]) & set(b["id"])), f"no id overlap: {names[i]} / {names[j]}")
            check(
                not (set(a["text"].str.lower()) & set(b["text"].str.lower())),
                f"no text overlap: {names[i]} / {names[j]}",
            )

check(REPORT.exists(), "processing report exists")
if REPORT.exists():
    json.loads(REPORT.read_text(encoding="utf-8"))
    check(True, "processing report is valid JSON")

print()
print("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED")
raise SystemExit(1 if failures else 0)