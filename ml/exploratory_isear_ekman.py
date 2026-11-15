"""Exploratory ISEAR view: group the predicted label into its Ekman group first.

Added AFTER the primary ISEAR results were seen; report it as exploratory.
Usage: python -m ml.exploratory_isear_ekman distilbert
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

from ml import config

NAME = sys.argv[1] if len(sys.argv) > 1 else "distilbert"
ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "ml" / "results" / "colab" / "eval" / f"{NAME}_isear_predictions.csv"
OUT = ROOT / "ml" / "results" / f"exploratory_isear_ekman_{NAME}.json"
MAIN = ["joy", "fear", "anger", "sadness", "disgust"]

to_group = {"neutral": "neutral"}
for group, members in config.EKMAN_MAPPING.items():
    for member in members:
        to_group[member] = group

df = pd.read_csv(PRED)
df = df[df["emotion_original"].isin(MAIN)].copy()
df["pred_group"] = df["pred"].map(to_group).fillna("unmapped")

true, pred = df["emotion_original"], df["pred_group"]
result = {
    "model": NAME,
    "note": "Exploratory view added after the primary ISEAR results were known.",
    "n": int(len(df)),
    "accuracy": float((true == pred).mean()),
    "macro_f1_5_classes": float(f1_score(true, pred, labels=MAIN, average="macro", zero_division=0)),
    "share_predicted_neutral": float((pred == "neutral").mean()),
    "share_predicted_surprise_group": float((pred == "surprise").mean()),
    "unmapped_predictions": int((pred == "unmapped").sum()),
}
OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

print(f"Exploratory Ekman view for {NAME} (n={result['n']})")
print(f"  accuracy {result['accuracy']:.4f} | macro F1 {result['macro_f1_5_classes']:.4f}")
print(f"  predicted neutral: {result['share_predicted_neutral']:.1%} | "
      f"surprise group: {result['share_predicted_surprise_group']:.1%} | unmapped: {result['unmapped_predictions']}")
print()
print(pd.crosstab(true, pred).to_string())
print(f"\nSaved: {OUT}")