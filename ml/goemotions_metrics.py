"""GoEmotions test precision, recall and accuracy from saved probabilities.

Usage: python -m ml.goemotions_metrics bert
       python -m ml.goemotions_metrics distilbert
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from ml.data_prep import labels as L

NAME = sys.argv[1] if len(sys.argv) > 1 else "bert"
ROOT = Path(__file__).resolve().parents[1]
COLAB = ROOT / "ml" / "results" / "colab"
OUT = ROOT / "ml" / "results" / f"goemotions_test_metrics_{NAME}.json"

res = json.loads((COLAB / f"{NAME}_goemotions_results.json").read_text(encoding="utf-8"))
prob = np.load(COLAB / f"{NAME}_goemotions_test_prob.npy")
df = pd.read_parquet(ROOT / "ml" / "data" / "processed" / "goemotions" / "test_clean.parquet")

y_true = L.multi_hot([list(x) for x in df["labels"]]).astype(int)
assert prob.shape == y_true.shape, f"shape mismatch: {prob.shape} vs {y_true.shape}"
pred = (prob >= res["threshold"]).astype(int)

p, r, f1, sup = precision_recall_fscore_support(y_true, pred, average=None, zero_division=0)
macro = precision_recall_fscore_support(y_true, pred, average="macro", zero_division=0)[:3]
micro = precision_recall_fscore_support(y_true, pred, average="micro", zero_division=0)[:3]
subset_accuracy = float((pred == y_true).all(axis=1).mean())  # all 28 labels exactly right
label_accuracy = float((pred == y_true).mean())               # share of single yes/no decisions right

out = {
    "model": NAME,
    "threshold": res["threshold"],
    "n_test": int(len(df)),
    "macro": {"precision": float(macro[0]), "recall": float(macro[1]), "f1": float(macro[2])},
    "micro": {"precision": float(micro[0]), "recall": float(micro[1]), "f1": float(micro[2])},
    "exact_match_accuracy": subset_accuracy,
    "label_wise_accuracy": label_accuracy,
    "per_label": {
        name: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]), "support": int(sup[i])}
        for i, name in enumerate(L.LABELS)
    },
}
OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

print(f"{NAME}: GoEmotions test, {len(df)} examples, threshold {res['threshold']}")
print(f"  macro  P {macro[0]:.3f}  R {macro[1]:.3f}  F1 {macro[2]:.3f}")
print(f"  micro  P {micro[0]:.3f}  R {micro[1]:.3f}  F1 {micro[2]:.3f}")
print(f"  exact-match accuracy (all 28 decisions right): {subset_accuracy:.3f}")
print(f"  label-wise accuracy (each yes/no decision):    {label_accuracy:.3f}")
print("\nlabel            P      R      F1    n")
for i in np.argsort(-f1):
    print(f"  {L.LABELS[i]:<15} {p[i]:.3f}  {r[i]:.3f}  {f1[i]:.3f}  {int(sup[i])}")
print(f"\nSaved: {OUT}")