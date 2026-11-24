"""Score VADER on the SemEval-2018 V-oc binary polarity TEST set.

Rule (fixed in ml/results/evaluation_plan.md): positive if compound >= 0.05, else negative.
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
)

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = ROOT / "ml" / "data" / "processed" / "semeval2018_voc_binary" / "test.parquet"
OUT_FILE = ROOT / "ml" / "results" / "vader_semeval_results.json"
THRESHOLD = 0.05


def main() -> None:
    df = pd.read_parquet(TEST_FILE)
    analyzer = SentimentIntensityAnalyzer()

    compound = df["text"].map(lambda t: analyzer.polarity_scores(t)["compound"])
    y_true = df["label_id"].to_numpy()
    y_pred = (compound >= THRESHOLD).astype(int).to_numpy()

    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro")
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    results = {
        "model": "VADER",
        "rule": f"positive if compound >= {THRESHOLD}, else negative",
        "n_test": int(len(df)),
        "accuracy": float(acc),
        "macro_f1": float(macro),
        "per_class": {
            name: {"precision": float(prec[i]), "recall": float(rec[i]),
                   "f1": float(f1[i]), "support": int(support[i])}
            for i, name in enumerate(["negative", "positive"])
        },
        "confusion_matrix_rows_true_cols_pred_[neg,pos]": cm.tolist(),
    }
    OUT_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"VADER on SemEval test ({len(df)} tweets)")
    print(f"  accuracy: {acc:.4f}   macro F1: {macro:.4f}")
    for name, v in results["per_class"].items():
        print(f"  {name:<9} precision {v['precision']:.3f}  recall {v['recall']:.3f}  f1 {v['f1']:.3f}  (n={v['support']})")
    print("  confusion matrix [rows=true neg,pos; cols=pred neg,pos]:", cm.tolist())
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()