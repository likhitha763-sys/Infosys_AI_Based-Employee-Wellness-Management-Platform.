"""Build frontend/model_metrics.json from the saved evaluation files.

The Streamlit page reads only this one small file. Run from the project root:
    python -m ml.make_frontend_metrics
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data_prep import labels as L

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "ml" / "results"
COLAB = RES / "colab"
EVAL = COLAB / "eval"
TEST_PARQUET = ROOT / "ml" / "data" / "processed" / "goemotions" / "test_clean.parquet"
OUT = ROOT / "frontend" / "model_metrics.json"

MODELS = {"bert": "BERT (fine-tuned)", "distilbert": "DistilBERT (fine-tuned)"}
ISEAR_MAIN = ["joy", "fear", "anger", "sadness", "disgust"]
CONF_KEY = "confusion_matrix_rows_true_cols_pred_[neg,pos]"
missing = []


def load(path):
    path = Path(path)
    if not path.exists():
        missing.append(str(path.relative_to(ROOT)))
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def with_simple_confusion_key(block):
    block = dict(block)
    block["confusion_matrix"] = block.pop(CONF_KEY, None)
    return block


def label_confusions(name, threshold):
    """tp, fp, fn, tn for each of the 28 labels on the GoEmotions test set."""
    npy = COLAB / f"{name}_goemotions_test_prob.npy"
    if not npy.exists() or not TEST_PARQUET.exists():
        missing.append(f"{npy.relative_to(ROOT)} or the processed test parquet")
        return None
    prob = np.load(npy)
    df = pd.read_parquet(TEST_PARQUET)
    y = L.multi_hot([list(x) for x in df["labels"]]).astype(int)
    pred = (prob >= threshold).astype(int)
    out = {}
    for i, label in enumerate(L.LABELS):
        t, p = y[:, i], pred[:, i]
        out[label] = {
            "tp": int(((t == 1) & (p == 1)).sum()),
            "fp": int(((t == 0) & (p == 1)).sum()),
            "fn": int(((t == 1) & (p == 0)).sum()),
            "tn": int(((t == 0) & (p == 0)).sum()),
        }
    return out


def isear_confusion(name):
    path = EVAL / f"{name}_isear_predictions.csv"
    if not path.exists():
        missing.append(str(path.relative_to(ROOT)))
        return None
    df = pd.read_csv(path)
    columns = ISEAR_MAIN + ["neutral", "other emotion"]
    df["bucket"] = df["pred"].map(
        lambda p: p if (p in ISEAR_MAIN or p == "neutral") else "other emotion"
    )
    table = (
        pd.crosstab(df["emotion_original"], df["bucket"])
        .reindex(index=ISEAR_MAIN, columns=columns, fill_value=0)
    )
    return {"rows": ISEAR_MAIN, "columns": columns, "matrix": table.values.tolist()}


def main():
    metrics = {
        "targets": {"goemotions_validation_macro_f1": 0.82, "semeval_polarity_accuracy": 0.90},
        "models": {},
        "vader": None,
    }
    vader = load(RES / "vader_semeval_results.json")
    if vader:
        metrics["vader"] = with_simple_confusion_key(vader)

    for key, title in MODELS.items():
        results = load(COLAB / f"{key}_goemotions_results.json")
        full = load(RES / f"goemotions_test_metrics_{key}.json")
        evaluation = load(EVAL / f"{key}_eval.json")
        if not (results or full or evaluation):
            continue
        entry = {"title": title}
        threshold = (results or full or {}).get("threshold")
        if results:
            entry["threshold"] = results["threshold"]
            entry["validation_macro_f1"] = results["validation_macro_f1"]
            entry["goemotions_views"] = results["views"]
        if full:
            entry["goemotions"] = full
        if threshold is not None:
            entry["goemotions_label_confusion"] = label_confusions(key, threshold)
        if evaluation:
            entry["semeval"] = with_simple_confusion_key(evaluation["semeval_polarity"])
            entry["isear"] = evaluation["isear_main"]
            entry["isear_confusion"] = isear_confusion(key)
        metrics["models"][key] = entry

    OUT.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Models included:", list(metrics["models"]))
    print("VADER included:", metrics["vader"] is not None)
    print("Missing inputs:", missing or "none")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()