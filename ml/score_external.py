"""Score a trained model on the SemEval polarity test set and the ISEAR held-out set.

Run in Colab from the folder that contains ml/ (rules are fixed in ml/results/evaluation_plan.md):
    python -m ml.score_external --name bert_goemotions --model-dir outputs/bert_goemotions_model
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ml.data_prep import labels as L

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "ml" / "data" / "processed"
MAX_LEN = 64
POSITIVE = ["admiration", "amusement", "approval", "caring", "desire", "excitement",
            "gratitude", "joy", "love", "optimism", "pride", "relief"]
NEGATIVE = ["anger", "annoyance", "disappointment", "disapproval", "disgust",
            "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness"]
ISEAR_MAIN = ["joy", "fear", "anger", "sadness", "disgust"]


def predict_probs(model, tokenizer, texts, device, batch_size=64):
    model.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], truncation=True, max_length=MAX_LEN,
                        padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out.append(torch.sigmoid(model(**enc).logits).float().cpu().numpy())
    return np.vstack(out)


def per_class(y_true, y_pred, labels):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {n: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, n in enumerate(labels)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = ROOT / "outputs" / "results_eval"
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    pos_idx = [L.LABEL_INDEX[n] for n in POSITIVE]
    neg_idx = [L.LABEL_INDEX[n] for n in NEGATIVE]

    # SemEval polarity: positive if the highest positive-label probability beats the highest negative one
    semeval = pd.read_parquet(PROC / "semeval2018_voc_binary" / "test.parquet")
    p = predict_probs(model, tokenizer, semeval["text"].tolist(), device)
    y_true = semeval["label_id"].to_numpy()
    y_pred = (p[:, pos_idx].max(axis=1) > p[:, neg_idx].max(axis=1)).astype(int)
    semeval_result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class": per_class(y_true, y_pred, [0, 1]),
        "confusion_matrix_rows_true_cols_pred_[neg,pos]": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    semeval_result["per_class"] = {"negative": semeval_result["per_class"][0], "positive": semeval_result["per_class"][1]}
    semeval.assign(pred=y_pred)[["id", "text", "label_id", "pred"]].to_csv(
        out / f"{args.name}_semeval_predictions.csv", index=False)

    # ISEAR: single predicted label = top-1 of 28; shame and guilt are not scored
    isear = pd.read_parquet(PROC / "isear" / "isear_clean.parquet")
    main_rows = isear[isear["in_main_eval"]].reset_index(drop=True)
    other_rows = isear[~isear["in_main_eval"]].reset_index(drop=True)
    p = predict_probs(model, tokenizer, main_rows["text"].tolist(), device)
    pred_names = np.array([L.LABELS[i] for i in p.argmax(axis=1)])
    true_names = main_rows["mapped_label"].to_numpy()
    isear_result = {
        "n": int(len(main_rows)),
        "accuracy": float((pred_names == true_names).mean()),
        "macro_f1_5_classes": float(f1_score(true_names, pred_names, labels=ISEAR_MAIN, average="macro", zero_division=0)),
        "predicted_outside_5_classes_share": float((~np.isin(pred_names, ISEAR_MAIN)).mean()),
        "per_class": per_class(true_names, pred_names, ISEAR_MAIN),
    }
    main_rows.assign(pred=pred_names)[["id", "text", "emotion_original", "pred"]].to_csv(
        out / f"{args.name}_isear_predictions.csv", index=False)

    p = predict_probs(model, tokenizer, other_rows["text"].tolist(), device)
    other_names = np.array([L.LABELS[i] for i in p.argmax(axis=1)])
    shame_guilt = {}
    for emotion in ["shame", "guilt"]:
        counts = pd.Series(other_names[(other_rows["emotion_original"] == emotion).to_numpy()]).value_counts()
        shame_guilt[emotion] = {k: int(v) for k, v in counts.head(6).items()}

    (out / f"{args.name}_eval.json").write_text(json.dumps(
        {"model": args.name, "semeval_polarity": semeval_result, "isear_main": isear_result,
         "isear_shame_guilt_top_predictions": shame_guilt}, indent=2), encoding="utf-8")
    print(f"SemEval polarity: accuracy {semeval_result['accuracy']:.4f} | macro F1 {semeval_result['macro_f1']:.4f}")
    print(f"ISEAR: accuracy {isear_result['accuracy']:.4f} | macro F1 {isear_result['macro_f1_5_classes']:.4f}")
    print("Saved to", out)


if __name__ == "__main__":
    main()