"""Fine-tune a transformer for 28-label multi-label GoEmotions. Needs a GPU (Google Colab).

Run from the folder that contains the ml/ package (after unzipping moodmentor_colab_bundle.zip):
    python -m ml.train_goemotions --model distilbert-base-uncased --name distilbert_goemotions
    python -m ml.train_goemotions --model bert-base-uncased --name bert_goemotions

Settings are the ones used for the Milestone 2 results: seed 42, max length 64,
learning rate 5e-5, batch size 32, 4 epochs, weight decay 0.01, 10% warmup,
square-root positive class weights capped at 10, global threshold chosen on the
validation set only, test set scored once.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
    Trainer, TrainingArguments, set_seed,
)

from ml.data_prep import labels as L

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ml" / "data" / "processed" / "goemotions"
SEED, MAX_LEN, BATCH, EPOCHS, LR = 42, 64, 32, 4, 5e-5


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load_split(name):
    frame = pd.read_parquet(DATA / f"{name}.parquet")
    y = L.multi_hot([list(x) for x in frame["labels"]]).astype("float32")
    return frame["text"].tolist(), y


class WeightedBCETrainer(Trainer):
    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.float()
        loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))(logits, labels.float())
        return (loss, outputs) if return_outputs else loss


def summary(name, y_true, y_pred):
    return {
        "view": name,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="e.g. bert-base-uncased")
    parser.add_argument("--name", required=True, help="output name, e.g. bert_goemotions")
    args = parser.parse_args()
    set_seed(SEED)
    out = ROOT / "outputs"
    (out / "results").mkdir(parents=True, exist_ok=True)
    print("GPU available:", torch.cuda.is_available())

    train_text, y_train = load_split("train")
    val_text, y_val = load_split("validation_clean")
    test_text, y_test = load_split("test_clean")
    print("sizes:", len(train_text), len(val_text), len(test_text))

    positives = y_train.sum(axis=0)
    negatives = len(y_train) - positives
    weights = np.minimum(np.sqrt(negatives / np.maximum(positives, 1)), 10.0)
    pos_weight = torch.tensor(weights, dtype=torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def make_ds(texts, y):
        ds = Dataset.from_dict({"text": texts, "labels": y.tolist()})
        return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=MAX_LEN),
                      batched=True, remove_columns=["text"])

    train_ds, val_ds, test_ds = make_ds(train_text, y_train), make_ds(val_text, y_val), make_ds(test_text, y_test)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=L.NUM_LABELS, problem_type="multi_label_classification",
        id2label={i: n for i, n in enumerate(L.LABELS)},
        label2id={n: i for i, n in enumerate(L.LABELS)},
    )

    def compute_metrics(eval_pred):
        logits, y = eval_pred
        preds = (sigmoid(logits) >= 0.5).astype(int)
        y = y.astype(int)
        return {"macro_f1": f1_score(y, preds, average="macro", zero_division=0),
                "micro_f1": f1_score(y, preds, average="micro", zero_division=0)}

    warmup = int(0.1 * math.ceil(len(train_ds) / BATCH) * EPOCHS)
    training_args = TrainingArguments(
        output_dir=str(out / f"{args.name}_checkpoints"), eval_strategy="epoch",
        save_strategy="epoch", save_total_limit=2, learning_rate=LR,
        per_device_train_batch_size=BATCH, per_device_eval_batch_size=64,
        num_train_epochs=EPOCHS, weight_decay=0.01, warmup_steps=warmup,
        logging_steps=200, load_best_model_at_end=True, metric_for_best_model="macro_f1",
        greater_is_better=True, fp16=torch.cuda.is_available(), seed=SEED, report_to="none",
    )
    trainer = WeightedBCETrainer(
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics, pos_weight=pos_weight,
    )
    trainer.train()

    # Threshold chosen on the VALIDATION set only
    val_prob = sigmoid(trainer.predict(val_ds).predictions)
    y_val_i = y_val.astype(int)
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.10, 0.91, 0.05):
        score = f1_score(y_val_i, (val_prob >= t).astype(int), average="macro", zero_division=0)
        print(f"threshold {t:.2f}: validation macro F1 = {score:.4f}")
        if score > best_f1:
            best_t, best_f1 = float(round(t, 2)), float(score)
    print(f"Chosen threshold: {best_t} (validation macro F1 {best_f1:.4f})")

    # Test set scored once
    test_prob = sigmoid(trainer.predict(test_ds).predictions)
    y_test_i = y_test.astype(int)
    test_pred = (test_prob >= best_t).astype(int)
    results = {
        "model": args.model, "threshold": best_t, "validation_macro_f1": best_f1,
        "views": [
            summary("28 labels", y_test_i, test_pred),
            summary("Ekman 6 + neutral", L.to_ekman_view(y_test_i), L.to_ekman_view(test_pred)),
            summary("12 headline emotions", L.to_headline_view(y_test_i), L.to_headline_view(test_pred)),
        ],
        "per_label_f1": {n: float(f) for n, f in zip(
            L.LABELS, f1_score(y_test_i, test_pred, average=None, zero_division=0))},
    }
    for view in results["views"]:
        print(f"{view['view']:<24} macro F1 {view['macro_f1']:.4f} | micro F1 {view['micro_f1']:.4f}")

    np.save(out / "results" / f"{args.name}_test_prob.npy", test_prob)
    (out / "results" / f"{args.name}_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    model_dir = out / f"{args.name}_model"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    print("Saved model to", model_dir)


if __name__ == "__main__":
    main()