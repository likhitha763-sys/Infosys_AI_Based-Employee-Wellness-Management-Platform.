"""Fine-tuned BERT emotion engine (Milestone 2).

Loads the MoodMentor BERT model from the Hugging Face Hub the first time it is
needed. This module contains no web code. If the model cannot be loaded,
EngineUnavailable is raised so the caller can fall back to another engine.
"""

import json
import logging
import os
import threading

logger = logging.getLogger("moodmentor")

REPO_ID = os.getenv("MOODMENTOR_BERT_REPO", "ramesh-t/moodmentor-bert-goemotions")
HF_TOKEN = os.getenv("HF_TOKEN") or None  # only needed while the repo is private
MAX_LENGTH = 64  # the model was trained and evaluated with this length
DEFAULT_THRESHOLD = 0.7
BATCH_SIZE = 16


class EngineUnavailable(RuntimeError):
    """The BERT model could not be loaded."""


_lock = threading.Lock()
_state = {}


def _load():
    """Load the model once. A failed load is remembered until the app restarts."""
    with _lock:
        if "error" in _state:
            raise EngineUnavailable(_state["error"])
        if "model" in _state:
            return _state
        try:
            from huggingface_hub import hf_hub_download
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(REPO_ID, token=HF_TOKEN)
            model = AutoModelForSequenceClassification.from_pretrained(
                REPO_ID, token=HF_TOKEN
            ).eval()

            threshold = DEFAULT_THRESHOLD
            try:
                path = hf_hub_download(REPO_ID, "moodmentor_config.json", token=HF_TOKEN)
                with open(path, encoding="utf-8") as handle:
                    threshold = float(json.load(handle)["threshold"])
            except Exception:
                logger.warning("No threshold file found; using %s", DEFAULT_THRESHOLD)

            labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
            _state.update(model=model, tokenizer=tokenizer, threshold=threshold, labels=labels)
            return _state
        except Exception as exc:
            _state["error"] = f"{type(exc).__name__}: {exc}"
            logger.warning("BERT engine unavailable: %s", _state["error"])
            raise EngineUnavailable(_state["error"]) from exc


def warm_up():
    """Load the model now, so the first request is not slow."""
    _load()


def predict(texts):
    """Score texts. Returns one dict per text:

    {"scores": [(label, probability), ...] sorted high to low,
     "selected": [labels at or above the threshold],
     "threshold": the threshold used}
    """
    import torch

    state = _load()
    results = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        encoded = state["tokenizer"](
            batch, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt"
        )
        with torch.no_grad():
            probs = torch.sigmoid(state["model"](**encoded).logits).numpy()
        for row in probs:
            order = row.argsort()[::-1]
            scores = [(state["labels"][i], float(row[i])) for i in order]
            results.append({
                "scores": scores,
                "selected": [label for label, score in scores if score >= state["threshold"]],
                "threshold": state["threshold"],
            })
    return results