"""Emotion detection for MoodMentor.

Uses a pretrained text-emotion model (GoEmotions). This is separate from the
VADER sentiment score:
  - Sentiment says whether the tone is positive, negative, or neutral.
  - Emotion says which specific feeling the text seems to express.

This module contains no web code. FastAPI (main.py) calls it.
"""

import os
import threading
from pathlib import Path

from dotenv import load_dotenv

from .sentiment import EmptyTextError

# Read settings from backend/.env if the file exists.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

DEFAULT_MODEL_NAME = "SamLowe/roberta-base-go_emotions"
MAX_TOKENS = 512  # longer texts are cut off (truncated) at this length
BATCH_SIZE = 16  # texts scored together
MIN_CONFIDENCE = 0.30  # best score below this means "no clear emotion"
NO_CLEAR_EMOTION = "no_clear_emotion"
TOP_N = 3  # how many top emotions are returned with each result

HEART = "\u2764\ufe0f"  # red heart (written as an escape so it is always exact)

# The 28 labels of the GoEmotions model.
MODEL_LABELS = (
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
)

# Model label -> (name shown to the user, emoji).
# The first group follows the emoji table in the project brief.
# The second group are extra emotions the model can detect; their emojis
# were chosen for this project and can be changed freely.
EMOTION_DISPLAY = {
    # From the brief
    "joy": ("Happiness", "😊"),
    "sadness": ("Sadness", "😢"),
    "anger": ("Anger", "😠"),
    "fear": ("Fear", "😨"),
    "surprise": ("Surprise", "😮"),
    "disgust": ("Disgust", "🤢"),
    "love": ("Love", HEART),
    "excitement": ("Excitement", "🤩"),
    "confusion": ("Confusion", "🤔"),
    "embarrassment": ("Embarrassment", "😳"),
    "pride": ("Pride", "🥹"),
    "nervousness": ("Nervousness", "😰"),
    # Extra emotions the model can detect
    "admiration": ("Admiration", "👏"),
    "amusement": ("Amusement", "😄"),
    "annoyance": ("Annoyance", "😤"),
    "approval": ("Approval", "👍"),
    "caring": ("Caring", "🤗"),
    "curiosity": ("Curiosity", "🧐"),
    "desire": ("Desire", "✨"),
    "disappointment": ("Disappointment", "😞"),
    "disapproval": ("Disapproval", "👎"),
    "gratitude": ("Gratitude", "🙏"),
    "grief": ("Grief", "💔"),
    "optimism": ("Optimism", "🌈"),
    "realization": ("Realization", "💡"),
    "relief": ("Relief", "😅"),
    "remorse": ("Remorse", "🙇"),
    "neutral": ("Neutral", "😐"),
    # Used when the model is not sure
    NO_CLEAR_EMOTION: ("No clear emotion", "😶"),
}

# The 16 emotions requested for the project, and the model label that
# detects each one. None means the model cannot detect it.
REQUESTED_EMOTIONS = {
    "Happiness": "joy",
    "Sadness": "sadness",
    "Anger": "anger",
    "Fear": "fear",
    "Surprise": "surprise",
    "Disgust": "disgust",
    "Love": "love",
    "Excitement": "excitement",
    "Confusion": "confusion",
    "Jealousy": None,
    "Embarrassment": "embarrassment",
    "Pride": "pride",
    "Shame": None,
    "Anxiety": "nervousness",  # close, but not identical
    "Loneliness": None,
    "Contentment": None,
}


# ---------------------------------------------------------------------------
# Errors and settings
# ---------------------------------------------------------------------------


class EmotionModelError(RuntimeError):
    """Raised when the emotion model or its libraries cannot be loaded."""


class EmotionDisabledError(RuntimeError):
    """Raised when emotion detection is switched off in the settings."""


def is_enabled():
    """True unless EMOTION_ENABLED is set to false/0/no/off in backend/.env."""
    value = os.getenv("EMOTION_ENABLED", "true").strip().lower()
    return value not in ("false", "0", "no", "off")


def get_model_name():
    """The Hugging Face model to use (can be changed in backend/.env)."""
    return os.getenv("EMOTION_MODEL_NAME", DEFAULT_MODEL_NAME)


# ---------------------------------------------------------------------------
# Display helpers and choosing the primary emotion (no model needed)
# ---------------------------------------------------------------------------


def describe_emotion(label):
    """Return (display name, emoji) for a label. Unknown labels get a safe fallback."""
    if label in EMOTION_DISPLAY:
        return EMOTION_DISPLAY[label]
    return (label.replace("_", " ").capitalize(), "❓")


def pick_primary_emotion(probabilities, min_confidence=MIN_CONFIDENCE, top_n=TOP_N):
    """Turn a {label: score} dictionary into one result.

    The primary emotion is the highest-scoring label. If even that score is
    below min_confidence, the result is "no clear emotion" and the confidence
    is None, because the model has no score for "no clear emotion".
    """
    if not probabilities:
        raise ValueError("No emotion scores were given.")

    # Highest score first; ties are broken alphabetically so results are stable.
    ranked = sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))
    top_label, top_score = ranked[0]

    if top_score < min_confidence:
        label, confidence = NO_CLEAR_EMOTION, None
    else:
        label, confidence = top_label, round(top_score, 4)

    name, emoji = describe_emotion(label)
    return {
        "predicted_emotion": label,
        "emotion_name": name,
        "emotion_emoji": emoji,
        "emotion_confidence": confidence,
        "top_emotions": [
            {"emotion": item_label, "score": round(score, 4)}
            for item_label, score in ranked[:top_n]
        ],
    }


# ---------------------------------------------------------------------------
# Loading the model (done once, then reused)
# ---------------------------------------------------------------------------

_model_bundle = None  # (tokenizer, model, labels) once loaded
_load_lock = threading.Lock()


def _load_model():
    """Load the tokenizer and model. Raises EmotionModelError on any problem."""
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise EmotionModelError(
            "The emotion libraries are not installed. "
            "Run: python -m pip install torch transformers"
        ) from exc

    name = get_model_name()
    try:
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name)
    except Exception as exc:  # any download or loading problem
        raise EmotionModelError(
            f"The emotion model '{name}' could not be loaded. The first run needs "
            "an internet connection to download it. Check your connection and try again."
        ) from exc

    model.eval()
    labels = [model.config.id2label[index] for index in range(model.config.num_labels)]
    return tokenizer, model, labels


def _get_model():
    """Return the loaded model, loading it on first use (safe with several threads)."""
    global _model_bundle
    if _model_bundle is None:
        with _load_lock:
            if _model_bundle is None:
                _model_bundle = _load_model()
    return _model_bundle


def reset_model_cache():
    """Forget the loaded model (used by tests)."""
    global _model_bundle
    _model_bundle = None


def _predict_probabilities(texts):
    """Score every text. Returns one {label: probability} dictionary per text."""
    tokenizer, model, labels = _get_model()
    import torch

    results = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS,
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        # The model is multi-label: each emotion gets its own 0-1 probability.
        probabilities = torch.sigmoid(logits)
        for row in probabilities:
            results.append({label: float(value) for label, value in zip(labels, row)})
    return results


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def _require_enabled():
    if not is_enabled():
        raise EmotionDisabledError("Emotion detection is turned off (EMOTION_ENABLED).")


def classify_emotion(text):
    """Detect the primary emotion of one text.

    Raises EmotionDisabledError, EmptyTextError, TypeError, or EmotionModelError.
    Texts longer than 512 tokens are cut off at that length.
    """
    _require_enabled()
    if not isinstance(text, str):
        raise TypeError("Text must be a string.")
    if not text.strip():
        raise EmptyTextError("Text must not be empty or only whitespace.")
    return pick_primary_emotion(_predict_probabilities([text])[0])


def classify_emotions(texts):
    """Detect the primary emotion of each text; results are in the same order.

    Raises EmptyTextError (naming the entry number) if any entry is blank.
    """
    _require_enabled()
    for position, text in enumerate(texts, start=1):
        if not isinstance(text, str):
            raise TypeError("Every text must be a string.")
        if not text.strip():
            raise EmptyTextError(f"Entry {position} is empty or only whitespace.")
    if not texts:
        return []
    return [pick_primary_emotion(scores) for scores in _predict_probabilities(list(texts))]