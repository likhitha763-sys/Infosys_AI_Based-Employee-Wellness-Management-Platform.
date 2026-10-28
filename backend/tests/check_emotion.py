"""Checks for backend/app/emotion.py that do not need the real model.

They use fake scores, so they are fast and give the same result every time.
Run from the project root:
    python -m backend.tests.check_emotion
"""

import os
import sys
from unittest.mock import patch

from backend.app import emotion
from backend.app.emotion import (
    EmotionDisabledError,
    EmotionModelError,
    MODEL_LABELS,
    REQUESTED_EMOTIONS,
    describe_emotion,
    pick_primary_emotion,
)
from backend.app.sentiment import EmptyTextError

passed = 0
failed = 0


def check(description, condition):
    """Print PASS or FAIL for one check and keep count."""
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {description}")
    else:
        failed += 1
        print(f"  FAIL  {description}")


def raises(error_type, function, *arguments):
    """True if calling the function raises the given error type."""
    try:
        function(*arguments)
    except error_type:
        return True
    except Exception:
        return False
    return False


def fake_scores(texts):
    """Pretend model: texts starting with 'happy' are joy, the rest sadness."""
    return [
        {"joy": 0.9, "sadness": 0.05, "neutral": 0.1}
        if text.startswith("happy")
        else {"joy": 0.05, "sadness": 0.8, "neutral": 0.1}
        for text in texts
    ]


print("1. Labels, names, and emojis")
check(
    "all 28 model labels have a display name and emoji",
    len(MODEL_LABELS) == 28 and all(label in emotion.EMOTION_DISPLAY for label in MODEL_LABELS),
)
emojis = [emoji for _, emoji in emotion.EMOTION_DISPLAY.values()]
check("no two labels share an emoji", len(set(emojis)) == len(emojis))
check("joy is shown as Happiness with a smiling face", describe_emotion("joy") == ("Happiness", "😊"))
brief_emojis = {
    "Happiness": "😊", "Sadness": "😢", "Anger": "😠", "Fear": "😨",
    "Surprise": "😮", "Disgust": "🤢", "Love": "\u2764\ufe0f", "Excitement": "🤩",
    "Confusion": "🤔", "Embarrassment": "😳", "Pride": "🥹", "Anxiety": "😰",
}
check(
    "the 12 supported emojis match the project brief",
    all(describe_emotion(REQUESTED_EMOTIONS[name])[1] == emoji for name, emoji in brief_emojis.items()),
)
check("an unknown label gets a safe fallback", describe_emotion("mystery") == ("Mystery", "❓"))
mapped = [label for label in REQUESTED_EMOTIONS.values() if label]
check(
    "16 emotions were requested and 12 map to real model labels",
    len(REQUESTED_EMOTIONS) == 16 and len(mapped) == 12 and all(label in MODEL_LABELS for label in mapped),
)
unsupported = {name for name, label in REQUESTED_EMOTIONS.items() if label is None}
check(
    "exactly Jealousy, Shame, Loneliness, and Contentment are unsupported",
    unsupported == {"Jealousy", "Shame", "Loneliness", "Contentment"},
)

print("\n2. Choosing the primary emotion")
result = pick_primary_emotion({"joy": 0.9, "excitement": 0.4, "neutral": 0.05, "anger": 0.01})
check(
    "the highest score wins, with name, emoji, and confidence",
    result["predicted_emotion"] == "joy" and result["emotion_name"] == "Happiness"
    and result["emotion_emoji"] == "😊" and result["emotion_confidence"] == 0.9,
)
check(
    "the top 3 emotions are listed, best first",
    [item["emotion"] for item in result["top_emotions"]] == ["joy", "excitement", "neutral"],
)
check("neutral can be the primary emotion", pick_primary_emotion({"neutral": 0.8, "joy": 0.1})["predicted_emotion"] == "neutral")
result = pick_primary_emotion({"joy": 0.2, "sadness": 0.1})
check(
    "a best score below the cutoff gives 'no clear emotion' and no confidence",
    result["predicted_emotion"] == emotion.NO_CLEAR_EMOTION and result["emotion_confidence"] is None,
)
check("a score exactly at the cutoff is accepted", pick_primary_emotion({"joy": 0.30})["predicted_emotion"] == "joy")
check("ties are broken alphabetically", pick_primary_emotion({"sadness": 0.6, "anger": 0.6})["predicted_emotion"] == "anger")
check("an empty score dictionary is rejected", raises(ValueError, pick_primary_emotion, {}))

print("\n3. Classifying texts (with fake scores)")
with patch.object(emotion, "_predict_probabilities", side_effect=fake_scores):
    results = emotion.classify_emotions(["happy day", "sad day"])
    check("one result per text, in the same order", len(results) == 2)
    check(
        "each result matches its text's scores",
        results[0]["predicted_emotion"] == "joy" and results[1]["predicted_emotion"] == "sadness",
    )
    check("an empty list gives an empty list", emotion.classify_emotions([]) == [])
    check("a blank entry is rejected and named", raises(EmptyTextError, emotion.classify_emotions, ["happy day", "   "]))
    check("one blank text is rejected", raises(EmptyTextError, emotion.classify_emotion, "   "))
    check("a non-text entry is rejected", raises(TypeError, emotion.classify_emotions, [123]))

print("\n4. Switching emotion detection off")
with patch.dict(os.environ, {"EMOTION_ENABLED": "false"}):
    check("EMOTION_ENABLED=false turns it off", not emotion.is_enabled())
    check("classifying while off raises a clear error", raises(EmotionDisabledError, emotion.classify_emotion, "happy day"))
with patch.dict(os.environ):
    os.environ.pop("EMOTION_ENABLED", None)
    check("it is on by default", emotion.is_enabled())

print("\n5. Missing model or libraries")
emotion.reset_model_cache()
with patch("transformers.AutoTokenizer.from_pretrained", side_effect=OSError("no connection")):
    try:
        emotion.classify_emotion("happy day")
        error = None
    except EmotionModelError as exc:
        error = str(exc)
check("a model that cannot load raises EmotionModelError", error is not None)
check("the message mentions the internet connection", error is not None and "internet" in error.lower())
check("a failed load is not remembered as a loaded model", emotion._model_bundle is None)
with patch.dict(sys.modules, {"transformers": None}):
    try:
        emotion._load_model()
        error = None
    except EmotionModelError as exc:
        error = str(exc)
check("missing libraries give an install instruction", error is not None and "pip install" in error)
emotion.reset_model_cache()

print("\n" + "=" * 50)
print(f"RESULT: {passed} passed, {failed} failed")
print("=" * 50)
sys.exit(1 if failed else 0)