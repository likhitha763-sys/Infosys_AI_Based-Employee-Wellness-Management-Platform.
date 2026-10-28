"""Checks the REAL emotion model: valid output structure, not accuracy.

You judge whether the emotions make sense by reading the printed output.
The first run may download the model, so it can take a while.

Run from the project root:
    python -m backend.tests.check_emotion_model
"""

import sys
import time

sys.stdout.reconfigure(encoding="utf-8")  # so emojis print on Windows

from backend.app import emotion  # noqa: E402

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


SAMPLES = [
    "I am very happy today. Everything is going well!",
    "I feel very sad and disappointed today.",
    "The meeting starts at 10 AM tomorrow.",
    "I am not happy with the result.",
    "I'm excited about the new job but also scared of failing.",
]
KNOWN_LABELS = set(emotion.MODEL_LABELS) | {emotion.NO_CLEAR_EMOTION}


def valid_structure(result):
    """True if the result has the expected fields with sensible values."""
    confidence = result["emotion_confidence"]
    scores = [item["score"] for item in result["top_emotions"]]
    return (
        result["predicted_emotion"] in KNOWN_LABELS
        and (confidence is None or 0 <= confidence <= 1)
        and len(result["top_emotions"]) == emotion.TOP_N
        and all(0 <= score <= 1 for score in scores)
        and scores == sorted(scores, reverse=True)
        and bool(result["emotion_name"])
        and bool(result["emotion_emoji"])
    )


print("Loading the model (the first run downloads it)...")
began = time.time()
try:
    emotion.classify_emotion("warm-up")
    loaded = True
except emotion.EmotionModelError as exc:
    loaded = False
    print("Could not load the model:", exc)
print(f"Load time: {time.time() - began:.1f} seconds\n")

check("the model loads", loaded)
if not loaded:
    print(f"\nRESULT: {passed} passed, {failed} failed")
    sys.exit(1)

singles = []
for text in SAMPLES:
    result = emotion.classify_emotion(text)
    singles.append(result)
    confidence = result["emotion_confidence"]
    shown = "none" if confidence is None else f"{confidence:.3f}"
    print(f'\n"{text}"')
    print(f"  Emotion: {result['emotion_emoji']} {result['emotion_name']}   (confidence: {shown})")
    print("  Top 3:", ", ".join(f"{item['emotion']} {item['score']:.3f}" for item in result["top_emotions"]))
    check("valid result structure", valid_structure(result))

batch = emotion.classify_emotions(SAMPLES)
same_labels = [a["predicted_emotion"] for a in batch] == [b["predicted_emotion"] for b in singles]
close_scores = all(
    abs(a["top_emotions"][0]["score"] - b["top_emotions"][0]["score"]) < 0.02
    for a, b in zip(batch, singles)
)
check("a batch gives the same labels and nearly the same scores as single texts", same_labels and close_scores)

began = time.time()
emotion.classify_emotions(SAMPLES * 4)
print(f"\nSpeed: {1000 * (time.time() - began) / (len(SAMPLES) * 4):.0f} ms per text (batch of {len(SAMPLES) * 4})")

print("\n" + "=" * 50)
print(f"RESULT: {passed} passed, {failed} failed")
print("=" * 50)
sys.exit(1 if failed else 0)