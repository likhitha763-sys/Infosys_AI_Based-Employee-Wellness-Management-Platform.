"""Checks that the fine-tuned engine is wired into emotion.py (no model download).

Run from the project root:
    python -m backend.tests.check_bert_integration
"""

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
os.environ["EMOTION_ENGINE"] = "finetuned"
os.environ["EMOTION_ENABLED"] = "true"

from backend.app import bert_engine, emotion  # noqa: E402

passed = 0
failed = 0


def check(description, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"PASS  {description}")
    else:
        failed += 1
        print(f"FAIL  {description}")


def fake_predict(texts):
    out = []
    for text in texts:
        if text == "ok":
            scores = [("approval", 0.76), ("neutral", 0.62), ("realization", 0.02)]
        else:
            scores = [("nervousness", 0.87), ("excitement", 0.82), ("fear", 0.61)]
        out.append({"scores": scores, "selected": [], "threshold": 0.7})
    return out


# 1. The fine-tuned engine is used, and the mixed-emotion result keeps both emotions.
bert_engine.predict = fake_predict
rows = emotion.attach_emotions([
    {"original_text": "I'm excited about the trip but also really nervous."},
    {"original_text": "ok"},
])
check("engine used is 'finetuned'", emotion.last_engine() == "finetuned")
check("mixed sentence: primary emotion is nervousness", rows[0]["predicted_emotion"] == "nervousness")
names = [item["emotion"] for item in rows[0]["top_emotions"]]
check("mixed sentence: excitement is also reported", "excitement" in names)
check("mixed sentence: no uncertainty note", rows[0]["emotion_note"] is None)
check("very short text gets an uncertainty note", "very short" in (rows[1]["emotion_note"] or ""))

# 2. If the fine-tuned engine is down, the pretrained model is tried; nothing crashes.
calls = {"pretrained": 0}


def fake_get_model():
    calls["pretrained"] += 1
    raise emotion.EmotionModelError("fake failure")


def unavailable(texts):
    raise bert_engine.EngineUnavailable("fake failure")


bert_engine.predict = unavailable
emotion._get_model = fake_get_model
rows = emotion.attach_emotions([{"original_text": "hello there my friend"}])
check("fallback tried the pretrained model", calls["pretrained"] == 1)
check("no crash: emotion empty, note explains why",
      rows[0]["predicted_emotion"] is None and "unavailable" in (rows[0]["emotion_note"] or ""))
check("the original result fields survive", rows[0]["original_text"] == "hello there my friend")

print(f"\n{passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)