"""Checks for the emotion display helpers in frontend/analysis_page.py.

They use made-up results shaped like the API's responses.
Run from the project root:
    python -m frontend.tests.check_emotion_display
"""

import sys
from pathlib import Path

# The frontend modules import each other by plain name, so add their folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analysis_page as page  # noqa: E402

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


def result(text="I am happy", compound=0.6, sentiment="Positive", **emotion_fields):
    """A made-up API result. Emotion fields default to 'none saved'."""
    base = {
        "original_text": text, "processed_text": text.lower(), "positive_score": 0.5,
        "negative_score": 0.0, "neutral_score": 0.5, "compound_score": compound,
        "sentiment": sentiment, "created_at": "2026-10-01T10:00:00Z",
        "predicted_emotion": None, "emotion_name": None, "emotion_emoji": None,
        "emotion_confidence": None, "top_emotions": None, "emotion_note": None,
    }
    base.update(emotion_fields)
    return base


JOY = result(
    predicted_emotion="joy", emotion_name="Happiness", emotion_emoji="😊", emotion_confidence=0.934,
    top_emotions=[{"emotion": "joy", "score": 0.934}, {"emotion": "excitement", "score": 0.31}, {"emotion": "neutral", "score": 0.05}],
)
NEUTRAL = result("The meeting is at 10 AM.", 0.0, "Neutral", predicted_emotion="neutral", emotion_name="Neutral", emotion_emoji="😐", emotion_confidence=0.8)
UNSURE = result("hmm", 0.0, "Neutral", predicted_emotion="no_clear_emotion", emotion_name="No clear emotion", emotion_emoji="😶")
OLD = result("An old entry")

print("1. Emotion and confidence text")
check("an emotion shows its emoji and name", page.emotion_text(JOY) == "😊 Happiness")
check("a result without an emotion says 'Not analyzed'", page.emotion_text(OLD) == "Not analyzed")
check("confidence is a whole-number percentage", page.confidence_text(JOY) == "93%")
check("no confidence shows a dash", page.confidence_text(UNSURE) == "—")

print("\n2. Explanations")
text = page.emotion_explanation(JOY)
check("a real emotion is named, called separate from sentiment, and said to be fallible", "Happiness" in text and "separate" in text and "can be wrong" in text)
check("neutral is explained as no strong emotion", "no strong emotion" in page.emotion_explanation(NEUTRAL))
check("'no clear emotion' is explained as low confidence", "not confident" in page.emotion_explanation(UNSURE))
check("a backend note is shown when there is no emotion", page.emotion_explanation(result(emotion_note="Emotion detection is turned off.")) == "Emotion detection is turned off.")
check("an old entry is explained", "before emotion detection" in page.emotion_explanation(OLD))

print("\n3. Top model scores")
check("fresh results list the top scores", page.top_emotions_text(JOY) == "Top model scores: joy 0.93, excitement 0.31, neutral 0.05")
check("saved results have none to show", page.top_emotions_text(OLD) is None and page.top_emotions_text(result(top_emotions=[])) is None)

print("\n4. Tables")
table = page.results_table([JOY, OLD])
check(
    "the results table has the emotion columns next to the sentiment columns",
    list(table.columns) == ["#", "Text", "Emotion", "Confidence", "Sentiment", "Positive", "Negative", "Neutral", "Compound"],
)
check(
    "emotion cells show the emotion, or 'Not analyzed' and a dash",
    list(table["Emotion"]) == ["😊 Happiness", "Not analyzed"] and list(table["Confidence"]) == ["93%", "—"],
)
check("sentiment scores are unchanged", list(table["Compound"]) == [0.6, 0.6] and list(table["Sentiment"]) == ["Positive", "Positive"])
history = page.history_table([JOY, OLD])
check("the history table has an Emotion column", list(history.columns) == ["Date", "Text", "Emotion", "Sentiment", "Compound"])
check("history emotion cells and dates are readable", list(history["Emotion"]) == ["😊 Happiness", "Not analyzed"] and history["Date"].iloc[0] == "2026-10-01 10:00 UTC")
long_entry = result("x" * 200)
check("long text is shortened in history", len(page.history_table([long_entry])["Text"].iloc[0]) == 80)

print("\n" + "=" * 50)
print(f"RESULT: {passed} passed, {failed} failed")
print("=" * 50)
sys.exit(1 if failed else 0)