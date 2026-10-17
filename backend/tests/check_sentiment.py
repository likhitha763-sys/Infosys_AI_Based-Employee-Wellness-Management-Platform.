"""Manual checks for backend/app/sentiment.py (Milestone 1, Step 5).

Run from the project root:
    python -m backend.tests.check_sentiment
"""

import sys

from backend.app.sentiment import (
    EmptyTextError,
    analyze_text,
    get_sentiment_label,
    preprocess_text,
)

# (name, text, the label we expect VADER to give)
SAMPLES = [
    ("Positive", "I am very happy today. Everything is going well!", "Positive"),
    ("Negative", "I feel very sad and disappointed today.", "Negative"),
    ("Neutral", "The meeting starts at 10 AM tomorrow.", "Neutral"),
    ("Negation", "I am not happy with the result.", "Negative"),
]

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


def expected_label_from_thresholds(compound):
    """Re-implemented here on purpose, so we test the app independently."""
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


print("=" * 60)
print("1. Sample sentences")
print("=" * 60)
for name, text, expected in SAMPLES:
    result = analyze_text(text)
    print(f"\n[{name}] {text}")
    print(f"  processed : {result['processed_text']!r}")
    print(
        f"  scores    : pos={result['positive_score']}  "
        f"neg={result['negative_score']}  neu={result['neutral_score']}  "
        f"compound={result['compound_score']}"
    )
    print(f"  label     : {result['sentiment']}")
    check(f"label is {expected}", result["sentiment"] == expected)
    check(
        "label follows the VADER thresholds",
        result["sentiment"] == expected_label_from_thresholds(result["compound_score"]),
    )
    total = (
        result["positive_score"] + result["negative_score"] + result["neutral_score"]
    )
    check("pos + neg + neu is about 1", abs(total - 1.0) < 0.01)
    check("original text is preserved", result["original_text"] == text)

print()
print("=" * 60)
print("2. Preprocessing stages (negation sentence)")
print("=" * 60)
steps = preprocess_text("I am not happy with the result.")
for key in ["cleaned_text", "tokens", "filtered_tokens", "lemmas", "processed_text"]:
    print(f"  {key:16}: {steps[key]!r}")
check("'not' survives stop-word filtering", "not" in steps["filtered_tokens"])
check("'not' is in the processed text", "not" in steps["processed_text"].split())

contraction = preprocess_text("I don't like this.")
print(f"\n  contraction test -> {contraction['cleaned_text']!r}")
check("contraction \"don't\" becomes 'do not'", "do not" in contraction["cleaned_text"])

print()
print("=" * 60)
print("3. Original text vs processed text scoring")
print("=" * 60)
text = "I am very happy today. Everything is going well!"
original_score = analyze_text(text)["compound_score"]
processed_score = analyze_text(text, use_processed_text=True)["compound_score"]
print(f"  compound (original text)  : {original_score}")
print(f"  compound (processed text) : {processed_score}")
print("  (the default scores the original text, as the project brief requires)")

print()
print("=" * 60)
print("4. Empty and whitespace-only input")
print("=" * 60)
for label, bad_text in [("empty string", ""), ("spaces and tabs", "   \n\t  ")]:
    try:
        analyze_text(bad_text)
        check(f"{label} is rejected", False)
    except EmptyTextError:
        check(f"{label} is rejected", True)

print()
print("=" * 60)
print("5. Threshold boundaries")
print("=" * 60)
check("0.05 -> Positive", get_sentiment_label(0.05) == "Positive")
check("0.049 -> Neutral", get_sentiment_label(0.049) == "Neutral")
check("0.0 -> Neutral", get_sentiment_label(0.0) == "Neutral")
check("-0.049 -> Neutral", get_sentiment_label(-0.049) == "Neutral")
check("-0.05 -> Negative", get_sentiment_label(-0.05) == "Negative")

print()
print("=" * 60)
print(f"RESULT: {passed} passed, {failed} failed")
print("=" * 60)
sys.exit(1 if failed else 0)