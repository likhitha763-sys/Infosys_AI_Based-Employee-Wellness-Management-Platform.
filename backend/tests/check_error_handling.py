"""Checks that errors become safe, friendly HTTP errors (Milestone 1, Step 11).

Run from the project root:
    python -m backend.tests.check_error_handling

Some red log lines will appear while it runs. That is expected: they are
the server log messages the real API would write for these errors.
"""

import sys
from unittest.mock import patch

from fastapi import HTTPException

from backend.app import main, sentiment

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


def get_http_error(function, *arguments):
    """Call main.run_analysis and return the HTTPException it raises (or None)."""
    try:
        main.run_analysis(function, *arguments)
    except HTTPException as exc:
        return exc
    return None


print("1. Whitespace-only text becomes a 400 error")
error = get_http_error(main.analyze_text, "   ")
check("status is 400", error is not None and error.status_code == 400)
check("message mentions empty text", error is not None and "empty" in error.detail.lower())

print("\n2. A blank entry in a batch names the entry")
error = get_http_error(main.analyze_batch, ["I am happy today", "   "])
check("status is 400", error is not None and error.status_code == 400)
check("message says 'Entry 2'", error is not None and "Entry 2" in error.detail)

print("\n3. Missing spaCy model becomes a 503 error")
sentiment._get_spacy_model.cache_clear()
with patch.object(sentiment.spacy, "load", side_effect=OSError("model missing")):
    error = get_http_error(main.analyze_text, "I am happy today")
sentiment._get_spacy_model.cache_clear()
check("status is 503", error is not None and error.status_code == 503)
check("message tells how to fix it", error is not None and "spacy download" in error.detail)

print("\n4. Missing NLTK tokenizer data becomes a 503 error")
with patch.object(sentiment, "word_tokenize", side_effect=LookupError("missing")):
    error = get_http_error(main.analyze_text, "I am happy today")
check("status is 503", error is not None and error.status_code == 503)
check("message tells how to fix it", error is not None and "nltk.downloader" in error.detail)

print("\n5. An unexpected error becomes a safe 500 error")
with patch.object(main, "analyze_text", side_effect=RuntimeError("secret internal detail")):
    error = get_http_error(main.analyze_text, "hello world")
check("status is 500", error is not None and error.status_code == 500)
check("internal details are not exposed", error is not None and "secret" not in error.detail)

print("\n6. Normal analysis still works after the patches")
result = sentiment.analyze_text("I am very happy today. Everything is going well!")
check("returns a label", result["sentiment"] in ("Positive", "Negative", "Neutral"))

print("\n" + "=" * 50)
print(f"RESULT: {passed} passed, {failed} failed")
print("=" * 50)
sys.exit(1 if failed else 0)