"""Checks for the page logic in frontend/analysis_page.py (no browser needed).

Run from the project root:
    python -m frontend.tests.check_analysis_page
"""

import sys
from pathlib import Path

import pandas as pd

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


def says(message, fragment):
    """True if the message exists and contains the fragment (ignoring case)."""
    return message is not None and fragment in message.lower()


def report(label, compound, positive=0.0, negative=0.0, processed="word"):
    """Build a fake backend result and return its report sentences as one string."""
    result = {
        "sentiment": label,
        "compound_score": compound,
        "positive_score": positive,
        "negative_score": negative,
        "processed_text": processed,
    }
    return " ".join(page.build_report(result))


print("1. Decoding text")
check("UTF-8 text decodes", page.decode_text(b"hello") == "hello")
check("a UTF-8 BOM is removed", page.decode_text(b"\xef\xbb\xbfhello") == "hello")
check("Windows-1252 text decodes", page.decode_text("café".encode("cp1252")) == "café")
check("undecodable bytes give None", page.decode_text(b"abc\x81\x9d") is None)

print("\n2. TXT files")
entries, _ = page.read_txt_entries(b"line one\nline two", False)
check("whole file is one entry", entries == ["line one\nline two"])
entries, _ = page.read_txt_entries(b"a\n\nb\n", True)
check("one entry per line splits the lines", entries == ["a", "", "b"])
_, error = page.read_txt_entries(b"   \n  ", False)
check("a whitespace-only file is rejected", says(error, "no text"))
_, error = page.read_txt_entries(b"abc\x81", True)
check("an unreadable file gives a UTF-8 hint", says(error, "utf-8"))

print("\n3. CSV files")
table, error = page.read_csv_table(b"date,entry\n2026-10-01,Hello there\n2026-10-02,Another one\n")
check("a valid CSV has the right columns", error is None and list(table.columns) == ["date", "entry"])
check("a valid CSV has the right number of rows", len(table) == 2)
table, _ = page.read_csv_table(b"\xef\xbb\xbfentry\nhello\n")
check("a BOM does not spoil the first column name", list(table.columns) == ["entry"])
_, error = page.read_csv_table(b"")
check("an empty CSV is rejected", says(error, "empty"))
_, error = page.read_csv_table(b"entry,date\n")
check("a CSV with headers only is rejected", says(error, "no data rows"))
_, error = page.read_csv_table(b"a,b\n1,2\n3,4,5,6\n")
check("a malformed CSV is rejected", says(error, "malformed"))
table, _ = page.read_csv_table("entry\ncafé\n".encode("cp1252"))
check("a Windows-1252 CSV is read correctly", table["entry"].iloc[0] == "café")

print("\n4. Finding the text column")
check("a column of words has text", page.column_has_text(pd.Series(["hello", "world"], dtype=str)))
check("a column of numbers is not text", not page.column_has_text(pd.Series(["1", " 2.5 ", "3"], dtype=str)))
check("a blank column is not text", not page.column_has_text(pd.Series(["", "  "], dtype=str)))
check("numbers mixed with words count as text", page.column_has_text(pd.Series(["1", "hello"], dtype=str)))
table = pd.DataFrame({"date": ["2026-10-01"], "entry": ["hi"], "rating": ["a"]})
check("a column named 'entry' is chosen by name", page.guess_text_column(table, list(table.columns)) == "entry")
table = pd.DataFrame({"a": ["x", "y"], "b": ["a much longer piece of text", "another long piece of text"]})
check("otherwise the column with the longest text is chosen", page.guess_text_column(table, ["a", "b"]) == "b")

print("\n5. Preparing entries")
entries, blank, too_long = page.prepare_entries(["  a  ", "", "   ", "x" * 10_001, "ok"])
check("blank and too-long entries are removed", entries == ["a", "ok"])
check("blank entries are counted", blank == 2)
check("too-long entries are counted", too_long == 1)

print("\n6. Text validation")
check("empty text is rejected", says(page.validate_text(""), "enter some text"))
check("whitespace-only text is rejected", says(page.validate_text("   "), "enter some text"))
check("text over 10,000 characters is rejected", says(page.validate_text("x" * 10_001), "too long"))
check("normal text is accepted", page.validate_text("I am happy") is None)

print("\n7. Report wording")
check("a high positive score says 'strongly positive'", "strongly positive" in report("Positive", 0.8))
check("a medium negative score says 'moderately negative'", "moderately negative" in report("Negative", -0.35))
text = report("Positive", 0.1)
check("a weak score says 'mildly' and mentions a weak signal", "mildly positive" in text and "weak signal" in text)
check("a clear neutral score has just one sentence", len(page.build_report({"sentiment": "Neutral", "compound_score": 0.0, "positive_score": 0.0, "negative_score": 0.0, "processed_text": "meeting"})) == 1)
check("a neutral score near the edge adds a warning", "edge of the neutral range" in report("Neutral", 0.04))
check("both positive and negative wording mentions mixed feelings", "mixed" in report("Positive", 0.4, positive=0.3, negative=0.2))
check("empty processed text adds a caution", "very little ordinary wording" in report("Neutral", 0.0, processed=""))

print("\n" + "=" * 50)
print(f"RESULT: {passed} passed, {failed} failed")
print("=" * 50)
sys.exit(1 if failed else 0)