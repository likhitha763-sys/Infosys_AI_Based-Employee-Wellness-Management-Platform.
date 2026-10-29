"""Checks for frontend/charts.py and the text helpers in dashboard_page.py.

They use made-up dashboard data and check the chart contents. Whether the
charts look good is checked by eye (see the manual test).

Run from the project root:
    python -m frontend.tests.check_charts
"""

import sys
from pathlib import Path

# The frontend modules import each other by plain name, so add their folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import charts  # noqa: E402
import dashboard_page as page  # noqa: E402

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


def point(number, score, sentiment, text, name=None, emoji=None):
    return {
        "number": number, "created_at": f"2026-10-01T10:{number:02d}:00Z",
        "compound_score": score, "sentiment": sentiment, "text_preview": text,
        "emotion_name": name, "emotion_emoji": emoji,
    }


def emotion(label, name, emoji, count, is_emotion=True):
    return {"emotion": label, "name": name, "emoji": emoji, "count": count, "percent": 0.0, "is_emotion": is_emotion}


RHYTHM = [
    point(1, 0.8, "Positive", "I am very happy today.", "Happiness", "😊"),
    point(2, -0.6, "Negative", "I feel very sad.", "Sadness", "😢"),
    point(3, 0.0, "Neutral", "The meeting is at 10 AM."),
    point(4, 0.5, "Positive", "<b>hi</b> there", "Happiness", "😊"),
]
EMOTIONS = [
    emotion("joy", "Happiness", "😊", 2),
    emotion("neutral", "Neutral", "😐", 3, False),
    emotion("sadness", "Sadness", "😢", 1),
    emotion("excitement", "Excitement", "🤩", 1),
    emotion("no_clear_emotion", "No clear emotion", "😶", 1, False),
]
SENTIMENTS = [
    {"sentiment": "Negative", "count": 1, "percent": 25.0},
    {"sentiment": "Positive", "count": 2, "percent": 50.0},
    {"sentiment": "Neutral", "count": 1, "percent": 25.0},
]

print("1. Emotional rhythm (line chart)")
figure = charts.build_rhythm_figure(RHYTHM)
trace = figure.data[0]
check("one line with markers", len(figure.data) == 1 and trace.mode == "lines+markers")
check("x is the entry number and y is the compound score", list(trace.x) == [1, 2, 3, 4] and list(trace.y) == [0.8, -0.6, 0.0, 0.5])
check(
    "the y axis covers -1 to +1 with ticks at both ends",
    figure.layout.yaxis.range[0] <= -1 and figure.layout.yaxis.range[1] >= 1
    and -1 in figure.layout.yaxis.tickvals and 1 in figure.layout.yaxis.tickvals,
)
check("a zero reference line is drawn", any(shape.y0 == 0 and shape.y1 == 0 for shape in figure.layout.shapes))
check("markers differ by sentiment in shape", set(trace.marker.symbol) == {"circle", "square", "diamond"})
rows = [list(row) for row in trace.customdata]
check(
    "hover details hold the date, sentiment, emotion, and text",
    rows[0] == ["2026-10-01 10:01 UTC", "Positive", "😊 Happiness", "I am very happy today."]
    and "Compound score" in trace.hovertemplate,
)
check("an entry without an emotion says 'Not analyzed'", rows[2][2] == "Not analyzed")
check("HTML in entry text is escaped", rows[3][3] == "&lt;b&gt;hi&lt;/b&gt; there")
check("no entries gives no chart", charts.build_rhythm_figure([]) is None)

print("\n2. Emotion mix (doughnut) and emotion counts (bars)")
pie = charts.build_emotion_doughnut(EMOTIONS).data[0]
check("a doughnut with only real emotions by default", pie.type == "pie" and pie.hole > 0 and list(pie.values) == [2, 1, 1])
check("slice labels hold the emoji and name", list(pie.labels)[0] == "😊 Happiness")
check("including neutral and no clear emotion adds their slices", len(charts.build_emotion_doughnut(EMOTIONS, True).data[0].values) == 5)
check("only non-emotions gives no doughnut by default", charts.build_emotion_doughnut([EMOTIONS[1]]) is None)
check("no emotions gives no doughnut", charts.build_emotion_doughnut([]) is None)
bar = charts.build_emotion_bar(EMOTIONS).data[0]
check("emotion bars are horizontal with the right counts", bar.orientation == "h" and list(bar.x) == [2, 1, 1])
check("bar labels show the count and the percent of the emotions shown", list(bar.text) == ["2 (50.0%)", "1 (25.0%)", "1 (25.0%)"])
check("no emotions gives no emotion bars", charts.build_emotion_bar([]) is None)

print("\n3. Sentiment mix (bars)")
bar = charts.build_sentiment_bar(SENTIMENTS)
data = bar.data[0]
check("bars are always in the order Positive, Neutral, Negative", list(data.x) == ["Positive", "Neutral", "Negative"] and list(data.y) == [2, 1, 1])
check("each bar has its sentiment color", list(data.marker.color) == [charts.SENTIMENT_COLORS[s] for s in ("Positive", "Neutral", "Negative")])
check("bar labels show the count and percent", list(data.text) == ["2 (50.0%)", "1 (25.0%)", "1 (25.0%)"])
check("the y axis is titled Entries and starts at zero", bar.layout.yaxis.title.text == "Entries" and bar.layout.yaxis.rangemode == "tozero")

print("\n4. Text helpers")
check("a positive average shows a plus sign", page.format_average(0.12857) == "+0.129")
check("a negative average shows a minus sign", page.format_average(-0.0129) == "-0.013")
check("an average of zero shows 0.000", page.format_average(0.0) == "0.000")
check("no average shows a dash", page.format_average(None) == "—")
check("one most common sentiment", page.describe_sentiment_leaders(["Positive"]) == "Positive")
check("tied sentiments say so", page.describe_sentiment_leaders(["Negative", "Positive"]) == "Negative / Positive (tie)")
check("no most common emotion shows a dash", page.describe_emotion_leaders([]) == "—")
check(
    "tied emotions show both, with emojis",
    page.describe_emotion_leaders([EMOTIONS[0], EMOTIONS[2]]) == "😊 Happiness / 😢 Sadness (tie)",
)
check("no coverage note when every entry has an emotion", page.coverage_note(5, 5) is None)
check("the coverage note gives the numbers", "3 of your 5" in page.coverage_note(5, 3))
check("the coverage note explains when none have an emotion", "No emotions" in page.coverage_note(5, 0))
table = page.recent_entries_table(RHYTHM, count=2)
check("the recent table is newest first and limited", list(table["#"]) == [4, 3])
check("the recent table shows emotion text or 'Not analyzed'", list(table["Emotion"]) == ["😊 Happiness", "Not analyzed"])
check("the recent table shows readable dates", table["Date"].iloc[0] == "2026-10-01 10:04 UTC")

print("\n" + "=" * 50)
print(f"RESULT: {passed} passed, {failed} failed")
print("=" * 50)
sys.exit(1 if failed else 0)