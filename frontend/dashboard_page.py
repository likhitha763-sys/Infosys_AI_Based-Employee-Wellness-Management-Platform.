"""The mood dashboard: summary cards, charts, emotion cards, and recent entries.

This file only handles the user interface. All numbers come from the
FastAPI backend (GET /api/dashboard), calculated from saved analyses.
"""

import pandas as pd
import streamlit as st

import api_client
import charts
import views

RHYTHM_LIMITS = [20, 50, 100, 200]
RECENT_ENTRIES = 10
EMOTION_CARDS = 6


# ---------------------------------------------------------------------------
# Text helpers (no screen code, so they can be tested)
# ---------------------------------------------------------------------------


def format_average(value):
    """The average compound score with a +/- sign and 3 decimals."""
    if value is None:
        return "—"
    if round(value, 3) == 0:
        return "0.000"
    return f"{value:+.3f}"


def describe_sentiment_leaders(labels):
    """Text for the most common sentiment (more than one label means a tie)."""
    if not labels:
        return "—"
    if len(labels) == 1:
        return labels[0]
    return " / ".join(labels) + " (tie)"


def describe_emotion_leaders(emotions):
    """Text for the most common emotion (more than one means a tie)."""
    if not emotions:
        return "—"
    text = " / ".join(charts.emotion_label(item["name"], item["emoji"]) for item in emotions)
    return text if len(emotions) == 1 else text + " (tie)"


def coverage_note(total, with_emotion):
    """Explain when some entries have no saved emotion. None if all have one."""
    if total == 0 or with_emotion >= total:
        return None
    if with_emotion == 0:
        return (
            "No emotions have been detected yet. Entries you analyze from now on "
            "will include one when emotion detection is switched on."
        )
    return (
        f"Emotion was detected for {with_emotion} of your {total} entries. Older entries, "
        "or entries analyzed while emotion detection was off, are left out of the emotion charts."
    )


def recent_entries_table(rhythm, count=RECENT_ENTRIES):
    """The newest entries first, as a table."""
    newest_first = list(reversed(rhythm))[:count]
    return pd.DataFrame(
        {
            "#": [point["number"] for point in newest_first],
            "Date": [charts.format_time(point["created_at"]) for point in newest_first],
            "Text": [point["text_preview"] for point in newest_first],
            "Emotion": [
                charts.emotion_label(point.get("emotion_name"), point.get("emotion_emoji"))
                for point in newest_first
            ],
            "Sentiment": [point["sentiment"] for point in newest_first],
            "Compound": [point["compound_score"] for point in newest_first],
        }
    )


# ---------------------------------------------------------------------------
# Screen parts
# ---------------------------------------------------------------------------


def render_summary_cards(summary):
    cards = [
        ("Journal entries", str(summary["total_entries"]), "Saved analyses in your account."),
        (
            "Average sentiment",
            format_average(summary["average_compound"]),
            "Average compound score, from -1 (very negative) to +1 (very positive).",
        ),
        (
            "Most common sentiment",
            describe_sentiment_leaders(summary["most_common_sentiments"]),
            "The sentiment label that appears most often.",
        ),
        (
            "Most common emotion",
            describe_emotion_leaders(summary["most_common_emotions"]),
            "The most frequent detected emotion. Neutral and 'no clear emotion' are not counted.",
        ),
    ]
    for column, (title, value, help_text) in zip(st.columns(4), cards):
        with column:
            with st.container(border=True):
                st.metric(title, value, help=help_text)


def show_chart(figure, key, empty_message):
    """Draw a chart, or an explanation if there is nothing to draw."""
    if figure is None:
        st.info(empty_message)
    else:
        st.plotly_chart(figure, key=key)


def render_emotion_cards(emotion_counts):
    """Small emoji cards for the most frequent emotions."""
    top = emotion_counts[:EMOTION_CARDS]
    if not top:
        st.info("No specific emotions have been detected yet.")
        return
    for column, item in zip(st.columns(len(top)), top):
        with column:
            with st.container(border=True):
                st.markdown(f"### {item['emoji']}")
                st.write(f"**{item['name']}**")
                st.caption(f"{item['count']} {'entry' if item['count'] == 1 else 'entries'}")


def render_mood_dashboard():
    """The whole dashboard tab."""
    token = st.session_state.get("token")

    title_column, limit_column = st.columns([3, 1])
    title_column.subheader("Your mood dashboard")
    limit = limit_column.selectbox(
        "Entries in the line chart", RHYTHM_LIMITS, index=1, key="dashboard_rhythm_limit"
    )

    data, error = api_client.get_dashboard(token, limit)
    if error:
        st.error(error)
        return

    summary = data["summary"]
    st.caption(
        "Everything here is calculated from your saved analyses. Sentiment (positive, "
        "negative, or neutral) and detected emotion are separate results from two "
        "different tools."
    )

    if summary["total_entries"] == 0:
        st.info(
            "You have no saved analyses yet. Analyze some text in the other tabs and "
            "your dashboard will fill in here."
        )
        return

    note = coverage_note(summary["total_entries"], summary["entries_with_emotion"])
    if note:
        st.caption(note)

    render_summary_cards(summary)

    rhythm = data["rhythm"]
    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Emotional rhythm")
        show_chart(charts.build_rhythm_figure(rhythm), "dashboard_rhythm", "No entries to chart yet.")
        caption = "Each point is one entry's compound sentiment score (VADER)."
        if len(rhythm) < summary["total_entries"]:
            caption += f" Showing your newest {len(rhythm)} of {summary['total_entries']} entries."
        st.caption(caption)

    with right:
        include = st.checkbox(
            "Include neutral and 'no clear emotion'",
            value=False,
            key="dashboard_include_non_emotions",
            help="These are not specific emotions, so they are hidden by default.",
        )
        emotion_tab, counts_tab, sentiment_tab = st.tabs(["Emotion mix", "Emotion counts", "Sentiment mix"])
        no_emotions = "No specific emotions to show yet. Analyze more text, or turn on the option above."
        with emotion_tab:
            show_chart(
                charts.build_emotion_doughnut(data["emotion_counts"], include),
                "dashboard_doughnut", no_emotions,
            )
            st.caption("Percentages are shares of the emotions shown.")
        with counts_tab:
            show_chart(
                charts.build_emotion_bar(data["emotion_counts"], include),
                "dashboard_emotion_bar", no_emotions,
            )
        with sentiment_tab:
            show_chart(
                charts.build_sentiment_bar(data["sentiment_counts"]),
                "dashboard_sentiment_bar", "No sentiment results yet.",
            )
            st.caption("Counts of entries by VADER sentiment label.")

    st.markdown("#### Emotions detected")
    render_emotion_cards(charts.filter_emotions(data["emotion_counts"], include))

    st.markdown("#### Recent journal entries")
    st.dataframe(recent_entries_table(rhythm), hide_index=True)
    st.caption("Open the My history tab to see every entry with its full report.")
    st.caption(views.DISCLAIMER)