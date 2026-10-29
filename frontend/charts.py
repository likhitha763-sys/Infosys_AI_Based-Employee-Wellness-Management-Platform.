"""Chart builders for the dashboard (Plotly).

This module contains no Streamlit code, so the charts can be tested on their own.
Every function takes the data returned by GET /api/dashboard and returns a
Plotly figure, or None when there is nothing to draw.
"""

import html
from datetime import datetime, timezone

import plotly.graph_objects as go

SENTIMENT_ORDER = ("Positive", "Neutral", "Negative")
SENTIMENT_COLORS = {"Positive": "#2A9D8F", "Neutral": "#8D99AE", "Negative": "#E76F51"}
# Different shapes as well as colors, so the chart is readable without color vision.
SENTIMENT_SYMBOLS = {"Positive": "circle", "Neutral": "square", "Negative": "diamond"}
LINE_COLOR = "#4C78A8"
# A color-blind friendly palette for the emotion charts.
PALETTE = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
    "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC",
]


def safe(text):
    """Escape text so it cannot be read as HTML formatting inside a chart."""
    return html.escape(str(text), quote=False)


def format_time(value):
    """Turn a stored UTC time (ISO text) into a readable form."""
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return str(value)


def emotion_label(name, emoji):
    """Emoji and name together, or a note when no emotion was saved."""
    return f"{emoji} {name}" if name else "Not analyzed"


def filter_emotions(emotion_counts, include_non_emotions=False):
    """Keep real emotions; optionally also 'neutral' and 'no clear emotion'."""
    return [item for item in emotion_counts if include_non_emotions or item["is_emotion"]]


def chart_layout(figure, height):
    """Shared spacing for every chart."""
    figure.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10))
    return figure


# ---------------------------------------------------------------------------
# Emotional rhythm (line chart): compound sentiment score per entry
# ---------------------------------------------------------------------------


def build_rhythm_figure(rhythm):
    """Line chart of each entry's compound score, in chronological order."""
    if not rhythm:
        return None

    numbers = [point["number"] for point in rhythm]
    scores = [point["compound_score"] for point in rhythm]
    details = [
        [
            format_time(point["created_at"]),
            point["sentiment"],
            emotion_label(point.get("emotion_name"), point.get("emotion_emoji")),
            safe(point["text_preview"]),
        ]
        for point in rhythm
    ]

    figure = go.Figure(
        go.Scatter(
            x=numbers,
            y=scores,
            mode="lines+markers",
            line=dict(color=LINE_COLOR, width=2),
            marker=dict(
                size=10,
                color=[SENTIMENT_COLORS.get(p["sentiment"], "#8D99AE") for p in rhythm],
                symbol=[SENTIMENT_SYMBOLS.get(p["sentiment"], "circle") for p in rhythm],
                line=dict(width=1, color="white"),
            ),
            customdata=details,
            hovertemplate=(
                "<b>Entry %{x}</b> (%{customdata[0]})<br>"
                "Compound score: %{y:.4f}<br>"
                "Sentiment: %{customdata[1]}<br>"
                "Emotion: %{customdata[2]}<br>"
                "%{customdata[3]}<extra></extra>"
            ),
        )
    )
    figure.add_hline(
        y=0, line_dash="dash", line_color="#888888",
        annotation_text="Neutral (0)", annotation_position="bottom right",
    )
    figure.update_layout(
        showlegend=False,
        yaxis=dict(
            title="Compound sentiment score",
            range=[-1.1, 1.1],
            tickvals=[-1, -0.5, 0, 0.5, 1],
            zeroline=False,
        ),
        xaxis=dict(title="Entry number", tickformat="d", dtick=1 if len(rhythm) <= 20 else None),
    )
    return chart_layout(figure, 380)


# ---------------------------------------------------------------------------
# Emotion mix (doughnut) and emotion counts (bars)
# ---------------------------------------------------------------------------


def build_emotion_doughnut(emotion_counts, include_non_emotions=False):
    """Doughnut chart: the share of each detected emotion."""
    shown = filter_emotions(emotion_counts, include_non_emotions)
    if not shown:
        return None

    figure = go.Figure(
        go.Pie(
            labels=[emotion_label(item["name"], item["emoji"]) for item in shown],
            values=[item["count"] for item in shown],
            hole=0.5,
            sort=False,  # keep the order given (most frequent first)
            marker=dict(
                colors=[PALETTE[index % len(PALETTE)] for index in range(len(shown))],
                line=dict(color="white", width=2),
            ),
            textinfo="percent",
            textposition="auto",
            hovertemplate="%{label}<br>%{value} entries (%{percent})<extra></extra>",
        )
    )
    figure.update_layout(legend=dict(orientation="v"))
    return chart_layout(figure, 380)


def build_emotion_bar(emotion_counts, include_non_emotions=False):
    """Horizontal bars: the number of entries per detected emotion."""
    shown = filter_emotions(emotion_counts, include_non_emotions)
    if not shown:
        return None

    total = sum(item["count"] for item in shown)
    figure = go.Figure(
        go.Bar(
            x=[item["count"] for item in shown],
            y=[emotion_label(item["name"], item["emoji"]) for item in shown],
            orientation="h",
            marker_color=[PALETTE[index % len(PALETTE)] for index in range(len(shown))],
            text=[f"{item['count']} ({round(100 * item['count'] / total, 1)}%)" for item in shown],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>%{x} entries<extra></extra>",
        )
    )
    figure.update_layout(
        showlegend=False,
        xaxis=dict(title="Entries", rangemode="tozero", tickformat="d"),
        yaxis=dict(autorange="reversed"),
    )
    return chart_layout(figure, max(260, 45 * len(shown) + 100))


# ---------------------------------------------------------------------------
# Sentiment mix (bars): Positive, Neutral, Negative counts
# ---------------------------------------------------------------------------


def build_sentiment_bar(sentiment_counts):
    """Bar chart comparing how many entries are Positive, Neutral, and Negative."""
    by_label = {item["sentiment"]: item for item in sentiment_counts}
    ordered = [by_label[label] for label in SENTIMENT_ORDER if label in by_label]
    if not ordered:
        return None

    figure = go.Figure(
        go.Bar(
            x=[item["sentiment"] for item in ordered],
            y=[item["count"] for item in ordered],
            marker_color=[SENTIMENT_COLORS[item["sentiment"]] for item in ordered],
            text=[f"{item['count']} ({item['percent']}%)" for item in ordered],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}<br>%{y} entries<extra></extra>",
        )
    )
    figure.update_layout(
        showlegend=False,
        xaxis=dict(title="Sentiment"),
        yaxis=dict(title="Entries", rangemode="tozero", tickformat="d"),
    )
    return chart_layout(figure, 380)