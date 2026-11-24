"""Model performance page: shows the saved Milestone 2 evaluation results.

Reads the result files in ml/results. No model is loaded here and nothing is
recalculated, so the numbers are exactly the ones from the evaluation runs.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RESULTS = Path(__file__).resolve().parents[1] / "ml" / "results"
COLAB = RESULTS / "colab"
EVAL = COLAB / "eval"

MODELS = {"BERT (used in the app)": "bert", "DistilBERT": "distilbert"}
ISEAR_CLASSES = ["joy", "fear", "anger", "sadness", "disgust"]
RARE_SUPPORT = 30  # fewer test examples than this: the score is unreliable
CM_KEY = "confusion_matrix_rows_true_cols_pred_[neg,pos]"
PROJECT_TARGET = 0.82


def expected_files():
    """Every result file this page reads."""
    files = [RESULTS / "vader_semeval_results.json"]
    for key in MODELS.values():
        files += [
            RESULTS / f"goemotions_test_metrics_{key}.json",
            COLAB / f"{key}_goemotions_results.json",
            EVAL / f"{key}_eval.json",
            EVAL / f"{key}_isear_predictions.csv",
        ]
    return files


@st.cache_data(show_spinner=False)
def load_json(path_text):
    path = Path(path_text)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_csv(path_text):
    path = Path(path_text)
    if not path.exists():
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def confusion_figure(matrix, row_labels, col_labels):
    """Heatmap with the count and the share of each row in every cell."""
    counts = [[int(value) for value in row] for row in matrix]
    totals = [max(sum(row), 1) for row in counts]
    shares = [[value / totals[i] for value in row] for i, row in enumerate(counts)]
    text = [[f"{value}<br>{shares[i][j]:.0%}" for j, value in enumerate(row)]
            for i, row in enumerate(counts)]
    figure = go.Figure(
        go.Heatmap(
            z=shares, x=col_labels, y=row_labels, text=text, texttemplate="%{text}",
            textfont={"color": "black", "size": 13},
            colorscale=[[0, "#f4f8fb"], [1, "#6baed6"]], zmin=0, zmax=1,
            showscale=False, hovertemplate="True %{y}<br>Predicted %{x}<br>%{text}<extra></extra>",
        )
    )
    figure.update_layout(
        xaxis_title="Predicted", yaxis_title="True", yaxis_autorange="reversed",
        height=110 + 60 * len(row_labels), margin={"l": 10, "r": 10, "t": 10, "b": 10},
    )
    return figure


def per_label_figure(per_label):
    """Horizontal bars: F1 for each of the 28 emotions, lowest at the bottom."""
    rows = sorted(per_label.items(), key=lambda item: item[1]["f1"])
    figure = go.Figure(
        go.Bar(
            x=[values["f1"] for _, values in rows],
            y=[name for name, _ in rows],
            orientation="h",
            marker_color=["#e07b39" if values["support"] < RARE_SUPPORT else "#4c78a8"
                          for _, values in rows],
            customdata=[[values["precision"], values["recall"], values["support"]]
                        for _, values in rows],
            hovertemplate=("%{y}<br>F1 %{x:.3f}<br>precision %{customdata[0]:.3f}"
                           "<br>recall %{customdata[1]:.3f}<br>test examples: %{customdata[2]}"
                           "<extra></extra>"),
        )
    )
    figure.update_layout(
        height=720, xaxis={"range": [0, 1], "title": "F1 score"},
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
    )
    return figure


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def render_goemotions(key):
    st.markdown("### 1. Emotion detection on GoEmotions (28 emotions)")
    metrics = load_json(str(RESULTS / f"goemotions_test_metrics_{key}.json"))
    if not metrics:
        st.info(f"Result file not found. Run: python -m ml.goemotions_metrics {key}")
        return

    st.caption(
        f"Held-out test set: {metrics['n_test']:,} Reddit comments. A comment can have "
        f"several emotions. An emotion counts as detected at {metrics['threshold']} "
        "confidence or higher (threshold chosen on the validation set)."
    )
    first = st.columns(3)
    first[0].metric("Macro F1", f"{metrics['macro']['f1']:.3f}",
                    help="Average F1 over all 28 emotions. Every emotion counts equally, so rare emotions matter.")
    first[1].metric("Micro F1", f"{metrics['micro']['f1']:.3f}",
                    help="F1 over all decisions pooled together. Common emotions dominate.")
    first[2].metric("Accuracy (each yes/no decision)", f"{metrics['label_wise_accuracy']:.1%}",
                    help="Looks high because most emotions are correctly absent in most comments.")
    second = st.columns(3)
    second[0].metric("Macro precision", f"{metrics['macro']['precision']:.3f}")
    second[1].metric("Macro recall", f"{metrics['macro']['recall']:.3f}")
    second[2].metric("Accuracy (all 28 decisions right)", f"{metrics['exact_match_accuracy']:.1%}",
                     help="Share of comments where every one of the 28 yes/no decisions is correct.")
    st.caption(
        f"The project target was a macro F1 of {PROJECT_TARGET:.2f}. It was not reached. "
        "Note how the 'each yes/no decision' accuracy is far higher than the macro F1: "
        "high accuracy can hide weak results on rare emotions."
    )

    st.plotly_chart(per_label_figure(metrics["per_label"]), key=f"perf_labels_{key}")
    st.caption(
        f"Orange bars have fewer than {RARE_SUPPORT} test examples, so their scores are "
        "unreliable. Hover over a bar for precision, recall and the number of examples. "
        "A multi-label task has no single confusion matrix, so results are shown per emotion."
    )
    table = pd.DataFrame(
        [{"Emotion": name, "Precision": values["precision"], "Recall": values["recall"],
          "F1": values["f1"], "Test examples": values["support"]}
         for name, values in metrics["per_label"].items()]
    ).sort_values("F1", ascending=False)
    with st.expander("Per-emotion table"):
        st.dataframe(table, hide_index=True)


def class_table(per_class):
    return pd.DataFrame(
        [{"Class": name, "Precision": values["precision"], "Recall": values["recall"],
          "F1": values["f1"], "Examples": values["support"]}
         for name, values in per_class.items()]
    )


def render_semeval(key, label):
    st.markdown("### 2. Positive vs negative on SemEval-2018 tweets")
    evaluation = load_json(str(EVAL / f"{key}_eval.json"))
    vader = load_json(str(RESULTS / "vader_semeval_results.json"))
    if not evaluation or not vader:
        st.info("SemEval result files not found.")
        return
    st.caption(
        "675 test tweets with neutral tweets removed. In each matrix, rows are the true "
        "label and columns the predicted label. Each cell shows a count and its share of the row. "
        "The emotion model's polarity comes from its emotion scores with a fixed rule; "
        "VADER is the baseline."
    )
    columns = st.columns(2)
    for column, name, data, suffix in (
        (columns[0], label, evaluation["semeval_polarity"], "model"),
        (columns[1], "VADER baseline", vader, "vader"),
    ):
        with column:
            st.markdown(f"**{name}**")
            a, b = st.columns(2)
            a.metric("Accuracy", f"{data['accuracy']:.1%}")
            b.metric("Macro F1", f"{data['macro_f1']:.3f}")
            st.plotly_chart(
                confusion_figure(data[CM_KEY], ["negative", "positive"], ["negative", "positive"]),
                key=f"perf_sem_{key}_{suffix}",
            )
            st.dataframe(class_table(data["per_class"]), hide_index=True)
    st.caption("The project target was 90% accuracy. It was not reached by any of the three.")


def render_isear(key):
    st.markdown("### 3. Held-out check on ISEAR (a very different kind of text)")
    evaluation = load_json(str(EVAL / f"{key}_eval.json"))
    frame = load_csv(str(EVAL / f"{key}_isear_predictions.csv"))
    if not evaluation or frame is None:
        st.info("ISEAR result files not found.")
        return
    isear = evaluation["isear_main"]
    a, b, c = st.columns(3)
    a.metric("Accuracy", f"{isear['accuracy']:.1%}")
    b.metric("Macro F1 (5 emotions)", f"{isear['macro_f1_5_classes']:.3f}")
    c.metric("Predictions outside the 5 emotions", f"{isear['predicted_outside_5_classes_share']:.0%}")

    frame = frame[frame["emotion_original"].isin(ISEAR_CLASSES)].copy()
    columns = ISEAR_CLASSES + ["neutral", "other label"]

    def bucket(prediction):
        if prediction in ISEAR_CLASSES:
            return prediction
        return "neutral" if prediction == "neutral" else "other label"

    frame["bucket"] = frame["pred"].map(bucket)
    table = (pd.crosstab(frame["emotion_original"], frame["bucket"])
             .reindex(index=ISEAR_CLASSES, columns=columns, fill_value=0))
    st.plotly_chart(confusion_figure(table.values.tolist(), ISEAR_CLASSES, columns),
                    key=f"perf_isear_{key}")
    st.caption(
        f"{len(frame):,} ISEAR reports with a matching emotion. ISEAR holds long first-person "
        "descriptions of past events, while the model learned from short Reddit comments. "
        "The model often answers 'neutral' or another label, so a prediction outside the five "
        "emotions counts as wrong. Read these numbers as a test of how well the model transfers "
        "to different text, not as its normal accuracy."
    )
    st.dataframe(class_table(isear["per_class"]), hide_index=True)


def view_value(key, view_name):
    data = load_json(str(COLAB / f"{key}_goemotions_results.json"))
    if not data:
        return "n/a"
    for view in data["views"]:
        if view["view"] == view_name:
            return f"{view['macro_f1']:.3f}"
    return "n/a"


def render_benchmark_table():
    st.markdown("### 4. Benchmark table")
    vader = load_json(str(RESULTS / "vader_semeval_results.json"))
    rows = []
    if vader:
        rows.append({"Model": "VADER (baseline)", "GoEmotions macro F1 (28 emotions)": "n/a",
                     "Ekman groups": "n/a", "12 main emotions": "n/a",
                     "SemEval polarity accuracy": f"{vader['accuracy']:.1%}",
                     "ISEAR accuracy": "n/a"})
    for name, key in (("DistilBERT", "distilbert"), ("BERT", "bert")):
        evaluation = load_json(str(EVAL / f"{key}_eval.json"))
        rows.append({
            "Model": name,
            "GoEmotions macro F1 (28 emotions)": view_value(key, "28 labels"),
            "Ekman groups": view_value(key, "Ekman 6 + neutral"),
            "12 main emotions": view_value(key, "12 headline emotions"),
            "SemEval polarity accuracy": (f"{evaluation['semeval_polarity']['accuracy']:.1%}"
                                          if evaluation else "n/a"),
            "ISEAR accuracy": (f"{evaluation['isear_main']['accuracy']:.1%}"
                               if evaluation else "n/a"),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption(
        "Each model was trained and scored once (seed 42), so small differences between "
        "DistilBERT and BERT should not be read as one being better."
    )


def render_model_performance():
    """The Model performance tab."""
    st.subheader("Model performance")
    st.write(
        "These are the results of the Milestone 2 evaluation on text the models never "
        "saw in training. They describe the models on those test sets. They do not "
        "guarantee how accurate the result is for your own text, and the emotion result "
        "is never a diagnosis."
    )
    missing = [path.name for path in expected_files() if not path.exists()]
    if missing:
        st.warning("Some result files are missing: " + ", ".join(missing))

    label = st.radio("Model", list(MODELS), horizontal=True, key="perf_model")
    key = MODELS[label]
    render_goemotions(key)
    render_semeval(key, label)
    render_isear(key)
    render_benchmark_table()