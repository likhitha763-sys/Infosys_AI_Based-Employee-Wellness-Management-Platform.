"""Model performance page: F1, accuracy, precision, recall and confusion matrices.

Reads frontend/model_metrics.json (made by `python -m ml.make_frontend_metrics`).
Nothing is calculated live: these are the saved results of the offline evaluation.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_FILE = Path(__file__).resolve().parent / "model_metrics.json"


def load_metrics():
    """The saved metrics, or None if the file is missing or unreadable."""
    try:
        return json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def pct(value):
    return "n/a" if value is None else f"{100 * value:.1f}%"


def num(value):
    return "n/a" if value is None else f"{value:.3f}"


def heatmap(matrix, rows, columns, title, row_title, column_title):
    """A confusion matrix: counts, with actual classes in rows."""
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=columns,
            y=rows,
            text=[[str(v) for v in row] for row in matrix],
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        )
    )
    figure.update_layout(
        title=title,
        xaxis_title=column_title,
        yaxis_title=row_title,
        yaxis={"autorange": "reversed"},
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        height=380,
    )
    return figure


def per_class_table(per_class):
    rows = [
        {"Class": name, "Precision": round(v["precision"], 3), "Recall": round(v["recall"], 3),
         "F1": round(v["f1"], 3), "Support": v["support"]}
        for name, v in per_class.items()
    ]
    return pd.DataFrame(rows)


def render_targets(metrics, entry):
    targets = metrics.get("targets", {})
    semeval = entry.get("semeval") or {}
    checks = [
        ("Macro F1 on the GoEmotions validation set",
         targets.get("goemotions_validation_macro_f1"), entry.get("validation_macro_f1")),
        ("Polarity accuracy on the SemEval binary test set",
         targets.get("semeval_polarity_accuracy"), semeval.get("accuracy")),
    ]
    rows = []
    for measure, target, actual in checks:
        if target is None or actual is None:
            status = "n/a"
        else:
            status = "Reached" if actual >= target else "Not reached"
        rows.append({"Measure": measure, "Planned target": pct(target), "Result": pct(actual), "Status": status})
    st.dataframe(pd.DataFrame(rows), hide_index=True)


def render_goemotions(key, entry):
    st.markdown("#### GoEmotions test set (28 emotions, multi-label)")
    full = entry.get("goemotions")
    if not full:
        st.info("No GoEmotions test metrics were saved for this model.")
        return
    st.caption(
        f"{full['n_test']:,} test comments. One decision threshold ({full['threshold']}) "
        "was chosen on the validation set only."
    )
    first = st.columns(3)
    first[0].metric("Macro F1", num(full["macro"]["f1"]))
    first[1].metric("Macro precision", num(full["macro"]["precision"]))
    first[2].metric("Macro recall", num(full["macro"]["recall"]))
    second = st.columns(3)
    second[0].metric("Micro F1", num(full["micro"]["f1"]))
    second[1].metric("Accuracy per yes/no decision", pct(full["label_wise_accuracy"]))
    second[2].metric("Exact match (all 28 decisions right)", pct(full["exact_match_accuracy"]))
    st.caption(
        "Accuracy looks high because most of the 28 decisions for any text are an easy "
        "'this emotion is absent'. Macro F1 gives every emotion equal weight, so weak "
        "results on rare emotions show up there."
    )

    views = entry.get("goemotions_views") or []
    if views:
        st.markdown("**Results by label grouping**")
        st.dataframe(
            pd.DataFrame(
                [{"View": v["view"], "Macro F1": round(v["macro_f1"], 3), "Micro F1": round(v["micro_f1"], 3)}
                 for v in views]
            ),
            hide_index=True,
        )

    st.markdown("**Per-emotion results**")
    table = per_class_table(full["per_label"]).sort_values("F1", ascending=False)
    st.dataframe(table, hide_index=True)
    st.caption("Emotions with very few test examples (such as grief, relief and pride) have unreliable scores.")

    confusion = entry.get("goemotions_label_confusion")
    if confusion:
        label = st.selectbox("Show the confusion matrix for one emotion:", list(confusion),
                             key=f"perf_label_{key}")
        c = confusion[label]
        st.plotly_chart(
            heatmap(
                [[c["tn"], c["fp"]], [c["fn"], c["tp"]]],
                ["Emotion absent", "Emotion present"],
                ["Not predicted", "Predicted"],
                f"Confusion matrix: {label}",
                "Actual", "Model said",
            ),
            key=f"perf_label_fig_{key}_{label}",
        )


def render_semeval(key, entry, vader):
    st.markdown("#### SemEval-2018 tweets: positive or negative")
    semeval = entry.get("semeval")
    if not semeval:
        st.info("No SemEval results were saved for this model.")
        return
    st.caption("Neutral tweets are excluded. Never used for training or tuning.")
    left, right = st.columns(2)
    with left:
        st.markdown(f"**{entry['title']}**")
        metric_columns = st.columns(2)
        metric_columns[0].metric("Accuracy", pct(semeval["accuracy"]))
        metric_columns[1].metric("Macro F1", num(semeval["macro_f1"]))
        st.dataframe(per_class_table(semeval["per_class"]), hide_index=True)
        if semeval.get("confusion_matrix"):
            st.plotly_chart(
                heatmap(semeval["confusion_matrix"], ["negative", "positive"], ["negative", "positive"],
                        "Confusion matrix", "Actual", "Predicted"),
                key=f"perf_semeval_{key}",
            )
    with right:
        st.markdown("**VADER (baseline)**")
        if vader:
            metric_columns = st.columns(2)
            metric_columns[0].metric("Accuracy", pct(vader["accuracy"]))
            metric_columns[1].metric("Macro F1", num(vader["macro_f1"]))
            st.dataframe(per_class_table(vader["per_class"]), hide_index=True)
            if vader.get("confusion_matrix"):
                st.plotly_chart(
                    heatmap(vader["confusion_matrix"], ["negative", "positive"], ["negative", "positive"],
                            "Confusion matrix", "Actual", "Predicted"),
                    key=f"perf_vader_{key}",
                )
        else:
            st.info("No VADER baseline results were saved.")


def render_isear(key, entry):
    st.markdown("#### ISEAR: long first-person reports (held out)")
    isear = entry.get("isear")
    if not isear:
        st.info("No ISEAR results were saved for this model.")
        return
    st.caption(
        f"{isear['n']:,} rows with a direct label match (joy, fear, anger, sadness, disgust). "
        "The model's single prediction is its top-scoring label out of 28."
    )
    cols = st.columns(3)
    cols[0].metric("Accuracy", pct(isear["accuracy"]))
    cols[1].metric("Macro F1 (5 classes)", num(isear["macro_f1_5_classes"]))
    cols[2].metric("Predicted a different emotion", pct(isear["predicted_outside_5_classes_share"]))
    st.dataframe(per_class_table(isear["per_class"]), hide_index=True)
    confusion = entry.get("isear_confusion")
    if confusion:
        st.plotly_chart(
            heatmap(confusion["matrix"], confusion["rows"], confusion["columns"],
                    "Confusion matrix", "Actual ISEAR emotion", "Model's top prediction"),
            key=f"perf_isear_{key}",
        )
    st.caption(
        "ISEAR is very different from the Reddit comments the model was trained on, and the "
        "model often answers 'neutral'. Low scores here show a domain shift, not a broken model."
    )


def render_model_performance():
    """The page: saved evaluation results for each model."""
    st.title("Model Performance")
    st.write(
        "Saved results of the offline evaluation. These numbers are not recalculated here, "
        "and they are not a clinical assessment."
    )
    metrics = load_metrics()
    if not metrics or not metrics.get("models"):
        st.warning(
            "No saved metrics were found. Run `python -m ml.make_frontend_metrics` "
            "from the project folder to create them."
        )
        return

    models = metrics["models"]
    key = st.radio(
        "Model", list(models), format_func=lambda k: models[k]["title"],
        horizontal=True, key="perf_model",
    )
    entry = models[key]

    st.markdown("#### Planned targets and results")
    render_targets(metrics, entry)
    st.caption("Results are reported as measured. No target was changed to fit the numbers.")

    render_goemotions(key, entry)
    render_semeval(key, entry, metrics.get("vader"))
    render_isear(key, entry)