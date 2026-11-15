"""Build the Milestone 2 benchmark table from saved result files.

Usage: python -m ml.make_benchmark_table
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "ml" / "results"
COLAB = RES / "colab"
OUT = RES / "benchmark_table.md"
MODELS = [("DistilBERT", "distilbert"), ("BERT", "bert")]


def load(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def view(results, name, key):
    for v in results["views"]:
        if v["view"] == name:
            return v[key]
    return None


def fmt(x):
    return "n/a" if x is None else f"{x:.4f}"


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


lines = ["# Milestone 2 benchmark table", ""]

lines += [
    "## GoEmotions test set (cleaned); threshold tuned on validation only", "",
    "| Model | Threshold | Validation macro F1 | 28 labels macro F1 | Ekman + neutral macro F1 | 12 headline macro F1 | 28 labels micro F1 |",
    "|---|---|---|---|---|---|---|",
]
for label, key in MODELS:
    r = load(COLAB / f"{key}_goemotions_results.json")
    if r is None:
        lines.append(f"| {label} | file missing | | | | | |")
        continue
    lines.append(
        f"| {label} | {r['threshold']} | {fmt(r['validation_macro_f1'])} "
        f"| {fmt(view(r, '28 labels', 'macro_f1'))} "
        f"| {fmt(view(r, 'Ekman 6 + neutral', 'macro_f1'))} "
        f"| {fmt(view(r, '12 headline emotions', 'macro_f1'))} "
        f"| {fmt(view(r, '28 labels', 'micro_f1'))} |"
    )

lines += [
    "", "## External evaluation (never used for training or tuning)", "",
    "| Model | SemEval polarity accuracy | SemEval macro F1 | ISEAR accuracy (5 classes) | ISEAR macro F1 | ISEAR predicted outside the 5 classes | ISEAR accuracy, Ekman-grouped (exploratory) |",
    "|---|---|---|---|---|---|---|",
]
vader = load(RES / "vader_semeval_results.json")
if vader:
    lines.append(f"| VADER | {pct(vader['accuracy'])} | {fmt(vader['macro_f1'])} | n/a | n/a | n/a | n/a |")
for label, key in MODELS:
    e = load(COLAB / "eval" / f"{key}_eval.json")
    x = load(RES / f"exploratory_isear_ekman_{key}.json")
    if e is None:
        lines.append(f"| {label} | file missing | | | | | |")
        continue
    s, i = e["semeval_polarity"], e["isear_main"]
    lines.append(
        f"| {label} | {pct(s['accuracy'])} | {fmt(s['macro_f1'])} "
        f"| {pct(i['accuracy'])} | {fmt(i['macro_f1_5_classes'])} "
        f"| {pct(i['predicted_outside_5_classes_share'])} "
        f"| {pct(x['accuracy']) if x else 'n/a'} |"
    )

lines += [
    "", "## Notes", "",
    "- GoEmotions scores come from one run per model with seed 42. Run-to-run variation was not measured, so small differences should not be read as a winner.",
    "- No model reaches the 82% macro F1 target in any GoEmotions view.",
    "- SemEval: 675 test tweets, neutral class excluded, emotion-to-polarity rule fixed in the evaluation plan before scoring. VADER rule: compound >= 0.05 means positive.",
    "- ISEAR: Hugging Face mirror subset, 3,752 rows with a direct label match; the predicted label is the top-1 of 28; shame and guilt rows are excluded from the score.",
    "- The Ekman-grouped ISEAR column is exploratory and was added after the primary ISEAR results were known.",
]
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"\nSaved: {OUT}")