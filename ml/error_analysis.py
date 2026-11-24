"""Error analysis from saved prediction files (no model needed).

Usage: python -m ml.error_analysis distilbert
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "ml" / "results" / "colab" / "eval"
NAME = sys.argv[1] if len(sys.argv) > 1 else "distilbert"
OUT = ROOT / "ml" / "results" / f"error_analysis_{NAME}.md"


def anonymise(text: str) -> str:
    return re.sub(r"@\w+", "@user", str(text))


sem = pd.read_csv(EVAL / f"{NAME}_semeval_predictions.csv")
isr = pd.read_csv(EVAL / f"{NAME}_isear_predictions.csv")

lines = [f"# Error analysis: {NAME}", ""]

# ---- SemEval polarity ----
errors = sem[sem["label_id"] != sem["pred"]]
neg_as_pos = errors[errors["label_id"] == 0]
pos_as_neg = errors[errors["label_id"] == 1]
lines += [
    "## SemEval polarity",
    f"Errors: {len(errors)} of {len(sem)} "
    f"({len(neg_as_pos)} negative tweets predicted positive, "
    f"{len(pos_as_neg)} positive tweets predicted negative).",
    "Mentions are replaced with @user. 10 random examples of each kind (seed 42).",
    "",
]
for title, df in [("Negative predicted as positive", neg_as_pos),
                  ("Positive predicted as negative", pos_as_neg)]:
    lines.append(f"### {title}")
    for _, r in df.sample(min(10, len(df)), random_state=42).iterrows():
        lines.append(f"- `{r['id']}` {anonymise(r['text'])[:240]}")
    lines.append("")

# ---- ISEAR ----
top_pred = isr["pred"].value_counts().head(8).index.tolist()
table = pd.crosstab(isr["emotion_original"], isr["pred"])
table = table.reindex(columns=top_pred, fill_value=0)
lines += [
    "## ISEAR: true emotion (rows) against the 8 most common predicted labels (columns)",
    "```",
    table.to_string(),
    "```",
    "",
]
OUT.write_text("\n".join(lines), encoding="utf-8")

print(f"Errors on SemEval: {len(errors)} of {len(sem)}")
print("\nISEAR true emotion (rows) vs most common predicted labels (columns):")
print(table.to_string())
print(f"\nSaved: {OUT}")