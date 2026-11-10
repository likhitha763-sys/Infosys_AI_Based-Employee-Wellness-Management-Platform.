"""Checks for the label views, cleaning, and loss weights (Milestone 2, Step 4).

Run the preparation script first:
    python -m ml.data_prep.prepare_goemotions
Then, from the project root:
    python -m ml.tests.check_prepare_goemotions
"""

import json
import sys

import numpy as np
import pandas as pd

from ml import config
from ml.data_prep import goemotions as go
from ml.data_prep import labels as lab
from ml.data_prep import prepare_goemotions as prep

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


print("1. The functions, on small made-up data")
table = lab.multi_hot([[0], [0, 2], [27]])
check("the 0/1 table has the right shape and ones", table.shape == (3, 28) and table.sum() == 4 and table[1, 2] == 1 and table[2, 27] == 1)
# Label numbers: annoyance = 3, admiration = 0, fear = 14, neutral = 27.
expected = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1]])
check(
    "the Ekman view groups labels correctly (annoyance -> anger, admiration + fear -> joy + fear, neutral -> neutral)",
    (lab.to_ekman_view(lab.multi_hot([[3], [0, 14], [27]])) == expected).all(),
)
check("the Ekman groups are in the planned order", lab.EKMAN_GROUPS == ("anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"))
check(
    "there are 12 distinct headline emotions, all real labels",
    len(set(lab.HEADLINE_EMOTIONS)) == 12 and all(name in lab.LABELS for name in lab.HEADLINE_EMOTIONS),
)
train = pd.DataFrame({"text": ["Hello there", "Other"]})
evaluation = pd.DataFrame({"text": ["hello   THERE", "New", "Another new"]})
cleaned, removed = prep.clean_eval_split(evaluation, train)
check("evaluation rows whose text is in train are removed (ignoring case and spaces)", removed == 1 and list(cleaned["text"]) == ["New", "Another new"])
check("cleaning does not change the original tables", len(train) == 2 and len(evaluation) == 3)
weights = prep.compute_pos_weights([10, 50, 0], 100)
check(
    "loss weights: 90/10 = 9, 50/50 = 1, and sqrt(9) = 3; a label with no positives gets 1 and is listed",
    weights["raw"] == [9.0, 1.0, 1.0] and weights["sqrt"] == [3.0, 1.0, 1.0] and weights["zero_positive_labels"] == [2],
)
plan = config.RESULTS_DIR / "evaluation_plan.md"
text = plan.read_text(encoding="utf-8") if plan.exists() else ""
check(
    "the evaluation plan exists and names the views, the models, and the external datasets",
    all(word in text for word in ("28", "Ekman", "validation", "ISEAR", "SemEval", "bert-base-cased")),
)

print("\n2. The real prepared data")
if not prep.REPORT_PATH.exists():
    check("the data has been prepared (run: python -m ml.data_prep.prepare_goemotions)", False)
else:
    report = json.loads(prep.REPORT_PATH.read_text(encoding="utf-8"))
    official = go.load_all_splits()
    processed = {name: prep.load_processed(name) for name in report["splits"]}
    check(
        "the training data is exactly the official training data",
        list(processed["train"]["id"]) == list(official["train"]["id"])
        and processed["train"]["labels"].map(tuple).tolist() == official["train"]["labels"].map(tuple).tolist(),
    )
    check(
        "the clean sets only contain rows of the official sets",
        all(set(processed[f"{name}_clean"]["id"]) <= set(official[name]["id"]) for name in ("validation", "test")),
    )
    train_texts = {go.normalize_text(text) for text in official["train"]["text"]}
    check(
        "no clean evaluation text appears in the training data",
        all(go.normalize_text(text) not in train_texts for name in ("validation_clean", "test_clean") for text in processed[name]["text"]),
    )
    check(
        "the number of removed rows in the report is correct",
        all(report["removed"][name] == len(official[name]) - len(processed[f"{name}_clean"]) for name in ("validation", "test")),
    )
    check("the label names in the report match the settings", report["label_names"] == list(config.GOEMOTIONS_LABELS))
    raw, sqrt = report["pos_weights"]["raw"], report["pos_weights"]["sqrt"]
    check(
        "there are 28 positive weights, all above zero, and the square-root version is never larger when the raw weight is at least 1",
        len(raw) == 28 and all(value > 0 for value in raw + sqrt) and all(s <= r + 1e-9 for r, s in zip(raw, sqrt) if r >= 1),
    )
    check(
        "every saved file matches its recorded checksum",
        all(go.sha256_of(config.PROJECT_ROOT / info["file"]) == info["sha256"] for info in report["splits"].values()),
    )
    check("the report names the same dataset version as the manifest", report["source_revision"] == go.load_manifest()["revision"])

print("\n" + "=" * 50)
print(f"RESULT: {passed} passed, {failed} failed")
print("=" * 50)
sys.exit(1 if failed else 0)