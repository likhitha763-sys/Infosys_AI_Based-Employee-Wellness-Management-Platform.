"""Checks that the saved metrics are consistent with each other.

Run from the project root:
    python -m frontend.tests.check_model_performance
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from frontend import model_performance_page as page  # noqa: E402

passed = 0
failed = 0


def check(description, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"PASS  {description}")
    else:
        failed += 1
        print(f"FAIL  {description}")


metrics = page.load_metrics()
check("the metrics file loads", metrics is not None)
models = (metrics or {}).get("models", {})
check("at least one model is included", len(models) >= 1)

for key, entry in models.items():
    if entry.get("threshold") is not None:
        check(f"{key}: threshold is between 0 and 1", 0 < entry["threshold"] < 1)

    semeval = entry.get("semeval")
    if semeval:
        matrix = semeval["confusion_matrix"]
        total = sum(sum(row) for row in matrix)
        support = sum(v["support"] for v in semeval["per_class"].values())
        check(f"{key}: SemEval confusion matrix adds up to the class supports", total == support)
        correct = matrix[0][0] + matrix[1][1]
        check(f"{key}: SemEval accuracy matches the confusion matrix",
              abs(correct / total - semeval["accuracy"]) < 1e-6)

    isear = entry.get("isear")
    confusion = entry.get("isear_confusion")
    if isear and confusion:
        rows = [sum(r) for r in confusion["matrix"]]
        supports = [isear["per_class"][c]["support"] for c in confusion["rows"]]
        check(f"{key}: ISEAR confusion row sums match the class supports", rows == supports)
        diagonal = sum(confusion["matrix"][i][i] for i in range(len(confusion["rows"])))
        check(f"{key}: ISEAR accuracy matches the confusion matrix",
              abs(diagonal / sum(rows) - isear["accuracy"]) < 1e-6)

    full = entry.get("goemotions")
    label_confusion = entry.get("goemotions_label_confusion")
    if full and label_confusion:
        ok = all(
            c["tp"] + c["fn"] == full["per_label"][label]["support"]
            and c["tp"] + c["fp"] + c["fn"] + c["tn"] == full["n_test"]
            for label, c in label_confusion.items()
        )
        check(f"{key}: all 28 per-emotion confusion matrices match the supports and test size", ok)

check("VADER baseline is included", bool((metrics or {}).get("vader")))

print(f"\n{passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)