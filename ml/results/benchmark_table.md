# Milestone 2 benchmark table

## GoEmotions test set (cleaned); threshold tuned on validation only

| Model | Threshold | Validation macro F1 | 28 labels macro F1 | Ekman + neutral macro F1 | 12 headline macro F1 | 28 labels micro F1 |
|---|---|---|---|---|---|---|
| DistilBERT | 0.65 | 0.5224 | 0.5155 | 0.6243 | 0.5300 | 0.5791 |
| BERT | 0.7 | 0.5320 | 0.5120 | 0.6213 | 0.5304 | 0.5805 |

## External evaluation (never used for training or tuning)

| Model | SemEval polarity accuracy | SemEval macro F1 | ISEAR accuracy (5 classes) | ISEAR macro F1 | ISEAR predicted outside the 5 classes | ISEAR accuracy, Ekman-grouped (exploratory) |
|---|---|---|---|---|---|---|
| VADER | 83.7% | 0.8357 | n/a | n/a | n/a | n/a |
| DistilBERT | 83.4% | 0.8320 | 29.4% | 0.4275 | 64.7% | 37.0% |
| BERT | file missing | | | | | |

## Notes

- GoEmotions scores come from one run per model with seed 42. Run-to-run variation was not measured, so small differences should not be read as a winner.
- No model reaches the 82% macro F1 target in any GoEmotions view.
- SemEval: 675 test tweets, neutral class excluded, emotion-to-polarity rule fixed in the evaluation plan before scoring. VADER rule: compound >= 0.05 means positive.
- ISEAR: Hugging Face mirror subset, 3,752 rows with a direct label match; the predicted label is the top-1 of 28; shame and guilt rows are excluded from the score.
- The Ekman-grouped ISEAR column is exploratory and was added after the primary ISEAR results were known.
