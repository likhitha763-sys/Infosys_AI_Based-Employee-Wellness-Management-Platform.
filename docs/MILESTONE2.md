# MoodMentor Milestone 2: fine-tuned emotion model, evaluation and safeguards

## What was built
- A multi-label emotion model (28 GoEmotions labels) fine-tuned from BERT and DistilBERT, with the same settings and a decision threshold chosen on the validation set only.
- A held-out evaluation on SemEval-2018 Task 1 (positive or negative tweets) and ISEAR (long first-person reports).
- An API and Streamlit app that show emotions with confidence next to the VADER baseline, with a fallback to the earlier emotion model if the new one cannot load.
- Privacy and safety features: consent before analysis, an option to analyse without saving, data export and deletion, an urgent-text support message, and an uncertainty note for very short text.

## Data
| Set | Use | Size after cleaning |
|---|---|---|
| GoEmotions (simplified) | train / validation / test | 43,410 / 5,377 / 5,385 |
| SemEval-2018 V-oc, binary | evaluation only | 675 test tweets (neutral excluded) |
| ISEAR (community mirror subset) | evaluation only | 5,239 rows; 3,752 with a direct label match |

Evaluation rules (thresholds, label mappings, the emotion-to-polarity rule) were written in `ml/results/evaluation_plan.md` before any model was scored on the external sets. No SemEval or ISEAR text appears in the training data.

## Results (one run per model, seed 42)
| | BERT | DistilBERT |
|---|---|---|
| GoEmotions test macro F1, 28 labels | 0.512 | 0.516 |
| Ekman 6 + neutral | 0.621 | 0.624 |
| 12 headline emotions | 0.530 | 0.530 |
| Validation macro F1 (threshold chosen here) | 0.532 | 0.522 |
| Macro precision / recall | 0.510 / 0.528 | 0.496 / 0.549 |
| Accuracy per yes/no decision | 96.6% | 96.4% |
| Exact match (all 28 right) | 45.0% | 42.6% |

| External test | VADER | DistilBERT | BERT |
|---|---|---|---|
| SemEval polarity accuracy | 83.7% | 83.4% | 81.6% |
| ISEAR accuracy (5 classes) | n/a | 29.4% | 31.3% |
| ISEAR predicted another emotion | n/a | 64.7% | 62.2% |

## Planned targets
| Target | Result | Status |
|---|---|---|
| Macro F1 82% on GoEmotions validation | 52.2% to 53.2% | Not reached |
| Polarity accuracy 90% on SemEval | 81.6% to 83.4% | Not reached |

Results are reported as measured, and no target was changed.

## What the results mean
- The two models are equal on the GoEmotions test set. Differences of about 0.004 come from one run each, so no model is declared better.
- The transformers do not beat VADER on positive or negative polarity. Their value is naming emotions, several at a time, with a confidence.
- High accuracy hides weak rare emotions: 96.6% of single yes/no decisions are right, but macro F1 is 0.51. Grief (5 test examples) scores 0, and relief (11), pride (16) and nervousness (23) are unreliable.
- On ISEAR, the model often answers "neutral" for long first-person reports. This is a domain shift (short Reddit comments versus long reports), not proof that the model is broken.

## Limitations
- Trained on English Reddit comments; the raters often disagree on subtle emotions.
- The text is cut at 64 tokens (about 45 words), so long entries are read only from the beginning.
- ISEAR was a mirror containing only part of the dataset. Shame and guilt have no matching label and are not scored.
- The urgent-text check is a short phrase list. It misses many cases and sometimes fires on harmless text.
- This is not a medical or psychological tool and does not diagnose anyone.

## Demo checklist
1. Consent box unticked: analysis is blocked.
2. Emotion Analytics, "I'm excited about the trip but also really nervous.": VADER gives one label, the model gives several emotions with confidence.
3. Error case, "ok": the model says approval, and the app shows the "very short text" warning.
4. "I don't want to live anymore": the support message appears.
5. Saving switched off: nothing appears in My History.
6. Model Performance: show the targets table (both "Not reached"), the SemEval matrices and the ISEAR result.
7. Privacy & Data: export, then delete.

## Reproducing
See `ml/results/evaluation_plan.md`, the data scripts in `ml/data_prep/`, `ml/train_goemotions.py` and `ml/score_external.py`, and the Colab notebook in `ml/notebooks/`. Rebuild the on-screen numbers with `python -m ml.make_frontend_metrics`.