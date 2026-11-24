# MoodMentor Milestone 2: Evaluation Plan

Written and committed **before** any model is trained. Any change after this
point must be listed in the "Changes after commit" section at the bottom, with
the reason. The date of this file is the date of its first Git commit.

## 1. Data

- GoEmotions, the "simplified" version, with its official train, validation, and
  test splits. The exact version is recorded in `ml/results/goemotions_manifest.json`.
- **Cleaned evaluation sets:** rows of the validation and test splits whose text
  (ignoring case and extra spaces) also appears in the training split are
  removed. The training split is not changed. Primary results use the cleaned
  sets; results on the official sets are also reported, for comparison with
  published numbers.
- The text is not modified (no lowercasing, no stop-word removal).

## 2. Task and models

- Multi-label classification over all 28 GoEmotions labels (27 emotions plus neutral).
- Two models, `bert-base-cased` and `distilbert-base-cased`, fine-tuned by us
  on the same data with the same seed and settings. Training settings are fixed
  in the committed training scripts before the test data is used.
- Loss: binary cross-entropy with logits, with a positive weight for each label
  equal to the square root of (examples without the label / examples with it),
  calculated on the training split only.
- The confidence score of a label is the model's sigmoid probability for it.

## 3. Choices made on validation data only

- The checkpoint (training epoch) is chosen by macro F1 over the 28 labels on
  the cleaned validation set.
- The prediction threshold is chosen on the cleaned validation set only: first
  one global threshold, and optionally per-label thresholds (also validation only).
- The test set is used once per final model, after the checkpoint and threshold
  are fixed.

## 4. Evaluation views and metrics

Views (all computed from the same 28-label predictions):

- **A. 28 labels.**
- **B. Ekman:** six basic emotions plus neutral, using the GoEmotions authors'
  grouping. A group counts as present if any of its member labels is present,
  in both the true labels and the predictions.
- **C. Headline emotions:** the 12 requested emotions that the dataset supports:
  joy, sadness, anger, fear, surprise, disgust, love, excitement, confusion,
  embarrassment, pride, nervousness.

For each view we report:

- Macro precision, recall, and F1 (computed for each label, then averaged with
  equal weight). Labels with no true example in an evaluation set are left out
  of the average and listed.
- Micro precision, recall, and F1 (pooled over all label decisions).
- Per-label precision, recall, F1, and the number of true examples.
- Exact-match accuracy (an example is correct only if every label in the view is
  right) and Hamming accuracy (the share of individual label decisions that are right).

## 5. The mentor's planned targets

These are targets, not results.

- **At least 82% macro F1 on GoEmotions validation.** We report macro F1 for all
  three views on both the cleaned and the official validation sets. The views
  and metrics are fixed here and will not be changed after seeing results in
  order to reach the target. If the target is not met we report the gap, the
  likely reasons, and possible improvements.
- **At least six emotion categories** with confidence scores: met by design
  (28 labels; view B has six emotions plus neutral).
- **At least 90% polarity accuracy on the SemEval binary subset.** The rule for
  deriving the binary subset, and for turning predicted emotions into a
  polarity, will be written down before running on SemEval. VADER is evaluated
  on the same subset.

## 6. External data

SemEval-2018 Task 1 and ISEAR are used for evaluation only. They are never used
for training, for choosing a checkpoint, or for tuning a threshold. Label
mappings are documented before evaluation.

## 7. Baselines

- VADER is compared only on tasks where the comparison is fair (polarity).
- The pretrained model already used in the app (`SamLowe/roberta-base-go_emotions`)
  may appear as a reference row, clearly labeled as trained by others and
  possibly trained on this training data.

## 8. Reproducibility

- Seed 42. The dataset version is pinned. The software environment is recorded in
  `ml/results/environment.json`. If compute allows, the runs are repeated with
  seeds 43 and 44 and the mean and spread are reported.

## 9. Known limitations (to be expanded in the final report)

- Reddit comments only, in English. Results on other kinds of text may be worse.
- Annotation is noisy and subjective; rare emotions have few examples.
- The simplified data has no author information, so same-author leakage
  between splits cannot be checked.
- Multi-label scores depend on the threshold. Confidence scores are not
  guaranteed to be calibrated probabilities.
- Predictions are guesses about text, not facts about a person's mental state.

## Changes after commit

None yet.

## ISEAR held-out evaluation (rules fixed before any model is scored on ISEAR)

**Source.** Hugging Face community mirror `TahaRasouli/ISEAR`, file `isear-train.csv` only. This is a subset of the full ISEAR (about 7.6k sentences), not the complete dataset.

**Cleaning.** 5,239 rows remain after removing 53 "[ No response.]" rows, 28 rows with conflicting duplicate texts and 35 duplicate texts. No ISEAR text overlaps with GoEmotions train or SemEval train.

**Held-out rule.** ISEAR is never used for training, hyperparameter tuning, threshold selection or early stopping. It is scored once per final model.

**Label mapping.**
- joy, fear, anger, sadness and disgust map to the same-named label. This gives 3,752 rows in the main evaluation.
- shame and guilt have no matching GoEmotions label. They are kept in the file (1,487 rows) but are excluded from the main score and reported separately.
- A model without a disgust label (the 6-class model: joy, sadness, love, anger, fear, surprise) is scored only on joy, fear, anger and sadness rows (3,003 rows).

**Scoring.** Accuracy and macro F1 on the mapped rows, plus a per-class table. If a model predicts a label that ISEAR does not contain (for example love or surprise), that prediction counts as wrong.

**Known limits.** ISEAR is long first-person reports written by students in the 1990s, a very different domain from Reddit comments. Lower scores than on GoEmotions are expected, and they will be reported as they are.

## SemEval polarity evaluation (rules fixed before any model is scored on SemEval)

**Data.** SemEval-2018 Task 1 V-oc, English, binary subset (neutral class 0 excluded): `ml/data/processed/semeval2018_voc_binary/`. The test file (675 tweets) is the scored set. SemEval data is never used to train or tune the emotion models.

**Emotion-to-polarity rule.** The model's 28 label probabilities are grouped following the positive/negative grouping used in the GoEmotions paper:
- Positive: admiration, amusement, approval, caring, desire, excitement, gratitude, joy, love, optimism, pride, relief.
- Negative: anger, annoyance, disappointment, disapproval, disgust, embarrassment, fear, grief, nervousness, remorse, sadness.
- Not used for polarity: confusion, curiosity, realization, surprise, neutral.

A tweet is predicted **positive** if the highest probability among the positive labels is greater than the highest probability among the negative labels. Otherwise (including ties) it is predicted **negative**. No threshold is used and nothing is tuned on SemEval.

**Baseline.** VADER on the same tweets: positive if the compound score is 0.05 or higher, otherwise negative.

**Metrics.** Accuracy, macro F1, and per-class precision and recall, for each transformer model and for VADER.

**Known limits.** SemEval is tweets, GoEmotions is Reddit comments. Neutral tweets are excluded, so these scores say nothing about how well a model handles neutral text.

**ISEAR prediction rule.** For each ISEAR text the model's single predicted label is the one with the highest probability among all 28 GoEmotions labels (no threshold). Predictions are compared with the mapped ISEAR label. A prediction such as neutral, love or surprise counts as wrong. Macro F1 is averaged over the five mapped classes (joy, fear, anger, sadness, disgust). For shame and guilt rows, only the distribution of predicted labels is reported. Texts are truncated at 64 tokens, the same as in training.