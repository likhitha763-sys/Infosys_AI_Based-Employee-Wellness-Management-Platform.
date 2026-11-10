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