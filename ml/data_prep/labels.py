"""Label sets and label conversions for GoEmotions (Milestone 2).

The model is trained on all 28 labels. Evaluation also looks at two other
"views" of the same predictions. Both are fixed in ml/results/evaluation_plan.md
before any model is trained.
"""

import numpy as np

from ml import config

LABELS = tuple(config.GOEMOTIONS_LABELS)
NUM_LABELS = len(LABELS)
LABEL_INDEX = {name: index for index, name in enumerate(LABELS)}

# View 2: Ekman's six basic emotions plus neutral, in this fixed order.
EKMAN_GROUPS = tuple(config.EKMAN_MAPPING) + ("neutral",)

# View 3: the requested emotions that the dataset can detect.
# ("nervousness" stands in for anxiety.)
HEADLINE_EMOTIONS = (
    "joy", "sadness", "anger", "fear", "surprise", "disgust", "love",
    "excitement", "confusion", "embarrassment", "pride", "nervousness",
)
HEADLINE_INDICES = [LABEL_INDEX[name] for name in HEADLINE_EMOTIONS]


def multi_hot(label_lists, number_of_labels=NUM_LABELS):
    """Turn lists of label numbers into a 0/1 table: one row per example,
    one column per label."""
    matrix = np.zeros((len(label_lists), number_of_labels), dtype=np.int8)
    for row, labels in enumerate(label_lists):
        for label in labels:
            matrix[row, label] = 1
    return matrix


def _group_matrix():
    """A 28 x 7 table: entry [label, group] is 1 if the label belongs to the group."""
    matrix = np.zeros((NUM_LABELS, len(EKMAN_GROUPS)), dtype=np.int8)
    for group_index, group in enumerate(EKMAN_GROUPS):
        members = ("neutral",) if group == "neutral" else config.EKMAN_MAPPING[group]
        for member in members:
            matrix[LABEL_INDEX[member], group_index] = 1
    return matrix


GROUP_MATRIX = _group_matrix()


def to_ekman_view(matrix):
    """28-label 0/1 table -> 7-group table. A group is 1 if any member label is 1."""
    return ((matrix.astype(np.int32) @ GROUP_MATRIX) > 0).astype(np.int8)


def to_headline_view(matrix):
    """28-label 0/1 table -> only the 12 headline emotions."""
    return matrix[:, HEADLINE_INDICES]