from typing import List, TypeVar

import numpy as np
from sklearn.metrics import precision_score, recall_score

# LT for label type
LT = TypeVar("LT", int, str)


def compute_f_beta(precision: float, recall: float, beta: float = 1.0) -> float:
    if np.isnan(precision) or np.isnan(recall):
        return float("nan")

    if precision == 0.0 and recall == 0.0:
        return 0.0

    return ((1 + beta ** 2) * precision * recall) / (((beta ** 2) * precision) + recall)


def compute_label_precision(
    y_true: List[LT], y_pred: List[LT], label_name: LT,
) -> float:
    """Compute recall for a designated label.

    This can calculate precision of a particular label for both binary and multi-class
    settings. The invoked sklearn function is not stable on string and integer mixed
    labels, may encouter ValueError. So mixed types in an argument is not allowed.
    """
    return precision_score(y_true, y_pred, average=None, labels=[label_name])[0]


def compute_label_recall(y_true: List[LT], y_pred: List[LT], label_name: LT,) -> float:
    """Compute recall for a designated label.

    This can calculate recall of a particular label for both binary and multi-class
    settings. The invoked sklearn function is not stable on string and integer mixed
    labels, may encouter ValueError. So mixed types in an arguments is not allowed.
    """
    return recall_score(y_true, y_pred, average=None, labels=[label_name])[0]
