import numpy as np


def compute_f_beta(precision: float, recall: float, beta: float) -> float:
    if np.isnan(precision) or np.isnan(recall) or (precision == 0 and recall == 0):
        return np.nan

    return ((1 + beta ** 2) * precision * recall) / (((beta ** 2) * precision) + recall)
