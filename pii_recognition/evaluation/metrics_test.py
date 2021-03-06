import numpy as np
from numpy.testing import assert_almost_equal

from .metrics import compute_f_beta, compute_label_precision, compute_label_recall


def test_compute_f_beta_for_zero_precision_recall():
    actual = compute_f_beta(0.0, 0.0)
    assert actual == 0.0


def test_compute_f_beta_for_nan_numerator():
    actual = compute_f_beta(float("nan"), 1.0)
    assert np.isnan(actual)


def test_test_compute_f_beta_for_nan_denominator():
    actual = compute_f_beta(1.0, float("nan"))
    assert np.isnan(actual)


def test_compute_f_beta():
    actual = compute_f_beta(1.0, 1.0)
    assert actual == 1.0

    actual = compute_f_beta(0.3, 0.5, beta=1.0)
    assert np.isclose(actual, 0.3749999999)

    actual = compute_f_beta(0.3, 0.5, beta=2.0)
    assert np.isclose(actual, 0.44117647058)


def test_compute_label_precision_for_str():
    y_true = ["1"]
    y_pred = ["1"]
    actual = compute_label_precision(y_true, y_pred, label_name="1")
    assert actual == 1.0


def test_compute_label_precision_for_int():
    y_true = [1]
    y_pred = [1]
    actual = compute_label_precision(y_true, y_pred, label_name=1)
    assert actual == 1.0


def test_compute_label_precision_for_binary():
    y_true = [0, 0, 1]
    y_pred = [0, 1, 1]
    actual = compute_label_precision(y_true, y_pred, label_name=1)
    assert actual == 0.5


def test_compute_label_precision_for_multiclass():
    y_true = [0, 0, 0, 1, 0, 0, 2, 0, 2]
    y_pred = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    actual = compute_label_precision(y_true, y_pred, label_name=1)
    assert_almost_equal(actual, 0.3333333)
    actual = compute_label_precision(y_true, y_pred, label_name=2)
    assert_almost_equal(actual, 0.6666666)


def test_compute_label_precision_for_nonexist_labels():
    y_true = [0, 1]
    y_pred = [1, 1]
    actual = compute_label_precision(y_true, y_pred, label_name=2)
    assert actual == 0.0


def test_compute_label_recall_for_binary():
    y_true = [0, 1, 1]
    y_pred = [0, 0, 1]
    actual = compute_label_recall(y_true, y_pred, label_name=1)
    assert actual == 0.5


def test_compute_label_recall_for_multiclass():
    y_true = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    y_pred = [0, 0, 0, 1, 0, 0, 2, 0, 2]
    actual = compute_label_recall(y_true, y_pred, label_name=1)
    assert_almost_equal(actual, 0.3333333)
    actual = compute_label_recall(y_true, y_pred, label_name=2)
    assert_almost_equal(actual, 0.6666666)


def test_compute_label_recall_for_nonexist_labels():
    y_true = [0, 1]
    y_pred = [0, 1]
    actual = compute_label_recall(y_true, y_pred, label_name=2)
    assert actual == 0.0
