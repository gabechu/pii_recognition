import numpy as np
from .metrics import compute_f_beta


def test_compute_f_beta():
    actual = compute_f_beta(0., 0.)
    assert np.isnan(actual)

    actual = compute_f_beta(np.nan, 1.)
    assert np.isnan(actual)

    actual = compute_f_beta(1., np.nan)
    assert np.isnan(actual)

    actual = compute_f_beta(1., 1.)
    assert actual == 1.

    actual = compute_f_beta(0.3, 0.5, beta=1.)
    assert np.isclose(actual, 0.3749999999)

    actual = compute_f_beta(0.3, 0.5, beta=2.)
    assert np.isclose(actual, 0.44117647058)
