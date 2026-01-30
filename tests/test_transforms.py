import numpy as np 
from housing_model.features import safe_ratio, safe_log1p


def test_safe_ratio_div_by_zero():
    X = np.array([[10, 2], [5, 0]])
    y = safe_ratio(X)
    assert y.shape == (2, 1)
    assert y[0, 0] == 5.0
    assert y[1, 0] == 0.0


def test_safe_log1p_handles_zero_and_negative():
    X = np.array([[0.0], [-3.0], [9.0]])
    y = safe_log1p(X)
    assert np.isfinite(y).all()



