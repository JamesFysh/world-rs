import numpy as np
import pytest
from metrics import rmse, psnr, pearson_r, max_abs_error, voicing_agreement

def test_rmse_basic():
    a = np.array([0.0, 1.0, 2.0])
    b = np.array([0.0, 1.0, 3.0])
    val = rmse(a, b)
    expected = np.sqrt(1/3)
    assert abs(val - expected) < 1e-12

def test_psnr_identical():
    a = np.array([1.0, 2.0, 3.0])
    assert psnr(a, a) == float('inf')

def test_psnr_known():
    ref = np.array([1.0, 0.0, -1.0])
    deg = np.array([1.0, 0.0, -0.9])
    val = psnr(ref, deg)
    mse = np.mean((ref - deg) ** 2)
    peak = np.max(np.abs(ref))
    expected = 10 * np.log10((peak ** 2) / mse)
    assert abs(val - expected) < 1e-12

def test_pearson_r_perfect():
    a = np.array([1, 2, 3, 4])
    b = np.array([2, 4, 6, 8])
    assert abs(pearson_r(a, b) - 1.0) < 1e-12

def test_pearson_r_negative():
    a = np.array([1, 2, 3])
    b = np.array([3, 2, 1])
    assert abs(pearson_r(a, b) + 1.0) < 1e-12

def test_max_abs_error():
    a = np.array([1.0, -2.0, 3.0])
    b = np.array([1.1, -1.9, 3.0])
    assert max_abs_error(a, b) == pytest.approx(0.1)

def test_voicing_agreement():
    v1 = np.array([True, False, True, True])
    v2 = np.array([True, False, False, True])
    val = voicing_agreement(v1, v2)
    assert val == pytest.approx(75.0)

def test_voicing_agreement_empty():
    v1 = np.array([], dtype=bool)
    v2 = np.array([], dtype=bool)
    assert voicing_agreement(v1, v2) == 0.0
