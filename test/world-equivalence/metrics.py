import numpy as np
import warnings
from scipy.signal import resample_poly

def rmse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Shapes must match for RMSE")
    return float(np.sqrt(np.mean((a - b) ** 2)))

def psnr(ref, deg):
    ref = np.asarray(ref, dtype=np.float64)
    deg = np.asarray(deg, dtype=np.float64)
    if ref.shape != deg.shape:
        raise ValueError("Shapes must match for PSNR")
    if np.allclose(ref, deg):
        return float('inf')
    mse = np.mean((ref - deg) ** 2)
    if mse <= 0:
        return float('inf')
    peak = np.max(np.abs(ref))
    if peak < 1e-12:
        return float('-inf')
    return float(10.0 * np.log10((peak ** 2) / mse))

def pearson_r(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Shapes must match for Pearson")
    a_flat = a.ravel()
    b_flat = b.ravel()
    if a_flat.size == 0:
        return float('nan')
    if np.allclose(a_flat, b_flat):
        return 1.0
    a_mean = np.mean(a_flat)
    b_mean = np.mean(b_flat)
    num = np.sum((a_flat - a_mean) * (b_flat - b_mean))
    den = np.sqrt(np.sum((a_flat - a_mean) ** 2) * np.sum((b_flat - b_mean) ** 2))
    if den == 0:
        return 0.0
    return float(num / den)

def max_abs_error(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Shapes must match for max_abs_error")
    return float(np.max(np.abs(a - b)))

def voicing_agreement(v1, v2):
    v1 = np.asarray(v1, dtype=np.bool_)
    v2 = np.asarray(v2, dtype=np.bool_)
    if v1.shape != v2.shape:
        raise ValueError("Shapes must match for voicing_agreement")
    if v1.size == 0:
        return 0.0
    agreement = np.mean(v1 == v2) * 100.0
    return float(agreement)

def _resample_to_16k(signal, fs):
    signal = np.asarray(signal, dtype=np.float64)
    if fs == 16000 or fs <= 0:
        return signal, fs, False
    up = 16000
    down = int(fs)
    g = np.gcd(up, down)
    up //= g
    down //= g
    resampled = resample_poly(signal, up, down)
    return resampled, 16000, True

def _level_align(ref, deg):
    ref = np.asarray(ref, dtype=np.float64)
    deg = np.asarray(deg, dtype=np.float64)
    rms_ref = np.sqrt(np.mean(ref ** 2))
    rms_deg = np.sqrt(np.mean(deg ** 2))
    if rms_deg > 1e-12:
        deg = deg * (rms_ref / rms_deg)
    return ref, deg

def pesq_score(ref, deg, fs=16000):
    ref = np.asarray(ref, dtype=np.float64)
    deg = np.asarray(deg, dtype=np.float64)
    duration = len(ref) / max(fs, 1)
    if duration < 0.5:
        warnings.warn(f"PESQ skipped: duration {duration:.2f}s < 0.5s")
        return None, False, fs
    try:
        import pesq
    except Exception:
        warnings.warn("pesq not installed; PESQ skipped")
        return None, False, fs
    ref_r, fs_r, resampled_r = _resample_to_16k(ref, fs)
    deg_r, fs_d, resampled_d = _resample_to_16k(deg, fs)
    # ensure same length
    min_len = min(len(ref_r), len(deg_r))
    ref_r = ref_r[:min_len]
    deg_r = deg_r[:min_len]
    ref_r, deg_r = _level_align(ref_r, deg_r)
    # pesq expects 16k or 8k
    try:
        score = pesq.pesq(fs_r, ref_r, deg_r, 'wb')
    except Exception as e:
        warnings.warn(f"PESQ computation failed: {e}")
        return None, False, fs
    resampled = resampled_r or resampled_d
    return float(score), resampled, fs_r

def stoi_score(ref, deg, fs=16000):
    ref = np.asarray(ref, dtype=np.float64)
    deg = np.asarray(deg, dtype=np.float64)
    try:
        import pystoi
    except Exception:
        warnings.warn("pystoi not installed; STOI skipped")
        return None, False, fs
    ref_r, fs_r, resampled_r = _resample_to_16k(ref, fs)
    deg_r, fs_d, resampled_d = _resample_to_16k(deg, fs)
    min_len = min(len(ref_r), len(deg_r))
    ref_r = ref_r[:min_len]
    deg_r = deg_r[:min_len]
    ref_r, deg_r = _level_align(ref_r, deg_r)
    try:
        score = pystoi.stoi(ref_r, deg_r, fs_r, extended=False)
    except Exception as e:
        warnings.warn(f"STOI computation failed: {e}")
        return None, False, fs
    resampled = resampled_r or resampled_d
    return float(score), resampled, fs_r
