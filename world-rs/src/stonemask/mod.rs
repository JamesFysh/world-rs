use crate::constants::{K_FLOOR_F0_STONE_MASK, K_LOG2, K_MY_SAFE_GUARD_MINIMUM, K_PI};
use crate::fft::forward_real_fft;
use crate::matlab::matlab_round;

/// Allocation budget for one `get_refined_f0` frame: at most
/// `MAX_STONEMASK_BASE_TIME` time samples (~48MB across the six per-frame
/// buffers) and `fft_size ≤ MAX_STONEMASK_FFT`. Degenerate plans (e.g. `fs`
/// near 1e9 with a low f0) return 0.0 (unvoiced) instead of attempting a
/// multi-GB allocation.
const MAX_STONEMASK_BASE_TIME: usize = 1_000_000;
const MAX_STONEMASK_FFT: usize = 1 << 20;

/// Error returned by [`stone_mask`] when the input is invalid.
#[derive(Debug, Clone, PartialEq)]
pub enum StoneMaskError {
    /// Sample rate `fs` is non-finite or non-positive.
    NonFiniteSampleRate,
    /// `f0` contains non-finite values.
    NonFiniteF0,
    /// `temporal_positions` and `f0` have different lengths.
    MismatchedLengths,
    /// `f0_length` does not match the length of `f0` or `temporal_positions`.
    MismatchedF0Length,
}

impl std::fmt::Display for StoneMaskError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoneMaskError::NonFiniteSampleRate => {
                write!(f, "sample rate must be finite and positive")
            }
            StoneMaskError::NonFiniteF0 => write!(f, "f0 contains non-finite values"),
            StoneMaskError::MismatchedLengths => {
                write!(f, "temporal_positions and f0 have different lengths")
            }
            StoneMaskError::MismatchedF0Length => {
                write!(f, "f0_length does not match input lengths")
            }
        }
    }
}

impl std::error::Error for StoneMaskError {}

pub fn stone_mask(
    x: &[f64],
    x_length: usize,
    fs: f64,
    temporal_positions: &[f64],
    f0: &[f64],
    f0_length: usize,
) -> Result<Vec<f64>, StoneMaskError> {
    if !fs.is_finite() || fs <= 0.0 || fs < 1.0 || fs > 1e9 {
        return Err(StoneMaskError::NonFiniteSampleRate);
    }
    if temporal_positions.len() != f0.len() {
        return Err(StoneMaskError::MismatchedLengths);
    }
    if f0_length != f0.len() || f0_length != temporal_positions.len() {
        return Err(StoneMaskError::MismatchedF0Length);
    }
    if f0.iter().any(|v| !v.is_finite()) {
        return Err(StoneMaskError::NonFiniteF0);
    }
    let mut refined = vec![0.0; f0_length];
    for i in 0..f0_length {
        let initial_f0 = f0[i];
        refined[i] = get_refined_f0(x, x_length, fs, temporal_positions[i], initial_f0);
    }
    Ok(refined)
}

fn get_refined_f0(
    x: &[f64],
    x_length: usize,
    fs: f64,
    current_position: f64,
    initial_f0: f64,
) -> f64 {
    // C++ guard (stonemask.cpp:187): reject outside (kFloorF0StoneMask, fs/12].
    if initial_f0 <= K_FLOOR_F0_STONE_MASK || initial_f0 > fs / 12.0 {
        return 0.0;
    }
    let half_window_length = (1.5 * fs / initial_f0 + 1.0) as usize;
    let base_time_length = 2 * half_window_length + 1;
    if base_time_length <= 1 {
        return 0.0;
    }
    // Budget-check before allocating: `fft_size` and `base_time_length`
    // both derive from `fs / initial_f0`, which is unbounded within the
    // validated ranges (`fs ≤ 1e9`, `f0 > 40`).
    let n = base_time_length as f64;
    let floor_log2 = (n.ln() / K_LOG2).floor() as usize;
    let fft_size = 1usize << (2 + floor_log2);
    if base_time_length > MAX_STONEMASK_BASE_TIME || fft_size > MAX_STONEMASK_FFT {
        return 0.0;
    }
    let window_length_in_time = base_time_length as f64 / fs;
    let base_time: Vec<f64> = (0..base_time_length)
        .map(|i| (-(half_window_length as f64) + i as f64) / fs)
        .collect();
    let mean_f0 = get_mean_f0(
        x,
        x_length,
        fs,
        current_position,
        initial_f0,
        fft_size,
        window_length_in_time,
        &base_time,
    );
    if (mean_f0 - initial_f0).abs() > initial_f0 * 0.2 {
        initial_f0
    } else {
        mean_f0
    }
}

#[allow(clippy::too_many_arguments)]
fn get_mean_f0(
    x: &[f64],
    x_length: usize,
    fs: f64,
    current_position: f64,
    initial_f0: f64,
    fft_size: usize,
    window_length_in_time: f64,
    base_time: &[f64],
) -> f64 {
    let base_time_length = base_time.len();
    let mut index_raw = vec![0i64; base_time_length];
    for i in 0..base_time_length {
        let pos = (current_position + base_time[i]) * fs;
        index_raw[i] = matlab_round(pos);
    }
    let mut main_window = vec![0.0; base_time_length];
    for i in 0..base_time_length {
        let tmp = (index_raw[i] as f64 - 1.0) / fs - current_position;
        main_window[i] = 0.42
            + 0.5 * (2.0 * K_PI * tmp / window_length_in_time).cos()
            + 0.08 * (4.0 * K_PI * tmp / window_length_in_time).cos();
    }
    let mut diff_window = vec![0.0; base_time_length];
    diff_window[0] = -main_window[1] / 2.0;
    for i in 1..base_time_length - 1 {
        diff_window[i] = -(main_window[i + 1] - main_window[i - 1]) / 2.0;
    }
    if base_time_length > 1 {
        diff_window[base_time_length - 1] = main_window[base_time_length - 2] / 2.0;
    }
    let mut index = vec![0usize; base_time_length];
    for i in 0..base_time_length {
        let raw = index_raw[i] - 1;
        let clamped = if raw < 0 {
            0
        } else if raw as usize >= x_length {
            x_length - 1
        } else {
            raw as usize
        };
        index[i] = clamped;
    }
    // Main spectrum
    let mut waveform = vec![0.0; fft_size];
    for i in 0..base_time_length {
        waveform[i] = x[index[i]] * main_window[i];
    }
    let main_spec = forward_real_fft(&waveform);
    // Diff spectrum
    for i in 0..base_time_length {
        waveform[i] = x[index[i]] * diff_window[i];
    }
    waveform[base_time_length..].fill(0.0);
    let diff_spec = forward_real_fft(&waveform);
    let spec_len = fft_size / 2 + 1;
    let mut power_spectrum = vec![0.0; spec_len];
    let mut numerator_i = vec![0.0; spec_len];
    for j in 0..spec_len {
        let mr = main_spec[j].re;
        let mi = main_spec[j].im;
        let dr = diff_spec[j].re;
        let di = diff_spec[j].im;
        numerator_i[j] = mr * di - mi * dr;
        power_spectrum[j] = mr * mr + mi * mi;
    }
    get_tentative_f0(&power_spectrum, &numerator_i, fft_size, fs, initial_f0)
}

fn get_tentative_f0(
    power_spectrum: &[f64],
    numerator_i: &[f64],
    fft_size: usize,
    fs: f64,
    initial_f0: f64,
) -> f64 {
    let tentative = fix_f0(power_spectrum, numerator_i, fft_size, fs, initial_f0, 2);
    if tentative <= 0.0 || tentative > initial_f0 * 2.0 {
        return 0.0;
    }
    fix_f0(power_spectrum, numerator_i, fft_size, fs, tentative, 6)
}

fn fix_f0(
    power_spectrum: &[f64],
    numerator_i: &[f64],
    fft_size: usize,
    fs: f64,
    initial_f0: f64,
    num_harmonics: usize,
) -> f64 {
    let mut amplitude_list = vec![0.0; num_harmonics];
    let mut inst_freq_list = vec![0.0; num_harmonics];
    for i in 0..num_harmonics {
        let idx_f = initial_f0 * fft_size as f64 / fs * (i as f64 + 1.0);
        let idx = matlab_round(idx_f) as usize;
        let idx_clamped = idx.min(fft_size / 2);
        let power = power_spectrum[idx_clamped];
        let inst = if power == 0.0 {
            0.0
        } else {
            idx_clamped as f64 * fs / fft_size as f64
                + numerator_i[idx_clamped] / power * fs / 2.0 / K_PI
        };
        inst_freq_list[i] = inst;
        amplitude_list[i] = power.sqrt();
    }
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for i in 0..num_harmonics {
        numerator += amplitude_list[i] * inst_freq_list[i];
        denominator += amplitude_list[i] * (i as f64 + 1.0);
    }
    numerator / (denominator + K_MY_SAFE_GUARD_MINIMUM)
}

#[cfg(test)]
// Expected values are full-precision f64 C++ reference vectors.
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    // Sine signal identical to the C++ reference driver
    // (/tmp/opencode/stonemask_vectors.cpp): x[i] = sin(2*pi*f0_true*i/fs).
    fn sine_signal(n: usize, fs: f64, f0_true: f64) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * K_PI * f0_true * (i as f64) / fs).sin())
            .collect()
    }

    // Tolerance for FFT-path vectors: the C++ reference uses the Ooura FFT
    // (libworld.a) while Rust uses rustfft/realfft, so the refined f0 differs
    // by a few ULPs in the spectra. The refined value is a robust ratio, so the
    // end-to-end difference is far below 1e-6.
    const TOL_REF: f64 = 1e-6;

    #[test]
    fn stone_mask_basic() {
        let x = vec![0.0; 1024];
        let temporal = vec![0.0; 10];
        let f0 = vec![100.0; 10];
        let out = stone_mask(&x, x.len(), 16000.0, &temporal, &f0, 10).unwrap();
        assert_eq!(out.len(), 10);
    }

    #[test]
    fn stone_mask_guard_low() {
        let x = vec![0.0; 1024];
        let temporal = vec![0.0];
        let f0 = vec![30.0];
        let out = stone_mask(&x, x.len(), 16000.0, &temporal, &f0, 1).unwrap();
        assert_eq!(out[0], 0.0);
    }

    #[test]
    fn stone_mask_guard_high() {
        let x = vec![0.0; 1024];
        let temporal = vec![0.0];
        let f0 = vec![2000.0];
        let out = stone_mask(&x, x.len(), 16000.0, &temporal, &f0, 1).unwrap();
        assert_eq!(out[0], 0.0);
    }

    #[test]
    fn stone_mask_sine_refinement() {
        let fs = 16000.0;
        let f0_true = 200.0;
        let n = 4096;
        let mut x = vec![0.0; n];
        for (i, xv) in x.iter_mut().enumerate() {
            let t = i as f64 / fs;
            *xv = (2.0 * std::f64::consts::PI * f0_true * t).sin();
        }
        let temporal = vec![n as f64 / 2.0 / fs];
        let f0 = vec![f0_true];
        let out = stone_mask(&x, n, fs, &temporal, &f0, 1).unwrap();
        // Refined f0 should be close to true f0 within 1 Hz
        assert!(out[0].abs() > 0.0);
        assert!(
            (out[0] - f0_true).abs() < 1.0,
            "refined {} vs {}",
            out[0],
            f0_true
        );
    }

    // ---------------------------------------------------------------------
    // C++ reference vectors.
    //
    // Reference values captured from the C++ StoneMask (mmorise/World) linked
    // against ext_src/world-cpp/build/libworld.a via the driver
    // /tmp/opencode/stonemask_vectors.cpp (see its header for the exact
    // compile/run commands). The driver prints refined_f0 at full f64
    // precision; the values below are hard-coded here. The Rust tests
    // re-generate the SAME deterministic input signals and assert the Rust
    // output matches the C++ reference.
    // ---------------------------------------------------------------------

    // A: pure sine 200 Hz, fs=16000, n=8192, single frame at center.
    // C++: refined[0] = 199.67434898305939
    #[test]
    fn matches_cpp_ref_a_sine200_fs16000() {
        let fs = 16000.0;
        let n = 8192;
        let x = sine_signal(n, fs, 200.0);
        let temporal = vec![n as f64 / 2.0 / fs];
        let f0 = vec![200.0];
        let out = stone_mask(&x, n, fs, &temporal, &f0, 1).unwrap();
        assert!(
            (out[0] - 199.67434898305939).abs() < TOL_REF,
            "got {}",
            out[0]
        );
    }

    // B: pure sine 150 Hz, fs=44100, n=16384, single frame at center.
    // C++: refined[0] = 149.8954510311552
    #[test]
    fn matches_cpp_ref_b_sine150_fs44100() {
        let fs = 44100.0;
        let n = 16384;
        let x = sine_signal(n, fs, 150.0);
        let temporal = vec![n as f64 / 2.0 / fs];
        let f0 = vec![150.0];
        let out = stone_mask(&x, n, fs, &temporal, &f0, 1).unwrap();
        assert!(
            (out[0] - 149.8954510311552).abs() < TOL_REF,
            "got {}",
            out[0]
        );
    }

    // C: all-zero spectrum, fs=16000, f0=200. Zero power -> tentative rejected
    // -> 20% clamp keeps the initial value exactly. C++: refined[0] = 200.
    #[test]
    fn matches_cpp_ref_c_zero_fs16000() {
        let fs = 16000.0;
        let n = 1024;
        let x = vec![0.0; n];
        let temporal = vec![n as f64 / 2.0 / fs];
        let f0 = vec![200.0];
        let out = stone_mask(&x, n, fs, &temporal, &f0, 1).unwrap();
        assert_eq!(out[0], 200.0);
    }

    // D: sine at 250 Hz but initial f0=200 (offset). The 20% clamp rejects the
    // correction and returns the initial value exactly. C++: refined[0] = 200.
    #[test]
    fn matches_cpp_ref_d_sine250_f0init200_fs16000() {
        let fs = 16000.0;
        let n = 8192;
        let x = sine_signal(n, fs, 250.0);
        let temporal = vec![n as f64 / 2.0 / fs];
        let f0 = vec![200.0];
        let out = stone_mask(&x, n, fs, &temporal, &f0, 1).unwrap();
        assert!((out[0] - 200.0).abs() < TOL_REF, "got {}", out[0]);
    }

    // E: guard cases (output independent of x). f0<=40 -> 0; f0>fs/12 -> 0.
    // C++: refined = [0, 0].
    #[test]
    fn matches_cpp_ref_e_guards_fs16000() {
        let fs = 16000.0;
        let n = 1024;
        let x = sine_signal(n, fs, 200.0);
        let temporal = vec![0.1, 0.2];
        let f0 = vec![30.0, 2000.0];
        let out = stone_mask(&x, n, fs, &temporal, &f0, 2).unwrap();
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 0.0);
    }

    // F: multi-frame sine 220 Hz, fs=16000, n=8192, frames at 0.1/0.2/0.3 s.
    // C++: refined = [219.92606774461879, 219.92606774461768, 219.92606774461754].
    #[test]
    fn matches_cpp_ref_f_multiframe_sine220_fs16000() {
        let fs = 16000.0;
        let n = 8192;
        let x = sine_signal(n, fs, 220.0);
        let temporal = vec![0.1, 0.2, 0.3];
        let f0 = vec![220.0, 220.0, 220.0];
        let out = stone_mask(&x, n, fs, &temporal, &f0, 3).unwrap();
        let expected = [219.92606774461879, 219.92606774461768, 219.92606774461754];
        for i in 0..3 {
            assert!(
                (out[i] - expected[i]).abs() < TOL_REF,
                "frame {}: got {}",
                i,
                out[i]
            );
        }
    }
}
