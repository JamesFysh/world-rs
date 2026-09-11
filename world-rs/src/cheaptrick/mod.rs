//! CheapTrick spectral envelope extraction (port of
//! `ext_src/world-cpp/src/cheaptrick.cpp`).
//!
//! Ported per the WORLD spec (`docs/impl/support/WORLD-spec.md`, §CheapTrick)
//! and the locked pre-phase decisions (`docs/impl/support/WORLD-decisions.md`):
//! f64 throughout the core math, FFT backend Route A (`rustfft`/`realfft`) with
//! tolerance validation, and crate-private FFT wrappers. The `crate::fft` and
//! `crate::matlab` primitives are consumed by the per-frame DSP pipeline ported
//! in PHASE2-3..6.
//!
//! Phase 2-1: module scaffolding and type definitions.
//! Phase 2-2: option initialization & FFT size utilities.
//! Phase 2-3: core DSP helpers.
//! Phase 2-4: F0-adaptive windowing.
//! Phase 2-5: power spectrum & cepstral smoothing.
//! Phase 2-6: CheapTrickGeneralBody & main function.
//! Phase 2-7: unit tests & accuracy validation.
//! Phase 2-8: DIO integration & performance optimization.

use crate::common::{
    dc_correction_into, dc_correction_scratch_capacity, linear_smoothing_into,
    linear_smoothing_scratch_capacity,
};
use crate::constants::{K_DEFAULT_F0, K_EPS, K_FLOOR_F0, K_LOG2, K_MY_SAFE_GUARD_MINIMUM, K_PI};
use crate::fft::{ForwardRealFFT, InverseRealFFT};
use crate::matlab::{matlab_round, randn, randn_reseed, RandnState};
use num_complex::Complex;

/// C++: `CheapTrickOption` (world/cheaptrick.h:16-20). Field order matches the
/// C++ struct.
#[derive(Debug, Clone, PartialEq)]
pub struct CheapTrickOption {
    /// Cepstral smoothing/recovery parameter (default `-0.15`).
    pub q1: f64,
    /// Lower F0 limit (Hz) used to determine `fft_size` (default `kFloorF0`).
    pub f0_floor: f64,
    /// FFT size; `get_fft_size_for_cheaptrick(fs, self)`.
    pub fft_size: i32,
}

/// C++: `InitializeCheapTrickOption` (cheaptrick.cpp:231-240).
///
/// Returns the default parameters matching the C++ reference: `q1 = -0.15`,
/// `f0_floor = kFloorF0`, and `fft_size = get_fft_size_for_cheaptrick(fs, ...)`.
pub fn initialize_cheaptrick_option(fs: f64) -> CheapTrickOption {
    let option = CheapTrickOption {
        q1: -0.15,
        f0_floor: K_FLOOR_F0,
        fft_size: 0,
    };
    let fft_size = get_fft_size_for_cheaptrick(fs, &option);
    CheapTrickOption { fft_size, ..option }
}

/// C++: `GetFFTSizeForCheapTrick` (cheaptrick.cpp:191-194).
///
/// `2^(1 + floor(ln(3*fs/f0_floor + 1) / kLog2))` where `kLog2 = ln 2`. The C++
/// `static_cast<int>` is a floor (the Python port uses `ceil`; the Rust port
/// matches the C++ floor).
pub fn get_fft_size_for_cheaptrick(fs: f64, option: &CheapTrickOption) -> i32 {
    if !fs.is_finite() || fs <= 0.0 || !option.f0_floor.is_finite() || option.f0_floor <= 0.0 {
        return 0;
    }
    let val = CHEAPTRICK_MULTIPLIER * fs / option.f0_floor + CHEAPTRICK_OFFSET;
    if !val.is_finite() || val <= 0.0 {
        return 0;
    }
    let exp = (val.ln() / K_LOG2).floor() as i32;
    let shift = CHEAPTRICK_FFT_EXP_OFFSET + exp;
    if !(0..31).contains(&shift) {
        return 0;
    }
    1i32 << shift
}

/// C++: `GetF0FloorForCheapTrick` (cheaptrick.cpp:196-198).
///
/// Actual lower F0 limit (Hz) for the given `fft_size`: `3*fs/(fft_size - 3)`.
/// Frames with `f0 <= f0_floor` are analyzed as unvoiced (using `kDefaultF0`).
pub fn get_f0_floor_for_cheaptrick(fs: f64, fft_size: i32) -> f64 {
    CHEAPTRICK_MULTIPLIER * fs / (fft_size as f64 - CHEAPTRICK_FFT_DENOM_OFFSET)
}

/// C++: `AddInfinitesimalNoise` (cheaptrick.cpp:147-151).
///
/// Safeguard against exact-zero bins before the log transform in the cepstral
/// smoothing stage: adds `|randn| * kEps` to every bin of the (linear-axis)
/// power spectrum. `input_spectrum` and `output_spectrum` are both
/// `fft_size/2 + 1` long and may alias (CheapTrick calls it in place).
///
/// Draws exactly `fft_size/2 + 1` samples from `randn_state`; the consumption
/// order is part of the algorithm (see `WORLD-spec.md`, §CheapTrick). No
/// allocations occur in the loop.
pub fn add_infinitesimal_noise(
    input_spectrum: &[f64],
    fft_size: i32,
    output_spectrum: &mut [f64],
    randn_state: &mut RandnState,
) {
    let half = (fft_size / 2) as usize;
    for i in 0..=half {
        let base = input_spectrum.get(i).copied().unwrap_or(0.0);
        let noise = randn(randn_state).abs() * K_EPS;
        if let Some(slot) = output_spectrum.get_mut(i) {
            *slot = base + noise;
        }
    }
}

/// C++: `SetParametersForGetWindowedWaveform` (cheaptrick.cpp:87-107).
///
/// Computes the safe index (clamped sample positions) and the F0-adaptive
/// window for a single frame. `safe_index` and `window` must have at least
/// `2 * half_window_length + 1` elements.
///
/// # Panics
///
/// Panics in debug builds if `safe_index` or `window` are too short (`debug_assert!`).
#[allow(clippy::needless_range_loop)]
pub(crate) fn set_parameters_for_get_windowed_waveform(
    half_window_length: usize,
    x_length: usize,
    current_position: f64,
    fs: f64,
    current_f0: f64,
    safe_index: &mut [i32],
    window: &mut [f64],
) {
    let n = 2 * half_window_length + 1;
    debug_assert!(safe_index.len() >= n);
    debug_assert!(window.len() >= n);
    let origin = matlab_round(current_position * fs + 0.001) as i32;
    let hwl = half_window_length as i32;
    for i in 0..n {
        let base = i as i32 - hwl;
        safe_index[i] = (origin + base).clamp(0, (x_length - 1) as i32);
    }
    let mut average = 0.0f64;
    for i in 0..n {
        let base = (i as i32 - hwl) as f64;
        let position = base / 1.5 / fs;
        window[i] = 0.5 * (K_PI * position * current_f0).cos() + 0.5;
        average += window[i] * window[i];
    }
    average = average.sqrt();
    for i in 0..n {
        window[i] /= average;
    }
}

/// C++: `GetWindowedWaveform` (cheaptrick.cpp:112-142).
///
/// Applies the F0-adaptive window to the waveform at `current_position`,
/// adding infinitesimal noise and removing the DC component. `waveform`,
/// `safe_index`, and `window` must have at least
/// `2 * matlab_round(1.5 * fs / current_f0) + 1` elements.
///
/// # Panics
///
/// Panics in debug builds if `waveform`, `safe_index`, or `window` are too short (`debug_assert!`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn get_windowed_waveform(
    x: &[f64],
    fs: f64,
    current_f0: f64,
    current_position: f64,
    waveform: &mut [f64],
    safe_index: &mut [i32],
    window: &mut [f64],
    randn_state: &mut RandnState,
) {
    let half_window_length = matlab_round(1.5 * fs / current_f0) as usize;
    let n = 2 * half_window_length + 1;
    debug_assert!(waveform.len() >= n);
    debug_assert!(safe_index.len() >= n);
    debug_assert!(window.len() >= n);
    set_parameters_for_get_windowed_waveform(
        half_window_length,
        x.len(),
        current_position,
        fs,
        current_f0,
        safe_index,
        window,
    );
    for i in 0..n {
        waveform[i] =
            x[safe_index[i] as usize] * window[i] + randn(randn_state) * K_MY_SAFE_GUARD_MINIMUM;
    }
    let mut tmp_weight1 = 0.0f64;
    let mut tmp_weight2 = 0.0f64;
    for i in 0..n {
        tmp_weight1 += waveform[i];
        tmp_weight2 += window[i];
    }
    let weighting_coefficient = if tmp_weight2 == 0.0 {
        0.0
    } else {
        tmp_weight1 / tmp_weight2
    };
    for i in 0..n {
        waveform[i] -= window[i] * weighting_coefficient;
    }
}

/// C++: `GetPowerSpectrum` (cheaptrick.cpp:64-82).
///
/// Computes the power spectrum of the (windowed) waveform stored in
/// `forward.waveform` and applies DC correction, storing the result back in
/// `forward.waveform[0..=fft_size/2]`. The C++ reuses the FFT `waveform`
/// buffer to hold the power spectrum, so the Rust port does the same: on
/// entry `forward.waveform` holds the time-domain signal, on exit
/// `forward.waveform[0..=fft_size/2]` holds the DC-corrected power spectrum.
///
/// Steps (matching the C++):
/// 1. Zero-pad `forward.waveform[2*half_window_length+1 .. fft_size]` where
///    `half_window_length = matlab_round(1.5 * fs / f0)`.
/// 2. Forward real FFT (r2c) using the cached plan.
/// 3. `power[i] = re^2 + im^2` for `i = 0..=fft_size/2`.
/// 4. DC correction in place (via the pre-allocated `temp` buffer).
///
/// `dc_scratch` is the pre-allocated working buffer for the DC correction
/// (see [`dc_correction_scratch_capacity`]); it is reused across frames so the
/// hot loop performs no heap allocations (PHASE2-8).
pub(crate) fn get_power_spectrum(
    fs: f64,
    f0: f64,
    forward: &mut ForwardRealFFT,
    dc_scratch: &mut [f64],
) {
    let fft_size = forward.fft_size;
    let half_window_length = matlab_round(1.5 * fs / f0) as usize;
    for i in (2 * half_window_length + 1)..fft_size {
        forward.waveform[i] = 0.0;
    }
    forward.forward();
    let half = fft_size / 2;
    for i in 0..=half {
        forward.waveform[i] = forward.spectrum[i].re * forward.spectrum[i].re
            + forward.spectrum[i].im * forward.spectrum[i].im;
    }
    let power_len = half + 1;
    forward.temp[..power_len].copy_from_slice(&forward.waveform[..power_len]);
    dc_correction_into(
        &forward.temp,
        f0,
        fs as i32,
        fft_size,
        &mut forward.waveform[..power_len],
        dc_scratch,
    );
}

/// C++: `SmoothingWithRecovery` (cheaptrick.cpp:22-57).
///
/// Performs spectral smoothing and recovery on the cepstrum domain. The input
/// power spectrum is read from `forward.waveform[0..=fft_size/2]` (which must
/// be strictly positive; `add_infinitesimal_noise` guarantees this upstream;
/// `log(0) = -inf` otherwise). The cepstral lifters are written into
/// `smoothing_lifter` and `compensation_lifter` (each length at least
/// `fft_size`), which the caller may allocate once and reuse across frames;
/// the resulting spectral envelope is written to
/// `spectral_envelope[0..=fft_size/2]`.
///
/// Steps (matching the C++):
/// 1. `smoothing_lifter[0] = 1.0`; `compensation_lifter[0] = (1-2*q1)+2*q1`.
/// 2. Lifter loop `i = 1..=fft_size/2` with `quefrency = i/fs`:
///    `smoothing = sin(kPi*f0*quefrency)/(kPi*f0*quefrency)`,
///    `compensation = (1-2*q1) + 2*q1*cos(2*kPi*quefrency*f0)`.
/// 3. `log` of the lower half of `forward.waveform` (in place), then even
///    mirror into the full-length buffer.
/// 4. Forward FFT (r2c) using the cached plan.
/// 5. Multiply the spectrum by the lifters, `÷ fft_size` (compensates the
///    unnormalized inverse), zero imaginary part, write to `inverse.spectrum`.
/// 6. Inverse FFT (c2r, unnormalized) using the cached plan.
/// 7. `exp` of the lower half of `inverse.waveform` into `spectral_envelope`.
///
/// # Panics
///
/// In debug builds, panics if `smoothing_lifter` or `compensation_lifter` are
/// shorter than `fft_size`, or if `spectral_envelope` is shorter than
/// `fft_size / 2 + 1`.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub(crate) fn smoothing_with_recovery(
    f0: f64,
    fs: f64,
    q1: f64,
    forward: &mut ForwardRealFFT,
    inverse: &mut InverseRealFFT,
    smoothing_lifter: &mut [f64],
    compensation_lifter: &mut [f64],
    spectral_envelope: &mut [f64],
) {
    let fft_size = forward.fft_size;
    let half = fft_size / 2;
    debug_assert!(smoothing_lifter.len() >= fft_size);
    debug_assert!(compensation_lifter.len() >= fft_size);
    debug_assert!(spectral_envelope.len() > half);

    smoothing_lifter[0] = 1.0;
    compensation_lifter[0] = (1.0 - 2.0 * q1) + 2.0 * q1;
    for i in 1..=half {
        let quefrency = i as f64 / fs;
        let denom = K_PI * f0 * quefrency;
        smoothing_lifter[i] = denom.sin() / denom;
        compensation_lifter[i] = (1.0 - 2.0 * q1) + 2.0 * q1 * (2.0 * K_PI * quefrency * f0).cos();
    }

    for i in 0..=half {
        forward.waveform[i] = forward.waveform[i].ln();
    }
    for i in 1..half {
        forward.waveform[fft_size - i] = forward.waveform[i];
    }

    forward.forward();

    let inv_n = 1.0 / fft_size as f64;
    for i in 0..=half {
        inverse.spectrum[i] = Complex::new(
            forward.spectrum[i].re * smoothing_lifter[i] * compensation_lifter[i] * inv_n,
            0.0,
        );
    }

    inverse.inverse();

    for i in 0..=half {
        spectral_envelope[i] = inverse.waveform[i].exp();
    }
}

/// Error returned by [`cheaptrick`] when the input is invalid.
#[derive(Debug, Clone, PartialEq)]
pub enum CheapTrickError {
    /// The input signal slice `x` is empty.
    EmptyInput,
    /// The sample rate `fs` is not strictly positive (zero or negative).
    NonPositiveSampleRate { fs: f64 },
    /// `f0` and `temporal_positions` have different lengths.
    MismatchedLengths {
        f0: usize,
        temporal_positions: usize,
    },
    /// The input signal contains NaN or infinite samples.
    NonFiniteInput,
    /// The FFT size is invalid (<=0, not power of two, or not even).
    InvalidFftSize { fft_size: i32 },
    /// `f0` contains NaN or infinite values.
    NonFiniteF0,
}

impl std::fmt::Display for CheapTrickError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CheapTrickError::EmptyInput => write!(f, "input signal is empty"),
            CheapTrickError::NonPositiveSampleRate { fs } => {
                write!(f, "sample rate must be positive, got {fs}")
            }
            CheapTrickError::MismatchedLengths {
                f0,
                temporal_positions,
            } => write!(
                f,
                "f0 length {f0} != temporal_positions length {temporal_positions}"
            ),
            CheapTrickError::NonFiniteInput => write!(f, "input contains non-finite samples"),
            CheapTrickError::InvalidFftSize { fft_size } => {
                write!(f, "fft_size must be positive power of two, got {fft_size}")
            }
            CheapTrickError::NonFiniteF0 => write!(f, "f0 contains non-finite values"),
        }
    }
}

impl std::error::Error for CheapTrickError {}

/// C++: `CheapTrickGeneralBody` (cheaptrick.cpp:159-187).
///
/// Calculates a spectral envelope at a single temporal position. The 5-step
/// sequence matches the C++: windowing, power spectrum + DC correction,
/// linear smoothing, infinitesimal noise, cepstral smoothing with recovery.
/// All buffers are pre-allocated by the caller; no allocations occur in this
/// function.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cheaptrick_general_body(
    x: &[f64],
    fs: f64,
    current_f0: f64,
    fft_size: i32,
    current_position: f64,
    q1: f64,
    forward: &mut ForwardRealFFT,
    inverse: &mut InverseRealFFT,
    spectral_envelope: &mut [f64],
    safe_index: &mut [i32],
    window: &mut [f64],
    smoothing_lifter: &mut [f64],
    compensation_lifter: &mut [f64],
    randn_state: &mut RandnState,
    dc_scratch: &mut [f64],
    smooth_scratch: &mut [f64],
) {
    let half = (fft_size / 2) as usize;
    let power_len = half + 1;

    get_windowed_waveform(
        x,
        fs,
        current_f0,
        current_position,
        &mut forward.waveform,
        safe_index,
        window,
        randn_state,
    );

    get_power_spectrum(fs, current_f0, forward, dc_scratch);

    forward.temp[..power_len].copy_from_slice(&forward.waveform[..power_len]);
    linear_smoothing_into(
        &forward.temp,
        current_f0 * 2.0 / 3.0,
        fs as i32,
        fft_size as usize,
        &mut forward.waveform[..power_len],
        smooth_scratch,
    );

    forward.temp[..power_len].copy_from_slice(&forward.waveform[..power_len]);
    add_infinitesimal_noise(
        &forward.temp,
        fft_size,
        &mut forward.waveform[..power_len],
        randn_state,
    );

    smoothing_with_recovery(
        current_f0,
        fs,
        q1,
        forward,
        inverse,
        smoothing_lifter,
        compensation_lifter,
        spectral_envelope,
    );
}

/// C++: `CheapTrick` (cheaptrick.cpp:200-229).
///
/// Estimates the spectral envelope (spectrogram) for each frame of the F0
/// contour. Returns an owned `f0_length x (fft_size/2 + 1)` spectrogram, where
/// `fft_size` is `option.fft_size`.
///
/// Validates the input before running the pipeline. Frames with
/// `f0[i] <= f0_floor` are analyzed as unvoiced using `kDefaultF0 = 500.0`.
/// The PRNG is reseeded exactly once before the frame loop; the same seed
/// produces the same output for all frames.
///
/// # Example
///
/// The natural caller is DIO: `dio()` returns the F0 contour and the temporal
/// grid it is aligned to, and both are consumed here directly.
///
/// ```
/// use world_rs_core::cheaptrick::{cheaptrick, initialize_cheaptrick_option};
/// use world_rs_core::dio::{dio, initialize_dio_option};
///
/// let fs = 16000.0;
/// let n = 16000; // 1 second at 16 kHz
/// let x: Vec<f64> = (0..n)
///     .map(|i| 0.5 * (2.0 * std::f64::consts::PI * 200.0 * i as f64 / fs).sin())
///     .collect();
///
/// // DIO produces the F0 contour and the 5 ms temporal grid it is aligned to.
/// let dio_result = dio(&x, fs, &initialize_dio_option()).expect("valid input");
///
/// // CheapTrick consumes DIO's `f0` and `temporal_positions` directly; both
/// // have length `f0_length` and `temporal_positions[i]` is in seconds.
/// let option = initialize_cheaptrick_option(fs);
/// let sp = cheaptrick(
///     &x,
///     fs,
///     &dio_result.temporal_positions,
///     &dio_result.f0,
///     &option,
/// )
/// .expect("valid input");
///
/// assert_eq!(sp.len(), dio_result.f0_length);
/// assert_eq!(sp[0].len(), (option.fft_size / 2) as usize + 1);
/// for row in &sp {
///     for &v in row {
///         assert!(v.is_finite() && v > 0.0);
///     }
/// }
/// ```
pub fn cheaptrick(
    x: &[f64],
    fs: f64,
    temporal_positions: &[f64],
    f0: &[f64],
    option: &CheapTrickOption,
) -> Result<Vec<Vec<f64>>, CheapTrickError> {
    if x.is_empty() {
        return Err(CheapTrickError::EmptyInput);
    }
    if !fs.is_finite() || fs <= 0.0 {
        return Err(CheapTrickError::NonPositiveSampleRate { fs });
    }
    if (fs as i32) == 0 {
        return Err(CheapTrickError::NonPositiveSampleRate { fs });
    }
    if f0.len() != temporal_positions.len() {
        return Err(CheapTrickError::MismatchedLengths {
            f0: f0.len(),
            temporal_positions: temporal_positions.len(),
        });
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(CheapTrickError::NonFiniteInput);
    }
    if option.fft_size <= 0 {
        return Err(CheapTrickError::InvalidFftSize {
            fft_size: option.fft_size,
        });
    }
    let fft_size_i32 = option.fft_size;
    if fft_size_i32 <= 3 {
        return Err(CheapTrickError::InvalidFftSize {
            fft_size: option.fft_size,
        });
    }
    if fft_size_i32 & (fft_size_i32 - 1) != 0 || fft_size_i32 % 2 != 0 {
        return Err(CheapTrickError::InvalidFftSize {
            fft_size: option.fft_size,
        });
    }
    // Allocation budget: FFT scratch, lifters, and windows all scale with
    // `fft_size`, and `ForwardRealFFT::new` would attempt multi-GB plans for
    // degenerate sizes (e.g. `i32::MAX`). 2^20 (≈8MB per f64 buffer) is
    // 256× the largest realistic size (8192).
    if fft_size_i32 > (1 << 20) {
        return Err(CheapTrickError::InvalidFftSize {
            fft_size: option.fft_size,
        });
    }
    if f0.iter().any(|v| !v.is_finite()) {
        return Err(CheapTrickError::NonFiniteF0);
    }

    let fft_size = option.fft_size as usize;
    let half = fft_size / 2;
    let f0_length = f0.len();

    let mut randn_state = RandnState::default();
    randn_reseed(&mut randn_state);

    let f0_floor = get_f0_floor_for_cheaptrick(fs, option.fft_size);

    let mut forward = ForwardRealFFT::new(fft_size);
    let mut inverse = InverseRealFFT::new(fft_size);
    let mut spectral_envelope = vec![0.0; half + 1];
    let mut smoothing_lifter = vec![0.0; fft_size];
    let mut compensation_lifter = vec![0.0; fft_size];
    let mut safe_index = vec![0i32; fft_size];
    let mut window = vec![0.0f64; fft_size];
    // PHASE2-8: pre-allocate the DC-correction and linear-smoothing working
    // buffers once (sized for the Nyquist worst case) so the per-frame loop
    // performs no heap allocations on the WASM target.
    let power_len = half + 1;
    let mut dc_scratch = vec![0.0f64; dc_correction_scratch_capacity(fft_size, power_len)];
    let mut smooth_scratch = vec![0.0f64; linear_smoothing_scratch_capacity(fft_size)];

    let mut spectrogram = vec![vec![0.0; half + 1]; f0_length];

    for i in 0..f0_length {
        let current_f0 = if f0[i] <= f0_floor {
            K_DEFAULT_F0
        } else {
            f0[i]
        };
        cheaptrick_general_body(
            x,
            fs,
            current_f0,
            option.fft_size,
            temporal_positions[i],
            option.q1,
            &mut forward,
            &mut inverse,
            &mut spectral_envelope,
            &mut safe_index,
            &mut window,
            &mut smoothing_lifter,
            &mut compensation_lifter,
            &mut randn_state,
            &mut dc_scratch,
            &mut smooth_scratch,
        );
        spectrogram[i].copy_from_slice(&spectral_envelope);
    }

    Ok(spectrogram)
}

const CHEAPTRICK_MULTIPLIER: f64 = 3.0;
const CHEAPTRICK_OFFSET: f64 = 1.0;
const CHEAPTRICK_FFT_EXP_OFFSET: i32 = 1;
const CHEAPTRICK_FFT_DENOM_OFFSET: f64 = 3.0;

#[cfg(test)]
// Reference values are captured from the C++ port at full f64 precision; the
// excess digits are intentional (documenting the exact reference), not noise.
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    #[test]
    fn test_initialize_cheaptrick_option_matches_cpp_reference() {
        let option = initialize_cheaptrick_option(16000.0);
        assert_eq!(option.q1, -0.15);
        assert_eq!(option.f0_floor, K_FLOOR_F0);
        assert_eq!(
            option.fft_size,
            get_fft_size_for_cheaptrick(16000.0, &option)
        );
        assert_eq!(option.fft_size, 1024);
    }

    #[test]
    fn test_get_fft_size_for_cheaptrick() {
        let option = initialize_cheaptrick_option(16000.0);
        assert_eq!(get_fft_size_for_cheaptrick(16000.0, &option), 1024);
        assert_eq!(get_fft_size_for_cheaptrick(48000.0, &option), 2048);
        let size = get_fft_size_for_cheaptrick(16000.0, &option);
        assert!(size > 0 && (size & (size - 1)) == 0);
    }

    // Edge cases (very low / very high fs) against the C++ reference
    // `GetFFTSizeForCheapTrick` (cheaptrick.cpp:191-194). Reference values
    // computed from the C++ formula `2^(1 + floor(log(3*fs/f0_floor+1)/kLog2))`.
    #[test]
    fn test_get_fft_size_for_cheaptrick_edge_cases() {
        let option = initialize_cheaptrick_option(16000.0);
        // (fs, expected fft_size)
        let cases: &[(f64, i32)] = &[
            (100.0, 8),
            (400.0, 32),
            (800.0, 64),
            (4000.0, 256),
            (8000.0, 512),
            (11025.0, 512),
            (16000.0, 1024),
            (22050.0, 1024),
            (44100.0, 2048),
            (48000.0, 2048),
            (88200.0, 4096),
            (96000.0, 4096),
            (192000.0, 8192),
        ];
        for &(fs, expected) in cases {
            let size = get_fft_size_for_cheaptrick(fs, &option);
            assert_eq!(
                size, expected,
                "fs={}: got {}, expected {}",
                fs, size, expected
            );
            // Always a power of two.
            assert!(
                size > 0 && (size & (size - 1)) == 0,
                "fs={}: not a power of two",
                fs
            );
        }
    }

    #[test]
    fn test_get_f0_floor_for_cheaptrick() {
        // C++: 3.0 * fs / (fft_size - 3.0)
        let f0 = get_f0_floor_for_cheaptrick(16000.0, 1024);
        assert!((f0 - 3.0 * 16000.0 / (1024.0 - 3.0)).abs() < 1e-9);
        assert!(f0 > 0.0);
    }

    // `get_f0_floor_for_cheaptrick` round-trips the effective floor for the
    // fft_size chosen by `get_fft_size_for_cheaptrick`, matching C++ within 1e-9.
    #[test]
    fn test_get_f0_floor_for_cheaptrick_matches_cpp_reference() {
        let option = initialize_cheaptrick_option(16000.0);
        for &fs in &[8000.0, 16000.0, 44100.0, 48000.0, 96000.0] {
            let fft_size = get_fft_size_for_cheaptrick(fs, &option);
            let f0 = get_f0_floor_for_cheaptrick(fs, fft_size);
            let expected = 3.0 * fs / (fft_size as f64 - 3.0);
            assert!(
                (f0 - expected).abs() < 1e-9,
                "fs={}: got {}, expected {}",
                fs,
                f0,
                expected
            );
            assert!(f0 > 0.0);
        }
    }

    use crate::matlab::randn_reseed;

    // Zero spectrum: each bin becomes |randn| * kEps. The expected values are
    // derived from the bit-exact C++ xorshift sequence (see
    // `matlab::tests::test_randn_bit_exact_vs_cpp`), so this test is
    // independent of the `randn` implementation.
    #[test]
    fn test_add_infinitesimal_noise_zero_spectrum_matches_cpp_reference() {
        let fft_size: i32 = 8;
        let n = (fft_size / 2) as usize + 1;
        let input = vec![0.0; n];
        let mut output = vec![0.0; n];
        let mut state = RandnState::default();
        randn_reseed(&mut state);
        add_infinitesimal_noise(&input, fft_size, &mut output, &mut state);

        let randn_seq: [f64; 5] = [
            -1.3276404961943626,
            -0.62285530939698219,
            -1.6091805659234524,
            1.1797650642693043,
            -0.25188251212239265,
        ];
        for i in 0..n {
            let expected = randn_seq[i].abs() * K_EPS;
            assert_eq!(
                output[i], expected,
                "bin {i}: {} != {}",
                output[i], expected
            );
        }
    }

    // A non-trivial spectrum keeps its shape: every bin is nudged up by a
    // tiny, non-negative perturbation bounded by |randn| * kEps (< 6 * kEps).
    #[test]
    fn test_add_infinitesimal_noise_preserves_shape_with_small_perturbation() {
        let fft_size: i32 = 16;
        let n = (fft_size / 2) as usize + 1;
        let input: Vec<f64> = (0..n).map(|i| (n - i) as f64 * 100.0).collect();
        let mut output = vec![0.0; n];
        let mut state = RandnState::default();
        randn_reseed(&mut state);
        add_infinitesimal_noise(&input, fft_size, &mut output, &mut state);

        for i in 0..n {
            let delta = output[i] - input[i];
            assert!(
                (0.0..6.0 * K_EPS).contains(&delta),
                "bin {i}: delta={}",
                delta
            );
        }
    }

    // Determinism: reseeding the state reproduces the exact same perturbation
    // (the PRNG is the sole source of non-determinism).
    #[test]
    fn test_add_infinitesimal_noise_deterministic() {
        let fft_size: i32 = 32;
        let n = (fft_size / 2) as usize + 1;
        let input: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5 + 1.0).collect();

        let mut out_a = vec![0.0; n];
        let mut out_b = vec![0.0; n];
        let mut s_a = RandnState::default();
        randn_reseed(&mut s_a);
        add_infinitesimal_noise(&input, fft_size, &mut out_a, &mut s_a);
        let mut s_b = RandnState::default();
        randn_reseed(&mut s_b);
        add_infinitesimal_noise(&input, fft_size, &mut out_b, &mut s_b);

        assert_eq!(out_a, out_b);
    }

    // The perturbation is strictly non-negative (fabs in the C++ reference), so
    // a spectrum that contains a zero bin becomes a small positive value — the
    // safeguard that keeps the subsequent log transform finite.
    #[test]
    fn test_add_infinitesimal_noise_lifts_zero_bins() {
        let fft_size: i32 = 8;
        let n = (fft_size / 2) as usize + 1;
        let input = vec![0.0, 5.0, 0.0, 7.0, 0.0];
        let mut output = vec![0.0; n];
        let mut state = RandnState::default();
        randn_reseed(&mut state);
        add_infinitesimal_noise(&input, fft_size, &mut output, &mut state);
        for (i, &v) in output.iter().enumerate() {
            assert!(v > 0.0, "bin {i}: {v}");
        }
    }

    // Test signal matching the C++ reference driver (phase24_cpp_ref.cpp).
    fn test_signal(fs: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let t = i as f64;
                0.5 * (2.0 * K_PI * 200.0 * t / fs).sin()
                    + 0.1 * (2.0 * K_PI * 1000.0 * t / fs).sin()
            })
            .collect()
    }

    // C++ reference: matlab_round(1.5 * fs / f0) for fs=16000.
    #[test]
    fn test_half_window_length_matches_cpp_reference() {
        let fs = 16000.0;
        let cases: &[(f64, usize)] = &[
            (71.0, 338),
            (100.0, 240),
            (200.0, 120),
            (500.0, 48),
            (1000.0, 24),
            (2000.0, 12),
            (5000.0, 5),
        ];
        for &(f0, expected) in cases {
            let hwl = matlab_round(1.5 * fs / f0) as usize;
            assert_eq!(hwl, expected, "f0={f0}");
        }
    }

    // Safe index at the start of the signal (position=0.0, f0=200, fs=16000,
    // x_length=1600). The first 120 entries clamp to 0; the rest are 0..=120.
    #[test]
    fn test_set_parameters_safe_index_start_matches_cpp_reference() {
        let fs = 1600.0;
        let x_length = 1600;
        let hwl = 120;
        let n = 2 * hwl + 1;
        let mut safe_index = vec![0i32; n];
        let mut window = vec![0.0f64; n];
        set_parameters_for_get_windowed_waveform(
            hwl,
            x_length,
            0.0,
            fs,
            200.0,
            &mut safe_index,
            &mut window,
        );
        for (i, &sv) in safe_index.iter().enumerate().take(n) {
            let expected = if i < hwl { 0 } else { (i - hwl) as i32 };
            assert_eq!(sv, expected, "i={i}");
        }
    }

    // Safe index at the end of the signal (position=(x_length-1)/fs, f0=200).
    // The last 120 entries clamp to x_length-1; the rest are 1479..=1599.
    #[test]
    fn test_set_parameters_safe_index_end_matches_cpp_reference() {
        let fs = 1600.0;
        let x_length = 1600;
        let hwl = 120;
        let n = 2 * hwl + 1;
        let mut safe_index = vec![0i32; n];
        let mut window = vec![0.0f64; n];
        set_parameters_for_get_windowed_waveform(
            hwl,
            x_length,
            (x_length - 1) as f64 / fs,
            fs,
            200.0,
            &mut safe_index,
            &mut window,
        );
        for (i, &sv) in safe_index.iter().enumerate().take(n) {
            let raw = (x_length - 1) as i32 - hwl as i32 + i as i32;
            let expected = raw.clamp(0, (x_length - 1) as i32);
            assert_eq!(sv, expected, "i={i}");
        }
    }

    // Normalized window for f0=200, fs=16000: 241 values, endpoints are 0,
    // center is ~0.1054, RMS is 1.0.
    #[test]
    fn test_set_parameters_window_200_matches_cpp_reference() {
        let fs = 16000.0;
        let x_length = 1600;
        let hwl = 120;
        let n = 2 * hwl + 1;
        let mut safe_index = vec![0i32; n];
        let mut window = vec![0.0f64; n];
        set_parameters_for_get_windowed_waveform(
            hwl,
            x_length,
            0.05,
            fs,
            200.0,
            &mut safe_index,
            &mut window,
        );

        let expected: &[f64] = &[
            0.0,
            0.000018060559574882178,
            0.00007222986049413321,
            0.00016247077782473497,
            0.00028872146494957182,
            0.00045089539595404476,
            0.00064888142492663535,
            0.00088254386213268074,
            0.0011517225670094364,
            0.001456233057918358,
            0.0017958666385796058,
            0.0021703905411020671,
            0.0025795480855108585,
            0.0030230588556628924,
            0.0035006188914301771,
            0.0040119008970188512,
            0.0045565544652814106,
            0.0051342063178681811,
            0.0057444605610537883,
            0.0063868989570628155,
            0.007061081210709157,
            0.0077665452711523086,
            0.0085028076485639745,
            0.0092693637454877766,
            0.01006568820266527,
            0.010891235259090746,
            0.011745439126048736,
            0.012627714374877277,
            0.01353745633819144,
            0.014474041524292214,
            0.015436828044476502,
            0.016425156052955622,
            0.017438348199080675,
            0.018475710091564755,
            0.019536530774384164,
            0.02062008321403206,
            0.021725624797790961,
            0.022852397842682368,
            0.023999630114744951,
            0.025166535358284881,
            0.026352313834736494,
            0.027556152870763011,
            0.028777227415222213,
            0.030014700604615353,
            0.031267724336631615,
            0.032535439851394996,
            0.033816978320015477,
            0.035111461440040938,
            0.036418002037401774,
            0.037735704674435717,
            0.03906366626357613,
            0.040400976686282869,
            0.041746719416792394,
            0.043099972150258632,
            0.044459807434854462,
            0.045825293307401128,
            0.047195493932089154,
            0.048569470241853409,
            0.049946280581962821,
            0.05132498135538352,
            0.052704627669473002,
            0.054084273983562491,
            0.05546297475698321,
            0.056839785097092622,
            0.05821376140685687,
            0.059583962031544903,
            0.060949447904091583,
            0.062309283188687399,
            0.06366253592215361,
            0.065008278652663162,
            0.066345589075369915,
            0.067673550664510293,
            0.06899125330154425,
            0.070297793898905087,
            0.071592277018930534,
            0.072873815487551022,
            0.074141531002314395,
            0.07539455473433064,
            0.076632027923723797,
            0.077853102468183014,
            0.079056941504209513,
            0.08024271998066114,
            0.081409625224201074,
            0.082556857496263639,
            0.08368363054115506,
            0.084789172124913964,
            0.085872724564561839,
            0.086933545247381255,
            0.087970907139865342,
            0.088984099285990381,
            0.089972427294469512,
            0.090935213814653804,
            0.091871799000754567,
            0.092781540964068751,
            0.09366381621289728,
            0.094518020079855283,
            0.095343567136280755,
            0.096139891593458218,
            0.09690644769038205,
            0.097642710067793709,
            0.098348174128236862,
            0.099022356381883198,
            0.099664794777892224,
            0.10027504902107784,
            0.1008527008736646,
            0.10139735444192716,
            0.10190863644751583,
            0.10238619648328313,
            0.10282970725343517,
            0.10323886479784394,
            0.1036133887003664,
            0.10395302228102765,
            0.10425753277193658,
            0.10452671147681335,
            0.10476037391401939,
            0.10495835994299198,
            0.10512053387399645,
            0.10524678456112128,
            0.10533702547845188,
            0.10539119477937113,
            0.10540925533894602,
            0.10539119477937113,
            0.10533702547845188,
            0.10524678456112128,
            0.10512053387399645,
            0.10495835994299198,
            0.10476037391401939,
            0.10452671147681335,
            0.10425753277193658,
            0.10395302228102765,
            0.1036133887003664,
            0.10323886479784394,
            0.10282970725343517,
            0.10238619648328313,
            0.10190863644751583,
            0.10139735444192716,
            0.1008527008736646,
            0.10027504902107784,
            0.099664794777892224,
            0.099022356381883198,
            0.098348174128236862,
            0.097642710067793709,
            0.09690644769038205,
            0.096139891593458218,
            0.095343567136280755,
            0.094518020079855283,
            0.09366381621289728,
            0.092781540964068751,
            0.091871799000754567,
            0.090935213814653804,
            0.089972427294469512,
            0.088984099285990381,
            0.087970907139865342,
            0.086933545247381255,
            0.085872724564561839,
            0.084789172124913964,
            0.08368363054115506,
            0.082556857496263639,
            0.081409625224201074,
            0.08024271998066114,
            0.079056941504209513,
            0.077853102468183014,
            0.076632027923723797,
            0.07539455473433064,
            0.074141531002314395,
            0.072873815487551022,
            0.071592277018930534,
            0.070297793898905087,
            0.06899125330154425,
            0.067673550664510293,
            0.066345589075369915,
            0.065008278652663162,
            0.06366253592215361,
            0.062309283188687399,
            0.060949447904091583,
            0.059583962031544903,
            0.05821376140685687,
            0.056839785097092622,
            0.05546297475698321,
            0.054084273983562491,
            0.052704627669473002,
            0.05132498135538352,
            0.049946280581962821,
            0.048569470241853409,
            0.047195493932089154,
            0.045825293307401128,
            0.044459807434854462,
            0.043099972150258632,
            0.041746719416792394,
            0.040400976686282869,
            0.03906366626357613,
            0.037735704674435717,
            0.036418002037401774,
            0.035111461440040938,
            0.033816978320015477,
            0.032535439851394996,
            0.031267724336631615,
            0.030014700604615353,
            0.028777227415222213,
            0.027556152870763011,
            0.026352313834736494,
            0.025166535358284881,
            0.023999630114744951,
            0.022852397842682368,
            0.021725624797790961,
            0.02062008321403206,
            0.019536530774384164,
            0.018475710091564755,
            0.017438348199080675,
            0.016425156052955622,
            0.015436828044476502,
            0.014474041524292214,
            0.01353745633819144,
            0.012627714374877277,
            0.011745439126048736,
            0.010891235259090746,
            0.01006568820266527,
            0.0092693637454877766,
            0.0085028076485639745,
            0.0077665452711523086,
            0.007061081210709157,
            0.0063868989570628155,
            0.0057444605610537883,
            0.0051342063178681811,
            0.0045565544652814106,
            0.0040119008970188512,
            0.0035006188914301771,
            0.0030230588556628924,
            0.0025795480855108585,
            0.0021703905411020671,
            0.0017958666385796058,
            0.001456233057918358,
            0.0011517225670094364,
            0.00088254386213268074,
            0.00064888142492663535,
            0.00045089539595404476,
            0.00028872146494957182,
            0.00016247077782473497,
            0.00007222986049413321,
            0.000018060559574882178,
            0.0,
        ];
        assert_eq!(window.len(), n);
        for i in 0..n {
            assert!(
                (window[i] - expected[i]).abs() < 1e-15,
                "window[{i}]: {} != {}",
                window[i],
                expected[i]
            );
        }
        let rms = (0..n).map(|i| window[i] * window[i]).sum::<f64>().sqrt();
        assert!((rms - 1.0).abs() < 1e-12, "RMS={rms}, expected ~1.0");
    }

    // Windowed waveform for a voiced frame (f0=200, position=0.05, fs=16000,
    // x_length=1600). PRNG is reseeded before the call.
    #[test]
    fn test_get_windowed_waveform_voiced_matches_cpp_reference() {
        let fs = 16000.0;
        let x_length = 1600;
        let x = test_signal(fs, x_length);
        let hwl = 120;
        let n = 2 * hwl + 1;
        let mut waveform = vec![0.0f64; n];
        let mut safe_index = vec![0i32; n];
        let mut window = vec![0.0f64; n];
        let mut state = RandnState::default();
        randn_reseed(&mut state);
        get_windowed_waveform(
            &x,
            fs,
            200.0,
            0.05,
            &mut waveform,
            &mut safe_index,
            &mut window,
            &mut state,
        );

        let expected: &[f64] = &[
            -1.3276404961943626e-12,
            -1.3996559019908326e-06,
            -1.0757043818255577e-05,
            -3.3974366369100839e-05,
            -7.348206640178053e-05,
            -0.00012793240232671511,
            -0.00019317584686287832,
            -0.00026433744140013981,
            -0.00033848277104613506,
            -0.00041714623208610106,
            -0.00050794779022757394,
            -0.00062467101882297425,
            -0.0007854943117054842,
            -0.0010094964795242677,
            -0.0013120060026791556,
            -0.0016997277616752895,
            -0.0021667704078835283,
            -0.0026926514644822976,
            -0.0032430630611301118,
            -0.00377367766195736,
            -0.0042366487258440463,
            -0.0045888370220776332,
            -0.0048003012821285574,
            -0.0048613484377646862,
            -0.0047865191782540777,
            -0.0046143051404066654,
            -0.0044021034813837134,
            -0.0042167995442608432,
            -0.0041222704845017468,
            -0.0041658466898937619,
            -0.0043661943160140868,
            -0.0047050792634522427,
            -0.0051250019476791718,
            -0.0055338008165892868,
            -0.0058161410254076798,
            -0.0058505293945592075,
            -0.0055293561164775256,
            -0.0047786794292100808,
            -0.0035742147706468567,
            -0.0019503534178295515,
            1.3350886312034674e-12,
            0.0021355437344916374,
            0.0042857323521189733,
            0.00627639310918293,
            0.0079579015324806705,
            0.0092312695952862407,
            0.010067514912178912,
            0.010516501562679083,
            0.010702982258138876,
            0.010809606978511976,
            0.011048873326338957,
            0.011628008296539205,
            0.012712230793922063,
            0.014392465458604412,
            0.016663206113358166,
            0.019414867226679535,
            0.02244279102163603,
            0.025472419114896023,
            0.028197414863684499,
            0.030325191762346174,
            0.031622776601538241,
            0.031955510484666994,
            0.031311891306564678,
            0.029809813060275848,
            0.027682288560970551,
            0.025244021982189366,
            0.022843401071730614,
            0.02080707158778972,
            0.01938578314189603,
            0.018710359640986617,
            0.018765366373847538,
            0.019385473035933805,
            0.02027602061311155,
            0.021055428311264616,
            0.021313445267429145,
            0.020676463578535545,
            0.018869649634260112,
            0.015765803228742825,
            0.011412647807237565,
            0.0060334512558215874,
            1.3960383297390683e-14,
            -0.0062186415750581665,
            -0.012124165390091259,
            -0.017263516909095881,
            -0.02129819504729704,
            -0.024057203776580459,
            -0.025564819155291726,
            -0.026038129057198246,
            -0.025854000923295862,
            -0.025490000751708236,
            -0.025448045385922338,
            -0.026172521252575957,
            -0.027975743449729844,
            -0.03098273750056682,
            -0.035104503705213716,
            -0.040044584067046536,
            -0.045338560404294487,
            -0.0504208837376805,
            -0.054709004886336586,
            -0.057691865222853793,
            -0.059008904477941715,
            -0.058507024585575246,
            -0.05626624310849785,
            -0.052589580713425774,
            -0.047958309176916403,
            -0.042959161447974147,
            -0.038194601183179862,
            -0.034190040563564852,
            -0.031312519620921443,
            -0.029713696922187997,
            -0.029306291909145583,
            -0.029777933783311256,
            -0.03064052010167332,
            -0.031307592435122038,
            -0.031187784332451281,
            -0.029779800770239163,
            -0.026754069100012032,
            -0.022008221970178276,
            -0.015687623117971835,
            -0.0081675953351895685,
            2.8707880047379284e-14,
            0.0081675953357324294,
            0.015687623118018069,
            0.022008221971290261,
            0.026754069099785917,
            0.029779800769296601,
            0.031187784334145371,
            0.031307592434290675,
            0.030640520101709697,
            0.029777933783378854,
            0.029306291911175095,
            0.029713696922391411,
            0.031312519623960623,
            0.034190040566000036,
            0.038194601184155484,
            0.042959161446707944,
            0.047958309176243351,
            0.05258958071220949,
            0.056266243109837889,
            0.058507024582779649,
            0.059008904476543077,
            0.057691865224704493,
            0.054709004886052855,
            0.050420883738128926,
            0.045338560405990241,
            0.04004458407002006,
            0.035104503703967858,
            0.030982737498882865,
            0.027975743446864536,
            0.026172521252039713,
            0.025448045383036137,
            0.025490000752660072,
            0.025854000924963153,
            0.026038129056179117,
            0.025564819156122309,
            0.0240572037800173,
            0.021298195048558205,
            0.017263516909098994,
            0.012124165389776193,
            0.006218641574289154,
            -1.1477879804563672e-12,
            -0.0060334512563045293,
            -0.011412647806966318,
            -0.015765803229413858,
            -0.018869649635156728,
            -0.020676463577973953,
            -0.02131344526807245,
            -0.021055428312177091,
            -0.020276020613712108,
            -0.019385473036144251,
            -0.018765366373242567,
            -0.018710359640908419,
            -0.019385783141431145,
            -0.020807071586518004,
            -0.022843401072882262,
            -0.025244021981258021,
            -0.027682288561280823,
            -0.029809813060735421,
            -0.031311891307475075,
            -0.031955510485988825,
            -0.031622776601186287,
            -0.030325191762512235,
            -0.028197414864160435,
            -0.025472419114493553,
            -0.022442791022872832,
            -0.019414867224674046,
            -0.016663206111903524,
            -0.014392465455452069,
            -0.012712230792301176,
            -0.011628008296789654,
            -0.011048873324810473,
            -0.010809606977562844,
            -0.010702982257274953,
            -0.010516501560730727,
            -0.010067514912469605,
            -0.009231269592704679,
            -0.007957901532625539,
            -0.0062763931074352524,
            -0.0042857323522679782,
            -0.0021355437339385095,
            -1.8072133627428468e-12,
            0.001950353414872201,
            0.0035742147684539312,
            0.0047786794274086424,
            0.0055293561171283435,
            0.0058505293930983908,
            0.0058161410240201552,
            0.0055338008157197038,
            0.0051250019467546284,
            0.0047050792609677709,
            0.0043661943163359673,
            0.0041658466876058741,
            0.0041222704845240033,
            0.0042167995455951032,
            0.0044021034795111583,
            0.0046143051420516135,
            0.0047865191774161004,
            0.0048613484365268309,
            0.0048003012831079207,
            0.0045888370235363666,
            0.0042366487276641746,
            0.0037736776612059641,
            0.0032430630609311599,
            0.0026926514636279883,
            0.0021667704086680535,
            0.0016997277627938004,
            0.0013120060003368409,
            0.0010094964781546406,
            0.00078549431132856685,
            0.00062467101775126751,
            0.00050794779057827507,
            0.00041714623240275629,
            0.00033848277097355922,
            0.00026433744232206435,
            0.0001931758462747783,
            0.0001279324023142821,
            7.3482068424739574e-05,
            3.3974366842275927e-05,
            1.0757041505174657e-05,
            1.3996556519288465e-06,
            -5.4568417370319369e-13,
        ];
        for i in 0..n {
            assert!(
                (waveform[i] - expected[i]).abs() < 1e-15,
                "waveform[{i}]: {} != {}",
                waveform[i],
                expected[i]
            );
        }
    }

    // Windowed waveform at the start of the signal (position=0.0, f0=200).
    #[test]
    fn test_get_windowed_waveform_edge_start_matches_cpp_reference() {
        let fs = 16000.0;
        let x_length = 1600;
        let x = test_signal(fs, x_length);
        let hwl = 120;
        let n = 2 * hwl + 1;
        let mut waveform = vec![0.0f64; n];
        let mut safe_index = vec![0i32; n];
        let mut window = vec![0.0f64; n];
        let mut state = RandnState::default();
        randn_reseed(&mut state);
        get_windowed_waveform(
            &x,
            fs,
            200.0,
            0.0,
            &mut waveform,
            &mut safe_index,
            &mut window,
            &mut state,
        );

        let expected: &[f64] = &[
            -1.3276404961943626e-12,
            -9.9548239245864813e-07,
            -3.9812464341140189e-06,
            -8.9552416933701735e-06,
            -1.5914067236285076e-05,
            -2.485294861208541e-05,
            -3.5765759543360465e-05,
            -4.8645021762284269e-05,
            -6.3481911326074877e-05,
            -8.026625384777366e-05,
            -9.8986549813745615e-05,
            -0.00011962997201583066,
            -0.00014218236683740095,
            -0.00016662828006663808,
            -0.00019295096177661793,
            -0.00022113236199168188,
            -0.00025115317612368085,
            -0.00028299282642715086,
            -0.00031662948971251503,
            -0.00035204011549917486,
            -0.00038920043309762214,
            -0.00042808497563175861,
            -0.00046866709479248184,
            -0.00051091897783903594,
            -0.00055481166384453132,
            -0.00060031507327165526,
            -0.00064739802235871209,
            -0.00069602823725158905,
            -0.00074617239414034018,
            -0.00079779612732232893,
            -0.00085086405178186005,
            -0.00090533980307329906,
            -0.0009611860390467881,
            -0.0010183644907991822,
            -0.0010768359736966675,
            -0.0011365604067959333,
            -0.0011974968608883146,
            -0.0012596035780298844,
            -0.0013228379877017339,
            -0.0013871567529447554,
            -0.0014525157914828512,
            -0.0015188703155524835,
            -0.0015861748451208029,
            -0.0016543832503993776,
            -0.0017234487893730307,
            -0.0017933241259452924,
            -0.0018639613732734145,
            -0.0019353121150545004,
            -0.0020073274561110597,
            -0.0020799580377895877,
            -0.0021531540835035617,
            -0.0022268654323417719,
            -0.0023010415549800968,
            -0.0023756316262356953,
            -0.0024505845218175362,
            -0.0025258488736202215,
            -0.0026013730985502107,
            -0.0026771054339149924,
            -0.0027529939789055192,
            -0.0028289867230719581,
            -0.0029050315857818143,
            -0.0029810764473263305,
            -0.0030570691916632898,
            -0.0031329577373558301,
            -0.0032086900740774106,
            -0.0032842142976216657,
            -0.0033594786477860322,
            -0.0034344315424724701,
            -0.003509021613387489,
            -0.0035831977415099071,
            -0.0036569090875798356,
            -0.0037301051333297479,
            -0.0038027357155933032,
            -0.0038747510570781136,
            -0.0039461017976411871,
            -0.004016739044074181,
            -0.004086614381124557,
            -0.0041556799206282401,
            -0.0042238883264130809,
            -0.0042911928566418548,
            -0.0043575473784403882,
            -0.0044229064199803741,
            -0.004487225184691134,
            -0.0045504595938675695,
            -0.0046125663081928387,
            -0.0046735027630473566,
            -0.0047332271990250478,
            -0.0047916986794572982,
            -0.0048488771315187869,
            -0.0049047233690668168,
            -0.0049591991214294534,
            -0.0050122670463151591,
            -0.0050638907784506734,
            -0.0051140349350187481,
            -0.0051626651520912384,
            -0.0052097480966154398,
            -0.0052552515060146493,
            -0.0052991441937087569,
            -0.0053413960753994176,
            -0.0053819781944331128,
            -0.005420862738593055,
            -0.0054580230571666969,
            -0.005493433681642278,
            -0.0055270703475830546,
            -0.0055589099959862517,
            -0.0055889308100913002,
            -0.0056171122119350802,
            -0.0056434348904651089,
            -0.0056678808041488865,
            -0.0056904332001628446,
            -0.0057110766208348217,
            -0.0057297969185299011,
            -0.0057465812609843476,
            -0.0057614181512286138,
            -0.0057742974117931112,
            -0.0057852102226575476,
            -0.0057941491044843424,
            -0.0058011079284455017,
            -0.0058060819270106525,
            -0.0058090676887360606,
            -0.0058100631712431029,
            0.0023585276462295759,
            0.0098815411915708293,
            0.01620711404289104,
            0.02095991999549766,
            0.023994590545952769,
            0.025413486922315306,
            0.025546174285448916,
            0.02489393884052668,
            0.024048136866064301,
            0.023595215290787189,
            0.024023263722031493,
            0.025644638818761674,
            0.028546605674809857,
            0.032577488971813333,
            0.037370230637804217,
            0.042399399181240929,
            0.047062510366806705,
            0.050772809428426796,
            0.053049001526862982,
            0.053588041738949627,
            0.052309887030437367,
            0.049367608810591275,
            0.045121739543995038,
            0.04008330889837787,
            0.034834835972267435,
            0.029941838553649076,
            0.025868702565271986,
            0.022911852670514179,
            0.021160254207202103,
            0.020488846263635906,
            0.020585277382516438,
            0.021005123792938743,
            0.02124643037691408,
            0.020831591957001933,
            0.01938370101499326,
            0.016685628739441107,
            0.012713057314894376,
            0.0076369402046506845,
            0.0017957351547033583,
            -0.0043575473796025428,
            -0.010324644111858558,
            -0.015636536133925398,
            -0.019921483149245023,
            -0.022956264015963748,
            -0.024693202621930317,
            -0.025259547066417493,
            -0.024930179367389387,
            -0.02407875632850541,
            -0.023115578168950258,
            -0.022422275459865748,
            -0.022293557382066412,
            -0.022894804755916175,
            -0.02424150312973412,
            -0.02620287972054243,
            -0.028528236278046244,
            -0.03089097863430779,
            -0.032942770797641657,
            -0.034368960499061511,
            -0.034936586933275025,
            -0.034527808186822544,
            -0.033154178486498473,
            -0.030950408843846498,
            -0.028149524548859932,
            -0.025044164121118397,
            -0.021940716099158365,
            -0.019113790635515725,
            -0.016768097083508619,
            -0.015013272349088785,
            -0.013854873726904093,
            -0.013202027409459799,
            -0.012889565016028767,
            -0.012710309713753748,
            -0.012451813676790801,
            -0.011931476285397223,
            -0.011024593720020771,
            -0.0096813503230907677,
            -0.0079307763588764933,
            -0.0058719071965819859,
            -0.0036544140496575913,
            -0.0014525157946254253,
            0.00056319666318537377,
            0.0022513767823068393,
            0.0035190758503404829,
            0.0043318592549728297,
            0.0047139689868494847,
            0.0047393050518685989,
            0.0045154363237124087,
            0.0041638159075068418,
            0.0037997394598390643,
            0.0035153302644641764,
            0.0033680505611712636,
            0.0033760980896017398,
            0.0035207713079332108,
            0.0037547054585570848,
            0.0040139900685314785,
            0.0042317075137555706,
            0.0043504294593875812,
            0.0043316341872966487,
            0.0041607520465308349,
            0.0038474482939851043,
            0.0034216375458503688,
            0.0029264335710700636,
            0.0024096586377585502,
            0.0019156172323980731,
            0.0014785954004256766,
            0.0011190550414067173,
            0.00084286819807242357,
            0.00064331194525491615,
            0.00050504104683849567,
            0.00040896123969356659,
            0.00033687997844436969,
            0.00027500086088333539,
            0.00021569241989095802,
            0.00015741008683205964,
            0.00010307945438570425,
            5.7568001440154096e-05,
            2.5019123969037183e-05,
            6.7757966801952286e-06,
            4.0417388231392693e-07,
            -5.4568417370319369e-13,
        ];
        for i in 0..n {
            assert!(
                (waveform[i] - expected[i]).abs() < 1e-15,
                "waveform[{i}]: {} != {}",
                waveform[i],
                expected[i]
            );
        }
    }

    // Windowed waveform at the end of the signal (position=(x_length-1)/fs, f0=200).
    #[test]
    fn test_get_windowed_waveform_edge_end_matches_cpp_reference() {
        let fs = 16000.0;
        let x_length = 1600;
        let x = test_signal(fs, x_length);
        let hwl = 120;
        let n = 2 * hwl + 1;
        let mut waveform = vec![0.0f64; n];
        let mut safe_index = vec![0i32; n];
        let mut window = vec![0.0f64; n];
        let mut state = RandnState::default();
        randn_reseed(&mut state);
        get_windowed_waveform(
            &x,
            fs,
            200.0,
            (x_length - 1) as f64 / fs,
            &mut waveform,
            &mut safe_index,
            &mut window,
            &mut state,
        );

        let expected: &[f64] = &[
            -1.3276404961943626e-12,
            1.6894768893480662e-06,
            1.1590886933635457e-06,
            -8.9980811991170061e-06,
            -3.3366240304079687e-05,
            -7.2577638897503959e-05,
            -0.00012340721077600095,
            -0.00018018085267446863,
            -0.00023722316501385913,
            -0.00029175263726278144,
            -0.00034644160975008719,
            -0.00041084966927246912,
            -0.00050112853123712936,
            -0.0006377546554811504,
            -0.00084150324949731079,
            -0.0011283368861032071,
            -0.0015042384044224527,
            -0.0019611801566152389,
            -0.0024753349697725605,
            -0.0030082914374197074,
            -0.0035114861608654103,
            -0.0039334046231521121,
            -0.0042284588568249435,
            -0.0043659603473423037,
            -0.0043373872517050507,
            -0.0041602680417165919,
            -0.0038774782026526032,
            -0.0035515124339298952,
            -0.0032542284123338512,
            -0.0030534924423858442,
            -0.0029989109318482058,
            -0.003109242226415853,
            -0.0033640412433769556,
            -0.0037015621335410431,
            -0.004023987871800693,
            -0.0042098125083761687,
            -0.00413187780179661,
            -0.0036783987555975492,
            -0.0027735292218944618,
            -0.0013937925427804803,
            0.00042288205046648675,
            0.0025777440841553063,
            0.0049221441046180901,
            0.0072777559180823024,
            0.0094633572414295659,
            0.011324078067959011,
            0.012758293935397274,
            0.013737396668068934,
            0.014314560474162361,
            0.014620236994738998,
            0.014844225116292034,
            0.015206436712536901,
            0.015920535854043478,
            0.0171561005316236,
            0.019005556909085085,
            0.021461713412363553,
            0.02441029099995861,
            0.027639593311232918,
            0.03086672348513568,
            0.033776969460961023,
            0.036070612716803827,
            0.037509885186864739,
            0.037958402725683377,
            0.037406261719287827,
            0.035976013735968912,
            0.033907647543936804,
            0.031524063754458896,
            0.029181787658949424,
            0.027214286772442919,
            0.025876781542483306,
            0.025301564870923034,
            0.025471502172161608,
            0.02621673265491322,
            0.027236019991104497,
            0.028140258156316527,
            0.028511957888744856,
            0.027971731726335436,
            0.026241348740969318,
            0.023193125553468898,
            0.018877278990015876,
            0.013522138634261457,
            0.0075063161999621405,
            0.0013064002410255997,
            -0.0045722277851592717,
            -0.0096709404869208399,
            -0.013647950086147978,
            -0.016331664005857813,
            -0.017748421397605178,
            -0.018119586703719843,
            -0.017827740901945919,
            -0.017356629927527956,
            -0.01721381553980713,
            -0.0178479245605902,
            -0.019573505685279845,
            -0.022515563072580597,
            -0.026582951878948373,
            -0.031475417752706895,
            -0.036723815897354822,
            -0.041757779730403503,
            -0.045990664731843198,
            -0.04890869289793142,
            -0.050150353977897716,
            -0.049563450299489713,
            -0.047230521945692758,
            -0.043458252456596333,
            -0.038732077969337454,
            -0.033642719705815435,
            -0.028795856408461152,
            -0.02471892404913912,
            -0.021779617862580626,
            -0.020128961933132829,
            -0.019678055125654736,
            -0.020112377398995563,
            -0.020941664239099794,
            -0.021577754964840528,
            -0.021428381829924244,
            -0.019992299628069862,
            -0.016940876135852767,
            -0.01217332587064216,
            -0.0058368567272010941,
            0.0016915281965539253,
            0.0016912383740678803,
            0.0016903691068316396,
            0.0016889209892273771,
            0.0016868950122703053,
            0.0016842925657898859,
            0.0016811154386670857,
            0.0016773658000755649,
            0.0016730462216014725,
            0.001668159668447234,
            0.0016627094861981669,
            0.0016566994067253576,
            0.0016501335571521974,
            0.0016430164306705554,
            0.0016353529064024319,
            0.0016271482379014261,
            0.0016184080487780771,
            0.0016091383283493342,
            0.001599345429309491,
            0.0015890360595448355,
            0.0015782172925671691,
            0.0015668965400260212,
            0.0015550815545538471,
            0.0015427804410593814,
            0.0015300016270863477,
            0.0015167538725533629,
            0.001503046252098717,
            0.0014888881637950728,
            0.0014742893106954027,
            0.0014592597000051832,
            0.0014438096265710952,
            0.0014279496851338858,
            0.0014116907433307209,
            0.0013950439403995546,
            0.0013780206920519528,
            0.0013606326607849039,
            0.0013428917624762737,
            0.0013248101581511485,
            0.0013064002398406093,
            0.0012876746253179172,
            0.0012686461462452225,
            0.0012493278474660389,
            0.0012297329636192138,
            0.0012098749279723523,
            0.0011897673478443727,
            0.0011694240070360285,
            0.0011488588426084596,
            0.0011280859565092937,
            0.001107119578057,
            0.0010859740804819353,
            0.0010646639561181269,
            0.0010432038063474744,
            0.001021608341332185,
            0.00099989236367958,
            0.00097807075149833073,
            0.00095615846728965698,
            0.00093417052048086133,
            0.00091212198460087924,
            0.00089002797026721796,
            0.00086790362006144203,
            0.00084576409875976657,
            0.00082362457410118696,
            0.00080150022416748115,
            0.00077940621186483325,
            0.00075735767585214655,
            0.00073536973214133067,
            0.00071345744366048012,
            0.00069163583470459181,
            0.00066991985437300696,
            0.0006483243925839619,
            0.00062686424235057015,
            0.00060555411662871136,
            0.00058440861916290955,
            0.00056344224191129215,
            0.00054266935262397642,
            0.00052210419137875986,
            0.00050176084686397778,
            0.00048165326938412337,
            0.00046179523328829589,
            0.00044220035005005688,
            0.00042288204732383481,
            0.00040385356913263102,
            0.00038512795529606369,
            0.0003667180371997585,
            0.00034863643376884966,
            0.00033089553628604562,
            0.00031350750536553282,
            0.00029648425322023782,
            0.00027983745322983323,
            0.0002635785107260051,
            0.00024771856932880813,
            0.00023226849606030665,
            0.00021723888430441032,
            0.00020264003337726474,
            0.00018848194448454179,
            0.00017477432720401187,
            0.00016152656888195624,
            0.00014874775494999994,
            0.00013644664158541286,
            0.00012463165860011134,
            0.00011331090479698434,
            0.0001024921348264467,
            9.2182768438734146e-05,
            8.2389868842693687e-05,
            7.312014869579269e-05,
            6.4379959286958396e-05,
            5.6175291194723604e-05,
            4.8511766178877393e-05,
            4.1394641748275171e-05,
            3.4828789836758401e-05,
            2.8818712082406096e-05,
            2.3368529565820785e-05,
            1.8481975923516806e-05,
            1.4162398257593331e-05,
            1.0412759027208668e-05,
            7.2356297767627636e-06,
            4.6331864986976408e-06,
            2.6072077487505745e-06,
            1.1590895985509935e-06,
            2.8982260582650268e-07,
            -5.4568417370319369e-13,
        ];
        for i in 0..n {
            assert!(
                (waveform[i] - expected[i]).abs() < 1e-15,
                "waveform[{i}]: {} != {}",
                waveform[i],
                expected[i]
            );
        }
    }
}

#[cfg(test)]
// Reference values are captured from the C++ port at full f64 precision; the
// excess digits are intentional (documenting the exact reference), not noise.
#[allow(clippy::approx_constant)]
#[allow(clippy::excessive_precision)]
mod phase25_tests {
    use super::*;
    use crate::matlab::randn_reseed;

    // C++ reference values (fs=16000, f0=200, q1=-0.15, fft_size=256),
    // generated from ext_src/world-cpp via scratch/phase25_cpp_ref_256.cpp.
    const PS_REF: [f64; 129] = [
        2805.5453637832188,
        750.93834560534935,
        86.966366023913039,
        3447.4760134999792,
        244.32070033382621,
        47.420388477863412,
        17.145015504931365,
        7.2590351563497855,
        3.0951014811101625,
        1.1866485357250149,
        0.33868597183622889,
        0.034849142054226324,
        0.015525162667743567,
        0.13617017024593003,
        0.31128440943725455,
        0.48929661616080639,
        144.00000000000009,
        0.74754789997429827,
        0.80608694740346554,
        0.81672699626950251,
        0.78524938835913904,
        0.72027840491665551,
        0.63179442142020781,
        0.52993940684032959,
        0.42409600790214158,
        0.32223020782477962,
        0.230484968946682,
        0.15300447131529377,
        0.091959336879096576,
        0.047735001666202211,
        0.01923965998641421,
        0.0042857489587088621,
        1.3045801027023451e-28,
        0.0032214340272458662,
        0.010853747575430308,
        0.020147515792545194,
        0.02889758413072822,
        0.035550971671698989,
        0.039229681312308487,
        0.039680280197487476,
        0.037167469349671203,
        0.032331844269875544,
        0.026032650736769438,
        0.019194785550385427,
        0.012675992260729397,
        0.0071656987077545631,
        0.003121841659756376,
        0.00074692521613506813,
        3.4771252522598511e-31,
        0.00063765116622376504,
        0.0022747191440714801,
        0.0044544600521266506,
        0.0067181369377300867,
        0.0086654347502170756,
        0.0099993233631962865,
        0.010551703735499386,
        0.010288994759741682,
        0.0092994103282800864,
        0.0077657572854486238,
        0.0059289628049829296,
        0.0040481237154800977,
        0.0023626726204650028,
        0.0010613804912868821,
        0.00026153822393944914,
        6.5298098264359444e-30,
        0.00023606267955785044,
        0.00086461603161409135,
        0.0017368060943206507,
        0.002684727422890637,
        0.0035464501441952595,
        0.004187976302202601,
        0.0045194274533806415,
        0.0045037658746624664,
        0.0041574919553779141,
        0.0035438813813854645,
        0.0027602832395661613,
        0.0019216816725484143,
        0.0011430605676448439,
        0.00052308389676879185,
        0.00013124326128916426,
        4.3236852488234927e-29,
        0.00012266259112243835,
        0.0004569064200389957,
        0.00093307648362789914,
        0.0014658124433125131,
        0.0019671699373353954,
        0.0023593170899410729,
        0.0025850575902101277,
        0.0026148329776621345,
        0.0024494198560092604,
        0.0021181775990986472,
        0.0016733278186365956,
        0.0011812740798783563,
        0.00071233218683510022,
        0.0003303964753285067,
        8.4004508471039976e-05,
        3.5955480853666833e-29,
        8.0576584111256524e-05,
        0.00030397718720196983,
        0.000628598897860795,
        0.00099978438282269074,
        0.0013582305619007527,
        0.001648757362679833,
        0.0018281739949007655,
        0.0018711516349009935,
        0.0017733299092250636,
        0.001551301073444552,
        0.0012395692989936339,
        0.00088500874772727697,
        0.00053968449836886497,
        0.00025310976139116524,
        6.5065384503430778e-05,
        9.6635460889573946e-30,
        6.378019890961121e-05,
        0.00024320793007370008,
        0.00050831853323499848,
        0.00081707516397064317,
        0.0011217396231252732,
        0.0013759718876287178,
        0.0015416212160876098,
        0.0015942310371029752,
        0.0015264774344560909,
        0.0013490672260497512,
        0.0010889918553363316,
        0.00078541455520258498,
        0.00048380646590067138,
        0.00022919506357470563,
        5.9510938532692478e-05,
        4.9758014989201937e-32,
    ];

    const SE_REF: [f64; 129] = [
        1201.8973475406485,
        742.18538139770624,
        930.05212263806538,
        568.45698315212019,
        458.83141236802646,
        75.294642786332048,
        17.888918014546221,
        7.3640007505297636,
        3.1548577856748516,
        1.0298724175935272,
        0.15385853532447974,
        0.026137182996468323,
        0.01580112894340575,
        0.040448053587166778,
        0.26736178563465868,
        5.5141380129612836,
        6.6972032007301356,
        8.1339453601437768,
        0.88979531954358204,
        0.6662155885183384,
        0.62655569503431496,
        0.72658338491723606,
        0.64912070229091012,
        0.59008904709074217,
        0.42084046934738245,
        0.38820085959493439,
        0.21584058852764121,
        0.31399426132434377,
        0.43586712346353984,
        0.29712402145581601,
        0.0096615647983844099,
        9.1840566771660461e-09,
        6.1391471797494467e-09,
        6.9473854507096429e-09,
        0.0054204173563081161,
        0.12800368980659682,
        0.13530543396807534,
        0.075919797940302056,
        0.035933075048939368,
        0.051412902464567783,
        0.035396965642748163,
        0.04182439022112909,
        0.024028338889907352,
        0.039725740393142285,
        0.053350867981227248,
        0.040420148856255479,
        0.0015808891295311545,
        4.1477715746034047e-09,
        2.7989508505793569e-09,
        3.5420879721932677e-09,
        0.001152343160367456,
        0.02514846194401223,
        0.028303381398436103,
        0.017953249846496936,
        0.0092505451579754233,
        0.013653084838057116,
        0.0098535436712901773,
        0.011997548569417158,
        0.0072459671160893073,
        0.011870570128506546,
        0.015446415447545971,
        0.011817098024779779,
        0.00054718371070848785,
        3.5385040066979132e-09,
        2.3841324816461366e-09,
        3.1946537626369146e-09,
        0.00044566000047087492,
        0.0086936383179232139,
        0.010239952017184327,
        0.0071102793267020558,
        0.0039048950545631971,
        0.005843121417129931,
        0.0043077403440237834,
        0.0053976942110186043,
        0.0032686590272551414,
        0.0057858667327917966,
        0.0082991333269681698,
        0.0066787372204185572,
        0.00026369435575685662,
        5.8253149034207927e-10,
        4.1300463362498608e-10,
        5.4439493164452838e-10,
        0.00023041239772808011,
        0.0054500852177367136,
        0.0063352944092281713,
        0.0041205329736184841,
        0.0021792617735679237,
        0.003351244740967263,
        0.0025081088335196728,
        0.0031555332662493709,
        0.0019897038696508941,
        0.003275304603928387,
        0.0042224294310589366,
        0.0032847967268551789,
        0.00017250278423956497,
        2.0581250329336169e-09,
        1.4049879018765794e-09,
        1.9737619098618931e-09,
        0.00015878328173461281,
        0.0028968500471193153,
        0.0035774690582238153,
        0.0026551688719862439,
        0.0015518108133364954,
        0.0023494321713588388,
        0.0018016428927151748,
        0.0022865205436388802,
        0.0014473134926930751,
        0.0025117192373931329,
        0.0035028697688396231,
        0.0028186642739187716,
        0.00012985798856790027,
        6.4459583046003191e-10,
        4.5750808252565086e-10,
        6.3197759141363465e-10,
        0.0001247352786310869,
        0.0026563967314166804,
        0.0032314820256355431,
        0.0022755718403631325,
        0.0012818531390915342,
        0.0019920078905120713,
        0.0015305670328531562,
        0.001970655872167066,
        0.0012598764568384462,
        0.002186849786498845,
        0.0030173585893802958,
        0.0024398407263202734,
        0.00011812877468357497,
        7.6460178589527662e-10,
        5.4328238309400256e-10,
    ];

    const LIFTER_SMOOTHING_REF: [f64; 129] = [
        1.0,
        0.99974299886925766,
        0.99897223324853845,
        0.9976884161904872,
        0.99589273524356137,
        0.99358685114420575,
        0.99077289598809126,
        0.98745347088261404,
        0.98363164308346596,
        0.97931094261871443,
        0.97449535840443269,
        0.96918933385654116,
        0.96339776200411587,
        0.95712598011001981,
        0.95037976380530087,
        0.94316532074438209,
        0.93548928378863905,
        0.92735870372652585,
        0.9187810415389589,
        0.90976416021921236,
        0.90031631615710606,
        0.89044615009778516,
        0.88016267768589562,
        0.86947527960644777,
        0.85839369133413979,
        0.84692799250337192,
        0.83508859591162976,
        0.82288623616934475,
        0.8103319580097551,
        0.79743710427268433,
        0.78421330357653718,
        0.77067245769317283,
        0.75682672864065703,
        0.74268852550922138,
        0.72827049103606123,
        0.71358548794489018,
        0.69864658506643418,
        0.68346704325629237,
        0.66806030112682113,
        0.65243996060989407,
        0.63661977236758138,
        0.62061362106794793,
        0.60443551054331424,
        0.58809954884843763,
        0.57161993323617344,
        0.55501093506824806,
        0.5382868846788279,
        0.52146215620860226,
        0.50455115242710458,
        0.48756828956098675,
        0.47052798214592223,
        0.45344462791976181,
        0.43633259277448361,
        0.41920619578438317,
        0.40207969432782176,
        0.38496726931971442,
        0.36788301057177419,
        0.3508409022973385,
        0.33385480877740875,
        0.31693846020429656,
        0.30010543871903539,
        0.28336916465844314,
        0.2667428830274397,
        0.25023965021191957,
        0.23387232094715982,
        0.21765353555639586,
        0.20159570747385158,
        0.1857110110661232,
        0.17001136976543207,
        0.15450844452785434,
        0.13921362262920453,
        0.12413800681082308,
        0.10929240478705181,
        0.094687319125730238,
        0.080332937512547997,
        0.066239123409613687,
        0.052415407118082215,
        0.038870977254174793,
        0.025614672647398463,
        0.012654974669231076,
        3.8981718325193755e-17,
        -0.012342506158879754,
        -0.024365176420696157,
        -0.036061027091222456,
        -0.04742346358302682,
        -0.058446285361423891,
        -0.069123690417773775,
        -0.079450279266417137,
        -0.089421058462133232,
        -0.099031443635600236,
        -0.10827726204493694,
        -0.11715475464199951,
        -0.12566057765271074,
        -0.13379180367129312,
        -0.14154592226887458,
        -0.14892084011753404,
        -0.15591488063143988,
        -0.16252678312732929,
        -0.16875570150715574,
        -0.1746012024663135,
        -0.18006326323142124,
        -0.18514226883221277,
        -0.1898390089126441,
        -0.19415467408687664,
        -0.19809085184633993,
        -0.20164952202461237,
        -0.2048330518273809,
        -0.2076441904352552,
        -0.21008606318771428,
        -0.2121621653569527,
        -0.21387635552087378,
        -0.21523284854494015,
        -0.21623620818304484,
        -0.2168913393080027,
        -0.21720347978268495,
        -0.21717819198322746,
        -0.21682135398613472,
        -0.21613915043147708,
        -0.21513806307473904,
        -0.21382486104021739,
        -0.21220659078919379,
        -0.21029056581641214,
        -0.20808435608868198,
        -0.20559577723969771,
        -0.20283287953541637,
        -0.19980393662456927,
        -0.19651743408909589,
        -0.19298205780948266,
        -0.18920668216016426,
    ];

    const LIFTER_COMPENSATION_REF: [f64; 129] = [
        1.0,
        1.0009247998800617,
        1.0036934978214587,
        1.008289023880697,
        1.0146830451114539,
        1.022836140246614,
        1.0326980427434898,
        1.0442079506937725,
        1.0572949016875159,
        1.0718782103199906,
        1.0878679656440358,
        1.105165585500945,
        1.123664424312258,
        1.1432504305852154,
        1.1638028500781359,
        1.1851949702904732,
        1.2072949016875159,
        1.2299663908432283,
        1.2530696604879308,
        1.2764622712816465,
        1.3,
        1.3235377287183534,
        1.3469303395120693,
        1.3700336091567715,
        1.3927050983124842,
        1.4148050297095269,
        1.436197149921864,
        1.4567495694147847,
        1.4763355756877419,
        1.4948344144990553,
        1.5121320343559643,
        1.5281217896800092,
        1.5427050983124841,
        1.5557920493062276,
        1.5673019572565103,
        1.5771638597533861,
        1.5853169548885462,
        1.5917109761193031,
        1.5963065021785412,
        1.5990752001199384,
        1.6000000000000001,
        1.5990752001199384,
        1.5963065021785414,
        1.5917109761193031,
        1.5853169548885462,
        1.5771638597533861,
        1.5673019572565103,
        1.5557920493062278,
        1.5427050983124844,
        1.5281217896800094,
        1.5121320343559643,
        1.4948344144990553,
        1.4763355756877421,
        1.456749569414785,
        1.4361971499218642,
        1.4148050297095272,
        1.3927050983124842,
        1.3700336091567715,
        1.3469303395120691,
        1.3235377287183536,
        1.3,
        1.2764622712816467,
        1.2530696604879308,
        1.2299663908432283,
        1.2072949016875159,
        1.1851949702904729,
        1.1638028500781361,
        1.1432504305852154,
        1.1236644243122582,
        1.1051655855009448,
        1.0878679656440355,
        1.0718782103199909,
        1.0572949016875159,
        1.0442079506937725,
        1.0326980427434898,
        1.0228361402466142,
        1.0146830451114539,
        1.0082890238806972,
        1.0036934978214589,
        1.0009247998800617,
        1.0,
        1.0009247998800617,
        1.0036934978214589,
        1.008289023880697,
        1.0146830451114539,
        1.0228361402466142,
        1.0326980427434895,
        1.0442079506937723,
        1.0572949016875155,
        1.0718782103199906,
        1.0878679656440358,
        1.1051655855009448,
        1.123664424312258,
        1.1432504305852151,
        1.1638028500781359,
        1.1851949702904729,
        1.2072949016875156,
        1.2299663908432286,
        1.2530696604879308,
        1.2764622712816467,
        1.3,
        1.3235377287183536,
        1.3469303395120689,
        1.3700336091567715,
        1.3927050983124842,
        1.4148050297095265,
        1.4361971499218638,
        1.456749569414785,
        1.4763355756877419,
        1.4948344144990551,
        1.5121320343559641,
        1.5281217896800092,
        1.5427050983124841,
        1.5557920493062278,
        1.5673019572565106,
        1.5771638597533859,
        1.5853169548885462,
        1.5917109761193031,
        1.5963065021785412,
        1.5990752001199384,
        1.6000000000000001,
        1.5990752001199384,
        1.5963065021785414,
        1.5917109761193031,
        1.5853169548885462,
        1.5771638597533861,
        1.5673019572565103,
        1.5557920493062278,
        1.5427050983124844,
    ];

    const FS: f64 = 16000.0;
    const F0: f64 = 200.0;
    const Q1: f64 = -0.15;
    const FFT_SIZE: i32 = 256;
    const HALF: usize = (FFT_SIZE / 2) as usize;

    fn synthetic_waveform(fs: f64) -> Vec<f64> {
        (0..FFT_SIZE as usize)
            .map(|i| {
                let t = i as f64 / fs;
                0.5 * (2.0 * K_PI * 200.0 * t).sin() + 0.1 * (2.0 * K_PI * 1000.0 * t).sin()
            })
            .collect()
    }

    fn max_err(a: &[f64], b: &[f64]) -> f64 {
        (0..a.len())
            .map(|i| (a[i] - b[i]).abs())
            .fold(0.0f64, f64::max)
    }

    #[test]
    fn get_power_spectrum_matches_cpp_reference() {
        let mut forward = ForwardRealFFT::new(FFT_SIZE as usize);
        forward.waveform.copy_from_slice(&synthetic_waveform(FS));
        let mut dc_scratch =
            vec![0.0f64; dc_correction_scratch_capacity(FFT_SIZE as usize, HALF + 1)];
        get_power_spectrum(FS, F0, &mut forward, &mut dc_scratch);
        let err = max_err(&forward.waveform[..PS_REF.len()], &PS_REF);
        assert!(err < 1e-6, "PS max_err = {err}");
    }

    #[test]
    fn get_power_spectrum_dc_correction_applied() {
        let waveform = synthetic_waveform(FS);
        let half_window_length = matlab_round(1.5 * FS / F0) as usize;

        let mut fwd_raw = ForwardRealFFT::new(FFT_SIZE as usize);
        fwd_raw.waveform.copy_from_slice(&waveform);
        for i in (2 * half_window_length + 1)..FFT_SIZE as usize {
            fwd_raw.waveform[i] = 0.0;
        }
        fwd_raw.forward();
        let mut raw = vec![0.0; HALF + 1];
        for (i, slot) in raw.iter_mut().enumerate().take(HALF + 1) {
            *slot = fwd_raw.spectrum[i].re * fwd_raw.spectrum[i].re
                + fwd_raw.spectrum[i].im * fwd_raw.spectrum[i].im;
        }

        let mut fwd = ForwardRealFFT::new(FFT_SIZE as usize);
        fwd.waveform.copy_from_slice(&waveform);
        let mut dc_scratch =
            vec![0.0f64; dc_correction_scratch_capacity(FFT_SIZE as usize, HALF + 1)];
        get_power_spectrum(FS, F0, &mut fwd, &mut dc_scratch);
        let diff = (0..5)
            .map(|i| (fwd.waveform[i] - raw[i]).abs())
            .fold(0.0f64, f64::max);
        assert!(diff > 0.0, "DC correction had no effect (diff={diff})");
    }

    #[test]
    fn smoothing_with_recovery_matches_cpp_reference() {
        let mut state = RandnState::default();
        randn_reseed(&mut state);
        let mut input = vec![0.0; HALF + 1];
        add_infinitesimal_noise(&PS_REF, FFT_SIZE, &mut input, &mut state);

        let mut forward = ForwardRealFFT::new(FFT_SIZE as usize);
        forward.waveform[..HALF + 1].copy_from_slice(&input);
        let mut inverse = InverseRealFFT::new(FFT_SIZE as usize);
        let mut smoothing_lifter = vec![0.0; FFT_SIZE as usize];
        let mut compensation_lifter = vec![0.0; FFT_SIZE as usize];
        let mut se = vec![0.0; HALF + 1];
        smoothing_with_recovery(
            F0,
            FS,
            Q1,
            &mut forward,
            &mut inverse,
            &mut smoothing_lifter,
            &mut compensation_lifter,
            &mut se,
        );
        let err = max_err(&se, &SE_REF);
        assert!(err < 1e-6, "SE max_err = {err}");
    }

    #[test]
    fn smoothing_lifters_match_reference() {
        let mut forward = ForwardRealFFT::new(FFT_SIZE as usize);
        forward.waveform[..HALF + 1].copy_from_slice(&PS_REF);
        let mut inverse = InverseRealFFT::new(FFT_SIZE as usize);
        let mut smoothing_lifter = vec![0.0; FFT_SIZE as usize];
        let mut compensation_lifter = vec![0.0; FFT_SIZE as usize];
        let mut se = vec![0.0; HALF + 1];
        smoothing_with_recovery(
            F0,
            FS,
            Q1,
            &mut forward,
            &mut inverse,
            &mut smoothing_lifter,
            &mut compensation_lifter,
            &mut se,
        );
        let ls_err = (0..=HALF)
            .map(|i| (smoothing_lifter[i] - LIFTER_SMOOTHING_REF[i]).abs())
            .fold(0.0f64, f64::max);
        let lc_err = (0..=HALF)
            .map(|i| (compensation_lifter[i] - LIFTER_COMPENSATION_REF[i]).abs())
            .fold(0.0f64, f64::max);
        assert!(ls_err < 1e-12, "LIFTER_SMOOTHING max_err = {ls_err}");
        assert!(lc_err < 1e-12, "LIFTER_COMPENSATION max_err = {lc_err}");
    }

    #[test]
    fn smoothing_with_recovery_envelope_positive() {
        let mut state = RandnState::default();
        randn_reseed(&mut state);
        let mut input = vec![0.0; HALF + 1];
        add_infinitesimal_noise(&PS_REF, FFT_SIZE, &mut input, &mut state);

        let mut forward = ForwardRealFFT::new(FFT_SIZE as usize);
        forward.waveform[..HALF + 1].copy_from_slice(&input);
        let mut inverse = InverseRealFFT::new(FFT_SIZE as usize);
        let mut smoothing_lifter = vec![0.0; FFT_SIZE as usize];
        let mut compensation_lifter = vec![0.0; FFT_SIZE as usize];
        let mut se = vec![0.0; HALF + 1];
        smoothing_with_recovery(
            F0,
            FS,
            Q1,
            &mut forward,
            &mut inverse,
            &mut smoothing_lifter,
            &mut compensation_lifter,
            &mut se,
        );
        for v in &se {
            assert!(
                v.is_finite() && *v > 0.0,
                "envelope not positive/finite: {v}"
            );
        }
    }

    #[test]
    fn get_power_spectrum_synthetic_sinusoid_peak() {
        let fs = 16000.0;
        let waveform: Vec<f64> = (0..FFT_SIZE as usize)
            .map(|i| (2.0 * K_PI * 1000.0 * i as f64 / fs).sin())
            .collect();
        let mut forward = ForwardRealFFT::new(FFT_SIZE as usize);
        forward.waveform.copy_from_slice(&waveform);
        let mut dc_scratch =
            vec![0.0f64; dc_correction_scratch_capacity(FFT_SIZE as usize, HALF + 1)];
        get_power_spectrum(fs, F0, &mut forward, &mut dc_scratch);
        let peak_bin = (1..=HALF)
            .max_by(|&a, &b| forward.waveform[a].total_cmp(&forward.waveform[b]))
            .unwrap();
        assert!(
            peak_bin.abs_diff(16) <= 1,
            "peak at bin {peak_bin}, expected ~16"
        );
    }
}
