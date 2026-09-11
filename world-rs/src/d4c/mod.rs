//! D4C aperiodicity extraction (port of https://github.com/mmorise/World/blob/master/src/d4c.cpp).
//!
//! Phase 1: module scaffolding and type definitions.
//! Phase 2: option initialization & input validation.
//! Phase 3: stage mapping (smoothed power spectrum → static/dynamic group delay → aperiodicity estimation/liftering).
//! Phase 4: accuracy validation with real extraction.
//!
//! C++ parity: internal helpers keep the C++ reference's argument lists, so
//! `clippy::too_many_arguments` is allowed module-wide.
#![allow(clippy::too_many_arguments)]

use crate::common::nuttall_window;
use crate::constants::{
    K_FLOOR_F0_D4C, K_FREQUENCY_INTERVAL, K_LOG2, K_MY_SAFE_GUARD_MINIMUM, K_THRESHOLD,
    K_UPPER_LIMIT,
};
use crate::d4c::body::{d4c_general_body, get_aperiodicity};
use crate::d4c::lovetrain::d4c_love_train;
use crate::fft::ForwardRealFFT;
use crate::matlab::{randn_reseed, RandnState};

mod body;
mod centroid;
mod lovetrain;
mod window;

/// C++: `D4COption` (world/d4c.h:16-18).
#[derive(Debug, Clone, PartialEq)]
pub struct D4COption {
    /// Threshold for aperiodicity estimation (default `kThreshold = 0.85`).
    pub threshold: f64,
}

/// C++: `InitializeD4COption` (d4c.cpp:405-406).
///
/// Returns the default parameters matching the C++ reference: `threshold = kThreshold`.
pub fn initialize_d4c_option() -> D4COption {
    D4COption {
        threshold: K_THRESHOLD,
    }
}

/// Errors for the D4C API.
#[derive(Debug, PartialEq)]
pub enum D4CError {
    EmptyInput,
    NonFiniteInput,
    NonPositiveSampleRate { fs: f64 },
    NonFiniteSampleRate,
    EmptyTemporalPositions,
    EmptyF0,
    LengthMismatch,
    InvalidFftSize,
    NonFiniteThreshold,
}

impl std::fmt::Display for D4CError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            D4CError::EmptyInput => write!(f, "input signal is empty"),
            D4CError::NonFiniteInput => write!(f, "input contains non-finite samples"),
            D4CError::NonPositiveSampleRate { fs } => {
                write!(f, "sample rate must be positive, got {fs}")
            }
            D4CError::NonFiniteSampleRate => write!(f, "sample rate is non-finite"),
            D4CError::EmptyTemporalPositions => write!(f, "temporal_positions is empty"),
            D4CError::EmptyF0 => write!(f, "f0 is empty"),
            D4CError::LengthMismatch => write!(f, "temporal_positions and f0 length mismatch"),
            D4CError::InvalidFftSize => write!(f, "fft_size is invalid"),
            D4CError::NonFiniteThreshold => write!(f, "threshold is non-finite"),
        }
    }
}

impl std::error::Error for D4CError {}

/// C++: `D4C` (d4c.h:35-37).
///
/// Validates inputs and returns a stub error until the full port is implemented.
/// The signature matches the C++ API: `x`, `fs`, `temporal_positions`, `f0`, `fft_size`, `option`.
/// Output is a `Vec<Vec<f64>>` with shape `[frames][fft_size/2+1]`.
pub fn d4c(
    x: &[f64],
    fs: f64,
    temporal_positions: &[f64],
    f0: &[f64],
    fft_size: i32,
    option: &D4COption,
) -> Result<Vec<Vec<f64>>, D4CError> {
    if x.is_empty() {
        return Err(D4CError::EmptyInput);
    }
    if !x.iter().all(|v| v.is_finite()) {
        return Err(D4CError::NonFiniteInput);
    }
    if !fs.is_finite() {
        return Err(D4CError::NonFiniteSampleRate);
    }
    if fs <= 0.0 {
        return Err(D4CError::NonPositiveSampleRate { fs });
    }
    if fs < 1000.0 {
        return Err(D4CError::NonPositiveSampleRate { fs });
    }
    if temporal_positions.is_empty() {
        return Err(D4CError::EmptyTemporalPositions);
    }
    if f0.is_empty() {
        return Err(D4CError::EmptyF0);
    }
    if temporal_positions.len() != f0.len() {
        return Err(D4CError::LengthMismatch);
    }
    if fft_size <= 0 {
        return Err(D4CError::InvalidFftSize);
    }
    // Allocation budget: the output is `frames × (fft_size/2+1)` f64 plus
    // FFT scratch scaling with `fft_size`. 2^20 (≈8MB per row) is 256× the
    // largest realistic size (8192).
    if fft_size > (1 << 20) {
        return Err(D4CError::InvalidFftSize);
    }
    if !option.threshold.is_finite() {
        return Err(D4CError::NonFiniteThreshold);
    }
    if !temporal_positions.iter().all(|v| v.is_finite()) {
        return Err(D4CError::NonFiniteInput);
    }
    if !f0.iter().all(|v| v.is_finite()) {
        return Err(D4CError::NonFiniteInput);
    }

    let fs_i = fs as i32;
    let fft_size_usize = fft_size as usize;
    let f0_length = f0.len();

    let mut randn_state = RandnState::default();
    randn_reseed(&mut randn_state);

    let mut aperiodicity: Vec<Vec<f64>> =
        vec![vec![1.0 - K_MY_SAFE_GUARD_MINIMUM; fft_size_usize / 2 + 1]; f0_length];

    let fft_size_d4c_f =
        2.0_f64.powi((((4.0 * fs / K_FLOOR_F0_D4C + 1.0).ln() / K_LOG2) as i32).saturating_add(1));
    let fft_size_d4c = fft_size_d4c_f as usize;
    // Allocation budget: `fft_size_d4c` derives from `fs` (not the capped
    // `fft_size` param), so a degenerate rate (e.g. `fs = 1e308`) saturates
    // the float→int cast to `usize::MAX` and `ForwardRealFFT::new` would
    // attempt a multi-GB plan. 2^20 covers sample rates up to ~10MHz
    // (reality: ≤192kHz → 32768).
    if fft_size_d4c > (1 << 20) {
        return Err(D4CError::InvalidFftSize);
    }
    let mut forward_real_fft = ForwardRealFFT::new(fft_size_d4c);

    let number_of_aperiodicities_f =
        K_UPPER_LIMIT.min(fs / 2.0 - K_FREQUENCY_INTERVAL) / K_FREQUENCY_INTERVAL;
    let number_of_aperiodicities = number_of_aperiodicities_f as usize;
    let window_length = ((K_FREQUENCY_INTERVAL * fft_size_d4c as f64 / fs) as usize) * 2 + 1;
    let mut window = vec![0.0f64; window_length];
    nuttall_window(window_length, &mut window);

    let mut aperiodicity0 = vec![0.0f64; f0_length];
    d4c_love_train(
        x,
        fs_i,
        x.len(),
        f0,
        f0_length,
        temporal_positions,
        &mut aperiodicity0,
        &mut randn_state,
    );

    let mut coarse_aperiodicity = vec![0.0f64; number_of_aperiodicities + 2];
    coarse_aperiodicity[0] = -60.0;
    coarse_aperiodicity[number_of_aperiodicities + 1] = -K_MY_SAFE_GUARD_MINIMUM;
    let mut coarse_frequency_axis = vec![0.0f64; number_of_aperiodicities + 2];
    for (i, slot) in coarse_frequency_axis
        .iter_mut()
        .enumerate()
        .take(number_of_aperiodicities + 1)
    {
        *slot = i as f64 * K_FREQUENCY_INTERVAL;
    }
    coarse_frequency_axis[number_of_aperiodicities + 1] = fs / 2.0;

    let mut frequency_axis = vec![0.0f64; fft_size_usize / 2 + 1];
    for (i, slot) in frequency_axis
        .iter_mut()
        .enumerate()
        .take(fft_size_usize / 2 + 1)
    {
        *slot = i as f64 * fs / fft_size as f64;
    }

    for i in 0..f0_length {
        if f0[i] == 0.0 || aperiodicity0[i] <= option.threshold {
            continue;
        }
        let cur_f0 = f0[i].max(K_FLOOR_F0_D4C);
        let mut coarse_body = vec![0.0f64; number_of_aperiodicities + 2];
        coarse_body[0] = -60.0;
        coarse_body[number_of_aperiodicities + 1] = -K_MY_SAFE_GUARD_MINIMUM;
        d4c_general_body(
            x,
            x.len(),
            fs_i,
            cur_f0,
            fft_size_d4c,
            temporal_positions[i],
            number_of_aperiodicities,
            &window,
            window_length,
            &mut forward_real_fft,
            &mut coarse_body[1..=number_of_aperiodicities],
            &mut randn_state,
        );
        let mut frame_aper = vec![0.0f64; fft_size_usize / 2 + 1];
        get_aperiodicity(
            &coarse_frequency_axis,
            &coarse_body,
            number_of_aperiodicities,
            &frequency_axis,
            fft_size_usize,
            &mut frame_aper,
        );
        aperiodicity[i] = frame_aper;
    }

    Ok(aperiodicity)
}
