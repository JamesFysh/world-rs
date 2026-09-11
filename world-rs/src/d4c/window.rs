use crate::constants::K_SAFE_GUARD_D4C;
use crate::constants::{K_HANNING, K_PI};
use crate::matlab::matlab_round;
use crate::matlab::randn;
use crate::matlab::RandnState;

pub(crate) fn set_parameters_for_windowed_waveform(
    half_window_length: usize,
    x_length: usize,
    current_position: f64,
    fs: i32,
    current_f0: f64,
    window_type: f64,
    window_length_ratio: f64,
    base_index: &mut [i32],
    safe_index: &mut [i32],
    window: &mut [f64],
) {
    let n = half_window_length * 2 + 1;
    let hwl = half_window_length as i32;
    for (i, slot) in base_index.iter_mut().enumerate().take(n) {
        *slot = i as i32 - hwl;
    }
    let origin = matlab_round(current_position * fs as f64 + 0.001) as i32;
    let x_len_i = x_length as i32 - 1;
    for i in 0..n {
        let idx = origin + base_index[i];
        let clamped = idx.clamp(0, x_len_i);
        safe_index[i] = clamped;
    }
    for i in 0..n {
        let base = base_index[i] as f64;
        let position = (2.0 * base / window_length_ratio) / fs as f64;
        let cos_val = (K_PI * position * current_f0).cos();
        if (window_type - K_HANNING).abs() < f64::EPSILON {
            window[i] = 0.5 * cos_val + 0.5;
        } else {
            window[i] = 0.42 + 0.5 * cos_val + 0.08 * (K_PI * position * current_f0 * 2.0).cos();
        }
    }
}

pub(crate) fn get_windowed_waveform(
    x: &[f64],
    x_length: usize,
    fs: i32,
    current_f0: f64,
    current_position: f64,
    window_type: f64,
    window_length_ratio: f64,
    waveform: &mut [f64],
    randn_state: &mut RandnState,
) {
    let half_window_length =
        matlab_round(window_length_ratio * fs as f64 / current_f0 / 2.0) as usize;
    let n = half_window_length * 2 + 1;
    let mut base_index = vec![0i32; n];
    let mut safe_index = vec![0i32; n];
    let mut window = vec![0.0f64; n];
    set_parameters_for_windowed_waveform(
        half_window_length,
        x_length,
        current_position,
        fs,
        current_f0,
        window_type,
        window_length_ratio,
        &mut base_index,
        &mut safe_index,
        &mut window,
    );
    for i in 0..n {
        let idx = safe_index[i] as usize;
        let sample = if idx < x.len() { x[idx] } else { 0.0 };
        waveform[i] = sample * window[i] + randn(randn_state) * K_SAFE_GUARD_D4C;
    }
    let mut tmp_weight1 = 0.0;
    let mut tmp_weight2 = 0.0;
    for i in 0..n {
        tmp_weight1 += waveform[i];
        tmp_weight2 += window[i];
    }
    let weighting_coefficient = if tmp_weight2 != 0.0 {
        tmp_weight1 / tmp_weight2
    } else {
        0.0
    };
    for i in 0..n {
        waveform[i] -= window[i] * weighting_coefficient;
    }
}
