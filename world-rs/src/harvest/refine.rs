#![allow(
    dead_code,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_memcpy,
    clippy::manual_range_contains
)]

use crate::constants::{K_MY_SAFE_GUARD_MINIMUM, K_PI};
use crate::fft::ForwardRealFFT;
use crate::matlab::matlab_round;
use rustfft::num_complex::Complex;

/// Per-frame allocation budget for `get_refined_f0` (mirrors the stonemask
/// frame cap): at most 1M time samples (~8MB `base_time`) and a 2^20 FFT.
const MAX_REFINE_BASE_TIME: usize = 1_000_000;
const MAX_REFINE_FFT: usize = 1 << 20;

pub(crate) struct RefineScratch {
    pub forward: ForwardRealFFT,
    pub forward_size: usize,
    pub base_time: Vec<f64>,
    pub base_index: Vec<i32>,
    pub safe_index: Vec<isize>,
    pub main_window: Vec<f64>,
    pub diff_window: Vec<f64>,
    pub main_spectrum: Vec<Complex<f64>>,
    pub diff_spectrum: Vec<Complex<f64>>,
    pub power_spectrum: Vec<f64>,
    pub numerator_i: Vec<f64>,
    pub amplitude_list: Vec<f64>,
    pub inst_freq_list: Vec<f64>,
}

impl RefineScratch {
    pub fn new() -> Self {
        let max_fft_spectrum = MAX_REFINE_FFT / 2 + 1;
        Self {
            forward: ForwardRealFFT::new(1),
            forward_size: 1,
            base_time: Vec::with_capacity(MAX_REFINE_BASE_TIME),
            base_index: Vec::with_capacity(MAX_REFINE_BASE_TIME),
            safe_index: Vec::with_capacity(MAX_REFINE_BASE_TIME),
            main_window: Vec::with_capacity(MAX_REFINE_BASE_TIME),
            diff_window: Vec::with_capacity(MAX_REFINE_BASE_TIME),
            main_spectrum: Vec::with_capacity(max_fft_spectrum),
            diff_spectrum: Vec::with_capacity(max_fft_spectrum),
            power_spectrum: Vec::with_capacity(max_fft_spectrum),
            numerator_i: Vec::with_capacity(max_fft_spectrum),
            amplitude_list: Vec::with_capacity(6),
            inst_freq_list: Vec::with_capacity(6),
        }
    }
}

pub(crate) fn get_base_index(
    current_position: f64,
    base_time: &[f64],
    fs: f64,
    base_index: &mut [i32],
) {
    if base_time.is_empty() || base_index.is_empty() {
        return;
    }
    let len = base_time.len().min(base_index.len());
    let basic = matlab_round((current_position + base_time[0]) * fs + 0.001) as i32;
    for i in 0..len {
        base_index[i] = basic + i as i32;
    }
}

pub(crate) fn get_main_window(
    current_position: f64,
    base_index: &[i32],
    fs: f64,
    window_length_in_time: f64,
    main_window: &mut [f64],
) {
    let len = base_index.len().min(main_window.len());
    for i in 0..len {
        let tmp = (base_index[i] as f64 - 1.0) / fs - current_position;
        let w = window_length_in_time;
        if w == 0.0 {
            main_window[i] = 0.0;
            continue;
        }
        let arg1 = 2.0 * K_PI * tmp / w;
        let arg2 = 4.0 * K_PI * tmp / w;
        main_window[i] = 0.42 + 0.5 * arg1.cos() + 0.08 * arg2.cos();
    }
}

pub(crate) fn get_diff_window(main_window: &[f64], diff_window: &mut [f64]) {
    let len = main_window.len().min(diff_window.len());
    if len < 2 {
        return;
    }
    diff_window[0] = -main_window[1] / 2.0;
    for i in 1..len - 1 {
        diff_window[i] = -(main_window[i + 1] - main_window[i - 1]) / 2.0;
    }
    diff_window[len - 1] = main_window[len - 2] / 2.0;
}

pub(crate) fn get_spectra(
    x: &[f64],
    x_length: usize,
    base_time_length: usize,
    scratch: &mut RefineScratch,
) {
    let len = base_time_length
        .min(scratch.base_index.len())
        .min(scratch.main_window.len())
        .min(scratch.diff_window.len());
    if x_length == 0 || len == 0 {
        scratch.forward.waveform.fill(0.0);
        scratch.forward.forward();
        let spec_len = scratch
            .main_spectrum
            .len()
            .min(scratch.forward.spectrum.len());
        for i in 0..spec_len {
            scratch.main_spectrum[i] = scratch.forward.spectrum[i];
        }
        scratch.forward.waveform.fill(0.0);
        scratch.forward.forward();
        let spec_len = scratch
            .diff_spectrum
            .len()
            .min(scratch.forward.spectrum.len());
        for i in 0..spec_len {
            scratch.diff_spectrum[i] = scratch.forward.spectrum[i];
        }
        return;
    }
    // safe indices
    if scratch.safe_index.len() < len {
        scratch.safe_index.resize(len, 0);
    }
    for i in 0..len {
        let idx = scratch.base_index[i] as isize - 1;
        let clamped = idx.clamp(0, x_length as isize - 1);
        scratch.safe_index[i] = clamped;
    }
    // main
    scratch.forward.waveform.fill(0.0);
    for i in 0..len {
        let xi = x
            .get(scratch.safe_index[i] as usize)
            .copied()
            .unwrap_or(0.0);
        scratch.forward.waveform[i] = xi * scratch.main_window[i];
    }
    scratch.forward.forward();
    let spec_len = scratch
        .main_spectrum
        .len()
        .min(scratch.forward.spectrum.len());
    for i in 0..spec_len {
        scratch.main_spectrum[i] = scratch.forward.spectrum[i];
    }
    // diff
    scratch.forward.waveform.fill(0.0);
    for i in 0..len {
        let xi = x
            .get(scratch.safe_index[i] as usize)
            .copied()
            .unwrap_or(0.0);
        scratch.forward.waveform[i] = xi * scratch.diff_window[i];
    }
    scratch.forward.forward();
    let spec_len = scratch
        .diff_spectrum
        .len()
        .min(scratch.forward.spectrum.len());
    for i in 0..spec_len {
        scratch.diff_spectrum[i] = scratch.forward.spectrum[i];
    }
}

pub(crate) fn fix_f0(
    fft_size: usize,
    fs: f64,
    current_f0: f64,
    number_of_harmonics: usize,
    refined_f0: &mut f64,
    score: &mut f64,
    scratch: &mut RefineScratch,
) {
    if number_of_harmonics == 0 || !current_f0.is_finite() || current_f0 <= 0.0 {
        *refined_f0 = 0.0;
        *score = 0.0;
        return;
    }
    if scratch.amplitude_list.len() < number_of_harmonics {
        scratch.amplitude_list.resize(number_of_harmonics, 0.0);
        scratch.inst_freq_list.resize(number_of_harmonics, 0.0);
    }
    let max_idx = (fft_size / 2).min(scratch.power_spectrum.len().saturating_sub(1));
    for i in 0..number_of_harmonics {
        let idx_f = current_f0 * fft_size as f64 / fs * (i as f64 + 1.0);
        let idx_raw = matlab_round(idx_f) as isize;
        let idx_clamped = if idx_raw < 0 {
            0
        } else if idx_raw > max_idx as isize {
            max_idx as isize
        } else {
            idx_raw
        };
        let idx_usize = idx_clamped as usize;
        let ps = *scratch.power_spectrum.get(idx_usize).unwrap_or(&0.0);
        if ps == 0.0 {
            scratch.inst_freq_list[i] = 0.0;
        } else {
            let num = *scratch.numerator_i.get(idx_usize).unwrap_or(&0.0);
            scratch.inst_freq_list[i] =
                idx_usize as f64 * fs / fft_size as f64 + num / ps * fs / 2.0 / K_PI;
        }
        scratch.amplitude_list[i] = ps.sqrt();
    }
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    let mut score_sum = 0.0;
    for i in 0..number_of_harmonics {
        let amp = scratch.amplitude_list[i];
        let inst = scratch.inst_freq_list[i];
        numerator += amp * inst;
        denominator += amp * (i as f64 + 1.0);
        if current_f0 != 0.0 {
            score_sum += ((inst / (i as f64 + 1.0) - current_f0) / current_f0).abs();
        }
    }
    *refined_f0 = numerator / (denominator + K_MY_SAFE_GUARD_MINIMUM);
    *score = 1.0 / (score_sum / number_of_harmonics as f64 + K_MY_SAFE_GUARD_MINIMUM);
}

pub(crate) fn get_mean_f0(
    x: &[f64],
    fs: f64,
    current_position: f64,
    current_f0: f64,
    fft_size: usize,
    window_length_in_time: f64,
    base_time_length: usize,
    refined_f0: &mut f64,
    refined_score: &mut f64,
    scratch: &mut RefineScratch,
) {
    if !current_f0.is_finite() || current_f0 <= 0.0 || !fs.is_finite() || fs <= 0.0 {
        *refined_f0 = 0.0;
        *refined_score = 0.0;
        return;
    }
    if scratch.forward_size != fft_size {
        scratch.forward = ForwardRealFFT::new(fft_size);
        scratch.forward_size = fft_size;
    }
    if scratch.base_index.len() < base_time_length {
        scratch.base_index.resize(base_time_length, 0);
    }
    if scratch.main_window.len() < base_time_length {
        scratch.main_window.resize(base_time_length, 0.0);
    }
    if scratch.diff_window.len() < base_time_length {
        scratch.diff_window.resize(base_time_length, 0.0);
    }
    get_base_index(
        current_position,
        &scratch.base_time[..base_time_length],
        fs,
        &mut scratch.base_index[..base_time_length],
    );
    get_main_window(
        current_position,
        &scratch.base_index[..base_time_length],
        fs,
        window_length_in_time,
        &mut scratch.main_window[..base_time_length],
    );
    get_diff_window(
        &scratch.main_window[..base_time_length],
        &mut scratch.diff_window[..base_time_length],
    );
    let spec_len = fft_size / 2 + 1;
    if scratch.main_spectrum.len() < spec_len {
        scratch
            .main_spectrum
            .resize(spec_len, Complex::new(0.0, 0.0));
    }
    if scratch.diff_spectrum.len() < spec_len {
        scratch
            .diff_spectrum
            .resize(spec_len, Complex::new(0.0, 0.0));
    }
    if scratch.power_spectrum.len() < spec_len {
        scratch.power_spectrum.resize(spec_len, 0.0);
    }
    if scratch.numerator_i.len() < spec_len {
        scratch.numerator_i.resize(spec_len, 0.0);
    }
    get_spectra(x, x.len(), base_time_length, scratch);
    for j in 0..spec_len {
        let ms = scratch.main_spectrum[j];
        let ds = scratch.diff_spectrum[j];
        scratch.numerator_i[j] = ms.re * ds.im - ms.im * ds.re;
        scratch.power_spectrum[j] = ms.re * ms.re + ms.im * ms.im;
    }
    let harmonics_f = fs / 2.0 / current_f0;
    let number_of_harmonics = if harmonics_f > 0.0 && harmonics_f.is_finite() {
        let h = if harmonics_f > usize::MAX as f64 {
            usize::MAX
        } else {
            harmonics_f as usize
        };
        h.min(6)
    } else {
        0
    };
    fix_f0(
        fft_size,
        fs,
        current_f0,
        number_of_harmonics,
        refined_f0,
        refined_score,
        scratch,
    );
}

pub(crate) fn get_refined_f0(
    x: &[f64],
    fs: f64,
    current_position: f64,
    current_f0: f64,
    f0_floor: f64,
    f0_ceil: f64,
    refined_f0: &mut f64,
    refined_score: &mut f64,
    scratch: &mut RefineScratch,
) {
    if !current_f0.is_finite() || current_f0 <= 0.0 || !fs.is_finite() || fs <= 0.0 {
        *refined_f0 = 0.0;
        *refined_score = 0.0;
        return;
    }
    let half_window_length_f = 1.5 * fs / current_f0 + 1.0;
    if !half_window_length_f.is_finite()
        || half_window_length_f <= 0.0
        || half_window_length_f > 1e7
    {
        *refined_f0 = 0.0;
        *refined_score = 0.0;
        return;
    }
    let half_window_length = half_window_length_f as usize;
    if half_window_length == 0 {
        *refined_f0 = 0.0;
        *refined_score = 0.0;
        return;
    }
    let window_length_in_time = (2 * half_window_length + 1) as f64 / fs;
    let base_time_len = 2 * half_window_length + 1;
    let n = base_time_len as f64;
    let log2_n = n.log2();
    let exponent = 2 + log2_n.floor() as i32;
    if exponent < 0 || exponent > 30 {
        *refined_f0 = 0.0;
        *refined_score = 0.0;
        return;
    }
    let fft_size = 1usize << exponent as u32;
    // Allocation budget (mirrors the stonemask frame cap): `fs` has no upper
    // bound at the `harvest()` entry, so a degenerate rate (e.g. `fs ≈ 1e8`
    // with a low f0) would plan ~160MB time vecs plus a 2^26 FFT per frame.
    // Bail to unvoiced instead of attempting a multi-GB allocation.
    if base_time_len > MAX_REFINE_BASE_TIME || fft_size > MAX_REFINE_FFT {
        *refined_f0 = 0.0;
        *refined_score = 0.0;
        return;
    }
    if scratch.base_time.len() < base_time_len {
        scratch.base_time.resize(base_time_len, 0.0);
    }
    for i in 0..base_time_len {
        scratch.base_time[i] = (-(half_window_length as isize) + i as isize) as f64 / fs;
    }
    get_mean_f0(
        x,
        fs,
        current_position,
        current_f0,
        fft_size,
        window_length_in_time,
        base_time_len,
        refined_f0,
        refined_score,
        scratch,
    );
    if *refined_f0 < f0_floor || *refined_f0 > f0_ceil || *refined_score < 2.5 {
        *refined_f0 = 0.0;
        *refined_score = 0.0;
    }
}

pub(crate) fn refine_f0_candidates(
    x: &[f64],
    fs: f64,
    temporal_positions: &[f64],
    f0_length: usize,
    max_candidates: usize,
    f0_floor: f64,
    f0_ceil: f64,
    refined_f0_candidates: &mut [Vec<f64>],
    f0_scores: &mut [Vec<f64>],
    scratch: &mut RefineScratch,
) {
    let len = f0_length
        .min(temporal_positions.len())
        .min(refined_f0_candidates.len())
        .min(f0_scores.len());
    for i in 0..len {
        let tpos = temporal_positions[i];
        let row_cand = &mut refined_f0_candidates[i];
        let row_score = &mut f0_scores[i];
        let n = max_candidates.min(row_cand.len()).min(row_score.len());
        for j in 0..n {
            let current_f0 = row_cand[j];
            let mut refined = 0.0;
            let mut score = 0.0;
            get_refined_f0(
                x,
                fs,
                tpos,
                current_f0,
                f0_floor,
                f0_ceil,
                &mut refined,
                &mut score,
                scratch,
            );
            row_cand[j] = refined;
            row_score[j] = score;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refined_f0_pure_sine_non_degenerate() {
        // T4 requirement: refinement returns a usable estimate on voiced
        // input. NOTE: a *pure* sine legitimately scores below the 2.5 gate
        // (C++ GetRefinedF0 zeroes it too — pure-tone pathology, same reason
        // the old sine fixtures failed). Use a harmonic complex, on which
        // C++ returns refined≈200.0, score≈7.5.
        let fs = 16000.0;
        let x: Vec<f64> = (0..32000)
            .map(|i| {
                let t = i as f64 / fs;
                (1..=4)
                    .map(|k| {
                        0.5 / k as f64 * (2.0 * std::f64::consts::PI * 200.0 * k as f64 * t).sin()
                    })
                    .sum()
            })
            .collect();
        let mut refined = 0.0;
        let mut score = 0.0;
        let mut scratch = RefineScratch::new();
        get_refined_f0(
            &x,
            fs,
            1.0,
            200.0,
            71.0,
            800.0,
            &mut refined,
            &mut score,
            &mut scratch,
        );
        assert!(
            (150.0..=250.0).contains(&refined),
            "refined {refined} outside loose band (score {score})"
        );
        assert!(score >= 2.5, "score {score} below C++ gate");
    }
}
