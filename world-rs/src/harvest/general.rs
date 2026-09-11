use crate::common::get_suitable_fft_size;
use crate::fft::{FftComplex, ForwardRealFFT, InverseRealFFT};
use crate::harvest::candidates::{
    detect_official_f0_candidates, get_raw_f0_candidates, overlap_f0_candidates,
    HarvestChannelScratch,
};
use crate::harvest::postprocess::remove_unreliable_candidates;
use crate::harvest::postprocess::{fix_f0_contour, smooth_f0_contour};
use crate::harvest::refine::{refine_f0_candidates, RefineScratch};
use crate::matlab::matlab_round;
use std::f64::consts::LN_2;

#[allow(clippy::too_many_arguments)]
pub(crate) fn harvest_general_body_sub(
    boundary_f0_list: &[f64],
    number_of_channels: usize,
    f0_length: usize,
    actual_fs: f64,
    y_length: usize,
    temporal_positions: &[f64],
    y_spectrum: &[FftComplex],
    fft_size: usize,
    f0_floor: f64,
    f0_ceil: f64,
    max_candidates: usize,
    f0_candidates: &mut [Vec<f64>],
    forward: &mut ForwardRealFFT,
    inverse: &mut InverseRealFFT,
) -> usize {
    let mut raw_f0_candidates: Vec<Vec<f64>> = vec![vec![0.0; f0_length]; number_of_channels];
    let mut scratch = HarvestChannelScratch::new(fft_size, y_length, f0_length);
    get_raw_f0_candidates(
        boundary_f0_list,
        number_of_channels,
        actual_fs,
        y_length,
        temporal_positions,
        f0_length,
        y_spectrum,
        fft_size,
        f0_floor,
        f0_ceil,
        &mut raw_f0_candidates,
        forward,
        inverse,
        &mut scratch,
    );
    let number_of_candidates = detect_official_f0_candidates(
        &raw_f0_candidates,
        number_of_channels,
        f0_length,
        max_candidates,
        f0_candidates,
    );
    overlap_f0_candidates(f0_length, number_of_candidates, f0_candidates);
    debug_assert!(
        number_of_candidates * 7 <= max_candidates,
        "7*N0 bound violated"
    );
    number_of_candidates
}

pub(crate) fn harvest_general_body(
    x: &[f64],
    fs: f64,
    frame_period: f64,
    f0_floor: f64,
    f0_ceil: f64,
    temporal_positions: &mut [f64],
    f0: &mut [f64],
) {
    let adjusted_f0_floor: f64 = f0_floor * 0.9;
    let adjusted_f0_ceil: f64 = f0_ceil * 1.1;
    let channels_in_octave: f64 = 40.0;
    let log_ratio = (adjusted_f0_ceil / adjusted_f0_floor).ln() / LN_2;
    let _ = (adjusted_f0_floor, adjusted_f0_ceil);
    let number_of_channels = 1 + (log_ratio * channels_in_octave).floor() as usize;
    let mut boundary_f0_list = vec![0.0f64; number_of_channels];
    for (i, b) in boundary_f0_list.iter_mut().enumerate() {
        *b = adjusted_f0_floor * 2f64.powf((i as f64 + 1.0) / channels_in_octave);
    }

    let dimension_ratio = matlab_round(fs / 8000.0) as i32;
    let speed = dimension_ratio.clamp(1, 12) as usize;
    let decimation_ratio = speed;
    let x_length = x.len();
    let y_length = ((x_length as f64) / decimation_ratio as f64).ceil() as usize;
    let actual_fs = fs / decimation_ratio as f64;
    let boundary0 = boundary_f0_list[0].max(1e-9);
    let fft_size_needed = y_length + 5 + 2 * ((2.0 * actual_fs / boundary0) as usize);
    let fft_size = get_suitable_fft_size(fft_size_needed);

    let mut forward = ForwardRealFFT::new(fft_size);
    let mut inverse = InverseRealFFT::new(fft_size);
    // Get waveform and spectrum. The DC-removed decimated waveform `y` is
    // snapshotted out (pre-FFT) for the later RefineF0Candidates call
    // (harvest.cpp:1194) instead of running the decimation twice: `forward()`
    // overwrites its input buffer, so the snapshot must happen inside.
    let mut y_waveform = vec![0.0f64; y_length];
    crate::harvest::candidates::get_waveform_and_spectrum(
        x,
        x_length,
        y_length,
        fft_size,
        decimation_ratio,
        &mut forward,
        Some(&mut y_waveform),
    );
    let y_spectrum: Vec<FftComplex> = forward.spectrum.iter().map(|c| [c.re, c.im]).collect();

    let f0_length = crate::harvest::get_samples_for_harvest(fs, x_length, frame_period);
    for i in 0..f0_length.min(temporal_positions.len()) {
        temporal_positions[i] = i as f64 * frame_period / 1000.0;
        if i < f0.len() {
            f0[i] = 0.0;
        }
    }

    let overlap_parameter = 7usize;
    let max_candidates = ((number_of_channels as f64) / 10.0).round() as usize * overlap_parameter;
    let mut f0_candidates: Vec<Vec<f64>> = vec![vec![0.0; max_candidates]; f0_length];
    let mut f0_scores: Vec<Vec<f64>> = vec![vec![0.0; max_candidates]; f0_length];

    let number_of_candidates = harvest_general_body_sub(
        &boundary_f0_list,
        number_of_channels,
        f0_length,
        actual_fs,
        y_length,
        &temporal_positions[..f0_length],
        &y_spectrum,
        fft_size,
        f0_floor,
        f0_ceil,
        max_candidates,
        &mut f0_candidates,
        &mut forward,
        &mut inverse,
    ) * overlap_parameter;

    let mut refine_scratch = RefineScratch::new();
    refine_f0_candidates(
        &y_waveform,
        actual_fs,
        &temporal_positions[..f0_length],
        f0_length,
        max_candidates,
        f0_floor,
        f0_ceil,
        &mut f0_candidates,
        &mut f0_scores,
        &mut refine_scratch,
    );
    remove_unreliable_candidates(
        f0_length,
        number_of_candidates,
        &mut f0_candidates,
        &mut f0_scores,
    );

    let mut best_f0_contour = vec![0.0f64; f0_length];
    fix_f0_contour(
        &f0_candidates,
        &f0_scores,
        f0_length,
        number_of_candidates,
        &mut best_f0_contour,
    );
    smooth_f0_contour(&best_f0_contour, f0_length, &mut f0[..f0_length]);
}
