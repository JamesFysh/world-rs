use crate::common::nuttall_window;
use crate::dio::{check_event, get_four_zero_crossing_intervals, DioZcScratch};
use crate::fft::{FftComplex, ForwardRealFFT, InverseRealFFT};
use crate::matlab::{decimate, interp1_into, matlab_round};
use rustfft::num_complex::Complex;
use std::f64::consts::PI;

pub(crate) struct HarvestChannelScratch {
    pub filtered_signal: Vec<f64>,
    pub zc: DioZcScratch,
    pub interp_buf0: Vec<f64>,
    pub interp_buf1: Vec<f64>,
    pub interp_buf2: Vec<f64>,
    pub interp_buf3: Vec<f64>,
    pub h: Vec<f64>,
    pub k: Vec<i32>,
}

impl HarvestChannelScratch {
    pub fn new(fft_size: usize, y_length: usize, f0_length: usize) -> Self {
        let mut zc = DioZcScratch {
            negative_interval_locations: Vec::with_capacity(y_length),
            negative_intervals: Vec::with_capacity(y_length),
            positive_interval_locations: Vec::with_capacity(y_length),
            positive_intervals: Vec::with_capacity(y_length),
            peak_interval_locations: Vec::with_capacity(y_length),
            peak_intervals: Vec::with_capacity(y_length),
            dip_interval_locations: Vec::with_capacity(y_length),
            dip_intervals: Vec::with_capacity(y_length),
            edges: Vec::with_capacity(y_length),
            fine_edges: Vec::with_capacity(y_length),
            number_of_negatives: 0,
            number_of_positives: 0,
            number_of_peaks: 0,
            number_of_dips: 0,
        };
        // Pre-allocate to avoid first-use reallocations
        zc.negative_interval_locations.resize(y_length, 0.0);
        zc.negative_intervals.resize(y_length, 0.0);
        zc.positive_interval_locations.resize(y_length, 0.0);
        zc.positive_intervals.resize(y_length, 0.0);
        zc.peak_interval_locations.resize(y_length, 0.0);
        zc.peak_intervals.resize(y_length, 0.0);
        zc.dip_interval_locations.resize(y_length, 0.0);
        zc.dip_intervals.resize(y_length, 0.0);
        zc.edges.resize(y_length, 0);
        zc.fine_edges.resize(y_length, 0.0);

        Self {
            filtered_signal: vec![0.0; fft_size],
            zc,
            interp_buf0: vec![0.0; f0_length],
            interp_buf1: vec![0.0; f0_length],
            interp_buf2: vec![0.0; f0_length],
            interp_buf3: vec![0.0; f0_length],
            h: vec![0.0; y_length],
            k: vec![0; f0_length],
        }
    }
}

#[allow(dead_code, clippy::needless_range_loop)]
pub(crate) fn get_waveform_and_spectrum_sub(
    x: &[f64],
    y_length: usize,
    decimation_ratio: usize,
    y: &mut [f64],
) {
    let x_len = x.len();
    if decimation_ratio == 0 {
        return;
    }
    if decimation_ratio == 1 {
        let copy_len = y_length.min(x_len).min(y.len());
        if copy_len > 0 {
            y[..copy_len].copy_from_slice(&x[..copy_len]);
        }
        return;
    }
    if x_len == 0 {
        return;
    }
    let lag_f = (140.0 / decimation_ratio as f64).ceil() as usize;
    let lag = lag_f * decimation_ratio;
    let new_x_len = x_len + lag * 2;
    let mut new_x = vec![0.0f64; new_x_len];
    // edge pad
    new_x[..lag].fill(x[0]);
    new_x[lag..lag + x_len].copy_from_slice(x);
    new_x[lag + x_len..].fill(x[x_len - 1]);
    let mut new_y = vec![0.0f64; new_x_len];
    decimate(&new_x, decimation_ratio, &mut new_y);
    let offset = lag / decimation_ratio;
    let copy_len = y_length
        .min(y.len())
        .min(new_y.len().saturating_sub(offset));
    for i in 0..copy_len {
        let idx = offset + i;
        if idx < new_y.len() {
            y[i] = new_y[idx];
        } else {
            y[i] = 0.0;
        }
    }
}

#[allow(dead_code, clippy::needless_range_loop)]
pub(crate) fn get_waveform_and_spectrum(
    x: &[f64],
    _x_length: usize,
    y_length: usize,
    fft_size: usize,
    decimation_ratio: usize,
    forward: &mut ForwardRealFFT,
    y_out: Option<&mut [f64]>,
) {
    debug_assert_eq!(forward.fft_size, fft_size);
    forward.waveform.fill(0.0);
    // fill waveform with decimated signal
    get_waveform_and_spectrum_sub(x, y_length, decimation_ratio, &mut forward.waveform);
    if y_length > 0 && y_length <= forward.waveform.len() {
        let sum: f64 = forward.waveform[..y_length].iter().copied().sum();
        let mean = sum / y_length as f64;
        for i in 0..y_length {
            forward.waveform[i] -= mean;
        }
    }
    // zero tail already zero
    // Snapshot the DC-removed decimated waveform before `forward()`
    // overwrites its input buffer (callers needing `y` for refinement).
    if let Some(y) = y_out {
        let len = y_length.min(y.len()).min(forward.waveform.len());
        y[..len].copy_from_slice(&forward.waveform[..len]);
    }
    forward.forward();
}

#[allow(dead_code, clippy::needless_range_loop, clippy::too_many_arguments)]
pub(crate) fn get_filtered_signal_harvest(
    boundary_f0: f64,
    fft_size: usize,
    fs: f64,
    y_spectrum: &[FftComplex],
    y_length: usize,
    forward: &mut ForwardRealFFT,
    inverse: &mut InverseRealFFT,
    filtered_signal: &mut [f64],
) {
    if boundary_f0 <= 0.0 || fs <= 0.0 || fft_size == 0 {
        filtered_signal.fill(0.0);
        return;
    }
    let filter_length_half = matlab_round(fs / boundary_f0 * 2.0) as usize;
    let filter_len = filter_length_half * 2 + 1;
    forward.waveform.fill(0.0);
    if filter_len <= fft_size {
        nuttall_window(filter_len, &mut forward.waveform[..filter_len]);
        for i in 0..filter_len {
            let k = i as isize - filter_length_half as isize;
            let phase = 2.0 * PI * boundary_f0 * k as f64 / fs;
            forward.waveform[i] *= phase.cos();
        }
    }
    forward.forward();
    let half = fft_size / 2;
    for i in 0..=half {
        let y_c = if i < y_spectrum.len() {
            Complex::new(y_spectrum[i][0], y_spectrum[i][1])
        } else {
            Complex::new(0.0, 0.0)
        };
        let f_c = if i < forward.spectrum.len() {
            forward.spectrum[i]
        } else {
            Complex::new(0.0, 0.0)
        };
        inverse.spectrum[i] = y_c * f_c;
    }
    inverse.inverse();
    let index_bias = filter_length_half + 1;
    let copy_len = y_length.min(filtered_signal.len());
    for i in 0..copy_len {
        let src = i + index_bias;
        if src < inverse.waveform.len() {
            filtered_signal[i] = inverse.waveform[src];
        } else {
            filtered_signal[i] = 0.0;
        }
    }
}

//-----------------------------------------------------------------------------
// GetF0CandidateContourSub
//-----------------------------------------------------------------------------
#[allow(dead_code, clippy::needless_range_loop)]
pub(crate) fn get_f0_candidate_contour_sub_harvest(
    interpolated_f0_set: [&[f64]; 4],
    f0_length: usize,
    f0_floor: f64,
    f0_ceil: f64,
    boundary_f0: f64,
    f0_candidate: &mut [f64],
) {
    let upper = boundary_f0 * 1.1;
    let lower = boundary_f0 * 0.9;
    for i in 0..f0_length {
        let v0 = interpolated_f0_set[0].get(i).copied().unwrap_or(0.0);
        let v1 = interpolated_f0_set[1].get(i).copied().unwrap_or(0.0);
        let v2 = interpolated_f0_set[2].get(i).copied().unwrap_or(0.0);
        let v3 = interpolated_f0_set[3].get(i).copied().unwrap_or(0.0);
        let avg = (v0 + v1 + v2 + v3) / 4.0;
        if avg > upper || avg < lower || avg > f0_ceil || avg < f0_floor {
            f0_candidate[i] = 0.0;
        } else {
            f0_candidate[i] = avg;
        }
    }
}

//-----------------------------------------------------------------------------
// GetF0CandidateContour
//-----------------------------------------------------------------------------
#[allow(dead_code, clippy::needless_range_loop, clippy::too_many_arguments)]
pub(crate) fn get_f0_candidate_contour_harvest(
    boundary_f0: f64,
    f0_floor: f64,
    f0_ceil: f64,
    temporal_positions: &[f64],
    f0_length: usize,
    f0_candidate: &mut [f64],
    scratch: &mut HarvestChannelScratch,
) {
    let zc = &scratch.zc;
    let n_neg = zc.number_of_negatives;
    let n_pos = zc.number_of_positives;
    let n_peak = zc.number_of_peaks;
    let n_dip = zc.number_of_dips;
    if check_event(n_neg - 2)
        * check_event(n_pos - 2)
        * check_event(n_peak - 2)
        * check_event(n_dip - 2)
        == 0
    {
        for i in 0..f0_length {
            f0_candidate[i] = 0.0;
        }
        return;
    }

    let n_neg_usize = n_neg.max(0) as usize;
    let n_pos_usize = n_pos.max(0) as usize;
    let n_peak_usize = n_peak.max(0) as usize;
    let n_dip_usize = n_dip.max(0) as usize;

    let x_neg = &zc.negative_interval_locations[..n_neg_usize];
    let y_neg = &zc.negative_intervals[..n_neg_usize];
    let x_pos = &zc.positive_interval_locations[..n_pos_usize];
    let y_pos = &zc.positive_intervals[..n_pos_usize];
    let x_peak = &zc.peak_interval_locations[..n_peak_usize];
    let y_peak = &zc.peak_intervals[..n_peak_usize];
    let x_dip = &zc.dip_interval_locations[..n_dip_usize];
    let y_dip = &zc.dip_intervals[..n_dip_usize];

    interp1_into(
        x_neg,
        y_neg,
        temporal_positions,
        &mut scratch.interp_buf0,
        &mut scratch.h,
        &mut scratch.k,
    );
    interp1_into(
        x_pos,
        y_pos,
        temporal_positions,
        &mut scratch.interp_buf1,
        &mut scratch.h,
        &mut scratch.k,
    );
    interp1_into(
        x_peak,
        y_peak,
        temporal_positions,
        &mut scratch.interp_buf2,
        &mut scratch.h,
        &mut scratch.k,
    );
    interp1_into(
        x_dip,
        y_dip,
        temporal_positions,
        &mut scratch.interp_buf3,
        &mut scratch.h,
        &mut scratch.k,
    );

    get_f0_candidate_contour_sub_harvest(
        [
            &scratch.interp_buf0[..],
            &scratch.interp_buf1[..],
            &scratch.interp_buf2[..],
            &scratch.interp_buf3[..],
        ],
        f0_length,
        f0_floor,
        f0_ceil,
        boundary_f0,
        f0_candidate,
    );
}

//-----------------------------------------------------------------------------
// GetF0CandidateFromRawEvent
//-----------------------------------------------------------------------------
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn get_f0_candidate_from_raw_event_harvest(
    boundary_f0: f64,
    fs: f64,
    y_spectrum: &[FftComplex],
    y_length: usize,
    fft_size: usize,
    f0_floor: f64,
    f0_ceil: f64,
    temporal_positions: &[f64],
    f0_length: usize,
    f0_candidate: &mut [f64],
    forward: &mut ForwardRealFFT,
    inverse: &mut InverseRealFFT,
    scratch: &mut HarvestChannelScratch,
) {
    get_filtered_signal_harvest(
        boundary_f0,
        fft_size,
        fs,
        y_spectrum,
        y_length,
        forward,
        inverse,
        &mut scratch.filtered_signal[..fft_size],
    );

    get_four_zero_crossing_intervals(&mut scratch.filtered_signal, y_length, fs, &mut scratch.zc);

    get_f0_candidate_contour_harvest(
        boundary_f0,
        f0_floor,
        f0_ceil,
        temporal_positions,
        f0_length,
        f0_candidate,
        scratch,
    );
}

//-----------------------------------------------------------------------------
// GetRawF0Candidates
//-----------------------------------------------------------------------------
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn get_raw_f0_candidates(
    boundary_f0_list: &[f64],
    number_of_bands: usize,
    actual_fs: f64,
    y_length: usize,
    temporal_positions: &[f64],
    f0_length: usize,
    y_spectrum: &[FftComplex],
    fft_size: usize,
    f0_floor: f64,
    f0_ceil: f64,
    raw_f0_candidates: &mut [Vec<f64>],
    forward: &mut ForwardRealFFT,
    inverse: &mut InverseRealFFT,
    scratch: &mut HarvestChannelScratch,
) {
    for i in 0..number_of_bands {
        let boundary = boundary_f0_list[i];
        let cand = &mut raw_f0_candidates[i];
        if cand.len() < f0_length {
            cand.resize(f0_length, 0.0);
        }
        get_f0_candidate_from_raw_event_harvest(
            boundary,
            actual_fs,
            y_spectrum,
            y_length,
            fft_size,
            f0_floor,
            f0_ceil,
            temporal_positions,
            f0_length,
            &mut cand[..f0_length],
            forward,
            inverse,
            scratch,
        );
    }
}

//-----------------------------------------------------------------------------
// DetectOfficialF0CandidatesSub1
//-----------------------------------------------------------------------------
#[allow(dead_code)]
pub(crate) fn detect_official_f0_candidates_sub1(
    vuv: &[i32],
    st: &mut [i32],
    ed: &mut [i32],
) -> usize {
    let number_of_channels = vuv.len();
    let mut number_of_voiced_sections = 0usize;
    for i in 1..number_of_channels {
        let tmp = vuv[i] - vuv[i - 1];
        if tmp == 1 {
            st[number_of_voiced_sections] = i as i32;
        }
        if tmp == -1 {
            ed[number_of_voiced_sections] = i as i32;
            number_of_voiced_sections += 1;
        }
    }
    number_of_voiced_sections
}

//-----------------------------------------------------------------------------
// DetectOfficialF0CandidatesSub2
//-----------------------------------------------------------------------------
#[allow(dead_code, clippy::too_many_arguments, clippy::needless_range_loop)]
pub(crate) fn detect_official_f0_candidates_sub2(
    _vuv: &[i32],
    raw_f0_candidates: &[&[f64]],
    index: usize,
    number_of_voiced_sections: usize,
    st: &[i32],
    ed: &[i32],
    max_candidates: usize,
    f0_list: &mut [f64],
) -> usize {
    let mut number_of_candidates = 0usize;
    for i in 0..number_of_voiced_sections {
        let s = st[i] as usize;
        let e = ed[i] as usize;
        if e - s < 10 {
            continue;
        }
        let mut tmp_f0 = 0.0;
        for j in s..e {
            if let Some(row) = raw_f0_candidates.get(j) {
                if let Some(v) = row.get(index) {
                    tmp_f0 += *v;
                }
            }
        }
        let cnt = (e - s) as f64;
        if cnt > 0.0 {
            tmp_f0 /= cnt;
        }
        f0_list[number_of_candidates] = tmp_f0;
        number_of_candidates += 1;
    }
    for i in number_of_candidates..max_candidates {
        f0_list[i] = 0.0;
    }
    number_of_candidates
}

//-----------------------------------------------------------------------------
// DetectOfficialF0Candidates
//-----------------------------------------------------------------------------
#[allow(dead_code, clippy::needless_range_loop)]
pub(crate) fn detect_official_f0_candidates(
    raw_f0_candidates: &[Vec<f64>],
    number_of_channels: usize,
    f0_length: usize,
    max_candidates: usize,
    f0_candidates: &mut [Vec<f64>],
) -> usize {
    let mut number_of_candidates = 0usize;
    let mut vuv = vec![0i32; number_of_channels];
    let mut st = vec![0i32; number_of_channels];
    let mut ed = vec![0i32; number_of_channels];
    let rows: Vec<&[f64]> = raw_f0_candidates.iter().map(|r| r.as_slice()).collect();

    for i in 0..f0_length {
        for j in 0..number_of_channels {
            let v = raw_f0_candidates[j].get(i).copied().unwrap_or(0.0);
            vuv[j] = if v > 0.0 { 1 } else { 0 };
        }
        if number_of_channels > 0 {
            vuv[0] = 0;
            vuv[number_of_channels - 1] = 0;
        }
        let number_of_voiced_sections = detect_official_f0_candidates_sub1(&vuv, &mut st, &mut ed);
        let f0_list = &mut f0_candidates[i];
        if f0_list.len() < max_candidates {
            f0_list.resize(max_candidates, 0.0);
        }
        let n = detect_official_f0_candidates_sub2(
            &vuv,
            &rows,
            i,
            number_of_voiced_sections,
            &st,
            &ed,
            max_candidates,
            &mut f0_list[..max_candidates],
        );
        if n > number_of_candidates {
            number_of_candidates = n;
        }
    }

    debug_assert!(number_of_candidates <= max_candidates, "N0 bound violated");
    number_of_candidates
}

//-----------------------------------------------------------------------------
// OverlapF0Candidates
//-----------------------------------------------------------------------------
#[allow(dead_code, clippy::needless_range_loop)]
pub(crate) fn overlap_f0_candidates(
    f0_length: usize,
    number_of_candidates: usize,
    f0_candidates: &mut [Vec<f64>],
) {
    let n = 3usize;
    let width = number_of_candidates;
    for i in 1..=n {
        for j in 0..number_of_candidates {
            let col_src = j;
            let col_dst = j + width * i;
            for k in i..f0_length {
                let src = f0_candidates[k - i].get(col_src).copied().unwrap_or(0.0);
                if let Some(row) = f0_candidates.get_mut(k) {
                    if col_dst < row.len() {
                        row[col_dst] = src;
                    }
                }
            }
            let col_dst2 = j + width * (i + n);
            for k in 0..f0_length.saturating_sub(i) {
                let src = f0_candidates[k + i].get(col_src).copied().unwrap_or(0.0);
                if let Some(row) = f0_candidates.get_mut(k) {
                    if col_dst2 < row.len() {
                        row[col_dst2] = src;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft::ForwardRealFFT;

    #[test]
    fn harvest_candidate_bound_guard() {
        // Pathologically alternating vuv pattern to stress N0 bound.
        // number_of_channels = 40 -> max_candidates = round(40/10)*7 = 28
        // With <10 frame skip, N0 <= floor(40/10) = 4, so 7*N0 <= 28.
        let number_of_channels = 40usize;
        let f0_length = 100usize;
        let max_candidates = ((number_of_channels as f64) / 10.0).round() as usize * 7;
        // Build raw candidates where every channel alternates voiced/unvoiced
        // to create many short sections (<10 frames) which must be skipped.
        let mut raw = vec![vec![0.0f64; f0_length]; number_of_channels];
        for (ch, row) in raw.iter_mut().enumerate() {
            for (i, item) in row.iter_mut().enumerate().take(f0_length) {
                // Alternate every frame, creating sections of length 1
                if (i + ch) % 2 == 0 {
                    *item = 200.0 + ch as f64;
                }
            }
        }
        let mut f0_candidates = vec![vec![0.0f64; max_candidates]; f0_length];
        let n0 = detect_official_f0_candidates(
            &raw,
            number_of_channels,
            f0_length,
            max_candidates,
            &mut f0_candidates,
        );
        let bound = number_of_channels / 10;
        assert!(
            n0 <= bound,
            "N0 {} exceeds floor(channels/10) {}",
            n0,
            bound
        );
        assert!(n0 * 7 <= max_candidates, "7*N0 exceeds max_candidates");
    }

    #[test]
    fn waveform_spectrum_sizes_15k_8k_48k() {
        // Spot-check sizes/offsets against C++ formulas
        for &(_fs, ratio) in &[(15000.0, 2), (8000.0, 1), (48000.0, 6)] {
            let x_len = 1024usize;
            let x = vec![1.0; x_len];
            let y_length = ((x_len as f64) / ratio as f64).ceil() as usize;
            let fft_size = 1024;
            let mut forward = ForwardRealFFT::new(fft_size);
            get_waveform_and_spectrum(&x, x_len, y_length, fft_size, ratio, &mut forward, None);
            assert_eq!(forward.waveform.len(), fft_size);
            assert_eq!(forward.spectrum.len(), fft_size / 2 + 1);
            // lag computation sanity
            if ratio != 1 {
                let lag_f = (140.0 / ratio as f64).ceil() as usize;
                let lag = lag_f * ratio;
                let offset = lag / ratio;
                assert!(offset > 0);
            }
        }
    }
}
