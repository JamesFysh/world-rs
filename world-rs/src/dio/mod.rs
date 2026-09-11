//! DIO pitch extraction (port of `ext_src/world-cpp/src/dio.cpp`).
//!
//! Phase 1-1: module scaffolding and type definitions.
//! Phase 1-2: spectrum estimation and core helpers
//! (`design_low_cut_filter`, `get_spectrum_for_estimation`).
//! Phase 1-3: zero-crossing engine and F0 candidate extraction
//! (`ZeroCrossings`, `get_filtered_signal`, `get_f0_candidate_from_raw_event`).
//! Phase 1-4: F0 candidate scoring and multi-band processing
//! (`get_number_of_bands`, `get_boundary_f0_list`, `get_decimation_ratio`,
//! `get_dio_fft_size`, `get_f0_candidates_and_scores`, `get_best_f0_contour`).
//! Phase 1-5: post-processing and F0 contour fixing
//! (`fix_step1`, `fix_step2`, `get_number_of_voiced_sections`,
//! `select_best_f0`, `fix_step3`, `fix_step4`, `fix_f0_contour`).
//! Phase 1-6: general body and public API
//! (`dio_general_body`, `dio`, `DioResult`).
//! Phase 1-7: unit tests & edge-case validation; `dio` validates its inputs and
//! returns `Result<DioResult, DioError>` (the `DioError` type is new).

use crate::constants::{
    K_CEIL_F0, K_CUT_OFF, K_FLOOR_F0, K_LOG2, K_MAXIMUM_VALUE, K_MY_SAFE_GUARD_MINIMUM, K_PI,
};
use crate::fft::FftComplex;
use crate::{common, fft, matlab};
use rustfft::num_complex::Complex;

/// Allocation budget for one `dio()` call: at most `MAX_DIO_CELLS`
/// `bands × frames` f64 cells (≈1GB for the two estimator matrices) and at
/// most `MAX_DIO_FRAMES` frames (≈192MB of contour vecs). Plans beyond this
/// fail with [`DioError::TooManyFrames`] before allocating.
const MAX_DIO_CELLS: u64 = 64_000_000;
const MAX_DIO_FRAMES: usize = 8_000_000;

/// C++: `DioOption` (world/dio.h:16-23). Field order matches the C++ struct.
#[derive(Debug, Clone, PartialEq)]
pub struct DioOption {
    pub f0_floor: f64,
    pub f0_ceil: f64,
    pub channels_in_octave: f64,
    /// Frame period in msec.
    pub frame_period: f64,
    /// Speed control, valid range (1, 2, ..., 12).
    pub speed: i32,
    /// Threshold used for fixing the F0 contour.
    pub allowed_range: f64,
}

/// C++: `InitializeDioOption` (dio.cpp:650-666).
///
/// Returns the default parameters matching the C++ reference.
/// Java cross-ref: `ConstantNumbers.framePeriod = 5.0` matches the C++
/// hardcoded default `option->frame_period = 5;` (dio.cpp:655).
pub fn initialize_dio_option() -> DioOption {
    DioOption {
        f0_floor: K_FLOOR_F0,
        f0_ceil: K_CEIL_F0,
        channels_in_octave: 2.0,
        frame_period: 5.0,
        speed: 1,
        allowed_range: 0.1,
    }
}

/// C++: `GetSamplesForDIO` (dio.cpp:639-641).
///
/// Number of samples required to store the results of `dio()`.
///
/// Total: returns `0` for non-finite or non-positive `fs`/`frame_period`, and
/// saturates on overflow, so it never panics for any input.
pub fn get_samples_for_dio(fs: f64, x_length: usize, frame_period: f64) -> usize {
    if !fs.is_finite() || fs <= 0.0 || !frame_period.is_finite() || frame_period <= 0.0 {
        return 0;
    }
    let v = 1000.0 * x_length as f64 / fs / frame_period;
    if !v.is_finite() {
        return 0;
    }
    let n = v as usize;
    n.saturating_add(1)
}

/// C++: `DesignLowCutFilter` (dio.cpp:40-53).
///
/// Designs the low-cut filter coefficients in place. `low_cut_filter` must
/// have length `fft_size` and `n <= fft_size` (the C++ caller always passes
/// `cutoff_in_sample * 2 + 1` below the DIO `fft_size`; the C++ code is
/// out-of-bounds otherwise).
///
/// # Panics
///
/// In debug builds, panics if `low_cut_filter.len() != fft_size` or if
/// `n > fft_size`.
pub(crate) fn design_low_cut_filter(n: usize, fft_size: usize, low_cut_filter: &mut [f64]) {
    debug_assert_eq!(low_cut_filter.len(), fft_size);
    debug_assert!(n <= fft_size);
    for i in 1..=n {
        low_cut_filter[i - 1] = 0.5 - 0.5 * ((i as f64) * 2.0 * K_PI / (n as f64 + 1.0)).cos();
    }
    low_cut_filter[n..].fill(0.0);
    let sum_of_amplitude: f64 = low_cut_filter[..n].iter().sum();
    for slot in low_cut_filter[..n].iter_mut() {
        *slot = -*slot / sum_of_amplitude;
    }
    for i in 0..(n - 1) / 2 {
        low_cut_filter[fft_size - (n - 1) / 2 + i] = low_cut_filter[i];
    }
    // In-place right shift: every read (`i + (n-1)/2`) is ahead of the write
    // (`i`), so the source values are always the original ones (as in C++).
    for i in 0..n {
        low_cut_filter[i] = low_cut_filter[i + (n - 1) / 2];
    }
    low_cut_filter[0] += 1.0;
}

/// C++: `GetSpectrumForEstimation` (dio.cpp:60-106).
///
/// Computes the low-cut-filtered half-spectrum of the (optionally downsampled,
/// DC-removed) signal `x`. `y_spectrum` must have length `fft_size`; only the
/// first `fft_size / 2 + 1` bins are written, matching the C++ output range.
/// `fft_size` must be a power of two (DIO always uses `GetSuitableFFTSize`).
///
/// The C++ reuses a single r2c `fft_plan` for the signal and filter FFTs (it
/// swaps `c_out`); the Rust port runs two independent `rustfft` plans instead.
pub(crate) fn get_spectrum_for_estimation(
    x: &[f64],
    y_length: usize,
    actual_fs: f64,
    fft_size: usize,
    decimation_ratio: usize,
    y_spectrum: &mut [FftComplex],
) {
    debug_assert_eq!(y_spectrum.len(), fft_size);
    let mut y = vec![0.0f64; fft_size];

    // Downsampling
    if decimation_ratio != 1 {
        matlab::decimate(x, decimation_ratio, &mut y);
    } else {
        y[..x.len()].copy_from_slice(x);
    }

    // Removal of the DC component (y = y - mean value of y)
    let mean_y: f64 = y[..y_length].iter().sum::<f64>() / y_length as f64;
    for slot in y[..y_length].iter_mut() {
        *slot -= mean_y;
    }
    y[y_length..].fill(0.0);

    // C++ writes the r2c output directly into `y_spectrum`; the Rust port
    // gets it from `forward_real_fft` and copies it over.
    let signal_spectrum = fft::forward_real_fft(&y);
    for (i, bin) in signal_spectrum.iter().enumerate() {
        y_spectrum[i] = [bin.re, bin.im];
    }

    // Low cut filtering (from 0.1.4). Cut off frequency is 50.0 Hz.
    let cutoff_in_sample = matlab::matlab_round(actual_fs / K_CUT_OFF) as usize;
    design_low_cut_filter(cutoff_in_sample * 2 + 1, fft_size, &mut y);
    let filter_spectrum = fft::forward_real_fft(&y);

    for i in 0..=fft_size / 2 {
        let (re, im) = (y_spectrum[i][0], y_spectrum[i][1]);
        let (filter_re, filter_im) = (filter_spectrum[i].re, filter_spectrum[i].im);
        y_spectrum[i][1] = re * filter_im + im * filter_re;
        y_spectrum[i][0] = re * filter_re - im * filter_im;
    }
}

/// C++: `ZeroCrossings` (dio.cpp:21-34). Holds the four zero-crossing
/// interval sources (negative-going, positive-going, peak, dip) as
/// location/interval arrays plus their counts.
///
/// "negative" means a zero-crossing going from positive to negative;
/// "positive" means going from negative to positive. Peaks and dips are the
/// zero-crossings of the (negated) differential of the waveform.
#[derive(Debug, Clone, Default)]
pub struct ZeroCrossings {
    pub negative_interval_locations: Vec<f64>,
    pub negative_intervals: Vec<f64>,
    pub number_of_negatives: i32,
    pub positive_interval_locations: Vec<f64>,
    pub positive_intervals: Vec<f64>,
    pub number_of_positives: i32,
    pub peak_interval_locations: Vec<f64>,
    pub peak_intervals: Vec<f64>,
    pub number_of_peaks: i32,
    pub dip_interval_locations: Vec<f64>,
    pub dip_intervals: Vec<f64>,
    pub number_of_dips: i32,
}

/// Caller-owned scratch for the zero-crossing path: the four interval
/// sources plus shared `edges`/`fine_edges` workspace and counts.
/// Sized `y_length` by the caller; only `..count` prefixes are read.
pub(crate) struct DioZcScratch {
    pub(crate) negative_interval_locations: Vec<f64>,
    pub(crate) negative_intervals: Vec<f64>,
    pub(crate) positive_interval_locations: Vec<f64>,
    pub(crate) positive_intervals: Vec<f64>,
    pub(crate) peak_interval_locations: Vec<f64>,
    pub(crate) peak_intervals: Vec<f64>,
    pub(crate) dip_interval_locations: Vec<f64>,
    pub(crate) dip_intervals: Vec<f64>,
    pub(crate) edges: Vec<i32>,
    pub(crate) fine_edges: Vec<f64>,
    pub(crate) number_of_negatives: i32,
    pub(crate) number_of_positives: i32,
    pub(crate) number_of_peaks: i32,
    pub(crate) number_of_dips: i32,
}

/// C++: `CheckEvent` (dio.cpp:349-351). Returns 1 iff `x > 0`.
#[inline]
pub(crate) fn check_event(x: i32) -> i32 {
    if x > 0 {
        1
    } else {
        0
    }
}

/// C++: `ZeroCrossingEngine` (dio.cpp:357-393). Detects negative-going
/// zero-crossings (positive to non-positive) with sub-sample linear
/// interpolation, and writes the interpolated interval lengths (`intervals`)
/// and their mid-point locations (`interval_locations`). Returns the number of
/// intervals (`count - 1`), or 0 when fewer than two crossings are found.
pub(crate) fn zero_crossing_engine(
    filtered_signal: &[f64],
    y_length: usize,
    fs: f64,
    interval_locations: &mut [f64],
    intervals: &mut [f64],
    edges: &mut [i32],
    fine_edges: &mut [f64],
) -> i32 {
    let mut edge_count = 0usize;
    for i in 0..y_length - 1 {
        if 0.0 < filtered_signal[i] && filtered_signal[i + 1] <= 0.0 {
            debug_assert!(edge_count < edges.len());
            edges[edge_count] = (i + 1) as i32;
            edge_count += 1;
        }
    }
    debug_assert!(edge_count <= y_length);

    if edge_count < 2 {
        return 0;
    }

    for i in 0..edge_count {
        let e = edges[i] as usize;
        debug_assert!(e < filtered_signal.len());
        fine_edges[i] =
            e as f64 - filtered_signal[e - 1] / (filtered_signal[e] - filtered_signal[e - 1]);
    }

    for i in 0..edge_count - 1 {
        intervals[i] = fs / (fine_edges[i + 1] - fine_edges[i]);
        interval_locations[i] = (fine_edges[i] + fine_edges[i + 1]) / 2.0 / fs;
    }

    (edge_count - 1) as i32
}

/// C++: `GetFourZeroCrossingIntervals` (dio.cpp:402-435). Computes the four
/// zero-crossing interval sources by repeatedly reusing the filtered signal:
/// negation for the positive-going crossings, then differentiation (and a
/// second negation) for the peaks and dips. The in-place mutation of
/// `filtered_signal` between passes is replicated.
pub(crate) fn get_four_zero_crossing_intervals(
    filtered_signal: &mut [f64],
    y_length: usize,
    actual_fs: f64,
    scratch: &mut DioZcScratch,
) {
    let negatives = zero_crossing_engine(
        &filtered_signal[..y_length],
        y_length,
        actual_fs,
        &mut scratch.negative_interval_locations,
        &mut scratch.negative_intervals,
        &mut scratch.edges,
        &mut scratch.fine_edges,
    );
    scratch.number_of_negatives = negatives;

    for v in filtered_signal.iter_mut().take(y_length) {
        *v = -*v;
    }
    let positives = zero_crossing_engine(
        &filtered_signal[..y_length],
        y_length,
        actual_fs,
        &mut scratch.positive_interval_locations,
        &mut scratch.positive_intervals,
        &mut scratch.edges,
        &mut scratch.fine_edges,
    );
    scratch.number_of_positives = positives;

    for i in 0..y_length - 1 {
        filtered_signal[i] -= filtered_signal[i + 1];
    }
    let peaks = zero_crossing_engine(
        &filtered_signal[..y_length - 1],
        y_length - 1,
        actual_fs,
        &mut scratch.peak_interval_locations,
        &mut scratch.peak_intervals,
        &mut scratch.edges,
        &mut scratch.fine_edges,
    );
    scratch.number_of_peaks = peaks;

    for v in filtered_signal.iter_mut().take(y_length - 1) {
        *v = -*v;
    }
    let dips = zero_crossing_engine(
        &filtered_signal[..y_length - 1],
        y_length - 1,
        actual_fs,
        &mut scratch.dip_interval_locations,
        &mut scratch.dip_intervals,
        &mut scratch.edges,
        &mut scratch.fine_edges,
    );
    scratch.number_of_dips = dips;
}

/// C++: `GetF0CandidateContourSub` (dio.cpp:441-465). Averages the four
/// interpolated F0 sources into `f0_candidate` and scores each frame with the
/// sample standard deviation (ddof=1, i.e. the sum of squared deviations
/// divided by 3 for four sources). Out-of-range candidates are zeroed with
/// `f0_score = kMaximumValue`.
fn get_f0_candidate_contour_sub(
    interpolated_f0_set: [&[f64]; 4],
    f0_length: usize,
    f0_floor: f64,
    f0_ceil: f64,
    boundary_f0: f64,
    f0_candidate: &mut [f64],
    f0_score: &mut [f64],
) {
    for i in 0..f0_length {
        f0_candidate[i] = (interpolated_f0_set[0][i]
            + interpolated_f0_set[1][i]
            + interpolated_f0_set[2][i]
            + interpolated_f0_set[3][i])
            / 4.0;

        let d0 = interpolated_f0_set[0][i] - f0_candidate[i];
        let d1 = interpolated_f0_set[1][i] - f0_candidate[i];
        let d2 = interpolated_f0_set[2][i] - f0_candidate[i];
        let d3 = interpolated_f0_set[3][i] - f0_candidate[i];
        f0_score[i] = ((d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3) / 3.0).sqrt();

        if f0_candidate[i] > boundary_f0
            || f0_candidate[i] < boundary_f0 / 2.0
            || f0_candidate[i] > f0_ceil
            || f0_candidate[i] < f0_floor
        {
            f0_candidate[i] = 0.0;
            f0_score[i] = K_MAXIMUM_VALUE;
        }
    }
}

/// C++: `GetF0CandidateContour` (dio.cpp:471-508). Interpolates the four
/// zero-crossing interval sources onto the temporal grid (one `interp1` per
/// band) and delegates scoring to `get_f0_candidate_contour_sub`. If any
/// source has fewer than three intervals, every candidate is zeroed with
/// `f0_score = kMaximumValue`.
// C++ parity: mirrors `GetF0CandidateContour` (dio.cpp:471).
#[allow(clippy::too_many_arguments)]
fn get_f0_candidate_contour(
    zero_crossings: &DioZcScratch,
    boundary_f0: f64,
    f0_floor: f64,
    f0_ceil: f64,
    temporal_positions: &[f64],
    f0_length: usize,
    f0_candidate: &mut [f64],
    f0_score: &mut [f64],
) {
    if 0 == check_event(zero_crossings.number_of_negatives - 2)
        * check_event(zero_crossings.number_of_positives - 2)
        * check_event(zero_crossings.number_of_peaks - 2)
        * check_event(zero_crossings.number_of_dips - 2)
    {
        for i in 0..f0_length {
            f0_score[i] = K_MAXIMUM_VALUE;
            f0_candidate[i] = 0.0;
        }
        return;
    }

    let mut interpolated_f0_set: [Vec<f64>; 4] = [
        vec![0.0; f0_length],
        vec![0.0; f0_length],
        vec![0.0; f0_length],
        vec![0.0; f0_length],
    ];

    let n_neg = zero_crossings.number_of_negatives as usize;
    let n_pos = zero_crossings.number_of_positives as usize;
    let n_peak = zero_crossings.number_of_peaks as usize;
    let n_dip = zero_crossings.number_of_dips as usize;

    matlab::interp1(
        &zero_crossings.negative_interval_locations[..n_neg],
        &zero_crossings.negative_intervals[..n_neg],
        temporal_positions,
        &mut interpolated_f0_set[0],
    );
    matlab::interp1(
        &zero_crossings.positive_interval_locations[..n_pos],
        &zero_crossings.positive_intervals[..n_pos],
        temporal_positions,
        &mut interpolated_f0_set[1],
    );
    matlab::interp1(
        &zero_crossings.peak_interval_locations[..n_peak],
        &zero_crossings.peak_intervals[..n_peak],
        temporal_positions,
        &mut interpolated_f0_set[2],
    );
    matlab::interp1(
        &zero_crossings.dip_interval_locations[..n_dip],
        &zero_crossings.dip_intervals[..n_dip],
        temporal_positions,
        &mut interpolated_f0_set[3],
    );

    get_f0_candidate_contour_sub(
        [
            &interpolated_f0_set[0],
            &interpolated_f0_set[1],
            &interpolated_f0_set[2],
            &interpolated_f0_set[3],
        ],
        f0_length,
        f0_floor,
        f0_ceil,
        boundary_f0,
        f0_candidate,
        f0_score,
    );
}

/// C++: `GetFilteredSignal` (dio.cpp:296-343). Low-pass filters the spectrum
/// with a Nuttall window (length `half_average_length * 4`) and inverse-FFT's
/// the product back to the time domain. The delay is compensated by shifting
/// `half_average_length * 2` samples. Note: `design_low_cut_filter` is NOT used
/// here (it is used in `get_spectrum_for_estimation`).
///
/// Replicates the C++ in-place convolution exactly, including mirror writes to
/// `fft_size - i - 1` which corrupt bins `fft_size/2 - 1` and `fft_size/2`.
fn get_filtered_signal(
    half_average_length: usize,
    fft_size: usize,
    y_spectrum: &[FftComplex],
    y_length: usize,
    filtered_signal: &mut [f64],
) {
    let mut low_pass_filter = vec![0.0f64; fft_size];
    let window_length = half_average_length * 4;
    common::nuttall_window(window_length, &mut low_pass_filter);

    let filter_spectrum = fft::forward_real_fft(&low_pass_filter);
    let mut lpf_spectrum = vec![Complex::new(0.0, 0.0); fft_size];
    lpf_spectrum[..(fft_size / 2 + 1)].copy_from_slice(&filter_spectrum[..(fft_size / 2 + 1)]);

    {
        let tmp = y_spectrum[0][0] * lpf_spectrum[0].re - y_spectrum[0][1] * lpf_spectrum[0].im;
        lpf_spectrum[0].im =
            y_spectrum[0][0] * lpf_spectrum[0].im + y_spectrum[0][1] * lpf_spectrum[0].re;
        lpf_spectrum[0].re = tmp;
        for i in 1..=fft_size / 2 {
            let tmp = y_spectrum[i][0] * lpf_spectrum[i].re - y_spectrum[i][1] * lpf_spectrum[i].im;
            lpf_spectrum[i].im =
                y_spectrum[i][0] * lpf_spectrum[i].im + y_spectrum[i][1] * lpf_spectrum[i].re;
            lpf_spectrum[i].re = tmp;
            lpf_spectrum[fft_size - i - 1].re = lpf_spectrum[i].re;
            lpf_spectrum[fft_size - i - 1].im = lpf_spectrum[i].im;
        }
    }

    let filtered = fft::inverse_real_fft(&lpf_spectrum[..fft_size / 2 + 1], fft_size);

    let index_bias = half_average_length * 2;
    filtered_signal[..y_length].copy_from_slice(&filtered[index_bias..(y_length + index_bias)]);
}

/// C++: `GetF0CandidateFromRawEvent` (dio.cpp:527-544). Combines low-pass
/// filtering and zero-crossing analysis to produce the F0 candidate contour
/// and scores for a single band, writing directly into the output arrays.
// C++ parity: mirrors `GetF0CandidateFromRawEvent` (dio.cpp:527).
#[allow(clippy::too_many_arguments)]
pub(crate) fn get_f0_candidate_from_raw_event(
    boundary_f0: f64,
    fs: f64,
    y_spectrum: &[FftComplex],
    y_length: usize,
    fft_size: usize,
    f0_floor: f64,
    f0_ceil: f64,
    temporal_positions: &[f64],
    f0_length: usize,
    f0_score: &mut [f64],
    f0_candidate: &mut [f64],
) {
    let half_average_length = matlab::matlab_round(fs / boundary_f0 / 2.0) as usize;
    let mut filtered_signal = vec![0.0f64; fft_size];
    get_filtered_signal(
        half_average_length,
        fft_size,
        y_spectrum,
        y_length,
        &mut filtered_signal,
    );

    let mut zero_crossings = DioZcScratch {
        negative_interval_locations: vec![0.0; y_length],
        negative_intervals: vec![0.0; y_length],
        positive_interval_locations: vec![0.0; y_length],
        positive_intervals: vec![0.0; y_length],
        peak_interval_locations: vec![0.0; y_length],
        peak_intervals: vec![0.0; y_length],
        dip_interval_locations: vec![0.0; y_length],
        dip_intervals: vec![0.0; y_length],
        edges: vec![0; y_length],
        fine_edges: vec![0.0; y_length],
        number_of_negatives: 0,
        number_of_positives: 0,
        number_of_peaks: 0,
        number_of_dips: 0,
    };
    get_four_zero_crossing_intervals(&mut filtered_signal, y_length, fs, &mut zero_crossings);

    get_f0_candidate_contour(
        &zero_crossings,
        boundary_f0,
        f0_floor,
        f0_ceil,
        temporal_positions,
        f0_length,
        f0_candidate,
        f0_score,
    );
}

/// C++: `number_of_bands` (dio.cpp:582-583).
///
/// `1 + floor(ln(f0_ceil / f0_floor) / kLog2 * channels_in_octave)` where
/// `kLog2 = ln 2` (constantnumbers.h:24). The C++ uses a truncating `int` cast
/// (floor for the non-negative values that occur in practice); the Python port
/// uses `ceil`, but the Rust port must match the C++ floor.
pub(crate) fn get_number_of_bands(f0_floor: f64, f0_ceil: f64, channels_in_octave: f64) -> usize {
    let n = 1 + ((f0_ceil / f0_floor).ln() / K_LOG2 * channels_in_octave) as usize;
    n.min(256)
}

/// C++: `boundary_f0_list` (dio.cpp:584-586).
///
/// `boundary_f0_list[i] = f0_floor * 2^((i + 1) / channels_in_octave)` for
/// `i = 0..number_of_bands`.
pub(crate) fn get_boundary_f0_list(
    f0_floor: f64,
    channels_in_octave: f64,
    number_of_bands: usize,
) -> Vec<f64> {
    (0..number_of_bands)
        .map(|i| f0_floor * 2.0f64.powf((i as f64 + 1.0) / channels_in_octave))
        .collect()
}

/// C++: `decimation_ratio` (dio.cpp:589).
///
/// `MyMaxInt(MyMinInt(speed, 12), 1)`, i.e. `speed` clamped to the integer
/// range `[1, 12]`.
pub(crate) fn get_decimation_ratio(speed: i32) -> usize {
    speed.clamp(1, 12) as usize
}

/// C++: `fft_size` (dio.cpp:592-594).
///
/// Combines the four components exactly as in the C++: the decimated signal
/// length `y_length`, plus the low-cut filter width
/// `matlab_round(actual_fs / kCutOff) * 2 + 1` (with `kCutOff = 50.0`), plus
/// the highest-band filter width
/// `4 * int(1.0 + actual_fs / boundary_f0_list[0] / 2.0)`, rounded up to the
/// next power of two by `get_suitable_fft_size`.
pub(crate) fn get_dio_fft_size(y_length: usize, actual_fs: f64, boundary_f0_list: &[f64]) -> usize {
    let low_cut_width = matlab::matlab_round(actual_fs / K_CUT_OFF) as usize * 2 + 1;
    let highest_band_width = 4 * (1.0 + actual_fs / boundary_f0_list[0] / 2.0) as usize;
    common::get_suitable_fft_size(y_length + low_cut_width + highest_band_width)
}

/// C++: `GetF0CandidatesAndScores` (dio.cpp:549-572).
///
/// Iterates over the boundary F0 bands, computing the per-band F0 candidate
/// contour and score via `get_f0_candidate_from_raw_event`, then normalizes
/// each score by the candidate value (with `kMySafeGuardMinimum` guarding
/// against division by zero) and stores both into the per-band output arrays.
// C++ parity: mirrors `GetF0CandidatesAndScores` (dio.cpp:549).
#[allow(clippy::too_many_arguments)]
pub(crate) fn get_f0_candidates_and_scores(
    boundary_f0_list: &[f64],
    actual_fs: f64,
    y_length: usize,
    temporal_positions: &[f64],
    f0_length: usize,
    y_spectrum: &[FftComplex],
    fft_size: usize,
    f0_floor: f64,
    f0_ceil: f64,
    f0_candidates: &mut [Vec<f64>],
    f0_scores: &mut [Vec<f64>],
) {
    debug_assert_eq!(boundary_f0_list.len(), f0_candidates.len());
    debug_assert_eq!(boundary_f0_list.len(), f0_scores.len());
    let mut f0_candidate = vec![0.0f64; f0_length];
    let mut f0_score = vec![0.0f64; f0_length];

    for (i, &boundary_f0) in boundary_f0_list.iter().enumerate() {
        get_f0_candidate_from_raw_event(
            boundary_f0,
            actual_fs,
            y_spectrum,
            y_length,
            fft_size,
            f0_floor,
            f0_ceil,
            temporal_positions,
            f0_length,
            &mut f0_score,
            &mut f0_candidate,
        );
        for j in 0..f0_length {
            f0_scores[i][j] = f0_score[j] / (f0_candidate[j] + K_MY_SAFE_GUARD_MINIMUM);
            f0_candidates[i][j] = f0_candidate[j];
        }
    }
}

/// C++: `GetBestF0Contour` (dio.cpp:109-126).
///
/// Selects the best (lowest-score) candidate per frame across all bands.
/// Tie-breaking uses a strict `>` comparison, so the first band wins ties.
///
/// # Panics
///
/// In debug builds, panics if `f0_candidates.len() != f0_scores.len()` or if
/// `f0_scores` is empty.
pub(crate) fn get_best_f0_contour(
    f0_length: usize,
    f0_candidates: &[Vec<f64>],
    f0_scores: &[Vec<f64>],
    best_f0_contour: &mut [f64],
) {
    debug_assert_eq!(f0_candidates.len(), f0_scores.len());
    debug_assert!(!f0_scores.is_empty());
    for i in 0..f0_length {
        let mut tmp = f0_scores[0][i];
        best_f0_contour[i] = f0_candidates[0][i];
        for j in 1..f0_scores.len() {
            if tmp > f0_scores[j][i] {
                tmp = f0_scores[j][i];
                best_f0_contour[i] = f0_candidates[j][i];
            }
        }
    }
}

/// C++: `FixStep1` (dio.cpp:132-150). The 1st post-processing step: zeroes the
/// first and last `voice_range_minimum` frames, then eliminates any frame whose
/// relative change from the previous frame exceeds `allowed_range`.
fn fix_step1(
    best_f0_contour: &[f64],
    f0_length: usize,
    voice_range_minimum: usize,
    allowed_range: f64,
    f0_step1: &mut [f64],
) {
    debug_assert_eq!(best_f0_contour.len(), f0_length);
    debug_assert_eq!(f0_step1.len(), f0_length);
    let mut f0_base = vec![0.0f64; f0_length];
    let end = f0_length.saturating_sub(voice_range_minimum);
    if voice_range_minimum < end {
        f0_base[voice_range_minimum..end]
            .copy_from_slice(&best_f0_contour[voice_range_minimum..end]);
    }

    for i in voice_range_minimum..f0_length {
        let jump = (f0_base[i] - f0_base[i - 1]) / (K_MY_SAFE_GUARD_MINIMUM + f0_base[i]);
        f0_step1[i] = if jump.abs() < allowed_range {
            f0_base[i]
        } else {
            0.0
        };
    }
}

/// C++: `FixStep2` (dio.cpp:156-169). The 2nd post-processing step: removes
/// the suspected F0 in the anlaut/auslaut by zeroing any frame that has a zero
/// within `center = (voice_range_minimum - 1) / 2` frames of it.
fn fix_step2(f0_step1: &[f64], f0_length: usize, voice_range_minimum: usize, f0_step2: &mut [f64]) {
    debug_assert_eq!(f0_step1.len(), f0_length);
    debug_assert_eq!(f0_step2.len(), f0_length);
    f0_step2.copy_from_slice(f0_step1);

    let center = (voice_range_minimum - 1) / 2;
    let end = f0_length.saturating_sub(center);
    // Sliding zero-count over the `[i-center, i+center]` window: O(n)
    // instead of a linear scan per frame. Zero-detection uses exact
    // `== 0.0`, matching the previous `.contains(&0.0)`.
    if center < end {
        let mut zeros = f0_step1[..(2 * center + 1).min(f0_length)]
            .iter()
            .filter(|&&v| v == 0.0)
            .count();
        for i in center..end {
            if zeros > 0 {
                f0_step2[i] = 0.0;
            }
            if f0_step1[i - center] == 0.0 {
                zeros -= 1;
            }
            if i + center + 1 < f0_length && f0_step1[i + center + 1] == 0.0 {
                zeros += 1;
            }
        }
    }
}

/// C++: `GetNumberOfVoicedSections` (dio.cpp:174-184). Counts the voiced
/// sections. `negative_index` holds the last voiced frame of each section
/// (voiced→unvoiced, i.e. section ends); `positive_index` holds the first
/// voiced frame (unvoiced→voiced, i.e. section starts). Both arrays are
/// pre-allocated with `f0_length` capacity by the caller. Returns
/// `(positive_count, negative_count)`.
fn get_number_of_voiced_sections(
    f0: &[f64],
    f0_length: usize,
    positive_index: &mut [usize],
    negative_index: &mut [usize],
) -> (usize, usize) {
    debug_assert_eq!(positive_index.len(), f0_length);
    debug_assert_eq!(negative_index.len(), f0_length);
    let mut positive_count = 0;
    let mut negative_count = 0;
    for i in 1..f0_length {
        if f0[i] == 0.0 && f0[i - 1] != 0.0 {
            negative_index[negative_count] = i - 1;
            negative_count += 1;
        } else if f0[i - 1] == 0.0 && f0[i] != 0.0 {
            positive_index[positive_count] = i;
            positive_count += 1;
        }
    }
    (positive_count, negative_count)
}

/// C++: `SelectBestF0` (dio.cpp:190-209). Corrects the F0 at `target_index`
/// based on a reference prediction `(current_f0 * 3.0 - past_f0) / 2.0`. Picks
/// the candidate closest to the reference; returns `0.0` if the best candidate
/// deviates from the reference by more than `allowed_range`. Scores are not
/// used in this step.
fn select_best_f0(
    current_f0: f64,
    past_f0: f64,
    f0_candidates: &[Vec<f64>],
    number_of_candidates: usize,
    target_index: usize,
    allowed_range: f64,
) -> f64 {
    let reference_f0 = (current_f0 * 3.0 - past_f0) / 2.0;

    let mut minimum_error = (reference_f0 - f0_candidates[0][target_index]).abs();
    let mut best_f0 = f0_candidates[0][target_index];

    for candidate in f0_candidates.iter().take(number_of_candidates).skip(1) {
        let current_error = (reference_f0 - candidate[target_index]).abs();
        if current_error < minimum_error {
            minimum_error = current_error;
            best_f0 = candidate[target_index];
        }
    }
    if (1.0 - best_f0 / reference_f0).abs() > allowed_range {
        0.0
    } else {
        best_f0
    }
}

/// C++: `FixStep3` (dio.cpp:215-231). The 3rd post-processing step: corrects
/// the F0 candidates from backward to forward, extending each voiced section
/// past its end (`negative_index`) as long as `select_best_f0` stays in range.
// C++ parity: mirrors `FixStep3` (dio.cpp:215).
#[allow(clippy::too_many_arguments)]
fn fix_step3(
    f0_step2: &[f64],
    f0_length: usize,
    f0_candidates: &[Vec<f64>],
    number_of_candidates: usize,
    allowed_range: f64,
    negative_index: &[usize],
    negative_count: usize,
    f0_step3: &mut [f64],
) {
    debug_assert_eq!(f0_step2.len(), f0_length);
    debug_assert_eq!(f0_step3.len(), f0_length);
    f0_step3.copy_from_slice(f0_step2);

    for i in 0..negative_count {
        let limit = if i == negative_count - 1 {
            f0_length - 1
        } else {
            negative_index[i + 1]
        };
        for j in negative_index[i]..limit {
            f0_step3[j + 1] = select_best_f0(
                f0_step3[j],
                f0_step3[j - 1],
                f0_candidates,
                number_of_candidates,
                j + 1,
                allowed_range,
            );
            if f0_step3[j + 1] == 0.0 {
                break;
            }
        }
    }
}

/// C++: `FixStep4` (dio.cpp:237-253). The 4th post-processing step: corrects
/// the F0 candidates from forward to backward, extending each voiced section
/// before its start (`positive_index`) as long as `select_best_f0` stays in
/// range.
// C++ parity: mirrors `FixStep4` (dio.cpp:237).
#[allow(clippy::too_many_arguments)]
fn fix_step4(
    f0_step3: &[f64],
    f0_length: usize,
    f0_candidates: &[Vec<f64>],
    number_of_candidates: usize,
    allowed_range: f64,
    positive_index: &[usize],
    positive_count: usize,
    f0_step4: &mut [f64],
) {
    debug_assert_eq!(f0_step3.len(), f0_length);
    debug_assert_eq!(f0_step4.len(), f0_length);
    f0_step4.copy_from_slice(f0_step3);

    for i in (0..positive_count).rev() {
        let limit = if i == 0 { 1 } else { positive_index[i - 1] };
        for j in (limit + 1..=positive_index[i]).rev() {
            f0_step4[j - 1] = select_best_f0(
                f0_step4[j],
                f0_step4[j + 1],
                f0_candidates,
                number_of_candidates,
                j - 1,
                allowed_range,
            );
            if f0_step4[j - 1] == 0.0 {
                break;
            }
        }
    }
}

/// C++: `FixF0Contour` (dio.cpp:259-289). Calculates the definitive F0 contour
/// by orchestrating the four post-processing steps. `voice_range_minimum` is
/// `int(0.5 + 1000.0 / frame_period / f0_floor) * 2 + 1`.
///
/// The C++ `fs` parameter is unused and is omitted here; `number_of_candidates`
/// is derived from `f0_candidates.len()`.
///
/// Early return: the C++ returns without touching `fixed_f0_contour` when
/// `f0_length <= voice_range_minimum` (leaving it uninitialized). The Rust
/// port zero-fills `fixed_f0_contour` for safety.
pub(crate) fn fix_f0_contour(
    frame_period: f64,
    f0_candidates: &[Vec<f64>],
    best_f0_contour: &[f64],
    f0_length: usize,
    f0_floor: f64,
    allowed_range: f64,
    fixed_f0_contour: &mut [f64],
) {
    debug_assert_eq!(best_f0_contour.len(), f0_length);
    debug_assert_eq!(fixed_f0_contour.len(), f0_length);
    let number_of_candidates = f0_candidates.len();
    let voice_range_minimum = (0.5 + 1000.0 / frame_period / f0_floor) as usize * 2 + 1;

    if f0_length <= voice_range_minimum {
        fixed_f0_contour.fill(0.0);
        return;
    }

    let mut f0_tmp1 = vec![0.0f64; f0_length];
    let mut f0_tmp2 = vec![0.0f64; f0_length];

    fix_step1(
        best_f0_contour,
        f0_length,
        voice_range_minimum,
        allowed_range,
        &mut f0_tmp1,
    );
    fix_step2(&f0_tmp1, f0_length, voice_range_minimum, &mut f0_tmp2);

    let mut positive_index = vec![0usize; f0_length];
    let mut negative_index = vec![0usize; f0_length];
    let (positive_count, negative_count) = get_number_of_voiced_sections(
        &f0_tmp2,
        f0_length,
        &mut positive_index,
        &mut negative_index,
    );

    fix_step3(
        &f0_tmp2,
        f0_length,
        f0_candidates,
        number_of_candidates,
        allowed_range,
        &negative_index,
        negative_count,
        &mut f0_tmp1,
    );
    fix_step4(
        &f0_tmp1,
        f0_length,
        f0_candidates,
        number_of_candidates,
        allowed_range,
        &positive_index,
        positive_count,
        fixed_f0_contour,
    );
}

/// C++: `DioGeneralBody` (dio.cpp:578-635). Orchestrates the full DIO
/// pipeline: spectrum estimation, per-band F0 candidate generation, best
/// contour selection, and post-processing. Field/step order mirrors the C++
/// exactly (see the numbered comments below against the C++ line numbers).
///
/// Memory management: the C++ performs 7 heap allocations — `boundary_f0_list`
/// (584), `y_spectrum` (597), the `f0_candidates`/`f0_scores` outer arrays
/// (601-602), the per-band `f0_candidates[i]`/`f0_scores[i]` buffers (605-606),
/// and `best_f0_contour` (618) — and frees each in reverse at lines 626-634.
/// Rust's ownership model releases all of these automatically at scope exit,
/// so no explicit cleanup is required here.
// C++ parity: mirrors `DioGeneralBody` (dio.cpp:578).
#[allow(clippy::too_many_arguments)]
pub(crate) fn dio_general_body(
    x: &[f64],
    fs: f64,
    frame_period: f64,
    f0_floor: f64,
    f0_ceil: f64,
    channels_in_octave: f64,
    speed: i32,
    allowed_range: f64,
) -> DioResult {
    let x_length = x.len();

    // 1. (dio.cpp:582-583) Number of frequency bands.
    let number_of_bands = get_number_of_bands(f0_floor, f0_ceil, channels_in_octave);

    // 2. (dio.cpp:584-586) Boundary F0 for each band.
    let boundary_f0_list = get_boundary_f0_list(f0_floor, channels_in_octave, number_of_bands);

    // 3. (dio.cpp:589-591) Clamp speed -> decimation ratio; decimated length
    // and the corresponding sample rate.
    let decimation_ratio = get_decimation_ratio(speed);
    let y_length = 1 + x_length / decimation_ratio;
    let actual_fs = fs / decimation_ratio as f64;

    // 4. (dio.cpp:592-594) FFT size for the estimation spectrum.
    let fft_size = get_dio_fft_size(y_length, actual_fs, &boundary_f0_list);

    // 5. (dio.cpp:597) Spectrum buffer.
    let mut y_spectrum = vec![[0.0, 0.0]; fft_size];

    // 6. (dio.cpp:598-599) Low-cut-filtered half-spectrum of the signal.
    get_spectrum_for_estimation(
        x,
        y_length,
        actual_fs,
        fft_size,
        decimation_ratio,
        &mut y_spectrum,
    );

    // 7. (dio.cpp:601-607) Per-band F0 candidate and score buffers.
    let f0_length = get_samples_for_dio(fs, x_length, frame_period);
    let mut f0_candidates = vec![vec![0.0; f0_length]; number_of_bands];
    let mut f0_scores = vec![vec![0.0; f0_length]; number_of_bands];

    // 8. (dio.cpp:609-610) Temporal grid. C++ computes `i * frame_period /
    // 1000.0` (with `i` an int); `i as f64 * frame_period / 1000.0` matches it
    // exactly. For precision-critical use `i as f64 * (frame_period / 1000.0)`
    // could be considered, but it is not used here to stay faithful to C++.
    let temporal_positions: Vec<f64> = (0..f0_length)
        .map(|i| i as f64 * frame_period / 1000.0)
        .collect();

    // 9. (dio.cpp:612-614) Per-band F0 candidates and scores.
    get_f0_candidates_and_scores(
        &boundary_f0_list,
        actual_fs,
        y_length,
        &temporal_positions,
        f0_length,
        &y_spectrum,
        fft_size,
        f0_floor,
        f0_ceil,
        &mut f0_candidates,
        &mut f0_scores,
    );

    // 10. (dio.cpp:618-620) Best (lowest-score) candidate per frame.
    let mut best_f0_contour = vec![0.0; f0_length];
    get_best_f0_contour(f0_length, &f0_candidates, &f0_scores, &mut best_f0_contour);

    // 11. (dio.cpp:623-624) Post-processing to the definitive F0 contour.
    let mut f0 = vec![0.0; f0_length];
    fix_f0_contour(
        frame_period,
        &f0_candidates,
        &best_f0_contour,
        f0_length,
        f0_floor,
        allowed_range,
        &mut f0,
    );

    // 12. (dio.cpp:626-634) The C++ frees `best_f0_contour`, `y_spectrum`,
    // `f0_scores[i]`/`f0_candidates[i]` per band, `f0_scores`, `f0_candidates`,
    // and `boundary_f0_list`. Rust drops all of these automatically.

    DioResult {
        f0,
        temporal_positions,
        f0_length,
        f0_candidates,
        f0_scores,
        number_of_bands,
    }
}

/// Result of [`dio`]. Exposes the definitive F0 contour, the temporal grid it
/// is aligned to, and the intermediate per-band candidates/scores.
#[derive(Debug, Clone, PartialEq)]
pub struct DioResult {
    /// Definitive F0 contour; `0.0` marks unvoiced frames. Length is
    /// `f0_length`.
    pub f0: Vec<f64>,
    /// Temporal position (seconds) of each frame: `i * frame_period / 1000.0`.
    pub temporal_positions: Vec<f64>,
    /// Number of frames in `f0` (and `temporal_positions`).
    pub f0_length: usize,
    /// Per-band F0 candidate contours, one row per band.
    pub f0_candidates: Vec<Vec<f64>>,
    /// Per-band F0 scores (standard deviation of the four zero-crossing
    /// sources), one row per band.
    pub f0_scores: Vec<Vec<f64>>,
    /// Number of frequency bands (rows in `f0_candidates`/`f0_scores`).
    pub number_of_bands: usize,
}

/// Error returned by [`dio`] when the input is invalid.
///
/// The C++ `Dio` performs no input validation (its `GetSamplesForDIO` returns
/// 1 for empty input); the Rust port validates the inputs up front and reports
/// a specific [`DioError`] instead of running the pipeline on degenerate input
/// (which could otherwise panic or produce meaningless output).
#[derive(Debug, Clone, PartialEq)]
pub enum DioError {
    /// The input signal slice `x` is empty.
    EmptyInput,
    /// The sample rate `fs` is not strictly positive (zero or negative).
    NonPositiveSampleRate { fs: f64 },
    /// The frame period (msec) is not strictly positive (zero or negative).
    NonPositiveFramePeriod { frame_period: f64 },
    /// `f0_floor` and/or `f0_ceil` are invalid (non-finite, <=0, or `f0_ceil <= f0_floor`).
    InvalidF0Range,
    /// The input signal contains NaN or infinite samples.
    NonFiniteInput,
    /// The input signal is too short to contain a single frame.
    TooShortInput,
    /// `f0_floor` is too low, causing an overflow in FFT size computation.
    F0FloorTooLow,
    /// `channels_in_octave` is invalid (non-finite, <=0, or >64).
    InvalidChannelsInOctave,
    /// `allowed_range` is NaN.
    InvalidAllowedRange,
    /// The frame plan (`bands × frames`) exceeds the allocation budget
    /// (`MAX_DIO_CELLS`) or `frames` exceeds `MAX_DIO_FRAMES`. Rejects
    /// degenerate plans (e.g. near-zero `frame_period`) before any
    /// frame-major buffer is allocated.
    TooManyFrames { bands: usize, frames: usize },
}

impl std::fmt::Display for DioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DioError::EmptyInput => write!(f, "input signal is empty"),
            DioError::NonPositiveSampleRate { fs } => {
                write!(f, "sample rate must be positive, got {fs}")
            }
            DioError::NonPositiveFramePeriod { frame_period } => {
                write!(f, "frame period must be positive, got {frame_period}")
            }
            DioError::InvalidF0Range => write!(f, "invalid f0 range"),
            DioError::NonFiniteInput => write!(f, "input contains non-finite samples"),
            DioError::TooShortInput => write!(f, "input too short for one frame"),
            DioError::F0FloorTooLow => write!(f, "f0_floor too low, would overflow FFT size"),
            DioError::InvalidChannelsInOctave => write!(f, "channels_in_octave must be in (0, 64]"),
            DioError::InvalidAllowedRange => write!(f, "allowed_range must be finite"),
            DioError::TooManyFrames { bands, frames } => write!(
                f,
                "frame plan too large: {bands} bands × {frames} frames exceeds budget"
            ),
        }
    }
}

impl std::error::Error for DioError {}

/// C++: `Dio` (dio.cpp:643-648). Public DIO pitch-extraction entry point.
///
/// Validates the input before running the pipeline. The C++ performs no
/// validation (its `GetSamplesForDIO` returns 1 for empty input); the Rust
/// port guards against empty input and non-positive sample rate/frame period
/// and returns the corresponding [`DioError`] rather than running the pipeline
/// on degenerate input (no misleading data, no panics).
pub fn dio(x: &[f64], fs: f64, option: &DioOption) -> Result<DioResult, DioError> {
    if x.is_empty() {
        return Err(DioError::EmptyInput);
    }
    if !fs.is_finite() || fs <= 0.0 {
        return Err(DioError::NonPositiveSampleRate { fs });
    }
    if (fs as i32) == 0 {
        return Err(DioError::NonPositiveSampleRate { fs });
    }
    if !option.frame_period.is_finite() || option.frame_period <= 0.0 {
        return Err(DioError::NonPositiveFramePeriod {
            frame_period: option.frame_period,
        });
    }
    if !option.f0_floor.is_finite()
        || !option.f0_ceil.is_finite()
        || option.f0_floor <= 0.0
        || option.f0_ceil <= option.f0_floor
    {
        return Err(DioError::InvalidF0Range);
    }
    if option.f0_floor < 1e-6 || fs / option.f0_floor > 1e7 {
        return Err(DioError::F0FloorTooLow);
    }
    if !option.channels_in_octave.is_finite()
        || option.channels_in_octave <= 0.0
        || option.channels_in_octave > 64.0
    {
        return Err(DioError::InvalidChannelsInOctave);
    }
    if !option.allowed_range.is_finite() {
        return Err(DioError::InvalidAllowedRange);
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(DioError::NonFiniteInput);
    }
    let min_samples = (fs * option.frame_period / 1000.0).ceil() as usize;
    if x.len() < min_samples {
        return Err(DioError::TooShortInput);
    }
    // Allocation budget: the estimator holds two `bands × frames` f64
    // matrices plus per-frame contour vecs. `bands` is already clamped to
    // 256 by `get_number_of_bands`; bound `frames` and the product so a
    // degenerate plan (e.g. `frame_period → 0`) fails here instead of
    // attempting a multi-GB allocation. 64M cells ≈ 1GB for the two
    // matrices; an hour of audio at 5ms (56 bands × 720k frames ≈ 40M
    // cells) still fits.
    let f0_length = get_samples_for_dio(fs, x.len(), option.frame_period);
    let bands = get_number_of_bands(option.f0_floor, option.f0_ceil, option.channels_in_octave);
    if f0_length > MAX_DIO_FRAMES || (bands as u128) * (f0_length as u128) > MAX_DIO_CELLS as u128 {
        return Err(DioError::TooManyFrames {
            bands,
            frames: f0_length,
        });
    }
    Ok(dio_general_body(
        x,
        fs,
        option.frame_period,
        option.f0_floor,
        option.f0_ceil,
        option.channels_in_octave,
        option.speed,
        option.allowed_range,
    ))
}

#[cfg(test)]
// Reference values are captured from the C++ port at full f64 precision; the
// excess digits are intentional (documenting the exact reference), not noise.
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    #[test]
    fn test_initialize_dio_option_matches_cpp_reference() {
        let option = initialize_dio_option();
        assert_eq!(option.f0_floor, 71.0);
        assert_eq!(option.f0_ceil, 800.0);
        assert_eq!(option.channels_in_octave, 2.0);
        assert_eq!(option.frame_period, 5.0);
        assert_eq!(option.speed, 1);
        assert_eq!(option.allowed_range, 0.1);
    }

    #[test]
    fn test_get_samples_for_dio() {
        // C++: static_cast<int>(1000.0 * x_length / fs / frame_period) + 1
        assert_eq!(get_samples_for_dio(16000.0, 16000, 5.0), 201);
        assert_eq!(get_samples_for_dio(44100.0, 44100, 5.0), 201);
        assert_eq!(get_samples_for_dio(48000.0, 96000, 10.0), 201);
        assert_eq!(get_samples_for_dio(48000.0, 960000, 10.0), 2001);
        assert_eq!(get_samples_for_dio(16000.0, 79, 5.0), 1);
        assert_eq!(get_samples_for_dio(16000.0, 0, 5.0), 1);
    }

    // Reference vectors from /tmp/opencode/dio_p12_vectors.out (C++ driver
    // with the dio.cpp:40-106 function bodies copied verbatim, linked against
    // libworld.a). Tolerance is the 1e-6 required by PHASE1-2.
    const TOL: f64 = 1e-6;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < TOL
    }

    // Relative 1e-6 for large-magnitude spectrum bins (FFT accumulation order
    // differs between Ooura split-radix and rustfft).
    fn close_rel(a: f64, b: f64) -> bool {
        (a - b).abs() < TOL * (1.0 + a.abs().max(b.abs()))
    }

    #[test]
    fn test_design_low_cut_filter_n5() {
        let mut f = vec![0.0; 64];
        design_low_cut_filter(5, 64, &mut f);
        let expected: &[f64] = &[
            0.66666666666666674,
            -0.25000000000000006,
            -0.083333333333333315,
        ];
        for (i, &e) in expected.iter().enumerate() {
            assert!(close(f[i], e), "i={}: {} vs {}", i, f[i], e);
        }
        // Middle is zero; tail is the mirrored head (pre-shift values).
        for (i, &v) in f.iter().enumerate().take(62).skip(3) {
            assert_eq!(v, 0.0, "i={}", i);
        }
        assert!(close(f[62], -0.083333333333333315));
        assert!(close(f[63], -0.24999999999999997));
    }

    #[test]
    fn test_design_low_cut_filter_n9() {
        let mut f = vec![0.0; 128];
        design_low_cut_filter(9, 128, &mut f);
        let head: &[f64] = &[
            0.80000000000000004,
            -0.18090169943749471,
            -0.13090169943749472,
            -0.069098300562505266,
            -0.019098300562505263,
        ];
        for (i, &e) in head.iter().enumerate() {
            assert!(close(f[i], e), "i={}: {} vs {}", i, f[i], e);
        }
        for (i, &v) in f.iter().enumerate().take(124).skip(5) {
            assert_eq!(v, 0.0, "i={}", i);
        }
        let tail: &[f64] = &[
            -0.019098300562505253,
            -0.069098300562505238,
            -0.13090169943749472,
            -0.18090169943749471,
        ];
        for (i, &e) in tail.iter().enumerate() {
            assert!(
                close(f[124 + i], e),
                "i={}: {} vs {}",
                124 + i,
                f[124 + i],
                e
            );
        }
    }

    #[test]
    fn test_design_low_cut_filter_n17() {
        let mut f = vec![0.0; 256];
        design_low_cut_filter(17, 256, &mut f);
        let head: &[f64] = &[
            0.88888888888888884,
            -0.1077607011547727,
            -0.098113580173276579,
            -0.083333333333333356,
            -0.065202676537051682,
            -0.045908434574059444,
            -0.027777777777777814,
            -0.012997530937834567,
            -0.0033504099563384207,
        ];
        for (i, &e) in head.iter().enumerate() {
            assert!(close(f[i], e), "i={}: {} vs {}", i, f[i], e);
        }
        for (i, &v) in f.iter().enumerate().take(248).skip(9) {
            assert_eq!(v, 0.0, "i={}", i);
        }
        let tail: &[f64] = &[
            -0.0033504099563384207,
            -0.012997530937834555,
            -0.027777777777777773,
            -0.045908434574059423,
            -0.065202676537051682,
            -0.083333333333333315,
            -0.098113580173276552,
            -0.1077607011547727,
        ];
        for (i, &e) in tail.iter().enumerate() {
            assert!(
                close(f[248 + i], e),
                "i={}: {} vs {}",
                248 + i,
                f[248 + i],
                e
            );
        }
    }

    #[test]
    fn test_design_low_cut_filter_n33() {
        let mut f = vec![0.0; 512];
        design_low_cut_filter(33, 512, &mut f);
        let head: &[f64] = &[
            0.94117647058823528,
            -0.05832273822599713,
            -0.056837418511892834,
            -0.054418151050871014,
            -0.051147321094725279,
            -0.047136312834684044,
            -0.042521716346368796,
            -0.037460676178590689,
            -0.032125539984214774,
            -0.026697989427549931,
            -0.02136285323317404,
            -0.016301813065395951,
            -0.011687216577080703,
            -0.0076762083170394567,
            -0.0044053783608937105,
            -0.0019861108998718881,
            -0.00050079118576759466,
        ];
        for (i, &e) in head.iter().enumerate() {
            assert!(close(f[i], e), "i={}: {} vs {}", i, f[i], e);
        }
        for (i, &v) in f.iter().enumerate().take(496).skip(17) {
            assert_eq!(v, 0.0, "i={}", i);
        }
        let tail: &[f64] = &[
            -0.00050079118576759466,
            -0.0019861108998718881,
            -0.0044053783608937001,
            -0.0076762083170394402,
            -0.011687216577080696,
            -0.016301813065395937,
            -0.021362853233174033,
            -0.026697989427549945,
            -0.032125539984214767,
            -0.037460676178590682,
            -0.042521716346368768,
            -0.047136312834684016,
            -0.051147321094725286,
            -0.054418151050871014,
            -0.056837418511892827,
            -0.05832273822599713,
        ];
        for (i, &e) in tail.iter().enumerate() {
            assert!(
                close(f[496 + i], e),
                "i={}: {} vs {}",
                496 + i,
                f[496 + i],
                e
            );
        }
    }

    // Realistic DIO size (fs=16000 -> cutoff=320 -> N=641). Head/tail values
    // plus structural invariants (zero middle, mirror tail, sum of first N).
    #[test]
    fn test_design_low_cut_filter_realistic_n641() {
        let fft_size = 16384;
        let mut f = vec![0.0; fft_size];
        design_low_cut_filter(641, fft_size, &mut f);
        let head: &[f64] = &[
            0.99688473520249221,
            -0.0031151902005373084,
            -0.0031149664167709557,
            -0.0031145934676432997,
            -0.0031140713888763443,
            -0.0031134002304761049,
            -0.0031125800567278171,
            -0.0031116109461897805,
            -0.0031104929916858366,
            -0.0031092263002964731,
            -0.0031078109933485713,
            -0.0031062472064037824,
        ];
        for (i, &e) in head.iter().enumerate() {
            assert!(close(f[i], e), "i={}: {} vs {}", i, f[i], e);
        }
        let tail: &[f64] = &[
            -0.0031045350892455457,
            -0.0031062472064037824,
            -0.0031078109933485713,
            -0.0031092263002964731,
            -0.0031104929916858366,
            -0.0031116109461897805,
            -0.0031125800567278171,
            -0.0031134002304761049,
            -0.0031140713888763443,
            -0.0031145934676432997,
            -0.0031149664167709557,
            -0.0031151902005373084,
        ];
        for (i, &e) in tail.iter().enumerate() {
            let idx = fft_size - 12 + i;
            assert!(close(f[idx], e), "i={}: {} vs {}", idx, f[idx], e);
        }
        // Zero middle: nonzero region is [0, 641) and [16384-320, 16384).
        for (i, &v) in f.iter().enumerate().take(fft_size - 320).skip(641) {
            assert_eq!(v, 0.0, "i={}", i);
        }
        let sum: f64 = f[..641].iter().sum();
        assert!(close(sum, 0.49844236760124611), "sum={}", sum);
    }

    // Full half-spectrum (129 bins) vs C++ reference: x[i] = 0.1i + 0.001i^2,
    // x_length=100, y_length=51, actual_fs=1600, fft_size=256, ratio=2.
    #[test]
    fn test_get_spectrum_for_estimation_ratio2_full() {
        let x: Vec<f64> = (0..100)
            .map(|i| 0.1 * i as f64 + 0.001 * i as f64 * i as f64)
            .collect();
        let mut spec = vec![[0.0, 0.0]; 256];
        get_spectrum_for_estimation(&x, 51, 1600.0, 256, 2, &mut spec);
        let expected: &[[f64; 2]] = &[
            [-0.0, 0.0],
            [-2.6378629544738854, -3.5791196390530757],
            [-28.633131945786182, -8.8256022394220608],
            [-70.919740657084489, 25.196444652859412],
            [-63.203919648780548, 96.21540431844825],
            [10.315007257552557, 120.12246608054259],
            [60.002706572453796, 56.531807442512275],
            [19.312379317326311, -13.021388159792421],
            [-52.582961872102622, -1.6074881393882723],
            [-61.155437709572055, 61.617147500865499],
            [-5.8944398766854489, 86.343964319273439],
            [36.183236182028985, 45.799546253820353],
            [18.619647054177403, -3.8208676798985626],
            [-26.573787976907948, -5.5290438139023754],
            [-40.219331906537171, 32.334309574435558],
            [-9.5068406730897657, 55.627063234362097],
            [22.335495511121131, 35.083943470769192],
            [15.417856094475004, -0.38828181973496056],
            [-17.128324044276923, -6.0151334102166514],
            [-31.237450484816335, 20.651930199489609],
            [-10.811634977447611, 41.136182985480644],
            [14.935456612881248, 28.826317103329199],
            [12.820002968366603, 1.3319450331642007],
            [-11.970834516361593, -5.8523846388461154],
            [-25.675670294462648, 13.969641875129305],
            [-11.413981419927071, 32.329516214575882],
            [10.347144271014006, 24.843876142807499],
            [10.956792946192042, 2.3715220725233355],
            [-8.8208524186065489, -5.588103254007537],
            [-21.898655594314437, 9.6692469897745301],
            [-11.68341736823378, 26.227341233133565],
            [7.1170719684800998, 21.907122142678677],
            [9.4987920191545516, 3.075081877813536],
            [-6.6818235775954165, -5.3086809462231983],
            [-19.196674139279111, 6.6859161550986466],
            [-11.838757249124432, 21.814333431171047],
            [4.7195020993741039, 19.683468297679529],
            [8.3157897619745462, 3.5871166992718595],
            [-5.111406303829801, -5.0230871544014226],
            [-17.091399534131813, 4.4654416473327405],
            [-11.914020035368763, 18.40823904146054],
            [2.8507446599055468, 17.929758052476558],
            [7.3405696652775783, 3.9871251816201543],
            [-3.9139150106655265, -4.7515964977530123],
            [-15.399099181124679, 2.7480430341320439],
            [-11.932793245117193, 15.665427594464695],
            [1.3341254192956675, 16.469262459492633],
            [6.5012872617023989, 4.3074983205014554],
            [-2.9638497948750402, -4.4904224695474522],
            [-14.000740826590958, 1.3763831176824406],
            [-11.927409513038297, 13.40635143124844],
            [0.069927274375168408, 15.23581275283969],
            [5.7643043897395305, 4.5725948994417447],
            [-2.1872631090278869, -4.235969237163955],
            [-12.800219874790388, 0.25032411016631689],
            [-11.896034050327552, 11.486404750251056],
            [-1.0100687039343952, 14.16607258191706],
            [5.1066334609356518, 4.7992795016433414],
            [-1.5398963381721227, -3.9904590795342596],
            [-11.753959341862032, -0.69283342529252312],
            [-11.845240671077066, 9.8209562838446054],
            [-1.9507605174237161, 13.213034123683304],
            [4.5060085304443733, 4.9940436735476066],
            [-0.98917228287828884, -3.7502306175170497],
            [-10.824767726573191, -1.4971700830225172],
            [-11.783683154897043, 8.3536983262456328],
            [-2.7848705418240383, 12.355740449461283],
            [3.9507014833381096, 5.1653235837404452],
            [-0.51315311517618112, -3.5135306315120682],
            [-9.981991997522579, -2.1931175568682892],
            [-11.708596858289303, 7.0366237700849448],
            [-3.5356387459858878, 11.569903310412437],
            [3.4305679362213244, 5.3188127012348172],
            [-0.09597168032472668, -3.2798543836368577],
            [-9.2095082189372768, -2.8038191183964574],
            [-11.623501992058243, 5.8385376918190559],
            [-4.2204969475688241, 10.838410756401672],
            [2.9362286843571876, 5.4571583567978834],
            [0.27462103669284255, -3.0463083475625643],
            [-8.4905173079183847, -3.3460488122012717],
            [-11.530665002033652, 4.7345688534120036],
            [-4.8546039243176109, 10.151348972370894],
            [2.4613137083428085, 5.5850434614622255],
            [0.60786582734588368, -2.8108168760404295],
            [-7.811194687572276, -3.8321481659923688],
            [-11.427784291873534, 3.7031184736591154],
            [-5.4488694236644699, 9.4960870265196924],
            [1.9990113485147611, 5.7053001310753926],
            [0.91098726117223627, -2.5707396780057081],
            [-7.1618718763176625, -4.2722737904762358],
            [-11.31567527106553, 2.727846455014785],
            [-6.0130416244271139, 8.8630763254012148],
            [1.5423558730925744, 5.819731239179732],
            [1.1890653288864987, -2.3218507684639351],
            [-6.5315119057581201, -4.6726353271792389],
            [-11.192277183556685, 1.7941217597808725],
            [-6.5558764193300219, 8.2435457254319644],
            [1.0842454702579181, 5.9305969764542326],
            [1.4452058822845926, -2.0596324957889274],
            [-5.9105871761420952, -5.0367481263367395],
            [-11.05263985991748, 0.88937036460741492],
            [-7.0830050424412772, 7.6262228213384846],
            [0.61636539454260952, 6.0374557045981625],
            [1.6798285333299028, -1.7786689984091573],
            [-5.2910770039226493, -5.3653531527824265],
            [-10.891628086717832, 0.003819451013752593],
            [-7.5983831316912331, 7.0001971408134569],
            [0.12991605827175259, 6.1374461332003483],
            [1.8898556704505551, -1.4734767527604662],
            [-4.6659253309511239, -5.654935747073135],
            [-10.700616685886695, -0.86909271822722711],
            [-8.1022413412831664, 6.3538206544033349],
            [-0.38339029699298066, 6.2246944409643348],
            [2.0687972053930284, -1.140484407564845],
            [-4.0320579301574719, -5.8994542761325501],
            [-10.469938719811998, -1.7311882163185441],
            [-8.5896828865835602, 5.6759248492940966],
            [-0.92941669980844066, 6.2889649468823281],
            [2.2072546884467625, -0.77968599990215937],
            [-3.3912545265497123, -6.0916736492758945],
            [-10.191747446612023, -2.5791705138938821],
            [-9.0523321715191809, 4.9592038560326168],
            [-1.5094465800196395, 6.3179686559409554],
            [2.2953013779704752, -0.39620446711511714],
            [-2.7502147842513662, -6.2256229458543801],
            [-9.8612798825364507, -3.4055317980775004],
            [-9.4794532816689099, 4.2013318738072858],
            [-2.1192968257386209, 6.299804494466092],
            [2.3255655274886164, 0.0],
        ];
        assert_eq!(expected.len(), 129);
        for (i, &e) in expected.iter().enumerate() {
            assert!(
                close_rel(spec[i][0], e[0]) && close_rel(spec[i][1], e[1]),
                "bin {}: {} {} vs {} {}",
                i,
                spec[i][0],
                spec[i][1],
                e[0],
                e[1]
            );
        }
    }

    // ratio=1 path: x_length=1600, actual_fs=16000 (cutoff=320, N=641),
    // fft_size=4096. Head/tail bins vs C++ reference.
    #[test]
    fn test_get_spectrum_for_estimation_ratio1() {
        let x: Vec<f64> = (0..1600)
            .map(|i| 0.1 * i as f64 + 0.001 * i as f64 * i as f64)
            .collect();
        let mut spec = vec![[0.0, 0.0]; 4096];
        get_spectrum_for_estimation(&x, 1600, 16000.0, 4096, 1, &mut spec);
        let head: &[[f64; 2]] = &[
            [1.0903950976326914e-25, -0.0],
            [-11628.386373424739, -3142.6757255607499],
            [-27580.915932839933, 50440.237617156155],
            [52841.992858726022, 35013.020577841176],
            [-43866.712221535476, -15424.557807749037],
            [-5315.2493901415755, 122107.40273176007],
            [62159.929322295749, -15587.048488763523],
            [-96177.580338660278, 57148.801608010035],
            [79146.374980371707, 112228.8544473887],
            [-23388.77794929433, -45439.072602579348],
            [-46988.516462710613, 142564.6574325148],
            [90126.512493315196, 13754.961905040185],
            [-91281.105432301512, 21571.93413705935],
            [48227.025237344154, 123062.67478536109],
            [9578.0998762345134, -39885.633958404302],
            [-58716.205516903734, 96500.816706551559],
        ];
        for (i, &e) in head.iter().enumerate() {
            assert!(
                close_rel(spec[i][0], e[0]) && close_rel(spec[i][1], e[1]),
                "bin {}: {} {} vs {} {}",
                i,
                spec[i][0],
                spec[i][1],
                e[0],
                e[1]
            );
        }
        let tail: &[[f64; 2]] = &[
            [359.74532328358998, 340.6078754049708],
            [-889.0062527389282, -785.4545712597353],
            [-639.12471986290961, 876.75846347611821],
            [223.54510692570716, -566.64315520830348],
            [-1359.1999999999996, 0.0],
        ];
        for (i, &e) in tail.iter().enumerate() {
            let idx = 4096 / 2 - 4 + i;
            assert!(
                close_rel(spec[idx][0], e[0]) && close_rel(spec[idx][1], e[1]),
                "bin {}: {} {} vs {} {}",
                idx,
                spec[idx][0],
                spec[idx][1],
                e[0],
                e[1]
            );
        }
    }

    // ratio=2 path at 16 kHz: y_length=801, actual_fs=8000 (cutoff=160,
    // N=321), fft_size=2048. Head/tail bins vs C++ reference.
    #[test]
    fn test_get_spectrum_for_estimation_ratio2_16k() {
        let x: Vec<f64> = (0..1600)
            .map(|i| 0.1 * i as f64 + 0.001 * i as f64 * i as f64)
            .collect();
        let mut spec = vec![[0.0, 0.0]; 2048];
        get_spectrum_for_estimation(&x, 801, 8000.0, 2048, 2, &mut spec);
        let head: &[[f64; 2]] = &[
            [-1.4445620896407168e-25, 0.0],
            [-5882.1623370159587, -1584.2433495609894],
            [-13879.241069281787, 25494.743681580938],
            [26631.213734167046, 17485.19927789849],
            [-22416.024759140051, -7578.0563491426074],
            [-2305.0633640066631, 61582.340416803556],
            [30768.277322305046, -8285.8081696901681],
            [-48519.760470330526, 29570.252344902889],
            [40230.937305761385, 55745.924588098424],
            [-12996.797233703262, -22605.409581842501],
            [-22771.625821517853, 72220.685312568181],
            [44562.665647525951, 5745.5013781626867],
            [-46454.445380658224, 12123.345618546697],
            [25119.057235342472, 61006.600780721586],
            [3168.2198087108268, -20155.03174760706],
            [-28703.328013986455, 49407.322582860761],
        ];
        for (i, &e) in head.iter().enumerate() {
            assert!(
                close_rel(spec[i][0], e[0]) && close_rel(spec[i][1], e[1]),
                "bin {}: {} {} vs {} {}",
                i,
                spec[i][0],
                spec[i][1],
                e[0],
                e[1]
            );
        }
        let tail: &[[f64; 2]] = &[
            [-1362.9426621607358, -373.24604196111909],
            [-13.372998435368448, 860.18432898591982],
            [-275.96715372603541, -950.36449937692464],
            [-1219.1338101271081, 615.72110571917949],
            [503.13249346708341, 0.0],
        ];
        for (i, &e) in tail.iter().enumerate() {
            let idx = 2048 / 2 - 4 + i;
            assert!(
                close_rel(spec[idx][0], e[0]) && close_rel(spec[idx][1], e[1]),
                "bin {}: {} {} vs {} {}",
                idx,
                spec[idx][0],
                spec[idx][1],
                e[0],
                e[1]
            );
        }
    }

    // DC removal: a constant signal has zero mean after removal, so the whole
    // output spectrum is zero.
    #[test]
    fn test_get_spectrum_for_estimation_constant_input_is_zero() {
        let x = vec![1.0; 100];
        let mut spec = vec![[0.0, 0.0]; 256];
        get_spectrum_for_estimation(&x, 100, 1600.0, 256, 1, &mut spec);
        for (i, bin) in spec.iter().take(129).enumerate() {
            assert!(
                bin[0].abs() < 1e-9 && bin[1].abs() < 1e-9,
                "bin {}: {:?}",
                i,
                bin
            );
        }
    }

    // ---- PHASE1-3 reference vectors from /tmp/opencode/dio_p13_vectors.out ----

    #[test]
    fn test_zero_crossing_engine_200hz_sine() {
        let fs: f64 = 16000.0;
        let l = 1600;
        let sig: Vec<f64> = (0..l)
            .map(|i| (2.0 * K_PI * 200.0 * i as f64 / fs).sin())
            .collect();
        let mut locations = vec![0.0; l];
        let mut intervals = vec![0.0; l];
        let mut edges = vec![0; l];
        let mut fine_edges = vec![0.0; l];
        let n = zero_crossing_engine(
            &sig,
            l,
            fs,
            &mut locations,
            &mut intervals,
            &mut edges,
            &mut fine_edges,
        ) as usize;
        assert_eq!(n, 19);
        let expected_locs: &[f64] = &[
            0.0050625000000000002,
            0.0100625,
            0.015062499999999998,
            0.0200625,
            0.025062500000000001,
            0.030062499999999999,
            0.035062500000000003,
            0.040062500000000001,
            0.045062499999999998,
            0.050062500000000003,
            0.0550625,
            0.060062499999999998,
            0.065062499999999995,
            0.0700625,
            0.075062500000000004,
            0.080062499999999995,
            0.085062499999999999,
            0.090062500000000004,
            0.095062499999999994,
        ];
        for i in 0..n {
            assert!(
                close(locations[i], expected_locs[i]),
                "loc[{}]: {} vs {}",
                i,
                locations[i],
                expected_locs[i]
            );
        }
        let expected_ints: &[f64] = &[
            200.0,
            200.0,
            200.00000000000014,
            199.99999999999986,
            200.0,
            200.0,
            200.0,
            199.99999999999972,
            200.00000000000028,
            200.0,
            199.99999999999972,
            200.00000000000028,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
        ];
        for i in 0..n {
            assert!(
                close(intervals[i], expected_ints[i]),
                "int[{}]: {} vs {}",
                i,
                intervals[i],
                expected_ints[i]
            );
        }
    }

    #[test]
    fn test_get_four_zero_crossing_intervals_200hz_sine() {
        let fs: f64 = 16000.0;
        let l = 1600;
        let mut sig: Vec<f64> = (0..l)
            .map(|i| (2.0 * K_PI * 200.0 * i as f64 / fs).sin())
            .collect();
        let mut zc = DioZcScratch {
            negative_interval_locations: vec![0.0; l],
            negative_intervals: vec![0.0; l],
            positive_interval_locations: vec![0.0; l],
            positive_intervals: vec![0.0; l],
            peak_interval_locations: vec![0.0; l],
            peak_intervals: vec![0.0; l],
            dip_interval_locations: vec![0.0; l],
            dip_intervals: vec![0.0; l],
            edges: vec![0; l],
            fine_edges: vec![0.0; l],
            number_of_negatives: 0,
            number_of_positives: 0,
            number_of_peaks: 0,
            number_of_dips: 0,
        };
        get_four_zero_crossing_intervals(&mut sig, l, fs, &mut zc);
        assert_eq!(zc.number_of_negatives, 19);
        assert_eq!(zc.number_of_positives, 18);
        assert_eq!(zc.number_of_peaks, 19);
        assert_eq!(zc.number_of_dips, 19);
        // Spot-check a few values.
        assert!(close(
            zc.negative_interval_locations[0],
            0.0050625000000000002
        ));
        assert!(close(zc.negative_intervals[0], 200.0));
        assert!(close(
            zc.positive_interval_locations[0],
            0.007562499999999998
        ));
        assert!(close(zc.positive_intervals[0], 200.00000000000003));
        assert!(close(zc.peak_interval_locations[0], 0.0037812499999999999));
        assert!(close(zc.peak_intervals[0], 200.0));
        assert!(close(zc.dip_interval_locations[0], 0.0062812500000000004));
        assert!(close(zc.dip_intervals[0], 200.0));
    }

    #[test]
    fn test_get_f0_candidate_contour_sub_normal() {
        let set0 = [100.0, 102.0, 98.0, 101.0, 99.0];
        let set1 = [101.0, 99.0, 103.0, 97.0, 102.0];
        let set2 = [99.5, 100.5, 101.5, 98.5, 100.0];
        let set3 = [100.5, 99.5, 99.0, 102.0, 101.0];
        let sets: [&[f64]; 4] = [&set0, &set1, &set2, &set3];
        let mut f0c = vec![0.0; 5];
        let mut f0s = vec![0.0; 5];
        get_f0_candidate_contour_sub(sets, 5, 65.0, 380.0, 100.0, &mut f0c, &mut f0s);
        // Only index 3 is in range [50, 100] ∩ [65, 380].
        assert_eq!(f0c[0], 0.0);
        assert!(close(f0s[0], K_MAXIMUM_VALUE));
        assert_eq!(f0c[1], 0.0);
        assert!(close(f0s[1], K_MAXIMUM_VALUE));
        assert_eq!(f0c[2], 0.0);
        assert!(close(f0s[2], K_MAXIMUM_VALUE));
        assert!(close(f0c[3], 99.625));
        assert!(close(f0s[3], 2.2867371223353739));
        assert_eq!(f0c[4], 0.0);
        assert!(close(f0s[4], K_MAXIMUM_VALUE));
    }

    #[test]
    fn test_get_f0_candidate_contour_sub_out_of_range() {
        let set0 = [50.0, 400.0, 30.0];
        let set1 = [50.0, 400.0, 30.0];
        let set2 = [50.0, 400.0, 30.0];
        let set3 = [50.0, 400.0, 30.0];
        let sets: [&[f64]; 4] = [&set0, &set1, &set2, &set3];
        let mut f0c = vec![0.0; 3];
        let mut f0s = vec![0.0; 3];
        get_f0_candidate_contour_sub(sets, 3, 65.0, 380.0, 100.0, &mut f0c, &mut f0s);
        for i in 0..3 {
            assert_eq!(f0c[i], 0.0);
            assert!(close(f0s[i], K_MAXIMUM_VALUE));
        }
    }

    #[test]
    fn test_get_f0_candidate_contour_boundary_insufficient() {
        let zc = DioZcScratch {
            negative_interval_locations: vec![0.0; 10],
            negative_intervals: vec![0.0; 10],
            positive_interval_locations: vec![0.0; 10],
            positive_intervals: vec![0.0; 10],
            peak_interval_locations: vec![0.0; 10],
            peak_intervals: vec![0.0; 10],
            dip_interval_locations: vec![0.0; 10],
            dip_intervals: vec![0.0; 10],
            edges: vec![0; 10],
            fine_edges: vec![0.0; 10],
            number_of_negatives: 1,
            number_of_positives: 5,
            number_of_peaks: 5,
            number_of_dips: 5,
        };
        let tp = [0.01, 0.02, 0.03, 0.04];
        let mut f0c = vec![0.0; 4];
        let mut f0s = vec![0.0; 4];
        get_f0_candidate_contour(&zc, 100.0, 65.0, 380.0, &tp, 4, &mut f0c, &mut f0s);
        for i in 0..4 {
            assert_eq!(f0c[i], 0.0);
            assert!(close(f0s[i], K_MAXIMUM_VALUE));
        }
    }

    #[test]
    fn test_get_filtered_signal_synthetic() {
        let fft_size = 1024;
        let y_length = 512;
        let half_avg = 8;
        let mut spec = vec![[0.0, 0.0]; fft_size / 2 + 1];
        for (i, slot) in spec.iter_mut().enumerate().take(fft_size / 2 + 1) {
            slot[0] = (0.1 * i as f64).cos() * (1.0 + 0.01 * i as f64);
            slot[1] = (0.1 * i as f64).sin() * (1.0 + 0.01 * i as f64);
        }
        let mut filtered = vec![0.0; fft_size];
        get_filtered_signal(half_avg, fft_size, &spec, y_length, &mut filtered);
        let head: &[f64] = &[
            -26.821602548665656,
            -23.032965991927519,
            -20.037897991340564,
            -17.666037750369171,
            -15.707212415868547,
            -14.085943628840596,
            -12.709501759012966,
            -11.539586695312025,
            -10.526680709053487,
            -9.6494978548109778,
            -8.8786480812561415,
            -8.2015254195043994,
            -7.599416241305561,
            -7.0644932839847634,
            -6.5842163598032819,
            -6.1535408395332034,
        ];
        for (i, &e) in head.iter().enumerate() {
            assert!(
                close_rel(filtered[i], e),
                "i={}: {} vs {}",
                i,
                filtered[i],
                e
            );
        }
        let tail: &[f64] = &[
            -0.05522097568983142,
            -0.055270161023841524,
            -0.055249539162037564,
            -0.055301156774323346,
        ];
        for (i, &e) in tail.iter().enumerate() {
            let idx = y_length - 4 + i;
            assert!(
                close_rel(filtered[idx], e),
                "i={}: {} vs {}",
                idx,
                filtered[idx],
                e
            );
        }
    }

    #[test]
    fn test_get_f0_candidate_from_raw_event_200hz() {
        let fs: f64 = 16000.0;
        let l = 1600;
        let fft_size = 4096;
        let x: Vec<f64> = (0..l)
            .map(|i| (2.0 * K_PI * 200.0 * i as f64 / fs).sin())
            .collect();
        let mut spec = vec![[0.0, 0.0]; fft_size];
        get_spectrum_for_estimation(&x, l, fs, fft_size, 1, &mut spec);

        let boundary_f0 = 100.0;
        let f0_floor = 65.0;
        let f0_ceil = 380.0;
        let f0_length = 20;
        let tp: Vec<f64> = (0..f0_length).map(|i| (i + 1) as f64 * 5.0 / fs).collect();
        let mut f0c = vec![0.0; f0_length];
        let mut f0s = vec![0.0; f0_length];
        get_f0_candidate_from_raw_event(
            boundary_f0,
            fs,
            &spec,
            l,
            fft_size,
            f0_floor,
            f0_ceil,
            &tp,
            f0_length,
            &mut f0s,
            &mut f0c,
        );
        let expected_c: &[f64] = &[
            68.828728077303055,
            70.206488406698213,
            71.584248736093343,
            72.962009065488488,
            74.339769394883632,
            75.717529724278791,
            77.095290053673921,
            78.473050383069065,
            79.850810712464209,
            81.228571041859354,
            82.606331371254498,
            83.984091700649657,
            85.361852030044815,
            86.73961235943996,
            88.11737268883509,
            89.495133018230263,
            90.872893347625393,
            92.250653677020523,
            93.628414006415682,
            95.006174335810826,
        ];
        let expected_s: &[f64] = &[
            247.8809504907575,
            245.36485314499788,
            242.848769310802,
            240.33269941253442,
            237.8166438925162,
            235.30060321198465,
            232.78457785211535,
            230.26856831511111,
            227.75257512536317,
            225.23659883068987,
            222.7206400036597,
            220.2046992430048,
            217.68877717513215,
            215.17287445574135,
            212.65699177155653,
            210.14112984218301,
            207.62528942209886,
            205.10947130279274,
            202.59367631506109,
            200.07790533147849,
        ];
        for i in 0..f0_length {
            assert!(
                close_rel(f0c[i], expected_c[i]),
                "f0c[{}]: {} vs {}",
                i,
                f0c[i],
                expected_c[i]
            );
            assert!(
                close_rel(f0s[i], expected_s[i]),
                "f0s[{}]: {} vs {}",
                i,
                f0s[i],
                expected_s[i]
            );
        }
    }

    #[test]
    fn test_get_f0_candidate_from_raw_event_440hz_out_of_range() {
        let fs: f64 = 16000.0;
        let l = 1600;
        let fft_size = 4096;
        let x: Vec<f64> = (0..l)
            .map(|i| (2.0 * K_PI * 440.0 * i as f64 / fs).sin())
            .collect();
        let mut spec = vec![[0.0, 0.0]; fft_size];
        get_spectrum_for_estimation(&x, l, fs, fft_size, 1, &mut spec);

        let boundary_f0 = 200.0;
        let f0_floor = 65.0;
        let f0_ceil = 380.0;
        let f0_length = 10;
        let tp: Vec<f64> = (0..f0_length).map(|i| (i + 1) as f64 * 10.0 / fs).collect();
        let mut f0c = vec![0.0; f0_length];
        let mut f0s = vec![0.0; f0_length];
        get_f0_candidate_from_raw_event(
            boundary_f0,
            fs,
            &spec,
            l,
            fft_size,
            f0_floor,
            f0_ceil,
            &tp,
            f0_length,
            &mut f0s,
            &mut f0c,
        );
        for i in 0..f0_length {
            assert_eq!(f0c[i], 0.0, "f0c[{}]", i);
            assert!(close(f0s[i], K_MAXIMUM_VALUE), "f0s[{}]: {}", i, f0s[i]);
        }
    }

    // ---- PHASE1-4 reference vectors from /tmp/opencode/dio_p14_vectors.out ----

    #[test]
    fn test_get_number_of_bands() {
        // C++: 1 + int(log(f0_ceil / f0_floor) / kLog2 * channels_in_octave).
        let cases = [
            (71.0, 800.0, 2.0, 7usize),
            (71.0, 800.0, 1.0, 4),
            (65.0, 380.0, 2.0, 6),
            (50.0, 1000.0, 3.0, 13),
            (100.0, 200.0, 2.0, 3),
            (71.0, 71.0, 2.0, 1),
        ];
        for (fl, fc, cio, expected) in cases {
            assert_eq!(
                get_number_of_bands(fl, fc, cio),
                expected,
                "fl={fl} fc={fc} cio={cio}"
            );
        }
    }

    #[test]
    fn test_get_boundary_f0_list() {
        // Default option: f0_floor=71, channels_in_octave=2 -> 7 bands.
        let bfl = get_boundary_f0_list(71.0, 2.0, 7);
        let expected: &[f64] = &[
            100.40916292848975,
            142.0,
            200.8183258569795,
            284.0,
            401.636651713959,
            568.0,
            803.27330342791799,
        ];
        assert_eq!(bfl.len(), 7);
        for (i, &e) in expected.iter().enumerate() {
            assert!(close(bfl[i], e), "i={}: {} vs {}", i, bfl[i], e);
        }
    }

    #[test]
    fn test_get_decimation_ratio() {
        // C++: max(min(speed, 12), 1).
        let cases = [
            (-3i32, 1usize),
            (0, 1),
            (1, 1),
            (2, 2),
            (5, 5),
            (11, 11),
            (12, 12),
            (13, 12),
            (100, 12),
        ];
        for (speed, expected) in cases {
            assert_eq!(get_decimation_ratio(speed), expected, "speed={speed}");
        }
    }

    #[test]
    fn test_get_dio_fft_size() {
        // f0_floor=71, channels_in_octave=2 -> boundary_f0_list[0]=71*2^0.5.
        let bfl = get_boundary_f0_list(71.0, 2.0, get_number_of_bands(71.0, 800.0, 2.0));
        // (x_length, fs, speed) -> expected fft_size (C++ reference).
        let cases = [
            (8000usize, 8000.0, 1i32, 16384usize),
            (16000, 16000.0, 1, 32768),
            (16000, 16000.0, 2, 16384),
            (16000, 16000.0, 12, 2048),
            (44100, 44100.0, 1, 65536),
            (48000, 48000.0, 4, 16384),
            (48000, 48000.0, 12, 8192),
        ];
        for (x_length, fs, speed, expected) in cases {
            let dec = get_decimation_ratio(speed);
            let actual_fs = fs / dec as f64;
            let y_length = 1 + x_length / dec;
            let got = get_dio_fft_size(y_length, actual_fs, &bfl);
            assert_eq!(got, expected, "x_length={x_length} fs={fs} speed={speed}");
        }
    }

    #[test]
    fn test_get_best_f0_contour_tie_breaking() {
        // 2 bands, 3 frames. Ties go to the first band (strict `>`).
        let cand: Vec<Vec<f64>> = vec![vec![100.0, 200.0, 300.0], vec![150.0, 250.0, 350.0]];
        let sc: Vec<Vec<f64>> = vec![vec![5.0, 1.0, 10.0], vec![3.0, 1.0, 8.0]];
        let mut best = vec![0.0; 3];
        get_best_f0_contour(3, &cand, &sc, &mut best);
        // frame 0: min(5,3)=3 -> band1 -> 150
        // frame 1: min(1,1)=1 -> tie -> band0 -> 200
        // frame 2: min(10,8)=8 -> band1 -> 350
        assert_eq!(best, [150.0, 200.0, 350.0]);
    }

    // Full multi-band pipeline on a 200Hz sine (fs=16000, x_length=1600,
    // default bands 71..800 cio=2). Exercises get_f0_candidates_and_scores
    // (band iteration + score normalization) and get_best_f0_contour together.
    #[test]
    fn test_get_f0_candidates_and_scores_200hz() {
        let fs: f64 = 16000.0;
        let l = 1600;
        let x: Vec<f64> = (0..l)
            .map(|i| (2.0 * K_PI * 200.0 * i as f64 / fs).sin())
            .collect();

        let f0_floor = 71.0;
        let f0_ceil = 800.0;
        let cio = 2.0;
        let number_of_bands = get_number_of_bands(f0_floor, f0_ceil, cio);
        assert_eq!(number_of_bands, 7);
        let bfl = get_boundary_f0_list(f0_floor, cio, number_of_bands);

        let dec = get_decimation_ratio(1);
        let actual_fs = fs / dec as f64;
        let y_length = 1 + l / dec;
        let fft_size = get_dio_fft_size(y_length, actual_fs, &bfl);
        assert_eq!(fft_size, 4096);

        let mut spec = vec![[0.0, 0.0]; fft_size];
        get_spectrum_for_estimation(&x, y_length, actual_fs, fft_size, dec, &mut spec);

        let f0_length = 20;
        let tp: Vec<f64> = (0..f0_length).map(|i| i as f64 * 5.0 / 1000.0).collect();
        let mut f0_candidates = vec![vec![0.0; f0_length]; number_of_bands];
        let mut f0_scores = vec![vec![0.0; f0_length]; number_of_bands];
        get_f0_candidates_and_scores(
            &bfl,
            actual_fs,
            y_length,
            &tp,
            f0_length,
            &spec,
            fft_size,
            f0_floor,
            f0_ceil,
            &mut f0_candidates,
            &mut f0_scores,
        );

        // Per-band normalization spot-checks (band 2 = boundary ~200.8,
        // band 3 = boundary ~284).
        assert!(
            close_rel(f0_candidates[2][0], 194.43080472053677),
            "band2 c0: {}",
            f0_candidates[2][0]
        );
        assert!(
            close_rel(f0_scores[2][0], 0.084426225349322703),
            "band2 s0: {}",
            f0_scores[2][0]
        );
        assert!(
            close_rel(f0_candidates[3][0], 197.31844489628489),
            "band3 c0: {}",
            f0_candidates[3][0]
        );
        assert!(
            close_rel(f0_scores[3][0], 0.030138258161605039),
            "band3 s0: {}",
            f0_scores[3][0]
        );
        // Out-of-range bands are zeroed with score kMaximumValue / (0 + guard)
        // = 1e17.
        assert_eq!(f0_candidates[0][0], 0.0);
        assert!(close(f0_scores[0][0], 1e17));

        let mut best = vec![0.0; f0_length];
        get_best_f0_contour(f0_length, &f0_candidates, &f0_scores, &mut best);
        let expected_best: &[f64] = &[
            197.31844489628489,
            198.85195367439272,
            199.98064966172711,
            199.98540270192768,
            200.03546192529635,
            199.99868402234449,
            199.99999999978309,
            200.0,
            200.00000000000003,
            199.99999999999994,
            200.0,
            199.99999999999994,
            200.0,
            200.0,
            199.99999999999991,
            199.99881710643697,
            200.03532475451709,
            199.98534126220929,
            199.98154928958817,
            198.85081323366342,
        ];
        for i in 0..f0_length {
            assert!(
                close_rel(best[i], expected_best[i]),
                "best[{}]: {} vs {}",
                i,
                best[i],
                expected_best[i]
            );
        }
    }

    // ---- PHASE1-5 reference vectors from /tmp/opencode/dio_p15_vectors.out ----

    #[test]
    fn test_voice_range_minimum_matches_cpp_reference() {
        // C++: static_cast<int>(0.5 + 1000.0 / frame_period / f0_floor) * 2 + 1
        let cases = [
            (5.0, 71.0, 7usize),
            (5.0, 65.0, 7),
            (10.0, 71.0, 3),
            (1.0, 71.0, 29),
            (5.0, 100.0, 5),
            (2.0, 50.0, 21),
        ];
        for (fp, fl, expected) in cases {
            let vrm = (0.5 + 1000.0 / fp / fl) as usize * 2 + 1;
            assert_eq!(vrm, expected, "fp={fp} fl={fl}");
        }
    }

    #[test]
    fn test_fix_step1_synthetic() {
        // best = [0,0,100,100,100,100,100,100,0,0], vrm=3, ar=0.1.
        // f0_base = [0,0,0,100,100,100,100,0,0,0]; the 0->100 and 100->0
        // jumps exceed allowed_range, leaving only the flat middle.
        let best: &[f64] = &[0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 0.0, 0.0];
        let mut step1 = vec![0.0; 10];
        fix_step1(best, 10, 3, 0.1, &mut step1);
        let expected: &[f64] = &[0.0, 0.0, 0.0, 0.0, 100.0, 100.0, 100.0, 0.0, 0.0, 0.0];
        for i in 0..10 {
            assert_eq!(
                step1[i], expected[i],
                "i={}: {} vs {}",
                i, step1[i], expected[i]
            );
        }
    }

    #[test]
    fn test_fix_step2_synthetic() {
        // step1 = [0,0,0,100,100,100,100,0,0,0], vrm=3 -> center=1. A frame
        // survives only if its 3-wide window is fully voiced, trimming the
        // region [3,7) down to [4,6).
        let step1: &[f64] = &[0.0, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 0.0, 0.0, 0.0];
        let mut step2 = vec![0.0; 10];
        fix_step2(step1, 10, 3, &mut step2);
        let expected: &[f64] = &[0.0, 0.0, 0.0, 0.0, 100.0, 100.0, 0.0, 0.0, 0.0, 0.0];
        for i in 0..10 {
            assert_eq!(
                step2[i], expected[i],
                "i={}: {} vs {}",
                i, step2[i], expected[i]
            );
        }
    }

    #[test]
    fn test_get_number_of_voiced_sections() {
        // f0 = [0,100,100,0,0,100,0,100,100,100].
        // Sections: [1,3), [5,6), [7,10).
        let f0: &[f64] = &[0.0, 100.0, 100.0, 0.0, 0.0, 100.0, 0.0, 100.0, 100.0, 100.0];
        let mut positive_index = vec![0usize; 10];
        let mut negative_index = vec![0usize; 10];
        let (pc, nc) =
            get_number_of_voiced_sections(f0, 10, &mut positive_index, &mut negative_index);
        assert_eq!(pc, 3);
        assert_eq!(nc, 2);
        assert_eq!(&positive_index[..pc], &[1, 5, 7]);
        assert_eq!(&negative_index[..nc], &[2, 5]);
    }

    #[test]
    fn test_select_best_f0() {
        // case 1: reference=(100*3-98)/2=101; candidates {100,101} -> 101, in range.
        let c: Vec<Vec<f64>> = vec![vec![100.0], vec![101.0]];
        assert_eq!(select_best_f0(100.0, 98.0, &c, 2, 0, 0.1), 101.0);
        // case 2: reference=100; candidates {200,300} -> best 200, out of range -> 0.
        let d: Vec<Vec<f64>> = vec![vec![200.0], vec![300.0]];
        assert_eq!(select_best_f0(100.0, 100.0, &d, 2, 0, 0.1), 0.0);
        // case 3: reference=100; candidates {100,105} -> best 100 (min error), in range.
        let e: Vec<Vec<f64>> = vec![vec![100.0], vec![105.0]];
        assert_eq!(select_best_f0(100.0, 100.0, &e, 2, 0, 0.1), 100.0);
    }

    #[test]
    fn test_fix_step3_direct() {
        // Voiced at [6,9); negative_index=[8] (last voiced frame). Candidates
        // in-range only for indices 6..=12, so the section extends to [6,13).
        let l = 20;
        let mut step2 = vec![0.0; l];
        step2[6] = 100.0;
        step2[7] = 100.0;
        step2[8] = 100.0;
        let mut band0 = vec![0.0; l];
        let mut band1 = vec![0.0; l];
        for i in 6..=12 {
            band0[i] = 100.0;
            band1[i] = 101.0;
        }
        let cands = vec![band0, band1];
        let negative_index = [8usize];
        let mut step3 = vec![0.0; l];
        fix_step3(&step2, l, &cands, 2, 0.1, &negative_index, 1, &mut step3);
        let expected: &[f64] = &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        for i in 0..l {
            assert_eq!(
                step3[i], expected[i],
                "i={}: {} vs {}",
                i, step3[i], expected[i]
            );
        }
    }

    #[test]
    fn test_fix_step4_direct() {
        // Voiced at [12,15); positive_index=[12] (first voiced frame). Candidates
        // in-range only for indices 7..=14, so the section extends to [7,15).
        let l = 20;
        let mut step3 = vec![0.0; l];
        step3[12] = 100.0;
        step3[13] = 100.0;
        step3[14] = 100.0;
        let mut band0 = vec![0.0; l];
        let mut band1 = vec![0.0; l];
        for i in 7..=14 {
            band0[i] = 100.0;
            band1[i] = 101.0;
        }
        let cands = vec![band0, band1];
        let positive_index = [12usize];
        let mut step4 = vec![0.0; l];
        fix_step4(&step3, l, &cands, 2, 0.1, &positive_index, 1, &mut step4);
        let expected: &[f64] = &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
            100.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        for i in 0..l {
            assert_eq!(
                step4[i], expected[i],
                "i={}: {} vs {}",
                i, step4[i], expected[i]
            );
        }
    }

    #[test]
    fn test_fix_f0_contour_hand_crafted() {
        // f0_length=30, vrm=7, center=3. best=200 everywhere -> FixStep1 keeps
        // [7,23), FixStep2 keeps [11,20). Candidates in-range for 8..=24, so
        // FixStep3/4 extend the region to [8,25).
        let l = 30;
        let mut band0 = vec![0.0; l];
        let mut band1 = vec![0.0; l];
        for i in 8..=24 {
            band0[i] = 200.0;
            band1[i] = 201.0;
        }
        let cands = vec![band0, band1];
        let best = vec![200.0; l];
        let mut fixed = vec![0.0; l];
        fix_f0_contour(5.0, &cands, &best, l, 71.0, 0.1, &mut fixed);
        let mut expected = vec![0.0; l];
        for slot in expected.iter_mut().take(24 + 1).skip(8) {
            *slot = 200.0;
        }
        for i in 0..l {
            assert_eq!(
                fixed[i], expected[i],
                "i={}: {} vs {}",
                i, fixed[i], expected[i]
            );
        }
    }

    #[test]
    fn test_fix_f0_contour_early_return_zero_fills() {
        // f0_length=5, frame_period=5, f0_floor=71 -> vrm=7; 5 <= 7 triggers the
        // early return, which zero-fills the (sentinel-pre-filled) output.
        let f0_length = 5;
        let cands = vec![vec![100.0; f0_length], vec![101.0; f0_length]];
        let best = vec![200.0; f0_length];
        let mut fixed = vec![999.0; f0_length];
        fix_f0_contour(5.0, &cands, &best, f0_length, 71.0, 0.1, &mut fixed);
        for (i, &v) in fixed.iter().enumerate().take(f0_length) {
            assert_eq!(v, 0.0, "i={}: {}", i, v);
        }
    }

    // Full multi-band pipeline on a 200Hz sine (fs=16000, x_length=3200, 40
    // frames) followed by fix_f0_contour. Exercises all four post-processing
    // steps end-to-end and must match the C++ reference FIXED contour.
    #[test]
    fn test_fix_f0_contour_full_pipeline_200hz() {
        let fs: f64 = 16000.0;
        let l = 3200;
        let x: Vec<f64> = (0..l)
            .map(|i| (2.0 * K_PI * 200.0 * i as f64 / fs).sin())
            .collect();

        let f0_floor = 71.0;
        let f0_ceil = 800.0;
        let cio = 2.0;
        let number_of_bands = get_number_of_bands(f0_floor, f0_ceil, cio);
        let bfl = get_boundary_f0_list(f0_floor, cio, number_of_bands);

        let dec = get_decimation_ratio(1);
        let actual_fs = fs / dec as f64;
        let y_length = 1 + l / dec;
        let fft_size = get_dio_fft_size(y_length, actual_fs, &bfl);
        assert_eq!(fft_size, 8192);

        let mut spec = vec![[0.0, 0.0]; fft_size];
        get_spectrum_for_estimation(&x, y_length, actual_fs, fft_size, dec, &mut spec);

        let f0_length = 40;
        let tp: Vec<f64> = (0..f0_length).map(|i| i as f64 * 5.0 / 1000.0).collect();
        let mut f0_candidates = vec![vec![0.0; f0_length]; number_of_bands];
        let mut f0_scores = vec![vec![0.0; f0_length]; number_of_bands];
        get_f0_candidates_and_scores(
            &bfl,
            actual_fs,
            y_length,
            &tp,
            f0_length,
            &spec,
            fft_size,
            f0_floor,
            f0_ceil,
            &mut f0_candidates,
            &mut f0_scores,
        );

        let mut best = vec![0.0; f0_length];
        get_best_f0_contour(f0_length, &f0_candidates, &f0_scores, &mut best);

        let mut fixed = vec![0.0; f0_length];
        fix_f0_contour(
            5.0,
            &f0_candidates,
            &best,
            f0_length,
            f0_floor,
            0.1,
            &mut fixed,
        );

        let expected: &[f64] = &[
            0.0,
            198.85195367439349,
            199.97301099386618,
            199.98540270192777,
            200.03546192529643,
            199.99868402234443,
            199.99999999978311,
            199.99999999999994,
            200.0,
            200.0,
            200.0,
            200.00000000000006,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            199.99999999999994,
            200.00000000000023,
            199.99881710643677,
            200.03532475451692,
            199.98534126220949,
            199.97498789301258,
            198.85081323366433,
        ];
        assert_eq!(expected.len(), f0_length);
        for i in 0..f0_length {
            assert!(
                close_rel(fixed[i], expected[i]),
                "fixed[{}]: {} vs {}",
                i,
                fixed[i],
                expected[i]
            );
        }
    }

    // ---- PHASE1-6: public API integration tests ----

    // Full public `dio()` on a 200 Hz sine (fs=16000, 0.2 s). Verifies the
    // structural contract (f0_length, temporal grid, band count) and that a
    // voiced sine yields a finite ~200 Hz contour with the first frame zeroed
    // by the post-processing step.
    #[test]
    fn test_dio_200hz_sine_integration() {
        let fs: f64 = 16000.0;
        let l = 3200;
        let x: Vec<f64> = (0..l)
            .map(|i| (2.0 * K_PI * 200.0 * i as f64 / fs).sin())
            .collect();
        let option = initialize_dio_option();
        let result = dio(&x, fs, &option).expect("valid input should succeed");

        let expected_f0_length = get_samples_for_dio(fs, l, option.frame_period);
        assert_eq!(result.f0_length, expected_f0_length);
        assert_eq!(result.f0.len(), expected_f0_length);
        assert_eq!(result.temporal_positions.len(), expected_f0_length);

        let expected_bands =
            get_number_of_bands(option.f0_floor, option.f0_ceil, option.channels_in_octave);
        assert_eq!(result.number_of_bands, expected_bands);
        assert_eq!(result.f0_candidates.len(), expected_bands);
        assert_eq!(result.f0_scores.len(), expected_bands);
        for band in result.f0_candidates.iter().chain(result.f0_scores.iter()) {
            assert_eq!(band.len(), expected_f0_length);
        }

        // Temporal grid: i * frame_period / 1000.0.
        for i in 0..expected_f0_length {
            assert!(
                close(
                    result.temporal_positions[i],
                    i as f64 * option.frame_period / 1000.0
                ),
                "tp[{}]: {}",
                i,
                result.temporal_positions[i]
            );
        }

        // Every F0 value is finite; the middle frame of a 200 Hz sine is
        // voiced near 200 Hz; the first frame is zeroed by FixStep1.
        for v in result.f0.iter() {
            assert!(v.is_finite(), "non-finite f0: {:?}", v);
        }
        let mid = expected_f0_length / 2;
        assert!(
            result.f0[mid] > 150.0 && result.f0[mid] < 250.0,
            "mid f0 = {}",
            result.f0[mid]
        );
        assert_eq!(result.f0[0], 0.0);
    }

    // ---- PHASE1-7: invalid input, edge cases, no-panic, property tests ----

    // Deterministic pseudo-random noise in [-1, 1) (no external crate, wasm-safe).
    fn lcg_noise(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 11) as f64) / (1u64 << 53) as f64 * 2.0 - 1.0
            })
            .collect()
    }

    fn sine(freq: f64, fs: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * K_PI * freq * i as f64 / fs).sin())
            .collect()
    }

    // Empty input returns a specific error (graceful, no panic).
    #[test]
    fn test_dio_empty_input_returns_error() {
        let option = initialize_dio_option();
        assert_eq!(dio(&[], 16000.0, &option), Err(DioError::EmptyInput));
    }

    // Zero and negative sample rates return a specific error.
    #[test]
    fn test_dio_non_positive_fs_returns_error() {
        let x = vec![0.0; 100];
        let option = initialize_dio_option();
        assert_eq!(
            dio(&x, 0.0, &option),
            Err(DioError::NonPositiveSampleRate { fs: 0.0 })
        );
        assert_eq!(
            dio(&x, -16000.0, &option),
            Err(DioError::NonPositiveSampleRate { fs: -16000.0 })
        );
    }

    // Zero and negative frame periods return a specific error.
    #[test]
    fn test_dio_non_positive_frame_period_returns_error() {
        let x = vec![0.0; 100];
        let mut option = initialize_dio_option();
        option.frame_period = 0.0;
        assert_eq!(
            dio(&x, 16000.0, &option),
            Err(DioError::NonPositiveFramePeriod { frame_period: 0.0 })
        );
        option.frame_period = -5.0;
        assert_eq!(
            dio(&x, 16000.0, &option),
            Err(DioError::NonPositiveFramePeriod { frame_period: -5.0 })
        );
    }

    // DioError implements Display and std::error::Error.
    #[test]
    fn test_dio_error_display() {
        let e = DioError::EmptyInput;
        assert_eq!(e.to_string(), "input signal is empty");
        let e = DioError::NonPositiveSampleRate { fs: -3.0 };
        assert_eq!(e.to_string(), "sample rate must be positive, got -3");
        let e = DioError::NonPositiveFramePeriod { frame_period: 0.0 };
        assert_eq!(e.to_string(), "frame period must be positive, got 0");
        // Usable as a boxed error.
        let boxed: Box<dyn std::error::Error> = Box::new(DioError::EmptyInput);
        assert_eq!(boxed.to_string(), "input signal is empty");
    }

    // Silence: all-zero input. The pipeline runs (x non-empty, fs > 0) and
    // every frame is unvoiced (0.0); no panic.
    #[test]
    fn test_dio_silence_input_all_unvoiced() {
        let fs: f64 = 16000.0;
        let x = vec![0.0; 3200];
        let option = initialize_dio_option();
        let result = dio(&x, fs, &option).expect("silence is valid input");
        let expected_f0_length = get_samples_for_dio(fs, x.len(), option.frame_period);
        assert_eq!(result.f0_length, expected_f0_length);
        for v in result.f0.iter() {
            assert_eq!(*v, 0.0, "silence should be fully unvoiced");
        }
        assert_eq!(
            result.number_of_bands,
            get_number_of_bands(option.f0_floor, option.f0_ceil, option.channels_in_octave)
        );
    }

    // Unvoiced noise: deterministic pseudo-random signal. Must not panic; every
    // F0 is finite and the signal is overwhelmingly unvoiced.
    #[test]
    fn test_dio_unvoiced_noise_no_panic_mostly_unvoiced() {
        let fs: f64 = 16000.0;
        let x = lcg_noise(3200, 0x1234_5678);
        let option = initialize_dio_option();
        let result = dio(&x, fs, &option).expect("noise is valid input");
        let n = result.f0_length;
        assert!(n > 0);
        for v in result.f0.iter() {
            assert!(v.is_finite(), "non-finite f0: {:?}", v);
        }
        let voiced = result.f0.iter().filter(|v| **v > 0.0).count();
        assert!(voiced * 2 < n, "voiced={voiced} of {n} for pure noise");
    }

    // Rapid pitch change: 200 Hz for the first half, 400 Hz for the second
    // (with a phase discontinuity at the boundary). Must not panic; every F0 is
    // finite and both halves contain voiced frames near their true frequencies.
    #[test]
    fn test_dio_rapid_pitch_change_no_panic() {
        let fs: f64 = 16000.0;
        let half = 1600;
        let mut x: Vec<f64> = Vec::with_capacity(half * 2);
        x.extend(sine(200.0, fs, half));
        x.extend(sine(400.0, fs, half));
        let option = initialize_dio_option();
        let result = dio(&x, fs, &option).expect("two-tone is valid input");
        for v in result.f0.iter() {
            assert!(v.is_finite(), "non-finite f0: {:?}", v);
        }
        let n = result.f0_length;
        let first_max = result.f0[..n / 2].iter().cloned().fold(0.0f64, f64::max);
        let second_max = result.f0[n / 2..].iter().cloned().fold(0.0f64, f64::max);
        assert!(first_max > 120.0, "first half max = {first_max}");
        assert!(second_max > 250.0, "second half max = {second_max}");
    }

    // Very low F0 (near the floor) and very high F0 (near the ceiling). Must not
    // panic; every F0 finite; a clearly in-range tone yields voiced frames.
    #[test]
    fn test_dio_very_low_f0() {
        let fs: f64 = 16000.0;
        let x = sine(75.0, fs, 3200);
        let option = initialize_dio_option();
        let result = dio(&x, fs, &option).expect("low tone is valid input");
        for v in result.f0.iter() {
            assert!(v.is_finite(), "non-finite f0: {:?}", v);
        }
        let mid = result.f0_length / 2;
        assert!(
            result.f0[mid] > 50.0 && result.f0[mid] < 130.0,
            "mid f0 = {}",
            result.f0[mid]
        );
    }

    #[test]
    fn test_dio_very_high_f0() {
        let fs: f64 = 16000.0;
        let x = sine(750.0, fs, 3200);
        let option = initialize_dio_option();
        let result = dio(&x, fs, &option).expect("high tone is valid input");
        for v in result.f0.iter() {
            assert!(v.is_finite(), "non-finite f0: {:?}", v);
        }
        let mid = result.f0_length / 2;
        assert!(
            result.f0[mid] > 500.0 && result.f0[mid] < 900.0,
            "mid f0 = {}",
            result.f0[mid]
        );
    }

    // No panics on boundary conditions: single sample, sub-frame input, and
    // extreme frame periods (which drive f0_length into the early-return path of
    // `fix_f0_contour`).
    #[test]
    fn test_dio_boundary_single_sample_no_panic() {
        let option = initialize_dio_option();
        let err = dio(&[1.0], 16000.0, &option).unwrap_err();
        assert_eq!(err, DioError::TooShortInput);
    }

    #[test]
    fn test_dio_boundary_sub_frame_input_no_panic() {
        let option = initialize_dio_option();
        let err = dio(&[0.5, -0.5, 0.25], 16000.0, &option).unwrap_err();
        assert_eq!(err, DioError::TooShortInput);
    }

    #[test]
    fn test_dio_boundary_huge_frame_period_no_panic() {
        // frame_period so large that input is too short for one frame.
        let mut option = initialize_dio_option();
        option.frame_period = 1000.0;
        let x = sine(200.0, 16000.0, 3200);
        let err = dio(&x, 16000.0, &option).unwrap_err();
        assert_eq!(err, DioError::TooShortInput);
    }

    #[test]
    fn test_dio_boundary_tiny_frame_period_no_panic() {
        let mut option = initialize_dio_option();
        option.frame_period = 0.1;
        let x = sine(200.0, 16000.0, 3200);
        let result = dio(&x, 16000.0, &option).expect("tiny frame period is valid");
        assert!(result.f0_length > 100);
        for v in result.f0.iter() {
            assert!(v.is_finite());
        }
    }

    // Property: `temporal_positions` is monotonically non-decreasing for every
    // valid (fs, x_length, frame_period) combination.
    #[test]
    fn property_temporal_positions_monotonic() {
        let fs_values = [8000.0, 16000.0, 44100.0];
        let x_lengths = [1usize, 10, 100, 1600];
        let frame_periods: [f64; 4] = [0.1, 1.0, 5.0, 100.0];
        for &fs in &fs_values {
            for &x_length in &x_lengths {
                for &fp in &frame_periods {
                    let min_samples = ((fs * fp) / 1000.0).ceil() as usize;
                    if x_length < min_samples {
                        continue;
                    }
                    let x = vec![0.0; x_length];
                    let option = DioOption {
                        frame_period: fp,
                        ..initialize_dio_option()
                    };
                    let result = dio(&x, fs, &option).expect("valid input");
                    for w in result.temporal_positions.windows(2) {
                        assert!(
                            w[0] <= w[1],
                            "temporal_positions not monotonic: {w:?} (fs={fs}, x_length={x_length}, fp={fp})"
                        );
                    }
                }
            }
        }
    }

    // Property: every per-band F0 score is non-negative across a variety of
    // signals (silence, noise, and tones spanning the F0 range).
    #[test]
    fn property_f0_scores_non_negative() {
        let fs: f64 = 16000.0;
        let n = 1600;
        let signals: Vec<Vec<f64>> = vec![
            vec![0.0; n],
            lcg_noise(n, 0xA5A5),
            sine(75.0, fs, n),
            sine(200.0, fs, n),
            sine(440.0, fs, n),
            sine(750.0, fs, n),
        ];
        let option = initialize_dio_option();
        for (idx, x) in signals.iter().enumerate() {
            let result = dio(x, fs, &option).expect("valid input");
            for (b, band) in result.f0_scores.iter().enumerate() {
                for (j, &s) in band.iter().enumerate() {
                    assert!(
                        s >= 0.0,
                        "negative score: signal={idx} band={b} frame={j} score={s}"
                    );
                }
            }
        }
    }
}
