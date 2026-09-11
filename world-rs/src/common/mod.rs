use crate::constants::*;
use crate::matlab::interp1q_into;

/// Returns the smallest power of two strictly greater than `sample`
/// (`0` returns `1`).
pub fn get_suitable_fft_size(sample: usize) -> usize {
    1 << (usize::BITS - sample.leading_zeros())
}

/// Worst-case scratch capacity (in `f64` elements) for [`dc_correction_into`],
/// assuming `current_f0 <= fs/2` (Nyquist). At that bound
/// `upper_limit = 2 + (f0 * fft_size / fs) <= 2 + fft_size/2`, and the three
/// working regions (`axis`, `replica`, `interp1q` delta) are all monotonic in
/// `upper_limit`, so this is the maximum the hot path will ever need. Callers
/// pre-allocate a buffer of this size once; out-of-range F0 (above Nyquist)
/// falls back to a local allocation inside [`dc_correction_into`].
pub fn dc_correction_scratch_capacity(fft_size: usize, input_len: usize) -> usize {
    let upper_limit_max = 2 + fft_size / 2;
    let upper_limit_replica_max = upper_limit_max.saturating_sub(1);
    let y_len_max = (upper_limit_max + 1).min(input_len);
    upper_limit_max + upper_limit_replica_max + y_len_max
}

/// C++: `DCCorrection` (common.cpp:56-75).
///
/// Corrects the DC (direct-current) component of a power spectrum. The low
/// bins `0..upper_limit-1` become `input + replica`, where `replica` is a
/// shifted copy of the low-frequency region produced by `interp1q` with a
/// **negative** shift of `-fs/fft_size` (so the interpolation samples the
/// region just below the DC bin). `upper_limit = 2 + (f0 * fft_size / fs) as
/// usize`; bins at or beyond `upper_limit - 1` are the unmodified input.
/// CheapTrick calls this in place on the power spectrum (cheaptrick.cpp:81).
pub fn dc_correction(input: &[f64], current_f0: f64, fs: i32, fft_size: usize, output: &mut [f64]) {
    let capacity = dc_correction_scratch_capacity(fft_size, input.len());
    let mut scratch = vec![0.0f64; capacity];
    dc_correction_into(input, current_f0, fs, fft_size, output, &mut scratch);
}

/// Allocation-free form of [`dc_correction`]. The working regions (the low
/// frequency axis, the interp1q replica, and the interp1q difference buffer)
/// are carved out of the caller-supplied `scratch`; when `scratch` is too small
/// for the current `upper_limit` (e.g. F0 above Nyquist) a local buffer is
/// allocated instead, preserving behaviour. See
/// [`dc_correction_scratch_capacity`] for sizing the buffer so the hot path
/// never allocates.
pub fn dc_correction_into(
    input: &[f64],
    current_f0: f64,
    fs: i32,
    fft_size: usize,
    output: &mut [f64],
    scratch: &mut [f64],
) {
    let fs_f = fs as f64;
    let fft_size_f = fft_size as f64;
    // Clamp to the Nyquist-derived bound assumed by
    // `dc_correction_scratch_capacity`: `current_f0` above Nyquist (or a
    // saturating float→int cast from a huge-but-finite f0) would otherwise
    // size the fallback buffer at ~`usize::MAX`. The C++ reads out of bounds
    // there (UB); the Rust port treats above-Nyquist as Nyquist.
    let upper_limit = 2_usize
        .saturating_add((current_f0 * fft_size_f / fs_f) as usize)
        .max(1)
        .min(2 + fft_size / 2);
    let upper_limit_replica = upper_limit.saturating_sub(1);
    let y_len = (upper_limit + 1).min(input.len());
    let needed = upper_limit + upper_limit_replica + y_len;

    // Reuse the caller's scratch when it is large enough; otherwise (F0 above
    // Nyquist) fall back to a local buffer so behaviour is preserved.
    let mut fallback = Vec::new();
    let buf: &mut [f64] = if scratch.len() >= needed {
        &mut scratch[..needed]
    } else {
        fallback.resize(needed, 0.0);
        &mut fallback[..]
    };
    dc_correction_run(
        input,
        current_f0,
        fs_f,
        fft_size_f,
        upper_limit,
        upper_limit_replica,
        y_len,
        output,
        buf,
    );
}

#[allow(clippy::too_many_arguments)]
fn dc_correction_run(
    input: &[f64],
    current_f0: f64,
    fs_f: f64,
    fft_size_f: f64,
    upper_limit: usize,
    upper_limit_replica: usize,
    y_len: usize,
    output: &mut [f64],
    buf: &mut [f64],
) {
    let (axis, rest) = buf.split_at_mut(upper_limit);
    let (replica, delta) = rest.split_at_mut(upper_limit_replica);
    let delta = &mut delta[..y_len];
    let step = fs_f / fft_size_f;
    for (i, slot) in axis.iter_mut().enumerate() {
        *slot = i as f64 * step;
    }
    let x0 = current_f0 - axis[0];
    let shift = -step;
    interp1q_into(
        x0,
        shift,
        &input[..y_len],
        &axis[..upper_limit_replica],
        replica,
        delta,
    );
    let copy_len = input.len().min(output.len());
    output[..copy_len].copy_from_slice(&input[..copy_len]);
    for i in 0..upper_limit_replica {
        if i < output.len() {
            output[i] += replica[i];
        }
    }
}

/// Worst-case scratch capacity (in `f64` elements) for
/// [`linear_smoothing_into`], assuming the CheapTrick window
/// `width = f0 * 2/3` with `f0 <= fs/2` (Nyquist), so `width <= fs/3` and
/// `boundary = (width * fft_size / fs) + 1 <= fft_size/3 + 1`. The seven
/// working regions (mirrored spectrum, cumulative segment, frequency axis,
/// low/high levels, and the two interp1q difference buffers) are all monotonic
/// in `boundary`, so this is the maximum the hot path will ever need. Callers
/// pre-allocate a buffer of this size once; a wider window (F0 above Nyquist)
/// falls back to a local allocation inside [`linear_smoothing_into`].
pub fn linear_smoothing_scratch_capacity(fft_size: usize) -> usize {
    let boundary_max = fft_size / 3 + 1;
    let mirroring_len_max = fft_size / 2 + boundary_max * 2 + 1;
    let n = fft_size / 2 + 1;
    4 * mirroring_len_max + 3 * n
}

/// C++: `LinearSmoothing` (common.cpp:77-111, `SetParametersForLinearSmoothing`
/// common.cpp:27-46).
///
/// Smooths a linear-axis power spectrum over a window of `width` Hz using even
/// boundary mirroring:
/// 1. Build a `fft_size/2 + 2*boundary + 1`-long mirrored spectrum from three
///    segments — a left mirror (`boundary` bins), the input, and a right
///    mirror — where `boundary = (width * fft_size / fs) as usize + 1`.
/// 2. Take the cumulative sum scaled by `fs/fft_size` (a discrete integral).
/// 3. Interpolate twice with `interp1q` (no clamping) on a frequency axis
///    offset by `-width/2` (low) and `+width/2` (high) relative to the
///    mirrored origin `-(boundary - 0.5) * fs / fft_size`.
/// 4. `output[i] = (high[i] - low[i]) / width` — a moving average over `width`.
///
/// CheapTrick calls this in place with `width = f0 * 2.0 / 3.0`
/// (cheaptrick.cpp:176).
pub fn linear_smoothing(input: &[f64], width: f64, fs: i32, fft_size: usize, output: &mut [f64]) {
    let capacity = linear_smoothing_scratch_capacity(fft_size);
    let mut scratch = vec![0.0f64; capacity];
    linear_smoothing_into(input, width, fs, fft_size, output, &mut scratch);
}

/// Allocation-free form of [`linear_smoothing`]. The seven working regions are
/// carved out of the caller-supplied `scratch`; when `scratch` is too small for
/// the current `boundary` (e.g. a window wider than the Nyquist-derived bound)
/// a local buffer is allocated instead, preserving behaviour. See
/// [`linear_smoothing_scratch_capacity`] for sizing the buffer so the hot path
/// never allocates.
///
/// For any finite `width` the function is total: mirror reads whose C++ source
/// index would be out of range (window wider than `fs/2`) are skipped, so no
/// panic or out-of-bounds access occurs for out-of-range F0.
pub fn linear_smoothing_into(
    input: &[f64],
    width: f64,
    fs: i32,
    fft_size: usize,
    output: &mut [f64],
    scratch: &mut [f64],
) {
    let fs_f = fs as f64;
    let fft_size_f = fft_size as f64;
    // Clamp to the Nyquist-derived bound assumed by
    // `linear_smoothing_scratch_capacity` (see `dc_correction_into` above
    // for why: a saturating cast would otherwise size the fallback at
    // ~`usize::MAX`). `saturating_add` guards the `+ 1` itself.
    let boundary = ((width * fft_size_f / fs_f) as usize)
        .saturating_add(1)
        .min(fft_size / 3 + 1);
    let n = fft_size / 2 + 1;
    let mirroring_len = fft_size / 2 + boundary * 2 + 1;
    let needed = 4 * mirroring_len + 3 * n;

    // Reuse the caller's scratch when it is large enough; otherwise (a window
    // wider than the Nyquist-derived bound) fall back to a local buffer.
    let mut fallback = Vec::new();
    let buf: &mut [f64] = if scratch.len() >= needed {
        &mut scratch[..needed]
    } else {
        fallback.resize(needed, 0.0);
        &mut fallback[..]
    };
    linear_smoothing_run(
        input,
        width,
        fs_f,
        fft_size,
        boundary,
        n,
        mirroring_len,
        output,
        buf,
    );
}

#[allow(clippy::too_many_arguments)]
fn linear_smoothing_run(
    input: &[f64],
    width: f64,
    fs_f: f64,
    fft_size: usize,
    boundary: usize,
    n: usize,
    mirroring_len: usize,
    output: &mut [f64],
    buf: &mut [f64],
) {
    let fft_size_f = fft_size as f64;
    let (mirroring_spectrum, rest) = buf.split_at_mut(mirroring_len);
    let (mirroring_segment, rest) = rest.split_at_mut(mirroring_len);
    let (frequency_axis, rest) = rest.split_at_mut(n);
    let (low_levels, rest) = rest.split_at_mut(n);
    let (high_levels, rest) = rest.split_at_mut(n);
    let (delta_low, delta_high) = rest.split_at_mut(mirroring_len);

    for (i, slot) in mirroring_spectrum[..boundary].iter_mut().enumerate() {
        let idx = boundary - i;
        if idx < input.len() {
            *slot = input[idx];
        }
    }
    for (idx, slot) in mirroring_spectrum[boundary..fft_size / 2 + boundary]
        .iter_mut()
        .enumerate()
    {
        if idx < input.len() {
            *slot = input[idx];
        }
    }
    for (i, slot) in mirroring_spectrum[fft_size / 2 + boundary..=fft_size / 2 + boundary * 2]
        .iter_mut()
        .enumerate()
    {
        // The C++ mirror index `fft_size/2 - (i - (fft_size/2 + boundary))` goes
        // negative (undefined behaviour) when `boundary > fft_size/2`, i.e. for a
        // window wider than `fs/2`. Skip those slots instead of underflowing; on
        // the hot path (`boundary <= fft_size/3 + 1`) this guard never triggers.
        if i <= fft_size / 2 {
            let idx = fft_size / 2 - i;
            if idx < input.len() {
                *slot = input[idx];
            }
        }
    }
    let scale = fs_f / fft_size_f;
    mirroring_segment[0] = mirroring_spectrum[0] * scale;
    for i in 1..mirroring_len {
        mirroring_segment[i] = mirroring_spectrum[i] * scale + mirroring_segment[i - 1];
    }
    for (i, slot) in frequency_axis.iter_mut().enumerate() {
        *slot = i as f64 * scale - width / 2.0;
    }
    let origin_of_mirroring_axis = -(boundary as f64 - 0.5) * scale;
    let discrete_frequency_interval = scale;
    interp1q_into(
        origin_of_mirroring_axis,
        discrete_frequency_interval,
        mirroring_segment,
        frequency_axis,
        low_levels,
        delta_low,
    );
    for slot in frequency_axis.iter_mut() {
        *slot += width;
    }
    interp1q_into(
        origin_of_mirroring_axis,
        discrete_frequency_interval,
        mirroring_segment,
        frequency_axis,
        high_levels,
        delta_high,
    );
    for i in 0..n {
        if i < output.len() {
            output[i] = (high_levels[i] - low_levels[i]) / width;
        }
    }
}

const NUTTALL_A0: f64 = 0.355768;
const NUTTALL_A1: f64 = 0.487396;
const NUTTALL_A2: f64 = 0.144232;
const NUTTALL_A3: f64 = 0.012604;

pub fn nuttall_window(y_length: usize, y: &mut [f64]) {
    if y_length < 2 {
        return;
    }
    for (n, slot) in y[..y_length].iter_mut().enumerate() {
        let phase = 2.0 * K_PI * n as f64 / (y_length as f64 - 1.0);
        let c = phase.cos();
        // cos(2θ) = 2c²−1 and cos(3θ) = 4c³−3c, so one cos() covers all three terms.
        let c2 = 2.0 * c * c - 1.0;
        let c3 = 4.0 * c * c * c - 3.0 * c;
        *slot = NUTTALL_A0 - NUTTALL_A1 * c + NUTTALL_A2 * c2 - NUTTALL_A3 * c3;
    }
}

const APERIODICITY_MIN: f64 = 0.001;
const APERIODICITY_MAX: f64 = 0.999999999999;

pub fn get_safe_aperiodicity(x: f64) -> f64 {
    x.clamp(APERIODICITY_MIN, APERIODICITY_MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Both the implementation and the C++ reference are f64, so results agree
    // to ~1e-13; 1e-9 leaves comfortable headroom while staying far below the
    // 1e-6 reference tolerance required by PHASE0.
    const TOL: f64 = 1e-9;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < TOL
    }

    #[test]
    fn test_get_suitable_fft_size() {
        assert_eq!(get_suitable_fft_size(100), 128);
        assert_eq!(get_suitable_fft_size(256), 512);
        assert_eq!(get_suitable_fft_size(0), 1);
    }

    // Strictly-greater power of two (A1 §3.1 / OQ2): exact powers of two are
    // doubled, values just below a power of two round up to it.
    #[test]
    fn test_get_suitable_fft_size_strictly_greater() {
        let cases = [
            (1usize, 2usize),
            (2, 4),
            (3, 4),
            (4, 8),
            (7, 8),
            (8, 16),
            (127, 128),
            (128, 256),
            (129, 256),
            (1023, 1024),
            (1024, 2048),
            (1025, 2048),
            (131071, 131072),
            (131072, 262144),
        ];
        for (input, expected) in cases {
            assert_eq!(get_suitable_fft_size(input), expected, "input={}", input);
        }
    }

    // Differential guard: the implementation must match the original float-based
    // algorithm (inlined below) exactly across the range where that algorithm is
    // correct. The dense low range covers every realistic FFT size; the
    // power-of-two boundaries extend into the larger range but stop at 2^45,
    // below the point (~2^48) where the original's `f64::log2` rounds `2^k - 1`
    // up to `2^k` and returns the wrong answer.
    #[test]
    fn test_get_suitable_fft_size_matches_original() {
        fn original(sample: usize) -> usize {
            let size = sample;
            if size == 0 {
                return 1;
            }
            let log2 = (size as f64).log2().floor() as usize;
            let pow2 = 1usize << (log2 + 1);
            if pow2 <= size {
                pow2 << 1
            } else {
                pow2
            }
        }
        for n in 0..100_000usize {
            assert_eq!(get_suitable_fft_size(n), original(n), "n={n}");
        }
        for k in [17usize, 24, 32, 40, 45] {
            for n in [(1usize << k) - 1, 1usize << k] {
                assert_eq!(
                    get_suitable_fft_size(n),
                    original(n),
                    "n={n} (2^{k} region)"
                );
            }
        }
    }

    // The bit_length-based implementation is integer-exact across the full
    // `usize` range, including the high end where the original float algorithm
    // loses precision (f64 rounding) and overflows near 2^63.
    #[test]
    fn test_get_suitable_fft_size_high_end_integer_exact() {
        let top = usize::BITS - 1;
        assert_eq!(get_suitable_fft_size((1usize << top) - 1), 1usize << top);
        assert_eq!(get_suitable_fft_size(1usize << (top - 1)), 1usize << top);
        assert_eq!(
            get_suitable_fft_size((1usize << (top - 1)) - 1),
            1usize << (top - 1)
        );
    }

    #[test]
    fn test_get_safe_aperiodicity() {
        assert_eq!(get_safe_aperiodicity(0.5), 0.5);
        assert_eq!(get_safe_aperiodicity(-1.0), APERIODICITY_MIN);
        assert_eq!(get_safe_aperiodicity(2.0), APERIODICITY_MAX);
        // Boundaries are inclusive (C++ GetSafeAperiodicity).
        assert_eq!(get_safe_aperiodicity(APERIODICITY_MIN), APERIODICITY_MIN);
        assert_eq!(get_safe_aperiodicity(APERIODICITY_MAX), APERIODICITY_MAX);
    }

    #[test]
    fn test_nuttall_window_basic() {
        let len = 8;
        let mut w = vec![0.0; len];
        nuttall_window(len, &mut w);
        assert_eq!(w.len(), len);
        for &v in &w {
            assert!((-1e-12..=1.0 + 1e-12).contains(&v));
        }
        // First and last samples should be ~0.0 for Nuttall
        assert!(w[0].abs() < 1e-6);
        assert!(w[len - 1].abs() < 1e-6);
    }

    #[test]
    fn test_nuttall_window_symmetry() {
        let len = 9;
        let mut w = vec![0.0; len];
        nuttall_window(len, &mut w);
        for i in 0..len / 2 {
            assert!((w[i] - w[len - 1 - i]).abs() < 1e-12);
        }
    }

    // Exact C++ coefficients (common.cpp:113-121). L=5 has closed-form values.
    #[test]
    fn test_nuttall_window_exact_coefficients() {
        let mut w = vec![0.0; 5];
        nuttall_window(5, &mut w);
        let expected = [0.0, 0.211536, 1.0, 0.211536, 0.0];
        for i in 0..5 {
            assert!(
                close(w[i], expected[i]),
                "i={}: {} vs {}",
                i,
                w[i],
                expected[i]
            );
        }
    }

    // Cross-check a longer window against the directly-evaluated C++ formula
    // (independent literals, so a wrong NUTTALL_A* constant would fail here).
    #[test]
    fn test_nuttall_window_matches_formula() {
        let len = 8;
        let mut w = vec![0.0; len];
        nuttall_window(len, &mut w);
        for (i, &wv) in w.iter().enumerate() {
            let t = i as f64 / (len as f64 - 1.0);
            let expected = 0.355768 - 0.487396 * (2.0 * K_PI * t).cos()
                + 0.144232 * (4.0 * K_PI * t).cos()
                - 0.012604 * (6.0 * K_PI * t).cos();
            assert!(close(wv, expected), "i={}: {} vs {}", i, wv, expected);
        }
    }

    // Differential guard: the triple-angle form (one cos) must match the direct
    // 3-cos C++ formula across the full phase range [0, 2π], for odd and even
    // lengths. Tolerance 1e-12 bounds the legitimate ~1e-15 polynomial deviation
    // while catching any wrong identity or coefficient.
    #[test]
    fn test_nuttall_window_triple_angle_matches_direct_cos() {
        for len in [2usize, 3, 5, 8, 64, 257, 1024] {
            let mut w = vec![0.0; len];
            nuttall_window(len, &mut w);
            for (i, &wv) in w.iter().enumerate() {
                let phase = 2.0 * K_PI * i as f64 / (len as f64 - 1.0);
                let expected = NUTTALL_A0 - NUTTALL_A1 * phase.cos()
                    + NUTTALL_A2 * (2.0 * phase).cos()
                    - NUTTALL_A3 * (3.0 * phase).cos();
                assert!(
                    (wv - expected).abs() < 1e-12,
                    "len={len} i={i}: {wv} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn test_nuttall_window_endpoints_and_noop() {
        // y_length < 2 is a no-op (C++ divides by y_length-1).
        let mut w = vec![0.5, 0.5];
        nuttall_window(1, &mut w);
        assert_eq!(w, [0.5, 0.5]);
        // len 2 -> both endpoints are t=0 and t=1 -> ~0.
        let mut w2 = vec![0.0; 2];
        nuttall_window(2, &mut w2);
        assert!(w2[0].abs() < 1e-6 && w2[1].abs() < 1e-6);
    }

    #[test]
    fn test_interp1q_linear() {
        use crate::matlab::interp1q;
        let y = [0.0, 1.0, 2.0, 3.0];
        let xi = [0.5, 1.5, 2.5];
        let mut yi = [0.0; 3];
        interp1q(0.0, 1.0, &y, &xi, &mut yi);
        assert!((yi[0] - 0.5).abs() < 1e-12);
        assert!((yi[1] - 1.5).abs() < 1e-12);
        assert!((yi[2] - 2.5).abs() < 1e-12);
    }

    // C++-derived: for a linear input the added low-frequency replica is also
    // linear, so the corrected bins collapse to a constant (14.8); the tail is
    // the unmodified input (common.cpp:56-75).
    #[test]
    fn test_dc_correction_matches_cpp_reference() {
        let input: Vec<f64> = (1..=16).map(|i| i as f64).collect();
        let mut output = vec![0.0; 16];
        dc_correction(&input, 100.0, 16000, 2048, &mut output);
        // upper_limit = 2 + int(100*2048/16000) = 14 -> corrected bins 0..12.
        for (i, &ov) in output.iter().enumerate().take(13) {
            assert!(close(ov, 14.8), "i={}: {}", i, ov);
        }
        assert!(close(output[13], 14.0));
        assert!(close(output[14], 15.0));
        assert!(close(output[15], 16.0));
    }

    // f0=0 edge: upper_limit=2 -> only bin 0 corrected to input[0]+input[0].
    #[test]
    fn test_dc_correction_zero_f0() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = [0.0; 4];
        dc_correction(&input, 0.0, 16000, 256, &mut output);
        assert!(close(output[0], 2.0));
        assert!(close(output[1], 2.0));
        assert!(close(output[2], 3.0));
        assert!(close(output[3], 4.0));
    }

    // Constant input is a fixed point of the smoothing: verifies the
    // cumsum/integral scaling and that NO clamping occurs at the edges
    // (A1-D11). A clamped rewrite (like the Python port) breaks the edges.
    #[test]
    fn test_linear_smoothing_constant_input_is_identity() {
        let n = 256 / 2 + 1;
        let input = vec![1.0; n];
        let mut output = vec![0.0; n];
        linear_smoothing(&input, 100.0, 16000, 256, &mut output);
        for (i, &ov) in output.iter().enumerate() {
            assert!(close(ov, 1.0), "i={}: {}", i, ov);
        }
    }

    // C++-derived expected sequence (common.cpp LinearSmoothing). Interior is
    // identity; the edges show the even-mirroring effect ([0]=2.0, [8]=8.0).
    #[test]
    fn test_linear_smoothing_matches_cpp_reference() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut output = vec![0.0; 9];
        linear_smoothing(&input, 4000.0, 16000, 16, &mut output);
        let expected = [2.0, 2.25, 3.0, 4.0, 5.0, 6.0, 7.0, 7.75, 8.0];
        for i in 0..9 {
            assert!(
                close(output[i], expected[i]),
                "i={}: {} vs {}",
                i,
                output[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_linear_smoothing_output_length() {
        let input = vec![1.0; 129];
        let mut output = vec![0.0; 129];
        linear_smoothing(&input, 100.0, 16000, 256, &mut output);
        assert_eq!(output.len(), 129);
        for &v in &output {
            assert!(v.is_finite());
        }
    }

    // Edge case: f0 near Nyquist pushes upper_limit close to fft_size/2, so the
    // corrected region spans most of the spectrum. Verifies no panic and that
    // the uncorrected tail is the unmodified input (common.cpp:56-75).
    #[test]
    fn test_dc_correction_f0_near_nyquist() {
        let fs = 16000;
        let fft_size = 1024usize;
        let n = fft_size / 2 + 1;
        let input: Vec<f64> = (0..n).map(|i| 1.0 + 0.01 * i as f64).collect();
        let mut output = vec![0.0; n];
        let f0 = 7900.0; // just below Nyquist (fs/2 = 8000)
        dc_correction(&input, f0, fs, fft_size, &mut output);

        let upper_limit = 2 + (f0 * fft_size as f64 / fs as f64) as usize;
        let upper_limit_replica = upper_limit - 1;
        for &v in &output {
            assert!(v.is_finite());
        }
        // Bins beyond the corrected region are the unmodified input.
        for i in upper_limit_replica..n {
            assert_eq!(output[i], input[i], "bin {i}");
        }
    }

    // Edge case: an all-zero spectrum is a fixed point of DC correction (the
    // interp1q replica of zeros is zero), so the output stays exactly zero.
    #[test]
    fn test_dc_correction_zero_spectrum() {
        let input = vec![0.0; 64];
        let mut output = vec![0.0; 64];
        dc_correction(&input, 100.0, 16000, 128, &mut output);
        for &v in &output {
            assert_eq!(v, 0.0);
        }
    }

    // Edge case: an all-zero spectrum is a fixed point of linear smoothing
    // (mirrored spectrum and cumulative sum are zero), so the output is zero.
    #[test]
    fn test_linear_smoothing_zero_spectrum() {
        let n = 128 / 2 + 1;
        let input = vec![0.0; n];
        let mut output = vec![0.0; n];
        linear_smoothing(&input, 100.0, 16000, 128, &mut output);
        for &v in &output {
            assert_eq!(v, 0.0);
        }
    }
}
