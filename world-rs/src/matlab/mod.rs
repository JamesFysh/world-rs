pub fn matlab_round(x: f64) -> i64 {
    if x >= 0.0 {
        (x + 0.5) as i64
    } else {
        (x - 0.5) as i64
    }
}

/// Circularly shift the elements of `x` by `n/2` into `y` (C++ `fftshift`).
///
/// # Panics
///
/// Panics if `y.len() != x.len()`.
pub fn fftshift(x: &[f64], y: &mut [f64]) {
    let n = x.len();
    assert_eq!(y.len(), n);
    let half = n / 2;
    y[..half].copy_from_slice(&x[half..half + half]);
    y[half..half + half].copy_from_slice(&x[..half]);
}

/// Histogram bin indices (C++ `histc`): for each edge, the first index in `x`
/// at or beyond that edge.
///
/// # Panics
///
/// Panics if `index.len() != edges.len()` or if `x` is empty.
pub fn histc(x: &[f64], edges: &[f64], index: &mut [i32]) {
    let edges_len = edges.len();
    assert_eq!(index.len(), edges_len);
    assert!(!x.is_empty());
    let mut count = 1usize;
    let mut i = 0usize;
    while i < edges_len {
        index[i] = 1;
        if edges[i] >= x[0] {
            break;
        }
        i += 1;
    }
    while i < edges_len {
        if count >= x.len() {
            break;
        }
        if edges[i] < x[count] {
            index[i] = count as i32;
            i += 1;
        } else {
            index[i] = count as i32;
            count += 1;
        }
        if count >= x.len() {
            break;
        }
    }
    count = count.saturating_sub(1);
    i += 1;
    while i < edges_len {
        index[i] = count as i32;
        i += 1;
    }
}

/// Linear interpolation (C++ `interp1`). `k` is derived internally via `histc`
/// from the query points `xi`, so `k` is always in [1, y.len()-1] and the OOB
/// guards below can never fire for valid in-scope call sites. The C++ original
/// performs the out-of-bounds read unconditionally (undefined behaviour); we
/// clamp to the nearest endpoint instead.
///
/// # Panics
///
/// Panics if `y.len() != x.len()` or if `yi.len() != xi.len()`.
pub(crate) fn interp1(x: &[f64], y: &[f64], xi: &[f64], yi: &mut [f64]) {
    let n = x.len();
    assert_eq!(y.len(), n);
    assert_eq!(yi.len(), xi.len());
    if n < 2 {
        return;
    }
    let mut h = vec![0.0f64; n - 1];
    for i in 0..n - 1 {
        h[i] = x[i + 1] - x[i];
    }
    let mut k = vec![0i32; xi.len()];
    histc(x, xi, &mut k);
    for i in 0..xi.len() {
        let ki = k[i] as usize;
        if ki == 0 || ki >= n {
            yi[i] = y[n - 1];
            continue;
        }
        let idx = ki - 1;
        if idx >= n - 1 {
            yi[i] = y[n - 1];
            continue;
        }
        let s = (xi[i] - x[idx]) / h[idx];
        yi[i] = y[idx] + s * (y[idx + 1] - y[idx]);
    }
}

pub(crate) fn interp1_into(
    x: &[f64],
    y: &[f64],
    xi: &[f64],
    yi: &mut [f64],
    h: &mut [f64],
    k: &mut [i32],
) {
    let n = x.len();
    assert_eq!(y.len(), n);
    assert_eq!(yi.len(), xi.len());
    if n < 2 {
        return;
    }
    debug_assert!(h.len() >= n - 1);
    debug_assert!(k.len() >= xi.len());
    for i in 0..n - 1 {
        h[i] = x[i + 1] - x[i];
    }
    histc(x, xi, &mut k[..xi.len()]);
    for i in 0..xi.len() {
        let ki = k[i] as usize;
        if ki == 0 || ki >= n {
            yi[i] = y[n - 1];
            continue;
        }
        let idx = ki - 1;
        if idx >= n - 1 {
            yi[i] = y[n - 1];
            continue;
        }
        let s = (xi[i] - x[idx]) / h[idx];
        yi[i] = y[idx] + s * (y[idx + 1] - y[idx]);
    }
}

// Coefficients are the exact C++ WORLD decimation filter taps
// (matlabfunctions.cpp); full f64 precision is intentional.
#[allow(clippy::excessive_precision)]
fn filter_for_decimate(x: &[f64], r: usize, y: &mut [f64]) {
    let (a0, a1, a2, b0, b1) = match r {
        11 => (
            2.450743295230728,
            -2.06794904601978,
            0.59574774438332101,
            0.0026822508007163792,
            0.0080467524021491377,
        ),
        12 => (
            2.4981398605924205,
            -2.1368928194784025,
            0.62187513816221485,
            0.0021097275904709001,
            0.0063291827714127002,
        ),
        10 => (
            2.3936475118069387,
            -1.9873904075111861,
            0.5658879979027055,
            0.0034818622251927556,
            0.010445586675578267,
        ),
        9 => (
            2.3236003491759578,
            -1.8921545617463598,
            0.53148928133729068,
            0.0046331164041389372,
            0.013899349212416812,
        ),
        8 => (
            2.2357462340187593,
            -1.7780899984041358,
            0.49152555365968692,
            0.0063522763407111993,
            0.019056829022133598,
        ),
        7 => (
            2.1225239019534703,
            -1.6395144861046302,
            0.44469707800587366,
            0.0090366882681608418,
            0.027110064804482525,
        ),
        6 => (
            1.9715352749512141,
            -1.4686795689225347,
            0.3893908434965701,
            0.013469181309343825,
            0.040407543928031475,
        ),
        5 => (
            1.7610939654280557,
            -1.2554914843859768,
            0.3237186507788215,
            0.021334858522387423,
            0.06400457556716227,
        ),
        4 => (
            1.4499664446880227,
            -0.98943497080950582,
            0.24578252340690215,
            0.036710750339322612,
            0.11013225101796784,
        ),
        3 => (
            0.95039378983237421,
            -0.67429146741526791,
            0.15412211621346475,
            0.071221945171178636,
            0.21366583551353591,
        ),
        2 => (
            0.041156734567757189,
            -0.42599112459189636,
            0.041037215479961225,
            0.16797464681802227,
            0.50392394045406674,
        ),
        _ => (0.0, 0.0, 0.0, 0.0, 0.0),
    };
    let mut w = [0.0f64, 0.0, 0.0];
    for i in 0..x.len() {
        let wt = x[i] + a0 * w[0] + a1 * w[1] + a2 * w[2];
        y[i] = b0 * wt + b1 * w[0] + b1 * w[1] + b0 * w[2];
        w[2] = w[1];
        w[1] = w[0];
        w[0] = wt;
    }
}

// Down-samples `x` by a factor `r` (2..=12) using the C++ WORLD three-pass
// zero-phase IIR/FIR filter. Mirrors `decimate()` in matlabfunctions.cpp
// line-for-line, including the final sample-extraction loop.
//
// Output count: the C++ loop `for (i = nbeg; i < x_length + kNFact; i += r)`
// writes `ceil((kNFact - r + r*nout) / r)` samples where `nout = (x_length-1)/r + 1`.
// For r < 9 this is strictly greater than `nout` (e.g. r=2 -> nout+4). Callers
// must size `y` to at least that count; DIO allocates `fft_size` and consumes
// only `y_length = 1 + x_length/r` samples, which is always within range.
//
// # Panics
//
// In debug builds, panics if `r` is outside `2..=12`, or if the extraction
// loop writes past the end of `y` (i.e. `y` was undersized for the C++ output
// count) or reads past the padded buffer.
pub(crate) fn decimate(x: &[f64], r: usize, y: &mut [f64]) {
    debug_assert!(matches!(r, 2..=12));
    if !(2..=12).contains(&r) {
        // C++ default case: all-zero filter coefficients -> silent zero output.
        y.fill(0.0);
        return;
    }
    let len = x.len();
    if len <= 9 {
        y.fill(0.0);
        return;
    }
    let k_n_fact = 9usize;
    let tmp_len = len + 2 * k_n_fact;
    let mut tmp1 = vec![0.0f64; tmp_len];
    let mut tmp2 = vec![0.0f64; tmp_len];
    for i in 0..k_n_fact {
        tmp1[i] = 2.0 * x[0] - x[k_n_fact - i];
    }
    tmp1[k_n_fact..k_n_fact + len].copy_from_slice(&x[..len]);
    for j in 0..k_n_fact {
        tmp1[k_n_fact + len + j] = 2.0 * x[len - 1] - x[len - 2 - j];
    }
    filter_for_decimate(&tmp1, r, &mut tmp2);
    for i in 0..tmp_len {
        tmp1[i] = tmp2[tmp_len - 1 - i];
    }
    filter_for_decimate(&tmp1, r, &mut tmp2);
    for i in 0..tmp_len {
        tmp1[i] = tmp2[tmp_len - 1 - i];
    }
    let nout = (len - 1) / r + 1;
    // Equivalent to C++ `r - r*nout + x_length` but computed to avoid usize
    // underflow: nbeg = len - r*(nout-1), and r*(nout-1) <= len-1 always.
    let nbeg = len - r * (nout - 1);
    let mut count = 0usize;
    let mut i = nbeg;
    while i < len + k_n_fact {
        debug_assert!(count < y.len());
        let src = i + k_n_fact - 1;
        debug_assert!(src < tmp_len);
        y[count] = tmp1[src];
        count += 1;
        i += r;
    }
}

// Piecewise-linear interpolation on a uniform grid (C++ `interp1q`).
//
// The OOB guard below is defensive: for in-scope call sites (DCCorrection,
// LinearSmoothing) the computed base index always falls within [0, y.len()-1].
// The C++ original reads `y[base]`/`y[base+1]` unconditionally, which is an
// out-of-bounds read (undefined behaviour) when base is outside range; we clamp
// to `y[0]` instead.
/// Allocation-free core of [`interp1q`]: the caller supplies a `delta` buffer
/// of at least `y.len()` elements (used for the `y[i+1] - y[i]` differences).
/// The public [`interp1q`] allocates a temporary buffer and delegates here; the
/// CheapTrick hot path passes a pre-allocated buffer so the per-frame loop
/// performs no heap allocations (PHASE2-8).
///
/// # Panics
///
/// Panics if `yi.len() != xi.len()` or if `delta.len() < y.len()`. An empty
/// `y` is handled (zero-filled) without panicking.
pub(crate) fn interp1q_into(
    x0: f64,
    shift: f64,
    y: &[f64],
    xi: &[f64],
    yi: &mut [f64],
    delta: &mut [f64],
) {
    let n = y.len();
    if n == 0 {
        for yi_slot in yi.iter_mut() {
            *yi_slot = 0.0;
        }
        return;
    }
    assert_eq!(yi.len(), xi.len());
    assert!(delta.len() >= n, "interp1q_into: delta buffer too small");
    let delta = &mut delta[..n];
    for i in 0..n - 1 {
        delta[i] = y[i + 1] - y[i];
    }
    delta[n - 1] = 0.0;
    for i in 0..xi.len() {
        let pos = (xi[i] - x0) / shift;
        let base = pos.trunc() as isize;
        let frac = pos - base as f64;
        if base < 0 || base >= n as isize {
            yi[i] = y[0];
            continue;
        }
        let idx = base as usize;
        yi[i] = y[idx] + delta[idx] * frac;
    }
}

#[allow(dead_code)] // test utility wrapper
pub fn interp1q(x0: f64, shift: f64, y: &[f64], xi: &[f64], yi: &mut [f64]) {
    let mut delta_y = vec![0.0f64; y.len()];
    interp1q_into(x0, shift, y, xi, yi, &mut delta_y);
}

#[derive(Clone, Copy, Default)]
pub struct RandnState {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub w: u32,
}

const RANDN_SEED_X: u32 = 123456789;
const RANDN_SEED_Y: u32 = 362436069;
const RANDN_SEED_Z: u32 = 521288629;
const RANDN_SEED_W: u32 = 88675123;

pub fn randn_reseed(state: &mut RandnState) {
    state.x = RANDN_SEED_X;
    state.y = RANDN_SEED_Y;
    state.z = RANDN_SEED_Z;
    state.w = RANDN_SEED_W;
}

pub fn randn(state: &mut RandnState) -> f64 {
    let t = state.x ^ state.x.wrapping_shl(11);
    state.x = state.y;
    state.y = state.z;
    state.z = state.w;
    state.w = state.w ^ state.w.wrapping_shr(19) ^ (t ^ t.wrapping_shr(8));
    let mut tmp = state.w >> 4;
    for _ in 0..11 {
        let t2 = state.x ^ state.x.wrapping_shl(11);
        state.x = state.y;
        state.y = state.z;
        state.z = state.w;
        state.w = state.w ^ state.w.wrapping_shr(19) ^ (t2 ^ t2.wrapping_shr(8));
        tmp = tmp.wrapping_add(state.w >> 4);
    }
    tmp as f64 / 268435456.0 - 6.0
}

#[cfg(test)]
// Expected values are full-precision f64 C++ reference vectors.
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    #[test]
    fn test_matlab_round() {
        assert_eq!(matlab_round(2.3), 2);
        assert_eq!(matlab_round(2.7), 3);
        assert_eq!(matlab_round(-2.3), -2);
        assert_eq!(matlab_round(-2.7), -3);
        assert_eq!(matlab_round(0.5), 1);
        assert_eq!(matlab_round(-0.5), -1);
    }

    #[test]
    fn test_fftshift() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let mut y = [0.0; 4];
        fftshift(&x, &mut y);
        assert_eq!(y, [3.0, 4.0, 1.0, 2.0]);
    }

    #[test]
    fn test_histc_basic() {
        let x = [0.0, 1.0, 2.0, 3.0];
        let edges = [0.5, 1.5, 2.5];
        let mut idx = [0; 3];
        histc(&x, &edges, &mut idx);
        assert_eq!(idx, [1, 2, 3]);
    }

    #[test]
    fn test_interp1_linear() {
        let x = [0.0, 1.0, 2.0];
        let y = [0.0, 10.0, 20.0];
        let xi = [0.5, 1.5];
        let mut yi = [0.0; 2];
        interp1(&x, &y, &xi, &mut yi);
        assert!((yi[0] - 5.0).abs() < 1e-9);
        assert!((yi[1] - 15.0).abs() < 1e-9);
    }

    #[test]
    fn test_interp1q() {
        let x0 = 0.0;
        let shift = 1.0;
        let y = [0.0, 10.0, 20.0, 30.0];
        let xi = [0.5, 1.5, 2.2];
        let mut yi = [0.0; 3];
        interp1q(x0, shift, &y, &xi, &mut yi);
        assert!((yi[0] - 5.0).abs() < 1e-9);
        assert!((yi[1] - 15.0).abs() < 1e-9);
        assert!((yi[2] - 22.0).abs() < 1e-9);
    }

    #[test]
    fn test_randn_repeatability() {
        let mut s1 = RandnState::default();
        randn_reseed(&mut s1);
        let mut s2 = RandnState::default();
        randn_reseed(&mut s2);
        let v1 = randn(&mut s1);
        let v2 = randn(&mut s2);
        assert!((v1 - v2).abs() < 1e-12);
    }

    // Bit-exactness: the first 12 samples after reseed must equal the C++
    // xorshift reference (matlabfunctions.cpp `randn`) to the last bit.
    #[test]
    fn test_randn_bit_exact_vs_cpp() {
        let expected = [
            -1.3276404961943626,
            -0.62285530939698219,
            -1.6091805659234524,
            1.1797650642693043,
            -0.25188251212239265,
            -0.68379516527056694,
            -0.10105277225375175,
            0.66826450079679489,
            -1.2365790456533432,
            0.10969294607639313,
            1.0698245503008366,
            -1.1044377610087395,
        ];
        let mut s = RandnState::default();
        randn_reseed(&mut s);
        for (i, &e) in expected.iter().enumerate() {
            let v = randn(&mut s);
            assert_eq!(v, e, "randn[{i}] mismatch: {v} != {e}");
        }
    }

    // Number of samples the C++ decimate loop writes for a given (len, r).
    fn dec_out_count(len: usize, r: usize) -> usize {
        let nout = (len - 1) / r + 1;
        let nbeg = len - r * (nout - 1);
        let mut cnt = 0usize;
        let mut i = nbeg;
        while i < len + 9 {
            cnt += 1;
            i += r;
        }
        cnt
    }

    fn run_decimate(x: &[f64], r: usize, expected: &[f64]) {
        assert_eq!(expected.len(), dec_out_count(x.len(), r));
        let mut y = vec![0.0f64; expected.len()];
        decimate(x, r, &mut y);
        for (i, (a, b)) in y.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-9,
                "decimate r={r} len={} idx {i}: {a} != {b}",
                x.len()
            );
        }
    }

    // Output-count defect: for r < 9 the C++ loop writes MORE than
    // (x_length-1)/r + 1 samples. Lock in the exact C++ count and values.
    #[test]
    fn test_decimate_r2_l20() {
        let x: Vec<f64> = (0..20)
            .map(|i| 0.1 * (i as f64) + 0.001 * (i as f64) * (i as f64))
            .collect();
        let expected = [
            0.10043951456722894,
            0.30928442818296925,
            0.52505898499974457,
            0.74907694306384165,
            0.98120572086180902,
            1.2207515175549211,
            1.4701406177715708,
            1.7221332899238961,
            1.9972118517140098,
            2.2391995596682208,
            2.5895874541434623,
            2.6525333427635625,
            3.41406196907149,
            2.2210359186420559,
        ];
        assert_eq!(dec_out_count(20, 2), 14);
        run_decimate(&x, 2, &expected);
    }

    #[test]
    fn test_decimate_r3_l20() {
        let x: Vec<f64> = (0..20)
            .map(|i| 0.1 * (i as f64) + 0.001 * (i as f64) * (i as f64))
            .collect();
        let expected = [
            0.1050530282570718,
            0.4145612330033886,
            0.75053352382882499,
            1.0977536038860085,
            1.476287448087239,
            1.8359940487085407,
            2.3171353605477472,
            2.5111305610847396,
            3.3636674213717659,
        ];
        assert_eq!(dec_out_count(20, 3), 9);
        run_decimate(&x, 3, &expected);
    }

    #[test]
    fn test_decimate_r2_l21() {
        let x: Vec<f64> = (0..21)
            .map(|i| 0.1 * (i as f64) + 0.001 * (i as f64) * (i as f64))
            .collect();
        let expected = [
            -0.0038805725646177507,
            0.20570345358346243,
            0.41539786301687437,
            0.63640564361368446,
            0.86394276857837293,
            1.1002606194699383,
            1.3437166380980174,
            1.5971971479370679,
            1.8529892545164666,
            2.1325986397045549,
            2.3771778329283957,
            2.7352328188888557,
            2.792798671042414,
            3.5818887250531981,
            2.3259317629617069,
        ];
        assert_eq!(dec_out_count(21, 2), 15);
        run_decimate(&x, 2, &expected);
    }

    #[test]
    fn test_decimate_r2_l100() {
        let x: Vec<f64> = (0..100)
            .map(|i| 0.1 * (i as f64) + 0.001 * (i as f64) * (i as f64))
            .collect();
        let expected = [
            0.10044264218510827,
            0.30928098806377091,
            0.52505806311350833,
            0.74909971272877118,
            0.98110541118504291,
            1.2210927730347407,
            1.469102574406097,
            1.7250965007452945,
            1.9890998982937895,
            2.2610981075652661,
            2.5410990149013526,
            2.8290985680374616,
            3.1250987833969881,
            3.4290986813895246,
            3.7410987290192206,
            4.0610987070495588,
            4.3890987170757381,
            4.7250987125438426,
            5.0690987145736184,
            5.4210987136742625,
            5.7810987140637806,
            6.1490987139104529,
            6.525098713936579,
            6.9090987140197004,
            7.3010987137659011,
            7.7010987143745977,
            8.1090987129764507,
            8.5250987161440328,
            8.9490987090306913,
            9.381098724872075,
            9.8210986899045949,
            10.269098766341134,
            10.725098601080191,
            11.189098953896448,
            11.66109821183756,
            12.141099744420888,
            12.629096651157983,
            13.125102706838257,
            13.629091351562103,
            14.141111270502686,
            14.661080271203648,
            15.189116425675689,
            15.72511604913289,
            16.268923281718958,
            16.821816515202602,
            17.378723966860875,
            17.956215995904294,
            18.504969824065981,
            19.163865683604204,
            19.555853402836146,
            20.668846232487283,
            19.920037707811105,
            23.84352374203878,
            14.902796526888633,
        ];
        assert_eq!(dec_out_count(100, 2), 54);
        run_decimate(&x, 2, &expected);
    }

    #[test]
    fn test_decimate_r11_l100() {
        let x: Vec<f64> = (0..100)
            .map(|i| 0.1 * (i as f64) + 0.001 * (i as f64) * (i as f64))
            .collect();
        let expected = [
            -0.0727820625243778,
            1.2469640117627494,
            2.6830899073903058,
            4.3947083530126925,
            6.3373528557214556,
            8.535995701742312,
            10.943004295638726,
            13.644600198798431,
            16.583139346360745,
            16.489221826782842,
        ];
        assert_eq!(dec_out_count(100, 11), 10);
        run_decimate(&x, 11, &expected);
    }

    // Large-length count checks (values verified on the first 12 samples).
    #[test]
    fn test_decimate_large_counts() {
        let x: Vec<f64> = (0..44100)
            .map(|i| 0.1 * (i as f64) + 0.001 * (i as f64) * (i as f64))
            .collect();
        assert_eq!(dec_out_count(44100, 2), 22054);
        assert_eq!(dec_out_count(44100, 11), 4010);

        let r2_head = [
            0.10044264218510827,
            0.30928098806377091,
            0.52505806311350833,
            0.74909971272877118,
            0.98110541118504291,
            1.2210927730347407,
            1.469102574406097,
            1.725096500745295,
            1.9890998982937897,
            2.2610981075652656,
            2.541099014901353,
            2.8290985680374598,
        ];
        let mut y = vec![0.0f64; 22054];
        decimate(&x, 2, &mut y);
        for (i, (a, b)) in y.iter().take(12).zip(r2_head.iter()).enumerate() {
            assert!((a - b).abs() < 1e-9, "r2 head {i}: {a} != {b}");
        }

        let r11_head = [
            -0.0727662321059049,
            1.24690472926236,
            2.6833041828243,
            4.39396825124285,
            6.33975279160223,
            8.52899687772974,
            10.959951529181,
            13.6329588012929,
            16.5479580059655,
            19.7049579536705,
            23.1039580268696,
            26.7449579945482,
        ];
        let mut y = vec![0.0f64; 4010];
        decimate(&x, 11, &mut y);
        for (i, (a, b)) in y.iter().take(12).zip(r11_head.iter()).enumerate() {
            assert!((a - b).abs() < 1e-9, "r11 head {i}: {a} != {b}");
        }
    }

    // matlab_round must round half-integers AWAY from zero (C++ `round`), not
    // toward zero (which `trunc`-based rounding would do).
    #[test]
    fn test_matlab_round_half_integers() {
        assert_eq!(matlab_round(2.5), 3);
        assert_eq!(matlab_round(-2.5), -3);
        assert_eq!(matlab_round(0.5), 1);
        assert_eq!(matlab_round(-0.5), -1);
        assert_eq!(matlab_round(3.5), 4);
        assert_eq!(matlab_round(-3.5), -4);
        assert_eq!(matlab_round(0.0), 0);
        assert_eq!(matlab_round(-0.0), 0);
        assert_eq!(matlab_round(1.499999), 1);
        assert_eq!(matlab_round(1.500001), 2);
    }

    #[test]
    fn test_interp1_exact_and_extrapolation() {
        let x = [0.0, 1.0, 2.0, 3.0];
        let y = [0.0, 10.0, 20.0, 30.0];
        let xi = [0.0, 1.0, 2.0, 3.0, 0.5, 1.5, -0.5, 3.5];
        let mut yi = [0.0; 8];
        interp1(&x, &y, &xi, &mut yi);
        // exact nodes
        assert!((yi[0] - 0.0).abs() < 1e-9);
        assert!((yi[1] - 10.0).abs() < 1e-9);
        assert!((yi[2] - 20.0).abs() < 1e-9);
        assert!((yi[3] - 30.0).abs() < 1e-9);
        // between nodes
        assert!((yi[4] - 5.0).abs() < 1e-9);
        assert!((yi[5] - 15.0).abs() < 1e-9);
        // extrapolation (C++ linearly extrapolates; guards do not fire)
        assert!((yi[6] - (-5.0)).abs() < 1e-9);
        assert!((yi[7] - 35.0).abs() < 1e-9);
    }

    // interp1q with a negative shift (as used by DCCorrection): (xi - x0)/shift
    // is positive for negative xi, so base stays in range and interpolates.
    #[test]
    fn test_interp1q_negative_shift() {
        let x0 = 0.0;
        let shift = -1.0;
        let y = [0.0, 10.0, 20.0, 30.0];
        let xi = [-0.5, -1.5, -2.2];
        let mut yi = [0.0; 3];
        interp1q(x0, shift, &y, &xi, &mut yi);
        // pos = xi/(-1) = -xi: 0.5, 1.5, 2.2 -> base 0,1,2; frac 0.5,0.5,0.2
        assert!((yi[0] - 5.0).abs() < 1e-9);
        assert!((yi[1] - 15.0).abs() < 1e-9);
        assert!((yi[2] - 22.0).abs() < 1e-9);
    }

    // OOB guard: base < 0 -> C++ reads out of bounds (UB); Rust clamps to y[0].
    #[test]
    fn test_interp1q_oob_clamps_to_y0() {
        let x0 = 0.0;
        let shift = 1.0;
        let y = [7.0, 10.0, 20.0];
        let xi = [-1.5]; // pos = -1.5, base = -1 < 0 -> OOB
        let mut yi = [0.0; 1];
        interp1q(x0, shift, &y, &xi, &mut yi);
        assert_eq!(yi[0], 7.0);
    }
}
