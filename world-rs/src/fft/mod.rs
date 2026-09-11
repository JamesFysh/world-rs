//! FFT/IFFT primitives for WORLD, matching the C++ reference conventions.
//!
//! Conventions (C++ authoritative, see WORLD-source-analysis/02-fft.md §4):
//! - r2c forward: standard unnormalized forward DFT, half-spectrum (`n/2+1` bins).
//! - c2r backward: **unnormalized** inverse, returns `N·x`. Call sites divide by
//!   `n` to normalize (the `÷ fft_size` is NOT baked in here).
//! - c2c forward: standard forward DFT of `conj(Z)` (conjugated relative to a
//!   standard library forward).
//! - c2c backward: `conj` of the standard forward DFT of `Z` (conjugated relative
//!   to a standard library backward).
//!
//! `rustfft`/`realfft` do not normalize, so `plan_fft_forward` is the unnormalized
//! forward DFT and `plan_fft_inverse` is the unnormalized inverse DFT (`N·x`).

use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::{num_complex::Complex, FftPlanner};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

/// C++: `typedef double fft_complex[2]` (world/fft.h:22). Index 0 is the real
/// part, index 1 the imaginary part.
pub type FftComplex = [f64; 2];

/// Cached r2c plan keyed by transform size.
type RealToComplexCache = OnceLock<Mutex<HashMap<usize, Arc<dyn RealToComplex<f64>>>>>;
/// Cached c2r plan keyed by transform size.
type ComplexToRealCache = OnceLock<Mutex<HashMap<usize, Arc<dyn ComplexToReal<f64>>>>>;
/// Cached complex FFT plan keyed by transform size.
// C++ parity (c2c FFT); exercised by unit tests only — minimum-phase is deferred.
#[allow(dead_code)]
type ComplexFftCache = OnceLock<Mutex<HashMap<usize, Arc<dyn rustfft::Fft<f64>>>>>;

static REAL_TO_COMPLEX_CACHE: RealToComplexCache = OnceLock::new();
static COMPLEX_TO_REAL_CACHE: ComplexToRealCache = OnceLock::new();
// C++ parity (c2c FFT); exercised by unit tests only — minimum-phase is deferred.
#[allow(dead_code)]
static COMPLEX_FFT_CACHE: ComplexFftCache = OnceLock::new();

/// Lock a plan-cache map, recovering the guard if a prior holder panicked
/// (poisoned lock). The map is rebuilt on every access, so a poisoned lock
/// never loses correctness; recovering keeps the total no-panic guarantee.
fn lock_map<'a, V>(m: &'a Mutex<HashMap<usize, V>>) -> MutexGuard<'a, HashMap<usize, V>> {
    match m.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn get_real_to_complex_plan(n: usize) -> Arc<dyn RealToComplex<f64>> {
    let cache = REAL_TO_COMPLEX_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = lock_map(cache);
    if let Some(p) = map.get(&n) {
        return p.clone();
    }
    let mut planner = RealFftPlanner::<f64>::new();
    let plan = planner.plan_fft_forward(n);
    map.insert(n, plan.clone());
    plan
}

fn get_complex_to_real_plan(n: usize) -> Arc<dyn ComplexToReal<f64>> {
    let cache = COMPLEX_TO_REAL_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = lock_map(cache);
    if let Some(p) = map.get(&n) {
        return p.clone();
    }
    let mut planner = RealFftPlanner::<f64>::new();
    let plan = planner.plan_fft_inverse(n);
    map.insert(n, plan.clone());
    plan
}

// C++ parity (c2c FFT); exercised by unit tests only — minimum-phase is deferred.
#[allow(dead_code)]
fn get_complex_fft_plan(n: usize) -> Arc<dyn rustfft::Fft<f64>> {
    let cache = COMPLEX_FFT_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = lock_map(cache);
    if let Some(p) = map.get(&n) {
        return p.clone();
    }
    let mut planner = FftPlanner::new();
    let plan = planner.plan_fft_forward(n);
    map.insert(n, plan.clone());
    plan
}

fn next_pow2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

/// Forward real FFT (r2c). Standard unnormalized forward DFT producing the
/// half-spectrum (`n/2+1` bins). The input is zero-padded to the next power of
/// two. Matches C++ `ForwardRealFFT`.
pub(crate) fn forward_real_fft(input: &[f64]) -> Vec<Complex<f64>> {
    let n_in = input.len();
    if n_in == 0 {
        return vec![];
    }
    let n = next_pow2(n_in);
    let mut data = vec![0.0; n];
    data[..n_in].copy_from_slice(input);
    let r2c = get_real_to_complex_plan(n);
    let mut buffer = vec![Complex::new(0.0, 0.0); n / 2 + 1];
    let mut scratch = vec![Complex::new(0.0, 0.0); r2c.get_scratch_len()];
    let _ = r2c.process_with_scratch(&mut data, &mut buffer, &mut scratch);
    buffer
}

/// Inverse real FFT (c2r). Returns the **unnormalized** inverse (`N·x`),
/// matching C++ `InverseRealFFT`. Call sites divide by `n` to normalize.
pub(crate) fn inverse_real_fft(spectrum: &[Complex<f64>], n: usize) -> Vec<f64> {
    if n == 0 || spectrum.is_empty() {
        return vec![];
    }
    let c2r = get_complex_to_real_plan(n);
    let mut buffer = vec![Complex::new(0.0, 0.0); n / 2 + 1];
    let len = spectrum.len().min(buffer.len());
    buffer[..len].copy_from_slice(&spectrum[..len]);
    let mut output = vec![0.0; n];
    let mut scratch = vec![Complex::new(0.0, 0.0); c2r.get_scratch_len()];
    let _ = c2r.process_with_scratch(&mut buffer, &mut output, &mut scratch);
    output
}

/// Forward complex FFT (c2c). Returns the C++ `FFT_FORWARD` convention: the
/// standard forward DFT of `conj(Z)`. Implemented as the unnormalized forward
/// DFT of `conj(Z)`. Matches C++ `MinimumPhaseAnalysis::forward_fft`.
// C++ parity (c2c FFT); exercised by unit tests only — minimum-phase is deferred.
#[allow(dead_code)]
pub(crate) fn forward_complex_fft(input: &[Complex<f64>], n: usize) -> Vec<Complex<f64>> {
    if n == 0 || input.is_empty() {
        return vec![];
    }
    let fft = get_complex_fft_plan(n);
    let mut buffer = vec![Complex::new(0.0, 0.0); n];
    for i in 0..input.len().min(n) {
        buffer[i] = input[i].conj();
    }
    let mut scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];
    fft.process_with_scratch(&mut buffer, &mut scratch);
    buffer
}

/// Inverse complex FFT (c2c). Returns the C++ `FFT_BACKWARD` convention: the
/// complex conjugate of the standard forward DFT of `Z`. Implemented as the
/// unnormalized forward DFT of `Z`, then conjugated. Matches C++
/// `InverseComplexFFT`.
// C++ parity (c2c FFT); exercised by unit tests only — minimum-phase is deferred.
#[allow(dead_code)]
pub(crate) fn inverse_complex_fft(input: &[Complex<f64>], n: usize) -> Vec<Complex<f64>> {
    if n == 0 || input.is_empty() {
        return vec![];
    }
    let fft = get_complex_fft_plan(n);
    let mut buffer = vec![Complex::new(0.0, 0.0); n];
    let len = input.len().min(n);
    buffer[..len].copy_from_slice(&input[..len]);
    let mut scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];
    fft.process_with_scratch(&mut buffer, &mut scratch);
    for c in buffer.iter_mut() {
        *c = c.conj();
    }
    buffer
}

/// Pre-allocated forward real FFT (r2c) with a cached plan.
/// Matches C++ `ForwardRealFFT` (world/common.h:18-23).
pub(crate) struct ForwardRealFFT {
    pub(crate) fft_size: usize,
    pub(crate) waveform: Vec<f64>,
    pub(crate) spectrum: Vec<Complex<f64>>,
    pub(crate) temp: Vec<f64>,
    scratch: Vec<Complex<f64>>,
    plan: Arc<dyn RealToComplex<f64>>,
}

impl ForwardRealFFT {
    pub(crate) fn new(fft_size: usize) -> Self {
        let mut planner = RealFftPlanner::<f64>::new();
        let plan = planner.plan_fft_forward(fft_size);
        let scratch_len = plan.get_scratch_len();
        let half = fft_size / 2 + 1;
        Self {
            fft_size,
            waveform: vec![0.0; fft_size],
            spectrum: vec![Complex::new(0.0, 0.0); half],
            temp: vec![0.0; half],
            scratch: vec![Complex::new(0.0, 0.0); scratch_len],
            plan,
        }
    }

    /// Runs the forward transform, filling `spectrum` from `waveform`.
    ///
    /// WARNING: `process_with_scratch` overwrites the input buffer, so
    /// `waveform` holds scrambled data after this returns — never read the
    /// time signal back out of it. Measured: sizes ≤ 1024 happen to be
    /// preserved, sizes ≥ 2048 are fully overwritten; do NOT rely on the
    /// threshold (it is a realfft implementation detail). If downstream code
    /// needs the waveform (e.g. Harvest's decimated `y` for refinement),
    /// snapshot or recompute it *before* calling this. (Bit us twice:
    /// HARVEST-T8's `y` snapshot and D4C-T2's centroid ramp — both silently
    /// fed garbage into the pipeline.)
    pub(crate) fn forward(&mut self) {
        let _ = self.plan.process_with_scratch(
            &mut self.waveform,
            &mut self.spectrum,
            &mut self.scratch,
        );
    }
}

/// Pre-allocated inverse real FFT (c2r) with a cached plan.
/// Matches C++ `InverseRealFFT` (world/common.h:26-31).
pub(crate) struct InverseRealFFT {
    // Write-only, kept for C++ API parity (matches `InverseRealFFT::fft_size`).
    #[allow(dead_code)]
    pub(crate) fft_size: usize,
    pub(crate) waveform: Vec<f64>,
    pub(crate) spectrum: Vec<Complex<f64>>,
    scratch: Vec<Complex<f64>>,
    plan: Arc<dyn ComplexToReal<f64>>,
}

impl InverseRealFFT {
    pub(crate) fn new(fft_size: usize) -> Self {
        let mut planner = RealFftPlanner::<f64>::new();
        let plan = planner.plan_fft_inverse(fft_size);
        let scratch_len = plan.get_scratch_len();
        let half = fft_size / 2 + 1;
        Self {
            fft_size,
            waveform: vec![0.0; fft_size],
            spectrum: vec![Complex::new(0.0, 0.0); half],
            scratch: vec![Complex::new(0.0, 0.0); scratch_len],
            plan,
        }
    }

    /// Runs the inverse transform, filling `waveform` from `spectrum`.
    ///
    /// WARNING: same aliasing hazard as [`ForwardRealFFT::forward`], but
    /// worse — `spectrum` (the input) is overwritten at *all* sizes
    /// (measured 64–32768). Never read `spectrum` back after this call.
    pub(crate) fn inverse(&mut self) {
        let _ = self.plan.process_with_scratch(
            &mut self.spectrum,
            &mut self.waveform,
            &mut self.scratch,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // Tight tolerance for internal consistency (both sides are f64).
    const TOL: f64 = 1e-9;
    // Stated requirement: match the C++ reference within 1e-6.
    const TOL_REF: f64 = 1e-6;

    fn close_c(a: Complex<f64>, b: Complex<f64>, tol: f64) -> bool {
        (a.re - b.re).abs() < tol && (a.im - b.im).abs() < tol
    }

    // Direct O(n^2) reference DFTs used to validate the FFT wrappers.
    fn dft_forward(x: &[Complex<f64>]) -> Vec<Complex<f64>> {
        let n = x.len();
        (0..n)
            .map(|k| {
                (0..n).fold(Complex::new(0.0, 0.0), |acc, j| {
                    let ang = -2.0 * PI * (j as f64) * (k as f64) / (n as f64);
                    acc + x[j] * Complex::new(ang.cos(), ang.sin())
                })
            })
            .collect()
    }

    fn dft_inverse_unnormalized(x: &[Complex<f64>]) -> Vec<Complex<f64>> {
        let n = x.len();
        (0..n)
            .map(|j| {
                (0..n).fold(Complex::new(0.0, 0.0), |acc, k| {
                    let ang = 2.0 * PI * (j as f64) * (k as f64) / (n as f64);
                    acc + x[k] * Complex::new(ang.cos(), ang.sin())
                })
            })
            .collect()
    }

    fn dft_real_forward(x: &[f64]) -> Vec<Complex<f64>> {
        let n = x.len();
        let half = n / 2 + 1;
        (0..half)
            .map(|k| {
                (0..n).fold(Complex::new(0.0, 0.0), |acc, j| {
                    let ang = -2.0 * PI * (j as f64) * (k as f64) / (n as f64);
                    acc + Complex::new(x[j], 0.0) * Complex::new(ang.cos(), ang.sin())
                })
            })
            .collect()
    }

    // (a) Known-value r2c: impulse at index 1, n=8 -> bin k = e^{-i*pi*k/4}.
    #[test]
    fn r2c_known_value_impulse() {
        let x = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let spec = forward_real_fft(&x);
        assert_eq!(spec.len(), 5); // n/2+1
        for (k, &bin) in spec.iter().enumerate() {
            let ang = -PI * (k as f64) / 4.0;
            let expected = Complex::new(ang.cos(), ang.sin());
            assert!(
                close_c(bin, expected, TOL),
                "bin {}: {} vs {}",
                k,
                bin,
                expected
            );
        }
    }

    // (a) Known-value r2c: constant [1,1,1,1] -> [4, 0, 0].
    #[test]
    fn r2c_known_value_dc() {
        let x = [1.0, 1.0, 1.0, 1.0];
        let spec = forward_real_fft(&x);
        assert_eq!(spec.len(), 3);
        assert!(close_c(spec[0], Complex::new(4.0, 0.0), TOL));
        assert!(spec[1].re.abs() < TOL && spec[1].im.abs() < TOL);
        assert!(spec[2].re.abs() < TOL && spec[2].im.abs() < TOL);
    }

    // (a) r2c matches a direct DFT on a non-trivial signal.
    #[test]
    fn r2c_matches_direct_dft() {
        let x: Vec<f64> = (0..16).map(|i| (i as f64 * 0.3).sin() + 0.5).collect();
        let spec = forward_real_fft(&x);
        let expected = dft_real_forward(&x);
        assert_eq!(spec.len(), expected.len());
        for k in 0..spec.len() {
            assert!(
                close_c(spec[k], expected[k], TOL),
                "bin {}: {} vs {}",
                k,
                spec[k],
                expected[k]
            );
        }
    }

    // (c) Unnormalized inverse: ifft(fft(x)) == N*x (NOT x).
    #[test]
    fn c2r_unnormalized_inverse_scales_by_n() {
        let input: Vec<f64> = (0..16).map(|i| (i as f64 * 0.7).cos()).collect();
        let n = 16;
        let spec = forward_real_fft(&input);
        let out = inverse_real_fft(&spec, n);
        assert_eq!(out.len(), n);
        for i in 0..n {
            let expected = if i < input.len() {
                n as f64 * input[i]
            } else {
                0.0
            };
            assert!(
                (out[i] - expected).abs() < TOL,
                "i={}: {} vs {}",
                i,
                out[i],
                expected
            );
        }
    }

    // (c) Normalized round-trip: ifft(fft(x))/N == x within 1e-6.
    #[test]
    fn fft_ifft_roundtrip_normalized() {
        let input: Vec<f64> = (0..256).map(|i| (i as f64).sin()).collect();
        let spec = forward_real_fft(&input);
        let n = 2 * (spec.len() - 1);
        let out = inverse_real_fft(&spec, n);
        for i in 0..input.len() {
            let normalized = out[i] / n as f64;
            assert!(
                (normalized - input[i]).abs() < TOL_REF,
                "i={}: {} vs {}",
                i,
                normalized,
                input[i]
            );
        }
    }

    // (b) c2c backward == conj(forward DFT of Z).
    #[test]
    fn c2c_backward_conjugation() {
        let n = 16;
        let z: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new((i as f64 * 0.5).sin(), (i as f64 * 0.3).cos()))
            .collect();
        let out = inverse_complex_fft(&z, n);
        let expected = dft_forward(&z).iter().map(|c| c.conj()).collect::<Vec<_>>();
        for k in 0..n {
            assert!(
                close_c(out[k], expected[k], TOL),
                "bin {}: {} vs {}",
                k,
                out[k],
                expected[k]
            );
        }
    }

    // (b) c2c forward == conj(inverse_unnormalized DFT of Z) == forward DFT of conj(Z).
    #[test]
    fn c2c_forward_conjugation() {
        let n = 16;
        let z: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new((i as f64 * 0.4).cos(), (i as f64 * 0.6).sin()))
            .collect();
        let out = forward_complex_fft(&z, n);
        let expected = dft_inverse_unnormalized(&z)
            .iter()
            .map(|c| c.conj())
            .collect::<Vec<_>>();
        for k in 0..n {
            assert!(
                close_c(out[k], expected[k], TOL),
                "bin {}: {} vs {}",
                k,
                out[k],
                expected[k]
            );
        }
    }

    // (d) Zero-padding: input length 3 -> n=4, FFT of [1,2,3,0].
    #[test]
    fn r2c_zero_padding() {
        let x = [1.0, 2.0, 3.0];
        let spec = forward_real_fft(&x);
        assert_eq!(spec.len(), 3); // n=4 -> n/2+1 = 3
        let expected = dft_real_forward(&[1.0, 2.0, 3.0, 0.0]);
        for k in 0..3 {
            assert!(
                close_c(spec[k], expected[k], TOL),
                "bin {}: {} vs {}",
                k,
                spec[k],
                expected[k]
            );
        }
    }

    // (e) Non-power-of-2 round-trip (length 10 -> n=16).
    #[test]
    fn fft_roundtrip_non_pow2() {
        let input: Vec<f64> = (0..10).map(|i| i as f64 * 0.1).collect();
        let spec = forward_real_fft(&input);
        let n = 2 * (spec.len() - 1);
        assert_eq!(n, 16);
        let out = inverse_real_fft(&spec, n);
        for i in 0..input.len() {
            let normalized = out[i] / n as f64;
            assert!(
                (normalized - input[i]).abs() < TOL_REF,
                "i={}: {} vs {}",
                i,
                normalized,
                input[i]
            );
        }
        for &val in &out[input.len()..] {
            assert!(val.abs() < 1e-9, "padded region not zero");
        }
    }

    // (f) Empty input.
    #[test]
    fn fft_empty_input() {
        assert!(forward_real_fft(&[]).is_empty());
        assert!(inverse_real_fft(&[], 8).is_empty());
        assert!(inverse_complex_fft(&[], 8).is_empty());
        assert!(forward_complex_fft(&[], 8).is_empty());
    }

    // (f) Edge: single element.
    #[test]
    fn fft_single_element() {
        let spec = forward_real_fft(&[5.0]);
        assert_eq!(spec.len(), 1);
        assert!(close_c(spec[0], Complex::new(5.0, 0.0), TOL));
    }

    // Reference vectors from /tmp/opencode/fft_vectors.out (A2-OQ2, C++ driver).
    // r2c impulse -> bin0 = 1; c2r DC wave -> out0 = N; c2c fwd impulse -> bin0 = 1;
    // c2c bwd DC -> bin0 = N.
    #[test]
    fn matches_cpp_reference_vectors() {
        for size in [64usize, 2048usize] {
            let n = size as f64;

            // r2c impulse [1,0,...,0] -> bin0 = 1+0i
            let mut impulse = vec![0.0; size];
            impulse[0] = 1.0;
            let r2c = forward_real_fft(&impulse);
            assert!(
                close_c(r2c[0], Complex::new(1.0, 0.0), TOL_REF),
                "size {} r2c impulse[0]={}",
                size,
                r2c[0]
            );

            // c2r DC wave [N,0,...,0] -> out[0] = N (unnormalized inverse)
            let mut dc_spec = vec![Complex::new(0.0, 0.0); size / 2 + 1];
            dc_spec[0] = Complex::new(n, 0.0);
            let c2r = inverse_real_fft(&dc_spec, size);
            assert!(
                (c2r[0] - n).abs() < TOL_REF,
                "size {} c2r dc[0]={}",
                size,
                c2r[0]
            );

            // c2c forward impulse [1,0,...,0] -> bin0 = 1-0i
            let mut c_impulse = vec![Complex::new(0.0, 0.0); size];
            c_impulse[0] = Complex::new(1.0, 0.0);
            let c2c_fwd = forward_complex_fft(&c_impulse, size);
            assert!(
                close_c(c2c_fwd[0], Complex::new(1.0, 0.0), TOL_REF),
                "size {} c2c fwd impulse[0]={}",
                size,
                c2c_fwd[0]
            );

            // c2c backward DC [N,0,...,0] -> bin0 = N-0i
            let mut c_dc = vec![Complex::new(0.0, 0.0); size];
            c_dc[0] = Complex::new(n, 0.0);
            let c2c_bwd = inverse_complex_fft(&c_dc, size);
            assert!(
                close_c(c2c_bwd[0], Complex::new(n, 0.0), TOL_REF),
                "size {} c2c bwd dc[0]={}",
                size,
                c2c_bwd[0]
            );
        }
    }
}
