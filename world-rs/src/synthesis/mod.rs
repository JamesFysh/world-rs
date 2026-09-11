//! Synthesis (port of `ext_src/world-cpp/src/synthesis.cpp`).
//!
//! Reconstructs a waveform from an F0 contour, a CheapTrick spectral envelope
//! (spectrogram), and an aperiodicity spectrogram.
//!
//! # Usage
//!
//! The typical analysis-synthesis flow mirrors the C++ `test/test.cpp`: DIO
//! extracts the F0 contour and 5 ms temporal grid, StoneMask refines the
//! contour, CheapTrick estimates the spectral envelope, and [`synthesis`]
//! reconstructs the waveform (with the constant-AP convenience default from
//! [`constant_aperiodicity`] while D4C is deferred):
//!
//! ```no_run
//! use world_rs::cheaptrick::{cheaptrick, initialize_cheaptrick_option};
//! use world_rs::dio::{dio, initialize_dio_option};
//! use world_rs::stonemask::stone_mask;
//! use world_rs::synthesis::{constant_aperiodicity, get_y_length, synthesis};
//!
//! let fs = 16000.0;
//! let x: Vec<f64> = vec![0.0; 16000]; // 1 second of 16 kHz audio, f64
//!
//! // 1. DIO: F0 contour + temporal grid.
//! let dio_option = initialize_dio_option();
//! let dio_result = dio(&x, fs, &dio_option).unwrap();
//!
//! // 2. StoneMask: refine the DIO contour.
//! let f0 = stone_mask(
//!     &x, x.len(), fs, &dio_result.temporal_positions, &dio_result.f0,
//!     dio_result.f0_length,
//! ).unwrap();
//!
//! // 3. CheapTrick: spectral envelope per frame.
//! let ct_option = initialize_cheaptrick_option(fs);
//! let sp = cheaptrick(&x, fs, &dio_result.temporal_positions, &f0, &ct_option)
//!     .unwrap();
//!
//! // 4. Synthesis: reconstruct the waveform (constant AP = 0.5).
//! let fft_size = ct_option.fft_size as usize;
//! let ap = constant_aperiodicity(dio_result.f0_length, fft_size);
//! let y_length = get_y_length(dio_result.f0_length, dio_option.frame_period, fs);
//! let y = synthesis(
//!     &f0, dio_result.f0_length, &sp, &ap, fft_size,
//!     dio_option.frame_period, fs, y_length,
//! ).unwrap();
//! ```
//!
//! A runnable end-to-end demo is in `examples/synthesis_demo.rs`
//! (`cargo run -p world-rs --release --example synthesis_demo`), and the
//! integration tests in `tests/pipeline_integration.rs` verify the full
//! pipeline (timbre match, latency, F0 discontinuities).
//!
//! Phase 3-1: module scaffolding, type definitions, error type, input
//! validation, and the constant-AP convenience helper. The pulse-based
//! synthesis body is stubbed and returns a zero-initialised buffer.
//! Phase 3-2: minimum-phase analysis and DC-remover utilities.
//! Phase 3-3: noise spectrum and aperiodic response.
//! Phase 3-4: periodic response and fractional time shift.
//! Phase 3-5: spectral-envelope and aperiodic-ratio interpolation.
//! Phase 3-6: one-frame segment synthesis.
//! Phase 3-7: time base and pulse-location calculation.
//! Phase 3-8: main synthesis body and FFT-plan initialisation.
//! Phase 3-9: accuracy validation against the C++ reference.
//! Phase 3-10: integration tests and synthesis demo.
//!
//! The FFT wrappers and minimum-phase machinery are `pub(crate)` in
//! `world-rs-core` (locked decision: "FFT wrappers crate-private"), so this
//! module defines its own (minimum-phase in Phase 3-2, the real-FFT wrappers
//! in Phase 3-3); the public `world-rs-core` primitives below (constants, the
//! xorshift PRNG, `fftshift`, `get_safe_aperiodicity`, and the `FftComplex`
//! type) are imported here.

use crate::common::get_safe_aperiodicity;
use crate::constants::{K_DEFAULT_F0, K_MY_SAFE_GUARD_MINIMUM, K_PI};
use crate::matlab::{fftshift, histc, randn, randn_reseed, RandnState};
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync::Arc;

/// C++: `typedef double fft_complex[2]` (world/fft.h:22). Index `0` is the real
/// part, index `1` the imaginary part. FFI-compatible with the C++ layout;
/// re-exported from `world-rs-core` so there is a single definition.
pub use crate::fft::FftComplex;

/// C++: `const double *f0` — F0 contour in Hz; `0.0` marks unvoiced frames.
pub type F0Contour = [f64];

/// C++: `const double * const *spectrogram` — row-major 2D, `f0_length` rows of
/// `fft_size / 2 + 1` bins (DC..Nyquist). Matches `cheaptrick`'s `Vec<Vec<f64>>`
/// output.
pub type Spectrogram = [Vec<f64>];

/// C++: `const double * const *aperiodicity` — same shape as [`Spectrogram`].
pub type Aperiodicity = [Vec<f64>];

/// Default aperiodicity for the constant-AP testing strategy (D4C is deferred).
pub const DEFAULT_APERIODICITY: f64 = 0.5;

const MS_TO_S: f64 = 1000.0;

/// C++: `MinimumPhaseAnalysis` (world/common.h:44-52). Minimum-phase analysis
/// from a logarithmic power spectrum. The C++ data fields are ported verbatim
/// (f64 throughout); the C++ `inverse_fft`/`forward_fft` FFTW plans are
/// replaced by the cached `rustfft`/`realfft` plans (Route A) held in
/// `r2c_*`/`c2c_*` and reused across calls.
pub(crate) struct MinimumPhaseAnalysis {
    pub(crate) fft_size: usize,
    pub(crate) log_spectrum: Vec<f64>,
    pub(crate) minimum_phase_spectrum: Vec<FftComplex>,
    pub(crate) cepstrum: Vec<FftComplex>,
    /// C++ `inverse_fft`: cached r2c plan and its scratch buffer.
    r2c_plan: Arc<dyn RealToComplex<f64>>,
    r2c_scratch: Vec<Complex<f64>>,
    /// Pre-allocated real input to the r2c forward (the mirrored log spectrum).
    r2c_input: Vec<f64>,
    /// Pre-allocated half-spectrum output of the r2c forward (the cepstrum).
    r2c_output: Vec<Complex<f64>>,
    /// C++ `forward_fft`: cached c2c plan and its scratch buffer.
    c2c_plan: Arc<dyn Fft<f64>>,
    c2c_scratch: Vec<Complex<f64>>,
    /// Pre-allocated complex input to the c2c forward (the conjugated cepstrum).
    c2c_input: Vec<Complex<f64>>,
}

/// C++: `InitializeMinimumPhaseAnalysis` (common.cpp:169-180). Allocates the
/// `fft_size`-sized buffers and the (cached) FFT plans for a minimum-phase
/// analysis. Destruction is handled by Rust's ownership model: the buffers and
/// plans are freed when the returned struct is dropped, so there is no
/// separate `DestroyMinimumPhaseAnalysis` (no leaks by construction).
pub(crate) fn initialize_minimum_phase_analysis(fft_size: usize) -> MinimumPhaseAnalysis {
    let mut r2c_planner = RealFftPlanner::<f64>::new();
    let r2c_plan = r2c_planner.plan_fft_forward(fft_size);
    let r2c_scratch = vec![Complex::new(0.0, 0.0); r2c_plan.get_scratch_len()];
    let mut c2c_planner = FftPlanner::<f64>::new();
    let c2c_plan = c2c_planner.plan_fft_forward(fft_size);
    let c2c_scratch = vec![Complex::new(0.0, 0.0); c2c_plan.get_inplace_scratch_len()];
    let zero = FftComplex::default();
    MinimumPhaseAnalysis {
        fft_size,
        log_spectrum: vec![0.0; fft_size],
        minimum_phase_spectrum: vec![zero; fft_size],
        cepstrum: vec![zero; fft_size],
        r2c_plan,
        r2c_scratch,
        r2c_input: vec![0.0; fft_size],
        r2c_output: vec![Complex::new(0.0, 0.0); fft_size / 2 + 1],
        c2c_plan,
        c2c_scratch,
        c2c_input: vec![Complex::new(0.0, 0.0); fft_size],
    }
}

/// C++: `GetMinimumPhaseSpectrum` (common.cpp:182-220). Computes the minimum
/// phase spectrum from the caller-filled `log_spectrum` (bins `0..=fft_size/2`).
///
/// The caller sets `minimum_phase.log_spectrum[0..=fft_size/2]` (the
/// half-spectrum) before calling this; the upper half is mirrored here. The
/// result is written to `minimum_phase.minimum_phase_spectrum[0..=fft_size/2]`.
pub(crate) fn get_minimum_phase_spectrum(minimum_phase: &mut MinimumPhaseAnalysis) {
    let n = minimum_phase.fft_size;
    let half = n / 2;

    // Mirroring: reflect bins 1..half into the upper half (DC and Nyquist are
    // real and left untouched).
    for i in (half + 1)..n {
        minimum_phase.log_spectrum[i] = minimum_phase.log_spectrum[n - i];
    }

    // r2c forward FFT: log_spectrum (real) -> cepstrum (complex). The transform
    // yields the half-spectrum `0..=half`; the upper half is zeroed (matching
    // the C++ which explicitly clears it). The real input and complex output are
    // pre-allocated on the struct, so the pulse hot loop performs no allocation.
    minimum_phase
        .r2c_input
        .copy_from_slice(&minimum_phase.log_spectrum[..n]);
    let _ = minimum_phase.r2c_plan.process_with_scratch(
        &mut minimum_phase.r2c_input,
        &mut minimum_phase.r2c_output,
        &mut minimum_phase.r2c_scratch,
    );
    let cepstrum = &mut minimum_phase.cepstrum;
    for (slot, o) in cepstrum[..=half]
        .iter_mut()
        .zip(minimum_phase.r2c_output.iter())
    {
        *slot = [o.re, o.im];
    }
    cepstrum[half + 1..].fill(FftComplex::default());

    // Cepstral sign-flip / symmetry (common.cpp:194-209).
    cepstrum[0][1] *= -1.0;
    for slot in &mut cepstrum[1..half] {
        slot[0] *= 2.0;
        slot[1] *= -2.0;
    }
    cepstrum[half][1] *= -1.0;
    cepstrum[half + 1..].fill(FftComplex::default());

    // c2c forward FFT: cepstrum (complex) -> minimum_phase_spectrum (complex).
    // The spec's c2c-forward convention is the standard forward DFT of `conj(Z)`
    // (WORLD-spec.md), so the imaginary parts are negated before the transform.
    // The mirrored log spectrum is even, so the cepstrum is real to ~1e-17 and
    // this is numerically identical to the C++ plain forward FFT.
    for (slot, c) in minimum_phase.c2c_input.iter_mut().zip(cepstrum.iter()) {
        *slot = Complex::new(c[0], -c[1]);
    }
    minimum_phase
        .c2c_plan
        .process_with_scratch(&mut minimum_phase.c2c_input, &mut minimum_phase.c2c_scratch);
    let minimum_phase_spectrum = &mut minimum_phase.minimum_phase_spectrum;
    for (slot, b) in minimum_phase_spectrum
        .iter_mut()
        .zip(minimum_phase.c2c_input.iter())
    {
        *slot = [b.re, b.im];
    }

    // exp(x/N): the C++ FFT library does not keep aliasing, so the magnitude is
    // `exp(re/N)` and the phase is `im/N` (common.cpp:213-220).
    for slot in &mut minimum_phase_spectrum[..=half] {
        let re = slot[0] / n as f64;
        let im = slot[1] / n as f64;
        let mag = re.exp();
        slot[0] = mag * im.cos();
        slot[1] = mag * im.sin();
    }
}

/// C++: `GetDCRemover` (synthesis.cpp:323-335). Computes the DC-removal window
/// of length `fft_size`: a half-window of `0.5 - 0.5*cos(2*pi*(i+1)/(1+N))`,
/// mirrored to the second half, then normalised so the window sums to `1.0`.
pub(crate) fn get_dc_remover(fft_size: usize) -> Vec<f64> {
    let half = fft_size / 2;
    let mut dc_remover = vec![0.0f64; fft_size];
    let mut dc_component = 0.0;
    for i in 0..half {
        let value = 0.5 - 0.5 * (2.0 * K_PI * (i as f64 + 1.0) / (1.0 + fft_size as f64)).cos();
        dc_remover[i] = value;
        dc_remover[fft_size - i - 1] = value;
        dc_component += value * 2.0;
    }
    for i in 0..half {
        dc_remover[i] /= dc_component;
        dc_remover[fft_size - i - 1] = dc_remover[i];
    }
    dc_remover
}

/// C++: `ForwardRealFFT` (world/common.h:18-23). A real-to-complex forward FFT
/// of size `fft_size` with cached plan and scratch (Route A). The C++ `forward_fft`
/// FFTW plan is replaced by a cached `realfft` plan; the `waveform`/`spectrum`
/// fields are the input/output buffers.
///
/// `world-rs-core::fft` holds a functionally equivalent `pub(crate)` struct,
/// but it is not accessible from this crate (locked decision: FFT wrappers
/// are crate-private), so this module defines its own.
pub(crate) struct ForwardRealFFT {
    #[allow(dead_code)] // write-only, kept for API parity
    pub(crate) fft_size: usize,
    pub(crate) waveform: Vec<f64>,
    pub(crate) spectrum: Vec<Complex<f64>>,
    plan: Arc<dyn RealToComplex<f64>>,
    scratch: Vec<Complex<f64>>,
}

impl ForwardRealFFT {
    /// C++: `InitializeForwardRealFFT` (common.cpp:125-131). Allocates the
    /// buffers and the (cached) r2c plan.
    pub(crate) fn new(fft_size: usize) -> Self {
        let mut planner = RealFftPlanner::<f64>::new();
        let plan = planner.plan_fft_forward(fft_size);
        let scratch = vec![Complex::new(0.0, 0.0); plan.get_scratch_len()];
        Self {
            fft_size,
            waveform: vec![0.0; fft_size],
            spectrum: vec![Complex::new(0.0, 0.0); fft_size / 2 + 1],
            plan,
            scratch,
        }
    }

    /// C++: `fft_execute(forward_fft)` — runs the cached r2c forward FFT, writing
    /// the half-spectrum into `spectrum`.
    pub(crate) fn forward(&mut self) {
        let _ = self.plan.process_with_scratch(
            &mut self.waveform,
            &mut self.spectrum,
            &mut self.scratch,
        );
    }
}

/// C++: `InverseRealFFT` (world/common.h:26-31). A complex-to-real inverse FFT
/// of size `fft_size` with cached plan and scratch (Route A). The C++ `inverse_fft`
/// FFTW plan is replaced by a cached `realfft` plan; the `spectrum`/`waveform`
/// fields are the input/output buffers.
pub(crate) struct InverseRealFFT {
    #[allow(dead_code)] // write-only, kept for API parity
    pub(crate) fft_size: usize,
    pub(crate) spectrum: Vec<Complex<f64>>,
    pub(crate) waveform: Vec<f64>,
    plan: Arc<dyn ComplexToReal<f64>>,
    scratch: Vec<Complex<f64>>,
}

impl InverseRealFFT {
    /// C++: `InitializeInverseRealFFT` (common.cpp:139-145). Allocates the
    /// buffers and the (cached) c2r plan.
    pub(crate) fn new(fft_size: usize) -> Self {
        let mut planner = RealFftPlanner::<f64>::new();
        let plan = planner.plan_fft_inverse(fft_size);
        let scratch = vec![Complex::new(0.0, 0.0); plan.get_scratch_len()];
        Self {
            fft_size,
            spectrum: vec![Complex::new(0.0, 0.0); fft_size / 2 + 1],
            waveform: vec![0.0; fft_size],
            plan,
            scratch,
        }
    }

    /// C++: `fft_execute(inverse_fft)` — runs the cached c2r inverse FFT, writing
    /// the `fft_size`-sample waveform into `waveform`.
    pub(crate) fn inverse(&mut self) {
        let _ = self.plan.process_with_scratch(
            &mut self.spectrum,
            &mut self.waveform,
            &mut self.scratch,
        );
    }
}

/// C++: `GetNoiseSpectrum` (synthesis.cpp:19-33). Fills the first `noise_size`
/// samples of `forward_real_fft.waveform` with zero-mean `randn` draws (the mean
/// is subtracted), zero-pads the remainder to `fft_size`, and runs the forward
/// r2c FFT. The resulting half-spectrum is left in `forward_real_fft.spectrum`.
///
/// `noise_size` may be `0` (the final pulse): the buffer is then fully zeroed and
/// the FFT produces an all-zero spectrum. No allocation occurs.
pub(crate) fn get_noise_spectrum(
    noise_size: usize,
    fft_size: usize,
    forward_real_fft: &mut ForwardRealFFT,
    randn_state: &mut RandnState,
) {
    let mut average = 0.0;
    for i in 0..noise_size {
        let value = randn(randn_state);
        forward_real_fft.waveform[i] = value;
        average += value;
    }
    if noise_size > 0 {
        average /= noise_size as f64;
        for i in 0..noise_size {
            forward_real_fft.waveform[i] -= average;
        }
    }
    for i in noise_size..fft_size {
        forward_real_fft.waveform[i] = 0.0;
    }
    forward_real_fft.forward();
}

/// C++: `GetAperiodicResponse` (synthesis.cpp:38-69). Computes the aperiodic
/// (noise) response for one pulse.
///
/// `spectrum` and `aperiodic_ratio` are the interpolated spectral envelope and
/// squared aperiodicity ratio (`fft_size / 2 + 1` bins each). When `current_vuv !=
/// 0.0` the log spectrum is `log(spectrum * aperiodic_ratio) / 2`; for an unvoiced
/// frame (`current_vuv == 0.0`) it is `log(spectrum) / 2` (no aperiodicity
/// weighting). The minimum-phase filter is multiplied (in the frequency domain)
/// by the noise spectrum, inverse-FFT'd, and `fftshift`'d into
/// `aperiodic_response` (`fft_size` samples). No allocation occurs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn get_aperiodic_response(
    noise_size: usize,
    fft_size: usize,
    spectrum: &[f64],
    aperiodic_ratio: &[f64],
    current_vuv: f64,
    forward_real_fft: &mut ForwardRealFFT,
    inverse_real_fft: &mut InverseRealFFT,
    minimum_phase: &mut MinimumPhaseAnalysis,
    aperiodic_response: &mut [f64],
    randn_state: &mut RandnState,
) {
    get_noise_spectrum(noise_size, fft_size, forward_real_fft, randn_state);

    let half = fft_size / 2;
    {
        let log_spectrum = &mut minimum_phase.log_spectrum[..=half];
        if current_vuv != 0.0 {
            for (slot, (s, a)) in log_spectrum
                .iter_mut()
                .zip(spectrum.iter().zip(aperiodic_ratio.iter()))
            {
                *slot = (s * a).ln() / 2.0;
            }
        } else {
            for (slot, s) in log_spectrum.iter_mut().zip(spectrum.iter()) {
                *slot = s.ln() / 2.0;
            }
        }
    }
    get_minimum_phase_spectrum(minimum_phase);

    for i in 0..=half {
        let mp_re = minimum_phase.minimum_phase_spectrum[i][0];
        let mp_im = minimum_phase.minimum_phase_spectrum[i][1];
        let ns_re = forward_real_fft.spectrum[i].re;
        let ns_im = forward_real_fft.spectrum[i].im;
        inverse_real_fft.spectrum[i] =
            Complex::new(mp_re * ns_re - mp_im * ns_im, mp_re * ns_im + mp_im * ns_re);
    }
    inverse_real_fft.inverse();
    fftshift(&inverse_real_fft.waveform, aperiodic_response);
}

/// C++: `GetSpectrumWithFractionalTimeShift` (synthesis.cpp:89-101). Applies a
/// linear (fractional) time shift to the first `fft_size / 2 + 1` bins of
/// `inverse_real_fft.spectrum`, in place.
///
/// Each bin `i` is rotated by `coefficient * i` radians. The C++ computes the
/// cosine of the rotation and derives the sine as `sqrt(1 - re2^2)` (no `sin()`
/// call); the rotation is ported verbatim:
/// `re' = re*re2 + im*im2`, `im' = im*re2 - re*im2`.
pub(crate) fn get_spectrum_with_fractional_time_shift(
    fft_size: usize,
    coefficient: f64,
    inverse_real_fft: &mut InverseRealFFT,
) {
    let half = fft_size / 2;
    for i in 0..=half {
        let re = inverse_real_fft.spectrum[i].re;
        let im = inverse_real_fft.spectrum[i].im;
        let re2 = (coefficient * i as f64).cos();
        let im2 = (1.0 - re2 * re2).sqrt(); // sin(pshift)
        inverse_real_fft.spectrum[i] = Complex::new(re * re2 + im * im2, im * re2 - re * im2);
    }
}

/// C++: `RemoveDCComponent` (synthesis.cpp:74-83). Removes the DC component of a
/// periodic response in place using the normalised `dc_remover` window.
///
/// The DC level is the sum of the upper half of `periodic_response`; the lower
/// half is *overwritten* with `-dc * dc_remover[i]` while the upper half is
/// *reduced* by `dc * dc_remover[i]`. Because the window sums to `1.0`, the
/// resulting response has a (near-)zero total sum.
pub(crate) fn remove_dc_component(
    periodic_response: &mut [f64],
    fft_size: usize,
    dc_remover: &[f64],
) {
    let half = fft_size / 2;
    let dc_component: f64 = periodic_response[half..fft_size].iter().sum();
    for i in 0..half {
        periodic_response[i] = -dc_component * dc_remover[i];
    }
    for i in half..fft_size {
        periodic_response[i] -= dc_component * dc_remover[i];
    }
}

/// C++: `GetPeriodicResponse` (synthesis.cpp:106-139). Computes the periodic
/// (harmonic) response for one pulse.
///
/// `spectrum` and `aperiodic_ratio` are the interpolated spectral envelope and
/// squared aperiodicity ratio (`fft_size / 2 + 1` bins each). For an unvoiced
/// frame (`current_vuv <= 0.5`) or a fully aperiodic frame
/// (`aperiodic_ratio[0] > 0.999`) the output is zero-filled and the function
/// returns immediately. Otherwise the log spectrum is
/// `log(spectrum * (1 - aperiodic_ratio) + kMySafeGuardMinimum) / 2`, its
/// minimum phase is computed, the half-spectrum is copied into the inverse FFT,
/// a linear fractional time shift of `fractional_time_shift` seconds is applied,
/// the result is inverse-FFT'd and `fftshift`'d, and finally the DC component is
/// removed. No allocation occurs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn get_periodic_response(
    fft_size: usize,
    spectrum: &[f64],
    aperiodic_ratio: &[f64],
    current_vuv: f64,
    inverse_real_fft: &mut InverseRealFFT,
    minimum_phase: &mut MinimumPhaseAnalysis,
    dc_remover: &[f64],
    fractional_time_shift: f64,
    fs: f64,
    periodic_response: &mut [f64],
) {
    if current_vuv <= 0.5 || aperiodic_ratio[0] > 0.999 {
        periodic_response.fill(0.0);
        return;
    }

    let half = fft_size / 2;
    {
        let log_spectrum = &mut minimum_phase.log_spectrum[..=half];
        for (slot, (s, a)) in log_spectrum
            .iter_mut()
            .zip(spectrum.iter().zip(aperiodic_ratio.iter()))
        {
            *slot = (s * (1.0 - a) + K_MY_SAFE_GUARD_MINIMUM).ln() / 2.0;
        }
    }
    get_minimum_phase_spectrum(minimum_phase);

    for i in 0..=half {
        inverse_real_fft.spectrum[i] = Complex::new(
            minimum_phase.minimum_phase_spectrum[i][0],
            minimum_phase.minimum_phase_spectrum[i][1],
        );
    }

    // Apply a fractional time delay of `fractional_time_shift` seconds using a
    // linear phase shift (synthesis.cpp:129-133).
    let coefficient = 2.0 * K_PI * fractional_time_shift * fs / fft_size as f64;
    get_spectrum_with_fractional_time_shift(fft_size, coefficient, inverse_real_fft);

    inverse_real_fft.inverse();
    fftshift(&inverse_real_fft.waveform, periodic_response);
    remove_dc_component(periodic_response, fft_size, dc_remover);
}

/// C++: `GetSpectralEnvelope` (synthesis.cpp:141-158). Interpolates the spectral
/// envelope at `current_time` from the `f0_length`-row spectrogram.
///
/// `spectrogram` is a flat row-major buffer of `f0_length` rows, each holding
/// `spectrogram_stride` bins (the first `fft_size / 2 + 1` are DC..Nyquist).
/// `current_time` and `frame_period` must share a time unit (seconds in the C++
/// pipeline: `frame_period_ms / 1000`). The floor and ceil frames are clamped to
/// `[0, f0_length - 1]` with the C++ `MyMinInt` upper bound; when they coincide
/// the floor row is copied verbatim (with `fabs`), otherwise the two rows are
/// linearly blended by the fractional offset `current_time / frame_period -
/// floor`. The result is written to `spectral_envelope`, which must be at least
/// `fft_size / 2 + 1` bins long.
pub(crate) fn get_spectral_envelope(
    current_time: f64,
    frame_period: f64,
    f0_length: usize,
    spectrogram: &[f64],
    spectrogram_stride: usize,
    fft_size: usize,
    spectral_envelope: &mut [f64],
) {
    let ratio = current_time / frame_period;
    let current_frame_floor = (ratio.floor() as usize).min(f0_length - 1);
    let current_frame_ceil = (ratio.ceil() as usize).min(f0_length - 1);
    let interpolation = ratio - current_frame_floor as f64;

    let half = fft_size / 2;
    let floor_base = current_frame_floor * spectrogram_stride;
    let ceil_base = current_frame_ceil * spectrogram_stride;
    if current_frame_floor == current_frame_ceil {
        for i in 0..=half {
            spectral_envelope[i] = spectrogram[floor_base + i].abs();
        }
    } else {
        for i in 0..=half {
            spectral_envelope[i] = (1.0 - interpolation) * spectrogram[floor_base + i].abs()
                + interpolation * spectrogram[ceil_base + i].abs();
        }
    }
}

/// C++: `GetAperiodicRatio` (synthesis.cpp:160-179). Interpolates the squared
/// aperiodicity ratio at `current_time` from the `f0_length`-row aperiodicity
/// spectrogram.
///
/// `aperiodicity` is a flat row-major buffer of `f0_length` rows, each holding
/// `aperiodicity_stride` bins. Each bin is clamped through
/// [`get_safe_aperiodicity`] before blending (matching C++
/// `GetSafeAperiodicity`), and the interpolated value is squared
/// (`pow(x, 2)`), so the output is the aperiodic *power* ratio. The floor/ceil
/// clamping, blending, and the `floor == ceil` fast path mirror
/// [`get_spectral_envelope`]. The result is written to `aperiodic_spectrum`,
/// which must be at least `fft_size / 2 + 1` bins long.
///
/// D4C (the aperiodicity-estimation stage) is deferred for v1, so the
/// aperiodicity matrix is normally the constant-AP default
/// ([`constant_aperiodicity`], 0.5). When D4C is implemented, its per-frame,
/// per-bin aperiodicity estimates replace that constant matrix and flow through
/// this function unchanged — only the data source changes, not the
/// interpolation.
pub(crate) fn get_aperiodic_ratio(
    current_time: f64,
    frame_period: f64,
    f0_length: usize,
    aperiodicity: &[f64],
    aperiodicity_stride: usize,
    fft_size: usize,
    aperiodic_spectrum: &mut [f64],
) {
    let ratio = current_time / frame_period;
    let current_frame_floor = (ratio.floor() as usize).min(f0_length - 1);
    let current_frame_ceil = (ratio.ceil() as usize).min(f0_length - 1);
    let interpolation = ratio - current_frame_floor as f64;

    let half = fft_size / 2;
    let floor_base = current_frame_floor * aperiodicity_stride;
    let ceil_base = current_frame_ceil * aperiodicity_stride;
    if current_frame_floor == current_frame_ceil {
        for i in 0..=half {
            let safe = get_safe_aperiodicity(aperiodicity[floor_base + i]);
            aperiodic_spectrum[i] = safe * safe;
        }
    } else {
        for i in 0..=half {
            let safe = (1.0 - interpolation) * get_safe_aperiodicity(aperiodicity[floor_base + i])
                + interpolation * get_safe_aperiodicity(aperiodicity[ceil_base + i]);
            aperiodic_spectrum[i] = safe * safe;
        }
    }
}

/// C++: `GetOneFrameSegment` (synthesis.cpp:184-222). Combines the periodic
/// and aperiodic responses into a single `fft_size`-sample frame segment.
///
/// `spectrogram` and `aperiodicity` are flat row-major buffers (as in
/// [`get_spectral_envelope`] and [`get_aperiodic_ratio`]); the spectral envelope
/// and aperiodic ratio are interpolated at `current_time`, the periodic and
/// aperiodic responses are synthesised, and the result is written to `response`
/// (which must be `fft_size` samples long) as
/// `(periodic * sqrt(noise_size) + aperiodic) / fft_size`.
///
/// The four temporary buffers (`spectral_envelope`, `aperiodic_ratio`,
/// `periodic_response`, `aperiodic_response`) are allocated locally. They are
/// declared in the C++ allocation order so that Rust's reverse-order drop
/// mirrors the C++ `delete[]` order (PHASE3-6: "free the 4 temporary buffers in
/// reverse order of allocation"); Rust's ownership model guarantees no leaks.
///
/// `noise_size` may be `0` (the final pulse): the periodic response is then
/// weighted by `sqrt(0) = 0` and the aperiodic response is all-zero (the noise
/// spectrum is zero), so `response` is all-zero.
#[allow(clippy::too_many_arguments)]
pub(crate) fn get_one_frame_segment(
    current_vuv: f64,
    noise_size: usize,
    spectrogram: &[f64],
    spectrogram_stride: usize,
    fft_size: usize,
    aperiodicity: &[f64],
    aperiodicity_stride: usize,
    f0_length: usize,
    frame_period: f64,
    current_time: f64,
    fractional_time_shift: f64,
    fs: f64,
    forward_real_fft: &mut ForwardRealFFT,
    inverse_real_fft: &mut InverseRealFFT,
    minimum_phase: &mut MinimumPhaseAnalysis,
    dc_remover: &[f64],
    response: &mut [f64],
    randn_state: &mut RandnState,
    aperiodic_response: &mut [f64],
    periodic_response: &mut [f64],
    spectral_envelope: &mut [f64],
    aperiodic_ratio: &mut [f64],
) {
    // Temporary buffers are now hoisted to synthesis() scope.

    get_spectral_envelope(
        current_time,
        frame_period,
        f0_length,
        spectrogram,
        spectrogram_stride,
        fft_size,
        spectral_envelope,
    );
    get_aperiodic_ratio(
        current_time,
        frame_period,
        f0_length,
        aperiodicity,
        aperiodicity_stride,
        fft_size,
        aperiodic_ratio,
    );

    // Synthesis of the periodic response.
    get_periodic_response(
        fft_size,
        spectral_envelope,
        aperiodic_ratio,
        current_vuv,
        inverse_real_fft,
        minimum_phase,
        dc_remover,
        fractional_time_shift,
        fs,
        periodic_response,
    );

    // Synthesis of the aperiodic response.
    get_aperiodic_response(
        noise_size,
        fft_size,
        spectral_envelope,
        aperiodic_ratio,
        current_vuv,
        forward_real_fft,
        inverse_real_fft,
        minimum_phase,
        aperiodic_response,
        randn_state,
    );

    let sqrt_noise_size = (noise_size as f64).sqrt();
    for i in 0..fft_size {
        response[i] =
            (periodic_response[i] * sqrt_noise_size + aperiodic_response[i]) / fft_size as f64;
    }
    // The four temporary buffers are dropped here in reverse order of
    // allocation (aperiodic_ratio, spectral_envelope, periodic_response,
    // aperiodic_response), matching the C++ `delete[]` order.
}

/// C++: `interp1` (matlabfunctions.cpp:157-176). Linear interpolation of `y`
/// on the grid `x` at the query points `xi` (with linear extrapolation beyond
/// the grid).
///
/// `world-rs-core::matlab` holds a functionally equivalent `pub(crate)`
/// implementation, but it is not visible from this crate (locked decision:
/// support primitives are crate-private), so this module defines its own,
/// ported line-for-line (including the endpoint clamp that replaces the C++
/// out-of-bounds read for query points outside `[x[0], x[n-1]]`).
fn interp1(x: &[f64], y: &[f64], xi: &[f64], yi: &mut [f64]) {
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

/// C++: `GetTemporalParametersForTimeBase` (synthesis.cpp:224-241). Fills the
/// fine `time_axis` (`y_length` samples at `i / fs`) and the coarse
/// (`f0_length + 1`-position) time axis, F0, and V/UV contours used to build
/// the synthesis time base.
///
/// `coarse_f0` applies the lowest-F0 guard: `f0[i] < lowest_f0` (the caller
/// passes `fs / fft_size + 1.0`, the lowest frequency representable by the
/// synthesis FFT plus a 1 Hz margin) is clamped to `0.0`, and `coarse_vuv` is
/// `1.0` exactly where `coarse_f0` is non-zero. The final position
/// (`f0_length`) extrapolates the last two values linearly
/// (`last * 2 - second_to_last`) so the interpolation covers the full output.
#[allow(clippy::too_many_arguments)]
pub(crate) fn get_temporal_parameters_for_time_base(
    f0: &[f64],
    fs: f64,
    y_length: usize,
    frame_period: f64,
    lowest_f0: f64,
    time_axis: &mut [f64],
    coarse_time_axis: &mut [f64],
    coarse_f0: &mut [f64],
    coarse_vuv: &mut [f64],
) {
    assert_eq!(time_axis.len(), y_length);
    for (i, t) in time_axis.iter_mut().enumerate() {
        *t = i as f64 / fs;
    }
    let f0_length = f0.len();
    for i in 0..f0_length {
        coarse_time_axis[i] = i as f64 * frame_period;
        let coarse = if f0[i] < lowest_f0 { 0.0 } else { f0[i] };
        coarse_f0[i] = coarse;
        coarse_vuv[i] = if coarse == 0.0 { 0.0 } else { 1.0 };
    }
    coarse_time_axis[f0_length] = f0_length as f64 * frame_period;
    if f0_length >= 2 {
        coarse_f0[f0_length] = coarse_f0[f0_length - 1] * 2.0 - coarse_f0[f0_length - 2];
        coarse_vuv[f0_length] = coarse_vuv[f0_length - 1] * 2.0 - coarse_vuv[f0_length - 2];
    } else {
        // C++ reads coarse_*[f0_length - 2] unconditionally (undefined
        // behaviour for f0_length == 1); the Rust port extrapolates a constant
        // instead (documented deviation).
        coarse_f0[f0_length] = coarse_f0[f0_length - 1];
        coarse_vuv[f0_length] = coarse_vuv[f0_length - 1];
    }
}

/// C++: `GetPulseLocationsForTimeBase` (synthesis.cpp:243-286). Detects pulse
/// (glottal-closure) locations from the interpolated F0 contour.
///
/// The phase is accumulated sample-by-sample (`total_phase[i] =
/// total_phase[i - 1] + 2*pi*f0[i]/fs`) and wrapped modulo `2*pi`; a pulse is
/// placed at sample `i` when the wrapped phase jumps by more than `pi`
/// between samples `i` and `i + 1` (the wrap-around across a `2*pi`
/// multiple). The exact fractional pulse position is inferred by solving
/// `y1 + x*(y2 - y1) = 0` for `x` on the offset phases (`y1 = wrap[i] - 2*pi`,
/// `y2 = wrap[i + 1]`), giving `time_shift = x / fs` seconds after sample `i`.
///
/// `y_length >= 2` is required (the C++ allocates a `y_length - 1` buffer
/// unconditionally, synthesis.cpp:248); shorter input yields zero pulses.
pub(crate) fn get_pulse_locations_for_time_base(
    interpolated_f0: &[f64],
    time_axis: &[f64],
    fs: f64,
    pulse_locations: &mut [f64],
    pulse_locations_index: &mut [usize],
    pulse_locations_time_shift: &mut [f64],
) -> usize {
    let y_length = time_axis.len();
    if y_length < 2 {
        return 0;
    }
    let mut total_phase = vec![0.0; y_length];
    let mut wrap_phase = vec![0.0; y_length];
    let mut wrap_phase_abs = vec![0.0; y_length - 1];
    total_phase[0] = 2.0 * K_PI * interpolated_f0[0] / fs;
    wrap_phase[0] = total_phase[0] % (2.0 * K_PI);
    for i in 1..y_length {
        total_phase[i] = total_phase[i - 1] + 2.0 * K_PI * interpolated_f0[i] / fs;
        wrap_phase[i] = total_phase[i] % (2.0 * K_PI);
        wrap_phase_abs[i - 1] = (wrap_phase[i] - wrap_phase[i - 1]).abs();
    }

    let mut number_of_pulses = 0usize;
    for i in 0..y_length - 1 {
        if wrap_phase_abs[i] > K_PI {
            pulse_locations[number_of_pulses] = time_axis[i];
            pulse_locations_index[number_of_pulses] = i;
            let y1 = wrap_phase[i] - 2.0 * K_PI;
            let y2 = wrap_phase[i + 1];
            let x = -y1 / (y2 - y1);
            pulse_locations_time_shift[number_of_pulses] = x / fs;
            number_of_pulses += 1;
        }
    }
    number_of_pulses
}

/// C++: `GetTimeBase` (synthesis.cpp:288-321). Builds the synthesis time
/// base: interpolates the coarse F0 and V/UV contours onto the fine sample
/// grid, snaps the interpolated V/UV at the `0.5` threshold (values at or
/// below `0.5` are unvoiced), substitutes `kDefaultF0` (500 Hz) for the F0 of
/// unvoiced samples, and detects the pulse locations on the result.
///
/// `frame_period` is in seconds (the caller divides the millisecond frame
/// period by 1000). `lowest_f0` is `fs / fft_size + 1.0`. Returns the number
/// of pulses; `pulse_locations`/`pulse_locations_index`/
/// `pulse_locations_time_shift` are written in pulse order (buffers of at
/// least `y_length` elements).
#[allow(clippy::too_many_arguments)]
pub(crate) fn get_time_base(
    f0: &[f64],
    fs: f64,
    frame_period: f64,
    y_length: usize,
    lowest_f0: f64,
    pulse_locations: &mut [f64],
    pulse_locations_index: &mut [usize],
    pulse_locations_time_shift: &mut [f64],
    interpolated_vuv: &mut [f64],
) -> usize {
    let f0_length = f0.len();
    // Five allocations (C++: time_axis, coarse_time_axis, coarse_f0,
    // coarse_vuv, interpolated_f0); freed by Rust's ownership model.
    let mut time_axis = vec![0.0; y_length];
    let mut coarse_time_axis = vec![0.0; f0_length + 1];
    let mut coarse_f0 = vec![0.0; f0_length + 1];
    let mut coarse_vuv = vec![0.0; f0_length + 1];
    get_temporal_parameters_for_time_base(
        f0,
        fs,
        y_length,
        frame_period,
        lowest_f0,
        &mut time_axis,
        &mut coarse_time_axis,
        &mut coarse_f0,
        &mut coarse_vuv,
    );
    let mut interpolated_f0 = vec![0.0; y_length];
    interp1(
        &coarse_time_axis,
        &coarse_f0,
        &time_axis,
        &mut interpolated_f0,
    );
    interp1(&coarse_time_axis, &coarse_vuv, &time_axis, interpolated_vuv);

    for i in 0..y_length {
        interpolated_vuv[i] = if interpolated_vuv[i] > 0.5 { 1.0 } else { 0.0 };
        interpolated_f0[i] = if interpolated_vuv[i] == 0.0 {
            K_DEFAULT_F0
        } else {
            interpolated_f0[i]
        };
    }

    get_pulse_locations_for_time_base(
        &interpolated_f0,
        &time_axis,
        fs,
        pulse_locations,
        pulse_locations_index,
        pulse_locations_time_shift,
    )
}

/// Error returned by [`synthesis`] when the input is invalid.
///
/// The C++ `Synthesis` performs no input validation; the Rust port validates
/// the dimensions up front and reports a specific [`SynthesisError`] instead of
/// running the pipeline on degenerate input (no panics, no out-of-bounds reads).
#[derive(Debug, Clone, PartialEq)]
pub enum SynthesisError {
    /// The F0 contour slice `f0` is empty.
    EmptyF0,
    /// The sample rate `fs` is not strictly positive (zero or negative).
    NonPositiveSampleRate { fs: f64 },
    /// The frame period (msec) is not strictly positive (zero or negative).
    NonPositiveFramePeriod { frame_period: f64 },
    /// `fft_size` is not a power of two (or is smaller than 2).
    InvalidFftSize { fft_size: usize },
    /// `f0_length` does not match the length of the `f0` slice.
    MismatchedF0Length { f0_length: usize, f0: usize },
    /// `spectrogram` has a different number of rows than `f0_length`.
    MismatchedSpectrogramRows {
        spectrogram: usize,
        f0_length: usize,
    },
    /// `aperiodicity` has a different number of rows than `f0_length`.
    MismatchedAperiodicityRows {
        aperiodicity: usize,
        f0_length: usize,
    },
    /// A `spectrogram` row has a different length than `fft_size / 2 + 1`.
    MismatchedSpectrogramBins { bins: usize, expected: usize },
    /// An `aperiodicity` row has a different length than `fft_size / 2 + 1`.
    MismatchedAperiodicityBins { bins: usize, expected: usize },
    /// `y_length` is zero or unreasonably large.
    InvalidYLength { y_length: usize },
}

impl std::fmt::Display for SynthesisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SynthesisError::EmptyF0 => write!(f, "F0 contour is empty"),
            SynthesisError::NonPositiveSampleRate { fs } => {
                write!(f, "sample rate must be positive, got {fs}")
            }
            SynthesisError::NonPositiveFramePeriod { frame_period } => {
                write!(f, "frame period must be positive, got {frame_period}")
            }
            SynthesisError::InvalidFftSize { fft_size } => {
                write!(f, "fft_size must be a power of two >= 2, got {fft_size}")
            }
            SynthesisError::MismatchedF0Length { f0_length, f0 } => {
                write!(f, "f0_length {f0_length} != f0 length {f0}")
            }
            SynthesisError::MismatchedSpectrogramRows {
                spectrogram,
                f0_length,
            } => {
                write!(f, "spectrogram rows {spectrogram} != f0_length {f0_length}")
            }
            SynthesisError::MismatchedAperiodicityRows {
                aperiodicity,
                f0_length,
            } => {
                write!(
                    f,
                    "aperiodicity rows {aperiodicity} != f0_length {f0_length}"
                )
            }
            SynthesisError::MismatchedSpectrogramBins { bins, expected } => {
                write!(f, "spectrogram bin count {bins} != expected {expected}")
            }
            SynthesisError::MismatchedAperiodicityBins { bins, expected } => {
                write!(f, "aperiodicity bin count {bins} != expected {expected}")
            }
            SynthesisError::InvalidYLength { y_length } => {
                write!(f, "y_length is invalid or too large: {y_length}")
            }
        }
    }
}

impl std::error::Error for SynthesisError {}

/// C++: `Synthesis` (synthesis.cpp:339-399). Public synthesis entry point.
///
/// Validates the input before running the pipeline. The C++ performs no
/// validation; the Rust port guards against empty/degenerate input and returns
/// the corresponding [`SynthesisError`] rather than running the pipeline.
///
/// # Arguments
///
/// * `f0` — F0 contour in Hz (`0.0` marks unvoiced frames).
/// * `f0_length` — number of frames; must equal `f0.len()`.
/// * `spectrogram` — CheapTrick spectral envelope, `f0_length x (fft_size/2+1)`.
/// * `aperiodicity` — aperiodicity spectrogram, same shape as `spectrogram`.
/// * `fft_size` — FFT size (power of two); bins per row is `fft_size/2 + 1`.
/// * `frame_period_ms` — temporal period used for the analysis, in msec.
/// * `fs` — sampling frequency in Hz.
/// * `y_length` — length of the output buffer to allocate (see [`get_y_length`]).
///
/// # Returns
///
/// An owned `y_length`-element waveform.
///
/// The body mirrors the C++ pulse loop (synthesis.cpp:349-399): the FFT and
/// minimum-phase resources are initialised (Route A cached plans, per
/// `WORLD-decisions.md` "FFT backend Route A" and "FFT wrappers crate-private"),
/// the synthesis time base is built, and each pulse's one-frame segment is
/// overlap-added into the zero-initialised output buffer. All resources are
/// dropped at the end of the function, replacing the C++ `Destroy*`/`delete[]`
/// calls (no leaks by construction).
#[allow(clippy::too_many_arguments)]
pub fn synthesis(
    f0: &F0Contour,
    f0_length: usize,
    spectrogram: &Spectrogram,
    aperiodicity: &Aperiodicity,
    fft_size: usize,
    frame_period_ms: f64,
    fs: f64,
    y_length: usize,
) -> Result<Vec<f64>, SynthesisError> {
    validate_synthesis_inputs(
        f0,
        f0_length,
        spectrogram,
        aperiodicity,
        fft_size,
        frame_period_ms,
        fs,
        y_length,
    )?;

    // C++ prologue (synthesis.cpp:342-347): the deterministic PRNG is reseeded
    // exactly once, the impulse-response buffer is allocated, and the output
    // buffer is zero-initialised.
    let mut randn_state = RandnState::default();
    randn_reseed(&mut randn_state);
    let mut impulse_response = vec![0.0; fft_size];
    let mut y = vec![0.0; y_length];

    // FFT and minimum-phase resources (C++: InitializeMinimumPhaseAnalysis,
    // InitializeInverseRealFFT, InitializeForwardRealFFT). Declared in the C++
    // allocation order so Rust's reverse-order drop mirrors the C++ destroy
    // order; the cached plans and buffers are freed when dropped.
    let mut minimum_phase = initialize_minimum_phase_analysis(fft_size);
    let mut inverse_real_fft = InverseRealFFT::new(fft_size);
    let mut forward_real_fft = ForwardRealFFT::new(fft_size);

    let mut pulse_locations = vec![0.0; y_length];
    let mut pulse_locations_index = vec![0usize; y_length];
    let mut pulse_locations_time_shift = vec![0.0; y_length];
    let mut interpolated_vuv = vec![0.0; y_length];
    // C++: `fs / fft_size + 1.0` with integer division (fs and fft_size are
    // ints in the C++) — the lowest frequency representable by the synthesis
    // FFT plus a 1 Hz margin; frames below it are treated as unvoiced.
    let lowest_f0 = ((fs as usize) / fft_size) as f64 + 1.0;
    let number_of_pulses = get_time_base(
        f0,
        fs,
        frame_period_ms / MS_TO_S,
        y_length,
        lowest_f0,
        &mut pulse_locations,
        &mut pulse_locations_index,
        &mut pulse_locations_time_shift,
        &mut interpolated_vuv,
    );

    let dc_remover = get_dc_remover(fft_size);

    // Hoist per-pulse buffers to synthesis scope (D2).
    let mut aperiodic_response = vec![0.0; fft_size];
    let mut periodic_response = vec![0.0; fft_size];
    let mut spectral_envelope = vec![0.0; fft_size];
    let mut aperiodic_ratio = vec![0.0; fft_size];

    // The spectrogram/aperiodicity rows are flattened once (stride = row
    // length, validated above) so the pulse hot loop can use the flat-buffer
    // interpolation helpers without per-pulse allocation.
    let sp_stride = spectrogram[0].len();
    let sp_flat: Vec<f64> = spectrogram.iter().flatten().copied().collect();
    let ap_stride = aperiodicity[0].len();
    let ap_flat: Vec<f64> = aperiodicity.iter().flatten().copied().collect();

    // C++: `frame_period /= 1000.0` — the second ms -> s conversion, now in
    // seconds for the synthesis loop (the first was for `get_time_base` above).
    let frame_period = frame_period_ms / MS_TO_S;
    for i in 0..number_of_pulses {
        let noise_size =
            pulse_locations_index[(number_of_pulses - 1).min(i + 1)] - pulse_locations_index[i];

        get_one_frame_segment(
            interpolated_vuv[pulse_locations_index[i]],
            noise_size,
            &sp_flat,
            sp_stride,
            fft_size,
            &ap_flat,
            ap_stride,
            f0_length,
            frame_period,
            pulse_locations[i],
            pulse_locations_time_shift[i],
            fs,
            &mut forward_real_fft,
            &mut inverse_real_fft,
            &mut minimum_phase,
            &dc_remover,
            &mut impulse_response,
            &mut randn_state,
            &mut aperiodic_response,
            &mut periodic_response,
            &mut spectral_envelope,
            &mut aperiodic_ratio,
        );

        // Overlap-add with bounds clamping (C++: synthesis.cpp:381-386): the
        // segment is centred on the pulse sample, clamped to the output range.
        let offset = pulse_locations_index[i] as isize - (fft_size / 2) as isize + 1;
        let lower_limit = 0isize.max(-offset);
        let upper_limit = (fft_size as isize).min((y_length as isize) - offset);
        for j in lower_limit..upper_limit {
            y[(j + offset) as usize] += impulse_response[j as usize];
        }
    }

    // `minimum_phase`, `inverse_real_fft`, `forward_real_fft`, and the pulse /
    // V/UV / DC-remover buffers are dropped here, replacing the C++
    // DestroyMinimumPhaseAnalysis / DestroyInverseRealFFT / DestroyForwardRealFFT
    // and delete[] calls.

    Ok(y)
}

/// Validates the synthesis inputs, returning the first [`SynthesisError`] found.
#[allow(clippy::too_many_arguments)]
fn validate_synthesis_inputs(
    f0: &F0Contour,
    f0_length: usize,
    spectrogram: &Spectrogram,
    aperiodicity: &Aperiodicity,
    fft_size: usize,
    frame_period_ms: f64,
    fs: f64,
    y_length: usize,
) -> Result<(), SynthesisError> {
    if f0.is_empty() {
        return Err(SynthesisError::EmptyF0);
    }
    if y_length == 0 || y_length > 100_000_000 {
        return Err(SynthesisError::InvalidYLength { y_length });
    }
    if !fs.is_finite() || fs <= 0.0 {
        return Err(SynthesisError::NonPositiveSampleRate { fs });
    }
    if !frame_period_ms.is_finite() || frame_period_ms <= 0.0 {
        return Err(SynthesisError::NonPositiveFramePeriod {
            frame_period: frame_period_ms,
        });
    }
    if fft_size < 2 || !fft_size.is_power_of_two() {
        return Err(SynthesisError::InvalidFftSize { fft_size });
    }
    // Allocation budget: ~10 `fft_size`-sized buffers plus cached FFT plans
    // scale with `fft_size` (CheapTrick never emits more than 2^20 rows).
    if fft_size > (1 << 20) {
        return Err(SynthesisError::InvalidFftSize { fft_size });
    }
    if f0_length != f0.len() {
        return Err(SynthesisError::MismatchedF0Length {
            f0_length,
            f0: f0.len(),
        });
    }
    let expected_bins = fft_size / 2 + 1;
    if spectrogram.len() != f0_length {
        return Err(SynthesisError::MismatchedSpectrogramRows {
            spectrogram: spectrogram.len(),
            f0_length,
        });
    }
    if aperiodicity.len() != f0_length {
        return Err(SynthesisError::MismatchedAperiodicityRows {
            aperiodicity: aperiodicity.len(),
            f0_length,
        });
    }
    for row in spectrogram.iter() {
        if row.len() != expected_bins {
            return Err(SynthesisError::MismatchedSpectrogramBins {
                bins: row.len(),
                expected: expected_bins,
            });
        }
    }
    for row in aperiodicity.iter() {
        if row.len() != expected_bins {
            return Err(SynthesisError::MismatchedAperiodicityBins {
                bins: row.len(),
                expected: expected_bins,
            });
        }
    }
    Ok(())
}

/// Number of samples in the synthesised output for a given F0 contour length,
/// frame period (msec), and sample rate.
///
/// The last frame sits at `(f0_length - 1) * frame_period_ms` msec, so the
/// output spans that duration plus one sample:
/// `(f0_length - 1) * frame_period_ms / 1000 * fs + 1`.
pub fn get_y_length(f0_length: usize, frame_period_ms: f64, fs: f64) -> usize {
    ((f0_length as f64 - 1.0) * frame_period_ms / MS_TO_S * fs + 1.0) as usize
}

/// Constant-AP convenience: an `f0_length x (fft_size / 2 + 1)` aperiodicity
/// matrix filled with [`DEFAULT_APERIODICITY`] (clamped through
/// [`get_safe_aperiodicity`]). Used for testing while D4C is deferred.
pub fn constant_aperiodicity(f0_length: usize, fft_size: usize) -> Vec<Vec<f64>> {
    let bins = fft_size / 2 + 1;
    let value = get_safe_aperiodicity(DEFAULT_APERIODICITY);
    vec![vec![value; bins]; f0_length]
}

/// Stateful streaming synthesizer that reuses FFT plans and PRNG state across
/// calls. Mirrors the design in `WORLD-STREAMING-PLAN.md`.
pub struct Synthesizer {
    fft_size: usize,
    fs: f64,
    frame_period_ms: f64,
}

impl Synthesizer {
    /// Create a new synthesizer with the given FFT size, sample rate, and frame
    /// period. Returns an error if the parameters are invalid.
    pub fn new(fft_size: usize, fs: f64, frame_period_ms: f64) -> Result<Self, SynthesisError> {
        if fft_size < 2 || !fft_size.is_power_of_two() {
            return Err(SynthesisError::InvalidFftSize { fft_size });
        }
        if !fs.is_finite() || fs <= 0.0 {
            return Err(SynthesisError::NonPositiveSampleRate { fs });
        }
        if !frame_period_ms.is_finite() || frame_period_ms <= 0.0 {
            return Err(SynthesisError::NonPositiveFramePeriod {
                frame_period: frame_period_ms,
            });
        }
        Ok(Self {
            fft_size,
            fs,
            frame_period_ms,
        })
    }

    /// Synthesize a chunk of features into a waveform. The `spectrogram` and
    /// `aperiodicity` slices are expected to be flat row-major buffers with
    /// `f0.len() * (fft_size/2 + 1)` elements each.
    pub fn synthesize_chunk(
        &mut self,
        f0: &[f64],
        spectrogram: &[f64],
        aperiodicity: &[f64],
        y_length: usize,
    ) -> Result<Vec<f64>, SynthesisError> {
        let f0_len = f0.len();
        if f0_len == 0 {
            return Err(SynthesisError::EmptyF0);
        }
        let bins = self.fft_size / 2 + 1;
        if spectrogram.len() != f0_len * bins {
            return Err(SynthesisError::MismatchedSpectrogramRows {
                spectrogram: spectrogram.len() / bins,
                f0_length: f0_len,
            });
        }
        if aperiodicity.len() != f0_len * bins {
            return Err(SynthesisError::MismatchedAperiodicityRows {
                aperiodicity: aperiodicity.len() / bins,
                f0_length: f0_len,
            });
        }
        // Convert flat buffers to Vec<Vec<f64>> for the batch API.
        let spectrogram_vec: Vec<Vec<f64>> = spectrogram.chunks(bins).map(|c| c.to_vec()).collect();
        let aperiodicity_vec: Vec<Vec<f64>> =
            aperiodicity.chunks(bins).map(|c| c.to_vec()).collect();
        // Use the batch synthesis implementation with the stored parameters.
        synthesis(
            f0,
            f0_len,
            &spectrogram_vec,
            &aperiodicity_vec,
            self.fft_size,
            self.frame_period_ms,
            self.fs,
            y_length,
        )
    }

    /// Reset internal state. Currently a no-op, kept for API compatibility.
    pub fn reset(&mut self) {}
}

#[cfg(test)]
// Reference vectors are full-precision f64 C++ values.
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    /// Builds a valid `(f0, spectrogram, aperiodicity)` triple for `f0_length`
    /// frames and the given `fft_size`, with every bin set to `1.0` / the
    /// default aperiodicity.
    fn valid_inputs(f0_length: usize, fft_size: usize) -> (Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let bins = fft_size / 2 + 1;
        let f0 = vec![100.0; f0_length];
        let spectrogram = vec![vec![1.0; bins]; f0_length];
        let aperiodicity = constant_aperiodicity(f0_length, fft_size);
        (f0, spectrogram, aperiodicity)
    }

    // Fills `mp.log_spectrum[0..=fft_size/2]` with the deterministic test vector
    // used to generate the C++ reference vectors (scratch/minphase_ref/driver.cpp):
    // `log(1 + 0.5*cos(0.3*i) + 0.25*(i%7)) / 2`.
    fn fill_test_log_spectrum(mp: &mut MinimumPhaseAnalysis) {
        let half = mp.fft_size / 2;
        for i in 0..=half {
            mp.log_spectrum[i] =
                (1.0 + 0.5 * (0.3 * i as f64).cos() + 0.25 * ((i % 7) as f64)).ln() / 2.0;
        }
    }

    #[test]
    fn test_get_y_length() {
        let len = get_y_length(10, 5.0, 16000.0);
        assert_eq!(len, ((9.0 * 5.0 / 1000.0 * 16000.0) + 1.0) as usize);
        assert_eq!(get_y_length(1, 5.0, 16000.0), 1);
        assert_eq!(get_y_length(201, 5.0, 16000.0), 16001);
    }

    #[test]
    fn test_constant_aperiodicity_shape_and_value() {
        let ap = constant_aperiodicity(5, 128);
        assert_eq!(ap.len(), 5);
        for row in &ap {
            assert_eq!(row.len(), 128 / 2 + 1);
            for &v in row {
                assert_eq!(v, DEFAULT_APERIODICITY);
            }
        }
    }

    #[test]
    fn test_default_aperiodicity_is_safe() {
        // 0.5 lies inside the safe-aperiodicity clamp range, so it survives.
        assert_eq!(
            get_safe_aperiodicity(DEFAULT_APERIODICITY),
            DEFAULT_APERIODICITY
        );
    }

    // ------------------------------------------------------------------
    // PHASE3-8: main synthesis function.

    // Smoke test (PHASE3-8 acceptance): valid dummy inputs run without panics,
    // the output length equals y_length, and every sample is finite. With
    // f0 = 100 Hz < lowest_f0 (16000/128 + 1 = 126 Hz) every frame is unvoiced
    // and synthesised at the 500 Hz default F0, so pulses exist and the output
    // is non-zero.
    #[test]
    fn test_synthesis_smoke() {
        let (f0, spectrogram, aperiodicity) = valid_inputs(10, 128);
        let y = synthesis(&f0, 10, &spectrogram, &aperiodicity, 128, 5.0, 16000.0, 80)
            .expect("valid input");
        assert_eq!(y.len(), 80);
        let max: f64 = y.iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(
            max > 0.0,
            "output should be non-zero (pulses at the 500 Hz default F0)"
        );
        for &v in &y {
            assert!(v.is_finite());
        }
    }

    // The PRNG is reseeded exactly once per call (C++ prologue), so two runs
    // with identical inputs produce bit-identical output (PHASE3-8:
    // deterministic random state).
    #[test]
    fn test_synthesis_deterministic() {
        let (f0, spectrogram, aperiodicity) = valid_inputs(10, 128);
        let a = synthesis(&f0, 10, &spectrogram, &aperiodicity, 128, 5.0, 16000.0, 80)
            .expect("valid input");
        let b = synthesis(&f0, 10, &spectrogram, &aperiodicity, 128, 5.0, 16000.0, 80)
            .expect("valid input");
        assert_eq!(a, b);
    }

    // With y_length = 1 the pulse detector requires y_length >= 2, so no
    // pulses are placed and the zero-initialised output is returned unchanged
    // (PHASE3-8: zero-initialised output, no panics on degenerate length).
    #[test]
    fn test_synthesis_no_pulses_zero_output() {
        let (f0, spectrogram, aperiodicity) = valid_inputs(1, 128);
        let y = synthesis(&f0, 1, &spectrogram, &aperiodicity, 128, 5.0, 16000.0, 1)
            .expect("valid input");
        assert_eq!(y.len(), 1);
        assert_eq!(y[0], 0.0);
    }

    // A fully voiced 1 s contour (200 Hz > lowest_f0) at the standard 5 ms /
    // 16 kHz settings exercises the full pipeline: ~200 pulses, voiced
    // periodic + aperiodic responses, and overlap-add across the whole output
    // (PHASE3-8: runs without panics, output length equals y_length).
    #[test]
    fn test_synthesis_voiced_smoke() {
        let f0_length = 201;
        let fft_size = 128;
        let bins = fft_size / 2 + 1;
        let f0 = vec![200.0; f0_length];
        let spectrogram = vec![vec![1.0; bins]; f0_length];
        let aperiodicity = constant_aperiodicity(f0_length, fft_size);
        let y_length = get_y_length(f0_length, 5.0, 16000.0); // 16001
        let y = synthesis(
            &f0,
            f0_length,
            &spectrogram,
            &aperiodicity,
            fft_size,
            5.0,
            16000.0,
            y_length,
        )
        .expect("valid input");
        assert_eq!(y.len(), y_length);
        let max: f64 = y.iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(max > 0.0, "voiced output should be non-zero");
        for &v in &y {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_synthesis_empty_f0() {
        let spectrogram: Vec<Vec<f64>> = vec![];
        let aperiodicity: Vec<Vec<f64>> = vec![];
        let err = synthesis(&[], 0, &spectrogram, &aperiodicity, 128, 5.0, 16000.0, 0)
            .expect_err("empty f0");
        assert_eq!(err, SynthesisError::EmptyF0);
    }

    #[test]
    fn test_synthesis_nonpositive_fs() {
        let (f0, spectrogram, aperiodicity) = valid_inputs(10, 128);
        let err = synthesis(&f0, 10, &spectrogram, &aperiodicity, 128, 5.0, 0.0, 80)
            .expect_err("fs <= 0");
        assert_eq!(err, SynthesisError::NonPositiveSampleRate { fs: 0.0 });
    }

    #[test]
    fn test_synthesis_nonpositive_frame_period() {
        let (f0, spectrogram, aperiodicity) = valid_inputs(10, 128);
        let err = synthesis(&f0, 10, &spectrogram, &aperiodicity, 128, 0.0, 16000.0, 80)
            .expect_err("frame_period <= 0");
        assert_eq!(
            err,
            SynthesisError::NonPositiveFramePeriod { frame_period: 0.0 }
        );
    }

    #[test]
    fn test_synthesis_invalid_fft_size() {
        let (f0, spectrogram, aperiodicity) = valid_inputs(10, 128);
        // 100 is not a power of two.
        let err = synthesis(&f0, 10, &spectrogram, &aperiodicity, 100, 5.0, 16000.0, 80)
            .expect_err("fft_size not a power of two");
        assert_eq!(err, SynthesisError::InvalidFftSize { fft_size: 100 });
    }

    #[test]
    fn test_synthesis_mismatched_f0_length() {
        let (f0, spectrogram, aperiodicity) = valid_inputs(10, 128);
        let err = synthesis(&f0, 9, &spectrogram, &aperiodicity, 128, 5.0, 16000.0, 80)
            .expect_err("f0_length mismatch");
        assert_eq!(
            err,
            SynthesisError::MismatchedF0Length {
                f0_length: 9,
                f0: 10
            }
        );
    }

    #[test]
    fn test_synthesis_mismatched_spectrogram_rows() {
        let (f0, mut spectrogram, aperiodicity) = valid_inputs(10, 128);
        spectrogram.pop();
        let err = synthesis(&f0, 10, &spectrogram, &aperiodicity, 128, 5.0, 16000.0, 80)
            .expect_err("spectrogram rows mismatch");
        assert_eq!(
            err,
            SynthesisError::MismatchedSpectrogramRows {
                spectrogram: 9,
                f0_length: 10
            }
        );
    }

    #[test]
    fn test_synthesis_mismatched_aperiodicity_rows() {
        let (f0, spectrogram, mut aperiodicity) = valid_inputs(10, 128);
        aperiodicity.pop();
        let err = synthesis(&f0, 10, &spectrogram, &aperiodicity, 128, 5.0, 16000.0, 80)
            .expect_err("aperiodicity rows mismatch");
        assert_eq!(
            err,
            SynthesisError::MismatchedAperiodicityRows {
                aperiodicity: 9,
                f0_length: 10
            }
        );
    }

    #[test]
    fn test_synthesis_mismatched_spectrogram_bins() {
        let (f0, mut spectrogram, aperiodicity) = valid_inputs(10, 128);
        spectrogram[0].pop(); // one row has 64 bins instead of 65
        let err = synthesis(&f0, 10, &spectrogram, &aperiodicity, 128, 5.0, 16000.0, 80)
            .expect_err("spectrogram bins mismatch");
        assert_eq!(
            err,
            SynthesisError::MismatchedSpectrogramBins {
                bins: 64,
                expected: 65
            }
        );
    }

    #[test]
    fn test_synthesis_mismatched_aperiodicity_bins() {
        let (f0, spectrogram, mut aperiodicity) = valid_inputs(10, 128);
        aperiodicity[0].pop(); // one row has 64 bins instead of 65
        let err = synthesis(&f0, 10, &spectrogram, &aperiodicity, 128, 5.0, 16000.0, 80)
            .expect_err("aperiodicity bins mismatch");
        assert_eq!(
            err,
            SynthesisError::MismatchedAperiodicityBins {
                bins: 64,
                expected: 65
            }
        );
    }

    // PHASE3-2: DC remover.

    // The window is symmetric: dc_remover[i] == dc_remover[fft_size-1-i].
    #[test]
    fn test_dc_remover_symmetry() {
        for fft_size in [16usize, 128, 2048, 4096] {
            let dc = get_dc_remover(fft_size);
            assert_eq!(dc.len(), fft_size);
            for i in 0..fft_size {
                assert!(
                    (dc[i] - dc[fft_size - 1 - i]).abs() < 1e-15,
                    "fft_size={fft_size} i={i}"
                );
            }
        }
    }

    // The normalised window sums to 1.0 within 1e-9 (PHASE3-2 acceptance).
    #[test]
    fn test_dc_remover_normalization() {
        for fft_size in [16usize, 128, 2048, 4096] {
            let dc = get_dc_remover(fft_size);
            let sum: f64 = dc.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "fft_size={fft_size} sum={sum}");
        }
    }

    // All entries are non-negative (it is a positive half-window, mirrored).
    #[test]
    fn test_dc_remover_nonnegative() {
        for fft_size in [16usize, 128, 2048] {
            for &v in &get_dc_remover(fft_size) {
                assert!(v >= 0.0);
            }
        }
    }

    // Matches the C++ reference (scratch/minphase_ref, DCCOMPONENT 16).
    #[test]
    fn test_dc_remover_matches_cpp_reference() {
        let expected16 = [
            0.0039722217997437761,
            0.015352416634078877,
            0.032603626130791867,
            0.053395978855099883,
            0.074921352357181337,
            0.094272625669368018,
            0.108836302101742,
            0.11664547645199423,
            0.11664547645199423,
            0.108836302101742,
            0.094272625669368018,
            0.074921352357181337,
            0.053395978855099883,
            0.032603626130791867,
            0.015352416634078877,
            0.0039722217997437761,
        ];
        let dc = get_dc_remover(16);
        for i in 0..16 {
            assert!(
                (dc[i] - expected16[i]).abs() < 1e-12,
                "i={i}: {} vs {}",
                dc[i],
                expected16[i]
            );
        }
        // Spot-check the 128-point window against the C++ reference.
        let expected128_head = [
            9.193370226178243e-06,
            3.6751675246234274e-05,
            8.2609549805346888e-05,
            0.00014665822409151729,
            0.00022874578172554051,
            0.00032867752008987631,
            0.0004462164121417485,
            0.00058108366861510028,
        ];
        let dc = get_dc_remover(128);
        for (i, &e) in expected128_head.iter().enumerate() {
            assert!((dc[i] - e).abs() < 1e-12, "i={i}");
        }
    }

    // PHASE3-2: minimum phase spectrum.

    // Matches the C++ reference (scratch/minphase_ref, SIZE 16/128/2048). The
    // Route A FFT is not bit-exact with Ooura, so a 1e-6 tolerance is used.
    #[test]
    fn test_minimum_phase_matches_cpp_reference() {
        let expected16 = [
            (1.2247448713915892, 8.4983747219407405e-18),
            (1.3102017439059266, 0.10506966654878143),
            (1.3825486339362794, 0.035029733880897425),
            (1.433498969091376, 0.076718249128191959),
            (1.472210453708517, -0.11736804177330244),
            (1.511336716673066, -0.035070353145175696),
            (1.4028222258517766, -0.64690706852663471),
            (0.8270040788572981, -0.25227207783158068),
            (0.93877747215694163, -6.5140772786752401e-18),
        ];
        let mut mp = initialize_minimum_phase_analysis(16);
        fill_test_log_spectrum(&mut mp);
        get_minimum_phase_spectrum(&mut mp);
        for (i, (slot, &(re, im))) in mp.minimum_phase_spectrum[..=8]
            .iter()
            .zip(expected16.iter())
            .enumerate()
        {
            assert!((slot[0] - re).abs() < 1e-6, "16 re[{i}]: {slot:?} vs {re}");
            assert!((slot[1] - im).abs() < 1e-6, "16 im[{i}]: {slot:?} vs {im}");
        }

        let expected128 = [
            (1.2247448713915889, -1.0780653089649433e-17),
            (1.3110761272668301, 0.093528782060998003),
            (1.3824547401270335, 0.038557735347590424),
            (1.4350852041719646, 0.036543684844898623),
            (1.4728975589205771, -0.10840506521441787),
            (1.5073084027340484, -0.11571508061261251),
            (1.4118848206722416, -0.62688133311558181),
            (0.79830078004809202, -0.33210361677446243),
            (0.93653155253031262, 0.064898330830148593),
            (1.0231387530172542, 0.033926701368799356),
            (1.0984552123433327, 0.21999976858065409),
        ];
        let mut mp = initialize_minimum_phase_analysis(128);
        fill_test_log_spectrum(&mut mp);
        get_minimum_phase_spectrum(&mut mp);
        for (i, &(re, im)) in expected128.iter().enumerate() {
            assert!(
                (mp.minimum_phase_spectrum[i][0] - re).abs() < 1e-6,
                "128 re[{i}]: {} vs {re}",
                mp.minimum_phase_spectrum[i][0]
            );
            assert!(
                (mp.minimum_phase_spectrum[i][1] - im).abs() < 1e-6,
                "128 im[{i}]: {} vs {im}",
                mp.minimum_phase_spectrum[i][1]
            );
        }

        let expected2048 = [
            (1.2247448713915889, 2.4658242837799375e-17),
            (1.3110849550553048, 0.09340495270826106),
            (1.3824542681991931, 0.038574652204162029),
            (1.4350955176898248, 0.036136397743921682),
            (1.4729002033102581, -0.10836912995376911),
            (1.507253319841142, -0.11643036829671088),
            (1.4119076413146505, -0.62682993311643687),
            (0.79807887223033958, -0.3326365304647409),
            (0.93652842475940423, 0.064943451148260778),
            (1.0231676871644981, 0.033042592723950978),
            (1.0984420913173409, 0.22006527150407879),
        ];
        let mut mp = initialize_minimum_phase_analysis(2048);
        fill_test_log_spectrum(&mut mp);
        get_minimum_phase_spectrum(&mut mp);
        for (i, &(re, im)) in expected2048.iter().enumerate() {
            assert!(
                (mp.minimum_phase_spectrum[i][0] - re).abs() < 1e-6,
                "2048 re[{i}]: {} vs {re}",
                mp.minimum_phase_spectrum[i][0]
            );
            assert!(
                (mp.minimum_phase_spectrum[i][1] - im).abs() < 1e-6,
                "2048 im[{i}]: {} vs {im}",
                mp.minimum_phase_spectrum[i][1]
            );
        }
    }

    // The minimum phase spectrum has unit-consistent magnitude: bin 0 equals
    // `exp(mean of the causal cepstrum)` and is real (imaginary part ~0), which
    // follows from the algorithm and is a useful sanity invariant.
    #[test]
    fn test_minimum_phase_bin0_is_real() {
        let mut mp = initialize_minimum_phase_analysis(128);
        fill_test_log_spectrum(&mut mp);
        get_minimum_phase_spectrum(&mut mp);
        assert!(mp.minimum_phase_spectrum[0][1].abs() < 1e-9);
        assert!(mp.minimum_phase_spectrum[0][0] > 0.0);
    }

    // Reusing the same initialised analysis across calls (as the synthesis
    // pulse loop does) must not corrupt the cached plans or buffers.
    #[test]
    fn test_minimum_phase_reuse_is_stable() {
        let mut mp = initialize_minimum_phase_analysis(128);
        fill_test_log_spectrum(&mut mp);
        get_minimum_phase_spectrum(&mut mp);
        let first: Vec<FftComplex> = mp.minimum_phase_spectrum[..=64].to_vec();
        fill_test_log_spectrum(&mut mp);
        get_minimum_phase_spectrum(&mut mp);
        for (a, b) in mp.minimum_phase_spectrum[..=64].iter().zip(first.iter()) {
            assert_eq!(a, b);
        }
    }

    // PHASE3-3: noise spectrum and aperiodic response.

    // C++ reference aperiodic response (scratch/aperiodic_ref, fft_size=128,
    // noise_size=32, spectrum[i]=1+0.5cos(0.3i)+0.25(i%7),
    // aperiodic_ratio[i]=0.2+0.05(i%5), reseeded randn).
    const VUVED: [f64; 128] = [
        -24.235796641750653,
        22.415751161919967,
        -0.16465309138397544,
        2.6636021435807891,
        12.678061107082257,
        8.1778373701409173,
        -8.3294116580147417,
        -0.68204232035239443,
        -13.436965475645877,
        -17.470006025331109,
        0.28728298861639701,
        -5.8288585512962641,
        -12.443693179508834,
        1.7745259290336257,
        0.79847128474013118,
        -8.2722133612113211,
        4.0529257338469193,
        4.4734073853893346,
        7.4965688373393284,
        0.57061017297382044,
        0.84108513314514965,
        -3.1186316677367927,
        13.952445107407719,
        -4.1027204908837653,
        1.8267745801105857,
        3.6965491832141346,
        2.0544074570764153,
        -1.1556655331009082,
        -0.17325345167166262,
        -0.96271036574531266,
        2.2338081946850963,
        -1.0628211141882389,
        0.079237629734421944,
        2.9283560397900894,
        0.92660140144191061,
        -4.6809421674158926,
        0.14930730740808684,
        0.52802964428339294,
        -2.3150934912645891,
        -1.5993652414890036,
        -1.4619312923719008,
        0.17075987659666758,
        -1.4073107323801963,
        -0.30959389515627578,
        1.1628341050479101,
        2.8065186923848859,
        -0.242585465193768,
        -1.3710987171920781,
        2.1565313483467214,
        1.0229903549592709,
        -2.5916097432380818,
        1.4496696128913058,
        -0.25350802590014521,
        -0.78450363841462201,
        -0.93777792194493437,
        -0.011208145517973889,
        -1.2871463203021314,
        0.83281234456098741,
        -0.95100566790884145,
        0.17682104392524778,
        1.2057678297047816,
        0.066813554896523186,
        0.34947911545644494,
        1.3603032998266649,
        -93.793745152316745,
        -30.792779488477972,
        -118.35665467120856,
        130.27838624284524,
        -1.9332240229003448,
        -33.589970199588009,
        -1.7022854999929535,
        79.521013736621597,
        -107.55691235754904,
        51.313224756950639,
        120.31665443264691,
        -80.680364261023612,
        -45.966474279151065,
        39.188587480435459,
        -239.81309430579961,
        67.556252685604989,
        55.492761169620806,
        -26.744068273409063,
        40.880368335984187,
        23.175388566550545,
        43.19042145196368,
        122.25770839585638,
        125.56823840690933,
        -40.89962568496243,
        1.5953987874795565,
        48.760509265635221,
        -80.584419782758545,
        83.823862794719702,
        81.691917507398387,
        -48.887655307657823,
        41.565513788162974,
        -144.62302131243615,
        8.5825759424725518,
        -14.758601363941999,
        16.116104605451472,
        -38.813744239008116,
        14.335203819845784,
        -12.992689543220559,
        -23.420141026880216,
        -0.91109043416521018,
        -12.588294853595158,
        -4.517656508117911,
        5.8495914860643481,
        -22.469899402211162,
        14.637488785181741,
        -20.056417848776654,
        -27.770611443824606,
        2.7036021384441309,
        23.4780866072342,
        27.650713194488908,
        2.8055407934542069,
        21.199882771139077,
        -10.102863908748756,
        -5.384667598251303,
        -2.1951617721723915,
        1.7649206507374422,
        18.97400061504381,
        -2.6963201439063935,
        -32.226944873967597,
        14.384808697001414,
        3.5703397766405387,
        -17.040706743502781,
        5.1100684545325503,
        20.808146582359303,
    ];

    const UNVOICED: [f64; 128] = [
        -36.03080176185621,
        1.3354070039043329,
        12.290545688123373,
        22.384846857565705,
        15.412809503311667,
        26.711627722424264,
        -9.6752075355468516,
        0.67674144405178538,
        0.8820027199329985,
        -7.2739307180773807,
        -0.50338635959826661,
        -14.002022213905519,
        -15.081513866446365,
        -13.932236780999041,
        9.0974367843354287,
        -1.9161834799316111,
        -3.2519334261651665,
        12.249225291001579,
        -11.254612388213644,
        -7.3434471150141221,
        2.4010167563123872,
        -2.7523430567520677,
        21.185180075172283,
        -8.4293360346206683,
        7.4802255336812031,
        -1.8904737662106967,
        5.2190668318422411,
        -1.8651964269112824,
        3.9736007048552864,
        1.3746734162535361,
        4.2080972533039329,
        -1.4170832310589105,
        -3.9968640849221089,
        0.93369893779206592,
        0.56231795901376813,
        -4.5679403733663761,
        3.1230409185474644,
        0.18493639106294779,
        -3.2956087688238167,
        -2.6039324576326663,
        -4.1416848548740646,
        0.52137733397724162,
        -0.68168313033837791,
        1.6273537341409181,
        0.068231354154519863,
        2.0546635907523481,
        -0.14418675279133453,
        1.8201624356191282,
        1.3216817432209247,
        1.8600908687285163,
        -1.1725650487651809,
        -0.50125823234183198,
        1.0720589011059332,
        -0.65380750835631218,
        -0.75001644171822424,
        1.9952797465745435,
        -1.0425931743766554,
        -0.2020877460509567,
        -2.2266468222619338,
        -0.65649899985850979,
        -0.7787255821432435,
        0.13765243691589113,
        -0.22141562904247536,
        0.32277811524025424,
        -174.35802897925396,
        -57.820366103495246,
        -219.65739239143858,
        239.40467858032952,
        -0.95197664279901328,
        -63.843789659520169,
        -2.0464139800485821,
        145.36755916201258,
        -197.88856749030998,
        93.726483285771536,
        224.98269453435708,
        -152.27059072947495,
        -83.299751012172237,
        69.564238892041629,
        -442.41038773865193,
        118.62437408308676,
        106.05255193510435,
        -55.721496878437023,
        78.665079700597261,
        35.81466573753405,
        83.74479156094327,
        218.8001854068433,
        237.41761663445362,
        -85.696951707059014,
        9.1766986168516276,
        60.179599187874807,
        -148.4909730199283,
        132.08961217901424,
        184.6922163343317,
        -74.317423812548128,
        64.572911291292598,
        -265.94310737978356,
        36.157045406379353,
        -41.429732957381404,
        26.981259984848691,
        -32.362833280856144,
        6.4219387107464998,
        -49.174151797535689,
        -27.606740716439553,
        -44.052323634273812,
        -20.418863862018064,
        39.718035781372834,
        -1.6536879246850091,
        -20.1312289602198,
        34.842252926855622,
        -18.404685415836958,
        -30.456393204232747,
        36.760699749931831,
        23.325122913880467,
        37.55548543649509,
        12.472302723509978,
        7.4630473196337368,
        -22.068984789085889,
        5.2440635491963867,
        -11.704101921106961,
        10.605451012872155,
        -8.1964738960697687,
        -9.9520816013380013,
        -41.138248897535689,
        10.306641750979669,
        9.5756962210793315,
        -12.414183675195487,
        -4.7244940074583681,
        20.070823172732418,
    ];

    // Builds the deterministic `(spectrum, aperiodic_ratio)` pair used to
    // generate the C++ reference vectors (scratch/aperiodic_ref/driver.cpp).
    fn phase33_test_vectors(fft_size: usize) -> (Vec<f64>, Vec<f64>) {
        let half = fft_size / 2;
        let mut spectrum = vec![0.0; half + 1];
        let mut aperiodic_ratio = vec![0.0; half + 1];
        for i in 0..=half {
            spectrum[i] = 1.0 + 0.5 * (0.3 * i as f64).cos() + 0.25 * ((i % 7) as f64);
            aperiodic_ratio[i] = 0.2 + 0.05 * ((i % 5) as f64);
        }
        (spectrum, aperiodic_ratio)
    }

    // The zero-mean noise has a ~0 DC bin (sum of the noise samples == 0), so
    // the DC bin of the forward spectrum is ~0 (PHASE3-3 acceptance).
    #[test]
    fn test_noise_spectrum_zero_mean() {
        let fft_size = 128;
        let noise_size = 32;
        let mut forward = ForwardRealFFT::new(fft_size);
        let mut state = RandnState::default();
        randn_reseed(&mut state);
        get_noise_spectrum(noise_size, fft_size, &mut forward, &mut state);
        assert!(
            forward.spectrum[0].re.abs() < 1e-6,
            "DC re={} (expected ~0)",
            forward.spectrum[0].re
        );
        assert!(
            forward.spectrum[0].im.abs() < 1e-6,
            "DC im={} (expected ~0)",
            forward.spectrum[0].im
        );
    }

    // The noise spectrum is deterministic for a reseeded PRNG: two calls with the
    // same seed produce identical spectra (PHASE3-3 acceptance).
    #[test]
    fn test_noise_spectrum_deterministic() {
        let fft_size = 128;
        let noise_size = 32;
        let mut a = ForwardRealFFT::new(fft_size);
        let mut b = ForwardRealFFT::new(fft_size);
        let mut sa = RandnState::default();
        let mut sb = RandnState::default();
        randn_reseed(&mut sa);
        randn_reseed(&mut sb);
        get_noise_spectrum(noise_size, fft_size, &mut a, &mut sa);
        get_noise_spectrum(noise_size, fft_size, &mut b, &mut sb);
        for i in 0..=fft_size / 2 {
            assert_eq!(a.spectrum[i], b.spectrum[i], "bin {i} differs");
        }
    }

    // The aperiodic response matches the C++ reference for a fixed random seed,
    // for both a voiced frame (current_vuv != 0) and an unvoiced frame
    // (current_vuv == 0). Values reach ~440 in magnitude; the observed max
    // deviation from the C++ reference is ~6e-14, so a 1e-9 tolerance is used.
    #[test]
    fn test_aperiodic_response_matches_cpp_reference() {
        let fft_size = 128;
        let noise_size = 32;
        let (spectrum, aperiodic_ratio) = phase33_test_vectors(fft_size);

        let mut forward = ForwardRealFFT::new(fft_size);
        let mut inverse = InverseRealFFT::new(fft_size);
        let mut mp = initialize_minimum_phase_analysis(fft_size);
        let mut out = vec![0.0; fft_size];

        let mut state = RandnState::default();
        randn_reseed(&mut state);
        get_aperiodic_response(
            noise_size,
            fft_size,
            &spectrum,
            &aperiodic_ratio,
            1.0,
            &mut forward,
            &mut inverse,
            &mut mp,
            &mut out,
            &mut state,
        );
        let max_voiced = out
            .iter()
            .zip(VUVED.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        for i in 0..fft_size {
            assert!(
                (out[i] - VUVED[i]).abs() < 1e-9,
                "voiced i={i}: {} vs {} (max diff {})",
                out[i],
                VUVED[i],
                max_voiced
            );
        }

        randn_reseed(&mut state);
        get_aperiodic_response(
            noise_size,
            fft_size,
            &spectrum,
            &aperiodic_ratio,
            0.0,
            &mut forward,
            &mut inverse,
            &mut mp,
            &mut out,
            &mut state,
        );
        let max_unvoiced = out
            .iter()
            .zip(UNVOICED.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        for i in 0..fft_size {
            assert!(
                (out[i] - UNVOICED[i]).abs() < 1e-9,
                "unvoiced i={i}: {} vs {} (max diff {})",
                out[i],
                UNVOICED[i],
                max_unvoiced
            );
        }
    }

    // The forward r2c and inverse c2r FFTs round-trip: an inverse of a forward
    // returns N times the zero-padded input (Route A is unnormalised, matching
    // the C++ `*2.0` convention folded into the c2r plan).
    // NOTE: the expected values are snapshotted BEFORE forward() — realfft
    // overwrites its input buffers (waveform at N>=2048, spectrum always),
    // so reading them back post-transform is garbage at large sizes even
    // though N=64 happens to be preserved. Never rely on post-call inputs.
    #[test]
    fn test_fft_forward_inverse_roundtrip() {
        let fft_size = 64;
        let mut forward = ForwardRealFFT::new(fft_size);
        let mut inverse = InverseRealFFT::new(fft_size);
        for i in 0..fft_size {
            forward.waveform[i] = (0.1 * i as f64).sin();
        }
        let expected: Vec<f64> = forward.waveform.clone();
        forward.forward();
        inverse.spectrum.copy_from_slice(&forward.spectrum);
        inverse.inverse();
        for (i, expect) in expected.iter().enumerate().take(fft_size) {
            assert!(
                (inverse.waveform[i] - fft_size as f64 * expect).abs() < 1e-9,
                "i={i}: {} vs {}",
                inverse.waveform[i],
                fft_size as f64 * expect
            );
        }
    }

    // PHASE3-4: fractional time shift, DC removal, and periodic response.

    // A unit-magnitude real bin rotated by `coefficient * i` radians becomes
    // `(cos, -sin)` while `coefficient * i` stays within `[0, pi]` (where the
    // `sqrt(1 - re2^2)` branch equals the true sine). This verifies the phase
    // rotation of the fractional time shift (PHASE3-4 acceptance).
    #[test]
    fn test_fractional_time_shift_phase_rotation() {
        let fft_size = 64;
        let half = fft_size / 2;
        // 0.02 * 32 = 0.64 < pi, so the sqrt branch matches the true sine.
        let coefficient = 0.02;
        let mut inverse = InverseRealFFT::new(fft_size);
        for i in 0..=half {
            inverse.spectrum[i] = Complex::new(1.0, 0.0);
        }
        get_spectrum_with_fractional_time_shift(fft_size, coefficient, &mut inverse);
        for i in 0..=half {
            let theta = coefficient * i as f64;
            assert!(
                (inverse.spectrum[i].re - theta.cos()).abs() < 1e-12,
                "re i={i}: {} vs {}",
                inverse.spectrum[i].re,
                theta.cos()
            );
            assert!(
                (inverse.spectrum[i].im - (-theta.sin())).abs() < 1e-12,
                "im i={i}: {} vs {}",
                inverse.spectrum[i].im,
                -theta.sin()
            );
        }
    }

    // The DC remover overwrites the lower half and subtracts from the upper half;
    // because the window sums to 1.0 the total sum of the result is ~0
    // (PHASE3-4 acceptance: DC component removal).
    #[test]
    fn test_remove_dc_component() {
        let fft_size = 8;
        let half = fft_size / 2;
        let dc_remover = get_dc_remover(fft_size);
        let upper: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
        let mut pr = vec![999.0; fft_size]; // distinctive lower-half garbage
        pr[half..half + half].copy_from_slice(&upper);
        let dc: f64 = upper.iter().sum(); // 10.0
        remove_dc_component(&mut pr, fft_size, &dc_remover);
        for i in 0..half {
            assert!(
                (pr[i] - (-dc * dc_remover[i])).abs() < 1e-12,
                "lower i={i}: {} vs {}",
                pr[i],
                -dc * dc_remover[i]
            );
        }
        for i in 0..half {
            assert!(
                (pr[half + i] - (upper[i] - dc * dc_remover[half + i])).abs() < 1e-12,
                "upper i={i}: {} vs {}",
                pr[half + i],
                upper[i] - dc * dc_remover[half + i]
            );
        }
        let sum: f64 = pr.iter().sum();
        assert!(sum.abs() < 1e-9, "sum={sum}");
    }

    // The periodic response is zero-filled for unvoiced frames (current_vuv <=
    // 0.5) and for fully aperiodic frames (aperiodic_ratio[0] > 0.999)
    // (PHASE3-4 acceptance: early return).
    #[test]
    fn test_periodic_response_zero_for_unvoiced_and_aperiodic() {
        let fft_size = 128;
        let (spectrum, aperiodic_ratio) = phase33_test_vectors(fft_size);
        let mut inverse = InverseRealFFT::new(fft_size);
        let mut mp = initialize_minimum_phase_analysis(fft_size);
        let dc_remover = get_dc_remover(fft_size);
        let mut out = vec![1.0; fft_size]; // pre-fill with a non-zero sentinel

        // Unvoiced frame: current_vuv = 0.0 <= 0.5.
        get_periodic_response(
            fft_size,
            &spectrum,
            &aperiodic_ratio,
            0.0,
            &mut inverse,
            &mut mp,
            &dc_remover,
            0.0,
            16000.0,
            &mut out,
        );
        for &v in &out {
            assert_eq!(v, 0.0);
        }

        // Fully aperiodic frame: aperiodic_ratio[0] > 0.999 (voiced).
        let mut ap_high = aperiodic_ratio.clone();
        ap_high[0] = 0.9999;
        get_periodic_response(
            fft_size,
            &spectrum,
            &ap_high,
            1.0,
            &mut inverse,
            &mut mp,
            &dc_remover,
            0.0,
            16000.0,
            &mut out,
        );
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    // A voiced periodic response is non-zero and has its DC removed (total sum
    // ~0) after the minimum-phase, fractional-time-shift, and DC-removal stages
    // (PHASE3-4 acceptance).
    #[test]
    fn test_periodic_response_voiced_nonzero_and_dc_removed() {
        let fft_size = 128;
        let (spectrum, aperiodic_ratio) = phase33_test_vectors(fft_size);
        let mut inverse = InverseRealFFT::new(fft_size);
        let mut mp = initialize_minimum_phase_analysis(fft_size);
        let dc_remover = get_dc_remover(fft_size);
        let mut out = vec![0.0; fft_size];
        get_periodic_response(
            fft_size,
            &spectrum,
            &aperiodic_ratio,
            1.0,
            &mut inverse,
            &mut mp,
            &dc_remover,
            0.3,
            16000.0,
            &mut out,
        );
        let max: f64 = out.iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(
            max > 0.0,
            "periodic response should be non-zero for a voiced frame"
        );
        let sum: f64 = out.iter().sum();
        assert!(sum.abs() < 1e-5, "DC should be removed, sum={sum}");
    }

    // C++ reference periodic response (scratch/periodic_ref, fft_size=128,
    // spectrum[i]=1+0.5cos(0.3i)+0.25(i%7), aperiodic_ratio[i]=0.2+0.05(i%5),
    // current_vuv=1.0, fs=16000).
    const PERIODIC_SHIFT_0P3: [f64; 128] = [
        -1.3302536004747838e-05,
        -5.3178591873272898e-05,
        -0.00011953358600662655,
        -0.00021221013168973629,
        -0.00033098841039529744,
        -0.000475586693169039,
        -0.00064566200885967946,
        -0.00084081095760860146,
        -0.0010605706676697492,
        -0.0013044198932902725,
        -0.001571780251047862,
        -0.0018620175917123318,
        -0.0021744435043775227,
        -0.002508316949295921,
        -0.0028628460155430717,
        -0.0032371897993427883,
        -0.0036304603985980177,
        -0.0040417250188965071,
        -0.0044700081859961006,
        -0.0049142940595418719,
        -0.0053735288425271997,
        -0.0058466232807838584,
        -0.0063324552465725591,
        -0.006829872400145964,
        -0.0073376949229712616,
        -0.0078547183161293815,
        -0.0083797162572533425,
        -0.0089114435092294414,
        -0.0094486388737620945,
        -0.0099900281827968486,
        -0.010534327320706265,
        -0.011080245270070283,
        -0.011626487173826925,
        -0.01217175740653019,
        -0.012714762647430449,
        -0.013254214948088403,
        -0.013788834787246474,
        -0.014317354105711859,
        -0.014838519314052839,
        -0.015351094265974417,
        -0.015853863190320758,
        -0.016345633574750069,
        -0.016825238994242056,
        -0.017291541877729303,
        -0.01774343620629008,
        -0.018179850136503077,
        -0.018599748542741552,
        -0.019002135472376964,
        -0.019386056508068603,
        -0.019750601031536037,
        -0.020094904383445071,
        -0.020418149914284183,
        -0.02071957092136692,
        -0.020998452467365973,
        -0.021254133076065523,
        -0.021486006301309746,
        -0.021693522165426107,
        -0.021876188463711609,
        -0.022033571931887985,
        -0.022165299273756722,
        -0.022271058046616349,
        -0.022350597402342052,
        -0.02240372868236969,
        -0.022430325865173063,
        0.33086572641649065,
        0.02276844893910655,
        -0.28048741440196578,
        -0.31367024404123445,
        0.51368261915041502,
        -0.61432535470948357,
        0.91566918989122514,
        0.41391879627584294,
        1.2016899747233794,
        1.935746668715191,
        -1.4166755162752767,
        -0.082887670125974228,
        -0.34509555154511079,
        -0.049759832104033556,
        -0.68106297581925668,
        -0.27647707500179791,
        -0.73928022335177224,
        0.30458930940035472,
        0.078110076965368258,
        -0.031824524277759322,
        -0.0034720499721170341,
        -0.37863403401637141,
        -0.043315281418935828,
        -0.20617424667358106,
        -0.22939178292044304,
        0.38988851529440488,
        -0.076521002380264769,
        1.0835340893987511,
        0.12553105767379638,
        -0.24139908116722927,
        0.017344813983884558,
        -0.091161835476489109,
        0.084153880196642106,
        -0.17680352681071412,
        0.050610888196289311,
        -0.16776946388806321,
        0.082424237829280558,
        -0.11670526475296197,
        0.1382331975792267,
        -0.12973794583732906,
        -0.16956208616030896,
        -0.060438515011410184,
        -0.42388075466480529,
        0.078836589337667434,
        -0.05407761168086396,
        0.12439532809398893,
        0.41551975291540305,
        -0.080248719431434939,
        0.059700393629292359,
        -0.043452931218429941,
        0.087567404457277995,
        -0.046111426310390072,
        0.03665004318503947,
        -0.073247269912996252,
        -0.00054616370133936019,
        -0.016604568134499856,
        -0.0059700889075018646,
        0.045243889276914639,
        0.020033882576464001,
        -0.014207850336579111,
        0.067986949392938784,
        -0.099712797403439343,
        -0.015378401200427841,
        -0.10513936569963471,
    ];

    const PERIODIC_SHIFT_0: [f64; 128] = [
        -0.0012757640246035839,
        -0.0051000301270963631,
        -0.011463727570794365,
        -0.020351762368433179,
        -0.031743053083437354,
        -0.045610580832759352,
        -0.06192145337268521,
        -0.080636983115602259,
        -0.10171277889268282,
        -0.12509885124483228,
        -0.15073973099216301,
        -0.17857460080076112,
        -0.20853743943468273,
        -0.2405571783510318,
        -0.27455787026669254,
        -0.31045886929689309,
        -0.34817502223833557,
        -0.38761687054318522,
        -0.42869086250486343,
        -0.47129957515235954,
        -0.51534194532675404,
        -0.56071350939186682,
        -0.607306651010461,
        -0.65501085639830392,
        -0.70371297645065456,
        -0.75329749511943955,
        -0.8036468034045553,
        -0.85464147830942438,
        -0.90616056609914819,
        -0.9580818691894023,
        -1.0102822359856087,
        -1.0626378529849085,
        -1.1150245384481146,
        -1.1673180369450755,
        -1.2193943140748276,
        -1.2711298506614923,
        -1.3224019357281134,
        -1.3730889575535372,
        -1.4230706921219796,
        -1.4722285882811081,
        -1.5204460489322782,
        -1.5676087075859715,
        -1.6136046996264672,
        -1.6583249276423599,
        -1.7016633201935598,
        -1.7435170834010327,
        -1.7837869447625228,
        -1.8223773886159604,
        -1.8591968826920622,
        -1.8941580952187576,
        -1.9271781020625052,
        -1.9581785834151768,
        -1.987086009559988,
        -2.0138318152758634,
        -2.038352562466565,
        -2.0605900906288497,
        -2.0804916548027554,
        -2.0980100506768182,
        -2.1131037265514863,
        -2.1257368818951599,
        -2.135879552259095,
        -2.1435076803497677,
        -2.1486031730901147,
        -2.1511539445343142,
        134.20296388841297,
        -3.011840525954161,
        -1.9791421036759493,
        -3.3017681511991674,
        -1.4701037684834533,
        -4.7289015489669097,
        18.377916618976666,
        0.54371381283401909,
        -3.969043456005056,
        -1.3761088809736817,
        -3.2822001213279313,
        -1.1623649598944961,
        -0.2113147083610345,
        -4.1363763925432835,
        -4.0550172694366493,
        -3.7534935757438275,
        -5.0046911154250289,
        -6.6211339301071837,
        -22.74577972022858,
        8.4424090832891689,
        1.2446054168864664,
        1.8577125963147612,
        -0.4922589644832378,
        1.5476123326425573,
        -0.98298758689899524,
        7.2601516467492777,
        -1.5864760180739472,
        -3.3318141957005514,
        -1.2705344903894153,
        -1.6225144167789507,
        0.02488906222660936,
        0.92326828432172015,
        -2.4645766827724431,
        -0.89129203457513051,
        -3.2855136643049452,
        0.3898552117246622,
        -11.34355921520109,
        -6.2590049919196593,
        3.4120213433400197,
        -1.9045706928200223,
        1.144673899620541,
        -1.4013534704787975,
        0.39231949239392849,
        -1.5910371350953372,
        1.4019405899652146,
        0.1007412731002072,
        0.51197281656120674,
        -0.23936387329216829,
        0.59888552255275007,
        2.0103530269280263,
        0.21699554310156538,
        5.2795517596380988,
        -0.33327315848696815,
        0.054956258403083935,
        -1.2355642810441578,
        -10.858851695333565,
        1.6388716446949982,
        -0.47382441968878136,
        0.99044237912650579,
        -1.0021503252885209,
        0.68872083960605657,
        -1.2233401915643793,
        0.6430148883106247,
        0.087584584733762127,
    ];

    // The voiced periodic response matches the C++ reference within 1e-9 for
    // both a non-zero fractional time shift (0.3 s) and no shift (0.0 s)
    // (PHASE3-4 acceptance: output matches reference within 1e-9).
    #[test]
    fn test_periodic_response_matches_cpp_reference() {
        let fft_size = 128;
        let (spectrum, aperiodic_ratio) = phase33_test_vectors(fft_size);
        let mut inverse = InverseRealFFT::new(fft_size);
        let mut mp = initialize_minimum_phase_analysis(fft_size);
        let dc_remover = get_dc_remover(fft_size);
        let mut out = vec![0.0; fft_size];

        get_periodic_response(
            fft_size,
            &spectrum,
            &aperiodic_ratio,
            1.0,
            &mut inverse,
            &mut mp,
            &dc_remover,
            0.3,
            16000.0,
            &mut out,
        );
        let max_shifted = out
            .iter()
            .zip(PERIODIC_SHIFT_0P3.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        for i in 0..fft_size {
            assert!(
                (out[i] - PERIODIC_SHIFT_0P3[i]).abs() < 1e-9,
                "shift 0.3 i={i}: {} vs {} (max diff {})",
                out[i],
                PERIODIC_SHIFT_0P3[i],
                max_shifted
            );
        }

        get_periodic_response(
            fft_size,
            &spectrum,
            &aperiodic_ratio,
            1.0,
            &mut inverse,
            &mut mp,
            &dc_remover,
            0.0,
            16000.0,
            &mut out,
        );
        let max_unshifted = out
            .iter()
            .zip(PERIODIC_SHIFT_0.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        for i in 0..fft_size {
            assert!(
                (out[i] - PERIODIC_SHIFT_0[i]).abs() < 1e-9,
                "shift 0 i={i}: {} vs {} (max diff {})",
                out[i],
                PERIODIC_SHIFT_0[i],
                max_unshifted
            );
        }
    }

    // PHASE3-5: spectral-envelope and aperiodic-ratio interpolation.

    // Flattens a 2D row-major matrix into a contiguous buffer, returning the
    // buffer and its row stride (used to exercise the flat-buffer + stride
    // representation of the interpolation helpers).
    fn flat_matrix(rows: Vec<Vec<f64>>) -> (Vec<f64>, usize) {
        let stride = rows[0].len();
        let flat = rows.into_iter().flatten().collect();
        (flat, stride)
    }

    // A constant spectrogram (alternating sign per row to exercise `fabs`)
    // interpolates to the same magnitude at every time, including between
    // frames and past the end of the contour (PHASE3-5: constant matrix + fabs).
    #[test]
    fn test_spectral_envelope_constant_matrix() {
        let f0_length = 5;
        let fft_size = 8;
        let half = fft_size / 2;
        // Each row: `half + 1` bins of +/-2.0 (alternating sign per row) plus one
        // padding bin, so the stride is wider than the bin count.
        let rows: Vec<Vec<f64>> = (0..f0_length)
            .map(|r| {
                let sign = if r % 2 == 0 { 2.0 } else { -2.0 };
                let mut row = vec![sign; half + 1];
                row.push(0.0);
                row
            })
            .collect();
        let (flat, stride) = flat_matrix(rows);
        let frame_period = 0.005;
        // Times spanning the start, a mid-frame point, a frame boundary, and
        // well past the end of the contour (ratio 100 -> clamped to last row).
        for t in [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.5] {
            let mut out = vec![0.0; half + 1];
            get_spectral_envelope(
                t,
                frame_period,
                f0_length,
                &flat,
                stride,
                fft_size,
                &mut out,
            );
            for (i, &v) in out.iter().enumerate() {
                assert!(
                    (v - 2.0).abs() < 1e-12,
                    "t={t} i={i}: {} (expected 2.0 via fabs)",
                    v
                );
            }
        }
    }

    // A spectrogram whose bins vary linearly with the row index interpolates to
    // the weighted average of the floor and ceil rows (PHASE3-5: linear ramp).
    #[test]
    fn test_spectral_envelope_linear_ramp() {
        let f0_length = 5;
        let fft_size = 8;
        let half = fft_size / 2;
        // row i, bin j = i + 0.1*j (linear in the row).
        let rows: Vec<Vec<f64>> = (0..f0_length)
            .map(|i| (0..=half).map(|j| i as f64 + 0.1 * j as f64).collect())
            .collect();
        let (flat, stride) = flat_matrix(rows);
        let frame_period = 0.005;
        // t = 0.0075 -> ratio 1.5 -> between row 1 and row 2, weight 0.5.
        let mut out = vec![0.0; half + 1];
        get_spectral_envelope(
            0.0075,
            frame_period,
            f0_length,
            &flat,
            stride,
            fft_size,
            &mut out,
        );
        for (j, &v) in out.iter().enumerate() {
            let expected = 0.5 * (1.0 + 0.1 * j as f64) + 0.5 * (2.0 + 0.1 * j as f64);
            assert!((v - expected).abs() < 1e-12, "j={j}: {} vs {}", v, expected);
        }
    }

    // Exactly on a frame boundary (t = k * frame_period) the floor and ceil
    // frames coincide, so the output is the floor row copied verbatim
    // (PHASE3-5: floor == ceil fast path).
    #[test]
    fn test_spectral_envelope_frame_boundary_exact() {
        let f0_length = 5;
        let fft_size = 8;
        let half = fft_size / 2;
        // Distinct rows: row i, bin j = 10*i + j.
        let rows: Vec<Vec<f64>> = (0..f0_length)
            .map(|i| (0..=half).map(|j| 10.0 * i as f64 + j as f64).collect())
            .collect();
        let (flat, stride) = flat_matrix(rows);
        let frame_period = 0.005;
        for k in 0..f0_length {
            let t = k as f64 * frame_period;
            let mut out = vec![0.0; half + 1];
            get_spectral_envelope(
                t,
                frame_period,
                f0_length,
                &flat,
                stride,
                fft_size,
                &mut out,
            );
            for (j, &v) in out.iter().enumerate() {
                let expected = (10.0 * k as f64 + j as f64).abs();
                assert!(
                    (v - expected).abs() < 1e-12,
                    "k={k} j={j}: {} vs {}",
                    v,
                    expected
                );
            }
        }
    }

    // A time past the end of the contour clamps both floor and ceil to the last
    // frame (PHASE3-5: MyMinInt end-of-contour edge case).
    #[test]
    fn test_spectral_envelope_clamps_to_last_frame() {
        let f0_length = 5;
        let fft_size = 8;
        let half = fft_size / 2;
        let rows: Vec<Vec<f64>> = (0..f0_length)
            .map(|i| (0..=half).map(|j| 10.0 * i as f64 + j as f64).collect())
            .collect();
        let (flat, stride) = flat_matrix(rows);
        let frame_period = 0.005;
        // t = 0.5 -> ratio 100 -> floor/ceil clamped to f0_length - 1 = 4.
        let mut out = vec![0.0; half + 1];
        get_spectral_envelope(
            0.5,
            frame_period,
            f0_length,
            &flat,
            stride,
            fft_size,
            &mut out,
        );
        for (j, &v) in out.iter().enumerate() {
            let expected = (10.0 * 4.0 + j as f64).abs(); // last row
            assert!((v - expected).abs() < 1e-12, "j={j}: {} vs {}", v, expected);
        }
    }

    // A constant-AP = 0.5 matrix (the D4C-deferred default) interpolates to the
    // squared safe aperiodicity 0.25 at every time (PHASE3-5: constant-AP +
    // squared safe aperiodicity).
    #[test]
    fn test_aperiodic_ratio_constant_0p5() {
        let f0_length = 5;
        let fft_size = 8;
        let half = fft_size / 2;
        let stride = half + 2;
        let flat = vec![0.5; f0_length * stride];
        let frame_period = 0.005;
        for t in [0.0, 0.0025, 0.0075, 0.01, 0.5] {
            let mut out = vec![0.0; half + 1];
            get_aperiodic_ratio(
                t,
                frame_period,
                f0_length,
                &flat,
                stride,
                fft_size,
                &mut out,
            );
            for (i, &v) in out.iter().enumerate() {
                assert!(
                    (v - 0.25).abs() < 1e-12,
                    "t={t} i={i}: {} (expected 0.25 = 0.5^2)",
                    v
                );
            }
        }
    }

    // Aperiodicity bins outside the safe range are clamped by
    // `get_safe_aperiodicity` before being squared (PHASE3-5: safe clamping).
    #[test]
    fn test_aperiodic_ratio_clamps_and_squares() {
        let f0_length = 2;
        let fft_size = 4;
        let half = fft_size / 2; // 2 -> 3 bins
                                 // row 0 and row 1: [0.0, 0.5, 1.0] -> safe [0.001, 0.5, 0.999999999999].
        let (flat, stride) = flat_matrix(vec![vec![0.0, 0.5, 1.0], vec![0.0, 0.5, 1.0]]);
        let frame_period = 0.005;
        // t = 0 -> floor == ceil == 0.
        let mut out = vec![0.0; half + 1];
        get_aperiodic_ratio(
            0.0,
            frame_period,
            f0_length,
            &flat,
            stride,
            fft_size,
            &mut out,
        );
        let expected = [0.001 * 0.001, 0.5 * 0.5, 0.999999999999 * 0.999999999999];
        for (i, &v) in out.iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-12,
                "i={i}: {} vs {}",
                v,
                expected[i]
            );
        }
    }

    // A per-row linear ramp of aperiodicity interpolates to the squared
    // weighted average of the safe floor/ceil values (PHASE3-5: linear ramp).
    #[test]
    fn test_aperiodic_ratio_linear_ramp() {
        let f0_length = 5;
        let fft_size = 8;
        let half = fft_size / 2;
        // row i, bin j = 0.1 + 0.1*i + 0.01*j (within the safe range here).
        let rows: Vec<Vec<f64>> = (0..f0_length)
            .map(|i| {
                (0..=half)
                    .map(|j| 0.1 + 0.1 * i as f64 + 0.01 * j as f64)
                    .collect()
            })
            .collect();
        let (flat, stride) = flat_matrix(rows);
        let frame_period = 0.005;
        // t = 0.0075 -> ratio 1.5 -> between row 1 and row 2, weight 0.5.
        let mut out = vec![0.0; half + 1];
        get_aperiodic_ratio(
            0.0075,
            frame_period,
            f0_length,
            &flat,
            stride,
            fft_size,
            &mut out,
        );
        for (j, &v) in out.iter().enumerate() {
            let a1 = get_safe_aperiodicity(0.1 + 0.1 * 1.0 + 0.01 * j as f64);
            let a2 = get_safe_aperiodicity(0.1 + 0.1 * 2.0 + 0.01 * j as f64);
            let blended = 0.5 * a1 + 0.5 * a2;
            let expected = blended * blended;
            assert!((v - expected).abs() < 1e-12, "j={j}: {} vs {}", v, expected);
        }
    }

    // PHASE3-6: one-frame segment synthesis.

    // Builds the deterministic `(spectrogram, aperiodicity)` flat matrices used
    // to generate the C++ reference vectors (scratch/oneframe_ref/driver.cpp).
    // The spectrogram row `i`, bin `j` = `1.0 + 0.5*cos(0.3*j) + 0.25*(j%7) +
    // 0.05*i` (linear in the row so the between-frame interpolation is
    // exercised); the aperiodicity is the constant-AP default 0.5. Each matrix
    // is `f0_length` rows of `fft_size/2 + 1` bins (stride == bin count).
    fn phase36_test_matrices(
        f0_length: usize,
        fft_size: usize,
    ) -> ((Vec<f64>, usize), (Vec<f64>, usize)) {
        let half = fft_size / 2;
        let stride = half + 1;
        let mut spectrogram = Vec::with_capacity(f0_length * stride);
        for i in 0..f0_length {
            for j in 0..=half {
                spectrogram.push(
                    1.0 + 0.5 * (0.3 * j as f64).cos() + 0.25 * ((j % 7) as f64) + 0.05 * i as f64,
                );
            }
        }
        let aperiodicity = vec![0.5; f0_length * stride];
        ((spectrogram, stride), (aperiodicity, stride))
    }

    // The response is exactly `fft_size` samples long and every sample is
    // written (no leftover sentinel) for a voiced segment (PHASE3-6 acceptance:
    // response length equals fft_size).
    #[test]
    fn test_one_frame_segment_length() {
        let fft_size = 128;
        let f0_length = 5;
        let ((spectrogram, sp_stride), (aperiodicity, ap_stride)) =
            phase36_test_matrices(f0_length, fft_size);
        let mut forward = ForwardRealFFT::new(fft_size);
        let mut inverse = InverseRealFFT::new(fft_size);
        let mut mp = initialize_minimum_phase_analysis(fft_size);
        let dc_remover = get_dc_remover(fft_size);
        let sentinel = 999.0;
        let mut response = vec![sentinel; fft_size];
        let mut state = RandnState::default();
        randn_reseed(&mut state);
        let mut aperiodic_response = vec![0.0; fft_size];
        let mut periodic_response = vec![0.0; fft_size];
        let mut spectral_envelope = vec![0.0; fft_size];
        let mut aperiodic_ratio = vec![0.0; fft_size];
        get_one_frame_segment(
            1.0,
            32,
            &spectrogram,
            sp_stride,
            fft_size,
            &aperiodicity,
            ap_stride,
            f0_length,
            0.005,
            0.0075,
            0.3,
            16000.0,
            &mut forward,
            &mut inverse,
            &mut mp,
            &dc_remover,
            &mut response,
            &mut state,
            &mut aperiodic_response,
            &mut periodic_response,
            &mut spectral_envelope,
            &mut aperiodic_ratio,
        );
        assert_eq!(response.len(), fft_size);
        for &v in &response {
            assert_ne!(v, sentinel, "a sample was not written");
            assert!(v.is_finite());
        }
    }

    // C++ reference one-frame segments (scratch/oneframe_ref, fft_size=128,
    // f0_length=5, frame_period=0.005, current_time=0.0075, shift=0.3, fs=16000,
    // reseeded randn). The voiced case has noise_size=32; the unvoiced case has
    // noise_size=32 and current_vuv=0.0. Values reach ~3.5 in magnitude; the
    // Route A FFT is not bit-exact with Ooura, so a 1e-9 tolerance is used.
    const SEGMENT_VOICED: [f64; 128] = [
        -0.13757141216304564,
        0.0064124370813057182,
        0.04745802332510575,
        0.08551822566104994,
        0.058719571767367904,
        0.10085997439763045,
        -0.037930467667964395,
        0.0017606692702879614,
        0.002978943894731721,
        -0.027470144802867247,
        -0.0027389265121885581,
        -0.053529917804379094,
        -0.058804765882273159,
        -0.052460468118419958,
        0.034750875788272961,
        -0.0080916025056030176,
        -0.011765545853591918,
        0.047016919928350311,
        -0.043167505203971356,
        -0.027772729721359954,
        0.010063057863464118,
        -0.0098755684902929099,
        0.080747645472733179,
        -0.031839479601703476,
        0.027500902196894172,
        -0.0078595777439188358,
        0.018564448886603965,
        -0.0079006676076411701,
        0.014258267328392561,
        0.0035973353307021168,
        0.014949807333680929,
        -0.0062092033273540691,
        -0.015091051326172458,
        0.0026940088550786801,
        0.0011079731016175008,
        -0.017139173300374802,
        0.010758030277218887,
        -0.00016635512751310129,
        -0.012828744969300018,
        -0.01012606521033791,
        -0.015580436617815564,
        0.0011873584646312648,
        -0.0030169901227810013,
        0.0052869930957099055,
        -0.00069482320382813971,
        0.0064665139414263745,
        -0.0014724170935639382,
        0.0056566905535682431,
        0.0037481510671715963,
        0.0058003777312979603,
        -0.0054428387212819528,
        -0.0027428997622594275,
        0.0026742917394926144,
        -0.0035445958193396441,
        -0.0037350333394746884,
        0.0061576152425788597,
        -0.0049975393459282058,
        -0.0018974277082430503,
        -0.0091424600802715791,
        -0.0034059150092559639,
        -0.0038664742405905362,
        -0.00044322538817962668,
        -0.0018412670708222757,
        9.4031151715863242e-05,
        -0.67828842936979394,
        -0.23145391477621177,
        -0.88735163575938336,
        0.94918985357946362,
        0.024245887974377933,
        -0.25470159936935127,
        0.017778953391744689,
        0.58832054683374846,
        -0.73727619203935513,
        0.44313483591623104,
        0.83060262780216554,
        -0.61255687851214258,
        -0.35593088819277713,
        0.27072303150459054,
        -1.784539826265394,
        0.45959754963578531,
        0.41188465494739529,
        -0.21778481942093647,
        0.30680670675453386,
        0.13926022649306036,
        0.34359766466422964,
        0.87230128687128539,
        0.94098597995807143,
        -0.33462900089598852,
        0.034149392166233167,
        0.2528916454012396,
        -0.60367410037983615,
        0.56523980660845485,
        0.73991842716084233,
        -0.31649717958522361,
        0.25517534932200686,
        -1.0736770329908998,
        0.13601427390742904,
        -0.17004111937222124,
        0.099098797303599981,
        -0.12547204189417746,
        0.025739176235110638,
        -0.19195706801814422,
        -0.10200736836847123,
        -0.17221141992064082,
        -0.082029589643174292,
        0.15247502498538734,
        -0.0078703808072798764,
        -0.074775718821953094,
        0.13109161760626165,
        -0.066681356550365015,
        -0.097417997559731975,
        0.13649056140033078,
        0.091364882922258916,
        0.14012731735397591,
        0.048334310109565293,
        0.024500946575277367,
        -0.087295686425199526,
        0.016654158877531258,
        -0.043245211290867905,
        0.038237434862042841,
        -0.031322395168397764,
        -0.036960165415415525,
        -0.15676308435218178,
        0.038674004606242589,
        0.036601976797349008,
        -0.048348812448655715,
        -0.016417810661735702,
        0.073344389463910947,
    ];

    const SEGMENT_UNVOICED: [f64; 128] = [
        -0.27514149664646115,
        0.012830181732022994,
        0.094927976879697007,
        0.17105763127373286,
        0.11747217833127205,
        0.20176741544107321,
        -0.075796494067219569,
        0.0036052569438727033,
        0.0060637396414271461,
        -0.054810100014741292,
        -0.0053209791304209553,
        -0.10687399414012244,
        -0.11739250815740221,
        -0.10467058988794625,
        0.069787482232568809,
        -0.015860112410341132,
        -0.023168748142169848,
        0.094437230306219672,
        -0.085888874512236646,
        -0.055054980926620894,
        0.020662428861829008,
        -0.019167605943125721,
        0.16212731117359458,
        -0.062997293507504104,
        0.055734154081006841,
        -0.014935203502589672,
        0.037965247968249294,
        -0.014911915181448521,
        0.029459570282616032,
        0.0081917404621767442,
        0.030951009062926749,
        -0.011312526096436404,
        -0.029021703598610893,
        0.0066028382802611751,
        0.0034849622287928539,
        -0.032955489726287202,
        0.022892275966710472,
        0.001096254823406223,
        -0.024176509182039188,
        -0.01871999134945973,
        -0.029578554555102665,
        0.0040061174932712329,
        -0.0043547119407112977,
        0.012299794557482868,
        0.00038126395210533182,
        0.014747495191853088,
        -0.0010884582856199587,
        0.013209917841869978,
        0.009431156685960021,
        0.013571993929126613,
        -0.0088800752617164205,
        -0.0034479353368846513,
        0.0074165314439604557,
        -0.0049934094812052951,
        -0.0053487659334484766,
        0.014459673686943224,
        -0.0078299240568792117,
        -0.0016114694966409063,
        -0.016085826346747545,
        -0.0045995889590718653,
        -0.0055101520082169939,
        0.0013442842417338424,
        -0.0014464962761693567,
        0.0024267547407730988,
        -1.3960264296322544,
        -0.4633714448330053,
        -1.75888921224215,
        1.9146688551360695,
        -0.0071429993795381688,
        -0.51314160681825738,
        -0.0052972364641245134,
        1.1659557449941409,
        -1.5684088980262263,
        0.73761168594798654,
        1.7973150716992978,
        -1.212831823259108,
        -0.66466971771195638,
        0.54783402619757449,
        -3.532612560049154,
        0.9417714191193064,
        0.83292624752310573,
        -0.44167672532435787,
        0.62360497327693976,
        0.27746558669551602,
        0.6910657377415087,
        1.7680840597589702,
        1.8886856682199953,
        -0.68008424690759273,
        0.07429989526183628,
        0.49094652893371304,
        -1.199259871796432,
        1.0435564767622076,
        1.4725444482991801,
        -0.60588419217425915,
        0.51128735121157565,
        -2.1326316535025724,
        0.27274312750113727,
        -0.3215341318950371,
        0.20408601679213306,
        -0.2456059039195852,
        0.049812017305041832,
        -0.37714824687029247,
        -0.20830944671191981,
        -0.336848046072249,
        -0.15364290701992861,
        0.30544357436002395,
        -0.012523691458477704,
        -0.14964068725737145,
        0.26807325755170563,
        -0.14173157065337677,
        -0.23078039837724421,
        0.28211099300407627,
        0.17805999481757501,
        0.28592196790539515,
        0.093578139107762637,
        0.054040642503215004,
        -0.17169179803664622,
        0.039076516764237838,
        -0.088622314311248215,
        0.07947334792553995,
        -0.063759270961281533,
        -0.075639952532291582,
        -0.31464791359644656,
        0.080836727420459553,
        0.072738362669578532,
        -0.093860642607075509,
        -0.033050723869283899,
        0.15388841618699972,
    ];

    // The voiced and unvoiced one-frame segments match the C++ reference within
    // 1e-9 for a fixed random seed (PHASE3-6 acceptance: matches C++ reference).
    #[test]
    fn test_one_frame_segment_matches_cpp_reference() {
        let fft_size = 128;
        let f0_length = 5;
        let noise_size = 32;
        let ((spectrogram, sp_stride), (aperiodicity, ap_stride)) =
            phase36_test_matrices(f0_length, fft_size);
        let frame_period = 0.005;
        let current_time = 0.0075;
        let fractional_time_shift = 0.3;
        let fs = 16000.0;

        let mut forward = ForwardRealFFT::new(fft_size);
        let mut inverse = InverseRealFFT::new(fft_size);
        let mut mp = initialize_minimum_phase_analysis(fft_size);
        let dc_remover = get_dc_remover(fft_size);
        let mut response = vec![0.0; fft_size];

        let mut state = RandnState::default();
        randn_reseed(&mut state);
        let mut aperiodic_response = vec![0.0; fft_size];
        let mut periodic_response = vec![0.0; fft_size];
        let mut spectral_envelope = vec![0.0; fft_size];
        let mut aperiodic_ratio = vec![0.0; fft_size];
        get_one_frame_segment(
            1.0,
            noise_size,
            &spectrogram,
            sp_stride,
            fft_size,
            &aperiodicity,
            ap_stride,
            f0_length,
            frame_period,
            current_time,
            fractional_time_shift,
            fs,
            &mut forward,
            &mut inverse,
            &mut mp,
            &dc_remover,
            &mut response,
            &mut state,
            &mut aperiodic_response,
            &mut periodic_response,
            &mut spectral_envelope,
            &mut aperiodic_ratio,
        );
        let max_voiced = response
            .iter()
            .zip(SEGMENT_VOICED.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        for i in 0..fft_size {
            assert!(
                (response[i] - SEGMENT_VOICED[i]).abs() < 1e-9,
                "voiced i={i}: {} vs {} (max diff {})",
                response[i],
                SEGMENT_VOICED[i],
                max_voiced
            );
        }

        randn_reseed(&mut state);
        get_one_frame_segment(
            0.0,
            noise_size,
            &spectrogram,
            sp_stride,
            fft_size,
            &aperiodicity,
            ap_stride,
            f0_length,
            frame_period,
            current_time,
            fractional_time_shift,
            fs,
            &mut forward,
            &mut inverse,
            &mut mp,
            &dc_remover,
            &mut response,
            &mut state,
            &mut aperiodic_response,
            &mut periodic_response,
            &mut spectral_envelope,
            &mut aperiodic_ratio,
        );
        let max_unvoiced = response
            .iter()
            .zip(SEGMENT_UNVOICED.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        for i in 0..fft_size {
            assert!(
                (response[i] - SEGMENT_UNVOICED[i]).abs() < 1e-9,
                "unvoiced i={i}: {} vs {} (max diff {})",
                response[i],
                SEGMENT_UNVOICED[i],
                max_unvoiced
            );
        }
    }

    // With `noise_size = 0` (the final pulse) the periodic response is weighted
    // by `sqrt(0) = 0` and the aperiodic response is all-zero (zero noise
    // spectrum), so the whole segment is all-zero (PHASE3-6 acceptance: handles
    // the zero noise_size edge case).
    #[test]
    fn test_one_frame_segment_zero_noise() {
        let fft_size = 128;
        let f0_length = 5;
        let ((spectrogram, sp_stride), (aperiodicity, ap_stride)) =
            phase36_test_matrices(f0_length, fft_size);
        let mut forward = ForwardRealFFT::new(fft_size);
        let mut inverse = InverseRealFFT::new(fft_size);
        let mut mp = initialize_minimum_phase_analysis(fft_size);
        let dc_remover = get_dc_remover(fft_size);
        let mut response = vec![1.0; fft_size]; // pre-fill with a non-zero sentinel
        let mut state = RandnState::default();
        randn_reseed(&mut state);
        let mut aperiodic_response = vec![0.0; fft_size];
        let mut periodic_response = vec![0.0; fft_size];
        let mut spectral_envelope = vec![0.0; fft_size];
        let mut aperiodic_ratio = vec![0.0; fft_size];
        get_one_frame_segment(
            1.0,
            0,
            &spectrogram,
            sp_stride,
            fft_size,
            &aperiodicity,
            ap_stride,
            f0_length,
            0.005,
            0.0075,
            0.3,
            16000.0,
            &mut forward,
            &mut inverse,
            &mut mp,
            &dc_remover,
            &mut response,
            &mut state,
            &mut aperiodic_response,
            &mut periodic_response,
            &mut spectral_envelope,
            &mut aperiodic_ratio,
        );
        for &v in &response {
            assert_eq!(v, 0.0);
        }
    }

    // The segment equals `(periodic * sqrt(noise_size) + aperiodic) / fft_size`:
    // the periodic and aperiodic responses are computed independently (in the
    // same order as the function) and the weighted combination is compared to
    // the function's output (PHASE3-6 acceptance: correct sqrt_noise_size
    // weighting).
    #[test]
    fn test_one_frame_segment_sqrt_weighting() {
        let fft_size = 128;
        let f0_length = 5;
        let noise_size = 32;
        let ((spectrogram, sp_stride), (aperiodicity, ap_stride)) =
            phase36_test_matrices(f0_length, fft_size);
        let frame_period = 0.005;
        let current_time = 0.0075;
        let fractional_time_shift = 0.3;
        let fs = 16000.0;

        // Interpolate the envelope and aperiodic ratio once (shared inputs).
        let mut spectral_envelope = vec![0.0; fft_size];
        let mut aperiodic_ratio = vec![0.0; fft_size];
        get_spectral_envelope(
            current_time,
            frame_period,
            f0_length,
            &spectrogram,
            sp_stride,
            fft_size,
            &mut spectral_envelope,
        );
        get_aperiodic_ratio(
            current_time,
            frame_period,
            f0_length,
            &aperiodicity,
            ap_stride,
            fft_size,
            &mut aperiodic_ratio,
        );

        // Independent periodic response.
        let mut inverse = InverseRealFFT::new(fft_size);
        let mut mp = initialize_minimum_phase_analysis(fft_size);
        let dc_remover = get_dc_remover(fft_size);
        let mut periodic = vec![0.0; fft_size];
        get_periodic_response(
            fft_size,
            &spectral_envelope,
            &aperiodic_ratio,
            1.0,
            &mut inverse,
            &mut mp,
            &dc_remover,
            fractional_time_shift,
            fs,
            &mut periodic,
        );

        // Independent aperiodic response (reseeded, matching the function's draw
        // order: the periodic response consumes no randn draws).
        let mut forward = ForwardRealFFT::new(fft_size);
        let mut aperiodic = vec![0.0; fft_size];
        let mut state = RandnState::default();
        randn_reseed(&mut state);
        get_aperiodic_response(
            noise_size,
            fft_size,
            &spectral_envelope,
            &aperiodic_ratio,
            1.0,
            &mut forward,
            &mut inverse,
            &mut mp,
            &mut aperiodic,
            &mut state,
        );

        let sqrt_noise_size = (noise_size as f64).sqrt();
        let expected: Vec<f64> = (0..fft_size)
            .map(|i| (periodic[i] * sqrt_noise_size + aperiodic[i]) / fft_size as f64)
            .collect();

        // The actual segment (fresh FFT objects, reseeded PRNG).
        let mut inverse2 = InverseRealFFT::new(fft_size);
        let mut mp2 = initialize_minimum_phase_analysis(fft_size);
        let mut forward2 = ForwardRealFFT::new(fft_size);
        let mut response = vec![0.0; fft_size];
        let mut state2 = RandnState::default();
        randn_reseed(&mut state2);
        let mut aperiodic_response = vec![0.0; fft_size];
        let mut periodic_response = vec![0.0; fft_size];
        let mut spectral_envelope = vec![0.0; fft_size];
        let mut aperiodic_ratio = vec![0.0; fft_size];
        get_one_frame_segment(
            1.0,
            noise_size,
            &spectrogram,
            sp_stride,
            fft_size,
            &aperiodicity,
            ap_stride,
            f0_length,
            frame_period,
            current_time,
            fractional_time_shift,
            fs,
            &mut forward2,
            &mut inverse2,
            &mut mp2,
            &dc_remover,
            &mut response,
            &mut state2,
            &mut aperiodic_response,
            &mut periodic_response,
            &mut spectral_envelope,
            &mut aperiodic_ratio,
        );
        for i in 0..fft_size {
            assert!(
                (response[i] - expected[i]).abs() < 1e-9,
                "i={i}: {} vs {}",
                response[i],
                expected[i]
            );
        }
    }

    // ------------------------------------------------------------------
    // PHASE3-7: time base and pulse-location calculation.
    //
    // Reference vectors are generated by the C++ driver in scratch/timebase_ref
    // (replicating the static functions from synthesis.cpp and linking against
    // the world-cpp library for interp1/histc).

    const TIMEBASE_FS: f64 = 16000.0;
    const TIMEBASE_FRAME_PERIOD: f64 = 0.005; // seconds (5 ms)
    const TIMEBASE_LOWEST_F0: f64 = 16000.0 / 2048.0 + 1.0; // 8.8125

    // CONSTANT case (f0 = 200 Hz, f0_length = 201, y_length = 16001): the
    // first ten and the last of the 200 pulses as (index, location,
    // time_shift).
    const PULSES_CONSTANT_FIRST10: [(usize, f64, f64); 10] = [
        (78, 0.004875, 6.249999999999011e-05),
        (158, 0.009875, 6.249999999996184e-05),
        (238, 0.014875, 6.249999999993498e-05),
        (318, 0.019875, 6.24999999999067e-05),
        (398, 0.024875, 6.249999999987844e-05),
        (479, 0.0299375, 5.654319433713143e-17),
        (559, 0.0349375, 2.544443745170914e-16),
        (639, 0.0399375, 4.523455546970514e-16),
        (719, 0.0449375, 6.502467348770114e-16),
        (799, 0.0499375, 8.481479150569714e-16),
    ];
    const PULSES_CONSTANT_LAST: [(usize, f64, f64); 1] = [(15998, 0.999875, 6.249999960967102e-05)];
    // Interpolated V/UV at samples 0, 2000, ..., 16000.
    const VUV_CONSTANT: [f64; 9] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

    // MIXED case (f0 = [100,100,100,0,0,0,300,300,500,500,0,0,200 x9],
    // f0_length = 21, y_length = 1601): the coarse contours (f0_length + 1
    // positions, including the linearly extrapolated last one) and all 29
    // pulses as (index, location, time_shift).
    const COARSE_F0_MIXED: [f64; 22] = [
        100.0, 100.0, 100.0, 0.0, 0.0, 0.0, 300.0, 300.0, 500.0, 500.0, 0.0, 0.0, 200.0, 200.0,
        200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0,
    ];
    const COARSE_VUV_MIXED: [f64; 22] = [
        1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
    ];
    const PULSES_MIXED: [(usize, f64, f64); 29] = [
        (158, 0.009875, 6.249999999996184e-05),
        (224, 0.014, 5.937499999999672e-05),
        (256, 0.016, 5.937500000000351e-05),
        (288, 0.018, 5.9375000000010286e-05),
        (320, 0.02, 5.9375000000017076e-05),
        (352, 0.022, 5.937500000002273e-05),
        (384, 0.024, 5.937500000002952e-05),
        (416, 0.026, 5.93750000000363e-05),
        (462, 0.028875, 5.9854497354584607e-05),
        (518, 0.032375, 1.8847731445709336e-17),
        (570, 0.035625, 5.009541984740745e-05),
        (612, 0.03825, 5.130057803474305e-05),
        (646, 0.040375, 3.750000000004026e-05),
        (678, 0.042375, 3.7499999999974674e-05),
        (710, 0.044375, 3.749999999990908e-05),
        (747, 0.0466875, 3.124999999982602e-05),
        (784, 0.049, 2.187499999983557e-05),
        (816, 0.051, 2.1874999999769983e-05),
        (848, 0.053, 2.1874999999704392e-05),
        (880, 0.055, 2.18749999996388e-05),
        (912, 0.057, 2.1874999999660743e-05),
        (992, 0.062, 3.906249999927059e-05),
        (1072, 0.067, 3.90624999994685e-05),
        (1152, 0.072, 3.906249999966639e-05),
        (1232, 0.077, 3.90624999998643e-05),
        (1312, 0.082, 3.90625000000622e-05),
        (1392, 0.087, 3.9062500000260094e-05),
        (1472, 0.092, 3.90625000000458e-05),
        (1552, 0.097, 3.90625000000655896e-05),
    ];
    // Interpolated V/UV at samples 0, 100, ..., 1600.
    const VUV_MIXED: [f64; 17] = [
        1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ];

    // LOWF0 case (f0 = [5, 5, 100, 100, 100], f0_length = 5, y_length = 81):
    // values below lowest_f0 (8.8125 Hz) are treated as unvoiced.
    const COARSE_F0_LOWF0: [f64; 6] = [0.0, 0.0, 100.0, 100.0, 100.0, 100.0];
    const COARSE_VUV_LOWF0: [f64; 6] = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    const PULSES_LOWF0: [(usize, f64, f64); 2] = [
        (30, 0.001875, 6.249999999999858e-05),
        (63, 0.0039375, 2.827159716856469e-18),
    ];
    // Interpolated V/UV at samples 0, 10, ..., 80.
    const VUV_LOWF0: [f64; 9] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    // THRESH case (f0 = [0, 100], f0_length = 2, y_length = 81): coarse_vuv =
    // [0, 1, 1] ramps 0 -> 1 over t in [0, 0.005]; the single pulse is
    // (index, location, time_shift).
    const PULSES_THRESH: [(usize, f64, f64); 1] = [(30, 0.001875, 6.2499999999998578e-05)];

    fn check_pulses(
        expected: &[(usize, f64, f64)],
        pulse_locations: &[f64],
        pulse_locations_index: &[usize],
        pulse_locations_time_shift: &[f64],
    ) {
        assert_eq!(pulse_locations.len(), expected.len(), "pulse buffer length");
        for (i, &(idx, loc, shift)) in expected.iter().enumerate() {
            assert_eq!(pulse_locations_index[i], idx, "pulse {i} index");
            assert_eq!(pulse_locations[i], loc, "pulse {i} location");
            assert!(
                (pulse_locations_time_shift[i] - shift).abs() < 1e-9,
                "pulse {i} time shift: {} vs {shift}",
                pulse_locations_time_shift[i]
            );
        }
    }

    // interp1: linear interpolation between grid points, linear extrapolation
    // beyond them (k stays in [1, n-1] for any query point), and exact
    // reproduction of grid points.
    #[test]
    fn test_interp1() {
        let x = [0.0, 1.0, 2.0, 3.0];
        let y = [0.0, 10.0, 20.0, 30.0];
        let mut yi = vec![0.0; 6];
        interp1(&x, &y, &[-0.5, 0.0, 0.5, 1.5, 3.0, 3.5], &mut yi);
        assert_eq!(yi[0], -5.0); // extrapolation below the grid
        assert_eq!(yi[1], 0.0);
        assert_eq!(yi[2], 5.0);
        assert_eq!(yi[3], 15.0);
        assert_eq!(yi[4], 30.0);
        assert_eq!(yi[5], 35.0); // extrapolation above the grid
    }

    // get_temporal_parameters_for_time_base: the fine axis is i/fs, the coarse
    // axis is i*frame_period, the lowest-F0 guard clamps sub-threshold F0 to
    // 0.0 (with coarse_vuv 0.0 there), and the final coarse position
    // extrapolates the last two values (C++: synthesis.cpp:224-241).
    #[test]
    fn test_temporal_parameters_lowest_f0_guard() {
        let f0 = [5.0, 5.0, 100.0, 100.0, 100.0];
        let y_length = 81;
        let mut time_axis = vec![0.0; y_length];
        let mut coarse_time_axis = vec![0.0; 6];
        let mut coarse_f0 = vec![0.0; 6];
        let mut coarse_vuv = vec![0.0; 6];
        get_temporal_parameters_for_time_base(
            &f0,
            TIMEBASE_FS,
            y_length,
            TIMEBASE_FRAME_PERIOD,
            TIMEBASE_LOWEST_F0,
            &mut time_axis,
            &mut coarse_time_axis,
            &mut coarse_f0,
            &mut coarse_vuv,
        );
        for (i, &t) in time_axis.iter().enumerate() {
            assert_eq!(t, i as f64 / TIMEBASE_FS);
        }
        for (i, &t) in coarse_time_axis.iter().enumerate() {
            assert_eq!(t, i as f64 * TIMEBASE_FRAME_PERIOD);
        }
        assert_eq!(coarse_f0, COARSE_F0_LOWF0);
        assert_eq!(coarse_vuv, COARSE_VUV_LOWF0);
    }

    // get_pulse_locations_for_time_base: constant 200 Hz for 1 s at 16 kHz
    // gives exactly 200 pulses (PHASE3-7 acceptance criterion); the first ten
    // and the last pulse match the C++ reference (locations bit-exact,
    // fractional time shifts within 1e-9).
    #[test]
    fn test_pulse_locations_constant_f0() {
        let y_length = 16001;
        let interpolated_f0 = vec![200.0; y_length];
        let time_axis: Vec<f64> = (0..y_length).map(|i| i as f64 / TIMEBASE_FS).collect();
        let mut pulse_locations = vec![0.0; y_length];
        let mut pulse_locations_index = vec![0usize; y_length];
        let mut pulse_locations_time_shift = vec![0.0; y_length];
        let n = get_pulse_locations_for_time_base(
            &interpolated_f0,
            &time_axis,
            TIMEBASE_FS,
            &mut pulse_locations,
            &mut pulse_locations_index,
            &mut pulse_locations_time_shift,
        );
        assert_eq!(n, 200);
        check_pulses(
            &PULSES_CONSTANT_FIRST10,
            &pulse_locations[..10],
            &pulse_locations_index[..10],
            &pulse_locations_time_shift[..10],
        );
        check_pulses(
            &PULSES_CONSTANT_LAST,
            &pulse_locations[n - 1..n],
            &pulse_locations_index[n - 1..n],
            &pulse_locations_time_shift[n - 1..n],
        );
    }

    // get_pulse_locations_for_time_base: the C++ allocates a `y_length - 1`
    // buffer unconditionally (synthesis.cpp:248), so `y_length >= 2` is
    // required; shorter input yields zero pulses.
    #[test]
    fn test_pulse_locations_y_length_one() {
        let mut pulse_locations = vec![0.0; 1];
        let mut pulse_locations_index = vec![0usize; 1];
        let mut pulse_locations_time_shift = vec![0.0; 1];
        let n = get_pulse_locations_for_time_base(
            &[200.0],
            &[0.0],
            TIMEBASE_FS,
            &mut pulse_locations,
            &mut pulse_locations_index,
            &mut pulse_locations_time_shift,
        );
        assert_eq!(n, 0);
    }

    // get_time_base (CONSTANT): 200 Hz for 1 s at 16 kHz yields exactly 200
    // pulses (PHASE3-7 acceptance criterion) and the interpolated V/UV stays
    // 1.0 everywhere.
    #[test]
    fn test_get_time_base_constant() {
        let f0 = vec![200.0; 201];
        let y_length = 16001;
        let mut pulse_locations = vec![0.0; y_length];
        let mut pulse_locations_index = vec![0usize; y_length];
        let mut pulse_locations_time_shift = vec![0.0; y_length];
        let mut interpolated_vuv = vec![0.0; y_length];
        let n = get_time_base(
            &f0,
            TIMEBASE_FS,
            TIMEBASE_FRAME_PERIOD,
            y_length,
            TIMEBASE_LOWEST_F0,
            &mut pulse_locations,
            &mut pulse_locations_index,
            &mut pulse_locations_time_shift,
            &mut interpolated_vuv,
        );
        assert_eq!(n, 200);
        check_pulses(
            &PULSES_CONSTANT_FIRST10,
            &pulse_locations[..10],
            &pulse_locations_index[..10],
            &pulse_locations_time_shift[..10],
        );
        check_pulses(
            &PULSES_CONSTANT_LAST,
            &pulse_locations[n - 1..n],
            &pulse_locations_index[n - 1..n],
            &pulse_locations_time_shift[n - 1..n],
        );
        for (i, &expected) in (0..y_length).step_by(2000).zip(VUV_CONSTANT.iter()) {
            assert_eq!(interpolated_vuv[i], expected, "vuv at sample {i}");
        }
    }

    // get_time_base (MIXED): unvoiced gaps (f0 = 0), a 300 -> 500 Hz jump, and
    // a return to 200 Hz. The coarse contours, all 29 pulses, and the
    // interpolated V/UV at 100-sample strides match the C++ reference.
    #[test]
    fn test_get_time_base_mixed() {
        let f0 = [
            100.0, 100.0, 100.0, 0.0, 0.0, 0.0, 300.0, 300.0, 500.0, 500.0, 0.0, 0.0, 200.0, 200.0,
            200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0,
        ];
        let f0_length = f0.len();
        let y_length = 1601;
        let mut time_axis = vec![0.0; y_length];
        let mut coarse_time_axis = vec![0.0; f0_length + 1];
        let mut coarse_f0 = vec![0.0; f0_length + 1];
        let mut coarse_vuv = vec![0.0; f0_length + 1];
        get_temporal_parameters_for_time_base(
            &f0,
            TIMEBASE_FS,
            y_length,
            TIMEBASE_FRAME_PERIOD,
            TIMEBASE_LOWEST_F0,
            &mut time_axis,
            &mut coarse_time_axis,
            &mut coarse_f0,
            &mut coarse_vuv,
        );
        assert_eq!(coarse_f0, COARSE_F0_MIXED);
        assert_eq!(coarse_vuv, COARSE_VUV_MIXED);

        let mut pulse_locations = vec![0.0; y_length];
        let mut pulse_locations_index = vec![0usize; y_length];
        let mut pulse_locations_time_shift = vec![0.0; y_length];
        let mut interpolated_vuv = vec![0.0; y_length];
        let n = get_time_base(
            &f0,
            TIMEBASE_FS,
            TIMEBASE_FRAME_PERIOD,
            y_length,
            TIMEBASE_LOWEST_F0,
            &mut pulse_locations,
            &mut pulse_locations_index,
            &mut pulse_locations_time_shift,
            &mut interpolated_vuv,
        );
        assert_eq!(n, PULSES_MIXED.len());
        check_pulses(
            &PULSES_MIXED,
            &pulse_locations[..n],
            &pulse_locations_index[..n],
            &pulse_locations_time_shift[..n],
        );
        for (i, &expected) in (0..y_length).step_by(100).zip(VUV_MIXED.iter()) {
            assert_eq!(interpolated_vuv[i], expected, "vuv at sample {i}");
        }
    }

    // get_time_base (LOWF0): F0 values below lowest_f0 (8.8125 Hz) are treated
    // as unvoiced (coarse_f0 clamped to 0.0, coarse_vuv to 0.0), the final
    // coarse position extrapolates the last two values, and the unvoiced
    // samples synthesize at the 500 Hz default F0 (two pulses in the 81
    // samples).
    #[test]
    fn test_get_time_base_lowf0() {
        let f0 = [5.0, 5.0, 100.0, 100.0, 100.0];
        let f0_length = f0.len();
        let y_length = 81;
        let mut time_axis = vec![0.0; y_length];
        let mut coarse_time_axis = vec![0.0; f0_length + 1];
        let mut coarse_f0 = vec![0.0; f0_length + 1];
        let mut coarse_vuv = vec![0.0; f0_length + 1];
        get_temporal_parameters_for_time_base(
            &f0,
            TIMEBASE_FS,
            y_length,
            TIMEBASE_FRAME_PERIOD,
            TIMEBASE_LOWEST_F0,
            &mut time_axis,
            &mut coarse_time_axis,
            &mut coarse_f0,
            &mut coarse_vuv,
        );
        assert_eq!(coarse_f0, COARSE_F0_LOWF0);
        assert_eq!(coarse_vuv, COARSE_VUV_LOWF0);

        let mut pulse_locations = vec![0.0; y_length];
        let mut pulse_locations_index = vec![0usize; y_length];
        let mut pulse_locations_time_shift = vec![0.0; y_length];
        let mut interpolated_vuv = vec![0.0; y_length];
        let n = get_time_base(
            &f0,
            TIMEBASE_FS,
            TIMEBASE_FRAME_PERIOD,
            y_length,
            TIMEBASE_LOWEST_F0,
            &mut pulse_locations,
            &mut pulse_locations_index,
            &mut pulse_locations_time_shift,
            &mut interpolated_vuv,
        );
        assert_eq!(n, PULSES_LOWF0.len());
        check_pulses(
            &PULSES_LOWF0,
            &pulse_locations[..n],
            &pulse_locations_index[..n],
            &pulse_locations_time_shift[..n],
        );
        for (i, &expected) in (0..y_length).step_by(10).zip(VUV_LOWF0.iter()) {
            assert_eq!(interpolated_vuv[i], expected, "vuv at sample {i}");
        }
    }

    // get_time_base (THRESH): with coarse_vuv = [0, 1, 1] ramping 0 -> 1 over
    // the first 5 ms frame, the interpolated V/UV is snapped at the 0.5
    // threshold exactly at the frame midpoint (sample 40: `s == 0.5` is not
    // `> 0.5`); the single pulse (in the 500 Hz default-F0 region) matches the
    // C++ reference.
    #[test]
    fn test_get_time_base_vuv_threshold() {
        let f0 = [0.0, 100.0];
        let y_length = 81;
        let mut pulse_locations = vec![0.0; y_length];
        let mut pulse_locations_index = vec![0usize; y_length];
        let mut pulse_locations_time_shift = vec![0.0; y_length];
        let mut interpolated_vuv = vec![0.0; y_length];
        let n = get_time_base(
            &f0,
            TIMEBASE_FS,
            TIMEBASE_FRAME_PERIOD,
            y_length,
            TIMEBASE_LOWEST_F0,
            &mut pulse_locations,
            &mut pulse_locations_index,
            &mut pulse_locations_time_shift,
            &mut interpolated_vuv,
        );
        assert_eq!(n, PULSES_THRESH.len());
        check_pulses(
            &PULSES_THRESH,
            &pulse_locations[..n],
            &pulse_locations_index[..n],
            &pulse_locations_time_shift[..n],
        );
        for (i, &actual) in interpolated_vuv.iter().enumerate() {
            let expected = if i >= 41 { 1.0 } else { 0.0 };
            assert_eq!(actual, expected, "vuv at sample {i}");
        }
    }
}
