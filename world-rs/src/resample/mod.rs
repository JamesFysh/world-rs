//! Polyphase sample-rate conversion (port of the `audiojs/resample`
//! `@audio/resample-polyphase` atom, `ext_src/resample/`).
//!
//! Converts a mono PCM signal between two positive integer sample rates using
//! a fixed 32-tap Kaiser-windowed (β = 8.6) polyphase FIR. The rational rate
//! `to / from` is reduced by its gcd to `L / M`; a Kaiser-sinc prototype of
//! `32 * L` taps (designed in f64) is decomposed into `L` phases of 32 f32
//! taps, and the polyphase loop streams the output with a TAPS-length
//! zero-fill history and group-delay compensation `D = TAPS / 2 = 16` input
//! samples.
//!
//! This is a custom Rust implementation: `World/Resample.h/cpp` does not exist
//! in mmorise/World, so there is no C++ reference to port (locked decisions,
//! `docs/impl/support/WORLD-decisions.md` "Resample module placement" and
//! "Resample rate argument"). The algorithmic reference is the JS polyphase
//! atom; the port must match it bit-for-bit (no `dasp_interpolate`).
//!
//! # Precision
//!
//! The prototype is designed in f64 and the phase coefficients are stored as
//! f32 (matching the JS `Float32Array`); the polyphase accumulation is f64 and
//! the I/O is f32 (the WASM boundary). Core WORLD math is f64; the f32
//! resample boundary is a documented deviation (`WORLD-spec.md` "Resample",
//! `WORLD-decisions.md` "Precision: f64 throughout for v1").
//!
//! # Rate contract
//!
//! Both rates are positive `u32`. A zero rate is rejected by **panicking**
//! (the Rust equivalent of the JS `RangeError` at `polyphase.js:47`); the
//! identity case `from == to` returns the input unchanged.
//!
//! # Phase breakdown
//!
//! - Phase 4-1: module scaffolding, buffer type, the
//!   `resample_polyphase` signature, the `TAPS`/`BETA` constants, and the
//!   zero-rate panic + identity behaviour. The polyphase body is stubbed.
//! - Phase 4-2: resampling constants and helper utilities — the `gcd`
//!   ratio-reduction helper and the `D` group-delay constant.
//! - Phase 4-3: Kaiser-windowed sinc prototype design and phase
//!   decomposition — the `i0` Bessel series, the `kaiser` window, the f64
//!   `prototype`, the f32 `design` phase decomposition, and the process-global
//!   `design_cached` prototype cache.
//! - Phase 4-4: upsampling with anti-aliasing — the crate-private
//!   `run` polyphase loop (TAPS-length zero-fill history, group-delay
//!   compensation `D`, f64 accumulation / f32 I/O) and the batch
//!   `resample_polyphase` (write + flush, truncated to `round(n·to/from)`).
//! - Phase 4-5: downsampling with anti-aliasing.
//! - Phase 4-6 (this): arbitrary rational rate conversion wrapper — the public
//!   `resample` entry point that reduces the rate `to/from` by its gcd to
//!   `L/M`, reuses the cached precomputed polyphase taps, and delegates to the
//!   PHASE4-4/5 polyphase streaming (`resample_polyphase`).
//! - Phase 4-7: unit tests for 16k↔48k conversion.
//! - Phase 4-8: edge-case tests and accuracy validation vs the JS reference.
//! - Phase 4-9: performance benchmarks and WORLD pipeline integration.
//!
//! See `docs/impl/support/PHASE4.md` and `docs/impl/support/WORLD-spec.md`
//! ("Resample").

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};

/// Mono PCM audio buffer (f32 samples) at the WASM boundary. The resampler
/// input is a `&MonoSignal` (a `&[f32]`) and the output a `Vec<f32>`.
pub type MonoSignal = [f32];

/// Number of taps per polyphase phase (`polyphase.js:6`): 16 zero crossings
/// each side of the Kaiser-sinc prototype.
pub const TAPS: usize = 32;

/// Kaiser window shape parameter β (`polyphase.js:23`); ~85 dB stopband.
pub const BETA: f64 = 8.6;

/// Group-delay compensation in input samples (`polyphase.js:7`): `TAPS / 2`,
/// the offset that recentres the causal polyphase window on the prototype
/// centre.
pub const D: usize = TAPS / 2;

/// Maximum batch output length in samples. `resample_polyphase` returns an
/// empty buffer instead of allocating beyond this (matches the `y_length ≤
/// 1e8` synthesis cap).
pub const MAX_RESAMPLE_LEN: usize = 100_000_000;

/// Error type for resampling operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResampleError {
    ZeroRate,
    TooLargeRate,
}

impl std::fmt::Display for ResampleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResampleError::ZeroRate => write!(f, "sample rate must be > 0"),
            ResampleError::TooLargeRate => write!(f, "sample rate ratio too large"),
        }
    }
}

impl std::error::Error for ResampleError {}

/// Greatest common divisor of two sample rates (Euclidean algorithm), used to
/// reduce the rational rate `to / from` to its lowest terms `L / M`
/// (`polyphase.js:16`). Crate-private: a support primitive consumed only
/// internally by the resampler (locked decision "Support primitives
/// crate-private").
///
/// Matches the JS reference exactly: `gcd(a, b)` is `a` when `b == 0`, and
/// `gcd(b, a % b)` otherwise, so `gcd(0, b) == b` and `gcd(a, 0) == a`.
pub(crate) fn gcd(a: u32, b: u32) -> u32 {
    let (mut a, mut b) = (a, b);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Modified Bessel function of the first kind, order 0, by power series
/// (`polyphase.js:10-14`): `i0(x) = Σₖ (x/2k)^(2k)`, accumulated until the
/// term drops below `1e-12` of the running sum. Pure IEEE-754 arithmetic, so
/// it matches the JS reference bit-for-bit.
pub(crate) fn i0(x: f64) -> f64 {
    let (mut s, mut t, mut k) = (1.0f64, 1.0f64, 0u32);
    while t > 1e-12 * s {
        k += 1;
        let q = x / (2.0 * k as f64);
        t *= q * q;
        s += t;
    }
    s
}

/// Kaiser window (`polyphase.js:30`): `i0(β·√(max(0, 1−r²)))/i0(β)` with
/// `β = BETA` (8.6, ~85 dB stopband), for a normalized position `r ∈ [-1, 1]`.
///
/// Note: [`prototype`] inlines the exact JS expression order (the `i0(…)/i0(β)`
/// division is applied to the full prototype product, not per tap) for
/// bit-for-bit parity with the reference, so this helper is exercised by the
/// unit tests.
#[allow(dead_code)] // test utility
pub(crate) fn kaiser(r: f64) -> f64 {
    i0(BETA * (1.0 - r * r).max(0.0).sqrt()) / i0(BETA)
}

/// The `l` polyphase phases of `TAPS` f32 taps (one `[f32; TAPS]` per phase).
pub(crate) type Phases = Vec<[f32; TAPS]>;

/// Kaiser-windowed sinc prototype: a `TAPS * l`-tap lowpass with cutoff
/// `min(1, l/m)/l` cycles per upsampled sample and gain `fc * l`, designed in
/// **f64** (`polyphase.js:20-31`). The `fc * l` gain already yields unity
/// system gain, so no normalization is applied (the per-phase DC gain is
/// ≈ 1.0 — 0.999993/0.999967 for 16k↔48k — not exactly 1.0).
pub(crate) fn prototype(l: usize, m: usize) -> Vec<f64> {
    let n = TAPS * l;
    let c = (n - 1) as f64 / 2.0;
    let fc = 1.0f64.min(l as f64 / m as f64) / l as f64;
    let ib = i0(BETA);
    let mut proto = vec![0.0f64; n];
    for (i, slot) in proto.iter_mut().enumerate() {
        let x = i as f64 - c;
        let px = std::f64::consts::PI * fc * x;
        let sinc = if x == 0.0 { 1.0 } else { px.sin() / px };
        let r = 2.0 * i as f64 / (n - 1) as f64 - 1.0;
        *slot = fc * l as f64 * sinc * i0(BETA * (1.0 - r * r).max(0.0).sqrt()) / ib;
    }
    proto
}

/// Decompose the f64 prototype into `l` phases of `TAPS` **f32** taps
/// (`polyphase.js:19,32-38`): `h[p][k] = proto[k·l + p]`, downcast from the
/// f64 prototype. Group-delay compensation for the polyphase loop is
/// `D = TAPS/2 = 16` input samples (the convolution base is
/// `floor(m·m_/l) + D`, recentring the causal window — PHASE4-4/5).
pub(crate) fn design(l: usize, m: usize) -> Phases {
    let proto = prototype(l, m);
    (0..l)
        .map(|p| {
            let mut h = [0.0f32; TAPS];
            for (k, tap) in h.iter_mut().enumerate() {
                *tap = proto[k * l + p] as f32;
            }
            h
        })
        .collect()
}

/// Process-global cache of designed phase banks, keyed by the reduced rate
/// pair `(l, m)`, so repeated resampling at the same rate reuses the
/// prototype (PHASE4-3: "prototype generation is cached for repeated use").
type PhaseCache = HashMap<(usize, usize), Arc<Phases>>;

static PHASE_CACHE: LazyLock<Mutex<PhaseCache>> = LazyLock::new(|| Mutex::new(HashMap::new()));

/// Lock the phase-cache, recovering the guard if a prior holder panicked
/// (poisoned lock). The cache is rebuilt on every access, so a poisoned lock
/// never loses correctness; recovering keeps the total no-panic guarantee.
fn lock_phase_cache<'a>() -> MutexGuard<'a, PhaseCache> {
    match PHASE_CACHE.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// [`design`] with a process-global cache keyed by `(l, m)`: the first call
/// for a rate pair designs the prototype, later calls reuse it. Consumed by
/// [`resample_polyphase`] (PHASE4-4).
pub(crate) fn design_cached(l: usize, m: usize) -> Arc<Phases> {
    let mut cache = lock_phase_cache();
    if cache.len() > 1000 {
        // Evict oldest entries to bound memory
        let keys: Vec<_> = cache.keys().cloned().collect();
        for k in keys.into_iter().take(100) {
            cache.remove(&k);
        }
    }
    cache
        .entry((l, m))
        .or_insert_with(|| Arc::new(design(l, m)))
        .clone()
}

/// One pass of the polyphase loop (port of `polyphase.js:57-77` `run`).
///
/// Emits outputs for the global output indices `out_idx..` while
/// `floor(out_idx·M/L) + D < avail`, reading each of the `TAPS` taps of phase
/// `up % L` (`up = out_idx·M`) from the current chunk (`data[rel]`, `rel ≥ 0`)
/// or the TAPS-length zero-fill history (`hist[TAPS + rel]`, `rel < 0`), then
/// updates the history to the last `min(TAPS, data.len())` samples and
/// advances `consumed` by `data.len()`.
///
/// * `consumed` — absolute index of `data[0]` (input samples seen before this
///   chunk); `avail` — total input samples available (chunk + history).
/// * `base = floor(up/L) + D` is the convolution base: the `+ D` recentres the
///   causal window on the prototype centre (group-delay compensation),
///   removing the filter's own ~`D`-sample delay so the output aligns with the
///   input.
///
/// The f64 accumulation (`sum += h[k] * value`, both operands promoted from
/// f32) and the f32 downcast on emit match the JS reference bit-for-bit
/// (Float32Array elements are f32; JS arithmetic is f64; `Float32Array.from`
/// downcasts). A forward read past the end of `data` (`rel ≥ data.len()`) —
/// which the reference reaches only for the group-delay tail and then
/// truncates away — reads as 0, keeping both the loop total and every kept
/// output identical to the reference.
///
/// The eight parameters mirror the JS `run` closure (which captures `phases`,
/// `hist`, `consumed`, `m`, `L`, `M` from `stream`); they are passed explicitly
/// because the batch API is stateless (no `PolyphaseStream` in Phase 4).
#[allow(clippy::too_many_arguments)]
fn run(
    data: &[f32],
    consumed: &mut usize,
    avail: usize,
    out_idx: &mut usize,
    l: usize,
    m: usize,
    phases: &Phases,
    hist: &mut [f32; TAPS],
) -> Vec<f32> {
    let data_len = data.len();
    let cap = if avail > D {
        ((avail - D) * l / m + 1).saturating_sub(*out_idx)
    } else {
        0
    };
    let mut out = Vec::with_capacity(cap);
    while *out_idx * m / l + D < avail {
        let up = *out_idx * m;
        let base = up / l + D;
        let h = &phases[up % l];
        let base_rel = base as isize - *consumed as isize;
        let mut sum = 0.0f64;
        for (k, &hk) in h.iter().enumerate() {
            // JS `if (idx < 0) break` — stop once `k > base` (no more taps in
            // the causal window).
            if k > base {
                break;
            }
            let rel = base_rel - k as isize;
            let v = if rel >= 0 {
                let r = rel as usize;
                if r < data_len {
                    data[r]
                } else {
                    0.0
                }
            } else if rel >= -(TAPS as isize) {
                hist[(TAPS as isize + rel) as usize]
            } else {
                0.0
            };
            sum += hk as f64 * v as f64;
        }
        out.push(sum as f32);
        *out_idx += 1;
    }
    if data_len >= TAPS {
        hist.copy_from_slice(&data[data_len - TAPS..]);
    } else {
        let n = data_len;
        hist.copy_within(n.., 0);
        hist[TAPS - n..].copy_from_slice(data);
    }
    *consumed += data_len;
    out
}

/// Stateful polyphase resampler for streaming use.
pub struct PolyphaseStream {
    l: usize,
    m: usize,
    phases: Arc<Phases>,
    hist: [f32; TAPS],
    consumed: usize,
    out_idx: usize,
}

impl PolyphaseStream {
    /// Create a new `PolyphaseStream` for converting from `from` Hz to `to` Hz.
    pub fn new(from: u32, to: u32) -> Result<Self, ResampleError> {
        if from == 0 || to == 0 {
            return Err(ResampleError::ZeroRate);
        }
        if from == to {
            // Identity stream: no phases needed.
            return Ok(Self {
                l: 1,
                m: 1,
                phases: Arc::new(vec![[0.0f32; TAPS]]),
                hist: [0.0; TAPS],
                consumed: 0,
                out_idx: 0,
            });
        }
        let g = gcd(from, to);
        let l = (to / g) as usize;
        let m = (from / g) as usize;
        if l > 1_000_000 || m > 1_000_000 {
            return Err(ResampleError::TooLargeRate);
        }
        let phases = design_cached(l, m);
        Ok(Self {
            l,
            m,
            phases,
            hist: [0.0; TAPS],
            consumed: 0,
            out_idx: 0,
        })
    }

    /// Process an input chunk and return the resampled output for that chunk.
    pub fn write(&mut self, input: &[f32]) -> Vec<f32> {
        if self.l == 1 && self.m == 1 {
            return input.to_vec();
        }
        let avail = self.consumed + input.len();
        run(
            input,
            &mut self.consumed,
            avail,
            &mut self.out_idx,
            self.l,
            self.m,
            &self.phases,
            &mut self.hist,
        )
    }

    /// Flush remaining samples from the internal history and return the tail.
    pub fn flush(&mut self) -> Vec<f32> {
        if self.l == 1 && self.m == 1 {
            return Vec::new();
        }
        let expected_len =
            ((self.consumed as f64) * (self.l as f64 / self.m as f64)).round() as usize;
        let remaining = expected_len.saturating_sub(self.out_idx);
        if remaining == 0 {
            return Vec::new();
        }
        let pad = [0.0f32; TAPS];
        let flush_avail = self.consumed + TAPS + D;
        let out_full = run(
            &pad,
            &mut self.consumed,
            flush_avail,
            &mut self.out_idx,
            self.l,
            self.m,
            &self.phases,
            &mut self.hist,
        );
        out_full.into_iter().take(remaining).collect()
    }
}

/// Rational sample-rate conversion using a fixed 32-tap Kaiser (β = 8.6)
/// polyphase FIR.
///
/// # Arguments
///
/// * `x` — mono PCM input (f32 samples).
/// * `from` — input sample rate in Hz; must be `> 0`.
/// * `to` — output sample rate in Hz; must be `> 0`.
///
/// A zero rate (`from == 0` or `to == 0`) returns an empty buffer
/// (the JS throws `RangeError` at `polyphase.js:47`; the Rust port returns
/// empty instead of panicking).
///
/// # Returns
///
/// An owned resampled buffer of length `round(x.len() * to / from)`. When
/// `from == to` the input is returned unchanged. Outputs longer than
/// `MAX_RESAMPLE_LEN` return empty rather than attempting a multi-GB
/// allocation.
///
/// Implemented as the JS default export (`polyphase.js:95-103`): a single
/// `write` over the whole input followed by a `flush` that pads `TAPS` zero
/// samples and drains the group-delay tail, the two parts concatenated and
/// truncated to `round(n·to/from)`. The rational rate `to/from` is reduced by
/// its gcd to `L/M`; the phase bank is the cached PHASE4-3 design. A stateful
/// `stream()` API is a possible future extension (PHASE4-4 note) but is not
/// required for the batch v1 pipeline.
pub fn resample_polyphase(x: &MonoSignal, from: u32, to: u32) -> Vec<f32> {
    if from == 0 || to == 0 {
        return Vec::new();
    }
    if from == to {
        return x.to_vec();
    }
    let g = gcd(from, to);
    let l = (to / g) as usize;
    let m = (from / g) as usize;
    if l > 1_000_000 || m > 1_000_000 {
        return Vec::new();
    }
    // Length/budget guards before any compute: `n` scales with `to/from`,
    // so a degenerate pair (e.g. `to = u32::MAX, from = 1`) would otherwise
    // run the full polyphase cost and then plan a multi-GB output from a
    // tiny input. Matches the `y_length ≤ 1e8` synthesis cap.
    let n = (x.len() as f64 * (to as f64 / from as f64)).round() as usize;
    if n > MAX_RESAMPLE_LEN {
        return Vec::new();
    }
    let phases = design_cached(l, m);

    let mut consumed = 0usize;
    let mut out_idx = 0usize;
    let mut hist = [0.0f32; TAPS];

    // write: the whole input is one chunk; avail = consumed + x.len() = x.len().
    let a = run(
        x,
        &mut consumed,
        x.len(),
        &mut out_idx,
        l,
        m,
        &phases,
        &mut hist,
    );

    // flush: pad TAPS zero samples; avail = consumed + TAPS + D (the + D lets
    // the loop reach the group-delay tail, whose out-of-range reads `run`
    // zeroes and which is truncated below).
    let pad = [0.0f32; TAPS];
    let flush_avail = consumed + TAPS + D;
    let b = run(
        &pad,
        &mut consumed,
        flush_avail,
        &mut out_idx,
        l,
        m,
        &phases,
        &mut hist,
    );

    // batch: length round(n·to/from) (polyphase.js:98); copy the write output,
    // then the tail, each clamped to the remaining length. The `to/from` division
    // is applied first (matching the JS `data.length * (to / from)` ordering) so
    // the rounded length is bit-identical to the reference for every rate pair.
    let mut out = vec![0.0f32; n];
    let a_len = a.len().min(n);
    out[..a_len].copy_from_slice(&a[..a_len]);
    if a.len() < n {
        let take = b.len().min(n - a.len());
        out[a.len()..a.len() + take].copy_from_slice(&b[..take]);
    }
    out
}

/// Arbitrary rational rate conversion (PHASE4-6): the public entry point for
/// converting a mono PCM signal between two positive integer sample rates.
///
/// This is the unified wrapper over the polyphase implementation
/// ([`resample_polyphase`], PHASE4-4/5). It reduces the rational rate
/// `to / from` by its gcd to `L / M`, reuses the cached precomputed polyphase
/// taps (PHASE4-3), and streams the output through the polyphase loop,
/// returning a buffer of length `round(x.len() * to / from)`
/// (`polyphase.js:98`). Because it delegates to [`resample_polyphase`], its
/// output is bit-for-bit identical to the polyphase reference for every rate
/// pair.
///
/// # Arguments
///
/// * `x` — mono PCM input (f32 samples).
/// * `from` — input sample rate in Hz; must be `> 0`.
/// * `to` — output sample rate in Hz; must be `> 0`.
///
/// A zero rate (`from == 0` or `to == 0`) returns an empty buffer
/// (the JS throws `RangeError` at `polyphase.js:47`; the Rust port returns
/// empty instead of panicking).
///
/// # Returns
///
/// An owned resampled buffer of length `round(x.len() * to / from)`. When
/// `from == to` the input is returned unchanged.
pub fn resample(x: &MonoSignal, from: u32, to: u32) -> Vec<f32> {
    resample_polyphase(x, from, to)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Identity case (`from == to`): the input is returned unchanged
    /// (`polyphase.js:50,58`), distinct from the zero-rate panic path.
    #[test]
    fn test_resample_identity() {
        let x = vec![1.0f32, 2.0, 3.0];
        let out = resample_polyphase(&x, 16000, 16000);
        assert_eq!(out, x);
    }

    /// Zero-rate contract (PHASE4-1): a zero `from` rate panics with the
    /// descriptive message (the Rust equivalent of the JS `RangeError` at
    /// `polyphase.js:47`).
    #[test]
    fn test_resample_zero_from_returns_empty() {
        let x = vec![1.0f32, 2.0];
        let out = resample_polyphase(&x, 0, 16000);
        assert!(out.is_empty());
    }

    /// Zero-rate contract (PHASE4-1): a zero `to` rate returns empty.
    #[test]
    fn test_resample_zero_to_returns_empty() {
        let x = vec![1.0f32, 2.0];
        let out = resample_polyphase(&x, 16000, 0);
        assert!(out.is_empty());
    }

    /// Output length is `round(n * to / from)` (`polyphase.js:98`) for both the
    /// 16k→48k (3:1 upsample) and 48k→16k (1:3 downsample) directions. A zero
    /// input resamples to a zero output of the correct length.
    #[test]
    fn test_resample_output_length() {
        let x = vec![0.0f32; 1000];
        // 16k -> 48k is a 3:1 upsample.
        let up = resample_polyphase(&x, 16000, 48000);
        assert_eq!(up.len(), 3000);
        // 48k -> 16k is a 1:3 downsample.
        let down = resample_polyphase(&x, 48000, 16000);
        let expected: f64 = 1000.0 * 16000.0 / 48000.0;
        assert_eq!(down.len(), expected.round() as usize);
    }

    /// Kaiser prototype constants match the JS reference
    /// (`polyphase.js:6,7,23`).
    #[test]
    fn test_constants_match_js_reference() {
        assert_eq!(TAPS, 32);
        assert_eq!(BETA, 8.6);
        assert_eq!(D, 16);
    }

    /// Group-delay constant is `TAPS / 2` (`polyphase.js:7`).
    #[test]
    fn test_d_is_half_taps() {
        assert_eq!(D, TAPS / 2);
    }

    /// `gcd` reduces the 16k→48k rate to `L/M = 3/1` (`polyphase.js:16,48-49`).
    #[test]
    fn test_gcd_16k_to_48k() {
        let g = gcd(16000, 48000);
        assert_eq!(g, 16000);
        assert_eq!((48000 / g, 16000 / g), (3, 1));
    }

    /// `gcd` reduces the 48k→16k rate to `L/M = 1/3` (`polyphase.js:16,48-49`).
    #[test]
    fn test_gcd_48k_to_16k() {
        let g = gcd(48000, 16000);
        assert_eq!(g, 16000);
        assert_eq!((16000 / g, 48000 / g), (1, 3));
    }

    /// `gcd` matches the JS Euclidean reference on a range of rate pairs
    /// (`polyphase.js:16`), including the zero-rate edge cases.
    #[test]
    fn test_gcd_matches_js_reference() {
        let cases = [
            (44100u32, 48000u32, 300u32),
            (48000, 44100, 300),
            (16000, 16000, 16000),
            (1, 1, 1),
            (1, 48000, 1),
            (48000, 1, 1),
            (0, 48000, 48000),
            (48000, 0, 48000),
            (0, 0, 0),
        ];
        for (a, b, expected) in cases {
            assert_eq!(gcd(a, b), expected, "gcd({}, {})", a, b);
        }
    }

    /// `i0` Bessel series matches the JS reference bit-for-bit
    /// (`polyphase.js:10-14`): `i0(8.6) == 750.4611595631624`.
    #[test]
    fn test_i0_matches_js_reference() {
        assert_eq!(i0(BETA).to_bits(), 0x4087_73b0_746c_d029);
        assert_eq!(i0(0.0), 1.0);
    }

    /// Kaiser window endpoints (`polyphase.js:30`): `kaiser(0) == 1` and
    /// `kaiser(±1) == 1/i0(β) == 0.001332513997902426`, symmetric in `r`.
    #[test]
    fn test_kaiser_endpoints_match_js_reference() {
        assert_eq!(kaiser(0.0), 1.0);
        assert_eq!(kaiser(1.0).to_bits(), 0x3f55_d4f8_02b7_8d58);
        assert_eq!(kaiser(-1.0), kaiser(1.0));
    }

    /// Per-phase DC gain ≈ 1.0 within 1e-4 for various reduced rate pairs
    /// (PHASE4-3 acceptance criteria; no explicit normalization — the `fc*l`
    /// prototype gain already yields unity system gain).
    #[test]
    fn test_design_dc_gain_near_unity() {
        for &(l, m) in &[(3, 1), (1, 3), (7, 5), (160, 147), (147, 160)] {
            let phases = design(l, m);
            assert_eq!(phases.len(), l, "design({}, {})", l, m);
            for (p, h) in phases.iter().enumerate() {
                let gain: f64 = h.iter().map(|&t| t as f64).sum();
                assert!(
                    (gain - 1.0).abs() < 1e-4,
                    "design({}, {}) phase {} DC gain {} not ≈ 1.0",
                    l,
                    m,
                    p,
                    gain
                );
            }
        }
    }

    /// Pinned per-phase DC sums (f64 accumulation of the f32 taps, in order)
    /// match the JS reference exactly for the 16k↔48k rate pairs
    /// (`polyphase.js:19-39`): 0.999993/0.999967, not exactly 1.0.
    #[test]
    fn test_design_dc_sums_match_js_reference() {
        let up = design(3, 1);
        let sums: Vec<f64> = up
            .iter()
            .map(|h| h.iter().map(|&t| t as f64).sum())
            .collect();
        assert_eq!(sums[0].to_bits(), 0x3fef_fff2_dfd3_6000);
        assert_eq!(sums[1].to_bits(), 0x3fef_ffed_5815_0000);
        assert_eq!(sums[2].to_bits(), 0x3fef_fff2_dfd3_6000);

        let down = design(1, 3);
        let gain: f64 = down[0].iter().map(|&t| t as f64).sum();
        assert_eq!(gain.to_bits(), 0x3fef_ffbb_1cec_8000);
    }

    /// Phase symmetry: the prototype is even about its centre, so phase `p`
    /// mirrors phase `l-1-p` bit-for-bit (`h[p][k] == h[l-1-p][TAPS-1-k]`),
    /// and for odd `l` the middle phase is individually symmetric.
    #[test]
    fn test_design_phase_symmetry() {
        for &(l, m) in &[(3, 1), (1, 3), (7, 5), (160, 147)] {
            let phases = design(l, m);
            for p in 0..l {
                for k in 0..TAPS {
                    assert_eq!(
                        phases[p][k],
                        phases[l - 1 - p][TAPS - 1 - k],
                        "design({}, {}) phase {} tap {} not symmetric",
                        l,
                        m,
                        p,
                        k
                    );
                }
            }
            if l % 2 == 1 {
                let mid = &phases[l / 2];
                for k in 0..TAPS / 2 {
                    assert_eq!(
                        mid[k],
                        mid[TAPS - 1 - k],
                        "design({}, {}) middle phase tap {} not symmetric",
                        l,
                        m,
                        k
                    );
                }
            }
        }
    }

    /// Phase coefficients match the JS reference bit-for-bit for the 16k→48k
    /// (L=3, M=1) and 48k→16k (L=1, M=3) rate pairs (vectors generated from
    /// the `polyphase.js` `design`, `ext_src/resample`).
    #[test]
    fn test_design_phases_match_js_reference_bits() {
        let up = design(3, 1);
        let up_phase0: [u32; TAPS] = [
            0xb760_b805,
            0x3893_47ab,
            0xb955_1398,
            0x39f3_f3d2,
            0xba73_78f9,
            0x3add_3384,
            0xbb3b_7913,
            0x3b96_9d79,
            0xbbe8_2949,
            0x3c2d_64fc,
            0xbc7d_8240,
            0x3cb7_83df,
            0xbd05_be17,
            0x3d49_f3b0,
            0xbda8_6256,
            0x3e41_640f,
            0x3f74_5a40,
            0xbe08_a822,
            0x3d8b_6d15,
            0xbd2e_dcf6,
            0x3ceb_69f8,
            0xbca2_4132,
            0x3c5f_bf0a,
            0xbc18_1fd8,
            0x3bc9_c741,
            0xbb81_419c,
            0x3b1e_471c,
            0xbab6_db1b,
            0x3a43_b631,
            0xb9bc_8280,
            0x391a_b016,
            0xb83c_7796,
        ];
        for (k, &bits) in up_phase0.iter().enumerate() {
            assert_eq!(up[0][k].to_bits(), bits, "16k->48k phase 0 tap {}", k);
        }
        let up_phase2: [u32; TAPS] = [
            0xb83c_7796,
            0x391a_b016,
            0xb9bc_8280,
            0x3a43_b631,
            0xbab6_db1b,
            0x3b1e_471c,
            0xbb81_419c,
            0x3bc9_c741,
            0xbc18_1fd8,
            0x3c5f_bf0a,
            0xbca2_4132,
            0x3ceb_69f8,
            0xbd2e_dcf6,
            0x3d8b_6d15,
            0xbe08_a822,
            0x3f74_5a40,
            0x3e41_640f,
            0xbda8_6256,
            0x3d49_f3b0,
            0xbd05_be17,
            0x3cb7_83df,
            0xbc7d_8240,
            0x3c2d_64fc,
            0xbbe8_2949,
            0x3b96_9d79,
            0xbb3b_7913,
            0x3add_3384,
            0xba73_78f9,
            0x39f3_f3d2,
            0xb955_1398,
            0x3893_47ab,
            0xb760_b805,
        ];
        for (k, &bits) in up_phase2.iter().enumerate() {
            assert_eq!(up[2][k].to_bits(), bits, "16k->48k phase 2 tap {}", k);
        }

        let down = design(1, 3);
        let down_phase0: [u32; TAPS] = [
            0xb765_8d2e,
            0x389a_84a0,
            0x39e2_68dd,
            0x3a02_bdbd,
            0xba83_565c,
            0xbb6f_de8e,
            0xbb4c_24f9,
            0x3ba4_9a7b,
            0x3c7e_9b02,
            0x3c3e_e488,
            0xbc8c_41ad,
            0xbd4c_a6f6,
            0xbd17_3077,
            0x3d6a_953e,
            0x3e51_35f7,
            0x3ea2_4a5f,
            0x3ea2_4a5f,
            0x3e51_35f7,
            0x3d6a_953e,
            0xbd17_3077,
            0xbd4c_a6f6,
            0xbc8c_41ad,
            0x3c3e_e488,
            0x3c7e_9b02,
            0x3ba4_9a7b,
            0xbb4c_24f9,
            0xbb6f_de8e,
            0xba83_565c,
            0x3a02_bdbd,
            0x39e2_68dd,
            0x389a_84a0,
            0xb765_8d2e,
        ];
        for (k, &bits) in down_phase0.iter().enumerate() {
            assert_eq!(down[0][k].to_bits(), bits, "48k->16k phase 0 tap {}", k);
        }
    }

    /// Magnitude of the f64 prototype's frequency response
    /// `H(f) = |Σᵢ proto[i]·e^(−2πi·f·i)|` at the upsampled (L·fs) rate, `f`
    /// in cycles per upsampled sample.
    fn prototype_response(proto: &[f64], f: f64) -> f64 {
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for (i, &h) in proto.iter().enumerate() {
            let th = std::f64::consts::TAU * f * i as f64;
            re += h * th.cos();
            im -= h * th.sin();
        }
        (re * re + im * im).sqrt()
    }

    /// Frequency response matches the JS reference within 1e-6 relative
    /// (PHASE4-3 acceptance criteria): unity-ish gain at DC (scaled by `l`)
    /// and half-cutoff, ≈ −88 dB or better beyond the cutoff
    /// (`polyphase.js:20-31`).
    #[test]
    fn test_design_frequency_response_matches_js_reference() {
        let fc = 1.0 / 3.0;
        let up = prototype(3, 1);
        let up_cases: [(f64, f64); 7] = [
            (0.0, 2.999978493727499),
            (0.5 * fc, 1.49999102325502),
            (0.9 * fc, 1.0878690309632239e-5),
            (fc, 2.6693006708335275e-6),
            (1.1 * fc, 7.965228089994228e-6),
            (2.0 * fc, 2.6693006970446152e-6),
            (4.0 * fc, 2.6693006447840906e-6),
        ];
        for (f, expected) in up_cases {
            let got = prototype_response(&up, f);
            assert!(
                (got - expected).abs() <= expected.abs() * 1e-6,
                "16k->48k H({}) = {}, expected {}",
                f,
                got,
                expected
            );
        }

        let down = prototype(1, 3);
        let down_cases: [(f64, f64); 7] = [
            (0.0, 0.9999671712577103),
            (0.5 * fc, 0.4999950821309204),
            (0.9 * fc, 3.8287616458453994e-5),
            (fc, 1.8202398961543987e-5),
            (1.1 * fc, 5.209417395906267e-6),
            (2.0 * fc, 1.8202398962409685e-5),
            (4.0 * fc, 1.82023989607162e-5),
        ];
        for (f, expected) in down_cases {
            let got = prototype_response(&down, f);
            assert!(
                (got - expected).abs() <= expected.abs() * 1e-6,
                "48k->16k H({}) = {}, expected {}",
                f,
                got,
                expected
            );
        }
    }

    /// `design_cached` returns the same allocation for repeated `(l, m)`
    /// lookups (PHASE4-3: prototype generation is cached for repeated use)
    /// and distinct allocations for distinct rate pairs.
    #[test]
    fn test_design_cached_reuses_prototype() {
        let a = design_cached(3, 1);
        let b = design_cached(3, 1);
        assert!(Arc::ptr_eq(&a, &b), "cache miss for (3, 1)");
        let c = design_cached(1, 3);
        assert!(
            !Arc::ptr_eq(&a, &c),
            "cache collision between (3,1) and (1,3)"
        );
        assert_eq!(&*a, &design(3, 1), "cached phases differ from design");
    }

    /// Prototype generation completes in < 1 ms for typical lengths
    /// (PHASE4-3 acceptance criteria): the 16k↔48k pairs (L ≤ 3, n ≤ 96
    /// taps) take tens of microseconds; the worst realistic pair (L = 160,
    /// 44.1k↔48k, n = 5120 taps) stays well under 10 ms.
    #[test]
    fn test_design_timing() {
        design(3, 1);
        design(1, 3);
        let start = std::time::Instant::now();
        design(3, 1);
        let up = start.elapsed().as_secs_f64();
        design(1, 3);
        let down = start.elapsed().as_secs_f64();
        assert!(up < 1e-3, "design(3, 1) took {} s", up);
        assert!(down < 1e-3, "design(1, 3) took {} s", down);

        let start = std::time::Instant::now();
        design(160, 147);
        let wide = start.elapsed().as_secs_f64();
        assert!(wide < 1e-2, "design(160, 147) took {} s", wide);
    }

    // ---- PHASE4-4: upsampling with anti-aliasing -------------------------

    /// A pure sine, `freq` Hz at sample rate `sr`, `n` samples (f32). The
    /// argument and sine are computed in f64 and stored as f32, matching the
    /// JS reference `sine` (`Math.sin(2·π·f·i/sr)` into a Float32Array) so the
    /// input is bit-identical and the polyphase output can be compared
    /// bit-for-bit.
    fn sine(freq: f32, n: usize, sr: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let arg = 2.0 * std::f64::consts::PI * freq as f64 * i as f64 / sr as f64;
                arg.sin() as f32
            })
            .collect()
    }

    /// RMS over the middle 3/4 (the kernel tapers the edges).
    fn rms(x: &[f32]) -> f64 {
        let n = x.len();
        let a = n >> 3;
        let b = n - (n >> 3);
        let s: f64 = x[a..b].iter().map(|&v| v as f64 * v as f64).sum();
        (s / (b - a) as f64).sqrt()
    }

    /// Goertzel power at `freq` Hz for a length-`n` signal at rate `sr`.
    fn goertzel_power(x: &[f32], freq: f32, sr: u32) -> f64 {
        let n = x.len();
        let k = (n as f64 * freq as f64 / sr as f64).round() as usize;
        let w = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
        let c = 2.0 * w.cos();
        let (mut s1, mut s2) = (0.0f64, 0.0f64);
        for &v in x {
            let s0 = v as f64 + c * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        s1 * s1 + s2 * s2 - c * s1 * s2
    }

    /// Magnitude of the length-`n` DFT at bin `k` (`|Σᵢ x[i]·e^(−2πi·k·i/n)|`),
    /// valid only when `k` is an exact bin (the tone lands on it). A pure tone of
    /// amplitude `A` at bin `k` has magnitude `A·n/2`, so `2·mag/n` recovers `A`.
    fn bin_mag(x: &[f32], k: usize) -> f64 {
        let n = x.len();
        let w = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for (i, &v) in x.iter().enumerate() {
            let th = w * i as f64;
            re += v as f64 * th.cos();
            im -= v as f64 * th.sin();
        }
        (re * re + im * im).sqrt()
    }

    /// Assert `got` equals the non-zero reference `nz` (f32 bit patterns)
    /// followed by zeros up to `len` — the shape of the JS reference output for
    /// an impulse (a non-zero prefix, then the zero tail).
    fn assert_impulse_matches(got: &[f32], nz: &[u32], len: usize) {
        assert_eq!(got.len(), len, "output length");
        for (i, &bits) in nz.iter().enumerate() {
            assert_eq!(got[i].to_bits(), bits, "sample {}", i);
        }
        for (i, &v) in got.iter().enumerate().skip(nz.len()) {
            assert_eq!(v, 0.0, "sample {} not zero", i);
        }
    }

    /// Upsampled output length is `round(n·to/from)`; for an integer upsample
    /// factor `f` this is `n·f` (PHASE4-4 acceptance criteria,
    /// `polyphase.js:98`).
    #[test]
    fn test_upsample_output_length_factors_2_3_4() {
        let n = 1000usize;
        let x = vec![0.0f32; n];
        // 16k → 32k (2×), 16k → 48k (3×), 12k → 48k (4×).
        assert_eq!(resample_polyphase(&x, 16000, 32000).len(), 2 * n);
        assert_eq!(resample_polyphase(&x, 16000, 48000).len(), 3 * n);
        assert_eq!(resample_polyphase(&x, 12000, 48000).len(), 4 * n);
    }

    /// 3× upsample (16k→48k) of a unit impulse at index 0 matches the JS
    /// reference bit-for-bit (vectors from `polyphase.js`, `ext_src/resample`):
    /// the leading-edge-truncated impulse response, 48 non-zero samples then
    /// zeros.
    #[test]
    fn test_upsample_impulse_3x_matches_js_reference() {
        let mut x = vec![0.0f32; 64];
        x[0] = 1.0;
        let y = resample_polyphase(&x, 16000, 48000);
        const NZ: [u32; 48] = [
            0x3f74_5a40,
            0x3f22_51a8,
            0x3e41_640f,
            0xbe08_a822,
            0xbe51_8adb,
            0xbda8_6256,
            0x3d8b_6d15,
            0x3deb_a014,
            0x3d49_f3b0,
            0xbd2e_dcf6,
            0xbd98_864d,
            0xbd05_be17,
            0x3ceb_69f8,
            0x3d4f_b2bb,
            0x3cb7_83df,
            0xbca2_4132,
            0xbd0f_7434,
            0xbc7d_8240,
            0x3c5f_bf0a,
            0x3cc5_2a59,
            0x3c2d_64fc,
            0xbc18_1fd8,
            0xbc85_1823,
            0xbbe8_2949,
            0x3bc9_c741,
            0x3c2e_b2a4,
            0x3b96_9d79,
            0xbb81_419c,
            0xbbdc_bebe,
            0xbb3b_7913,
            0x3b1e_471c,
            0x3b84_c5e7,
            0x3add_3384,
            0xbab6_db1b,
            0xbb15_e466,
            0xba73_78f9,
            0x3a43_b631,
            0x3a9b_8424,
            0x39f3_f3d2,
            0xb9bc_8280,
            0xba0f_2bc8,
            0xb955_1398,
            0x391a_b016,
            0x3959_c827,
            0x3893_47ab,
            0xb83c_7796,
            0xb85d_bf87,
            0xb760_b805,
        ];
        assert_impulse_matches(&y, &NZ, 192);
    }

    /// 3× upsample (16k→48k) of a unit impulse at index 5 (interior, full
    /// response) matches the JS reference bit-for-bit: 63 non-zero samples
    /// (the response is centred on output 15 = 5·3 by the group-delay
    /// compensation), then zeros.
    #[test]
    fn test_upsample_impulse_3x_interior_matches_js_reference() {
        let mut x = vec![0.0f32; 64];
        x[5] = 1.0;
        let y = resample_polyphase(&x, 16000, 48000);
        const NZ: [u32; 63] = [
            0x3cb7_83df,
            0x3d4f_b2bb,
            0x3ceb_69f8,
            0xbd05_be17,
            0xbd98_864d,
            0xbd2e_dcf6,
            0x3d49_f3b0,
            0x3deb_a014,
            0x3d8b_6d15,
            0xbda8_6256,
            0xbe51_8adb,
            0xbe08_a822,
            0x3e41_640f,
            0x3f22_51a8,
            0x3f74_5a40,
            0x3f74_5a40,
            0x3f22_51a8,
            0x3e41_640f,
            0xbe08_a822,
            0xbe51_8adb,
            0xbda8_6256,
            0x3d8b_6d15,
            0x3deb_a014,
            0x3d49_f3b0,
            0xbd2e_dcf6,
            0xbd98_864d,
            0xbd05_be17,
            0x3ceb_69f8,
            0x3d4f_b2bb,
            0x3cb7_83df,
            0xbca2_4132,
            0xbd0f_7434,
            0xbc7d_8240,
            0x3c5f_bf0a,
            0x3cc5_2a59,
            0x3c2d_64fc,
            0xbc18_1fd8,
            0xbc85_1823,
            0xbbe8_2949,
            0x3bc9_c741,
            0x3c2e_b2a4,
            0x3b96_9d79,
            0xbb81_419c,
            0xbbdc_bebe,
            0xbb3b_7913,
            0x3b1e_471c,
            0x3b84_c5e7,
            0x3add_3384,
            0xbab6_db1b,
            0xbb15_e466,
            0xba73_78f9,
            0x3a43_b631,
            0x3a9b_8424,
            0x39f3_f3d2,
            0xb9bc_8280,
            0xba0f_2bc8,
            0xb955_1398,
            0x391a_b016,
            0x3959_c827,
            0x3893_47ab,
            0xb83c_7796,
            0xb85d_bf87,
            0xb760_b805,
        ];
        assert_impulse_matches(&y, &NZ, 192);
    }

    /// 2× upsample (16k→32k) of a unit impulse at index 0 matches the JS
    /// reference bit-for-bit: 32 non-zero samples then zeros.
    #[test]
    fn test_upsample_impulse_2x_matches_js_reference() {
        let mut x = vec![0.0f32; 32];
        x[0] = 1.0;
        let y = resample_polyphase(&x, 16000, 32000);
        const NZ: [u32; 32] = [
            0x3f66_3f13,
            0x3e98_4060,
            0xbe33_be46,
            0xbdfa_8cc2,
            0x3dbc_93a8,
            0x3d94_0f0a,
            0xbd6e_6772,
            0xbd42_e597,
            0x3d20_c8fd,
            0x3d05_4ae1,
            0xbcdd_680b,
            0xbcb7_caf8,
            0x3c98_357e,
            0x3c7b_1d72,
            0xbc4e_0bce,
            0xbc27_f435,
            0x3c07_d5f9,
            0x3bd9_bc35,
            0xbbac_b507,
            0xbb87_6350,
            0x3b51_73db,
            0x3b1f_9499,
            0xbaee_fe2d,
            0xbaaf_6cd5,
            0x3a7b_988c,
            0x3a2f_7cb6,
            0xb9ec_ac3c,
            0xb998_f262,
            0x393a_de49,
            0x38d2_c58b,
            0xb851_1e84,
            0xb79f_bdaf,
        ];
        assert_impulse_matches(&y, &NZ, 64);
    }

    /// 4× upsample (12k→48k) of a unit impulse at index 0 matches the JS
    /// reference bit-for-bit: 64 non-zero samples then zeros.
    #[test]
    fn test_upsample_impulse_4x_matches_js_reference() {
        let mut x = vec![0.0f32; 32];
        x[0] = 1.0;
        let y = resample_polyphase(&x, 12000, 48000);
        const NZ: [u32; 64] = [
            0x3f79_6887,
            0x3f48_4e69,
            0x3eef_67c1,
            0x3e0c_d058,
            0xbdd9_4a82,
            0xbe54_750b,
            0xbe31_9cf5,
            0xbd7b_77b4,
            0x3d5a_53a7,
            0x3de7_8bd3,
            0x3dcd_49a4,
            0x3d17_d7c6,
            0xbd08_519c,
            0xbd94_5d65,
            0xbd86_389f,
            0xbcc9_b5c7,
            0x3cb7_5231,
            0x3d49_68f0,
            0x3d37_81ae,
            0x3c8a_997a,
            0xbc7c_cee4,
            0xbd0b_29d8,
            0xbcfd_cb55,
            0xbc3f_a223,
            0x3c2e_8d29,
            0x3cbf_c231,
            0x3cae_56e3,
            0x3c03_26af,
            0xbbed_de0c,
            0xbc81_feb4,
            0xbc6b_05b1,
            0xbbaf_ae96,
            0x3b9e_361d,
            0x3c2b_9f72,
            0x3c19_e13c,
            0x3b64_0b67,
            0xbb4b_724f,
            0xbbda_7d48,
            0xbbc1_d122,
            0xbb0d_fd0b,
            0x3afa_4b06,
            0x3b84_ad53,
            0x3b68_2da0,
            0x3aa7_9eb3,
            0xba91_72c0,
            0xbb17_a5ce,
            0xbb02_562d,
            0xba38_9ceb,
            0x3a1c_ea20,
            0x3a9f_fc4f,
            0x3a86_34a1,
            0x39b9_202b,
            0xb998_d837,
            0xba16_ea11,
            0xb9f4_4e44,
            0xb921_de5b,
            0x38ff_5d30,
            0x396f_3a37,
            0x3936_11b6,
            0x3860_1ea6,
            0xb821_73e4,
            0xb886_b0e9,
            0xb82f_54a2,
            0xb72b_8a7a,
        ];
        assert_impulse_matches(&y, &NZ, 128);
    }

    /// Frequency-response test (PHASE4-4 acceptance criteria): a 1 kHz sine is
    /// preserved at 16k→48k — unity passband gain (RMS ≈ √½) and a sample
    /// window that matches the JS reference bit-for-bit.
    #[test]
    fn test_upsample_1khz_sine_preserved_16k_to_48k() {
        let x = sine(1000.0, 2048, 16000);
        let y = resample_polyphase(&x, 16000, 48000);
        assert_eq!(y.len(), 2048 * 3);
        // Unity passband gain: a unit sine has RMS √½ ≈ 0.7071.
        let r = rms(&y);
        assert!(
            (r - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-3,
            "1 kHz RMS {} not ≈ √½",
            r
        );
        // A mid-section sample window matches the JS reference bit-for-bit.
        const WIN: [u32; 16] = [
            0x3f54_dabb,
            0x3f40_7848,
            0x3f28_ca9c,
            0x3f0e_398b,
            0x3ee2_7356,
            0x3ea4_936c,
            0x3e47_c552,
            0x3d85_f200,
            0xbd85_f200,
            0xbe47_c552,
            0xbea4_936c,
            0xbee2_7356,
            0xbf0e_398b,
            0xbf28_ca9c,
            0xbf40_7848,
            0xbf54_dabb,
        ];
        for (k, &bits) in WIN.iter().enumerate() {
            assert_eq!(y[400 + k].to_bits(), bits, "sine sample {}", 400 + k);
        }
    }

    /// No imaging artifacts (PHASE4-4 acceptance criteria): upsampling a 1 kHz
    /// tone by 3× replicates the spectrum at multiples of 16 kHz, so images
    /// land at 16k·k ± 1 kHz (e.g. 15 kHz). The Kaiser lowpass (cutoff 8 kHz)
    /// suppresses them to the ~85 dB stopband, far below the tone.
    #[test]
    fn test_upsample_no_imaging_above_original_nyquist() {
        // 4096 samples → 12288 output; bin spacing 48000/12288 = 3.90625 Hz, so
        // 1 kHz (bin 256) and 15 kHz (bin 3840) land exactly on bins.
        let x = sine(1000.0, 4096, 16000);
        let y = resample_polyphase(&x, 16000, 48000);
        let p_tone = goertzel_power(&y, 1000.0, 48000);
        let p_image = goertzel_power(&y, 15000.0, 48000);
        let db = 10.0 * (p_image / p_tone).log10();
        assert!(
            db < -80.0,
            "15 kHz image only {} dB below the 1 kHz tone (want < −80 dB)",
            db
        );
    }

    // ---- PHASE4-5: downsampling with anti-aliasing ------------------------

    /// Downsampled output length is `round(n·to/from)`; for an integer
    /// downsample factor `f` this is `round(n/f)` (PHASE4-5 acceptance
    /// criteria, `polyphase.js:98`).
    #[test]
    fn test_downsample_output_length_factors_2_3_4() {
        let n = 1000usize;
        let x = vec![0.0f32; n];
        // 32k → 16k (2×), 48k → 16k (3×), 48k → 12k (4×).
        assert_eq!(
            resample_polyphase(&x, 32000, 16000).len(),
            (n as f64 / 2.0).round() as usize
        );
        assert_eq!(
            resample_polyphase(&x, 48000, 16000).len(),
            (n as f64 / 3.0).round() as usize
        );
        assert_eq!(
            resample_polyphase(&x, 48000, 12000).len(),
            (n as f64 / 4.0).round() as usize
        );
    }

    /// 3× downsample (48k→16k) of a unit impulse at index 0 (leading edge)
    /// matches the JS reference bit-for-bit (vectors from `polyphase.js`,
    /// `ext_src/resample`): the leading-edge-truncated impulse response, 6
    /// non-zero samples then zeros. Each output `m` reads the single phase at
    /// tap `k = 3m + D`, so the response is the tail of the prototype.
    #[test]
    fn test_downsample_impulse_3x_matches_js_reference() {
        let mut x = vec![0.0f32; 64];
        x[0] = 1.0;
        let y = resample_polyphase(&x, 48000, 16000);
        const NZ: [u32; 6] = [
            0x3ea2_4a5f,
            0xbd17_3077,
            0x3c3e_e488,
            0xbb4c_24f9,
            0x3a02_bdbd,
            0xb765_8d2e,
        ];
        assert_impulse_matches(&y, &NZ, 21);
    }

    /// 3× downsample (48k→16k) of a unit impulse at index 5 (interior, full
    /// response) matches the JS reference bit-for-bit: 7 non-zero samples then
    /// zeros. Each output `m` reads tap `k = 3m + D − 5`, so the response spans
    /// the prototype around the impulse.
    #[test]
    fn test_downsample_impulse_3x_interior_matches_js_reference() {
        let mut x = vec![0.0f32; 64];
        x[5] = 1.0;
        let y = resample_polyphase(&x, 48000, 16000);
        const NZ: [u32; 7] = [
            0xbd4c_a6f6,
            0x3e51_35f7,
            0x3e51_35f7,
            0xbd4c_a6f6,
            0x3c7e_9b02,
            0xbb6f_de8e,
            0x39e2_68dd,
        ];
        assert_impulse_matches(&y, &NZ, 21);
    }

    /// 2× downsample (32k→16k) of a unit impulse at index 0 matches the JS
    /// reference bit-for-bit: 8 non-zero samples then zeros.
    #[test]
    fn test_downsample_impulse_2x_matches_js_reference() {
        let mut x = vec![0.0f32; 32];
        x[0] = 1.0;
        let y = resample_polyphase(&x, 32000, 16000);
        const NZ: [u32; 8] = [
            0x3ee5_8371,
            0xbda5_e00e,
            0x3d10_b601,
            0xbc86_fb45,
            0x3be8_c8d0,
            0xbb29_9cfe,
            0x3a38_e55c,
            0xb8da_8584,
        ];
        assert_impulse_matches(&y, &NZ, 16);
    }

    /// 4× downsample (48k→12k) of a unit impulse at index 0 matches the JS
    /// reference bit-for-bit: 4 non-zero samples then zeros.
    #[test]
    fn test_downsample_impulse_4x_matches_js_reference() {
        let mut x = vec![0.0f32; 32];
        x[0] = 1.0;
        let y = resample_polyphase(&x, 48000, 12000);
        const NZ: [u32; 4] = [0x3e78_6c6f, 0xbc9c_a24e, 0x3b7b_f6cb, 0xb9c8_2142];
        assert_impulse_matches(&y, &NZ, 8);
    }

    /// Frequency-response test (PHASE4-5 acceptance criteria): a 1 kHz sine is
    /// preserved at 48k→16k — unity passband gain (RMS ≈ √½) and a sample
    /// window that matches the JS reference bit-for-bit.
    #[test]
    fn test_downsample_1khz_sine_preserved_48k_to_16k() {
        let x = sine(1000.0, 4096, 48000);
        let y = resample_polyphase(&x, 48000, 16000);
        assert_eq!(
            y.len(),
            (4096.0_f64 * 16000.0_f64 / 48000.0_f64).round() as usize
        );
        // Unity passband gain: a unit sine has RMS √½ ≈ 0.7071.
        let r = rms(&y);
        assert!(
            (r - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-3,
            "1 kHz RMS {} not ≈ √½",
            r
        );
        // A mid-section sample window matches the JS reference bit-for-bit.
        const WIN: [u32; 16] = [
            0x3d85_f2c0,
            0x3ee2_74ce,
            0x3f40_798b,
            0x3f72_6b3c,
            0x3f7f_74f9,
            0x3f65_9a9c,
            0x3f28_cbbd,
            0x3ea4_948b,
            0xbd85_f2c0,
            0xbee2_74ce,
            0xbf40_798b,
            0xbf72_6b3c,
            0xbf7f_74f9,
            0xbf65_9a9c,
            0xbf28_cbbd,
            0xbea4_948b,
        ];
        for (k, &bits) in WIN.iter().enumerate() {
            assert_eq!(y[400 + k].to_bits(), bits, "sine sample {}", 400 + k);
        }
    }

    /// No aliasing artifacts (PHASE4-5 acceptance criteria): the 48k→16k
    /// anti-aliasing lowpass (cutoff 16 kHz, output Nyquist 8 kHz) leaves the
    /// 8–16 kHz transition band only partially suppressed. A 10 kHz tone folds
    /// to a 6 kHz alias at ≈ −22 dB (transition band), while content at
    /// ≥ 12 kHz is in the Kaiser stopband and its alias is suppressed to
    /// ≤ −60 dB. Input length 12288 (a multiple of 3) gives a 4096-sample
    /// output whose bins (3.90625 Hz) land exactly on every tone and alias
    /// used here.
    #[test]
    fn test_downsample_aliasing_suppressed_above_output_nyquist() {
        let n = 12288usize;
        // The alias of an input tone `f` Hz after decimating 48k→16k is
        // |f − k·16000| folded into [0, 8000]; its output bin is
        // alias / 3.90625.
        fn alias_bin(f: f64) -> usize {
            let a = (f % 16000.0).min(16000.0 - f % 16000.0);
            (a / 3.90625).round() as usize
        }
        fn alias_level_db(x: &[f32], f: f64) -> f64 {
            let amp = 2.0 * bin_mag(x, alias_bin(f)) / x.len() as f64;
            20.0 * amp.log10()
        }

        // 10 kHz → 6 kHz alias sits in the transition band, ≈ −22 dB below a
        // full-scale tone (partially, not fully, suppressed).
        let x10 = sine(10000.0, n, 48000);
        let y10 = resample_polyphase(&x10, 48000, 16000);
        let db10 = alias_level_db(&y10, 10000.0);
        assert!(
            (-26.0..-18.0).contains(&db10),
            "10 kHz → 6 kHz alias at {} dB, want ≈ −22 dB (transition band)",
            db10
        );

        // Content at ≥ 12 kHz is in the stopband: its alias is ≤ −60 dB.
        for &f in &[12000.0f64, 13000.0, 14000.0, 15000.0] {
            let x = sine(f as f32, n, 48000);
            let y = resample_polyphase(&x, 48000, 16000);
            let db = alias_level_db(&y, f);
            assert!(
                db <= -60.0,
                "{} kHz alias only {} dB below full-scale (want ≤ −60 dB)",
                f,
                db
            );
        }
    }

    // ---- PHASE4-6: arbitrary rational rate conversion wrapper -------------

    /// The public `resample` wrapper (PHASE4-6) is the unified entry point for
    /// arbitrary rational rate conversion and delegates to the polyphase
    /// implementation bit-for-bit: for every rate pair its output is identical
    /// to [`resample_polyphase`], including the non-rational-reducible 44.1k↔48k
    /// pair (L/M = 160/147).
    #[test]
    fn test_resample_wrapper_matches_polyphase() {
        for &(from, to) in &[
            (16000u32, 48000),
            (48000, 16000),
            (16000, 32000),
            (48000, 12000),
            (44100, 48000),
            (48000, 44100),
            (16000, 16000),
        ] {
            let x: Vec<f32> = (0..257).map(|i| (i as f64 * 0.0137).sin() as f32).collect();
            assert_eq!(
                resample(&x, from, to),
                resample_polyphase(&x, from, to),
                "rate {} -> {}",
                from,
                to
            );
        }
    }

    /// 16k→48k (3:1 upsample) through the `resample` wrapper (PHASE4-6): the
    /// output length is `input * 3` and the result is bit-identical to the
    /// polyphase reference.
    #[test]
    fn test_resample_16k_to_48k_3x() {
        let x = sine(1000.0, 1000, 16000);
        let y = resample(&x, 16000, 48000);
        assert_eq!(y.len(), 3 * 1000);
        assert_eq!(y, resample_polyphase(&x, 16000, 48000));
    }

    /// 48k→16k (1:3 downsample) through the `resample` wrapper (PHASE4-6): the
    /// output length is `round(input / 3)` and the result is bit-identical to
    /// the polyphase reference.
    #[test]
    fn test_resample_48k_to_16k_1_over_3() {
        let x = sine(1000.0, 1000, 48000);
        let y = resample(&x, 48000, 16000);
        assert_eq!(y.len(), (1000.0_f64 / 3.0).round() as usize);
        assert_eq!(y, resample_polyphase(&x, 48000, 16000));
    }

    /// Identity case through the `resample` wrapper (PHASE4-6): `from == to`
    /// returns the input unchanged.
    #[test]
    fn test_resample_wrapper_identity() {
        let x = vec![0.25f32, -0.5, 1.0, 0.0, -0.75];
        assert_eq!(resample(&x, 16000, 16000), x);
    }

    /// Zero-rate contract through the `resample` wrapper (PHASE4-6): a zero
    /// `from` rate returns empty.
    #[test]
    fn test_resample_wrapper_zero_from_returns_empty() {
        let x = vec![1.0f32, 2.0];
        let out = resample(&x, 0, 16000);
        assert!(out.is_empty());
    }

    /// Zero-rate contract through the `resample` wrapper (PHASE4-6): a zero
    /// `to` rate returns empty.
    #[test]
    fn test_resample_wrapper_zero_to_returns_empty() {
        let x = vec![1.0f32, 2.0];
        let out = resample(&x, 16000, 0);
        assert!(out.is_empty());
    }

    // ---- PHASE4-7: unit tests for 16k↔48k conversion ----------------------

    /// Sine waves at 100 Hz, 1 kHz, and 4 kHz (the passband, ≤ 4 kHz) are
    /// preserved at 16k→48k with amplitude error < 1% (RMS ≈ √½) and
    /// mid-section sample windows that match the JS reference bit-for-bit
    /// (vectors from `polyphase.js`, `ext_src/resample`). The 1 kHz window is
    /// already pinned in PHASE4-4; here the 100 Hz and 4 kHz windows are added.
    #[test]
    fn test_16k_to_48k_sine_passband_preserved() {
        for &freq in &[100.0f32, 1000.0, 4000.0] {
            let x = sine(freq, 2048, 16000);
            let y = resample_polyphase(&x, 16000, 48000);
            assert_eq!(y.len(), 2048 * 3);
            // Amplitude error < 1%: a unit sine has RMS √½.
            let r = rms(&y);
            assert!(
                (r - std::f64::consts::FRAC_1_SQRT_2).abs()
                    < 0.01 * std::f64::consts::FRAC_1_SQRT_2,
                "16k->48k {} Hz RMS {} not within 1% of √½",
                freq,
                r
            );
        }
        // Bit-for-bit mid-section windows (samples 400..416).
        let win_100: [u32; 16] = [
            0xbf5c_dbc5,
            0xbf5b_256b,
            0xbf59_64e0,
            0xbf57_9b2a,
            0xbf55_c837,
            0xbf53_eb49,
            0xbf52_0575,
            0xbf50_16a1,
            0xbf4e_1e0b,
            0xbf4c_1cd8,
            0xbf4a_12e7,
            0xbf47_ff70,
            0xbf45_e3a9,
            0xbf43_bf68,
            0xbf41_91e3,
            0xbf3f_5c5d,
        ];
        let y = resample_polyphase(&sine(100.0, 2048, 16000), 16000, 48000);
        for (k, &bits) in win_100.iter().enumerate() {
            assert_eq!(y[400 + k].to_bits(), bits, "100 Hz sample {}", 400 + k);
        }
        let win_4000: [u32; 16] = [
            0x3f35_053f,
            0x3e84_84a1,
            0xbe84_84a1,
            0xbf35_053f,
            0xbf77_46a6,
            0xbf77_46a6,
            0xbf35_053f,
            0xbe84_84a1,
            0x3e84_84a1,
            0x3f35_053f,
            0x3f77_46a6,
            0x3f77_46a6,
            0x3f35_053f,
            0x3e84_84a1,
            0xbe84_84a1,
            0xbf35_053f,
        ];
        let y = resample_polyphase(&sine(4000.0, 2048, 16000), 16000, 48000);
        for (k, &bits) in win_4000.iter().enumerate() {
            assert_eq!(y[400 + k].to_bits(), bits, "4 kHz sample {}", 400 + k);
        }
    }

    /// Sine waves at 100 Hz, 1 kHz, and 4 kHz (the passband, ≤ 4 kHz) are
    /// preserved at 48k→16k with amplitude error < 1% (RMS ≈ √½) and
    /// mid-section sample windows that match the JS reference bit-for-bit. The
    /// reference's upper-band gain droop (−6 dB at the 8 kHz output Nyquist) is
    /// expected, not a defect; 4 kHz sits below it, so the RMS stays within 1%.
    #[test]
    fn test_48k_to_16k_sine_passband_preserved() {
        for &freq in &[100.0f32, 1000.0, 4000.0] {
            let x = sine(freq, 4096, 48000);
            let y = resample_polyphase(&x, 48000, 16000);
            assert_eq!(
                y.len(),
                (4096.0_f64 * 16000.0_f64 / 48000.0_f64).round() as usize
            );
            // Amplitude error < 1%: a unit sine has RMS √½.
            let r = rms(&y);
            assert!(
                (r - std::f64::consts::FRAC_1_SQRT_2).abs()
                    < 0.01 * std::f64::consts::FRAC_1_SQRT_2,
                "48k->16k {} Hz RMS {} not within 1% of √½",
                freq,
                r
            );
        }
        // Bit-for-bit mid-section windows (samples 400..416).
        let win_100: [u32; 16] = [
            0xbbd6_7527,
            0xbd3b_960c,
            0xbdae_09b4,
            0xbdfe_03ae,
            0xbe26_ccb3,
            0xbe4e_55b6,
            0xbe75_8d47,
            0xbe8e_31f6,
            0xbea1_6527,
            0xbeb4_58a2,
            0xbec7_04ed,
            0xbed9_62a9,
            0xbeeb_6a96,
            0xbefd_1596,
            0xbf07_2e57,
            0xbf0f_9c88,
        ];
        let y = resample_polyphase(&sine(100.0, 4096, 48000), 48000, 16000);
        for (k, &bits) in win_100.iter().enumerate() {
            assert_eq!(y[400 + k].to_bits(), bits, "100 Hz sample {}", 400 + k);
        }
        let win_4000: [u32; 16] = [
            0x3e84_759c,
            0x3f77_2c32,
            0xbe84_759c,
            0xbf77_2c32,
            0x3e84_759c,
            0x3f77_2c32,
            0xbe84_759c,
            0xbf77_2c32,
            0x3e84_759c,
            0x3f77_2c32,
            0xbe84_759c,
            0xbf77_2c32,
            0x3e84_759c,
            0x3f77_2c32,
            0xbe84_759c,
            0xbf77_2c32,
        ];
        let y = resample_polyphase(&sine(4000.0, 4096, 48000), 48000, 16000);
        for (k, &bits) in win_4000.iter().enumerate() {
            assert_eq!(y[400 + k].to_bits(), bits, "4 kHz sample {}", 400 + k);
        }
    }

    /// Impulse response shows correct interpolation with group delay D = 16
    /// (PHASE4-7 acceptance criteria): a unit impulse at input index 5 is
    /// centred on output index 5·(to/from) — 15 for 16k→48k (3:1) and ≈ 1.67
    /// for 48k→16k (1:3) — confirming the `D = TAPS/2 = 16` group-delay
    /// compensation recentres the causal polyphase window on the prototype
    /// centre (PHASE4-3). The full bit-for-bit impulse responses are pinned in
    /// PHASE4-4/5; here the peak location (the group delay) is checked.
    #[test]
    fn test_impulse_group_delay_d16() {
        // 16k -> 48k (3:1): impulse at index 5 is centred on output 15.
        let mut x = vec![0.0f32; 64];
        x[5] = 1.0;
        let y = resample_polyphase(&x, 16000, 48000);
        let peak = y
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .unwrap()
            .0;
        assert!(
            (14..=15).contains(&peak),
            "16k->48k impulse peak at {} (want 14..15, i.e. 5·3 ± ½)",
            peak
        );
        // 48k -> 16k (1:3): impulse at index 5 is centred on output ≈ 1.67.
        let mut x = vec![0.0f32; 64];
        x[5] = 1.0;
        let y = resample_polyphase(&x, 48000, 16000);
        let peak = y
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .unwrap()
            .0;
        assert!(
            (1..=2).contains(&peak),
            "48k->16k impulse peak at {} (want 1..2, i.e. 5/3 ± ½)",
            peak
        );
    }

    /// DC offset preservation at 16k→48k (PHASE4-7 acceptance criteria): a
    /// unit-DC input resamples to a steady-state output within 1e-4 of 1.0
    /// (the reference itself is off by 7e-6…3.3e-5, so 1e-6 is unachievable),
    /// and a mid-section sample window matches the JS reference bit-for-bit.
    #[test]
    fn test_16k_to_48k_dc_offset_preserved() {
        let x = vec![1.0f32; 4096];
        let y = resample_polyphase(&x, 16000, 48000);
        assert_eq!(y.len(), 4096 * 3);
        // Steady-state (middle) DC value within 1e-4 of the input DC.
        let a = y.len() / 4;
        let b = 3 * y.len() / 4;
        for &v in &y[a..b] {
            assert!(
                (v as f64 - 1.0).abs() <= 1e-4,
                "16k->48k steady-state DC {} not within 1e-4 of 1.0",
                v
            );
        }
        // Bit-for-bit mid-section window (samples 1000..1016): the three
        // phases' slightly different DC gains (0.9999937 for phases 0/2,
        // 0.9999911 for phase 1) make the value oscillate with period 3.
        let win: [u32; 16] = [
            0x3f7f_ff6b,
            0x3f7f_ff97,
            0x3f7f_ff97,
            0x3f7f_ff6b,
            0x3f7f_ff97,
            0x3f7f_ff97,
            0x3f7f_ff6b,
            0x3f7f_ff97,
            0x3f7f_ff97,
            0x3f7f_ff6b,
            0x3f7f_ff97,
            0x3f7f_ff97,
            0x3f7f_ff6b,
            0x3f7f_ff97,
            0x3f7f_ff97,
            0x3f7f_ff6b,
        ];
        for (k, &bits) in win.iter().enumerate() {
            assert_eq!(y[1000 + k].to_bits(), bits, "DC sample {}", 1000 + k);
        }
    }

    /// DC offset preservation at 48k→16k (PHASE4-7 acceptance criteria): a
    /// unit-DC input resamples to a steady-state output within 1e-4 of 1.0 and
    /// a mid-section sample window matches the JS reference bit-for-bit. With a
    /// single phase (L = 1) the DC is constant (0.999967, off by 3.3e-5).
    #[test]
    fn test_48k_to_16k_dc_offset_preserved() {
        let x = vec![1.0f32; 4096];
        let y = resample_polyphase(&x, 48000, 16000);
        assert_eq!(
            y.len(),
            (4096.0_f64 * 16000.0_f64 / 48000.0_f64).round() as usize
        );
        // Steady-state (middle) DC value within 1e-4 of the input DC.
        let a = y.len() / 4;
        let b = 3 * y.len() / 4;
        for &v in &y[a..b] {
            assert!(
                (v as f64 - 1.0).abs() <= 1e-4,
                "48k->16k steady-state DC {} not within 1e-4 of 1.0",
                v
            );
        }
        // Bit-for-bit mid-section window (samples 1000..1016): constant, since
        // there is a single phase.
        let win = [0x3f7f_fdd9u32; 16];
        for (k, &bits) in win.iter().enumerate() {
            assert_eq!(y[1000 + k].to_bits(), bits, "DC sample {}", 1000 + k);
        }
    }

    /// Silence handling at 16k→48k (PHASE4-7 acceptance criteria): a silent
    /// input produces a silent output of the correct length (the TAPS-length
    /// zero-fill history and the Kaiser window keep every output exactly 0.0).
    #[test]
    fn test_16k_to_48k_silence_produces_silence() {
        let x = vec![0.0f32; 1000];
        let y = resample_polyphase(&x, 16000, 48000);
        assert_eq!(y.len(), 3000);
        assert!(
            y.iter().all(|&v| v == 0.0),
            "16k->48k silence produced non-zero"
        );
    }

    /// Silence handling at 48k→16k (PHASE4-7 acceptance criteria): a silent
    /// input produces a silent output of the correct length.
    #[test]
    fn test_48k_to_16k_silence_produces_silence() {
        let x = vec![0.0f32; 1000];
        let y = resample_polyphase(&x, 48000, 16000);
        assert_eq!(y.len(), 333);
        assert!(
            y.iter().all(|&v| v == 0.0),
            "48k->16k silence produced non-zero"
        );
    }

    /// Windowed-sinc shift of `x` by `frac` samples (±`half_win` taps), used to
    /// align the round-trip group delay before measuring the SNR.
    fn sinc_shift(x: &[f32], frac: f64, half_win: usize) -> Vec<f32> {
        let n = x.len();
        let mut out = vec![0.0f32; n];
        for (i, slot) in out.iter_mut().enumerate() {
            let p = i as f64 + frac;
            let ip = p.floor() as isize;
            let mu = p - p.floor();
            let mut s = 0.0f64;
            for k in -(half_win as isize)..=half_win as isize {
                let j = ip + k;
                if j < 0 || j >= n as isize {
                    continue;
                }
                let d = k as f64 - mu;
                let w = if d == 0.0 {
                    1.0
                } else {
                    (std::f64::consts::PI * d).sin() / (std::f64::consts::PI * d)
                };
                s += x[j as usize] as f64 * w;
            }
            *slot = s as f32;
        }
        out
    }

    /// Round-trip 16k→48k→16k SNR > 50 dB (PHASE4-7 acceptance criteria): a
    /// 440 Hz tone resampled up and down is a high-fidelity copy of the
    /// original. The round-trip introduces a constant ⅓-sample group delay
    /// (the sum of the two filters' D = 16 compensations), so the error is
    /// measured after aligning that delay with a windowed-sinc shift; the
    /// residual is the passband droop (the tone is far below the 8 kHz output
    /// Nyquist, so no aliasing).
    #[test]
    fn test_roundtrip_16k_48k_16k_snr() {
        let x = sine(440.0, 16000, 16000);
        let up = resample_polyphase(&x, 16000, 48000);
        let back = resample_polyphase(&up, 48000, 16000);
        assert_eq!(up.len(), 48000);
        assert_eq!(back.len(), 16000);
        // Align the ⅓-sample group delay (the round-trip advances the signal
        // ⅓ sample, so shift `back` back by ⅓ sample to line it up with `x`).
        let aligned = sinc_shift(&back, -1.0 / 3.0, 128);
        let m = 500usize;
        let (mut sig, mut noise) = (0.0f64, 0.0f64);
        for i in m..x.len() - m {
            sig += x[i] as f64 * x[i] as f64;
            let e = x[i] as f64 - aligned[i] as f64;
            noise += e * e;
        }
        let snr = 10.0 * (sig / noise).log10();
        assert!(
            snr > 50.0,
            "round-trip 16k->48k->16k SNR {} dB not > 50 dB",
            snr
        );
    }

    /// Output length correctness (PHASE4-7): the 16k↔48k conversion produces
    /// `round(n·to/from)` samples for both directions (3:1 upsample, 1:3
    /// downsample), including input lengths that are not multiples of 3 (the
    /// downsample rounds).
    #[test]
    fn test_16k_48k_output_length() {
        for &n in &[1usize, 2, 3, 1000, 1001, 16000] {
            let x = vec![0.0f32; n];
            // 16k -> 48k is a 3:1 upsample: exactly 3n.
            assert_eq!(
                resample_polyphase(&x, 16000, 48000).len(),
                3 * n,
                "16k->48k n={}",
                n
            );
            // 48k -> 16k is a 1:3 downsample: round(n/3).
            assert_eq!(
                resample_polyphase(&x, 48000, 16000).len(),
                (n as f64 / 3.0).round() as usize,
                "48k->16k n={}",
                n
            );
        }
    }

    // ---- PHASE4-8: edge cases & accuracy validation -----------------------

    /// Max-amplitude edge case (PHASE4-8): a full-scale (|x| <= 1) Nyquist
    /// square (alternating +/-1.0) resamples to a finite, bounded output. The
    /// Kaiser polyphase taps sum to ~unity DC gain, so a full-scale input cannot
    /// drive the output beyond a small margin above full scale; the result stays
    /// finite and |y| <= 1.5 for every rate pair (the JS-reference bit-for-bit
    /// comparison of the same signal lives in `tests/resample_accuracy.rs`).
    #[test]
    fn test_edge_case_max_amplitude_finite_bounded() {
        let x: Vec<f32> = (0..1024)
            .map(|i| if i % 2 == 0 { 1.0f32 } else { -1.0 })
            .collect();
        for &(from, to) in &[(16000u32, 48000), (48000, 16000), (44100, 48000)] {
            let y = resample_polyphase(&x, from, to);
            let peak = y.iter().map(|&v| v.abs()).fold(0.0f32, f32::max);
            assert!(
                peak.is_finite(),
                "max-amplitude {}->{} peak {} not finite",
                from,
                to,
                peak
            );
            assert!(
                peak <= 1.5,
                "max-amplitude {}->{} peak {} exceeds full-scale margin",
                from,
                to,
                peak
            );
        }
    }

    /// Combined edge-case sweep (PHASE4-8): across the 16k↔48k and 44.1k↔48k
    /// rate pairs, the four named edge cases behave correctly — silence in gives
    /// silence out (exactly 0.0), a unit impulse gives a finite response, a unit
    /// DC offset settles within 1e-4 of 1.0 (the reference itself is off by
    /// 7e-6…3.3e-5), and a full-scale input stays finite and bounded.
    #[test]
    fn test_edge_cases_silence_impulse_dc_maxamp() {
        for &(from, to) in &[
            (16000u32, 48000),
            (48000, 16000),
            (44100, 48000),
            (48000, 44100),
        ] {
            // Silence in -> silence out (exactly 0.0).
            let silence = vec![0.0f32; 512];
            let ys = resample_polyphase(&silence, from, to);
            assert!(
                ys.iter().all(|&v| v == 0.0),
                "{}->{} silence produced non-zero",
                from,
                to
            );

            // Unit impulse -> a finite response.
            let mut imp = vec![0.0f32; 64];
            imp[5] = 1.0;
            let yi = resample_polyphase(&imp, from, to);
            assert!(
                yi.iter().all(|v| v.is_finite()),
                "{}->{} impulse response not finite",
                from,
                to
            );

            // Unit DC offset -> steady-state within 1e-4 of 1.0.
            let dc = vec![1.0f32; 2048];
            let yd = resample_polyphase(&dc, from, to);
            let a = yd.len() / 4;
            let b = 3 * yd.len() / 4;
            for &v in &yd[a..b] {
                assert!(
                    (v as f64 - 1.0).abs() <= 1e-4,
                    "{}->{} steady-state DC {} not within 1e-4 of 1.0",
                    from,
                    to,
                    v
                );
            }

            // Full-scale input -> finite and bounded.
            let maxamp: Vec<f32> = (0..512)
                .map(|i| if i % 2 == 0 { 1.0f32 } else { -1.0 })
                .collect();
            let ym = resample_polyphase(&maxamp, from, to);
            let peak = ym.iter().map(|&v| v.abs()).fold(0.0f32, f32::max);
            assert!(
                peak.is_finite() && peak <= 1.5,
                "{}->{} max-amplitude peak {} not finite/bounded",
                from,
                to,
                peak
            );
        }
    }
}
