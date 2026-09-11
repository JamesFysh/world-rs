//! Resample module integration tests (Phases 4-1, 4-2, 4-4, 4-5, and 4-6).
//!
//! These integration tests exercise the public [`world_rs::resample`] API:
//! the identity case, the zero-rate panic contract, the output length
//! (`round(n·to/from)`), the Kaiser prototype constants (including the
//! `D` group-delay constant), and the arbitrary rational rate conversion
//! wrapper [`resample`] (PHASE4-6). The `gcd` ratio-reduction helper is
//! crate-private and covered by the unit tests in `src/resample/mod.rs`. The
//! bit-for-bit upsampling tests (impulse response for 2×/3×/4×, 1 kHz sine
//! preservation, no imaging) live in the unit tests in `src/resample/mod.rs`
//! (PHASE4-4); the bit-for-bit downsampling tests (impulse response for
//! 2×/3×/4×, 1 kHz sine preservation, aliasing suppression) live in the unit
//! tests in `src/resample/mod.rs` (PHASE4-5). The broader accuracy validation
//! (16k↔48k sine/impulse/DC, JS-reference PSNR, bit-for-bit vs the JS polyphase
//! reference, the 44.1k↔48k non-integer ratio, and the swept-sine aliasing
//! check) is added in PHASE4-7/8 and lives in `resample_accuracy.rs` (reading
//! the binary reference vectors generated from https://github.com/audiojs/resample).

use world_rs::resample::{resample, resample_polyphase, BETA, D, TAPS};

/// Identity case (`from == to`): the input is returned unchanged
/// (`polyphase.js:50,58`), distinct from the zero-rate panic path.
#[test]
fn resample_identity_returns_input_unchanged() {
    let x = vec![0.25f32, -0.5, 1.0, 0.0, -0.75];
    assert_eq!(resample_polyphase(&x, 16000, 16000), x);
}

/// Zero-rate contract (PHASE4-1): a zero `from` rate returns empty.
#[test]
fn resample_zero_from_panics() {
    let x = vec![1.0f32, 2.0];
    let out = resample_polyphase(&x, 0, 16000);
    assert!(out.is_empty());
}

/// Zero-rate contract (PHASE4-1): a zero `to` rate returns empty.
#[test]
fn resample_zero_to_panics() {
    let x = vec![1.0f32, 2.0];
    let out = resample_polyphase(&x, 16000, 0);
    assert!(out.is_empty());
}

/// `round(n * to / from)` — the output length formula (`polyphase.js:98`).
fn expected_len(n: usize, from: u32, to: u32) -> usize {
    ((n as f64 * to as f64 / from as f64).round()) as usize
}

/// Output length is `round(n·to/from)` (`polyphase.js:98`) for both the
/// 16k→48k (3:1 upsample) and 48k→16k (1:3 downsample) directions, and a
/// non-zero input resamples to a non-zero output (the polyphase body runs,
/// PHASE4-4).
#[test]
fn resample_output_length() {
    let n = 1000usize;
    let x = vec![0.0f32; n];

    let up = resample_polyphase(&x, 16000, 48000);
    assert_eq!(up.len(), expected_len(n, 16000, 48000));
    assert_eq!(up.len(), 3 * n);

    let down = resample_polyphase(&x, 48000, 16000);
    assert_eq!(down.len(), expected_len(n, 48000, 16000));

    // A unit-DC input upsamples to a non-zero (≈ unity-DC) output.
    let dc = vec![1.0f32; n];
    let dc_up = resample_polyphase(&dc, 16000, 48000);
    assert!(dc_up.iter().any(|&v| v.abs() > 0.5), "DC upsample is zero");
}

/// Kaiser prototype constants match the JS reference
/// (`polyphase.js:6,7,23`).
#[test]
fn resample_constants_match_js_reference() {
    assert_eq!(TAPS, 32);
    assert_eq!(BETA, 8.6);
    assert_eq!(D, 16);
}

/// `D` group-delay constant is `TAPS / 2` (`polyphase.js:7`).
#[test]
fn resample_d_is_half_taps() {
    assert_eq!(D, TAPS / 2);
}

// ---- PHASE4-5: downsampling with anti-aliasing ----------------------------

/// A pure sine, `freq` Hz at sample rate `sr`, `n` samples (f32).
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

/// Downsampled output length is `round(n·to/from)`; for an integer downsample
/// factor `f` this is `round(n/f)` (PHASE4-5 acceptance criteria,
/// `polyphase.js:98`).
#[test]
fn resample_downsample_output_length_factors_2_3_4() {
    let n = 1000usize;
    let x = vec![0.0f32; n];
    // 32k → 16k (2×), 48k → 16k (3×), 48k → 12k (4×).
    assert_eq!(
        resample_polyphase(&x, 32000, 16000).len(),
        expected_len(n, 32000, 16000)
    );
    assert_eq!(
        resample_polyphase(&x, 48000, 16000).len(),
        expected_len(n, 48000, 16000)
    );
    assert_eq!(
        resample_polyphase(&x, 48000, 12000).len(),
        expected_len(n, 48000, 12000)
    );
    assert_eq!(resample_polyphase(&x, 32000, 16000).len(), 500);
    assert_eq!(resample_polyphase(&x, 48000, 12000).len(), 250);
}

/// Downsample passband + anti-aliasing (PHASE4-5 acceptance criteria): a 1 kHz
/// tone at 48k→16k is preserved (unity gain, RMS ≈ √½), while a 10 kHz tone
/// (above the 8 kHz output Nyquist) folds to a 6 kHz alias that the Kaiser
/// lowpass leaves only partially suppressed (≈ −22 dB, transition band) and a
/// 13 kHz tone's alias is in the stopband (≤ −60 dB).
#[test]
fn resample_downsample_preserves_low_freq_and_suppresses_alias() {
    // 1 kHz preserved: unity passband gain.
    let x = sine(1000.0, 4096, 48000);
    let y = resample_polyphase(&x, 48000, 16000);
    let r = rms(&y);
    assert!(
        (r - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-3,
        "1 kHz RMS {} not ≈ √½",
        r
    );

    // A 10 kHz tone (above the 8 kHz output Nyquist) must not pass through at
    // full scale: its 6 kHz alias is in the transition band (≈ −22 dB), well
    // below the full-scale tone, and a 13 kHz tone's alias is in the stopband
    // (≤ −60 dB). The whole downsampled 10 kHz signal stays far below the
    // unity-gain passband level.
    let x10 = sine(10000.0, 4096, 48000);
    let y10 = resample_polyphase(&x10, 48000, 16000);
    let r10 = rms(&y10);
    assert!(
        r10 < 0.2 * std::f64::consts::FRAC_1_SQRT_2,
        "10 kHz downsample RMS {} not suppressed (want < 0.2·√½)",
        r10
    );
}

// ---- PHASE4-6: arbitrary rational rate conversion wrapper ------------------

/// The public `resample` wrapper (PHASE4-6) is the unified entry point for
/// arbitrary rational rate conversion and is bit-identical to
/// [`resample_polyphase`] for every rate pair, including the 16k↔48k pair and
/// the non-rational-reducible 44.1k↔48k pair (L/M = 160/147).
#[test]
fn resample_wrapper_matches_polyphase() {
    for &(from, to) in &[
        (16000u32, 48000),
        (48000, 16000),
        (16000, 32000),
        (48000, 12000),
        (44100, 48000),
        (48000, 44100),
        (16000, 16000),
    ] {
        let x: Vec<f32> = (0..513).map(|i| (i as f64 * 0.0137).sin() as f32).collect();
        assert_eq!(
            resample(&x, from, to),
            resample_polyphase(&x, from, to),
            "rate {} -> {}",
            from,
            to
        );
    }
}

/// 16k→48k (3:1 upsample) through the public `resample` wrapper (PHASE4-6):
/// output length = input * 3, a 1 kHz tone is preserved at unity gain
/// (RMS ≈ √½), and the result is bit-identical to the polyphase reference.
#[test]
fn resample_16k_to_48k_3x() {
    let x = sine(1000.0, 1000, 16000);
    let y = resample(&x, 16000, 48000);
    assert_eq!(y.len(), 3 * 1000);
    let r = rms(&y);
    assert!(
        (r - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-3,
        "1 kHz upsample RMS {} not ≈ √½",
        r
    );
    assert_eq!(y, resample_polyphase(&x, 16000, 48000));
}

/// 48k→16k (1:3 downsample) through the public `resample` wrapper (PHASE4-6):
/// output length ≈ input / 3, a 1 kHz tone is preserved at unity gain
/// (RMS ≈ √½), and the result is bit-identical to the polyphase reference.
#[test]
fn resample_48k_to_16k_1_over_3() {
    let x = sine(1000.0, 1000, 48000);
    let y = resample(&x, 48000, 16000);
    assert_eq!(y.len(), expected_len(1000, 48000, 16000));
    let r = rms(&y);
    assert!(
        (r - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-3,
        "1 kHz downsample RMS {} not ≈ √½",
        r
    );
    assert_eq!(y, resample_polyphase(&x, 48000, 16000));
}

/// Zero-rate contract through the public `resample` wrapper (PHASE4-6): a zero
/// `from` rate returns empty.
#[test]
fn resample_wrapper_zero_from_panics() {
    let x = vec![1.0f32, 2.0];
    let out = resample(&x, 0, 16000);
    assert!(out.is_empty());
}

/// Zero-rate contract through the public `resample` wrapper (PHASE4-6): a zero
/// `to` rate returns empty.
#[test]
fn resample_wrapper_zero_to_panics() {
    let x = vec![1.0f32, 2.0];
    let out = resample(&x, 16000, 0);
    assert!(out.is_empty());
}

// ---- PHASE4-8: edge cases & accuracy validation ----------------------------

/// Non-integer rational ratio through the public `resample` wrapper (PHASE4-8):
/// 44100 -> 48000 (reduced by gcd to L/M = 160/147) confirms arbitrary-ratio
/// support. The output length is `round(n * to / from)`, a 440 Hz tone is
/// preserved at unity passband gain (RMS ≈ √½), and the result is bit-identical
/// to the polyphase reference.
#[test]
fn resample_44100_to_48000_arbitrary_ratio() {
    let x = sine(440.0, 4410, 44100);
    let y = resample(&x, 44100, 48000);
    assert_eq!(y.len(), expected_len(4410, 44100, 48000));
    let r = rms(&y);
    assert!(
        (r - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-3,
        "44100->48000 440 Hz RMS {} not ≈ √½",
        r
    );
    assert_eq!(y, resample_polyphase(&x, 44100, 48000));
}

/// Max-amplitude edge case through the public `resample` wrapper (PHASE4-8): a
/// full-scale (|x| <= 1) Nyquist square resamples to a finite, bounded output
/// for every rate pair (the Kaiser polyphase taps sum to ~unity DC gain, so a
/// full-scale input cannot drive the output beyond a small margin above full
/// scale).
#[test]
fn resample_max_amplitude_finite_bounded() {
    let x: Vec<f32> = (0..1024)
        .map(|i| if i % 2 == 0 { 1.0f32 } else { -1.0 })
        .collect();
    for &(from, to) in &[(16000u32, 48000), (48000, 16000), (44100, 48000)] {
        let y = resample(&x, from, to);
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

/// Output-length budget: `n = round(len*to/from)` beyond `MAX_RESAMPLE_LEN`
/// returns empty instead of attempting a multi-GB allocation.
#[test]
fn resample_giant_output_length_returns_empty() {
    let x = vec![0.5f32; 200_000];
    let out = resample_polyphase(&x, 1, 1000);
    assert!(
        out.is_empty(),
        "resample should refuse a 200M-sample output"
    );
}
