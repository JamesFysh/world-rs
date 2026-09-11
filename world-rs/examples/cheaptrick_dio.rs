//! End-to-end DIO -> CheapTrick integration example (Phase 2-8).
//!
//! Runs the full analysis chain on one second of 16 kHz audio:
//!   PCM -> DIO (F0 contour + 5 ms temporal grid) -> CheapTrick (spectral
//!   envelope / spectrogram).
//!
//! The key integration contract verified here is *frame alignment*: DIO emits
//! `temporal_positions[i] = i * frame_period / 1000.0` (seconds) and an F0 of
//! the same length, and CheapTrick consumes both directly. This example checks
//! that the grid CheapTrick sees is exactly DIO's 5 ms grid, and that the
//! resulting spectrogram is finite and strictly positive on every frame.
//!
//! Run with:
//!   cargo run -p world-rs-core --example cheaptrick_dio

use world_rs::cheaptrick::{cheaptrick, initialize_cheaptrick_option};
use world_rs::dio::{dio, initialize_dio_option};

const FS: f64 = 16000.0;
const N: usize = 16000; // 1 second at 16 kHz
const PI: f64 = std::f64::consts::PI;

/// A deterministic voiced test signal: a 200 Hz fundamental with a slowly
/// varying pitch and a couple of harmonics, so DIO has a stable contour to
/// track and CheapTrick has structure to smooth.
fn test_signal(n: usize) -> Vec<f64> {
    let mut phase = 0.0;
    (0..n)
        .map(|i| {
            let t = i as f64 / FS;
            let f0 = 200.0 + 40.0 * (2.0 * PI * 0.5 * t).sin();
            phase += 2.0 * PI * f0 / FS;
            let mut v = 0.0;
            for h in 1..=6 {
                v += (h as f64 * phase).sin() / h as f64;
            }
            0.5 * v
        })
        .collect()
}

fn main() {
    let x = test_signal(N);
    let duration_s = N as f64 / FS;

    // 1. DIO: extract the F0 contour and the temporal grid it is aligned to.
    let dio_option = initialize_dio_option();
    let dio_result = dio(&x, FS, &dio_option).expect("valid DIO input");
    let f0_length = dio_result.f0_length;
    assert_eq!(dio_result.f0.len(), f0_length);
    assert_eq!(dio_result.temporal_positions.len(), f0_length);

    // 2. Frame alignment: CheapTrick consumes DIO's `temporal_positions` and
    //    `f0` directly. Verify the grid is exactly the 5 ms DIO grid.
    let frame_period = dio_option.frame_period;
    for (i, &tp) in dio_result.temporal_positions.iter().enumerate() {
        let expected = i as f64 * frame_period / 1000.0;
        assert!(
            (tp - expected).abs() < 1e-12,
            "frame {i}: temporal_positions {tp} != expected {expected}"
        );
    }
    let voiced = dio_result.f0.iter().filter(|&&v| v > 0.0).count();

    // 3. CheapTrick: estimate the spectral envelope for each frame.
    let ct_option = initialize_cheaptrick_option(FS);
    let sp = cheaptrick(
        &x,
        FS,
        &dio_result.temporal_positions,
        &dio_result.f0,
        &ct_option,
    )
    .expect("valid CheapTrick input");

    // 4. Validate the spectrogram: one row per DIO frame, each row is the
    //    half-spectrum, and every value is finite and strictly positive.
    assert_eq!(sp.len(), f0_length, "spectrogram frame count mismatch");
    let bins = (ct_option.fft_size / 2) as usize + 1;
    for (i, row) in sp.iter().enumerate() {
        assert_eq!(row.len(), bins, "frame {i} bin count mismatch");
        for &v in row {
            assert!(
                v.is_finite() && v > 0.0,
                "frame {i} not finite/positive: {v}"
            );
        }
    }

    println!("DIO -> CheapTrick integration (16 kHz, {duration_s:.1} s)");
    println!("  frames (f0_length):      {f0_length}");
    println!("  frame period:            {frame_period} ms");
    println!(
        "  first temporal position: {:.4} s",
        dio_result.temporal_positions[0]
    );
    println!(
        "  last temporal position:  {:.4} s",
        dio_result.temporal_positions[f0_length - 1]
    );
    println!("  voiced frames:           {voiced}/{f0_length}");
    println!("  cheaptrick fft_size:     {}", ct_option.fft_size);
    println!("  spectrogram:             {f0_length} x {bins} (all finite & positive)");
    println!(
        "  frame alignment:         OK (temporal_positions match DIO's {frame_period} ms grid)"
    );
    println!("OK");
}
