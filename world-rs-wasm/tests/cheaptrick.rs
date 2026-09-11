//! WASM-runtime correctness tests for the CheapTrick binding (spectral
//! envelope extraction).
//!
//! Run with `wasm-pack test --node`. These tests run in the real WASM runtime
//! and exercise the exported `cheaptrick` function and the `CheapTrickOption`
//! class with small simulated inputs and inlined expected outputs (C++
//! reference defaults, envelope shape, and the fundamental peak location).

use wasm_bindgen_test::*;

use js_sys::Float64Array;
use world_rs_wasm::cheaptrick::{cheaptrick, initialize_cheaptrick_option, CheapTrickOption};

/// A pure sine of `hz` Hz at sample rate `fs`, `n` samples, amplitude `amp`.
fn sine(n: usize, hz: f64, fs: f64, amp: f64) -> Vec<f64> {
    (0..n)
        .map(|i| amp * (2.0 * std::f64::consts::PI * hz * i as f64 / fs).sin())
        .collect()
}

/// `initialize_cheaptrick_option` returns the C++ reference defaults at 16 kHz.
#[wasm_bindgen_test]
fn initialize_cheaptrick_option_matches_cpp_defaults() {
    let option = initialize_cheaptrick_option(16000.0);
    assert_eq!(option.q1(), -0.15);
    assert_eq!(option.f0_floor(), 71.0);
    assert_eq!(option.fft_size(), 1024);
}

/// `CheapTrickOption` constructor + getters/setters round-trip.
#[wasm_bindgen_test]
fn cheaptrick_option_constructor_and_accessors_roundtrip() {
    let mut option = CheapTrickOption::new(-0.2, 80.0, 2048);
    assert_eq!(option.q1(), -0.2);
    assert_eq!(option.f0_floor(), 80.0);
    assert_eq!(option.fft_size(), 2048);
    option.set_q1(-0.1);
    option.set_fft_size(512);
    assert_eq!(option.q1(), -0.1);
    assert_eq!(option.fft_size(), 512);
}

/// A 200 Hz sine yields a spectrogram of the expected shape (201 frames ×
/// 513 bins) whose per-frame peak sits at the fundamental (bin ≈ 12.8).
#[wasm_bindgen_test]
fn cheaptrick_sine_envelope_shape_and_peak() {
    let fs = 16000.0;
    let x = sine(16000, 200.0, fs, 0.5);
    let tpos: Vec<f64> = (0..201).map(|i| i as f64 * 5.0 / 1000.0).collect();
    let f0: Vec<f64> = vec![200.0; 201];
    let option = initialize_cheaptrick_option(fs);
    let sp = cheaptrick(
        Float64Array::from(x.as_slice()),
        fs,
        Float64Array::from(tpos.as_slice()),
        Float64Array::from(f0.as_slice()),
        &option,
    )
    .expect("cheaptrick should succeed on a 200 Hz sine");

    // Inlined expected: 201 frames, 1024/2 + 1 = 513 bins per frame.
    let frames = 201;
    let bins = 513;
    assert_eq!(sp.length() as usize, frames * bins);

    // The spectral peak of a 200 Hz sine is near bin 200 * 1024 / 16000 = 12.8.
    let row_offset = 100 * bins;
    let row: Vec<f64> = sp.to_vec()[row_offset..row_offset + bins].to_vec();
    let mut peak = 0;
    for (k, &v) in row.iter().enumerate() {
        if v > row[peak] {
            peak = k;
        }
    }
    assert!(
        (peak as f64 - 12.8).abs() <= 2.0,
        "peak bin {} not near the fundamental (12.8)",
        peak
    );
}

/// A `f0`/`temporal_positions` length mismatch is rejected.
#[wasm_bindgen_test]
fn cheaptrick_length_mismatch_errors() {
    let fs = 16000.0;
    let x = sine(16000, 200.0, fs, 0.5);
    let tpos: Vec<f64> = (0..201).map(|i| i as f64 * 5.0 / 1000.0).collect();
    let f0: Vec<f64> = vec![200.0; 100]; // mismatched length
    let option = initialize_cheaptrick_option(fs);
    let result = cheaptrick(
        Float64Array::from(x.as_slice()),
        fs,
        Float64Array::from(tpos.as_slice()),
        Float64Array::from(f0.as_slice()),
        &option,
    );
    assert!(result.is_err());
}

/// An empty input signal is rejected.
#[wasm_bindgen_test]
fn cheaptrick_empty_input_errors() {
    let option = initialize_cheaptrick_option(16000.0);
    let tpos: Vec<f64> = vec![0.0];
    let f0: Vec<f64> = vec![200.0];
    let result = cheaptrick(
        Float64Array::from(&[] as &[f64]),
        16000.0,
        Float64Array::from(tpos.as_slice()),
        Float64Array::from(f0.as_slice()),
        &option,
    );
    assert!(result.is_err());
}
