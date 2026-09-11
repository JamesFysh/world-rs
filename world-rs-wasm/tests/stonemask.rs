//! WASM-runtime correctness tests for the StoneMask binding (F0 refinement).
//!
//! Run with `wasm-pack test --node`. These tests run in the real WASM runtime
//! and exercise the exported `stone_mask` function with small simulated inputs
//! and inlined expected outputs (C++ reference behaviour: guard rejection to
//! 0.0, and refinement of a steady sine to its true pitch).
//
// Test target: unwrap/panic in setup and assertions is expected.
#![allow(clippy::unwrap_used)]
#![allow(clippy::panic)]

use wasm_bindgen_test::*;

use js_sys::Float64Array;
use world_rs_wasm::stonemask::stone_mask;

/// A pure sine of `hz` Hz at sample rate `fs`, `n` samples, amplitude `amp`.
fn sine(n: usize, hz: f64, fs: f64, amp: f64) -> Vec<f64> {
    (0..n)
        .map(|i| amp * (2.0 * std::f64::consts::PI * hz * i as f64 / fs).sin())
        .collect()
}

/// The refined contour has the same length as the input `f0`.
#[wasm_bindgen_test]
fn stone_mask_output_length_matches_input() {
    let x = sine(16000, 200.0, 16000.0, 0.5);
    let tpos: Vec<f64> = (0..201).map(|i| i as f64 * 5.0 / 1000.0).collect();
    let f0: Vec<f64> = vec![200.0; 201];
    let out = stone_mask(
        Float64Array::from(x.as_slice()),
        16000.0,
        Float64Array::from(tpos.as_slice()),
        Float64Array::from(f0.as_slice()),
    )
    .unwrap();
    assert_eq!(out.length(), 201);
}

/// A steady 200 Hz sine with a 200 Hz DIO contour refines to ~200 Hz
/// (inlined C++ reference behaviour).
#[wasm_bindgen_test]
fn stone_mask_refines_steady_sine_to_f0() {
    let x = sine(16000, 200.0, 16000.0, 0.5);
    let tpos: Vec<f64> = (0..201).map(|i| i as f64 * 5.0 / 1000.0).collect();
    let f0: Vec<f64> = vec![200.0; 201];
    let out = stone_mask(
        Float64Array::from(x.as_slice()),
        16000.0,
        Float64Array::from(tpos.as_slice()),
        Float64Array::from(f0.as_slice()),
    )
    .unwrap();
    let refined: Vec<f64> = out.to_vec();
    let voiced: Vec<f64> = refined.iter().copied().filter(|&v| v > 0.0).collect();
    assert!(!voiced.is_empty(), "a 200 Hz sine should stay voiced");
    let mean = voiced.iter().sum::<f64>() / voiced.len() as f64;
    assert!(
        (mean - 200.0).abs() < 1.0,
        "refined mean {} not near 200",
        mean
    );
    // The interior frame is essentially exactly 200 Hz.
    assert!((refined[100] - 200.0).abs() < 1.0);
}

/// Guard (low): `f0 <= 40` is rejected to 0.0 (independent of the signal).
#[wasm_bindgen_test]
fn stone_mask_guard_low_rejects_to_zero() {
    let x = sine(16000, 200.0, 16000.0, 0.5);
    let out = stone_mask(
        Float64Array::from(x.as_slice()),
        16000.0,
        Float64Array::from([0.0f64].as_slice()),
        Float64Array::from([30.0f64].as_slice()),
    )
    .unwrap();
    assert_eq!(out.get_index(0), 0.0);
}

/// Guard (high): `f0 > fs/12` is rejected to 0.0 (independent of the signal).
#[wasm_bindgen_test]
fn stone_mask_guard_high_rejects_to_zero() {
    let x = sine(16000, 200.0, 16000.0, 0.5);
    let out = stone_mask(
        Float64Array::from(x.as_slice()),
        16000.0,
        Float64Array::from([0.0f64].as_slice()),
        Float64Array::from([2000.0f64].as_slice()),
    )
    .unwrap();
    assert_eq!(out.get_index(0), 0.0);
}
