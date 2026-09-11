//! WASM-runtime correctness tests for the Synthesis binding (waveform
//! reconstruction) and its helpers `get_y_length` / `constant_aperiodicity`.
//!
//! Run with `wasm-pack test --node`. These tests run in the real WASM runtime
//! and exercise the exported `synthesis`, `get_y_length`, and
//! `constant_aperiodicity` functions with small simulated inputs and inlined
//! expected outputs (C++ reference behaviour: output length, constant-AP
//! value, error paths, and a full CheapTrick→Synthesis round trip).
//
// Test target: unwrap/panic in setup and assertions is expected.
#![allow(clippy::unwrap_used)]
#![allow(clippy::panic)]

use wasm_bindgen_test::*;

use js_sys::Float64Array;
use world_rs_wasm::synthesis::{constant_aperiodicity, get_y_length, synthesis};

/// `get_y_length` matches the formula `(f0_length - 1) * frame_period_ms / 1000 * fs + 1`.
#[wasm_bindgen_test]
fn get_y_length_matches_formula() {
    assert_eq!(get_y_length(10, 5.0, 16000.0), 721);
    assert_eq!(get_y_length(1, 5.0, 16000.0), 1);
    assert_eq!(get_y_length(201, 5.0, 16000.0), 16001);
    assert_eq!(get_y_length(201, 5.0, 48000.0), 48001);
}

/// `constant_aperiodicity` returns an `f0_length x (fft_size/2+1)` matrix
/// filled with the default aperiodicity 0.5.
#[wasm_bindgen_test]
fn constant_aperiodicity_shape_and_value() {
    let ap = constant_aperiodicity(5, 128);
    assert_eq!(ap.length() as usize, 5 * 65); // 128/2 + 1
    for v in ap.to_vec() {
        assert_eq!(v, 0.5);
    }
}

/// `synthesis` returns a waveform whose length is exactly `get_y_length`.
#[wasm_bindgen_test]
fn synthesis_output_length_matches_get_y_length() {
    let f0_length = 10usize;
    let fft_size = 128usize;
    let bins = fft_size / 2 + 1;
    let f0: Vec<f64> = vec![100.0; f0_length];
    let sp: Vec<f64> = vec![1.0; f0_length * bins];
    let ap = constant_aperiodicity(f0_length as u32, fft_size as u32);
    let fs = 16000.0;
    let y_length = get_y_length(f0_length as u32, 5.0, fs) as usize;
    let y = synthesis(
        Float64Array::from(f0.as_slice()),
        f0_length as u32,
        Float64Array::from(sp.as_slice()),
        ap,
        fft_size as u32,
        5.0,
        fs,
        y_length as u32,
    )
    .expect("synthesis should succeed");
    assert_eq!(y.length(), y_length as u32);
}

/// A non-power-of-two `fft_size` is rejected.
#[wasm_bindgen_test]
fn synthesis_rejects_non_power_of_two_fft() {
    let f0_length = 10usize;
    let bins = 51; // 100/2 + 1, consistent with the (invalid) fft_size
    let f0: Vec<f64> = vec![100.0; f0_length];
    let sp: Vec<f64> = vec![1.0; f0_length * bins];
    let ap = constant_aperiodicity(f0_length as u32, 100);
    let result = synthesis(
        Float64Array::from(f0.as_slice()),
        f0_length as u32,
        Float64Array::from(sp.as_slice()),
        ap,
        100, // not a power of two
        5.0,
        16000.0,
        81,
    );
    assert!(result.is_err());
}

/// An empty F0 contour is rejected.
#[wasm_bindgen_test]
fn synthesis_empty_f0_errors() {
    let result = synthesis(
        Float64Array::from(&[] as &[f64]),
        0,
        Float64Array::from(&[] as &[f64]),
        Float64Array::from(&[] as &[f64]),
        128,
        5.0,
        16000.0,
        1,
    );
    assert!(result.is_err());
}

/// Full CheapTrick → Synthesis round trip: a 200 Hz sine analyzed and
/// re-synthesized reconstructs a finite waveform whose 200 Hz fundamental
/// dominates the spectral projection.
#[wasm_bindgen_test]
fn synthesis_round_trip_reconstructs_fundamental() {
    use world_rs_wasm::cheaptrick::{cheaptrick, initialize_cheaptrick_option};

    let fs = 16000.0;
    let x: Vec<f64> = (0..16000)
        .map(|i| 0.5 * (2.0 * std::f64::consts::PI * 200.0 * i as f64 / fs).sin())
        .collect();
    let f0_length = 201usize;
    let tpos: Vec<f64> = (0..f0_length).map(|i| i as f64 * 5.0 / 1000.0).collect();
    let f0: Vec<f64> = vec![200.0; f0_length];

    let option = initialize_cheaptrick_option(fs);
    let sp = cheaptrick(
        Float64Array::from(x.as_slice()),
        fs,
        Float64Array::from(tpos.as_slice()),
        Float64Array::from(f0.as_slice()),
        &option,
    )
    .expect("cheaptrick should succeed");
    let fft = option.fft_size() as u32;
    let ap = constant_aperiodicity(f0_length as u32, fft);
    let y_length = get_y_length(f0_length as u32, 5.0, fs);

    let y = synthesis(
        Float64Array::from(f0.as_slice()),
        f0_length as u32,
        sp,
        ap,
        fft,
        5.0,
        fs,
        y_length,
    )
    .expect("synthesis should succeed");
    let y: Vec<f64> = y.to_vec();
    assert_eq!(y.len(), y_length as usize);
    assert!(y.iter().all(|v| v.is_finite()), "output must be finite");

    // WORLD is parametric (no absolute phase): the 200 Hz fundamental must
    // dominate the spectral projection.
    let proj = |hz: f64| {
        let (mut s, mut c) = (0.0f64, 0.0f64);
        for (i, &yv) in y.iter().enumerate() {
            s += yv * (2.0 * std::f64::consts::PI * hz * i as f64 / fs).sin();
            c += yv * (2.0 * std::f64::consts::PI * hz * i as f64 / fs).cos();
        }
        (s * s + c * c).sqrt()
    };
    assert!(
        proj(200.0) > 10.0 * proj(100.0),
        "200 Hz must dominate 100 Hz"
    );
    assert!(
        proj(200.0) > 10.0 * proj(300.0),
        "200 Hz must dominate 300 Hz"
    );
}
