//! WASM-runtime correctness tests for the DIO binding (pitch extraction).
//!
//! Run with `wasm-pack test --node`. These tests run in the real WASM runtime
//! and exercise the exported `dio` function, the `DioOption` class, and the
//! `DioResult` getters with small simulated inputs and inlined expected
//! outputs (C++ reference values for a 1 s @ 16 kHz, 5 ms frame-period signal).

use wasm_bindgen_test::*;

use js_sys::Float64Array;
use world_rs_wasm::dio::{dio, initialize_dio_option, DioOption};

/// A pure sine of `hz` Hz at sample rate `fs`, `n` samples, amplitude `amp`.
fn sine(n: usize, hz: f64, fs: f64, amp: f64) -> Vec<f64> {
    (0..n)
        .map(|i| amp * (2.0 * std::f64::consts::PI * hz * i as f64 / fs).sin())
        .collect()
}

/// `initialize_dio_option` returns the C++ reference defaults.
#[wasm_bindgen_test]
fn initialize_dio_option_matches_cpp_defaults() {
    let option = initialize_dio_option();
    assert_eq!(option.f0_floor(), 71.0);
    assert_eq!(option.f0_ceil(), 800.0);
    assert_eq!(option.channels_in_octave(), 2.0);
    assert_eq!(option.frame_period(), 5.0);
    assert_eq!(option.speed(), 1);
    assert_eq!(option.allowed_range(), 0.1);
}

/// `DioOption` constructor + getters/setters round-trip.
#[wasm_bindgen_test]
fn dio_option_constructor_and_accessors_roundtrip() {
    let mut option = DioOption::new(70.0, 750.0, 1.5, 10.0, 4, 0.2);
    assert_eq!(option.f0_floor(), 70.0);
    assert_eq!(option.f0_ceil(), 750.0);
    assert_eq!(option.channels_in_octave(), 1.5);
    assert_eq!(option.frame_period(), 10.0);
    assert_eq!(option.speed(), 4);
    assert_eq!(option.allowed_range(), 0.2);
    option.set_f0_floor(60.0);
    option.set_speed(2);
    assert_eq!(option.f0_floor(), 60.0);
    assert_eq!(option.speed(), 2);
}

/// A steady 200 Hz sine recovers the expected F0 contour, temporal grid, and
/// band structure (inlined C++ reference values).
#[wasm_bindgen_test]
fn dio_sine_200hz_recovers_contour() {
    let fs = 16000.0;
    let x = sine(16000, 200.0, fs, 0.5);
    let option = initialize_dio_option();
    let result = dio(Float64Array::from(x.as_slice()), fs, &option)
        .expect("dio should succeed on a 200 Hz sine");

    // Inlined expected outputs (1 s @ 16 kHz, 5 ms frame period).
    assert_eq!(result.f0_length(), 201);
    assert_eq!(result.f0().length(), 201);
    assert_eq!(result.temporal_positions().length(), 201);
    assert_eq!(result.temporal_positions().get_index(0), 0.0);
    assert_eq!(result.temporal_positions().get_index(100), 0.5);
    assert_eq!(result.temporal_positions().get_index(200), 1.0);
    assert_eq!(result.number_of_bands(), 7);
    // Candidates/scores are not copied by default to avoid WASM boundary overhead
    assert_eq!(result.f0_candidates().len(), 0);
    assert_eq!(result.f0_scores().len(), 0);

    // The first frame is zeroed by FixF0Contour; a steady 200 Hz sine
    // converges to ~200 Hz in the interior.
    assert_eq!(result.f0().get_index(0), 0.0);
    assert!((result.f0().get_index(100) - 200.0).abs() < 1e-3);

    let f0: Vec<f64> = result.f0().to_vec();
    let voiced: Vec<f64> = f0.iter().copied().filter(|&v| v > 0.0).collect();
    assert!(voiced.len() > 201 / 2, "most frames should be voiced");
    let mean = voiced.iter().sum::<f64>() / voiced.len() as f64;
    assert!(
        (mean - 200.0).abs() < 5.0,
        "voiced mean {} not near 200",
        mean
    );
}

/// Silence is entirely unvoiced (every F0 frame is 0.0).
#[wasm_bindgen_test]
fn dio_silence_is_all_unvoiced() {
    let fs = 16000.0;
    let x: Vec<f64> = vec![0.0; 16000];
    let option = initialize_dio_option();
    let result =
        dio(Float64Array::from(x.as_slice()), fs, &option).expect("dio should succeed on silence");
    assert_eq!(result.f0_length(), 201);
    let f0: Vec<f64> = result.f0().to_vec();
    assert!(f0.iter().all(|&v| v == 0.0), "silence must be all-unvoiced");
}

/// An empty input signal is rejected.
#[wasm_bindgen_test]
fn dio_empty_input_errors() {
    let option = initialize_dio_option();
    assert!(dio(Float64Array::from(&[] as &[f64]), 16000.0, &option).is_err());
}

/// A non-positive sample rate is rejected.
#[wasm_bindgen_test]
fn dio_nonpositive_fs_errors() {
    let x: Vec<f64> = vec![0.0; 16000];
    let option = initialize_dio_option();
    assert!(dio(Float64Array::from(x.as_slice()), 0.0, &option).is_err());
    assert!(dio(Float64Array::from(x.as_slice()), -1.0, &option).is_err());
}

/// A non-positive frame period is rejected.
#[wasm_bindgen_test]
fn dio_nonpositive_frame_period_errors() {
    let x: Vec<f64> = vec![0.0; 16000];
    let option = DioOption::new(71.0, 800.0, 2.0, 0.0, 1, 0.1);
    assert!(dio(Float64Array::from(x.as_slice()), 16000.0, &option).is_err());
}
