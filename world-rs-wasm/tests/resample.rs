//! WASM-runtime correctness tests for the `resample` binding (polyphase
//! sample-rate conversion, Float32Array I/O).
//!
//! Run with `wasm-pack test --node` (see the crate `README`/`PHASE6-1`). These
//! tests run in the real WASM runtime and exercise the exported `resample`
//! function with small simulated inputs and inlined expected outputs. The
//! inlined reference outputs are the exact `@audio/resample-polyphase`
//! (https://github.com/audiojs/resample) results for the same inputs, so a correct port matches
//! them to f32 precision.
//
// Test target: unwrap/panic in setup and assertions is expected; inlined
// reference outputs are exact f32 literals.
#![allow(clippy::excessive_precision)]
#![allow(clippy::unwrap_used)]
#![allow(clippy::panic)]

use wasm_bindgen_test::*;

use js_sys::Float32Array;
use world_rs_wasm::resample::resample;

/// f32 comparison tolerance: the Rust port reproduces the JS reference to
/// within a few f32 ulps (design is f64, coefficients/accumulation differ by
/// rounding order only).
fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-5
}

/// Identity case (`from == to`): the input is returned unchanged.
#[wasm_bindgen_test]
fn resample_identity_returns_input_unchanged() {
    let x: Vec<f32> = vec![0.25, -0.5, 1.0, 0.0, -0.75];
    let out = resample(Float32Array::from(x.as_slice()), 16000, 16000).unwrap();
    assert_eq!(out.to_vec(), x);
}

/// Output length is `round(n * to / from)`: 16k→48k is a 3:1 upsample, 48k→16k
/// a 1:3 downsample.
#[wasm_bindgen_test]
fn resample_output_length_matches_ratio() {
    let x: Vec<f32> = vec![0.0; 1000];
    let up = resample(Float32Array::from(x.as_slice()), 16000, 48000).unwrap();
    assert_eq!(up.length(), 3000);
    let down = resample(Float32Array::from(x.as_slice()), 48000, 16000).unwrap();
    assert_eq!(down.length(), 333);
}

/// 8-sample alternating pattern, 16k → 48k (3:1 upsample). Expected output is
/// the inlined JS polyphase reference result.
#[wasm_bindgen_test]
fn resample_upsample_matches_js_reference() {
    let x: Vec<f32> = vec![1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0];
    let expected: Vec<f32> = vec![
        1.01954186,
        0.799237728,
        0.299427420,
        -0.295409203,
        -0.774345875,
        -0.997378826,
        -0.935989797,
        -0.649163425,
        -0.230316281,
        0.230316281,
        0.649163425,
        0.935989797,
        0.997378826,
        0.774345875,
        0.295409203,
        -0.299427420,
        -0.799237728,
        -1.01954186,
        -0.901503623,
        -0.545646548,
        -0.151372179,
        0.101284221,
        0.148932412,
        0.0579547733,
    ];
    let out = resample(Float32Array::from(x.as_slice()), 16000, 48000).unwrap();
    let got: Vec<f32> = out.to_vec();
    assert_eq!(got.len(), expected.len());
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(close(*g, *e), "i={}: got {} expected {}", i, g, e);
    }
}

/// 12-sample signal, 48k → 16k (1:3 downsample). Expected output is the
/// inlined JS polyphase reference result.
#[wasm_bindgen_test]
fn resample_downsample_matches_js_reference() {
    let x: Vec<f32> = vec![
        0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5, 0.0, 0.5, 1.0, 0.5,
    ];
    let expected: Vec<f32> = vec![0.425273985, 0.261321187, -0.687852204, 0.634749115];
    let out = resample(Float32Array::from(x.as_slice()), 48000, 16000).unwrap();
    let got: Vec<f32> = out.to_vec();
    assert_eq!(got.len(), expected.len());
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(close(*g, *e), "i={}: got {} expected {}", i, g, e);
    }
}

/// Zero-rate contract: a zero `from` rate returns an error.
#[wasm_bindgen_test]
fn resample_zero_from_err() {
    let x: Vec<f32> = vec![1.0, 2.0];
    let res = resample(Float32Array::from(x.as_slice()), 0, 16000);
    assert!(res.is_err());
}

/// Zero-rate contract: a zero `to` rate returns an error.
#[wasm_bindgen_test]
fn resample_zero_to_err() {
    let x: Vec<f32> = vec![1.0, 2.0];
    let res = resample(Float32Array::from(x.as_slice()), 16000, 0);
    assert!(res.is_err());
}
