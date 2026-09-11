#![allow(clippy::unwrap_used)]
#![allow(clippy::panic)]

//! D4C accuracy tests with real AP extraction.
//!
//! Per EQ-TASK-15 §4: accuracy test `d4c_accuracy.rs` with §3.1 AP thresholds on real extraction.

use std::fs;
use std::path::PathBuf;

use world_rs_core::d4c::{d4c, initialize_d4c_option, D4CError, D4COption};

mod common;
use common::{manifest_dir, psnr_2d};
/// Test target: unwrap/panic in setup and assertions is expected.
const TEST_NAMES: &[&str] = &["speech_clean", "speech_noise", "music_vocal"];

fn vectors_dir() -> PathBuf {
    manifest_dir().join("../../test-vector-data/vectors/d4c")
}

fn read_f64s(bytes: &[u8]) -> Vec<f64> {
    let (chunks, _rest) = bytes.as_chunks::<8>();
    chunks.iter().map(|c| f64::from_le_bytes(*c)).collect()
}

struct D4CReference {
    x: Vec<f64>,
    fs: f64,
    f0: Vec<f64>,
    temporal_positions: Vec<f64>,
    aperiodicity: Vec<Vec<f64>>,
    fft_size: i32,
}

fn read_reference(name: &str) -> D4CReference {
    let dir = vectors_dir();
    let in_path = dir.join(format!("{name}.in"));
    let out_path = dir.join(format!("{name}.out"));
    let in_bytes = fs::read(&in_path).expect("in missing");
    let out_bytes = fs::read(&out_path).expect("out missing");

    let x_length = u32::from_le_bytes(in_bytes[0..4].try_into().unwrap()) as usize;
    let fs = f64::from_le_bytes(in_bytes[4..12].try_into().unwrap());
    let x = read_f64s(&in_bytes[12..]);
    assert_eq!(x.len(), x_length, "x length mismatch");

    let f0_length = u32::from_le_bytes(out_bytes[0..4].try_into().unwrap()) as usize;
    let fft_size = i32::from_le_bytes(out_bytes[4..8].try_into().unwrap());
    let f0 = read_f64s(&out_bytes[8..8 + f0_length * 8]);
    let tp = read_f64s(&out_bytes[8 + f0_length * 8..8 + f0_length * 16]);
    let half = (fft_size / 2) as usize + 1;
    let ap_bytes = &out_bytes[8 + f0_length * 16..];
    let aperiodicity: Vec<Vec<f64>> = ap_bytes.chunks_exact(half * 8).map(read_f64s).collect();
    assert_eq!(aperiodicity.len(), f0_length);

    D4CReference {
        x,
        fs,
        f0,
        temporal_positions: tp,
        aperiodicity,
        fft_size,
    }
}

fn pearson_corr(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let mut av = Vec::new();
    let mut bv = Vec::new();
    for (ra, rb) in a.iter().zip(b) {
        av.extend_from_slice(ra);
        bv.extend_from_slice(rb);
    }
    let n = av.len() as f64;
    let mean_a = av.iter().sum::<f64>() / n;
    let mean_b = bv.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for (x, y) in av.iter().zip(bv) {
        let da = x - mean_a;
        let db = y - mean_b;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    cov / (va.sqrt() * vb.sqrt())
}

#[test]
fn test_d4c_option_defaults() {
    let opt = initialize_d4c_option();
    assert_eq!(opt.threshold, 0.85);
}

#[test]
fn test_d4c_invalid_inputs() {
    // Empty input
    let opt = initialize_d4c_option();
    let err = d4c(&[], 16000.0, &[0.0], &[0.0], 1024, &opt).unwrap_err();
    assert_eq!(err, D4CError::EmptyInput);

    // Non-positive fs
    let err = d4c(&[0.0], -1.0, &[0.0], &[0.0], 1024, &opt).unwrap_err();
    assert!(matches!(err, D4CError::NonPositiveSampleRate { .. }));

    // Non-finite threshold
    let bad_opt = D4COption {
        threshold: f64::NAN,
    };
    let err = d4c(&[0.0], 16000.0, &[0.0], &[0.0], 1024, &bad_opt).unwrap_err();
    assert_eq!(err, D4CError::NonFiniteThreshold);

    // f0 NaN
    let err = d4c(&[0.0], 16000.0, &[0.01], &[f64::NAN], 1024, &opt).unwrap_err();
    assert_eq!(err, D4CError::NonFiniteInput);

    // f0 Inf
    let err = d4c(&[0.0], 16000.0, &[0.01], &[f64::INFINITY], 1024, &opt).unwrap_err();
    assert_eq!(err, D4CError::NonFiniteInput);

    // temporal_positions NaN
    let err = d4c(&[0.0], 16000.0, &[f64::NAN], &[100.0], 1024, &opt).unwrap_err();
    assert_eq!(err, D4CError::NonFiniteInput);

    // temporal_positions Inf
    let err = d4c(&[0.0], 16000.0, &[f64::INFINITY], &[100.0], 1024, &opt).unwrap_err();
    assert_eq!(err, D4CError::NonFiniteInput);

    // mismatched lengths
    let err = d4c(&[0.0], 16000.0, &[0.01, 0.02], &[100.0], 1024, &opt).unwrap_err();
    assert_eq!(err, D4CError::LengthMismatch);

    // fft_size <= 0
    let err = d4c(&[0.0], 16000.0, &[0.01], &[100.0], 0, &opt).unwrap_err();
    assert_eq!(err, D4CError::InvalidFftSize);

    let err = d4c(&[0.0], 16000.0, &[0.01], &[100.0], -1, &opt).unwrap_err();
    assert_eq!(err, D4CError::InvalidFftSize);

    // empty temporal_positions
    let err = d4c(&[0.0], 16000.0, &[], &[100.0], 1024, &opt).unwrap_err();
    assert_eq!(err, D4CError::EmptyTemporalPositions);

    // empty f0
    let err = d4c(&[0.0], 16000.0, &[0.01], &[], 1024, &opt).unwrap_err();
    assert_eq!(err, D4CError::EmptyF0);
}

#[test]
fn test_d4c_accuracy_real_extraction() {
    let opt = initialize_d4c_option();
    let mut failures = Vec::new();
    for name in TEST_NAMES {
        let ref_data = read_reference(name);
        let result = d4c(
            &ref_data.x,
            ref_data.fs,
            &ref_data.temporal_positions,
            &ref_data.f0,
            ref_data.fft_size,
            &opt,
        )
        .expect("d4c should succeed");

        assert_eq!(
            result.len(),
            ref_data.f0.len(),
            "{name}: frame count mismatch"
        );
        for (i, row) in result.iter().enumerate() {
            assert_eq!(
                row.len(),
                (ref_data.fft_size / 2) as usize + 1,
                "{name}: frame {i} length mismatch"
            );
        }

        let psnr = psnr_2d(&result, &ref_data.aperiodicity);
        let r = pearson_corr(&result, &ref_data.aperiodicity);
        let max_abs = result
            .iter()
            .zip(&ref_data.aperiodicity)
            .flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs()))
            .fold(0.0f64, |m, v| m.max(v));

        if std::env::var("WORLD_TEST_VERBOSE").unwrap_or_default() == "1" {
            eprintln!("{name}: PSNR={psnr:.3} dB, r={r:.6}, max_abs={max_abs:.6}");
        }

        let psnr_ok = psnr >= 40.0;
        let r_ok = r >= 0.99;
        let max_abs_ok = max_abs < 1e-3;
        if !psnr_ok || !r_ok || !max_abs_ok {
            failures.push(format!(
                "{name}: psnr_ok={psnr_ok} r_ok={r_ok} max_abs_ok={max_abs_ok}"
            ));
        }
    }
    assert!(failures.is_empty(), "Failures: {}", failures.join("; "));
}

#[test]
fn test_d4c_determinism() {
    let opt = initialize_d4c_option();
    for name in TEST_NAMES {
        let ref_data = read_reference(name);
        let out1 = d4c(
            &ref_data.x,
            ref_data.fs,
            &ref_data.temporal_positions,
            &ref_data.f0,
            ref_data.fft_size,
            &opt,
        )
        .expect("d4c should succeed");
        let out2 = d4c(
            &ref_data.x,
            ref_data.fs,
            &ref_data.temporal_positions,
            &ref_data.f0,
            ref_data.fft_size,
            &opt,
        )
        .expect("d4c should succeed");
        assert_eq!(out1.len(), out2.len(), "{name}: determinism frame count");
        for (i, (r1, r2)) in out1.iter().zip(out2).enumerate() {
            assert_eq!(r1.len(), r2.len(), "{name}: frame {i} len");
            for (j, (v1, v2)) in r1.iter().zip(r2).enumerate() {
                assert!(
                    v1.to_bits() == v2.to_bits(),
                    "{name}: frame {i} bin {j} not bit-identical: {v1} vs {v2}"
                );
            }
        }
    }
}
