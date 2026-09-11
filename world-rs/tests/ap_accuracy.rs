// Test target: unwrap/panic in setup and assertions is expected.
#![allow(clippy::unwrap_used)]
#![allow(clippy::panic)]

use std::fs;
use std::io::Read;
use std::path::PathBuf;
use std::process::Command;

use world_rs::synthesis::{get_y_length, synthesis};

mod common;
use common::{cpp_driver_path, manifest_dir, max_abs, skip_if_missing};

// Constant-AP fallback synthesis parity tests (EQ-TASK-8).
// These tests validate synthesis parity with constant aperiodicity matrices
// (ones, zeros, half) without D4C. Real AP extraction parity will be added
// in EQ-TASK-15. // UPGRADE(EQ-TASK-15) resolved: fallback remains valid until D4C implementation lands; D4C stub created in EQ-TASK-15.

const TEST_NAMES: &[&str] = &[
    "sine_220_16k",
    "chirp_16k",
    "jump_16k",
    "transition_16k",
    "unvoiced_16k",
    "silence_16k",
    "sine_440_48k",
];

fn vectors_dir() -> PathBuf {
    manifest_dir().join("../test-vector-data/vectors/synthesis")
}

fn read_f64s(bytes: &[u8]) -> Vec<f64> {
    let (chunks, _rest) = bytes.as_chunks::<8>();
    chunks.iter().map(|c| f64::from_le_bytes(*c)).collect()
}

struct SynthesisInput {
    f0: Vec<f64>,
    fs: f64,
    frame_period_ms: f64,
    fft_size: usize,
    spectrogram: Vec<Vec<f64>>,
}

fn read_synthesis_in(name: &str) -> SynthesisInput {
    let dir = vectors_dir();
    let in_path = dir.join(format!("{name}.in"));
    let bytes = fs::read(&in_path).expect("read in");
    let f0_length = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let fft_size = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let frame_period_ms = f64::from_le_bytes(bytes[8..16].try_into().unwrap());
    let fs = f64::from_le_bytes(bytes[16..24].try_into().unwrap());
    let half = fft_size / 2 + 1;
    let f0_span = 24..24 + f0_length * 8;
    let sp_span = f0_span.end..f0_span.end + f0_length * half * 8;
    let f0 = read_f64s(&bytes[f0_span]);
    let spectrogram: Vec<Vec<f64>> = read_f64s(&bytes[sp_span])
        .chunks(half)
        .map(|c| c.to_vec())
        .collect();
    SynthesisInput {
        f0,
        fs,
        frame_period_ms,
        fft_size,
        spectrogram,
    }
}

fn cpp_synthesis_vec_const_ap(name: &str, ap_val: f64) -> Vec<f64> {
    let cpp_driver = cpp_driver_path();
    skip_if_missing(&cpp_driver, "cpp_driver for AP accuracy");
    let in_path = vectors_dir().join(format!("{name}.in"));
    skip_if_missing(&in_path, &format!("vector {}", name));
    let in_path = vectors_dir().join(format!("{name}.in"));
    let uniq = format!(
        "{:x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let mut out_path = std::env::temp_dir();
    out_path.push(format!("world_ap_out_{}_{}.raw", std::process::id(), uniq));
    let phase = format!("synthesis_vec_const_ap_{}", ap_val);
    let status = Command::new(&cpp_driver)
        .args([
            phase.as_str(),
            in_path.to_str().unwrap(),
            out_path.to_str().unwrap(),
        ])
        .status()
        .expect("run cpp_driver");
    assert!(status.success(), "cpp_driver failed for {name} ap {ap_val}");
    let mut f = fs::File::open(&out_path).expect("open cpp out");
    let mut len_buf = [0u8; 4];
    f.read_exact(&mut len_buf).unwrap();
    let y_len = u32::from_le_bytes(len_buf) as usize;
    let mut y = vec![0.0f64; y_len];
    for slot in y.iter_mut() {
        let mut buf = [0u8; 8];
        f.read_exact(&mut buf).unwrap();
        *slot = f64::from_le_bytes(buf);
    }
    let _ = fs::remove_file(&out_path);
    y
}

#[test]
fn constant_ap_ones() {
    for name in TEST_NAMES {
        let input = read_synthesis_in(name);
        let f0_len = input.f0.len();
        let bins = input.fft_size / 2 + 1;
        let ap = vec![vec![1.0; bins]; f0_len];
        let y_len = get_y_length(f0_len, input.frame_period_ms, input.fs);
        let rust_y = synthesis(
            &input.f0,
            f0_len,
            &input.spectrogram,
            &ap,
            input.fft_size,
            input.frame_period_ms,
            input.fs,
            y_len,
        )
        .expect("synthesis");
        let cpp_y = cpp_synthesis_vec_const_ap(name, 1.0);
        assert_eq!(rust_y.len(), cpp_y.len(), "{name}: length mismatch");
        let err = max_abs(&rust_y, &cpp_y);
        assert!(err < 1e-6, "{name} AP=1.0 max abs error {err} >= 1e-6");
    }
}

#[test]
fn constant_ap_zeros() {
    for name in TEST_NAMES {
        let input = read_synthesis_in(name);
        let f0_len = input.f0.len();
        let bins = input.fft_size / 2 + 1;
        let ap = vec![vec![0.0; bins]; f0_len];
        let y_len = get_y_length(f0_len, input.frame_period_ms, input.fs);
        let rust_y = synthesis(
            &input.f0,
            f0_len,
            &input.spectrogram,
            &ap,
            input.fft_size,
            input.frame_period_ms,
            input.fs,
            y_len,
        )
        .expect("synthesis");
        let cpp_y = cpp_synthesis_vec_const_ap(name, 0.0);
        assert_eq!(rust_y.len(), cpp_y.len(), "{name}: length mismatch");
        let err = max_abs(&rust_y, &cpp_y);
        assert!(err < 1e-6, "{name} AP=0.0 max abs error {err} >= 1e-6");
    }
}

#[test]
fn constant_ap_half() {
    for name in TEST_NAMES {
        let input = read_synthesis_in(name);
        let f0_len = input.f0.len();
        let bins = input.fft_size / 2 + 1;
        let ap = vec![vec![0.5; bins]; f0_len];
        let y_len = get_y_length(f0_len, input.frame_period_ms, input.fs);
        let rust_y = synthesis(
            &input.f0,
            f0_len,
            &input.spectrogram,
            &ap,
            input.fft_size,
            input.frame_period_ms,
            input.fs,
            y_len,
        )
        .expect("synthesis");
        let cpp_y = cpp_synthesis_vec_const_ap(name, 0.5);
        assert_eq!(rust_y.len(), cpp_y.len(), "{name}: length mismatch");
        let err = max_abs(&rust_y, &cpp_y);
        assert!(err < 1e-6, "{name} AP=0.5 max abs error {err} >= 1e-6");
    }
}
