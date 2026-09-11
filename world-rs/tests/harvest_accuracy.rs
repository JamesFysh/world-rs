// Test target: unwrap/panic in setup and assertions is expected.
#![allow(clippy::unwrap_used)]
#![allow(clippy::panic)]

use std::fs;
use std::path::PathBuf;
use world_rs::harvest::{harvest, initialize_harvest_option};

const TEST_NAMES: &[&str] = &[
    "harm_100",
    "harm_200",
    "harm_500",
    "harm_700",
    "harm_chirp",
    "speech_like",
    "noise",
    "silence",
];

fn vectors_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test-vector-data/vectors/harvest")
}

#[allow(clippy::chunks_exact_to_as_chunks)]
fn read_f64s(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

struct Reference {
    x: Vec<f64>,
    fs: f64,
    #[allow(dead_code)]
    frame_period: f64,
    f0: Vec<f64>,
    temporal_positions: Vec<f64>,
}

fn read_reference(name: &str) -> Reference {
    let dir = vectors_dir();
    let in_path = dir.join(format!("{name}.in"));
    let out_path = dir.join(format!("{name}.out"));
    let hint = "run `bash test-vector-data/generate-harvest-vectors.sh`";
    assert!(in_path.exists(), "missing {in_path:?}; {hint}");
    assert!(out_path.exists(), "missing {out_path:?}; {hint}");
    let in_bytes = fs::read(&in_path).unwrap();
    let x_length = u32::from_le_bytes(in_bytes[0..4].try_into().unwrap()) as usize;
    let fs = f64::from_le_bytes(in_bytes[4..12].try_into().unwrap());
    let x = read_f64s(&in_bytes[12..]);
    assert_eq!(x.len(), x_length);
    let out_bytes = fs::read(&out_path).unwrap();
    let f0_length = u32::from_le_bytes(out_bytes[0..4].try_into().unwrap()) as usize;
    let frame_period = f64::from_le_bytes(out_bytes[4..12].try_into().unwrap());
    let f0 = read_f64s(&out_bytes[12..12 + f0_length * 8]);
    let temporal_positions = read_f64s(&out_bytes[12 + f0_length * 8..]);
    Reference {
        x,
        fs,
        frame_period,
        f0,
        temporal_positions,
    }
}

fn f0_rmse(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let se: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    (se / a.len() as f64).sqrt()
}

fn voicing_accuracy(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let matches = a
        .iter()
        .zip(b)
        .filter(|(x, y)| (**x > 0.0) == (**y > 0.0))
        .count();
    matches as f64 / a.len() as f64
}

fn run_case(name: &str) -> (f64, f64) {
    let reference = read_reference(name);
    // Process guard: a degenerate reference must fail loudly, never pass
    // vacuously (pure-tone refs voiced ~nothing — see HARVEST-fixture-redesign).
    let ref_voiced = reference.f0.iter().filter(|v| **v > 0.0).count();
    if name == "noise" || name == "silence" {
        assert!(
            ref_voiced * 20 < reference.f0.len(),
            "{name}: regressed generator voicing an unvoiced reference"
        );
    } else {
        assert!(
            ref_voiced * 2 > reference.f0.len(),
            "{name}: degenerate reference ({ref_voiced} voiced)"
        );
    }
    let option = initialize_harvest_option();
    let result = harvest(&reference.x, reference.fs, &option).expect("harvest failed");
    assert_eq!(result.f0.len(), reference.f0.len(), "{name} f0 length");
    assert_eq!(result.temporal_positions.len(), reference.f0.len());
    for i in 0..result.f0.len() {
        assert!(
            (result.temporal_positions[i] - reference.temporal_positions[i]).abs() < 1e-9,
            "{name} temporal mismatch"
        );
    }
    if name == "harm_200" {
        let n = result.f0.len();
        println!("{} len {}", name, n);
        println!("{} first 10 f0 result: {:?}", name, &result.f0[..10.min(n)]);
        println!(
            "{} first 10 f0 ref:    {:?}",
            name,
            &reference.f0[..10.min(n)]
        );
        println!(
            "{} last 10 f0 result: {:?}",
            name,
            &result.f0[n - 10.min(n)..]
        );
        println!(
            "{} last 10 f0 ref:    {:?}",
            name,
            &reference.f0[n - 10.min(n)..]
        );
    }
    let rmse = f0_rmse(&result.f0, &reference.f0);
    let voicing = voicing_accuracy(&result.f0, &reference.f0);
    (rmse, voicing)
}

/// Parity gate (31s): `#[ignore]`d for fast edit-compile-run cycles;
/// run explicitly with `cargo test -- --ignored`.
#[test]
#[ignore]
fn harvest_f0_rmse_below_1hz() {
    for name in TEST_NAMES {
        let (rmse, voicing) = run_case(name);
        println!("{} RMSE {:.6} voicing {:.6}", name, rmse, voicing);
        assert!(rmse < 1.0, "{name}: RMSE {rmse} >= 1 Hz");
    }
}

/// Parity gate (31s): see `harvest_f0_rmse_below_1hz`.
#[test]
#[ignore]
fn harvest_voicing_accuracy_above_99_percent() {
    for name in TEST_NAMES {
        let (_, voicing) = run_case(name);
        assert!(voicing >= 0.99, "{name}: voicing {voicing} < 0.99");
    }
}

#[test]
fn harvest_determinism() {
    let name = "harm_200";
    let reference = read_reference(name);
    let option = initialize_harvest_option();
    let result1 = harvest(&reference.x, reference.fs, &option).expect("harvest failed");
    let result2 = harvest(&reference.x, reference.fs, &option).expect("harvest failed");
    assert_eq!(result1.f0, result2.f0, "f0 not deterministic");
    assert_eq!(
        result1.temporal_positions, result2.temporal_positions,
        "temporal_positions not deterministic"
    );
}
