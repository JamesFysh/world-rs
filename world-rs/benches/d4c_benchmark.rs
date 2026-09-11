#![allow(clippy::unwrap_used)]
#![allow(clippy::panic)]

use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use criterion::{black_box, Criterion};
use world_rs::d4c::{d4c, initialize_d4c_option};

fn vectors_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test-vector-data/vectors/d4c")
}

fn read_f64s(bytes: &[u8]) -> Vec<f64> {
    let (chunks, _remainder) = bytes.as_chunks::<8>();
    chunks.iter().map(|c| f64::from_le_bytes(*c)).collect()
}

struct D4cBenchInput {
    x: Vec<f64>,
    fs: f64,
    f0: Vec<f64>,
    temporal: Vec<f64>,
    fft_size: i32,
}

fn load_d4c_input() -> D4cBenchInput {
    let in_path = vectors_dir().join("speech_clean.in");
    let out_path = vectors_dir().join("speech_clean.out");

    let in_bytes = fs::read(&in_path).unwrap();
    let x_length = u32::from_le_bytes(in_bytes[0..4].try_into().unwrap()) as usize;
    let fs = f64::from_le_bytes(in_bytes[4..12].try_into().unwrap());
    let x = read_f64s(&in_bytes[12..]);
    assert_eq!(x.len(), x_length);

    let out_bytes = fs::read(&out_path).unwrap();
    let f0_length = u32::from_le_bytes(out_bytes[0..4].try_into().unwrap()) as usize;
    let fft_size_out = u32::from_le_bytes(out_bytes[4..8].try_into().unwrap()) as i32;
    let f0 = read_f64s(&out_bytes[8..8 + f0_length * 8]);
    let temporal = read_f64s(&out_bytes[8 + f0_length * 8..8 + 2 * f0_length * 8]);

    // Select first 20 voiced frames
    let mut f0_sub = Vec::new();
    let mut temporal_sub = Vec::new();
    for (&f, &t) in f0.iter().zip(temporal.iter()) {
        if f > 0.0 {
            f0_sub.push(f);
            temporal_sub.push(t);
            if f0_sub.len() >= 20 {
                break;
            }
        }
    }

    D4cBenchInput {
        x,
        fs,
        f0: f0_sub,
        temporal: temporal_sub,
        fft_size: fft_size_out,
    }
}

fn main() {
    let data = load_d4c_input();
    let option = initialize_d4c_option();

    let mut criterion = Criterion::default();

    let mut group = criterion.benchmark_group("d4c_frames");
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_millis(250));

    let x = data.x;
    let fs = data.fs;
    let f0 = data.f0;
    let temporal = data.temporal;
    let fft_size = data.fft_size;

    group.bench_function("d4c_20frames", |b| {
        b.iter(|| black_box(d4c(&x, fs, &temporal, &f0, fft_size, &option).expect("valid input")))
    });
    group.finish();
}
