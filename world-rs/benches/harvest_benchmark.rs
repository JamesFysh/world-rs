#![allow(clippy::unwrap_used)]
#![allow(clippy::panic)]

use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use criterion::{black_box, BenchmarkId, Criterion};
use world_rs_core::constants::K_PI;
use world_rs_core::dio::{dio, initialize_dio_option};
use world_rs_core::harvest::{harvest, initialize_harvest_option};
use world_rs_core::stonemask::stone_mask;

const FS: f64 = 16000.0;
const N: usize = 16000;
const PI: f64 = K_PI;

fn vectors_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test-vector-data/vectors")
}

fn read_f64s(bytes: &[u8]) -> Vec<f64> {
    let (chunks, _remainder) = bytes.as_chunks::<8>();
    chunks.iter().map(|c| f64::from_le_bytes(*c)).collect()
}

fn gen_speech(n: usize) -> Vec<f64> {
    let mut phase = 0.0;
    (0..n)
        .map(|i| {
            let t = i as f64 / FS;
            let f0 = 150.0 + 30.0 * (2.0 * PI * 0.5 * t).sin();
            phase += 2.0 * PI * f0 / FS;
            let mut v = 0.0;
            for h in 1..=8 {
                v += (h as f64 * phase).sin() / h as f64;
            }
            let env = 0.5 + 0.5 * (2.0 * PI * 3.0 * t).sin();
            0.5 * env * v
        })
        .collect()
}

fn load_harvest_signal() -> Vec<f64> {
    let path = vectors_dir().join("harvest/speech_like.in");
    if path.exists() {
        let bytes = fs::read(&path).unwrap();
        let x_length = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
        let _fs = f64::from_le_bytes(bytes[4..12].try_into().unwrap());
        let samples = read_f64s(&bytes[12..]);
        let target_len = N.min(x_length);
        return samples.into_iter().take(target_len).collect();
    }
    gen_speech(N)
}

fn load_dio_speech() -> (Vec<f64>, f64, usize) {
    let path = vectors_dir().join("dio/speech.in");
    let bytes = fs::read(&path).unwrap();
    let x_length = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let fs = f64::from_le_bytes(bytes[4..12].try_into().unwrap());
    let samples = read_f64s(&bytes[12..]);
    (samples, fs, x_length)
}

fn load_dio_f0() -> (Vec<f64>, f64) {
    let path = vectors_dir().join("dio/speech.out");
    let bytes = fs::read(&path).unwrap();
    let f0_length = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let frame_period = f64::from_le_bytes(bytes[4..12].try_into().unwrap());
    let f0 = read_f64s(&bytes[12..12 + f0_length * 8]);
    (f0, frame_period)
}

fn main() {
    let harvest_option = initialize_harvest_option();
    let dio_option = initialize_dio_option();

    let mut criterion = Criterion::default();

    // harvest_1s
    let harvest_input = load_harvest_signal();
    let mut group = criterion.benchmark_group("harvest_1s");
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_millis(250));
    group.bench_with_input(
        BenchmarkId::new("harvest", "speech_like"),
        &harvest_input,
        |b, input| {
            b.iter(|| black_box(harvest(input, FS, &harvest_option).unwrap()));
        },
    );
    group.finish();

    // stonemask_contour
    let (x, fs, x_length) = load_dio_speech();
    let (f0, frame_period) = load_dio_f0();
    let f0_length = f0.len();
    let temporal_positions: Vec<f64> = (0..f0_length)
        .map(|i| i as f64 * frame_period / 1000.0)
        .collect();

    let mut group = criterion.benchmark_group("stonemask_contour");
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_millis(250));
    group.bench_function("stone_mask", |b| {
        b.iter(|| {
            black_box(stone_mask(&x, x_length, fs, &temporal_positions, &f0, f0_length).unwrap())
        });
    });
    group.finish();

    // dio_frame_micro - per-frame path micro bench
    let dio_input = load_harvest_signal();
    let mut group = criterion.benchmark_group("dio_frame_micro");
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_millis(250));
    group.bench_with_input(
        BenchmarkId::new("dio", "frame_micro"),
        &dio_input,
        |b, input| {
            b.iter(|| black_box(dio(input, FS, &dio_option).unwrap()));
        },
    );
    group.finish();
}
