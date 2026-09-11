#![allow(clippy::unwrap_used)]
#![allow(clippy::panic)]

use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use criterion::{black_box, Criterion};
use world_rs::synthesis::{get_y_length, synthesis};

fn vectors_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test-vector-data/vectors/synthesis")
}

fn read_f64s(bytes: &[u8]) -> Vec<f64> {
    let (chunks, _remainder) = bytes.as_chunks::<8>();
    chunks.iter().map(|c| f64::from_le_bytes(*c)).collect()
}

struct SynthCase {
    f0: Vec<f64>,
    fs: f64,
    frame_period_ms: f64,
    fft_size: usize,
    spectrogram: Vec<Vec<f64>>,
    aperiodicity: Vec<Vec<f64>>,
    y_length: usize,
}

fn load_synth(name: &str) -> SynthCase {
    let in_path = vectors_dir().join(format!("{name}.in"));
    let in_bytes = fs::read(&in_path).unwrap();
    let f0_length = u32::from_le_bytes(in_bytes[0..4].try_into().unwrap()) as usize;
    let fft_size = u32::from_le_bytes(in_bytes[4..8].try_into().unwrap()) as usize;
    let frame_period_ms = f64::from_le_bytes(in_bytes[8..16].try_into().unwrap());
    let fs = f64::from_le_bytes(in_bytes[16..24].try_into().unwrap());
    let half = fft_size / 2 + 1;

    let f0_span = 24..24 + f0_length * 8;
    let sp_span = f0_span.end..f0_span.end + f0_length * half * 8;
    let ap_span = sp_span.end..sp_span.end + f0_length * half * 8;

    let f0 = read_f64s(&in_bytes[f0_span]);
    let spectrogram: Vec<Vec<f64>> = read_f64s(&in_bytes[sp_span])
        .chunks(half)
        .map(|c| c.to_vec())
        .collect();
    let aperiodicity: Vec<Vec<f64>> = read_f64s(&in_bytes[ap_span])
        .chunks(half)
        .map(|c| c.to_vec())
        .collect();

    let y_length = get_y_length(f0_length, frame_period_ms, fs);

    SynthCase {
        f0,
        fs,
        frame_period_ms,
        fft_size,
        spectrogram,
        aperiodicity,
        y_length,
    }
}

fn main() {
    let case = load_synth("chirp_16k");

    let mut criterion = Criterion::default();

    let mut group = criterion.benchmark_group("synthesis_1s");
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_millis(250));

    let f0 = case.f0;
    let f0_length = f0.len();
    let spectrogram = case.spectrogram;
    let aperiodicity = case.aperiodicity;
    let fft_size = case.fft_size;
    let frame_period_ms = case.frame_period_ms;
    let fs = case.fs;
    let y_length = case.y_length;

    group.bench_function("synthesis_chirp_16k", |b| {
        b.iter(|| {
            black_box(
                synthesis(
                    &f0,
                    f0_length,
                    &spectrogram,
                    &aperiodicity,
                    fft_size,
                    frame_period_ms,
                    fs,
                    y_length,
                )
                .expect("valid input"),
            )
        })
    });
    group.finish();
}
