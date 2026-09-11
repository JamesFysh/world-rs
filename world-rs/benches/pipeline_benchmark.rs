//! End-to-end pipeline benchmark driven by the committed test vectors.
//!
//! Loads the exact C++/JS reference inputs (`test-vector-data/vectors/{dio,cheaptrick,synthesis,resample}`)
//! and runs every user-facing API function on each: `dio`, `stone_mask`,
//! `cheaptrick`, `synthesis`, and `resample`, plus the full `dio() + cheaptrick()`
//! analysis chain. Because each vector is a full second of audio (or the full
//! contour / waveform), the per-frame low-level helpers are exercised ~1000 times
//! per run: `nuttall_window` inside DIO and `dc_correction` / `linear_smoothing`
//! inside CheapTrick. This captures the aggregate impact of changes to those
//! helpers at the API level, on the identical data the accuracy tests validate.
//!
//! Run: `cargo bench -p world-rs --bench pipeline_benchmark`.

// Benchmark target: unwrap/panic in setup is expected.
#![allow(clippy::unwrap_used)]
#![allow(clippy::panic)]

use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use criterion::{black_box, BenchmarkId, Criterion};
use world_rs::cheaptrick::{cheaptrick, initialize_cheaptrick_option};
use world_rs::dio::{dio, initialize_dio_option};
use world_rs::resample::resample;
use world_rs::stonemask::stone_mask;
use world_rs::synthesis::{get_y_length, synthesis};

/// DIO vectors (also used as the input contour for `stone_mask`).
const DIO_NAMES: &[&str] = &[
    "sine_71", "sine_200", "sine_500", "sine_800", "chirp", "speech", "noise", "silence",
];

/// CheapTrick vectors (each carries a C++ reference F0 contour + temporal grid).
const CT_NAMES: &[&str] = &[
    "sine_440_16k",
    "sine_440_48k",
    "unvoiced_16k",
    "transition_16k",
    "chirp_16k",
];

/// Synthesis vectors.
const SYNTH_NAMES: &[&str] = &[
    "sine_220_16k",
    "chirp_16k",
    "jump_16k",
    "transition_16k",
    "unvoiced_16k",
    "silence_16k",
    "sine_440_48k",
];

/// Resample vectors.
const RESAMPLE_NAMES: &[&str] = &[
    "sine440_16k_to_48k",
    "sine440_48k_to_16k",
    "sine440_44100_to_48000",
    "maxamp_16k_to_48k",
    "swept_48k_to_16k",
];

fn vectors_dir(sub: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("../../test-vector-data/vectors/{sub}"))
}

#[allow(clippy::chunks_exact_to_as_chunks)]
fn read_f64s(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn read_f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn read_in(sub: &str, name: &str) -> Vec<u8> {
    let dir = vectors_dir(sub);
    let hint = format!("run `bash test-vector-data/generate-{sub}-vectors.sh` to generate them");
    fs::read(dir.join(format!("{name}.in"))).unwrap_or_else(|e| panic!("{name}.in: {e}; {hint}"))
}

fn read_out(sub: &str, name: &str) -> Vec<u8> {
    let dir = vectors_dir(sub);
    let hint = format!("run `bash test-vector-data/generate-{sub}-vectors.sh` to generate them");
    fs::read(dir.join(format!("{name}.out"))).unwrap_or_else(|e| panic!("{name}.out: {e}; {hint}"))
}

/// A DIO case: the input signal (`.in`) plus the C++ reference F0 contour and
/// temporal grid (`.out`), which also serve as the `stone_mask` input.
struct DioCase {
    x: Vec<f64>,
    fs: f64,
    f0: Vec<f64>,
    temporal: Vec<f64>,
}

fn load_dio(name: &str) -> DioCase {
    let in_bytes = read_in("dio", name);
    let x_length = u32::from_le_bytes(in_bytes[0..4].try_into().unwrap()) as usize;
    let fs = f64::from_le_bytes(in_bytes[4..12].try_into().unwrap());
    let x = read_f64s(&in_bytes[12..]);
    assert_eq!(x.len(), x_length, "{name}: input length mismatch");

    // <name>.out: u32 f0_length, f64 frame_period, f0[f0_length], temporal[f0_length]
    let out_bytes = read_out("dio", name);
    let f0_length = u32::from_le_bytes(out_bytes[0..4].try_into().unwrap()) as usize;
    let f0 = read_f64s(&out_bytes[12..12 + f0_length * 8]);
    let temporal = read_f64s(&out_bytes[12 + f0_length * 8..12 + 2 * f0_length * 8]);
    assert_eq!(f0.len(), f0_length, "{name}: f0 length mismatch");
    assert_eq!(
        temporal.len(),
        f0_length,
        "{name}: temporal length mismatch"
    );

    DioCase {
        x,
        fs,
        f0,
        temporal,
    }
}

/// A CheapTrick case: input signal (`.in`) plus the C++ reference F0 contour /
/// temporal grid (`.out`).
struct CtCase {
    x: Vec<f64>,
    fs: f64,
    f0: Vec<f64>,
    temporal: Vec<f64>,
}

fn load_ct(name: &str) -> CtCase {
    let in_bytes = read_in("cheaptrick", name);
    let x_length = u32::from_le_bytes(in_bytes[0..4].try_into().unwrap()) as usize;
    let fs = f64::from_le_bytes(in_bytes[4..12].try_into().unwrap());
    let x = read_f64s(&in_bytes[12..]);
    assert_eq!(x.len(), x_length, "{name}: input length mismatch");

    // <name>.out: u32 f0_length, u32 fft_size, f0[f0_length], temporal[f0_length], spectrogram
    let out_bytes = read_out("cheaptrick", name);
    let f0_length = u32::from_le_bytes(out_bytes[0..4].try_into().unwrap()) as usize;
    let f0 = read_f64s(&out_bytes[8..8 + f0_length * 8]);
    let temporal = read_f64s(&out_bytes[8 + f0_length * 8..8 + 2 * f0_length * 8]);
    assert_eq!(f0.len(), f0_length, "{name}: f0 length mismatch");
    assert_eq!(
        temporal.len(),
        f0_length,
        "{name}: temporal length mismatch"
    );

    CtCase {
        x,
        fs,
        f0,
        temporal,
    }
}

/// A Synthesis case: the full synthesis input (all from `.in`).
struct SynthCase {
    f0: Vec<f64>,
    fs: f64,
    frame_period_ms: f64,
    fft_size: usize,
    spectrogram: Vec<Vec<f64>>,
    aperiodicity: Vec<Vec<f64>>,
}

fn load_synth(name: &str) -> SynthCase {
    let in_bytes = read_in("synthesis", name);
    let f0_length = u32::from_le_bytes(in_bytes[0..4].try_into().unwrap()) as usize;
    let fft_size = u32::from_le_bytes(in_bytes[4..8].try_into().unwrap()) as usize;
    let frame_period_ms = f64::from_le_bytes(in_bytes[8..16].try_into().unwrap());
    let fs = f64::from_le_bytes(in_bytes[16..24].try_into().unwrap());
    let half = fft_size / 2 + 1;

    let f0_span = 24..24 + f0_length * 8;
    let sp_span = f0_span.end..f0_span.end + f0_length * half * 8;
    let ap_span = sp_span.end..sp_span.end + f0_length * half * 8;
    let f0 = read_f64s(&in_bytes[f0_span.clone()]);
    let spectrogram: Vec<Vec<f64>> = read_f64s(&in_bytes[sp_span.clone()])
        .chunks(half)
        .map(|c| c.to_vec())
        .collect();
    let aperiodicity: Vec<Vec<f64>> = read_f64s(&in_bytes[ap_span.clone()])
        .chunks(half)
        .map(|c| c.to_vec())
        .collect();
    assert_eq!(f0.len(), f0_length, "{name}: f0 length mismatch");
    assert_eq!(
        spectrogram.len(),
        f0_length,
        "{name}: spectrogram frame count mismatch"
    );
    assert_eq!(
        aperiodicity.len(),
        f0_length,
        "{name}: aperiodicity frame count mismatch"
    );

    SynthCase {
        f0,
        fs,
        frame_period_ms,
        fft_size,
        spectrogram,
        aperiodicity,
    }
}

/// A Resample case: the f32 input PCM (`.in`) plus the from/to rates (`.out`).
struct ResampleCase {
    from: u32,
    to: u32,
    x: Vec<f32>,
}

fn load_resample(name: &str) -> ResampleCase {
    let in_bytes = read_in("resample", name);
    let n = u32::from_le_bytes(in_bytes[0..4].try_into().unwrap()) as usize;
    let x = read_f32s(&in_bytes[4..]);
    assert_eq!(x.len(), n, "{name}: input length mismatch");

    // <name>.out: u32 from, u32 to, u32 m, m * f32 reference output
    let out_bytes = read_out("resample", name);
    let from = u32::from_le_bytes(out_bytes[0..4].try_into().unwrap());
    let to = u32::from_le_bytes(out_bytes[4..8].try_into().unwrap());

    ResampleCase { from, to, x }
}

fn main() {
    let dio_option = initialize_dio_option();

    // Pre-warm the CPU to a steady (boosted) frequency with a short burst of
    // representative work before any samples are collected.
    {
        let warm = load_dio("speech");
        for _ in 0..20 {
            let _ = dio(&warm.x, warm.fs, &dio_option).expect("valid input");
        }
    }

    let mut criterion = Criterion::default();

    // DIO: exercises nuttall_window (get_filtered_signal) once per frame.
    {
        let mut group = criterion.benchmark_group("pipeline_dio");
        group.measurement_time(Duration::from_secs(1));
        group.warm_up_time(Duration::from_millis(250));
        for name in DIO_NAMES {
            let case = load_dio(name);
            let x = case.x;
            let fs = case.fs;
            group.bench_with_input(BenchmarkId::new("dio", name), &x, |b, x| {
                b.iter(|| black_box(dio(x, fs, &dio_option)).expect("valid input"));
            });
        }
        group.finish();
    }

    // StoneMask: refines the DIO F0 contour (one call over the full contour).
    {
        let mut group = criterion.benchmark_group("pipeline_stonemask");
        group.measurement_time(Duration::from_secs(1));
        group.warm_up_time(Duration::from_millis(250));
        for name in DIO_NAMES {
            let case = load_dio(name);
            let x = case.x;
            let fs = case.fs;
            let f0 = case.f0;
            let temporal = case.temporal;
            let x_length = x.len();
            let f0_length = f0.len();
            group.bench_with_input(BenchmarkId::new("stone_mask", name), &x, |b, x| {
                b.iter(|| black_box(stone_mask(x, x_length, fs, &temporal, &f0, f0_length)));
            });
        }
        group.finish();
    }

    // CheapTrick: exercises dc_correction + linear_smoothing once per frame.
    {
        let mut group = criterion.benchmark_group("pipeline_cheaptrick");
        group.measurement_time(Duration::from_secs(1));
        group.warm_up_time(Duration::from_millis(250));
        for name in CT_NAMES {
            let case = load_ct(name);
            let ct_option = initialize_cheaptrick_option(case.fs);
            let x = case.x;
            let fs = case.fs;
            let f0 = case.f0;
            let temporal = case.temporal;
            group.bench_with_input(BenchmarkId::new("cheaptrick", name), &x, |b, x| {
                b.iter(|| {
                    black_box(cheaptrick(x, fs, &temporal, &f0, &ct_option)).expect("valid input")
                });
            });
        }
        group.finish();
    }

    // Synthesis: waveform generation from the C++ reference (f0, SP, AP).
    {
        let mut group = criterion.benchmark_group("pipeline_synthesis");
        group.measurement_time(Duration::from_secs(1));
        group.warm_up_time(Duration::from_millis(250));
        for name in SYNTH_NAMES {
            let case = load_synth(name);
            let y_length = get_y_length(case.f0.len(), case.frame_period_ms, case.fs);
            let f0 = case.f0;
            let fs = case.fs;
            let frame_period_ms = case.frame_period_ms;
            let fft_size = case.fft_size;
            let spectrogram = case.spectrogram;
            let aperiodicity = case.aperiodicity;
            let f0_length = f0.len();
            group.bench_with_input(BenchmarkId::new("synthesis", name), &f0, |b, f0| {
                b.iter(|| {
                    black_box(synthesis(
                        f0,
                        f0_length,
                        &spectrogram,
                        &aperiodicity,
                        fft_size,
                        frame_period_ms,
                        fs,
                        y_length,
                    ))
                    .expect("valid input")
                });
            });
        }
        group.finish();
    }

    // Resample: polyphase resampling (f32 I/O).
    {
        let mut group = criterion.benchmark_group("pipeline_resample");
        group.measurement_time(Duration::from_secs(1));
        group.warm_up_time(Duration::from_millis(250));
        for name in RESAMPLE_NAMES {
            let case = load_resample(name);
            let x = case.x;
            let from = case.from;
            let to = case.to;
            group.bench_with_input(BenchmarkId::new("resample", name), &x, |b, x| {
                b.iter(|| black_box(resample(x, from, to)));
            });
        }
        group.finish();
    }

    // Full analysis chain (DIO -> CheapTrick): the total end-to-end analysis.
    {
        let mut group = criterion.benchmark_group("pipeline_chain");
        group.measurement_time(Duration::from_secs(1));
        group.warm_up_time(Duration::from_millis(250));
        for name in CT_NAMES {
            let case = load_ct(name);
            let ct_option = initialize_cheaptrick_option(case.fs);
            let x = case.x;
            let fs = case.fs;
            group.bench_with_input(BenchmarkId::new("integration", name), &x, |b, x| {
                b.iter(|| {
                    let dr = dio(x, fs, &dio_option).expect("valid input");
                    black_box(cheaptrick(
                        x,
                        fs,
                        &dr.temporal_positions,
                        &dr.f0,
                        &ct_option,
                    ))
                    .expect("valid input")
                });
            });
        }
        group.finish();
    }
}
