//! Micro-benchmarks for the small scalar helpers in `common` (Phase 6).
//!
//! Lightweight criterion benchmarks used to confirm that refactors of the
//! helpers are performance-neutral. The heavy real-time DSP benchmarks live in
//! `dio_benchmark` / `cheaptrick_benchmark`.

use std::time::Duration;

use criterion::{black_box, BatchSize, Criterion};
use world_rs_core::common::{
    dc_correction_into, dc_correction_scratch_capacity, get_suitable_fft_size,
    linear_smoothing_into, linear_smoothing_scratch_capacity, nuttall_window,
};

/// The pre-optimization 3-cos implementation, inlined so we can benchmark
/// before/after without reverting the source.
fn nuttall_window_direct_cos(y_length: usize, y: &mut [f64]) {
    if y_length < 2 {
        return;
    }
    for (n, slot) in y[..y_length].iter_mut().enumerate() {
        let phase = 2.0 * std::f64::consts::PI * n as f64 / (y_length as f64 - 1.0);
        *slot = 0.355768 - 0.487396 * phase.cos() + 0.144232 * (2.0 * phase).cos()
            - 0.012604 * (3.0 * phase).cos();
    }
}

fn dc_correction_benchmark(c: &mut Criterion) {
    let fs = 16000;
    let mut group = c.benchmark_group("dc_correction");
    group.measurement_time(Duration::from_millis(500));
    group.warm_up_time(Duration::from_millis(100));
    for &fft_size in &[1024usize, 2048] {
        for &f0 in &[100.0, 400.0] {
            let input_len = fft_size / 2 + 1;
            group.bench_function(format!("{fft_size}/f0{f0}"), |b| {
                b.iter_batched(
                    || {
                        let input: Vec<f64> =
                            (0..input_len).map(|i| 1.0 + 0.01 * i as f64).collect();
                        let output = vec![0.0f64; input_len];
                        let scratch =
                            vec![0.0f64; dc_correction_scratch_capacity(fft_size, input_len)];
                        (input, output, scratch)
                    },
                    |(input, mut output, mut scratch)| {
                        dc_correction_into(&input, f0, fs, fft_size, &mut output, &mut scratch);
                        black_box(output)
                    },
                    BatchSize::LargeInput,
                )
            });
        }
    }
    group.finish();
}

fn linear_smoothing_benchmark(c: &mut Criterion) {
    let fs = 16000;
    let mut group = c.benchmark_group("linear_smoothing");
    group.measurement_time(Duration::from_millis(500));
    group.warm_up_time(Duration::from_millis(100));
    for &fft_size in &[1024usize, 2048] {
        for &width in &[100.0, 400.0] {
            let input_len = fft_size / 2 + 1;
            group.bench_function(format!("{fft_size}/w{width}"), |b| {
                b.iter_batched(
                    || {
                        let input: Vec<f64> =
                            (0..input_len).map(|i| 1.0 + 0.01 * i as f64).collect();
                        let output = vec![0.0f64; input_len];
                        let scratch = vec![0.0f64; linear_smoothing_scratch_capacity(fft_size)];
                        (input, output, scratch)
                    },
                    |(input, mut output, mut scratch)| {
                        linear_smoothing_into(
                            &input,
                            width,
                            fs,
                            fft_size,
                            &mut output,
                            &mut scratch,
                        );
                        black_box(output)
                    },
                    BatchSize::LargeInput,
                )
            });
        }
    }
    group.finish();
}

fn main() {
    let mut criterion = Criterion::default();
    let mut group = criterion.benchmark_group("common");
    group.measurement_time(Duration::from_secs(1));
    group.warm_up_time(Duration::from_millis(100));

    // A spread of realistic FFT-ish sizes plus the power-of-two boundaries.
    let sizes = [1usize, 127, 128, 129, 1024, 2048];
    for &n in &sizes {
        group.bench_function(format!("get_suitable_fft_size/{n}"), |b| {
            // black_box the input too, so the compiler can't constant-fold the
            // call for each fixed size (which would measure a load, not the fn).
            b.iter(|| black_box(get_suitable_fft_size(black_box(n))))
        });
    }
    group.finish();

    let mut group = criterion.benchmark_group("nuttall_window");
    group.measurement_time(Duration::from_millis(500));
    group.warm_up_time(Duration::from_millis(100));

    for &len in &[256usize, 1024, 2048] {
        group.bench_function(format!("triple_angle/{len}"), |b| {
            b.iter_batched(
                || vec![0.0f64; len],
                |mut buf| {
                    nuttall_window(len, &mut buf);
                    black_box(buf)
                },
                BatchSize::SmallInput,
            )
        });
        group.bench_function(format!("direct_cos/{len}"), |b| {
            b.iter_batched(
                || vec![0.0f64; len],
                |mut buf| {
                    nuttall_window_direct_cos(len, &mut buf);
                    black_box(buf)
                },
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();

    dc_correction_benchmark(&mut criterion);
    linear_smoothing_benchmark(&mut criterion);
}
