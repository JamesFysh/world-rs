//! CheapTrick performance benchmark (Phase 2-8).
//!
//! Uses `criterion` to measure the real-time performance of [`cheaptrick`] (and
//! the full DIO -> CheapTrick integration) on one second of 16 kHz audio for a
//! set of deterministic test signals. For each signal the F0 contour is first
//! extracted once with [`dio`] (setup, not benchmarked); the benchmark then
//! times `cheaptrick()` over that contour, and separately times the full
//! `dio() + cheaptrick()` chain.
//!
//! The real-time requirement is RTF < 0.1 for 16 kHz audio, i.e. one second of
//! audio must be processed in under 100 ms. After the criterion run, the
//! harness (re)generates the summary report at
//! `docs/benchmarks/cheaptrick_benchmark_results.md`.
//
// Benchmark target: unwrap/panic in setup is expected.
#![allow(clippy::unwrap_used)]
#![allow(clippy::panic)]

use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use criterion::{black_box, BenchmarkId, Criterion};
use world_rs::cheaptrick::{cheaptrick, initialize_cheaptrick_option};
use world_rs::dio::{dio, initialize_dio_option};

const FS: f64 = 16000.0;
const N: usize = 16000; // 1 second
const PI: f64 = std::f64::consts::PI;

/// Deterministic 16 kHz test signals (mirrors the DIO benchmark set).
const TEST_NAMES: &[&str] = &["sine_200", "sine_500", "chirp", "speech"];

// ---- Signal generation (mirrors test-vector-data/dio_reference.cpp) ----

fn gen_sine(n: usize, freq: f64) -> Vec<f64> {
    (0..n)
        .map(|i| (2.0 * PI * freq * i as f64 / FS).sin())
        .collect()
}

fn gen_chirp(n: usize) -> Vec<f64> {
    let (f_start, f_end) = (100.0, 600.0);
    let mut phase = 0.0;
    (0..n)
        .map(|i| {
            let t = i as f64 / FS;
            let f = f_start + (f_end - f_start) * t / (n as f64 / FS);
            phase += 2.0 * PI * f / FS;
            phase.sin()
        })
        .collect()
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

fn load_signal(name: &str) -> Vec<f64> {
    match name {
        "sine_200" => gen_sine(N, 200.0),
        "sine_500" => gen_sine(N, 500.0),
        "chirp" => gen_chirp(N),
        "speech" => gen_speech(N),
        _ => panic!("unknown signal {name}"),
    }
}

/// Root of criterion's on-disk results for this benchmark.
fn criterion_dir() -> PathBuf {
    if let Ok(td) = std::env::var("CARGO_TARGET_DIR") {
        return PathBuf::from(td).join("criterion");
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/criterion")
}

/// Reads the criterion median (ms) for a benchmark from its on-disk JSON.
fn criterion_median_ms(group: &str, bench: &str, name: &str) -> Option<f64> {
    let path = criterion_dir()
        .join(group)
        .join(bench)
        .join(name)
        .join("new")
        .join("estimates.json");
    let text = fs::read_to_string(path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    let ns = v.get("median")?.get("point_estimate")?.as_f64()?;
    Some(ns / 1e6)
}

struct Row {
    name: &'static str,
    duration_s: f64,
    cheaptrick_ms: Option<f64>,
    integration_ms: Option<f64>,
}

fn cpu_model() -> String {
    if let Ok(info) = fs::read_to_string("/proc/cpuinfo") {
        for line in info.lines() {
            if let Some(rest) = line.strip_prefix("model name") {
                if let Some(value) = rest.split(':').nth(1) {
                    return value.trim().to_string();
                }
            }
        }
    }
    String::from("unknown")
}

fn fmt_opt(v: Option<f64>, digits: usize) -> String {
    match v {
        Some(x) => format!("{x:.prec$}", prec = digits),
        None => String::from("n/a"),
    }
}

/// Builds the per-signal report rows and writes the markdown summary. Called
/// after the criterion run so the processing times reflect criterion's own
/// median measurements.
fn write_report() {
    let mut rows = Vec::new();
    for name in TEST_NAMES {
        let input = load_signal(name);
        let duration_s = input.len() as f64 / FS;
        rows.push(Row {
            name,
            duration_s,
            cheaptrick_ms: criterion_median_ms("cheaptrick_1s", "cheaptrick", name),
            integration_ms: criterion_median_ms("cheaptrick_1s", "integration", name),
        });
    }

    // Typical (median) CheapTrick processing time across signals, and the worst
    // single-signal value, for the RTF verdict.
    let times: Vec<f64> = rows.iter().filter_map(|r| r.cheaptrick_ms).collect();
    let (typical_ms, max_ms) = if times.is_empty() {
        (f64::NAN, f64::NAN)
    } else {
        let mut t = times.clone();
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (t[t.len() / 2], t.iter().cloned().fold(f64::MIN, f64::max))
    };
    // RTF = processing time / audio duration. For 1 s of audio this is
    // processing_ms / 1000.
    let rtf = typical_ms / 1000.0;
    let rtf_ok = rtf < 0.1;

    let mut md = String::new();
    md.push_str("# CheapTrick Benchmark Results\n\n");
    md.push_str("Phase 2-8 performance benchmark for the Rust CheapTrick port\n");
    md.push_str("(`crates/world-rs`). Generated by `cargo bench -p world-rs-core --bench cheaptrick_benchmark`.\n\n");
    md.push_str("## Environment\n\n");
    md.push_str(&format!("- CPU: {}\n", cpu_model()));
    md.push_str(&format!("- OS: {}\n", std::env::consts::OS));
    md.push_str("- Build: `cargo bench` (release profile, `opt-level = 3`, `lto = true`)\n");
    md.push_str("- Benchmark harness: criterion 0.5\n\n");
    md.push_str("## Results\n\n");
    md.push_str("Each signal is exactly 1 second of audio at 16 kHz. The F0 contour is\n");
    md.push_str("extracted once with `dio()` (setup, not benchmarked); `cheaptrick` is the\n");
    md.push_str("criterion median of `cheaptrick()` over that contour, and `integration` is\n");
    md.push_str("the median of the full `dio() + cheaptrick()` chain. RTF = processing time\n");
    md.push_str("/ audio duration (1 s), so the RTF < 0.1 requirement is a < 100 ms budget.\n\n");
    md.push_str("| Audio file | Duration (s) | CheapTrick (ms) | DIO+CheapTrick (ms) | RTF (cheaptrick) |\n");
    md.push_str("|------------|--------------|-----------------|---------------------|------------------|\n");
    for r in &rows {
        let rtf = r.cheaptrick_ms.map(|ms| ms / 1000.0);
        md.push_str(&format!(
            "| {} | {:.1} | {} | {} | {} |\n",
            r.name,
            r.duration_s,
            fmt_opt(r.cheaptrick_ms, 3),
            fmt_opt(r.integration_ms, 3),
            fmt_opt(rtf, 4),
        ));
    }
    md.push_str("\n## Real-time requirement\n\n");
    md.push_str(&format!(
        "Requirement: RTF < 0.1 (1 second of 16 kHz audio in < 100 ms). **{}** (typical/median CheapTrick processing time is {:.3} ms per second of audio, RTF = {:.4}).\n\n",
        if rtf_ok { "PASS" } else { "FAIL" },
        typical_ms,
        rtf
    ));
    md.push_str("Note: this is a single-threaded measurement on a thermally-managed APU\n");
    md.push_str("(powersave governor, 2-3 GHz). The per-second processing time varies with the\n");
    md.push_str(&format!(
        "core's thermal/frequency state (worst single-signal value this run: {:.3} ms).\n\n",
        max_ms
    ));
    md.push_str("## Hot-loop allocation notes\n\n");
    md.push_str("- The per-frame loop reuses the FFT plans, the spectral-envelope buffer, and\n");
    md.push_str(
        "  the DC-correction / linear-smoothing scratch buffers (pre-allocated once, sized\n",
    );
    md.push_str(
        "  for the Nyquist worst case), so no heap allocations occur on the hot path for\n",
    );
    md.push_str("  `f0 <= fs/2` (see `dc_correction_scratch_capacity` /\n");
    md.push_str("  `linear_smoothing_scratch_capacity`).\n");

    if std::env::var("WORLD_BENCH_REPORT").unwrap_or_default() == "1" {
        let out = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../docs/benchmarks/cheaptrick_benchmark_results.md");
        if let Some(parent) = out.parent() {
            let _ = fs::create_dir_all(parent);
        }
        if let Err(e) = fs::write(&out, md) {
            eprintln!("bench report write failed: {e}");
        } else {
            println!("\nWrote benchmark report to {}\n", out.display());
        }
    } else {
        println!("bench report suppressed (set WORLD_BENCH_REPORT=1 to write)");
    }
}

fn main() {
    let dio_option = initialize_dio_option();
    let ct_option = initialize_cheaptrick_option(FS);

    // Pre-warm the CPU so it reaches its boosted frequency before the first
    // benchmark is measured (a short burst of representative work).
    let warm = gen_sine(N, 200.0);
    let warm_dio = dio(&warm, FS, &dio_option).expect("valid input");
    for _ in 0..50 {
        let _ = cheaptrick(
            &warm,
            FS,
            &warm_dio.temporal_positions,
            &warm_dio.f0,
            &ct_option,
        )
        .expect("valid input");
    }

    let mut criterion = Criterion::default();
    let mut group = criterion.benchmark_group("cheaptrick_1s");
    group.measurement_time(Duration::from_secs(1));
    group.warm_up_time(Duration::from_millis(250));
    for name in TEST_NAMES {
        let input = load_signal(name);
        // Setup: extract the F0 contour once (not benchmarked).
        let dio_result = dio(&input, FS, &dio_option).expect("valid input");
        let f0 = dio_result.f0;
        let temporal_positions = dio_result.temporal_positions;

        group.bench_with_input(BenchmarkId::new("cheaptrick", name), &input, |b, input| {
            b.iter(|| {
                black_box(
                    cheaptrick(input, FS, &temporal_positions, &f0, &ct_option)
                        .expect("valid input"),
                )
            });
        });
        group.bench_with_input(BenchmarkId::new("integration", name), &input, |b, input| {
            b.iter(|| {
                let dr = dio(input, FS, &dio_option).expect("valid input");
                black_box(
                    cheaptrick(input, FS, &dr.temporal_positions, &dr.f0, &ct_option)
                        .expect("valid input"),
                )
            });
        });
    }
    group.finish();

    // The criterion results are now on disk; build the report from them.
    write_report();
}
