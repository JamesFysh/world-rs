//! Full analysis-synthesis demo (Phase 3-10, with Phase 4-9 resample).
//!
//! Loads the committed 16 kHz sample audio (`test-vector-data/vectors/test_sine.wav`,
//! 1 s of a 220 Hz sine), runs the complete WORLD pipeline
//!   PCM -> DIO -> StoneMask -> CheapTrick -> Synthesis
//! (the C++ `test/test.cpp` flow, with the constant-AP convenience default
//! 0.5 since D4C is deferred), resamples the reconstructed 16 kHz waveform to
//! 48 kHz (the v1 playback rate, Phase 4-9), prints per-stage timings and
//! quality metrics, and writes the reconstructed waveform to
//! `scratch/synthesis_demo_output.wav` (16-bit PCM, 16 kHz) and
//! `scratch/synthesis_demo_output_48k.wav` (16-bit PCM, 48 kHz). All internal
//! math is f64; the resample boundary is f32.
//!
//! Run with:
//!   cargo run -p world-rs --release --example synthesis_demo

// Example target: unwrap/panic in setup is expected.
#![allow(clippy::unwrap_used)]
#![allow(clippy::panic)]

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use realfft::RealFftPlanner;
use rustfft::num_complex::Complex;
use world_rs::cheaptrick::{cheaptrick, initialize_cheaptrick_option};
use world_rs::dio::{dio, initialize_dio_option};
use world_rs::resample::resample;
use world_rs::stonemask::stone_mask;
use world_rs::synthesis::{constant_aperiodicity, get_y_length, synthesis};

const FS: f64 = 16000.0;
const PI: f64 = std::f64::consts::PI;

/// Pinned full-scale constant for the PSNR metric (PHASE2-7: "MAX pinned,
/// e.g. 1e6"), matching the accuracy tests.
const PSNR_MAX: f64 = 1e6;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

fn sample_wav_path() -> PathBuf {
    workspace_root().join("test-vector-data/vectors/test_sine.wav")
}

fn output_wav_path() -> PathBuf {
    workspace_root().join("scratch/synthesis_demo_output.wav")
}

fn output_wav_path_48k() -> PathBuf {
    workspace_root().join("scratch/synthesis_demo_output_48k.wav")
}

/// Reads a 16-bit PCM mono 16 kHz WAV file into an f64 vector in [-1, 1].
// duplicated from tests/common (examples cannot depend on tests/)
fn read_wav(path: &PathBuf) -> Vec<f64> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("cannot read {path:?}: {e}"));
    assert_eq!(&bytes[0..4], b"RIFF", "not a RIFF file");
    assert_eq!(&bytes[8..12], b"WAVE", "not a WAVE file");
    let mut pos = 12usize;
    let mut data: Option<&[u8]> = None;
    while pos + 8 <= bytes.len() {
        let chunk_id = &bytes[pos..pos + 4];
        let chunk_size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        if chunk_id == b"fmt " {
            let fmt = &bytes[pos + 8..pos + 8 + chunk_size];
            let audio_format = u16::from_le_bytes(fmt[0..2].try_into().unwrap());
            let channels = u16::from_le_bytes(fmt[2..4].try_into().unwrap());
            let sample_rate = u32::from_le_bytes(fmt[4..8].try_into().unwrap());
            let bits = u16::from_le_bytes(fmt[14..16].try_into().unwrap());
            assert_eq!(audio_format, 1, "expected PCM");
            assert_eq!(channels, 1, "expected mono");
            assert_eq!(sample_rate, 16000, "expected 16 kHz");
            assert_eq!(bits, 16, "expected 16-bit");
        }
        if chunk_id == b"data" {
            let end = (pos + 8 + chunk_size).min(bytes.len());
            data = Some(&bytes[pos + 8..end]);
        }
        pos += 8 + chunk_size + chunk_size % 2; // chunks are word-aligned
    }
    let data = data.expect("no data chunk");
    let (chunks, _rest) = data.as_chunks::<2>();
    chunks
        .iter()
        .map(|c| i16::from_le_bytes(*c) as f64 / 32768.0)
        .collect()
}

/// Writes `y` (f64, clipped to [-1, 1]) as a 16-bit PCM mono WAV at `fs`.
fn write_wav(path: &PathBuf, y: &[f64], fs: u32) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create output directory");
    }
    let data_len = y.len() * 2;
    let mut out: Vec<u8> = Vec::with_capacity(44 + data_len);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&fs.to_le_bytes());
    out.extend_from_slice(&(fs * 2).to_le_bytes()); // byte rate
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_len as u32).to_le_bytes());
    for &v in y {
        let clamped = v.clamp(-1.0, 1.0);
        let s = (clamped * 32767.0).round().clamp(-32768.0, 32767.0) as i16;
        out.extend_from_slice(&s.to_le_bytes());
    }
    fs::write(path, &out).expect("write output wav");
}

/// Dominant frequency (Hz) of `x`: argmax of the Hann-windowed r2c magnitude
/// spectrum, bin resolution `fs / x.len()`.
fn dominant_frequency(x: &[f64], fs: f64) -> f64 {
    let n = x.len();
    let mut planner = RealFftPlanner::<f64>::new();
    let plan = planner.plan_fft_forward(n);
    let mut windowed: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &v)| v * (0.5 * (1.0 - (2.0 * PI * i as f64 / n as f64).cos())))
        .collect();
    let mut buf = vec![Complex { re: 0.0, im: 0.0 }; n / 2 + 1];
    let _ = plan.process(&mut windowed, &mut buf);
    let (bin, _) = buf
        .iter()
        .enumerate()
        .skip(1) // skip DC
        .fold((0usize, f64::NEG_INFINITY), |best, (i, c)| {
            let m = c.norm_sqr();
            if m > best.1 {
                (i, m)
            } else {
                best
            }
        });
    bin as f64 * fs / n as f64
}

/// `PSNR = 10 * log10(MAX^2 / MSE)` with `MAX` pinned to [`PSNR_MAX`].
// duplicated from tests/common (examples cannot depend on tests/)
fn psnr(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    let se: f64 = x[..n]
        .iter()
        .zip(&y[..n])
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
    let mse = se / n as f64;
    if mse == 0.0 {
        return f64::INFINITY;
    }
    (PSNR_MAX * PSNR_MAX / mse).log10() * 10.0
}

fn main() {
    let input_path = sample_wav_path();
    println!("WORLD synthesis demo (DIO -> StoneMask -> CheapTrick -> Synthesis)");
    println!("  input:  {}", input_path.display());

    let x = read_wav(&input_path);
    let duration_s = x.len() as f64 / FS;
    println!(
        "  audio:  {} samples, {duration_s:.1} s at {FS:.0} Hz (internal math: f64)",
        x.len()
    );

    // 1. DIO: F0 contour + 5 ms temporal grid.
    let dio_option = initialize_dio_option();
    let start = Instant::now();
    let dio_result = dio(&x, FS, &dio_option).expect("valid DIO input");
    let dio_ms = start.elapsed().as_secs_f64() * 1000.0;
    let f0_length = dio_result.f0_length;

    // 2. StoneMask: refine the DIO contour.
    let start = Instant::now();
    let f0 = stone_mask(
        &x,
        x.len(),
        FS,
        &dio_result.temporal_positions,
        &dio_result.f0,
        f0_length,
    )
    .unwrap();
    let stone_mask_ms = start.elapsed().as_secs_f64() * 1000.0;
    let voiced = f0.iter().filter(|&&v| v > 0.0).count();
    let f0_mean = f0.iter().filter(|&&v| v > 0.0).copied().sum::<f64>() / voiced.max(1) as f64;

    // 3. CheapTrick: spectral envelope per frame.
    let ct_option = initialize_cheaptrick_option(FS);
    let start = Instant::now();
    let sp = cheaptrick(&x, FS, &dio_result.temporal_positions, &f0, &ct_option)
        .expect("valid CheapTrick input");
    let cheaptrick_ms = start.elapsed().as_secs_f64() * 1000.0;
    let fft_size = ct_option.fft_size as usize;

    // 4. Synthesis: reconstruct the waveform (constant AP = 0.5, D4C deferred).
    let ap = constant_aperiodicity(f0_length, fft_size);
    let frame_period_ms = dio_option.frame_period;
    let y_length = get_y_length(f0_length, frame_period_ms, FS);
    let start = Instant::now();
    let y = synthesis(
        &f0,
        f0_length,
        &sp,
        &ap,
        fft_size,
        frame_period_ms,
        FS,
        y_length,
    )
    .expect("valid synthesis input");
    let synthesis_ms = start.elapsed().as_secs_f64() * 1000.0;

    // 5. Resample: 16 kHz -> 48 kHz (the v1 playback rate, Phase 4-9). The
    //    f64 synthesis output is downcast to the f32 resample boundary and
    //    resampled; the result is written as a 48 kHz WAV.
    let start = Instant::now();
    let y16_f32: Vec<f32> = y.iter().map(|&v| v as f32).collect();
    let y48 = resample(&y16_f32, 16000, 48000);
    let resample_ms = start.elapsed().as_secs_f64() * 1000.0;
    let y48_f64: Vec<f64> = y48.iter().map(|&v| v as f64).collect();
    let f48 = dominant_frequency(&y48_f64, 48000.0);

    let total_ms = dio_ms + stone_mask_ms + cheaptrick_ms + synthesis_ms + resample_ms;
    let rtf = total_ms / (duration_s * 1000.0);
    let value = psnr(&x, &y);
    let f = dominant_frequency(&y, FS);
    let out_path = output_wav_path();
    write_wav(&out_path, &y, 16000);
    let out_path_48k = output_wav_path_48k();
    write_wav(&out_path_48k, &y48_f64, 48000);

    println!();
    println!("  frames (f0_length):   {f0_length} @ {frame_period_ms} ms");
    println!("  voiced frames:        {voiced}/{f0_length} (mean F0 {f0_mean:.1} Hz)");
    println!("  cheaptrick fft_size:  {fft_size}");
    println!("  output:               {} samples (get_y_length)", y.len());
    println!();
    println!("  stage timings (1 s of audio):");
    println!("    DIO:                {dio_ms:.2} ms");
    println!("    StoneMask:          {stone_mask_ms:.2} ms");
    println!("    CheapTrick:         {cheaptrick_ms:.2} ms");
    println!("    Synthesis:          {synthesis_ms:.2} ms");
    println!("    Resample 16k->48k:  {resample_ms:.2} ms");
    println!("    total:              {total_ms:.2} ms (RTF {rtf:.4})");
    println!();
    println!("  quality vs original:");
    println!("    PSNR (MAX=1e6):     {value:.1} dB");
    println!("    dominant frequency: {f:.1} Hz @ 16k, {f48:.1} Hz @ 48k (original: 220 Hz)");
    println!();
    println!("  output written:       {}", out_path.display());
    println!("  output written:       {}", out_path_48k.display());
    println!("OK");
}
