// Example target: unwrap/panic in setup is expected.
#![allow(clippy::unwrap_used)]
#![allow(clippy::panic)]

use serde_json::json;
use std::env;
use std::path::Path;
use world_rs::cheaptrick::{cheaptrick, initialize_cheaptrick_option};
use world_rs::dio::{dio, initialize_dio_option, DioError};
use world_rs::stonemask::stone_mask;
use world_rs::synthesis::{constant_aperiodicity, get_y_length, synthesis};

fn read_wav(path: &Path) -> Result<(f64, Vec<f64>), String> {
    let mut reader = hound::WavReader::open(path).map_err(|e| e.to_string())?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate as f64;
    let channels = spec.channels;
    if channels != 1 {
        return Err(format!("only mono supported, got {} channels", channels));
    }
    let mut samples = Vec::new();
    match spec.sample_format {
        hound::SampleFormat::Int => match spec.bits_per_sample {
            16 => {
                for s in reader.samples::<i16>() {
                    let s = s.map_err(|e| e.to_string())?;
                    let v = s as f64 / 32768.0;
                    samples.push(v);
                }
            }
            32 => {
                for s in reader.samples::<i32>() {
                    let s = s.map_err(|e| e.to_string())?;
                    let v = s as f64 / 2147483648.0;
                    samples.push(v);
                }
            }
            _ => {
                return Err(format!(
                    "unsupported bits per sample: {}",
                    spec.bits_per_sample
                ))
            }
        },
        hound::SampleFormat::Float => match spec.bits_per_sample {
            32 => {
                for s in reader.samples::<f32>() {
                    let s = s.map_err(|e| e.to_string())?;
                    samples.push(s as f64);
                }
            }
            64 => return Err("64-bit float WAV not supported by hound".to_string()),
            _ => return Err(format!("unsupported float bits: {}", spec.bits_per_sample)),
        },
    }
    Ok((sample_rate, samples))
}

fn run_dio(x: &[f64], fs: f64) -> Result<(Vec<f64>, Vec<f64>), DioError> {
    let option = initialize_dio_option();
    let res = dio(x, fs, &option)?;
    Ok((res.f0, res.temporal_positions))
}

fn run_stonemask(x: &[f64], fs: f64) -> Result<Vec<f64>, DioError> {
    let option = initialize_dio_option();
    let dio_res = dio(x, fs, &option)?;
    let refined = stone_mask(
        x,
        x.len(),
        fs,
        &dio_res.temporal_positions,
        &dio_res.f0,
        dio_res.f0_length,
    )
    .unwrap();
    Ok(refined)
}

fn run_cheaptrick(x: &[f64], fs: f64) -> Result<Vec<Vec<f64>>, String> {
    let dio_option = initialize_dio_option();
    let dio_res = dio(x, fs, &dio_option).map_err(|e| e.to_string())?;
    let f0 = dio_res.f0;
    let ct_option = initialize_cheaptrick_option(fs);
    cheaptrick(x, fs, &dio_res.temporal_positions, &f0, &ct_option).map_err(|e| e.to_string())
}

fn run_synthesis(x: &[f64], fs: f64) -> Result<Vec<f64>, String> {
    let dio_option = initialize_dio_option();
    let dio_res = dio(x, fs, &dio_option).map_err(|e| e.to_string())?;
    let f0 = dio_res.f0;
    let ct_option = initialize_cheaptrick_option(fs);
    let sp = cheaptrick(x, fs, &dio_res.temporal_positions, &f0, &ct_option)
        .map_err(|e| e.to_string())?;
    let fft_size = ct_option.fft_size as usize;
    let ap = constant_aperiodicity(f0.len(), fft_size);
    let frame_period_ms = dio_option.frame_period;
    let y_length = get_y_length(f0.len(), frame_period_ms, fs);
    let y = synthesis(
        &f0,
        f0.len(),
        &sp,
        &ap,
        fft_size,
        frame_period_ms,
        fs,
        y_length,
    )
    .map_err(|e| e.to_string())?;
    Ok(y)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: equiv_dump <wav_path> <phase>");
        std::process::exit(1);
    }
    let wav_path = Path::new(&args[1]);
    let phase = args[2].to_lowercase();
    let (fs, x) = match read_wav(wav_path) {
        Ok(v) => v,
        Err(e) => {
            let out = json!({"error": format!("wav_read_error: {}", e)});
            println!("{}", out);
            return;
        }
    };
    let result = match phase.as_str() {
        "dio" => match run_dio(&x, fs) {
            Ok((f0, t)) => json!({"f0": f0, "t": t}),
            Err(e) => json!({"error": format!("dio_error: {}", e)}),
        },
        "stonemask" => match run_stonemask(&x, fs) {
            Ok(f0) => json!({"f0": f0}),
            Err(e) => json!({"error": format!("dio_error: {}", e)}),
        },
        "cheaptrick" => match run_cheaptrick(&x, fs) {
            Ok(sp) => json!({"sp": sp}),
            Err(e) => json!({"error": e}),
        },
        "synthesis" => match run_synthesis(&x, fs) {
            Ok(y) => json!({"y": y}),
            Err(e) => json!({"error": e}),
        },
        _ => json!({"error": format!("unknown phase: {}", phase)}),
    };
    println!("{}", result);
}
