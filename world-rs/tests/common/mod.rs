#![allow(dead_code)]
// Test helper module: unwrap/panic in setup and assertions is expected.
#![allow(clippy::unwrap_used)]
#![allow(clippy::panic)]

use std::fs;
use std::path::PathBuf;

// mirrored with world-rs-core/tests/common
pub fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

pub fn workspace_dir() -> PathBuf {
    manifest_dir().join("..")
}

pub fn world_equivalence_dir() -> PathBuf {
    workspace_dir().join("test/world-equivalence")
}

pub fn cpp_driver_path() -> PathBuf {
    world_equivalence_dir().join("cpp_driver")
}

pub fn fixture_path(name: &str) -> PathBuf {
    world_equivalence_dir()
        .join("fixtures")
        .join(format!("{name}.wav"))
}

pub fn skip_if_missing<P: AsRef<std::path::Path>>(path: P, msg: &str) {
    if !path.as_ref().exists() {
        eprintln!("skip: {msg} (missing {})", path.as_ref().display());
        std::process::exit(0);
    }
}

pub fn read_wav_16k_mono(path: &PathBuf) -> Vec<f64> {
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
        pos += 8 + chunk_size + chunk_size % 2;
    }
    let data = data.expect("no data chunk");
    let (chunks, _rest) = data.as_chunks::<2>();
    chunks.iter().map(f16_to_f64).collect()
}

fn f16_to_f64(c: &[u8; 2]) -> f64 {
    (i16::from_le_bytes(*c) as f64) / 32768.0
}

// mirrored with world-rs-core/tests/common
pub const PSNR_MAX: f64 = 1e6;

pub fn psnr(x: &[f64], y: &[f64]) -> f64 {
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

pub fn psnr_f32(x: &[f32], y: &[f32]) -> f64 {
    let n = x.len().min(y.len());
    let se: f64 = x[..n]
        .iter()
        .zip(&y[..n])
        .map(|(a, b)| {
            let d = *a as f64 - *b as f64;
            d * d
        })
        .sum();
    let mse = se / n as f64;
    if mse == 0.0 {
        return f64::INFINITY;
    }
    (PSNR_MAX * PSNR_MAX / mse).log10() * 10.0
}

pub fn psnr_end_to_end(ref_sig: &[f64], test_sig: &[f64]) -> f64 {
    let n = ref_sig.len().min(test_sig.len());
    if n == 0 {
        return f64::NEG_INFINITY;
    }
    let mse: f64 = ref_sig[..n]
        .iter()
        .zip(&test_sig[..n])
        .map(|(a, b)| {
            let d = a - b;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    if mse == 0.0 {
        return f64::INFINITY;
    }
    let max_val = ref_sig[..n].iter().cloned().fold(0.0_f64, f64::max).abs();
    if max_val == 0.0 {
        return f64::NEG_INFINITY;
    }
    10.0 * (max_val * max_val / mse).log10()
}

pub fn rmse(x: &[f64]) -> f64 {
    (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt()
}

pub fn max_abs(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

pub fn psnr_2d(rust: &[Vec<f64>], reference: &[Vec<f64>]) -> f64 {
    assert_eq!(rust.len(), reference.len(), "frame count mismatch");
    let mut se = 0.0f64;
    let mut n = 0usize;
    for (r, ref_row) in rust.iter().zip(reference) {
        assert_eq!(r.len(), ref_row.len(), "bin count mismatch");
        for (a, b) in r.iter().zip(ref_row) {
            let d = a - b;
            se += d * d;
            n += 1;
        }
    }
    let mse = se / n as f64;
    if mse == 0.0 {
        return f64::INFINITY;
    }
    (PSNR_MAX * PSNR_MAX / mse).log10() * 10.0
}
