// Test target: unwrap/panic in setup and assertions is expected.
#![allow(clippy::unwrap_used)]
#![allow(clippy::panic)]

use world_rs::synthesis::{get_y_length, synthesis, Synthesizer};

#[test]
fn test_synthesizer_matches_batch() {
    let fft_size = 1024;
    let fs = 16000.0;
    let frame_period_ms = 5.0;
    let f0_len = 100;
    let f0: Vec<f64> = vec![220.0; f0_len];
    let bins = fft_size / 2 + 1;
    let spectrogram: Vec<f64> = vec![1.0; f0_len * bins];
    let aperiodicity: Vec<f64> = vec![0.5; f0_len * bins];
    let y_length = get_y_length(f0_len, frame_period_ms, fs);
    // Build Vec<Vec> for batch
    let sp_vec: Vec<Vec<f64>> = spectrogram.chunks(bins).map(|c| c.to_vec()).collect();
    let ap_vec: Vec<Vec<f64>> = aperiodicity.chunks(bins).map(|c| c.to_vec()).collect();
    let batch = synthesis(
        &f0,
        f0_len,
        &sp_vec,
        &ap_vec,
        fft_size,
        frame_period_ms,
        fs,
        y_length,
    )
    .unwrap();
    let mut synth = Synthesizer::new(fft_size, fs, frame_period_ms).unwrap();
    let out = synth
        .synthesize_chunk(&f0, &spectrogram, &aperiodicity, y_length)
        .unwrap();
    assert_eq!(batch.len(), out.len());
    for (a, b) in batch.iter().zip(out.iter()) {
        assert!((a - b).abs() < 1e-9);
    }
}

#[test]
fn test_synthesis_giant_fft_size_rejected_before_alloc() {
    // Review finding #2: `fft_size = 2^30` passed validation (`>= 2 &&
    // pow2`) then allocated ~10 multi-GB buffers. Must fail fast.
    let f0_len = 2;
    let f0: Vec<f64> = vec![220.0; f0_len];
    let fft_size = 1usize << 30;
    // NOTE: rows are intentionally short; the fft_size cap
    // must fire before any row-length check or allocation.
    let sp_vec: Vec<Vec<f64>> = vec![vec![1.0; 4]; f0_len];
    let ap_vec: Vec<Vec<f64>> = vec![vec![0.5; 4]; f0_len];
    let res = synthesis(&f0, f0_len, &sp_vec, &ap_vec, fft_size, 5.0, 16000.0, 160);
    assert!(res.is_err(), "synthesis should reject giant fft_size");
}
