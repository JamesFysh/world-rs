use proptest::prelude::*;
use world_rs::synthesis::{constant_aperiodicity, get_y_length, synthesis};

proptest! {
    #[test]
    fn synthesis_no_nan_output(
        f0_len in 1usize..32usize,
        fs in 8000.0..48000.0,
        frame_period_ms in 1.0..20.0,
        power in 8u32..13,
    ) {
        let fft_size = 1usize << power;
        let y_length = get_y_length(f0_len, frame_period_ms, fs);
        let f0: Vec<f64> = vec![200.0; f0_len];
        let spectrogram = vec![vec![1.0; fft_size / 2 + 1]; f0_len];
        let aperiodicity = constant_aperiodicity(f0_len, fft_size);
        let res = synthesis(&f0, f0_len, &spectrogram, &aperiodicity, fft_size, frame_period_ms, fs, y_length);
        if let Ok(y) = res {
            for v in y {
                prop_assert!(v.is_finite());
            }
        }
    }

    #[test]
    fn synthesis_no_panic_arbitrary(
        f0 in proptest::collection::vec(any::<f64>(), 0..32),
        f0_len in 0usize..32,
        spectrogram in proptest::collection::vec(proptest::collection::vec(any::<f64>(), 0..256), 0..16),
        aperiodicity in proptest::collection::vec(proptest::collection::vec(any::<f64>(), 0..256), 0..16),
        fft_size in 0usize..4096,
        frame_period_ms in any::<f64>(),
        fs in any::<f64>(),
        y_length in 0usize..100_000,
    ) {
        let result = std::panic::catch_unwind(|| {
            synthesis(&f0, f0_len, &spectrogram, &aperiodicity, fft_size, frame_period_ms, fs, y_length)
        });
        prop_assert!(result.is_ok(), "synthesis panicked");
    }
}
