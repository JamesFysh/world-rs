// Test target: unwrap/panic in setup and assertions is expected.
#![allow(clippy::unwrap_used)]
#![allow(clippy::panic)]

use world_rs::resample::{resample_polyphase, PolyphaseStream};

#[test]
fn test_polyphase_stream_various_chunks() {
    let x: Vec<f32> = (0..1234).map(|i| (i as f32).sin()).collect();
    for &from_to in &[(16000u32, 48000u32), (48000, 16000), (44100, 48000)] {
        let (from, to) = from_to;
        let batch = resample_polyphase(&x, from, to);
        for chunk_size in [1, 7, 32, 128, 256, 512] {
            let mut stream = PolyphaseStream::new(from, to).unwrap();
            let mut out = Vec::new();
            for chunk in x.chunks(chunk_size) {
                out.extend(stream.write(chunk));
            }
            out.extend(stream.flush());
            assert_eq!(
                batch.len(),
                out.len(),
                "len mismatch for chunk {}",
                chunk_size
            );
            for (a, b) in batch.iter().zip(out.iter()) {
                assert!((a - b).abs() < 1e-6, "mismatch for chunk {}", chunk_size);
            }
        }
    }
}
