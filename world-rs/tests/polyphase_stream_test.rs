// Test target: unwrap/panic in setup and assertions is expected.
#![allow(clippy::unwrap_used)]
#![allow(clippy::panic)]

use world_rs::resample::{resample_polyphase, PolyphaseStream};

#[test]
fn test_polyphase_stream_matches_batch() {
    let x: Vec<f32> = (0..1000).map(|i| (i as f32).sin()).collect();
    let from = 16000u32;
    let to = 48000u32;
    let batch = resample_polyphase(&x, from, to);
    let mut stream = PolyphaseStream::new(from, to).unwrap();
    let mut out = Vec::new();
    for chunk in x.chunks(256) {
        out.extend(stream.write(chunk));
    }
    out.extend(stream.flush());
    assert_eq!(batch.len(), out.len());
    for (a, b) in batch.iter().zip(out.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}
