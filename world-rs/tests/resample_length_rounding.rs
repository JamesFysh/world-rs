use world_rs::resample::{resample, resample_polyphase};

fn gcd(a: u32, b: u32) -> u32 {
    let (mut a, mut b) = (a, b);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn len_direct(n: usize, from: u32, to: u32) -> usize {
    ((n as f64 * to as f64 / from as f64).round()) as usize
}

fn len_reduced(n: usize, from: u32, to: u32) -> usize {
    let g = gcd(from, to);
    let l = to / g;
    let m = from / g;
    ((n as f64 * l as f64 / m as f64).round()) as usize
}

#[test]
fn resample_length_rounding_direct_vs_reduced_matches() {
    // Audible range rates
    let rates = [
        (8000u32, 16000u32),
        (16000, 48000),
        (48000, 16000),
        (44100, 48000),
        (48000, 44100),
        (16000, 32000),
        (32000, 16000),
        (22050, 48000),
        (48000, 22050),
        (192000, 48000),
        (48000, 192000),
    ];
    // Adversarial coprime pairs
    let adversarial = [(147u32, 160u32), (160, 147), (5u32, 6u32), (6, 5)];

    let all_rates: Vec<(u32, u32)> = rates
        .iter()
        .cloned()
        .chain(adversarial.iter().cloned())
        .collect();

    // Sample n values covering small to large, including near mantissa boundary
    let ns = [
        1usize, 2, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000,
    ];

    for &(from, to) in &all_rates {
        for &n in &ns {
            let d = len_direct(n, from, to);
            let r = len_reduced(n, from, to);
            assert_eq!(d, r, "len mismatch for n={} from={} to={}", n, from, to);
        }
    }
}

#[test]
fn resample_length_rounding_spaced_sweep() {
    // Sweep n 1..=1e7 for a representative pair with large L/M
    let from = 147u32;
    let to = 160u32;
    // Spaced sample to keep test fast: every 1000th value
    for n in (1..=1_000_000usize).step_by(1000) {
        let d = len_direct(n, from, to);
        let r = len_reduced(n, from, to);
        assert_eq!(d, r, "len mismatch for n={} from={} to={}", n, from, to);
    }
    // Also test near mantissa boundary conceptually: n*L ~ 4.5e15
    // For L=160, M=147, n ~ 2.8e13 which is far beyond audio, skip actual loop.
    // Verify formula holds for max realistic n (1e7)
    for n in [1usize, 1_000_000, 10_000_000] {
        let d = len_direct(n, from, to);
        let r = len_reduced(n, from, to);
        assert_eq!(d, r);
    }
}

#[test]
fn resample_output_length_matches_direct_formula() {
    // Ensure resample_polyphase uses round(n * to/from) as documented
    for &(from, to) in &[
        (16000u32, 48000u32),
        (48000, 16000),
        (44100, 48000),
        (147, 160),
    ] {
        for &n in &[1usize, 10, 1000, 100_000] {
            let x = vec![0.0f32; n];
            let y = resample_polyphase(&x, from, to);
            let expected = len_direct(n, from, to);
            assert_eq!(
                y.len(),
                expected,
                "resample length for n={} from={} to={}",
                n,
                from,
                to
            );
        }
    }
}

#[test]
fn resample_wrapper_length_matches_polyphase() {
    for &(from, to) in &[
        (16000u32, 48000u32),
        (48000, 16000),
        (44100, 48000),
        (147, 160),
    ] {
        for &n in &[1usize, 10, 1000] {
            let x = vec![0.0f32; n];
            let y1 = resample(&x, from, to);
            let y2 = resample_polyphase(&x, from, to);
            assert_eq!(y1.len(), y2.len());
            assert_eq!(y1.len(), len_direct(n, from, to));
        }
    }
}
