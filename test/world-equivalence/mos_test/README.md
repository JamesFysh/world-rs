# MOS Blind Discrimination Test — Scaffolding

**EQ-TASK-14** — Optional MOS blind discrimination test for WORLD-rs equivalence.

## Purpose

Run the §4.3 double-blind discrimination protocol from `docs/impl/support/EQUIVALENCE-TESTING.md`:
C++ vs world-rs outputs, "same recording?" yes/no, ≥ 80% "same" = majority cannot tell the difference.

This directory contains scaffolding for stimulus preparation, listener response collection, and analysis.

## Protocol Summary

1. **Stimulus prep**
   - Pick 8–12 short excerpts (5–10 s) from `speech_clean`, `speech_noise`, `music_vocal`, `long.wav` outputs (both implementations).
   - Musical + speech mix, no silence-only clips.
   - Loudness-normalize pairs to **-23 LUFS integrated** using `pyloudnorm` (target chosen per EQ-TASK-14 resolution). Do **not** normalize peak.
   - Fix tool + target before rendering stimuli.
   - Randomize order per listener, double-blind administrator where feasible.

2. **Run**
   - 10+ listeners, not developers.
   - Each hears each pair once: "Are these the same recording?" yes/no.
   - Optional 1–5 quality rating per stimulus.
   - Record raw responses in `test/world-equivalence/mos_responses.csv` with anonymized IDs only.

3. **Analyze**
   - Agreement rate = % of "same" responses.
   - Binomial confidence interval for agreement rate.
   - Per-fixture breakdown.
   - Quality-rating comparison if collected.

4. **Write up**
   - Append method + results + threats to validity (listener pool bias, equipment variance, excerpt selection) to `equivalence-report.md`.
   - If agreement < 80%, file discriminable excerpts as perceptual defects with fixture + timestamps.

## Files

- `README.md` — this protocol.
- `prepare_stimuli.py` — template script to build stimulus set and manifest.
- `../mos_manifest.json` — stimulus manifest (pair id, fixture, excerpt start/end, paths, loudness).
- `../mos_responses.csv` — anonymized listener responses.

## Stimulus Manifest Schema

`mos_manifest.json` is a list of pairs:

```json
[
  {
    "pair_id": "speech_clean_01",
    "fixture": "speech_clean",
    "excerpt_start_s": 1.0,
    "excerpt_end_s": 7.0,
    "cpp_path": "test/world-equivalence/results-cpp/.../speech_clean.synthesis.y.npy",
    "rs_path": "test/world-equivalence/results-rs/.../speech_clean.synthesis.y.npy",
    "stimulus_cpp_wav": "test/world-equivalence/mos_test/stimuli/speech_clean_01_cpp.wav",
    "stimulus_rs_wav": "test/world-equivalence/mos_test/stimuli/speech_clean_01_rs.wav",
    "loudness_lufs_cpp": -23.0,
    "loudness_lufs_rs": -23.0,
    "duration_s": 6.0
  }
]
```

## Response CSV Schema

`mos_responses.csv` header:

```
listener_id,pair_id,order,response_same,quality_cpp,quality_rs,notes
```

- `listener_id`: anonymized (e.g., L01)
- `response_same`: yes/no
- `quality_cpp` / `quality_rs`: optional 1–5
- `order`: which stimulus played first (A/B)

## Template Script Usage

```bash
# Install deps
pip install pyloudnorm soundfile numpy

# Prepare stimuli from existing synthesis outputs
python test/world-equivalence/mos_test/prepare_stimuli.py \
  --cpp-dir test/world-equivalence/results-cpp \
  --rs-dir test/world-equivalence/results-rs \
  --fixtures speech_clean speech_noise music_vocal long \
  --out-dir test/world-equivalence/mos_test/stimuli \
  --manifest test/world-equivalence/mos_manifest.json \
  --target-lufs -23

# The script will:
# - Load .npy synthesis outputs
# - Extract 5-10s excerpts
# - Loudness normalize each pair to target LUFS
# - Write WAV files and update manifest
# - Generate a randomized playlist per listener
```

## Analysis Notes

Agreement rate ≥ 80% is the pass threshold per §4.3.

Compute binomial CI:
```
n = listeners * pairs
k = number of "same" responses
p_hat = k / n
CI = p_hat ± z * sqrt(p_hat*(1-p_hat)/n)
```

Report per-fixture breakdown and threats to validity.

## Dependencies

- EQ-TASK-1 fixtures
- EQ-TASK-3/4 synthesis outputs
- Coordinate with EQ-TASK-12 for report placement

## Open Questions Resolved

- Recruitment: internal non-dev team, lead time 2 weeks
- Playback: controlled machine with calibrated headphones
- Loudness normalization: `pyloudnorm` target -23 LUFS integrated
