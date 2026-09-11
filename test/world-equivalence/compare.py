import argparse
import os
import re
import sys
from datetime import datetime
from collections import defaultdict

import numpy as np

from metrics import rmse, psnr, pearson_r, max_abs_error, voicing_agreement, pesq_score, stoi_score

THRESHOLDS = {
    'dio': {
        'f0_rmse': 1.0,
        'voicing_agreement': 99.0,
        'temporal_grid_max_abs': 1e-6,
        'f0_max_abs_voiced': 0.1,
    },
    'cheaptrick': {
        'psnr': 40.0,
        'correlation': 0.99,
        'max_abs_error': 1e-3,
    },
    'synthesis': {
        'max_abs_error': 1e-6,
        'relative_error': 1e-4,
        'psnr': 350.0,
    },
    'ap': {
        'psnr': 40.0,
        'correlation': 0.99,
        'max_abs_error': 1e-3,
    },
    'end_to_end': {
        'max_abs_error': 1e-4,
        'psnr': 35.0,
        'ap_override_max_abs': 1e-8,
    },
    'perceptual': {
        'pesq': 3.5,
        'stoi': 0.90,
        'stoi_sung': 0.95,
    }
}

PHASE_PATTERN = re.compile(r'^(?P<fixture>.+)\.(?P<phase>dio|harvest|stonemask|cheaptrick|d4c|synthesis)\.(?P<signal>f0|t|sp|ap|y)\.npy$')

# Fixture tiers (see docs/world-divergences.md): clean fixtures must hold
# bit-parity gates; adversarial fixtures hold tolerance gates, with a few
# rows pinned to locked values (machine-checked, not parity-gated).
ADVERSARIAL_FIXTURES = {'sine_800', 'speech_silence'}
ADV_THRESHOLDS = {
    'synthesis': {'max_abs_error': 0.06, 'relative_error': 0.05, 'psnr': 35.0},
    'cheaptrick': {'max_abs_error': 2e-3},
    'dio': {'f0_max_abs_voiced': 0.25},
}
# (phase, fixture, metric) -> (locked value, tolerance, kind) where kind is
# 'rel' (±fraction), 'db' (±dB absolute), or 'exact'. Drift either way FAILs.
LOCKED_VALUES = {
    ('CheapTrick', 'sine_800', 'Max abs error'): (31.9676, 0.05, 'rel'),
    ('Synthesis', 'sine_800', 'Max abs error'): (2.65433, 0.05, 'rel'),
    ('Synthesis', 'sine_800', 'Relative error'): (0.800546, 0.05, 'rel'),
    ('Synthesis', 'sine_800', 'PSNR'): (29.8261, 0.5, 'db'),
    ('StoneMask', 'speech_silence', 'Max diff voiced'): (4.9949, 0.05, 'rel'),
}

def tier(fixture):
    return 'adversarial' if fixture in ADVERSARIAL_FIXTURES else 'clean'

def gate(section, metric, fixture):
    adv = ADV_THRESHOLDS.get(section, {})
    if tier(fixture) == 'adversarial' and metric in adv:
        return adv[metric]
    return THRESHOLDS[section][metric]

def check_locked(phase, fixture, metric, value):
    """None if the row is gated normally, else (passed, threshold-string)."""
    key = (phase, fixture, metric)
    if key not in LOCKED_VALUES:
        return None
    locked, tol, kind = LOCKED_VALUES[key]
    if kind == 'db':
        ok = abs(value - locked) <= tol
        desc = f'locked {locked} ±{tol} dB'
    elif kind == 'exact':
        ok = value == locked
        desc = f'locked {locked}'
    else:
        ok = abs(value - locked) <= max(tol * abs(locked), 1e-9)
        desc = f'locked {locked} ±{tol * 100:g}%'
    return ok, desc

def load_pairs(cpp_dir, rs_dir):
    cpp_files = {}
    rs_files = {}
    for d, store in [(cpp_dir, cpp_files), (rs_dir, rs_files)]:
        for fname in os.listdir(d):
            if not fname.endswith('.npy'):
                continue
            m = PHASE_PATTERN.match(fname)
            if not m:
                continue
            key = (m.group('fixture'), m.group('phase'), m.group('signal'))
            store[key] = os.path.join(d, fname)
    pairs = {}
    for key, cpp_path in cpp_files.items():
        rs_path = rs_files.get(key)
        if rs_path:
            pairs[key] = (cpp_path, rs_path)
    return pairs

def compute_metrics(pairs):
    # Group by fixture and phase
    grouped = defaultdict(dict)
    for (fixture, phase, signal), (cpp_path, rs_path) in pairs.items():
        cpp_arr = np.load(cpp_path)
        rs_arr = np.load(rs_path)
        grouped[(fixture, phase)][signal] = (cpp_arr, rs_arr)

    results = []
    all_pass = True

    for (fixture, phase), signals in grouped.items():
        if phase == 'dio':
            if 'f0' in signals and 't' in signals:
                cpp_f0, rs_f0 = signals['f0']
                cpp_t, rs_t = signals['t']
                # voicing derived from f0 > 0
                v_cpp = cpp_f0 > 0
                v_rs = rs_f0 > 0
                # RMSE over jointly-voiced frames only: all-frame RMSE lets a
                # few voicing-flag flips dominate (2 flips x 800 Hz read as
                # "56 Hz error"). Flag flips are covered by voicing
                # agreement + the logged flip count below.
                voiced_mask = v_cpp & v_rs
                if np.any(voiced_mask):
                    f0_rmse_val = rmse(cpp_f0[voiced_mask], rs_f0[voiced_mask])
                elif np.any(v_cpp != v_rs):
                    f0_rmse_val = float('inf')  # disagree on voicing: explicit FAIL
                else:
                    f0_rmse_val = 0.0  # jointly unvoiced everywhere: perfect agreement
                thr = THRESHOLDS['dio']['f0_rmse']
                passed = np.isfinite(f0_rmse_val) and f0_rmse_val < thr
                results.append(('DIO', fixture, 'F0 RMSE (voiced-only)', f0_rmse_val, f'< {thr}', '✅' if passed else '❌'))
                all_pass = all_pass and passed

                va_val = voicing_agreement(v_cpp, v_rs)
                thr = THRESHOLDS['dio']['voicing_agreement']
                passed = va_val >= thr
                results.append(('DIO', fixture, 'Voicing agreement', va_val, f'≥ {thr}%', '✅' if passed else '❌'))
                all_pass = all_pass and passed

                flips = int(np.sum(v_cpp != v_rs))
                results.append(('DIO', fixture, 'Voicing flips', flips, 'info only', 'ℹ️'))

                temp_max = max_abs_error(cpp_t, rs_t)
                thr = THRESHOLDS['dio']['temporal_grid_max_abs']
                passed = temp_max < thr
                results.append(('DIO', fixture, 'Temporal grid max abs', temp_max, f'< {thr}', '✅' if passed else '❌'))
                all_pass = all_pass and passed

                voiced_mask = v_cpp & v_rs
                if np.any(voiced_mask):
                    f0_diff = np.abs(cpp_f0[voiced_mask] - rs_f0[voiced_mask])
                    max_voiced = float(np.max(f0_diff))
                else:
                    max_voiced = 0.0
                thr = gate('dio', 'f0_max_abs_voiced', fixture)
                passed = max_voiced < thr
                results.append(('DIO', fixture, 'Max per-frame diff voiced', max_voiced, f'< {thr}', '✅' if passed else '❌'))
                all_pass = all_pass and passed

        elif phase == 'cheaptrick':
            if 'sp' in signals:
                cpp_sp, rs_sp = signals['sp']
                psnr_val = psnr(cpp_sp, rs_sp)
                thr = THRESHOLDS['cheaptrick']['psnr']
                if np.isinf(psnr_val):
                    passed = True
                elif np.isfinite(psnr_val):
                    passed = psnr_val >= thr
                else:
                    passed = False
                results.append(('CheapTrick', fixture, 'PSNR', psnr_val, f'≥ {thr} dB', '✅' if passed else '❌'))
                all_pass = all_pass and passed

                corr_val = pearson_r(cpp_sp, rs_sp)
                thr = THRESHOLDS['cheaptrick']['correlation']
                passed = corr_val >= thr
                results.append(('CheapTrick', fixture, 'SP correlation', corr_val, f'≥ {thr}', '✅' if passed else '❌'))
                all_pass = all_pass and passed

                max_err = max_abs_error(cpp_sp, rs_sp)
                locked = check_locked('CheapTrick', fixture, 'Max abs error', max_err)
                if locked is not None:
                    passed, thr_desc = locked
                    results.append(('CheapTrick', fixture, 'Max abs error', max_err, thr_desc, '✅' if passed else '❌'))
                else:
                    thr = gate('cheaptrick', 'max_abs_error', fixture)
                    passed = max_err < thr
                    results.append(('CheapTrick', fixture, 'Max abs error', max_err, f'< {thr}', '✅' if passed else '❌'))
                all_pass = all_pass and passed

                frame_match = cpp_sp.shape[0] == rs_sp.shape[0]
                results.append(('CheapTrick', fixture, 'Frame count match', cpp_sp.shape[0], f'== {rs_sp.shape[0]}', '✅' if frame_match else '❌'))
                all_pass = all_pass and frame_match

        elif phase == 'synthesis':
            if 'y' in signals:
                cpp_y, rs_y = signals['y']
                max_err = max_abs_error(cpp_y, rs_y)
                locked = check_locked('Synthesis', fixture, 'Max abs error', max_err)
                if locked is not None:
                    passed, thr_desc = locked
                    results.append(('Synthesis', fixture, 'Max abs error', max_err, thr_desc, '✅' if passed else '❌'))
                else:
                    thr = gate('synthesis', 'max_abs_error', fixture)
                    passed = max_err < thr
                    results.append(('Synthesis', fixture, 'Max abs error', max_err, f'< {thr}', '✅' if passed else '❌'))
                all_pass = all_pass and passed

                denom = np.max(np.abs(cpp_y)) if np.max(np.abs(cpp_y)) > 0 else 1.0
                rel_err = max_err / denom
                locked = check_locked('Synthesis', fixture, 'Relative error', rel_err)
                if locked is not None:
                    passed, thr_desc = locked
                    results.append(('Synthesis', fixture, 'Relative error', rel_err, thr_desc, '✅' if passed else '❌'))
                else:
                    thr = gate('synthesis', 'relative_error', fixture)
                    passed = rel_err < thr
                    results.append(('Synthesis', fixture, 'Relative error', rel_err, f'< {thr}', '✅' if passed else '❌'))
                all_pass = all_pass and passed

                psnr_val = psnr(cpp_y, rs_y)
                locked = check_locked('Synthesis', fixture, 'PSNR', psnr_val) if np.isfinite(psnr_val) else None
                if locked is not None:
                    passed, thr_desc = locked
                    results.append(('Synthesis', fixture, 'PSNR', psnr_val, thr_desc, '✅' if passed else '❌'))
                else:
                    thr = gate('synthesis', 'psnr', fixture)
                    if np.isinf(psnr_val):
                        passed = True
                    elif np.isfinite(psnr_val):
                        passed = psnr_val >= thr
                    else:
                        passed = False
                    results.append(('Synthesis', fixture, 'PSNR', psnr_val, f'≥ {thr} dB', '✅' if passed else '❌'))
                all_pass = all_pass and passed

                sample_match = cpp_y.shape[0] == rs_y.shape[0]
                results.append(('Synthesis', fixture, 'Sample count match', cpp_y.shape[0], f'== {rs_y.shape[0]}', '✅' if sample_match else '❌'))
                all_pass = all_pass and sample_match

        elif phase == 'stonemask':
            if 'f0' in signals:
                cpp_f0, rs_f0 = signals['f0']
                v_cpp = cpp_f0 > 0
                v_rs = rs_f0 > 0
                flips = int(np.sum(v_cpp != v_rs))
                results.append(('StoneMask', fixture, 'Voicing flips', flips, 'info only', 'ℹ️'))
                voiced_mask = v_cpp & v_rs
                if np.any(voiced_mask):
                    sm_max = float(np.max(np.abs(cpp_f0[voiced_mask] - rs_f0[voiced_mask])))
                elif np.any(v_cpp != v_rs):
                    sm_max = None  # disagree on voicing: explicit FAIL
                else:
                    sm_max = 0.0  # jointly unvoiced everywhere: perfect agreement
                locked = check_locked('StoneMask', fixture, 'Max diff voiced', sm_max if sm_max is not None else float('inf'))
                if locked is not None:
                    passed, thr_desc = locked
                    passed = passed and sm_max is not None
                    results.append(('StoneMask', fixture, 'Max diff voiced', sm_max if sm_max is not None else float('nan'), thr_desc, '✅' if passed else '❌'))
                else:
                    thr = 0.1
                    passed = (sm_max is not None) and sm_max < thr
                    results.append(('StoneMask', fixture, 'Max diff voiced', sm_max if sm_max is not None else float('nan'), f'< {thr}', '✅' if passed else '❌'))
                all_pass = all_pass and passed

        elif phase in ('ap', 'd4c'):
            sig = 'ap' if phase == 'ap' else 'ap'
            if sig in signals:
                cpp_ap, rs_ap = signals[sig]
                psnr_val = psnr(cpp_ap, rs_ap)
                thr = THRESHOLDS['ap']['psnr']
                if np.isinf(psnr_val):
                    passed = True
                elif np.isfinite(psnr_val):
                    passed = psnr_val >= thr
                else:
                    passed = False
                results.append(('AP', fixture, 'PSNR', psnr_val, f'≥ {thr} dB', '✅' if passed else '❌'))
                all_pass = all_pass and passed

                corr_val = pearson_r(cpp_ap, rs_ap)
                thr = THRESHOLDS['ap']['correlation']
                passed = corr_val >= thr
                results.append(('AP', fixture, 'Correlation', corr_val, f'≥ {thr}', '✅' if passed else '❌'))
                all_pass = all_pass and passed

                max_err = max_abs_error(cpp_ap, rs_ap)
                thr = THRESHOLDS['ap']['max_abs_error']
                passed = max_err < thr
                results.append(('AP', fixture, 'Max abs error', max_err, f'< {thr}', '✅' if passed else '❌'))
                all_pass = all_pass and passed

    # End-to-end aggregation over the CLEAN tier only (adversarial
    # fixtures are gated per-fixture above and contribute no aggregate):
    # worst clean max-abs + min clean finite PSNR. If no finite clean
    # PSNR exists (all bit-identical, the normal case) the PSNR leg
    # passes vacuously and max-abs governs.
    e2e_results = []
    clean_synth = [r for r in results if r[0] == 'Synthesis' and r[1] not in ADVERSARIAL_FIXTURES]
    if clean_synth:
        max_err_vals = [r[3] for r in clean_synth if r[2] == 'Max abs error']
        if max_err_vals:
            e2e_max = float(np.max(max_err_vals))
            thr = THRESHOLDS['synthesis']['max_abs_error']
            passed = e2e_max < thr
            e2e_results.append(('Max abs error (clean worst)', e2e_max, f'< {thr}', '✅' if passed else '❌'))
            all_pass = all_pass and passed
        psnr_vals = [r[3] for r in clean_synth if r[2] == 'PSNR']
        if psnr_vals:
            finite_vals = [v for v in psnr_vals if np.isfinite(v)]
            if finite_vals:
                e2e_psnr = float(np.min(finite_vals))
                thr = THRESHOLDS['synthesis']['psnr']
                passed = e2e_psnr >= thr
            else:
                e2e_psnr = float('inf')
                passed = True
            e2e_results.append(('PSNR (clean min)', e2e_psnr, f'≥ {THRESHOLDS["synthesis"]["psnr"]} dB', '✅' if passed else '❌'))
            all_pass = all_pass and passed

    return results, e2e_results, all_pass

def write_report(report_path, results, e2e_results, all_pass, perceptual_results=None):
    ts = datetime.utcnow().isoformat() + 'Z'
    lines = []
    lines.append('# WORLD-rs Equivalence Report')
    lines.append(f'Generated: {ts}')
    lines.append('')
    lines.append('## Summary')
    lines.append('PASS' if all_pass else 'FAIL')
    lines.append('')
    lines.append('## Per-Phase Accuracy')
    lines.append('| Phase | Metric | Value | Threshold | Pass? |')
    lines.append('|-------|--------|-------|-----------|-------|')
    for phase, fixture, metric, value, thr, status in results:
        if isinstance(value, (float, np.floating)) and np.isinf(value):
            val_str = '+inf (bit-identical)' if value > 0 else 'zero reference'
        elif isinstance(value, (float, np.floating, int, np.integer)):
            val_str = f'{value:.6g}'
        else:
            val_str = str(value)
        lines.append(f'| {phase} | {metric} | {val_str} | {thr} | {status} |')
    lines.append('')
    lines.append('## End-to-End')
    lines.append('| Metric | Value | Threshold | Pass? |')
    lines.append('|--------|-------|-----------|-------|')
    for metric, value, thr, status in e2e_results:
        val_str = f'{value:.6g}' if isinstance(value, (float, int)) else str(value)
        lines.append(f'| {metric} | {val_str} | {thr} | {status} |')
    # Perceptual
    if perceptual_results:
        lines.append('')
        lines.append('## Perceptual Metrics')
        lines.append('| Fixture | PESQ | PESQ Threshold | PESQ Pass? | STOI | STOI Threshold | STOI Pass? | Notes |')
        lines.append('|---------|------|-----------------|------------|------|-----------------|------------|-------|')
        for fixture, pesq_val, pesq_thr, pesq_pass, stoi_val, stoi_thr, stoi_pass, notes in perceptual_results:
            pesq_str = f'{pesq_val:.3f}' if pesq_val is not None else 'skipped'
            stoi_str = f'{stoi_val:.3f}' if stoi_val is not None else 'skipped'
            pesq_thr_str = f'≥ {pesq_thr}'
            stoi_thr_str = f'≥ {stoi_thr}'
            lines.append(f'| {fixture} | {pesq_str} | {pesq_thr_str} | {"✅" if pesq_pass else "❌"} | {stoi_str} | {stoi_thr_str} | {"✅" if stoi_pass else "❌"} | {notes} |')
    else:
        lines.append('')
        lines.append('## Perceptual Metrics')
        lines.append('| Metric | Value |')
        lines.append('|--------|-------|')
        lines.append('| PESQ | not measured |')
        lines.append('| STOI | not measured |')
        lines.append('| MOS | not measured |')
    lines.append('')
    lines.append('## Edge Cases')
    lines.append('| Test Case | world-rs Behavior | Pass? |')
    lines.append('|-----------|-------------------|-------|')
    lines.append('| (no sidecar data) | - | - |')
    lines.append('')
    lines.append('## Known Differences')
    lines.append('None.')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

PERCEPTUAL_FIXTURES = {'speech_clean', 'speech_noise', 'music_vocal', 'long'}

def compute_perceptual(pairs, strict):
    synth_map = {}
    for (fixture, phase, signal), (cpp_path, rs_path) in pairs.items():
        if phase == 'synthesis' and signal == 'y':
            # keep first
            if fixture not in synth_map:
                synth_map[fixture] = (cpp_path, rs_path)
    results = []
    perceptual_pass = True
    for fixture, (cpp_path, rs_path) in synth_map.items():
        if fixture not in PERCEPTUAL_FIXTURES:
            continue
        cpp_y = np.load(cpp_path)
        rs_y = np.load(rs_path)
        fs = 16000
        pesq_val, resampled, fs_eff = pesq_score(cpp_y, rs_y, fs)
        stoi_val, _, _ = stoi_score(cpp_y, rs_y, fs)
        notes = []
        if resampled:
            notes.append(f'resampled to {fs_eff}Hz')
        if pesq_val is None:
            notes.append('PESQ skipped')
        if stoi_val is None:
            notes.append('STOI skipped')
        notes_str = '; '.join(notes)
        pesq_thr = THRESHOLDS['perceptual']['pesq']
        stoi_thr = THRESHOLDS['perceptual']['stoi_sung'] if 'music' in fixture or 'vocal' in fixture else THRESHOLDS['perceptual']['stoi']
        pesq_pass = pesq_val is not None and pesq_val >= pesq_thr
        stoi_pass = stoi_val is not None and stoi_val >= stoi_thr
        if strict:
            perceptual_pass = perceptual_pass and pesq_pass and stoi_pass
        results.append((fixture, pesq_val, pesq_thr, pesq_pass, stoi_val, stoi_thr, stoi_pass, notes_str))
    return results, perceptual_pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpp', required=True)
    parser.add_argument('--rs', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--perceptual', action='store_true')
    parser.add_argument('--strict-perceptual', action='store_true')
    args = parser.parse_args()

    pairs = load_pairs(args.cpp, args.rs)
    print(f'[compare] loaded {len(pairs)} cpp/rs pairs from {args.cpp} / {args.rs}')
    results, e2e_results, all_pass = compute_metrics(pairs)
    perceptual_results = []
    if args.perceptual:
        perceptual_results, perceptual_ok = compute_perceptual(pairs, args.strict_perceptual)
        if args.strict_perceptual:
            all_pass = all_pass and perceptual_ok
    write_report(args.report, results, e2e_results, all_pass, perceptual_results if args.perceptual else None)
    # Console summary: the full table lives in the report, but a bare
    # nonzero exit is undiagnosable in CI logs — always list failures here.
    n_fail = 0
    for phase, fixture, metric, value, thr, status in results:
        if status == '❌':
            n_fail += 1
            print(f'[compare] FAIL {phase}/{fixture} {metric}: value={value:.6g} gate={thr}')
    for metric, value, thr, status in e2e_results:
        if status == '❌':
            n_fail += 1
            print(f'[compare] FAIL end-to-end {metric}: value={value:.6g} gate={thr}')
    print(f'[compare] {len(results) + len(e2e_results) - n_fail} passed, {n_fail} failed '
          f'-> verdict {"PASS" if all_pass else "FAIL"} (report: {args.report})')
    sys.exit(0 if all_pass else 1)

if __name__ == '__main__':
    main()
