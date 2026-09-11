# WORLD-rs Equivalence Report
Generated: 2026-09-09T07:58:04.749431Z
Commit: e6153e75aa9e13f6208cc015c205f548f270a28f
Toolchain:
- rustc 1.98.0 (88d9e12ae 2026-08-18)
- g++ (GCC) 16.1.1 20260501 (Red Hat 16.1.1-1)
- Python 3.14.4
- numpy 2.5.0


## Summary
FAIL

## Per-Phase Accuracy
| Phase | Metric | Value | Threshold | Pass? |
|-------|--------|-------|-----------|-------|
| Synthesis | Max abs error | 0.0488628 | < 1e-06 | ❌ |
| Synthesis | Relative error | 0.0413708 | < 0.0001 | ❌ |
| Synthesis | PSNR | 53.0317 | ≥ 350.0 dB | ❌ |
| Synthesis | Sample count match | 80001 | == 80001 | ✅ |
| CheapTrick | PSNR | 115.735 | ≥ 40.0 dB | ✅ |
| CheapTrick | SP correlation | 1 | ≥ 0.99 | ✅ |
| CheapTrick | Max abs error | 0.00110002 | < 0.001 | ❌ |
| CheapTrick | Frame count match | 1001 | == 1001 | ✅ |
| DIO | F0 RMSE | 0.00657611 | < 1.0 | ✅ |
| DIO | Voicing agreement | 100 | ≥ 99.0% | ✅ |
| DIO | Temporal grid max abs | 0 | < 1e-06 | ✅ |
| DIO | Max per-frame diff voiced | 0.167388 | < 0.1 | ❌ |
| Synthesis | Max abs error | 1.82168e-11 | < 1e-06 | ✅ |
| Synthesis | Relative error | 1.75489e-11 | < 0.0001 | ✅ |
| Synthesis | PSNR | +inf (bit-identical) | ≥ 350.0 dB | ✅ |
| Synthesis | Sample count match | 80001 | == 80001 | ✅ |
| CheapTrick | PSNR | +inf (bit-identical) | ≥ 40.0 dB | ✅ |
| CheapTrick | SP correlation | 1 | ≥ 0.99 | ✅ |
| CheapTrick | Max abs error | 3.59712e-14 | < 0.001 | ✅ |
| CheapTrick | Frame count match | 1001 | == 1001 | ✅ |
| DIO | F0 RMSE | 3.11097e-13 | < 1.0 | ✅ |
| DIO | Voicing agreement | 100 | ≥ 99.0% | ✅ |
| DIO | Temporal grid max abs | 0 | < 1e-06 | ✅ |
| DIO | Max per-frame diff voiced | 3.97904e-12 | < 0.1 | ✅ |
| Synthesis | Max abs error | 8.37741e-12 | < 1e-06 | ✅ |
| Synthesis | Relative error | 7.02844e-12 | < 0.0001 | ✅ |
| Synthesis | PSNR | +inf (bit-identical) | ≥ 350.0 dB | ✅ |
| Synthesis | Sample count match | 80001 | == 80001 | ✅ |
| CheapTrick | PSNR | +inf (bit-identical) | ≥ 40.0 dB | ✅ |
| CheapTrick | SP correlation | 1 | ≥ 0.99 | ✅ |
| CheapTrick | Max abs error | 3.19744e-14 | < 0.001 | ✅ |
| CheapTrick | Frame count match | 1001 | == 1001 | ✅ |
| DIO | F0 RMSE | 2.59014e-13 | < 1.0 | ✅ |
| DIO | Voicing agreement | 100 | ≥ 99.0% | ✅ |
| DIO | Temporal grid max abs | 0 | < 1e-06 | ✅ |
| DIO | Max per-frame diff voiced | 2.30216e-12 | < 0.1 | ✅ |
| Synthesis | Max abs error | 2.65433 | < 1e-06 | ❌ |
| Synthesis | Relative error | 0.800546 | < 0.0001 | ❌ |
| Synthesis | PSNR | 29.8261 | ≥ 350.0 dB | ❌ |
| Synthesis | Sample count match | 32001 | == 32001 | ✅ |
| CheapTrick | PSNR | 42.3286 | ≥ 40.0 dB | ✅ |
| CheapTrick | SP correlation | 0.994087 | ≥ 0.99 | ✅ |
| CheapTrick | Max abs error | 31.9676 | < 0.001 | ❌ |
| CheapTrick | Frame count match | 401 | == 401 | ✅ |
| DIO | F0 RMSE | 56.498 | < 1.0 | ❌ |
| DIO | Voicing agreement | 99.5012 | ≥ 99.0% | ✅ |
| DIO | Temporal grid max abs | 0 | < 1e-06 | ✅ |
| DIO | Max per-frame diff voiced | 2.27374e-12 | < 0.1 | ✅ |
| Synthesis | Max abs error | 1.40568e-08 | < 1e-06 | ✅ |
| Synthesis | Relative error | 8.37453e-09 | < 0.0001 | ✅ |
| Synthesis | PSNR | +inf (bit-identical) | ≥ 350.0 dB | ✅ |
| Synthesis | Sample count match | 32001 | == 32001 | ✅ |
| CheapTrick | PSNR | +inf (bit-identical) | ≥ 40.0 dB | ✅ |
| CheapTrick | SP correlation | 1 | ≥ 0.99 | ✅ |
| CheapTrick | Max abs error | 2.90447e-09 | < 0.001 | ✅ |
| CheapTrick | Frame count match | 401 | == 401 | ✅ |
| DIO | F0 RMSE | 1.01855e-14 | < 1.0 | ✅ |
| DIO | Voicing agreement | 100 | ≥ 99.0% | ✅ |
| DIO | Temporal grid max abs | 0 | < 1e-06 | ✅ |
| DIO | Max per-frame diff voiced | 1.42109e-13 | < 0.1 | ✅ |
| Synthesis | Max abs error | 3.48999e-09 | < 1e-06 | ✅ |
| Synthesis | Relative error | 1.08051e-09 | < 0.0001 | ✅ |
| Synthesis | PSNR | +inf (bit-identical) | ≥ 350.0 dB | ✅ |
| Synthesis | Sample count match | 32001 | == 32001 | ✅ |
| CheapTrick | PSNR | +inf (bit-identical) | ≥ 40.0 dB | ✅ |
| CheapTrick | SP correlation | 1 | ≥ 0.99 | ✅ |
| CheapTrick | Max abs error | 1.33081e-10 | < 0.001 | ✅ |
| CheapTrick | Frame count match | 401 | == 401 | ✅ |
| DIO | F0 RMSE | 6.95318e-15 | < 1.0 | ✅ |
| DIO | Voicing agreement | 100 | ≥ 99.0% | ✅ |
| DIO | Temporal grid max abs | 0 | < 1e-06 | ✅ |
| DIO | Max per-frame diff voiced | 1.13687e-13 | < 0.1 | ✅ |
| Synthesis | Max abs error | 4.43454e-07 | < 1e-06 | ✅ |
| Synthesis | Relative error | 1.6231e-07 | < 0.0001 | ✅ |
| Synthesis | PSNR | 144.248 | ≥ 350.0 dB | ❌ |
| Synthesis | Sample count match | 32001 | == 32001 | ✅ |
| CheapTrick | PSNR | +inf (bit-identical) | ≥ 40.0 dB | ✅ |
| CheapTrick | SP correlation | 1 | ≥ 0.99 | ✅ |
| CheapTrick | Max abs error | 1.48853e-07 | < 0.001 | ✅ |
| CheapTrick | Frame count match | 401 | == 401 | ✅ |
| DIO | F0 RMSE | 1.26151e-14 | < 1.0 | ✅ |
| DIO | Voicing agreement | 100 | ≥ 99.0% | ✅ |
| DIO | Temporal grid max abs | 0 | < 1e-06 | ✅ |
| DIO | Max per-frame diff voiced | 1.13687e-13 | < 0.1 | ✅ |
| Synthesis | Max abs error | 2.20857e-22 | < 1e-06 | ✅ |
| Synthesis | Relative error | 4.26454e-15 | < 0.0001 | ✅ |
| Synthesis | PSNR | +inf (bit-identical) | ≥ 350.0 dB | ✅ |
| Synthesis | Sample count match | 32001 | == 32001 | ✅ |
| CheapTrick | PSNR | +inf (bit-identical) | ≥ 40.0 dB | ✅ |
| CheapTrick | SP correlation | 1 | ≥ 0.99 | ✅ |
| CheapTrick | Max abs error | 3.8457e-30 | < 0.001 | ✅ |
| CheapTrick | Frame count match | 401 | == 401 | ✅ |
| DIO | F0 RMSE | 0 | < 1.0 | ✅ |
| DIO | Voicing agreement | 100 | ≥ 99.0% | ✅ |
| DIO | Temporal grid max abs | 0 | < 1e-06 | ✅ |
| DIO | Max per-frame diff voiced | 0 | < 0.1 | ✅ |
| Synthesis | Max abs error | 1.28268e-08 | < 1e-06 | ✅ |
| Synthesis | Relative error | 3.74025e-09 | < 0.0001 | ✅ |
| Synthesis | PSNR | +inf (bit-identical) | ≥ 350.0 dB | ✅ |
| Synthesis | Sample count match | 801 | == 801 | ✅ |
| CheapTrick | PSNR | +inf (bit-identical) | ≥ 40.0 dB | ✅ |
| CheapTrick | SP correlation | 1 | ≥ 0.99 | ✅ |
| CheapTrick | Max abs error | 5.12252e-10 | < 0.001 | ✅ |
| CheapTrick | Frame count match | 11 | == 11 | ✅ |
| DIO | F0 RMSE | 0 | < 1.0 | ✅ |
| DIO | Voicing agreement | 100 | ≥ 99.0% | ✅ |
| DIO | Temporal grid max abs | 0 | < 1e-06 | ✅ |
| DIO | Max per-frame diff voiced | 0 | < 0.1 | ✅ |
| Synthesis | Max abs error | 2.60661e-12 | < 1e-06 | ✅ |
| Synthesis | Relative error | 1.08874e-12 | < 0.0001 | ✅ |
| Synthesis | PSNR | +inf (bit-identical) | ≥ 350.0 dB | ✅ |
| Synthesis | Sample count match | 80001 | == 80001 | ✅ |
| CheapTrick | PSNR | +inf (bit-identical) | ≥ 40.0 dB | ✅ |
| CheapTrick | SP correlation | 1 | ≥ 0.99 | ✅ |
| CheapTrick | Max abs error | 1.46194e-12 | < 0.001 | ✅ |
| CheapTrick | Frame count match | 1001 | == 1001 | ✅ |
| DIO | F0 RMSE | 4.92563e-13 | < 1.0 | ✅ |
| DIO | Voicing agreement | 100 | ≥ 99.0% | ✅ |
| DIO | Temporal grid max abs | 0 | < 1e-06 | ✅ |
| DIO | Max per-frame diff voiced | 5.17275e-12 | < 0.1 | ✅ |
| Synthesis | Max abs error | 4.2345e-12 | < 1e-06 | ✅ |
| Synthesis | Relative error | 3.52302e-12 | < 0.0001 | ✅ |
| Synthesis | PSNR | +inf (bit-identical) | ≥ 350.0 dB | ✅ |
| Synthesis | Sample count match | 160001 | == 160001 | ✅ |
| CheapTrick | PSNR | +inf (bit-identical) | ≥ 40.0 dB | ✅ |
| CheapTrick | SP correlation | 1 | ≥ 0.99 | ✅ |
| CheapTrick | Max abs error | 6.35048e-14 | < 0.001 | ✅ |
| CheapTrick | Frame count match | 2001 | == 2001 | ✅ |
| DIO | F0 RMSE | 3.79321e-13 | < 1.0 | ✅ |
| DIO | Voicing agreement | 100 | ≥ 99.0% | ✅ |
| DIO | Temporal grid max abs | 0 | < 1e-06 | ✅ |
| DIO | Max per-frame diff voiced | 8.10019e-12 | < 0.1 | ✅ |
| Synthesis | Max abs error | 2.43899e-10 | < 1e-06 | ✅ |
| Synthesis | Relative error | 3.47065e-10 | < 0.0001 | ✅ |
| Synthesis | PSNR | +inf (bit-identical) | ≥ 350.0 dB | ✅ |
| Synthesis | Sample count match | 32001 | == 32001 | ✅ |
| CheapTrick | PSNR | +inf (bit-identical) | ≥ 40.0 dB | ✅ |
| CheapTrick | SP correlation | 1 | ≥ 0.99 | ✅ |
| CheapTrick | Max abs error | 1.8221e-11 | < 0.001 | ✅ |
| CheapTrick | Frame count match | 401 | == 401 | ✅ |
| DIO | F0 RMSE | 0 | < 1.0 | ✅ |
| DIO | Voicing agreement | 100 | ≥ 99.0% | ✅ |
| DIO | Temporal grid max abs | 0 | < 1e-06 | ✅ |
| DIO | Max per-frame diff voiced | 0 | < 0.1 | ✅ |
| Synthesis | Max abs error | 1.51723e-12 | < 1e-06 | ✅ |
| Synthesis | Relative error | 4.93522e-13 | < 0.0001 | ✅ |
| Synthesis | PSNR | +inf (bit-identical) | ≥ 350.0 dB | ✅ |
| Synthesis | Sample count match | 32001 | == 32001 | ✅ |
| CheapTrick | PSNR | +inf (bit-identical) | ≥ 40.0 dB | ✅ |
| CheapTrick | SP correlation | 1 | ≥ 0.99 | ✅ |
| CheapTrick | Max abs error | 8.52651e-13 | < 0.001 | ✅ |
| CheapTrick | Frame count match | 401 | == 401 | ✅ |
| DIO | F0 RMSE | 5.31134e-13 | < 1.0 | ✅ |
| DIO | Voicing agreement | 100 | ≥ 99.0% | ✅ |
| DIO | Temporal grid max abs | 0 | < 1e-06 | ✅ |
| DIO | Max per-frame diff voiced | 5.51381e-12 | < 0.1 | ✅ |
| Synthesis | Max abs error | 6.6902e-13 | < 1e-06 | ✅ |
| Synthesis | Relative error | 2.28964e-13 | < 0.0001 | ✅ |
| Synthesis | PSNR | +inf (bit-identical) | ≥ 350.0 dB | ✅ |
| Synthesis | Sample count match | 32001 | == 32001 | ✅ |
| CheapTrick | PSNR | +inf (bit-identical) | ≥ 40.0 dB | ✅ |
| CheapTrick | SP correlation | 1 | ≥ 0.99 | ✅ |
| CheapTrick | Max abs error | 1.54188e-12 | < 0.001 | ✅ |
| CheapTrick | Frame count match | 401 | == 401 | ✅ |
| DIO | F0 RMSE | 1.74675e-12 | < 1.0 | ✅ |
| DIO | Voicing agreement | 100 | ≥ 99.0% | ✅ |
| DIO | Temporal grid max abs | 0 | < 1e-06 | ✅ |
| DIO | Max per-frame diff voiced | 3.20597e-11 | < 0.1 | ✅ |

## End-to-End
| Metric | Value | Threshold | Pass? |
|--------|-------|-----------|-------|
| Max abs error | 0.193085 | < 0.0001 | ❌ |
| PSNR | 75.7018 | ≥ 35.0 dB | ✅ |

| Metric | Value |
|--------|-------|
| PESQ | not measured |
| STOI | not measured |
| MOS | not measured |

## Edge Cases
| Test Case | world-rs Behavior | Pass? |
|-----------|-------------------|-------|
| (no sidecar data) | - | - |

## Known Differences
None.