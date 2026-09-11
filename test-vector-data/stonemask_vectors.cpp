// C++ reference driver for StoneMask test vectors (PHASE0-7).
//
// Links against the pre-built mmorise/World reference library
// (ext_src/world-cpp/build/libworld.a). C++ is authoritative.
//
// Regenerate (from the repo root):
//   g++ -O2 -I ext_src/world-cpp/src test-vector-data/stonemask_vectors.cpp \
//       ext_src/world-cpp/build/libworld.a -o /tmp/stonemask_vectors -lm
//   /tmp/stonemask_vectors
//
// (If libworld.a is missing, rebuild it first: see ext_src/world-cpp
//  CMakeLists.txt, e.g. cmake -B build && cmake --build build.)
//
// The driver prints refined_f0 at full f64 precision (%.17g). The reference
// values are hard-coded in the Rust tests
// (crates/world-rs-core/src/stonemask/mod.rs, `matches_cpp_ref_*`). The Rust
// tests re-generate the SAME deterministic input signals (the `sine` formula
// below) and assert the Rust StoneMask output matches these values within 1e-6
// (empirically ~1e-14, the residual being the Ooura-vs-rustfft FFT difference).
//
// Reference values captured 2026-09-05:
//   A_sine200_fs16000:            refined[0] = 199.67434898305939
//   B_sine150_fs44100:            refined[0] = 149.8954510311552
//   C_zero_fs16000:               refined[0] = 200
//   D_sine250_f0init200_fs16000:  refined[0] = 200
//   E_guards_fs16000:             refined[0] = 0, refined[1] = 0
//   F_multiframe_sine220_fs16000: refined = [219.92606774461879,
//                                        219.92606774461768,
//                                        219.92606774461754]
#include <cmath>
#include <cstdio>
#include <vector>
#include "world/stonemask.h"

static const double PI = 3.1415926535897932384;

// Sine signal: x[i] = sin(2*pi*f0_true*i/fs), i in [0,n).
static std::vector<double> sine(int n, int fs, double f0_true) {
  std::vector<double> x(n);
  for (int i = 0; i < n; ++i) x[i] = sin(2.0 * PI * f0_true * (double)i / (double)fs);
  return x;
}

static void run(const char *name, const std::vector<double> &x, int fs,
    const std::vector<double> &temporal, const std::vector<double> &f0) {
  int f0_length = (int)f0.size();
  std::vector<double> refined(f0_length, -999.0);
  StoneMask(x.data(), (int)x.size(), fs, temporal.data(), f0.data(), f0_length,
      refined.data());
  printf("case %s:\n", name);
  for (int i = 0; i < f0_length; ++i) printf("  refined[%d] = %.17g\n", i, refined[i]);
  printf("\n");
}

int main() {
  // A: pure sine 200 Hz, fs=16000, n=8192, single frame at center.
  {
    int fs = 16000, n = 8192;
    double f0_true = 200.0;
    std::vector<double> x = sine(n, fs, f0_true);
    std::vector<double> temporal = { (double)n / 2.0 / (double)fs };
    std::vector<double> f0 = { f0_true };
    run("A_sine200_fs16000", x, fs, temporal, f0);
  }
  // B: pure sine 150 Hz, fs=44100, n=16384, single frame at center.
  {
    int fs = 44100, n = 16384;
    double f0_true = 150.0;
    std::vector<double> x = sine(n, fs, f0_true);
    std::vector<double> temporal = { (double)n / 2.0 / (double)fs };
    std::vector<double> f0 = { f0_true };
    run("B_sine150_fs44100", x, fs, temporal, f0);
  }
  // C: all-zero spectrum, fs=16000, f0=200 (exercises zero-power path ->
  //     tentative rejected -> 20% clamp keeps initial).
  {
    int fs = 16000, n = 1024;
    std::vector<double> x(n, 0.0);
    std::vector<double> temporal = { (double)n / 2.0 / (double)fs };
    std::vector<double> f0 = { 200.0 };
    run("C_zero_fs16000", x, fs, temporal, f0);
  }
  // D: sine at 250 Hz but initial f0=200 (offset; exercises 20% clamp).
  {
    int fs = 16000, n = 8192;
    std::vector<double> x = sine(n, fs, 250.0);
    std::vector<double> temporal = { (double)n / 2.0 / (double)fs };
    std::vector<double> f0 = { 200.0 };
    run("D_sine250_f0init200_fs16000", x, fs, temporal, f0);
  }
  // E: guard cases (output independent of x).
  {
    int fs = 16000, n = 1024;
    std::vector<double> x = sine(n, fs, 200.0);
    std::vector<double> temporal = { 0.1, 0.2 };
    std::vector<double> f0 = { 30.0, 2000.0 };  // <=40 -> 0 ; >fs/12 -> 0
    run("E_guards_fs16000", x, fs, temporal, f0);
  }
  // F: multi-frame sine 220 Hz, fs=16000, n=8192, frames at 0.1/0.2/0.3 s.
  {
    int fs = 16000, n = 8192;
    std::vector<double> x = sine(n, fs, 220.0);
    std::vector<double> temporal = { 0.1, 0.2, 0.3 };
    std::vector<double> f0 = { 220.0, 220.0, 220.0 };
    run("F_multiframe_sine220_fs16000", x, fs, temporal, f0);
  }
  return 0;
}
