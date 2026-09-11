//-----------------------------------------------------------------------------
// DIO reference vector generator (Phase 1-8).
//
// Generates deterministic test signals, runs the C++ reference Dio(), and dumps
// the input signal plus the resulting F0 contour and temporal positions to
// binary files. The Rust accuracy tests
// (crates/world-rs-core/tests/dio_accuracy.rs) read these files and compare
// against the Rust dio() output.
//
// Binary format (little-endian, native on x86):
//   <name>.in : u32 x_length, f64 fs, x_length * f64 samples
//   <name>.out: u32 f0_length, f64 frame_period,
//               f0_length * f64 f0, f0_length * f64 temporal_positions
//
// Build + run via test-vector-data/generate-dio-vectors.sh.
//-----------------------------------------------------------------------------
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>

#include "world/dio.h"

namespace {

const double kFs = 16000.0;
const int kLength = 16000;  // 1 second

// Deterministic 64-bit LCG producing values in [-1, 1).
double lcg_next(unsigned long long *state) {
  *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
  return (double)(*state >> 11) / (double)(1ULL << 53) * 2.0 - 1.0;
}

void gen_sine(double *x, int n, double freq) {
  for (int i = 0; i < n; ++i) x[i] = sin(2.0 * M_PI * freq * (double)i / kFs);
}

// Linear chirp sweeping 100 Hz -> 600 Hz over the signal duration.
void gen_chirp(double *x, int n, double) {
  const double f_start = 100.0;
  const double f_end = 600.0;
  double phase = 0.0;
  for (int i = 0; i < n; ++i) {
    double t = (double)i / kFs;
    double f = f_start + (f_end - f_start) * t / ((double)n / kFs);
    phase += 2.0 * M_PI * f / kFs;
    x[i] = sin(phase);
  }
}

// Voiced, speech-like signal: a quasi-periodic glottal source (8 harmonics)
// with a slowly varying fundamental (~120-180 Hz) and syllable-rate amplitude
// modulation. Deterministic and self-contained (no external audio file).
void gen_speech(double *x, int n, double) {
  double phase = 0.0;
  for (int i = 0; i < n; ++i) {
    double t = (double)i / kFs;
    double f0 = 150.0 + 30.0 * sin(2.0 * M_PI * 0.5 * t);
    phase += 2.0 * M_PI * f0 / kFs;
    double v = 0.0;
    for (int h = 1; h <= 8; ++h) v += sin(h * phase) / (double)h;
    double env = 0.5 + 0.5 * sin(2.0 * M_PI * 3.0 * t);
    x[i] = 0.5 * env * v;
  }
}

// Unvoiced noise.
void gen_noise(double *x, int n, double) {
  unsigned long long s = 0x123456789abcdefULL;
  for (int i = 0; i < n; ++i) x[i] = lcg_next(&s);
}

// Silence.
void gen_silence(double *x, int n, double) {
  for (int i = 0; i < n; ++i) x[i] = 0.0;
}

void write_in(const std::string &path, double fs, int n, const double *x) {
  std::ofstream f(path, std::ios::binary);
  if (!f) {
    fprintf(stderr, "cannot open %s\n", path.c_str());
    exit(1);
  }
  f.write(reinterpret_cast<const char *>(&n), sizeof(n));
  f.write(reinterpret_cast<const char *>(&fs), sizeof(fs));
  f.write(reinterpret_cast<const char *>(x), (std::streamsize)(n * sizeof(double)));
}

void write_out(const std::string &path, double frame_period, int f0_length,
               const double *f0, const double *tp) {
  std::ofstream f(path, std::ios::binary);
  if (!f) {
    fprintf(stderr, "cannot open %s\n", path.c_str());
    exit(1);
  }
  f.write(reinterpret_cast<const char *>(&f0_length), sizeof(f0_length));
  f.write(reinterpret_cast<const char *>(&frame_period), sizeof(frame_period));
  f.write(reinterpret_cast<const char *>(f0),
          (std::streamsize)(f0_length * sizeof(double)));
  f.write(reinterpret_cast<const char *>(tp),
          (std::streamsize)(f0_length * sizeof(double)));
}

struct Case {
  const char *name;
  void (*gen)(double *, int, double);
  double param;
};

}  // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <out_dir>\n", argv[0]);
    return 1;
  }
  std::string outdir = argv[1];

  DioOption option;
  InitializeDioOption(&option);

  const Case cases[] = {
      {"sine_71", gen_sine, 71.0},
      {"sine_200", gen_sine, 200.0},
      {"sine_500", gen_sine, 500.0},
      {"sine_800", gen_sine, 800.0},
      {"chirp", gen_chirp, 0.0},
      {"speech", gen_speech, 0.0},
      {"noise", gen_noise, 0.0},
      {"silence", gen_silence, 0.0},
  };

  for (const Case &c : cases) {
    double x[kLength];
    c.gen(x, kLength, c.param);
    int f0_length = GetSamplesForDIO((int)kFs, kLength, option.frame_period);
    double *tp = new double[f0_length];
    double *f0 = new double[f0_length];
    Dio(x, kLength, (int)kFs, &option, tp, f0);
    write_in(outdir + "/" + c.name + ".in", kFs, kLength, x);
    write_out(outdir + "/" + c.name + ".out", option.frame_period, f0_length, f0, tp);
    delete[] tp;
    delete[] f0;
    printf("%s: f0_length=%d\n", c.name, f0_length);
  }

  return 0;
}
