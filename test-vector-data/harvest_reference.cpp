// Harvest reference vector generator
// Generates deterministic test signals, runs C++ reference Harvest(), and dumps
// input signal plus F0 contour and temporal positions to binary files.
// Binary format matches harvest_accuracy.rs expectations:
//   <name>.in : u32 x_length, f64 fs, x_length * f64 samples
//   <name>.out: u32 f0_length, f64 frame_period, f0_length * f64 f0, f0_length * f64 temporal_positions
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include "world/harvest.h"

namespace {
const double kFs = 16000.0;
const double kDuration = 2.0;
const int kLength = static_cast<int>(kFs * kDuration);

double lcg_next(unsigned long long *state) {
  *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
  return (double)(*state >> 11) / (double)(1ULL << 53) * 2.0 - 1.0;
}

void gen_harm(double *x, int n, double f0, int harmonics) {
  for (int i = 0; i < n; ++i) {
    double t = (double)i / kFs;
    double v = 0.0;
    for (int k = 1; k <= harmonics; ++k) {
      v += (0.5 / k) * sin(2.0 * M_PI * f0 * k * t);
    }
    x[i] = v;
  }
}

void gen_harm_chirp(double *x, int n) {
  for (int i = 0; i < n; ++i) {
    double t = (double)i / kFs;
    double phase = 2.0 * M_PI * (71.0 * t + (800.0 - 71.0) * t * t / (2.0 * kDuration));
    double v = 0.0;
    for (int k = 1; k <= 4; ++k) {
      v += (0.5 / k) * sin(k * phase);
    }
    x[i] = v;
  }
}

void gen_speech_like(double *x, int n) {
  for (int i = 0; i < n; ++i) {
    double t = (double)i / kFs;
    double f0 = 150.0 + 100.0 * t / kDuration;
    double phase = 2.0 * M_PI * (150.0 * t + 100.0 * t * t / (2.0 * kDuration));
    double v = 0.0;
    for (int k = 1; k <= 5; ++k) {
      v += (0.4 / k) * sin(k * phase);
    }
    double env = 0.7 + 0.3 * sin(2.0 * M_PI * 3.0 * t);
    x[i] = v * env;
  }
}

void gen_noise(double *x, int n) {
  unsigned long long s = 0x123456789abcdefULL;
  for (int i = 0; i < n; ++i) {
    x[i] = 0.1 * lcg_next(&s);
  }
}

void gen_silence(double *x, int n) {
  for (int i = 0; i < n; ++i) x[i] = 0.0;
}

void write_in(const std::string &path, double fs, int n, const double *x) {
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char *>(&n), sizeof(n));
  f.write(reinterpret_cast<const char *>(&fs), sizeof(fs));
  f.write(reinterpret_cast<const char *>(x), (std::streamsize)(n * sizeof(double)));
}

void write_out(const std::string &path, double frame_period, int f0_length,
               const double *f0, const double *tp) {
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char *>(&f0_length), sizeof(f0_length));
  f.write(reinterpret_cast<const char *>(&frame_period), sizeof(frame_period));
  f.write(reinterpret_cast<const char *>(f0), (std::streamsize)(f0_length * sizeof(double)));
  f.write(reinterpret_cast<const char *>(tp), (std::streamsize)(f0_length * sizeof(double)));
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <out_dir>\n", argv[0]);
    return 1;
  }
  std::string outdir = argv[1];
  HarvestOption option;
  InitializeHarvestOption(&option);
  const int fs_int = static_cast<int>(kFs);
  const int f0_length = GetSamplesForHarvest(fs_int, kLength, option.frame_period);
  double *tp = new double[f0_length];
  double *f0 = new double[f0_length];
  double x[kLength];

  auto process = [&](const char *name, void (*gen)(double *, int)) {
    gen(x, kLength);
    Harvest(x, kLength, fs_int, &option, tp, f0);
    write_in(outdir + "/" + name + ".in", kFs, kLength, x);
    write_out(outdir + "/" + name + ".out", option.frame_period, f0_length, f0, tp);
    printf("%s generated\n", name);
  };

  // harm_100
  gen_harm(x, kLength, 100.0, 4);
  Harvest(x, kLength, fs_int, &option, tp, f0);
  write_in(outdir + "/harm_100.in", kFs, kLength, x);
  write_out(outdir + "/harm_100.out", option.frame_period, f0_length, f0, tp);
  printf("harm_100 generated\n");

  gen_harm(x, kLength, 200.0, 4);
  Harvest(x, kLength, fs_int, &option, tp, f0);
  write_in(outdir + "/harm_200.in", kFs, kLength, x);
  write_out(outdir + "/harm_200.out", option.frame_period, f0_length, f0, tp);
  printf("harm_200 generated\n");

  gen_harm(x, kLength, 500.0, 4);
  Harvest(x, kLength, fs_int, &option, tp, f0);
  write_in(outdir + "/harm_500.in", kFs, kLength, x);
  write_out(outdir + "/harm_500.out", option.frame_period, f0_length, f0, tp);
  printf("harm_500 generated\n");

  gen_harm(x, kLength, 700.0, 4);
  Harvest(x, kLength, fs_int, &option, tp, f0);
  write_in(outdir + "/harm_700.in", kFs, kLength, x);
  write_out(outdir + "/harm_700.out", option.frame_period, f0_length, f0, tp);
  printf("harm_700 generated\n");

  gen_harm_chirp(x, kLength);
  Harvest(x, kLength, fs_int, &option, tp, f0);
  write_in(outdir + "/harm_chirp.in", kFs, kLength, x);
  write_out(outdir + "/harm_chirp.out", option.frame_period, f0_length, f0, tp);
  printf("harm_chirp generated\n");

  gen_speech_like(x, kLength);
  Harvest(x, kLength, fs_int, &option, tp, f0);
  write_in(outdir + "/speech_like.in", kFs, kLength, x);
  write_out(outdir + "/speech_like.out", option.frame_period, f0_length, f0, tp);
  printf("speech_like generated\n");

  gen_noise(x, kLength);
  Harvest(x, kLength, fs_int, &option, tp, f0);
  write_in(outdir + "/noise.in", kFs, kLength, x);
  write_out(outdir + "/noise.out", option.frame_period, f0_length, f0, tp);
  printf("noise generated\n");

  gen_silence(x, kLength);
  Harvest(x, kLength, fs_int, &option, tp, f0);
  write_in(outdir + "/silence.in", kFs, kLength, x);
  write_out(outdir + "/silence.out", option.frame_period, f0_length, f0, tp);
  printf("silence generated\n");

  delete[] tp;
  delete[] f0;
  return 0;
}
