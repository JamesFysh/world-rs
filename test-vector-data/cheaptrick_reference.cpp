//-----------------------------------------------------------------------------
// CheapTrick reference vector generator (Phase 2-7).
//
// Generates deterministic test signals and F0 contours, runs the C++ reference
// CheapTrick(), and dumps the input signal plus the resulting spectrogram to
// binary files. The Rust accuracy tests
// (crates/world-rs-core/tests/cheaptrick_accuracy.rs) read these files and
// compare against the Rust cheaptrick() output using the PSNR metric.
//
// Binary format (little-endian, native on x86):
//   <name>.in : u32 x_length, f64 fs, x_length * f64 samples
//   <name>.out: u32 f0_length, u32 fft_size,
//               f0_length * f64 f0, f0_length * f64 temporal_positions,
//               f0_length * (fft_size/2 + 1) * f64 spectrogram
//
// Build + run via test-vector-data/generate-cheaptrick-vectors.sh.
//-----------------------------------------------------------------------------
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>

#include "world/cheaptrick.h"
#include "world/dio.h"

namespace {

const double kFramePeriod = 5.0;  // ms, DIO's 5 ms grid

void gen_sine(double *x, int n, double fs, double freq) {
  for (int i = 0; i < n; ++i) x[i] = sin(2.0 * M_PI * freq * (double)i / fs);
}

// Linear chirp sweeping 100 Hz -> 600 Hz over the signal duration.
void gen_chirp(double *x, int n, double fs, double) {
  const double f_start = 100.0;
  const double f_end = 600.0;
  double phase = 0.0;
  for (int i = 0; i < n; ++i) {
    double t = (double)i / fs;
    double f = f_start + (f_end - f_start) * t / ((double)n / fs);
    phase += 2.0 * M_PI * f / fs;
    x[i] = sin(phase);
  }
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

void write_out(const std::string &path, int f0_length, int fft_size,
               const double *f0, const double *tp, double **spectrogram) {
  std::ofstream f(path, std::ios::binary);
  if (!f) {
    fprintf(stderr, "cannot open %s\n", path.c_str());
    exit(1);
  }
  f.write(reinterpret_cast<const char *>(&f0_length), sizeof(f0_length));
  f.write(reinterpret_cast<const char *>(&fft_size), sizeof(fft_size));
  f.write(reinterpret_cast<const char *>(f0),
          (std::streamsize)(f0_length * sizeof(double)));
  f.write(reinterpret_cast<const char *>(tp),
          (std::streamsize)(f0_length * sizeof(double)));
  for (int i = 0; i < f0_length; ++i) {
    f.write(reinterpret_cast<const char *>(spectrogram[i]),
            (std::streamsize)((fft_size / 2 + 1) * sizeof(double)));
  }
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <out_dir>\n", argv[0]);
    return 1;
  }
  std::string outdir = argv[1];

  struct Case {
    const char *name;
    double fs;
    int x_length;
    void (*gen)(double *, int, double, double);
    double param;  // frequency for sine, ignored for chirp
  };

  const Case cases[] = {
      {"sine_440_16k", 16000.0, 16000, gen_sine, 440.0},
      {"sine_440_48k", 48000.0, 48000, gen_sine, 440.0},
      {"unvoiced_16k", 16000.0, 16000, gen_sine, 440.0},
      {"transition_16k", 16000.0, 16000, gen_sine, 440.0},
      {"chirp_16k", 16000.0, 16000, gen_chirp, 0.0},
  };

  for (const Case &c : cases) {
    double *x = new double[c.x_length];
    c.gen(x, c.x_length, c.fs, c.param);

    int fs = (int)c.fs;
    int f0_length = GetSamplesForDIO(fs, c.x_length, kFramePeriod);

    // Temporal grid: the 5 ms grid is DIO's (temporal_positions[i] = i * fp / 1000).
    double *tp = new double[f0_length];
    for (int i = 0; i < f0_length; ++i) tp[i] = i * kFramePeriod / 1000.0;

    // F0 contour per case.
    double *f0 = new double[f0_length];
    for (int i = 0; i < f0_length; ++i) {
      double v;
      if (std::string(c.name) == "unvoiced_16k") {
        v = 0.0;  // all frames unvoiced -> kDefaultF0 path
      } else if (std::string(c.name) == "transition_16k") {
        v = (i < f0_length / 2) ? 440.0 : 0.0;  // voiced -> unvoiced
      } else if (std::string(c.name) == "chirp_16k") {
        v = 100.0 + (600.0 - 100.0) * (double)i / (double)(f0_length - 1);
      } else {
        v = 440.0;
      }
      f0[i] = v;
    }

    CheapTrickOption option;
    InitializeCheapTrickOption(fs, &option);

    double **spectrogram = new double *[f0_length];
    for (int i = 0; i < f0_length; ++i)
      spectrogram[i] = new double[option.fft_size / 2 + 1];
    CheapTrick(x, c.x_length, fs, tp, f0, f0_length, &option, spectrogram);

    write_in(outdir + "/" + c.name + ".in", c.fs, c.x_length, x);
    write_out(outdir + "/" + c.name + ".out", f0_length, option.fft_size,
              f0, tp, spectrogram);

    printf("%s: fs=%d x_length=%d f0_length=%d fft_size=%d\n", c.name, fs,
           c.x_length, f0_length, option.fft_size);

    for (int i = 0; i < f0_length; ++i) delete[] spectrogram[i];
    delete[] spectrogram;
    delete[] f0;
    delete[] tp;
    delete[] x;
  }

  return 0;
}
