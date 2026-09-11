//-----------------------------------------------------------------------------
// Synthesis reference vector generator (Phase 3-9).
//
// Generates deterministic F0 contours, CheapTrick-style spectral envelopes,
// and constant-AP aperiodicity matrices, runs the C++ reference Synthesis(),
// and dumps the inputs plus the resulting waveform to binary files. The Rust
// accuracy tests (crates/world-rs/tests/synthesis_accuracy.rs) read these
// files and compare against the Rust synthesis() output using the PSNR metric
// and a key-sample relative-error bound.
//
// Binary format (little-endian, native on x86):
//   <name>.in : u32 f0_length, u32 fft_size, f64 frame_period_ms, f64 fs,
//               f0_length * f64 f0,
//               f0_length * (fft_size/2 + 1) * f64 spectrogram,
//               f0_length * (fft_size/2 + 1) * f64 aperiodicity
//   <name>.out: u32 y_length, y_length * f64 y
//
// Build + run via test-vector-data/generate-synthesis-vectors.sh.
//-----------------------------------------------------------------------------
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>

#include "world/synthesis.h"

namespace {

const double kFramePeriod = 5.0;  // ms, the standard 5 ms analysis grid

// Deterministic spectral envelope: row i, bin j. Smooth in the bin (so the
// minimum-phase filter is well behaved), linear in the row (so the
// between-frame interpolation is exercised). `scale` shrinks the whole matrix
// for the silence case.
double envelope(int i, int j, double scale) {
  return scale * (1.0 + 0.5 * cos(0.3 * (double)j) +
                  0.25 * (double)(j % 7) + 0.05 * (double)i);
}

void write_in(const std::string &path, int f0_length, int fft_size,
              double frame_period_ms, double fs, const double *f0,
              double **spectrogram, double **aperiodicity) {
  std::ofstream f(path, std::ios::binary);
  if (!f) {
    fprintf(stderr, "cannot open %s\n", path.c_str());
    exit(1);
  }
  f.write(reinterpret_cast<const char *>(&f0_length), sizeof(f0_length));
  f.write(reinterpret_cast<const char *>(&fft_size), sizeof(fft_size));
  f.write(reinterpret_cast<const char *>(&frame_period_ms),
          sizeof(frame_period_ms));
  f.write(reinterpret_cast<const char *>(&fs), sizeof(fs));
  f.write(reinterpret_cast<const char *>(f0),
          (std::streamsize)(f0_length * sizeof(double)));
  for (int i = 0; i < f0_length; ++i) {
    f.write(reinterpret_cast<const char *>(spectrogram[i]),
            (std::streamsize)((fft_size / 2 + 1) * sizeof(double)));
  }
  for (int i = 0; i < f0_length; ++i) {
    f.write(reinterpret_cast<const char *>(aperiodicity[i]),
            (std::streamsize)((fft_size / 2 + 1) * sizeof(double)));
  }
}

void write_out(const std::string &path, int y_length, const double *y) {
  std::ofstream f(path, std::ios::binary);
  if (!f) {
    fprintf(stderr, "cannot open %s\n", path.c_str());
    exit(1);
  }
  f.write(reinterpret_cast<const char *>(&y_length), sizeof(y_length));
  f.write(reinterpret_cast<const char *>(y),
          (std::streamsize)(y_length * sizeof(double)));
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
    int fft_size;
    int f0_length;
    double envelope_scale;  // 1.0 normally, 1e-6 for the silence case
  };

  const Case cases[] = {
      // 1 s of audio at the 5 ms grid -> 201 frames.
      {"sine_220_16k", 16000.0, 1024, 201, 1.0},
      {"chirp_16k", 16000.0, 1024, 201, 1.0},
      {"jump_16k", 16000.0, 1024, 201, 1.0},
      {"transition_16k", 16000.0, 1024, 201, 1.0},
      {"unvoiced_16k", 16000.0, 1024, 201, 1.0},
      {"silence_16k", 16000.0, 1024, 201, 1e-6},
      {"sine_440_48k", 48000.0, 2048, 201, 1.0},
  };

  for (const Case &c : cases) {
    double *f0 = new double[c.f0_length];
    for (int i = 0; i < c.f0_length; ++i) {
      double v;
      if (std::string(c.name) == "chirp_16k") {
        v = 100.0 + (600.0 - 100.0) * (double)i / (double)(c.f0_length - 1);
      } else if (std::string(c.name) == "jump_16k") {
        // Hard F0 discontinuity: 200 Hz -> 500 Hz at the midpoint.
        v = (i < c.f0_length / 2) ? 200.0 : 500.0;
      } else if (std::string(c.name) == "transition_16k") {
        // Voiced -> unvoiced at the midpoint.
        v = (i < c.f0_length / 2) ? 440.0 : 0.0;
      } else if (std::string(c.name) == "unvoiced_16k" ||
                 std::string(c.name) == "silence_16k") {
        v = 0.0;  // all frames unvoiced -> kDefaultF0 path
      } else if (std::string(c.name) == "sine_220_16k") {
        v = 220.0;
      } else {  // sine_440_48k
        v = 440.0;
      }
      f0[i] = v;
    }

    double **spectrogram = new double *[c.f0_length];
    double **aperiodicity = new double *[c.f0_length];
    for (int i = 0; i < c.f0_length; ++i) {
      spectrogram[i] = new double[c.fft_size / 2 + 1];
      aperiodicity[i] = new double[c.fft_size / 2 + 1];
      for (int j = 0; j <= c.fft_size / 2; ++j) {
        spectrogram[i][j] = envelope(i, j, c.envelope_scale);
        aperiodicity[i][j] = 0.5;  // constant-AP default (D4C deferred)
      }
    }

    int y_length =
      (int)((double)(c.f0_length - 1) * kFramePeriod / 1000.0 * c.fs + 1.0);
    double *y = new double[y_length];
    Synthesis(f0, c.f0_length, spectrogram, aperiodicity, c.fft_size,
              kFramePeriod, (int)c.fs, y_length, y);

    write_in(outdir + "/" + c.name + ".in", c.f0_length, c.fft_size,
             kFramePeriod, c.fs, f0, spectrogram, aperiodicity);
    write_out(outdir + "/" + c.name + ".out", y_length, y);

    printf("%s: fs=%d fft_size=%d f0_length=%d y_length=%d\n", c.name,
           (int)c.fs, c.fft_size, c.f0_length, y_length);

    for (int i = 0; i < c.f0_length; ++i) {
      delete[] spectrogram[i];
      delete[] aperiodicity[i];
    }
    delete[] spectrogram;
    delete[] aperiodicity;
    delete[] y;
    delete[] f0;
  }

  return 0;
}
