//-----------------------------------------------------------------------------
// D4C reference vector generator (EQ-TASK-15).
//
// Generates reference vectors for the 3-fixture subset (speech_clean,
// speech_noise, music_vocal) by reading WAV fixtures, running DIO to obtain
// F0/temporal positions, then running the C++ D4C() reference. Writes binary
// .in/.out pairs matching the test-vector-data convention.
//
// .in layout: u32 x_length, f64 fs, x_length * f64 samples
// .out layout: u32 f0_length, u32 fft_size,
//              f0_length * f64 f0,
//              f0_length * f64 temporal_positions,
//              f0_length * (fft_size/2+1) * f64 aperiodicity (row-major)
//-----------------------------------------------------------------------------
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "world/d4c.h"
#include "world/dio.h"
#include "tools/audioio.h"

namespace {

void write_in(const std::string &path, double fs, int x_length, const double *x) {
  std::ofstream f(path, std::ios::binary);
  if (!f) {
    fprintf(stderr, "cannot open %s\n", path.c_str());
    exit(1);
  }
  f.write(reinterpret_cast<const char *>(&x_length), sizeof(x_length));
  f.write(reinterpret_cast<const char *>(&fs), sizeof(fs));
  f.write(reinterpret_cast<const char *>(x), (std::streamsize)(x_length * sizeof(double)));
}

void write_out(const std::string &path, int f0_length, int fft_size,
               const double *f0, const double *tp, double **aperiodicity) {
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
  int bins = fft_size / 2 + 1;
  for (int i = 0; i < f0_length; ++i) {
    f.write(reinterpret_cast<const char *>(aperiodicity[i]),
            (std::streamsize)(bins * sizeof(double)));
  }
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <out_dir>\n", argv[0]);
    return 1;
  }
  std::string outdir = argv[1];

  const char *fixtures[] = {"speech_clean", "speech_noise", "music_vocal"};
  const int kNumFixtures = 3;
  const int kFftSize = 1024;
  const double kMaxDurationSec = 1.0;  // keep vectors ~0.8 MB

  for (int fi = 0; fi < kNumFixtures; ++fi) {
    const char *name = fixtures[fi];
    std::string wav_path = std::string("../../test/world-equivalence/fixtures/") + name + ".wav";

    int fs = 0, nbit = 0;
    int x_length_full = GetAudioLength(wav_path.c_str());
    if (x_length_full <= 0) {
      fprintf(stderr, "cannot get length for %s\n", wav_path.c_str());
      continue;
    }

    double *x_full = new double[x_length_full];
    wavread(wav_path.c_str(), &fs, &nbit, x_full);

    int x_length = x_length_full;
    if (fs > 0) {
      int max_len = static_cast<int>(kMaxDurationSec * fs);
      if (x_length > max_len) x_length = max_len;
    }

    // Copy first x_length samples into compact buffer
    double *x = new double[x_length];
    for (int i = 0; i < x_length; ++i) x[i] = x_full[i];
    delete[] x_full;

    // DIO to obtain F0 and temporal positions
    DioOption dio_opt;
    InitializeDioOption(&dio_opt);
    int f0_length = GetSamplesForDIO(fs, x_length, dio_opt.frame_period);
    double *f0 = new double[f0_length];
    double *tp = new double[f0_length];
    Dio(x, x_length, fs, &dio_opt, tp, f0);

    // D4C
    D4COption d4c_opt;
    InitializeD4COption(&d4c_opt);
    double **aperiodicity = new double *[f0_length];
    int bins = kFftSize / 2 + 1;
    for (int i = 0; i < f0_length; ++i) {
      aperiodicity[i] = new double[bins];
    }
    D4C(x, x_length, fs, tp, f0, f0_length, kFftSize, &d4c_opt, aperiodicity);

    write_in(outdir + "/" + name + ".in", static_cast<double>(fs), x_length, x);
    write_out(outdir + "/" + name + ".out", f0_length, kFftSize, f0, tp, aperiodicity);

    printf("%s: fs=%d x_length=%d f0_length=%d fft_size=%d\n",
           name, fs, x_length, f0_length, kFftSize);

    for (int i = 0; i < f0_length; ++i) delete[] aperiodicity[i];
    delete[] aperiodicity;
    delete[] f0;
    delete[] tp;
    delete[] x;
  }

  return 0;
}

