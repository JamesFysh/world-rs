#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <cstdint>
#include "world/dio.h"
#include "world/cheaptrick.h"
#include "world/stonemask.h"
#include "world/synthesis.h"

static void read_raw_input(const char* path, double& fs, int& x_length, double*& x) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    uint32_t len = 0;
    f.read(reinterpret_cast<char*>(&len), sizeof(len));
    f.read(reinterpret_cast<char*>(&fs), sizeof(fs));
    x_length = static_cast<int>(len);
    x = new double[x_length];
    f.read(reinterpret_cast<char*>(x), sizeof(double) * x_length);
}

static void write_raw(const char* path, const void* data, size_t size) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open %s for write\n", path); exit(1); }
    f.write(reinterpret_cast<const char*>(data), size);
}

static void run_dio(const char* in_path, const char* out_path) {
    double fs; int x_length; double* x;
    read_raw_input(in_path, fs, x_length, x);
    DioOption opt; InitializeDioOption(&opt);
    int f0_length = GetSamplesForDIO((int)fs, x_length, opt.frame_period);
    double* tp = new double[f0_length];
    double* f0 = new double[f0_length];
    Dio(x, x_length, (int)fs, &opt, tp, f0);
    std::ofstream f(out_path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(&f0_length), sizeof(f0_length));
    f.write(reinterpret_cast<const char*>(&opt.frame_period), sizeof(opt.frame_period));
    f.write(reinterpret_cast<const char*>(f0), sizeof(double)*f0_length);
    f.write(reinterpret_cast<const char*>(tp), sizeof(double)*f0_length);
    delete[] x; delete[] tp; delete[] f0;
}

static void run_stonemask(const char* in_path, const char* out_path) {
    double fs; int x_length; double* x;
    read_raw_input(in_path, fs, x_length, x);
    DioOption opt; InitializeDioOption(&opt);
    int f0_length = GetSamplesForDIO((int)fs, x_length, opt.frame_period);
    double* tp = new double[f0_length];
    double* f0 = new double[f0_length];
    Dio(x, x_length, (int)fs, &opt, tp, f0);
    double* refined = new double[f0_length];
    StoneMask(x, x_length, (int)fs, tp, f0, f0_length, refined);
    std::ofstream f(out_path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(&f0_length), sizeof(f0_length));
    f.write(reinterpret_cast<const char*>(refined), sizeof(double)*f0_length);
    delete[] x; delete[] tp; delete[] f0; delete[] refined;
}

static void run_cheaptrick(const char* in_path, const char* out_path) {
    double fs; int x_length; double* x;
    read_raw_input(in_path, fs, x_length, x);
    DioOption dio_opt; InitializeDioOption(&dio_opt);
    int f0_length = GetSamplesForDIO((int)fs, x_length, dio_opt.frame_period);
    double* tp = new double[f0_length];
    double* f0 = new double[f0_length];
    Dio(x, x_length, (int)fs, &dio_opt, tp, f0);
    CheapTrickOption ct_opt; InitializeCheapTrickOption((int)fs, &ct_opt);
    int fft_size = ct_opt.fft_size;
    double** sp = new double*[f0_length];
    for (int i=0;i<f0_length;i++) sp[i] = new double[fft_size/2+1];
    CheapTrick(x, x_length, (int)fs, tp, f0, f0_length, &ct_opt, sp);
    std::ofstream f(out_path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(&f0_length), sizeof(f0_length));
    f.write(reinterpret_cast<const char*>(&fft_size), sizeof(fft_size));
    f.write(reinterpret_cast<const char*>(f0), sizeof(double)*f0_length);
    f.write(reinterpret_cast<const char*>(tp), sizeof(double)*f0_length);
    for (int i=0;i<f0_length;i++) {
        f.write(reinterpret_cast<const char*>(sp[i]), sizeof(double)*(fft_size/2+1));
    }
    for (int i=0;i<f0_length;i++) delete[] sp[i];
    delete[] sp; delete[] x; delete[] tp; delete[] f0;
}

static void run_synthesis_with_ap(const char* in_path, const char* out_path, double ap_val) {
    double fs; int x_length; double* x;
    read_raw_input(in_path, fs, x_length, x);
    DioOption dio_opt; InitializeDioOption(&dio_opt);
    int f0_length = GetSamplesForDIO((int)fs, x_length, dio_opt.frame_period);
    double* tp = new double[f0_length];
    double* f0 = new double[f0_length];
    Dio(x, x_length, (int)fs, &dio_opt, tp, f0);
    CheapTrickOption ct_opt; InitializeCheapTrickOption((int)fs, &ct_opt);
    int fft_size = ct_opt.fft_size;
    double** sp = new double*[f0_length];
    for (int i=0;i<f0_length;i++) sp[i] = new double[fft_size/2+1];
    CheapTrick(x, x_length, (int)fs, tp, f0, f0_length, &ct_opt, sp);
    double** ap = new double*[f0_length];
    for (int i=0;i<f0_length;i++) {
        ap[i] = new double[fft_size/2+1];
        for (int j=0;j<=fft_size/2;j++) ap[i][j] = ap_val;
    }
    double frame_period = dio_opt.frame_period;
    int y_length = static_cast<int>((double)(f0_length-1)*frame_period/1000.0*(int)fs + 1.0);
    double* y = new double[y_length];
    Synthesis(f0, f0_length, const_cast<const double* const*>(reinterpret_cast<const double* const*>(sp)), const_cast<const double* const*>(reinterpret_cast<const double* const*>(ap)), fft_size, frame_period, (int)fs, y_length, y);
    std::ofstream f(out_path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(&y_length), sizeof(y_length));
    f.write(reinterpret_cast<const char*>(y), sizeof(double)*y_length);
    for (int i=0;i<f0_length;i++) { delete[] sp[i]; delete[] ap[i]; }
    delete[] sp; delete[] ap; delete[] y; delete[] x; delete[] tp; delete[] f0;
}

static void run_synthesis(const char* in_path, const char* out_path) {
    run_synthesis_with_ap(in_path, out_path, 0.5);
}

static void run_chain(const char* in_path, const char* out_path) {
    double fs; int x_length; double* x;
    read_raw_input(in_path, fs, x_length, x);
    DioOption dio_opt; InitializeDioOption(&dio_opt);
    int f0_length = GetSamplesForDIO((int)fs, x_length, dio_opt.frame_period);
    double* tp = new double[f0_length];
    double* f0 = new double[f0_length];
    Dio(x, x_length, (int)fs, &dio_opt, tp, f0);
    CheapTrickOption ct_opt; InitializeCheapTrickOption((int)fs, &ct_opt);
    int fft_size = ct_opt.fft_size;
    double** sp = new double*[f0_length];
    for (int i=0;i<f0_length;i++) sp[i] = new double[fft_size/2+1];
    CheapTrick(x, x_length, (int)fs, tp, f0, f0_length, &ct_opt, sp);
    double** ap = new double*[f0_length];
    for (int i=0;i<f0_length;i++) {
        ap[i] = new double[fft_size/2+1];
        for (int j=0;j<=fft_size/2;j++) ap[i][j] = 0.5;
    }
    double frame_period = dio_opt.frame_period;
    int y_length = static_cast<int>((double)(f0_length-1)*frame_period/1000.0*(int)fs + 1.0);
    double* y = new double[y_length];
    Synthesis(f0, f0_length, const_cast<const double* const*>(reinterpret_cast<const double* const*>(sp)), const_cast<const double* const*>(reinterpret_cast<const double* const*>(ap)), fft_size, frame_period, (int)fs, y_length, y);
    std::ofstream f(out_path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(&y_length), sizeof(y_length));
    f.write(reinterpret_cast<const char*>(y), sizeof(double)*y_length);
    for (int i=0;i<f0_length;i++) { delete[] sp[i]; delete[] ap[i]; }
    delete[] sp; delete[] ap; delete[] y; delete[] x; delete[] tp; delete[] f0;
}

static void run_synthesis_from_vec_const_ap(const char* in_path, const char* out_path, double ap_val) {
    std::ifstream f(in_path, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open %s\n", in_path); exit(1); }
    uint32_t f0_length = 0;
    uint32_t fft_size = 0;
    double frame_period_ms = 0.0;
    double fs = 0.0;
    f.read(reinterpret_cast<char*>(&f0_length), sizeof(f0_length));
    f.read(reinterpret_cast<char*>(&fft_size), sizeof(fft_size));
    f.read(reinterpret_cast<char*>(&frame_period_ms), sizeof(frame_period_ms));
    f.read(reinterpret_cast<char*>(&fs), sizeof(fs));
    int n = static_cast<int>(f0_length);
    int fft = static_cast<int>(fft_size);
    int half = fft / 2 + 1;
    double* f0 = new double[n];
    f.read(reinterpret_cast<char*>(f0), sizeof(double) * n);
    double** sp = new double*[n];
    for (int i=0;i<n;i++) {
        sp[i] = new double[half];
        f.read(reinterpret_cast<char*>(sp[i]), sizeof(double) * half);
    }
    // skip original aperiodicity
    for (int i=0;i<n;i++) {
        f.seekg(sizeof(double) * half, std::ios::cur);
    }
    double** ap = new double*[n];
    for (int i=0;i<n;i++) {
        ap[i] = new double[half];
        for (int j=0;j<half;j++) ap[i][j] = ap_val;
    }
    int y_length = static_cast<int>((double)(n - 1) * frame_period_ms / 1000.0 * fs + 1.0);
    double* y = new double[y_length];
    Synthesis(f0, n, const_cast<const double* const*>(reinterpret_cast<const double* const*>(sp)), const_cast<const double* const*>(reinterpret_cast<const double* const*>(ap)), fft, frame_period_ms, static_cast<int>(fs), y_length, y);
    std::ofstream out(out_path, std::ios::binary);
    if (!out) { fprintf(stderr, "cannot open %s for write\n", out_path); exit(1); }
    out.write(reinterpret_cast<const char*>(&y_length), sizeof(y_length));
    out.write(reinterpret_cast<const char*>(y), sizeof(double) * y_length);
    for (int i=0;i<n;i++) { delete[] sp[i]; delete[] ap[i]; }
    delete[] sp; delete[] ap; delete[] y; delete[] f0;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <phase> <input_raw> <output_raw>\n", argv[0]);
        return 1;
    }
    const char* phase = argv[1];
    const char* in_path = argv[2];
    const char* out_path = argv[3];
    if (strcmp(phase, "dio")==0) run_dio(in_path, out_path);
    else if (strcmp(phase, "stonemask")==0) run_stonemask(in_path, out_path);
    else if (strcmp(phase, "cheaptrick")==0) run_cheaptrick(in_path, out_path);
    else if (strcmp(phase, "synthesis")==0) run_synthesis(in_path, out_path);
    else if (strcmp(phase, "chain")==0) run_chain(in_path, out_path);
    else if (strncmp(phase, "synthesis_const_ap_", 19)==0) {
        double ap_val = atof(phase + 19);
        run_synthesis_with_ap(in_path, out_path, ap_val);
    }
    else if (strncmp(phase, "synthesis_vec_const_ap_", 23)==0) {
        double ap_val = atof(phase + 23);
        run_synthesis_from_vec_const_ap(in_path, out_path, ap_val);
    }
    else { fprintf(stderr, "unknown phase %s\n", phase); return 1; }
    return 0;
}
