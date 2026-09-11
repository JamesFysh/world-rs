// interp1q reference vector generator
// Generates deterministic inputs for interp1Q and dumps C++ reference output.
// Binary format (little-endian):
//   <name>.in : u32 y_len, f64 x0, f64 shift, y_len * f64 y, u32 xi_len, xi_len * f64 xi
//   <name>.out: u32 yi_len, yi_len * f64 yi
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "world/matlabfunctions.h"

void write_in(const std::string &path, double x0, double shift,
              const std::vector<double> &y,
              const std::vector<double> &xi) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { std::cerr << "cannot open " << path << "\n"; std::exit(1); }
    uint32_t y_len = static_cast<uint32_t>(y.size());
    f.write(reinterpret_cast<const char*>(&y_len), sizeof(y_len));
    f.write(reinterpret_cast<const char*>(&x0), sizeof(x0));
    f.write(reinterpret_cast<const char*>(&shift), sizeof(shift));
    f.write(reinterpret_cast<const char*>(y.data()), y_len * sizeof(double));
    uint32_t xi_len = static_cast<uint32_t>(xi.size());
    f.write(reinterpret_cast<const char*>(&xi_len), sizeof(xi_len));
    f.write(reinterpret_cast<const char*>(xi.data()), xi_len * sizeof(double));
}

void write_out(const std::string &path, const std::vector<double> &yi) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { std::cerr << "cannot open " << path << "\n"; std::exit(1); }
    uint32_t yi_len = static_cast<uint32_t>(yi.size());
    f.write(reinterpret_cast<const char*>(&yi_len), sizeof(yi_len));
    f.write(reinterpret_cast<const char*>(yi.data()), yi_len * sizeof(double));
}

int main(int argc, char **argv) {
    if (argc != 2) { std::cerr << "usage: " << argv[0] << " <out_dir>\n"; return 1; }
    std::string outdir = argv[1];

    // Case 1: grid-aligned exact hits, value-mode, boundary, negative shift
    // y = [0,10,20,30,40], x0=0, shift=1
    // xi includes exact nodes 0,1,2,3,4 and mid points 0.5,1.5, etc.
    // Also negative shift case: shift=-1, xi negative
    // We'll generate one combined case with multiple sub-cases concatenated?
    // Simpler: generate one case covering all requirements.
    double x0 = 0.0;
    double shift = 1.0;
    std::vector<double> y = {0.0, 10.0, 20.0, 30.0, 40.0};
    // exact hits
    std::vector<double> xi;
    for (int i = 0; i < 5; ++i) xi.push_back(static_cast<double>(i));
    // mid points
    for (int i = 0; i < 4; ++i) xi.push_back(i + 0.5);
    // boundary: just outside range
    xi.push_back(-0.5);
    xi.push_back(4.5);
    // negative shift case: we will handle separately with shift=-1
    // For now generate positive shift case
    std::vector<double> yi(xi.size(), 0.0);
    interp1Q(x0, shift, y.data(), static_cast<int>(y.size()), xi.data(), static_cast<int>(xi.size()), yi.data());

    write_in(outdir + "/interp1q_grid.in", x0, shift, y, xi);
    write_out(outdir + "/interp1q_grid.out", yi);

    // Negative shift case matching dc_correction usage
    double x0_neg = 0.0;
    double shift_neg = -1.0;
    std::vector<double> y_neg = {0.0, 10.0, 20.0, 30.0, 40.0};
    std::vector<double> xi_neg = {-0.5, -1.5, -2.2, -4.0, -4.5};
    std::vector<double> yi_neg(xi_neg.size(), 0.0);
    interp1Q(x0_neg, shift_neg, y_neg.data(), static_cast<int>(y_neg.size()), xi_neg.data(), static_cast<int>(xi_neg.size()), yi_neg.data());

    write_in(outdir + "/interp1q_negative.in", x0_neg, shift_neg, y_neg, xi_neg);
    write_out(outdir + "/interp1q_negative.out", yi_neg);

    // Boundary behavior: query points far outside
    double x0_b = 0.0;
    double shift_b = 1.0;
    std::vector<double> y_b = {7.0, 10.0, 20.0};
    std::vector<double> xi_b = {-1.5, 0.0, 1.0, 2.0, 3.0};
    std::vector<double> yi_b(xi_b.size(), 0.0);
    interp1Q(x0_b, shift_b, y_b.data(), static_cast<int>(y_b.size()), xi_b.data(), static_cast<int>(xi_b.size()), yi_b.data());

    write_in(outdir + "/interp1q_boundary.in", x0_b, shift_b, y_b, xi_b);
    write_out(outdir + "/interp1q_boundary.out", yi_b);

    std::cout << "interp1q vectors written\n";
    return 0;
}
