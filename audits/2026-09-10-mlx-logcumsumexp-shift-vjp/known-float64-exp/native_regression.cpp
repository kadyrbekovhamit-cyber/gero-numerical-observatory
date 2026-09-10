#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>
#include "mlx/mlx.h"
namespace mx = mlx::core;
int checks = 0, failures = 0;
void number(double x) {
  if (std::isfinite(x)) std::cout << std::setprecision(17) << x;
  else std::cout << (std::isnan(x) ? "\"NaN\"" : x > 0 ? "\"Infinity\"" : "\"-Infinity\"");
}
void test(std::vector<double> data, mx::Shape shape, bool strided, mx::Dtype dtype) {
  auto x = mx::astype(mx::array(data.data(), shape, mx::float64), dtype);
  if (strided) x = mx::swapaxes(mx::contiguous(mx::swapaxes(x, 0, 1)), 0, 1);
  auto actual = mx::contiguous(mx::astype(mx::exp(x), mx::float64));
  auto quantized = mx::contiguous(mx::astype(x, mx::float64));
  mx::eval(actual, quantized);
  for (int i = 0; i < x.size(); ++i) {
    double v = quantized.data<double>()[i], got = actual.data<double>()[i];
    double expected = std::exp(v);
    bool ok = std::isnan(expected) ? std::isnan(got) :
        std::isinf(expected) ? got == expected :
        expected == 0 ? got == 0 :
        std::isfinite(got) && std::abs(got/expected-1) <= (dtype == mx::float64 ? 2e-14 : 3e-6);
    ++checks; failures += !ok;
    std::cout << "CASE {\"dtype\":\"" << (dtype == mx::float64 ? "float64" : "float32")
              << "\",\"size\":" << x.size() << ",\"strided\":" << (strided ? "true" : "false")
              << ",\"input\":"; number(v);
    std::cout << ",\"actual\":"; number(got);
    std::cout << ",\"expected\":"; number(expected);
    std::cout << ",\"passed\":" << (ok ? "true" : "false") << "}\n";
  }
}
int main() {
  mx::set_default_device(mx::Device::cpu);
  std::vector<double> v{-700., -100., -89., -1., -1e-8, 0., 1e-8, .25, 1., 89., 100., 700.};
  for (double x : v) test({x}, {1}, false, mx::float64);
  test(v, {12}, false, mx::float64);
  test(v, {2,6}, true, mx::float64);
  test({-INFINITY, INFINITY, NAN, 0., 1e-8}, {5}, false, mx::float64);
  test({-10., -1., 0., .25, 1., 10.}, {6}, false, mx::float32);
  std::cout << "SUMMARY {\"checks\":" << checks << ",\"failures\":" << failures << "}\n";
  return failures ? 1 : 0;
}
