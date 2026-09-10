#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "mlx/mlx.h"
namespace mx = mlx::core;
using A = mx::array;
int checks = 0, failures = 0;

std::vector<double> values(const A& a) {
  auto x = mx::contiguous(mx::astype(a, mx::float64));
  mx::eval(x);
  return {x.data<double>(), x.data<double>() + x.size()};
}
void json_vector(const std::vector<double>& v) {
  std::cout << '[';
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) std::cout << ',';
    if (std::isfinite(v[i])) std::cout << std::setprecision(17) << v[i];
    else std::cout << (std::isnan(v[i]) ? "\"NaN\"" : (v[i] > 0 ? "\"Infinity\"" : "\"-Infinity\""));
  }
  std::cout << ']';
}
A from(const std::vector<double>& v, const mx::Shape& shape, mx::Dtype dtype) {
  return mx::astype(A(v.data(), shape, mx::float64), dtype);
}

void test(mx::Shape shape, int axis, bool reverse, bool inclusive,
          int pattern, int cotangent, mx::Dtype dtype, bool strided = false) {
  int n = 1, stride = 1;
  for (int d : shape) n *= d;
  for (int d = axis + 1; d < shape.size(); ++d) stride *= shape[d];
  int length = shape[axis];
  double shift = pattern == 1 ? 1000. : pattern == 2 ? 100000000. :
      pattern == 3 ? -100000000. : pattern == 4 ? 1e20 : 0.;
  std::vector<double> input(n), cot(n);
  for (int i = 0; i < n; ++i) {
    int position = (i / stride) % length;
    input[i] = shift + (pattern == 5 ? (position == length - 1 ? 1e8 : 0.) :
        pattern == 6 ? (position == 0 ? 1e8 : 0.) : (i % 3) * .25);
    cot[i] = cotangent == 0 ? 1. : cotangent == 1 ? (i % 2 ? -.75 : .25) :
        cotangent == 2 ? (position == (reverse ? 0 : length - 1) ? 1. : 0.) : 0.;
    if (!inclusive && position == (reverse ? length - 1 : 0)) cot[i] = 0.;
  }
  A x = from(input, shape, dtype);
  if (strided) x = mx::swapaxes(mx::contiguous(mx::swapaxes(x, -1, -2)), -1, -2);
  input = values(x);
  A g = from(cot, shape, dtype);
  std::vector<double> expected(n, 0.);
  for (int row = 0; row < n; ++row) {
    int pos = (row / stride) % length;
    int base = row - pos * stride;
    std::vector<int> ids;
    for (int k = 0; k < length; ++k) {
      bool inside = reverse ? (inclusive ? k >= pos : k > pos) :
          (inclusive ? k <= pos : k < pos);
      if (inside) ids.push_back(base + k * stride);
    }
    if (ids.empty()) continue;
    double maximum = -INFINITY;
    for (int k : ids) maximum = std::max(maximum, input[k]);
    double denominator = 0.;
    for (int k : ids) denominator += std::exp(input[k] - maximum);
    for (int k : ids) expected[k] += cot[row] * std::exp(input[k] - maximum) / denominator;
  }
  std::string label = "n" + std::to_string(n) + "-a" + std::to_string(axis) +
      "-r" + std::to_string(reverse) + "-i" + std::to_string(inclusive) +
      "-p" + std::to_string(pattern) + "-g" + std::to_string(cotangent) +
      (dtype == mx::float64 ? "-f64" : "-f32") + (strided ? "-strided" : "");
  ++checks;
  try {
    auto result = mx::vjp([&](const std::vector<A>& a) {
      return std::vector<A>{mx::logcumsumexp(a[0], axis, reverse, inclusive)};
    }, {x}, {g}).second[0];
    auto actual = values(result);
    bool ok = result.shape() == shape && actual.size() == expected.size();
    double error = 0.;
    double tolerance = dtype == mx::float64 ? 2e-12 : 2e-5;
    for (int i = 0; i < n; ++i) {
      error = std::max(error, std::abs(actual[i] - expected[i]));
      ok &= std::isfinite(actual[i]) &&
          std::abs(actual[i] - expected[i]) <= tolerance * (1 + std::abs(expected[i]));
    }
    if (!ok) ++failures;
    std::cout << "CASE {\"label\":\"" << label << "\",\"passed\":" << (ok ? "true" : "false")
              << ",\"input\":";
    json_vector(input);
    std::cout << ",\"cotangent\":";
    json_vector(cot);
    std::cout << ",\"actual\":";
    json_vector(actual);
    std::cout << ",\"expected\":";
    json_vector(expected);
    std::cout << "}\n";
  } catch (const std::exception& e) {
    ++failures;
    std::cout << "ERROR " << label << ' ' << e.what() << '\n';
  }
}
int main() {
  mx::set_default_device(mx::Device(mx::Device::cpu));
  for (auto dtype : {mx::float32, mx::float64})
    for (auto shape : {mx::Shape{1}, mx::Shape{2}, mx::Shape{4}, mx::Shape{2, 3}})
      for (int axis = 0; axis < shape.size(); ++axis)
        for (bool reverse : {false, true})
          for (bool inclusive : {false, true})
            for (int pattern = 0; pattern < 7; ++pattern)
              for (int cotangent = 0; cotangent < 4; ++cotangent)
                test(shape, axis, reverse, inclusive, pattern, cotangent, dtype);
  test({2, 3}, 1, false, true, 4, 1, mx::float32, true);
  test({2, 3}, 0, true, false, 4, 1, mx::float64, true);
  std::cout << "SUMMARY checks=" << checks << " failures=" << failures << '\n';
  return failures ? 1 : 0;
}
