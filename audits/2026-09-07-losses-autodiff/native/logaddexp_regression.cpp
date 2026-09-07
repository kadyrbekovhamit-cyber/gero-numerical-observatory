#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include "mlx/mlx.h"

using namespace mlx::core;
int checks = 0;
int failures = 0;

void check(const std::string& name, const array& output, double expected, double rtol) {
  ++checks;
  auto flat = reshape(astype(output, float32), {-1});
  eval(flat);
  bool ok = true;
  for (size_t i = 0; i < flat.size(); ++i) {
    double got = flat.data<float>()[i];
    ok &= std::isnan(expected) ? std::isnan(got)
                              : std::isfinite(got) && std::abs(got - expected) <= rtol * std::abs(expected);
  }
  if (!ok) {
    ++failures;
    std::cout << "FAIL " << name << " got=" << flat.data<float>()[0]
              << " expected=" << expected << '\n';
  }
}

int main(int argc, char** argv) {
  const bool gpu = argc > 1 && std::string(argv[1]) == "gpu";
  set_default_device(gpu ? Device::gpu : Device::cpu);
  auto fun = [](const std::vector<array>& inputs) {
    return std::vector<array>{logaddexp(inputs[0], inputs[1])};
  };
  for (auto dtype : {float32, float16, bfloat16}) {
    const double rtol = dtype == float32 ? 2e-6 : (dtype == float16 ? 0.003 : 0.02);
    const std::vector<float> gaps = dtype == float16 ? std::vector<float>{0, 1, 5, 8}
                                                    : std::vector<float>{0, 1, 5, 10, 20, 40, 80};
    for (auto shape : {Shape{1}, Shape{2, 3}, Shape{2, 3, 5}}) {
      for (float gap : gaps) {
        auto a = full(shape, gap, dtype);
        auto b = zeros(shape, dtype);
        auto one = ones(shape, dtype);
        auto zero = zeros(shape, dtype);
        const double small = 1.0 / (1.0 + std::exp(double(gap)));
        auto out = vjp(fun, {a, b}, {multiply(one, array(3.0f, dtype))}).second;
        check("vjp_first", out[0], 3 * (1 - small), rtol);
        check("vjp_second", out[1], 3 * small, rtol);
        auto swapped = vjp(fun, {b, a}, {multiply(one, array(3.0f, dtype))}).second;
        check("vjp_swapped_first", swapped[0], 3 * small, rtol);
        check("vjp_swapped_second", swapped[1], 3 * (1 - small), rtol);
        check("jvp_second", jvp(fun, {a, b}, {zero, one}).second[0], small, rtol);
        check("jvp_swapped_first", jvp(fun, {b, a}, {one, zero}).second[0], small, rtol);
        check("translation_direction", jvp(fun, {a, b}, {one, one}).second[0], 1, rtol);
        check("forward", logaddexp(a, b), double(gap) + std::log1p(std::exp(-double(gap))), rtol);
      }
    }
  }
  const float inf = std::numeric_limits<float>::infinity();
  for (float first : {-inf, inf}) {
    auto out = vjp(fun, {array(first), array(2.0f)}, {array(1.0f)}).second;
    check("infinite_first", out[0], first < 0 ? 0 : 1, 0);
    check("infinite_second", out[1], first < 0 ? 1 : 0, 0);
  }
  auto undefined = vjp(fun, {array(-inf), array(-inf)}, {array(1.0f)}).second;
  check("undefined_first", undefined[0], NAN, 0);
  check("undefined_second", undefined[1], NAN, 0);
  std::cout << (gpu ? "Metal" : "CPU") << ": " << checks << " checks, " << failures << " failures\n";
  return failures ? 1 : 0;
}
