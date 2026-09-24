#include "mlx/mlx.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>

namespace mx = mlx::core;

static uint32_t bits(float value) {
  return std::bit_cast<uint32_t>(value);
}

int main() {
  mx::set_default_device(mx::Device(mx::Device::cpu));
  std::cout << "exponent,forward_bits,d1_bits,d2_bits,d3_bits,forward,d1,d2,d3\n";
  for (int exponent : {0, 32, 64, 80}) {
    float c = std::ldexp(1.0f, exponent);
    auto fun = [c](mx::array t) {
      return mx::arctan(mx::multiply(mx::array(c), t));
    };
    auto d1fun = mx::grad(fun);
    auto d2fun = mx::grad(d1fun);
    auto d3fun = mx::grad(d2fun);
    mx::array t(1.0f);
    auto forward = fun(t);
    auto d1 = d1fun(t);
    auto d2 = d2fun(t);
    auto d3 = d3fun(t);
    mx::eval(forward, d1, d2, d3);
    float fv = forward.item<float>();
    float a = d1.item<float>();
    float b = d2.item<float>();
    float c3 = d3.item<float>();
    std::cout << exponent << ',' << bits(fv) << ',' << bits(a) << ','
              << bits(b) << ',' << bits(c3) << ',' << fv << ',' << a << ','
              << b << ',' << c3 << '\n';
  }
}
