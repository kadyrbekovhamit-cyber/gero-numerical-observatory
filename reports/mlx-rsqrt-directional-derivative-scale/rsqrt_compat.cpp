#include "mlx/mlx.h"
#include <bit>
#include <cstdint>
#include <iostream>
#include <limits>
namespace mx = mlx::core;
uint32_t bits(float x) { return std::bit_cast<uint32_t>(x); }
int main() {
  mx::set_default_device(mx::Device(mx::Device::cpu));
  float inf = std::numeric_limits<float>::infinity();
  float nan = std::numeric_limits<float>::quiet_NaN();
  std::cout << "op,x_bits,v_bits,forward_bits,jvp_bits,vjp_bits\n";
  for (bool reciprocal : {false, true}) {
    auto f = [reciprocal](const mx::array& a) {
      return reciprocal ? mx::rsqrt(a) : mx::sqrt(a);
    };
    for (float a : {-inf, -4.f, -0.f, 0.f, 1.f, 4.f, inf, nan}) {
      for (float b : {-3.f, -0.f, 0.f, 4.f}) {
        auto j = mx::jvp(f, mx::array(a), mx::array(b));
        auto v = mx::vjp(f, mx::array(a), mx::array(b));
        mx::eval(j.first, j.second, v.second);
        std::cout << (reciprocal ? "rsqrt" : "sqrt") << ',' << bits(a)
                  << ',' << bits(b) << ',' << bits(j.first.item<float>())
                  << ',' << bits(j.second.item<float>())
                  << ',' << bits(v.second.item<float>()) << '\n';
      }
    }
  }
}
