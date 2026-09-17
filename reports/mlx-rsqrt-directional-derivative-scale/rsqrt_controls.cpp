#include "mlx/mlx.h"
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>

namespace mx = mlx::core;
int main() {
  mx::set_default_device(mx::Device(mx::Device::cpu));
  std::function<mx::array(const mx::array&)> f =
      [](const mx::array& x) { return mx::rsqrt(x); };
  auto g1 = mx::grad(f);
  auto g2 = mx::grad(g1);
  auto g3 = mx::grad(g2);
  int count = 0, failed = 0;
  std::cout << "x,order,bits,passed\n";
  for (float v : {.125f, .25f, .5f, .75f, .999f, 1.f, 1.001f, 1.25f, 1.5f, 2.f, 4.f, 8.f, 16.f}) {
    mx::array x(v);
    auto d1 = g1(x), d2 = g2(x), d3 = g3(x);
    mx::eval(d1, d2, d3);
    float actual[] = {d1.item<float>(), d2.item<float>(), d3.item<float>()};
    double expected[] = {-.5 * std::pow(double(v), -1.5),
                          .75 * std::pow(double(v), -2.5),
                        -1.875 * std::pow(double(v), -3.5)};
    for (int j=0; j<3; ++j) {
      bool ok = std::isfinite(actual[j]) && std::abs(actual[j]-expected[j]) <= 2e-5*std::abs(expected[j]);
      ++count; if (!ok) ++failed;
      std::cout << v << ',' << j+1 << ',' << std::bit_cast<uint32_t>(actual[j]) << ',' << ok << '\n';
    }
  }
  std::cerr << "checks=" << count << " failures=" << failed << '\n';
  return failed ? 1 : 0;
}
