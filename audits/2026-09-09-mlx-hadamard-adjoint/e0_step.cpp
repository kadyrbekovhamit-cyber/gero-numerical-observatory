#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include "mlx/mlx.h"
namespace mx=mlx::core;
int main() {
  mx::set_default_device(mx::Device(mx::Device::cpu));
  int checks=0, failures=0;
  std::cout << std::setprecision(10);
  for(int n : {20,28,40,56}) {
    std::vector<float> v(n,0); v[0]=1;
    mx::array x(v.data(),{n},mx::float32);
    auto loss=[](const mx::array& a) {
      return mx::multiply(mx::array(.5f),mx::sum(mx::square(mx::hadamard_transform(a))));
    };
    auto g=mx::grad(loss)(x);
    auto next=mx::subtract(x,mx::multiply(mx::array(.125f),g));
    float first=mx::take(g,mx::array(0)).item<float>();
    float before=loss(x).item<float>();
    float after=loss(next).item<float>();
    std::cout << "n=" << n << " gradient0=" << first << " loss_before=" << before << " loss_after=" << after << '\n';
    checks+=2;
    failures+=!(std::isfinite(first)&&std::abs(first-1.f)<1e-5);
    failures+=!(std::isfinite(after)&&std::abs(after-.3828125f)<1e-5);
  }
  std::cout << "SUMMARY supplemental_checks=" << checks << " failures=" << failures << '\n';
  return failures?1:0;
}
