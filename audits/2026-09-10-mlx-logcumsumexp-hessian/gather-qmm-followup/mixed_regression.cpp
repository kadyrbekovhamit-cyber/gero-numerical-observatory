#include <cmath>
#include <functional>
#include <iostream>
#include "mlx/mlx.h"
namespace mx = mlx::core;
using A = mx::array;
int checks = 0, failures = 0;
void check(float actual, float expected) {
  ++checks;
  if (!std::isfinite(actual) || std::abs(actual-expected)>1e-4) ++failures;
}
int main() {
  mx::set_default_device(mx::Device(mx::Device::cpu));
  for (bool gathered : {true,false}) for (bool trans : {true,false})
    for (int param : {1,2}) {
      A x = mx::reshape(mx::concatenate({A({1.f}),mx::zeros({31})}),{1,1,32});
      A s = mx::ones({1,32,1}), b = mx::zeros({1,32,1});
      A w = mx::full({1,32,4},A(uint32_t(0x11111111)));
      A us = mx::reshape(x,s.shape()), ids = A({uint32_t(0)});
      if (!gathered) {
        w=mx::reshape(w,{32,4}); s=mx::reshape(s,{32,1});
        b=mx::reshape(b,{32,1}); us=mx::reshape(us,{32,1});
      }
      auto f = [&](const A& z,const A& q) {
        A scale=param==1 ? q : s, bias=param==2 ? q : b;
        return mx::sum(gathered
            ? mx::gather_qmm(z,w,scale,bias,std::nullopt,ids,trans,32,4)
            : mx::quantized_matmul(z,w,scale,bias,trans,32,4));
      };
      A q = param==1 ? s : b;
      std::function<A(const A&)> gx = [&](const A& m) {
        return mx::sum(mx::multiply(mx::grad(std::function<A(const A&)>(
            [&](const A& z){return f(z,m);}))(x),x));
      };
      std::function<A(const A&)> gq = [&](const A& z) {
        return mx::sum(mx::multiply(mx::grad(std::function<A(const A&)>(
            [&](const A& m){return f(z,m);}))(q),us));
      };
      float a = mx::sum(mx::multiply(mx::grad(gx)(q),us)).item<float>();
      float c = mx::sum(mx::multiply(mx::grad(gq)(x),x)).item<float>();
      float eps = 1.f/1024;
      float fd = mx::divide(mx::subtract(gx(mx::add(q,mx::multiply(A(eps),us))),
          gx(mx::subtract(q,mx::multiply(A(eps),us)))),A(2*eps)).item<float>();
      float expected = trans ? 1 : 32;
      check(a,expected); check(c,expected); check(fd,expected);
      std::cout << "EVIDENCE gathered=" << gathered << " transpose=" << trans
                << " param=" << param << " x_then_parameter=" << a
                << " parameter_then_x=" << c << " fd=" << fd
                << " expected=" << expected << '\n';
    }
  std::cout << "SUMMARY checks=" << checks << " failures=" << failures << '\n';
  return failures ? 1 : 0;
}
