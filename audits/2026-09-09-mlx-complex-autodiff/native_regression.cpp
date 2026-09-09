#include <cmath>
#include <complex>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "mlx/mlx.h"

namespace mx = mlx::core;
using C = std::complex<double>;
using A = mx::array;

struct Op {
  std::string name;
  std::function<A(const A&)> apply;
  std::function<C(C)> forward;
  std::function<C(C)> derivative;
  double real_point;
};

A arr(C z) { return A(mx::complex64_t(float(z.real()), float(z.imag()))); }
C value(const A& a) {
  mx::eval(a);
  auto z = a.item<mx::complex64_t>();
  return C(z.real(), z.imag());
}
bool near(C a, C b, double atol = 3e-6, double rtol = 3e-5) {
  return std::isfinite(std::abs(a)) && std::abs(a-b) <= atol + rtol*std::abs(b);
}

int main() {
  mx::set_default_device(mx::Device::cpu);
  std::cout << std::setprecision(15);
  std::vector<Op> ops{
    {"cos", [](const A& z){return mx::cos(z);}, [](C z){return std::cos(z);}, [](C z){return -std::sin(z);}, 0.25},
    {"arcsin", [](const A& z){return mx::arcsin(z);}, [](C z){return std::asin(z);}, [](C z){return 1.0/std::sqrt(1.0-z*z);}, 0.25},
    {"arccos", [](const A& z){return mx::arccos(z);}, [](C z){return std::acos(z);}, [](C z){return -1.0/std::sqrt(1.0-z*z);}, 0.25},
    {"arctan", [](const A& z){return mx::arctan(z);}, [](C z){return std::atan(z);}, [](C z){return 1.0/(1.0+z*z);}, 0.25},
    {"arcsinh", [](const A& z){return mx::arcsinh(z);}, [](C z){return std::asinh(z);}, [](C z){return 1.0/std::sqrt(1.0+z*z);}, 0.25},
    {"arccosh", [](const A& z){return mx::arccosh(z);}, [](C z){return std::acosh(z);}, [](C z){return 1.0/(std::sqrt(z-1.0)*std::sqrt(z+1.0));}, 1.5},
    {"arctanh", [](const A& z){return mx::arctanh(z);}, [](C z){return std::atanh(z);}, [](C z){return 1.0/(1.0-z*z);}, 0.25},
    {"exp", [](const A& z){return mx::exp(z);}, [](C z){return std::exp(z);}, [](C z){return std::exp(z);}, 0.25},
    {"sin", [](const A& z){return mx::sin(z);}, [](C z){return std::sin(z);}, [](C z){return std::cos(z);}, 0.25},
    {"log", [](const A& z){return mx::log(z);}, [](C z){return std::log(z);}, [](C z){return 1.0/z;}, 1.5}
  };
  int cases = 0, failed_cases = 0, assertions = 0, failures = 0;
  std::map<std::string,int> failure_kinds;
  for (const auto& op : ops) {
    auto fn = [&](const std::vector<A>& x){return std::vector<A>{op.apply(x[0])};};
    for (C z : {C(.25,.5), C(.25,-.5), C(-.25,.5), C(-.25,-.5)}) {
      for (C cot : {C(1,0), C(0,1), C(.75,-.375)}) {
        ++cases;
        int previous = failures;
        auto check = [&](const std::string& kind, C actual, C expected, double tol=3e-6) {
          ++assertions;
          if (!near(actual,expected,tol)) {
            ++failures;
            ++failure_kinds[kind];
            std::cout << "FAIL " << op.name << " " << kind << " z=" << z << " cot=" << cot
                      << " actual=" << actual << " expected=" << expected << '\n';
          }
        };
        C tangent(.5,.25), deriv = op.derivative(z);
        C v = value(mx::vjp(fn,{arr(z)},{arr(cot)}).second[0]);
        C j = value(mx::jvp(fn,{arr(z)},{arr(tangent)}).second[0]);
        check("forward",value(op.apply(arr(z))),op.forward(z));
        check("vjp",v,cot*std::conj(deriv));
        check("jvp",j,tangent*deriv);
        check("adjoint",C(std::real(std::conj(cot)*j)),C(std::real(std::conj(v)*tangent)));
        auto objective = [&](C x){return std::real(std::conj(cot)*value(op.apply(arr(x))));};
        double h = std::ldexp(1.0,-10);
        C fd((objective(z+h)-objective(z-h))/(2*h),
             (objective(z+C(0,h))-objective(z-C(0,h)))/(2*h));
        check("finite_difference",fd,cot*std::conj(deriv),3e-4);
        if (op.name == "arccosh" && z == C(-.25,.5) && cot == C(1,0)) {
          std::cout << "BRANCH arccosh jvp=" << j << " expected=" << tangent*deriv
                    << " vjp=" << v << " expected_vjp=" << std::conj(deriv) << '\n';
        }
        failed_cases += failures != previous;
      }
    }
    ++cases;
    int previous=failures;
    A x(float(op.real_point)), cot(0.75f), tangent(0.5f);
    auto v=mx::vjp(fn,{x},{cot}).second[0];
    auto j=mx::jvp(fn,{x},{tangent}).second[0];
    C d=op.derivative(C(op.real_point));
    assertions+=2;
    if (!near(C(v.item<float>()),0.75*d) || v.dtype()!=mx::float32) {
      ++failures; ++failure_kinds["real_vjp"];
    }
    if (!near(C(j.item<float>()),0.5*d) || j.dtype()!=mx::float32) {
      ++failures; ++failure_kinds["real_jvp"];
    }
    failed_cases += failures != previous;
  }
  ++cases;
  int previous = failures;
  auto loss = [](const A& x) { return mx::real(mx::cos(mx::multiply(arr(C(0,1)),x))); };
  auto objective = [&](const std::vector<A>& x) { return std::vector<A>{loss(x[0])}; };
  A start(1.0f);
  A gradient = mx::vjp(objective,{start},{A(1.0f)}).second[0];
  double g=gradient.item<float>();
  A next = mx::subtract(start,mx::multiply(A(0.01f),gradient));
  double old_loss=loss(start).item<float>(), new_loss=loss(next).item<float>();
  assertions+=2;
  if (!near(C(g),C(std::sinh(1.0)))) {++failures; ++failure_kinds["real_chain_gradient"];}
  if (!(new_loss < old_loss)) {++failures; ++failure_kinds["real_chain_descent"];}
  failed_cases += failures != previous;
  std::cout << "REAL_LOSS gradient=" << g << " expected=" << std::sinh(1.0)
            << " old_loss=" << old_loss << " new_loss=" << new_loss << '\n';
  std::cout << "SUMMARY cases=" << cases << " failed_cases=" << failed_cases
            << " assertions=" << assertions << " failures=" << failures << '\n';
  for (const auto& [kind,count] : failure_kinds) std::cout << "KIND " << kind << " " << count << '\n';
  return failures ? 1 : 0;
}
