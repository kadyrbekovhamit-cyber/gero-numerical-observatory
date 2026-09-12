#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>
#include "mlx/mlx.h"

namespace mx = mlx::core;
using A = mx::array;
using VF = std::function<std::vector<A>(const std::vector<A>&)>;
int checks = 0, failures = 0;

A arr(const std::vector<double>& v, const mx::Shape& s, mx::Dtype d) {
  return mx::astype(A(v.data(), s, mx::float64), d);
}
std::vector<double> values(const A& x) {
  auto a = mx::contiguous(mx::astype(x, mx::float64));
  mx::eval(a);
  if (!a.size()) return {};
  return {a.data<double>(), a.data<double>() + a.size()};
}
std::string dtype(mx::Dtype d) { std::ostringstream s; s << d; return s.str(); }
void numbers(const std::vector<double>& v) {
  std::cout << '[';
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) std::cout << ',';
    if (std::isfinite(v[i])) std::cout << std::setprecision(17) << v[i];
    else std::cout << (std::isnan(v[i]) ? "\"NaN\"" : v[i] > 0 ? "\"Infinity\"" : "\"-Infinity\"");
  }
  std::cout << ']';
}
void check(const std::string& name, const A& actual, const A& expected, double tol = 0) {
  auto a = values(actual), b = values(expected);
  bool pass = actual.dtype() == expected.dtype() && actual.shape() == expected.shape();
  for (size_t i = 0; pass && i < a.size(); ++i) {
    pass = (a[i] == b[i]) || (std::isnan(a[i]) && std::isnan(b[i])) ||
           (std::isfinite(a[i]) && std::isfinite(b[i]) && std::abs(a[i]-b[i]) <= tol * (1+std::abs(b[i])));
  }
  ++checks; failures += !pass;
  std::cout << "CASE {\"name\":\"" << name << "\",\"actual_dtype\":\"" << actual.dtype()
            << "\",\"expected_dtype\":\"" << expected.dtype() << "\",\"actual\":";
  numbers(a); std::cout << ",\"expected\":"; numbers(b);
  std::cout << ",\"tolerance\":" << tol << ",\"passed\":" << (pass ? "true" : "false") << "}\n";
}
std::vector<double> bias_values(mx::Dtype d) {
  if (d == mx::float16) return {2., .125, -4., std::ldexp(1., -14)};
  if (d == mx::bfloat16) return {1048576., 1.0078125, std::ldexp(1., -30), -1048576.};
  if (d == mx::float32) return {70000., 1.000244140625, std::ldexp(1., -30), -70000.};
  return {std::ldexp(1., 50), 1.+std::ldexp(1., -40), std::ldexp(1., -160), -std::ldexp(1., 50)};
}
void constant_cases() {
  const std::vector<mx::Dtype> ds{mx::float16, mx::bfloat16, mx::float32, mx::float64};
  for (auto xd : ds) for (auto bd : ds) for (auto shape : {mx::Shape{4}, mx::Shape{2,4}, mx::Shape{2,3,4}}) {
    A x = mx::zeros(shape, xd), b = arr(bias_values(bd), {4}, bd);
    auto out = mx::promote_types(xd, bd);
    auto name = "constant/" + dtype(xd) + "/" + dtype(bd) + "/rank" + std::to_string(shape.size());
    A expected = mx::broadcast_to(mx::astype(b, out), shape);
    A y = mx::fast::layer_norm(x, std::nullopt, b, 1e-5f);
    A with_one = mx::fast::layer_norm(x, mx::ones({4}, xd), b, 1e-5f);
    check(name + "/bias_identity", y, expected);
    check(name + "/unit_weight_control", with_one, expected);
    check(name + "/unit_weight_invariance", y, with_one);
    VF f = [x](const std::vector<A>& z) { return std::vector<A>{mx::fast::layer_norm(x, std::nullopt, z[0], 1e-5f)}; };
    A tangent = arr(bias_values(bd), {4}, bd);
    check(name + "/bias_jvp", mx::jvp(f, {b}, {tangent}).second[0], expected);
    A g = mx::ones(shape, out);
    int rows = x.size()/4;
    A grad_expected = mx::full({4}, A(rows, bd), bd);
    check(name + "/bias_vjp", mx::vjp(f, {b}, {g}).second[0], grad_expected);
  }
}
void precision_gradient() {
  A x = mx::zeros({4}, mx::float16), b = mx::zeros({4}, mx::float32);
  A g = arr({70000., 1.000244140625, std::ldexp(1.,-30), -70000.}, {4}, mx::float32);
  VF f = [x](const std::vector<A>& z) { return std::vector<A>{mx::fast::layer_norm(x, std::nullopt, z[0], 1e-5f)}; };
  check("gradient/constant_bias_vjp_identity", mx::vjp(f, {b}, {g}).second[0], g);
}
void nonconstant_cases() {
  const std::vector<mx::Dtype> ds{mx::float16, mx::bfloat16, mx::float32, mx::float64};
  for (auto xd : ds) for (auto bd : ds) for (bool strided : {false, true}) {
    auto label = "nonconstant/" + dtype(xd) + "/" + dtype(bd) + "/strided" + std::to_string(strided);
    A x = arr({-1,1,-1,1,-1,1,-1,1}, {2,4}, xd);
    if (strided) x = mx::swapaxes(mx::contiguous(mx::swapaxes(x,0,1)),0,1);
    A b = arr(bias_values(bd), {4}, bd);
    auto out = mx::promote_types(xd, bd);
    auto bv = values(b);
    std::vector<double> y;
    for (int row = 0; row < 2; ++row) for (int j = 0; j < 4; ++j)
      y.push_back(bv[j] + (j%2 ? .5 : -.5));
    auto f = [b](const std::vector<A>& z) {
      return std::vector<A>{mx::fast::layer_norm(z[0], std::nullopt, b, 3.f)};
    };
    check(label+"/forward", f({x})[0], arr(y,{2,4},out));
    check(label+"/unit_weight", f({x})[0], mx::fast::layer_norm(x,mx::ones({4},xd),b,3.f));
    A t = arr({1,0,0,0,1,0,0,0}, {2,4}, xd);
    A g = arr({1,-2,3,-4,1,-2,3,-4}, {2,4}, out);
    check(label+"/input_jvp", mx::jvp(f,{x},{t}).second[0],
          arr({.34375,-.09375,-.15625,-.09375,.34375,-.09375,-.15625,-.09375}, {2,4}, out));
    check(label+"/input_vjp", mx::vjp(f,{x},{g}).second[0],
          arr({.4375,-.4375,1.4375,-1.4375,.4375,-.4375,1.4375,-1.4375}, {2,4}, xd));
  }
}
void controls() {
  const std::vector<mx::Dtype> ds{mx::float16, mx::bfloat16, mx::float32, mx::float64};
  for (auto d : ds) {
    A x=arr({-1,1,-1,1}, {4}, d), w=arr({2,4,-2,-4}, {4}, d);
    A b=arr({1,2,3,4}, {4}, d);
    auto name="control/"+dtype(d);
    check(name+"/no_params", mx::fast::layer_norm(x,std::nullopt,std::nullopt,3.f),arr({-.5,.5,-.5,.5},{4},d));
    check(name+"/weight_only",mx::fast::layer_norm(x,w,std::nullopt,3.f),arr({-1,2,1,-2},{4},d));
    check(name+"/both",mx::fast::layer_norm(x,w,b,3.f),arr({0,4,4,2},{4},d));
    check(name+"/bias_only_same_dtype",mx::fast::layer_norm(x,std::nullopt,b,3.f),arr({.5,2.5,2.5,4.5},{4},d));
  }
}
int main() {
  mx::set_default_device(mx::Device::cpu);
  constant_cases();
  precision_gradient();
  nonconstant_cases();
  controls();
  std::cout << "SUMMARY {\"checks\":" << checks << ",\"failures\":" << failures << "}\n";
  return failures ? 1 : 0;
}
