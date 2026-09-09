#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include "mlx/mlx.h"
#include "mlx/backend/common/hadamard.h"

namespace mx = mlx::core;
using A = mx::array;
int cases = 0, failed_cases = 0, checks = 0, failures = 0;

std::vector<double> values(const A& x) {
  auto a = mx::contiguous(mx::astype(x, mx::float32));
  mx::eval(a);
  return std::vector<double>(a.data<float>(), a.data<float>() + a.size());
}

void check(const std::string& label, const A& actual,
           const std::vector<double>& expected, double tolerance = 3e-5) {
  ++checks;
  auto a = values(actual);
  bool ok = a.size() == expected.size();
  double error = 0;
  for (size_t i = 0; i < a.size() && i < expected.size(); ++i) {
    error = std::max(error, std::abs(a[i] - expected[i]));
    ok &= std::isfinite(a[i]) &&
          std::abs(a[i] - expected[i]) <= tolerance * (1 + std::abs(expected[i]));
  }
  if (!ok) {
    ++failures;
    std::cout << "FAIL " << label << " max_error=" << error << '\n';
  }
}

std::vector<double> matrix(int size) {
  auto [n, m] = mx::decompose_hadamard(size);
  std::vector<double> h(m * m, 1);
  if (m > 1) {
    int i = 0;
    auto matrix_text = mx::hadamard_matrices().at(m);
    for (char c : matrix_text) {
      if (c == '+' || c == '-') h[i++] = c == '+' ? 1 : -1;
    }
  }
  std::vector<double> out(size * size);
  for (int j = 0; j < size; ++j)
    for (int i = 0; i < size; ++i)
      out[j * size + i] = h[(j / n) * m + i / n] *
          (__builtin_popcount((j % n) & (i % n)) % 2 ? -1 : 1);
  return out;
}

std::vector<double> multiply(const std::vector<double>& h,
                             const std::vector<float>& x, int n,
                             float scale, bool transpose = false) {
  std::vector<double> out(x.size());
  for (int b = 0; b < x.size() / n; ++b)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i)
        out[b * n + j] += scale * x[b * n + i] * h[transpose ? i * n + j : j * n + i];
  return out;
}

void test(int n, float scale, int batch, mx::Dtype dtype) {
  ++cases;
  int before = failures;
  std::string label = "n=" + std::to_string(n) + " scale=" + std::to_string(scale) +
      " batch=" + std::to_string(batch) + " dtype=" + std::to_string(static_cast<int>(dtype.val()));
  try {
    mx::Shape shape = batch == 1 ? mx::Shape{n} : mx::Shape{batch, n};
    std::vector<float> xs(batch * n), ts(batch * n), cs(batch * n);
    for (int i = 0; i < xs.size(); ++i) {
      xs[i] = ((i * 7) % 13 - 6) / 8.f;
      ts[i] = ((i * 3) % 11 - 5) / 8.f;
      cs[i] = ((i * 5) % 7 - 3) / 4.f;
    }
    auto arr = [&](const std::vector<float>& v) {return mx::astype(A(v.data(), shape, mx::float32), dtype);};
    A x = arr(xs), t = arr(ts), c = arr(cs);
    auto h = matrix(n);
    double tolerance = dtype == mx::float32 ? 3e-5 : 0.045;
    auto f = [scale](const A& x) {return mx::hadamard_transform(x, scale);};
    auto vf = [&](const std::vector<A>& x) {return std::vector<A>{f(x[0])};};
    auto [outputs, gradient] = mx::vjp(vf, {x}, {c});
    check(label + " forward", outputs[0], multiply(h, xs, n, scale), tolerance);
    check(label + " vjp", gradient[0], multiply(h, cs, n, scale, true), tolerance);
    check(label + " jvp", mx::jvp(vf, {x}, {t}).second[0], multiply(h, ts, n, scale), tolerance);
    auto negative = cs;
    for (auto& v : negative) v *= -2;
    check(label + " vjp_linearity", mx::vjp(vf, {x}, {mx::multiply(c, A(-2.f))}).second[0],
          multiply(h, negative, n, scale, true), tolerance);
    check(label + " zero_cotangent", mx::vjp(vf, {x}, {mx::zeros_like(c)}).second[0],
          std::vector<double>(xs.size(), 0), tolerance);
    auto loss = [&](const A& x) {return mx::multiply(A(.5f), mx::sum(mx::square(f(x))));};
    auto g = mx::grad(loss);
    double factor = double(n) * scale * scale;
    std::vector<double> gx(xs.size()), ht(ts.size());
    for (int i = 0; i < xs.size(); ++i) {gx[i] = factor * xs[i]; ht[i] = factor * ts[i];}
    check(label + " energy_gradient", g(x), gx, tolerance * 3);
    auto second = mx::grad([&](const A& x) {return mx::sum(mx::multiply(g(x), t));});
    check(label + " Hessian_reverse", second(x), ht, tolerance * 5);
    check(label + " Hessian_forward", mx::jvp([&](const std::vector<A>& a) {
      return std::vector<A>{g(a[0])};}, {x}, {t}).second[0], ht, tolerance * 5);
    if (dtype == mx::float32) {
      auto scalar_loss = [&](const A& a) {return loss(mx::add(x, mx::multiply(a, t)));};
      auto third = mx::grad(mx::grad(mx::grad(scalar_loss)));
      check(label + " third_derivative", third(A(0.f)), {0.}, 1e-4);
      if (n == 20 || n == 28) {
        float eps = 1.f / 512;
        auto hi = xs, lo = xs;
        hi[0] += eps; lo[0] -= eps;
        check(label + " first_coordinate_fd", mx::divide(mx::subtract(loss(arr(hi)), loss(arr(lo))), A(2 * eps)),
              {gx[0]}, 0.004);
      }
    }
    if (batch > 1) {
      auto vmapped_loss = mx::vmap(loss);
      check(label + " grad_vmap", mx::grad([&](const A& x) {return mx::sum(vmapped_loss(x));})(x), gx, tolerance * 3);
      check(label + " vmap_grad", mx::vmap(g)(x), gx, tolerance * 3);
      auto transposed = mx::swapaxes(x, 0, 1);
      check(label + " vmap_last", mx::swapaxes(mx::vmap(g, 1, 1)(transposed), 0, 1), gx, tolerance * 3);
    }
  } catch (const std::exception& e) {
    ++failures;
    std::cout << "FAIL " << label << " exception=" << e.what() << '\n';
  }
  if (failures != before) ++failed_cases;
}

int main() {
  mx::set_default_device(mx::Device(mx::Device::cpu));
  std::cout << std::setprecision(10);
  for (int n : {1, 2, 4, 12, 24, 48, 20, 40, 80, 28, 56, 112})
    for (float scale : {1.f / std::sqrt(float(n)), .25f, -.125f, 0.f})
      test(n, scale, 1, mx::float32);
  for (int n : {4, 12, 20, 28, 40, 56})
    test(n, 1.f / std::sqrt(float(n)), 2, mx::float32);
  for (auto dtype : {mx::float16, mx::bfloat16})
    for (int n : {12, 20, 28, 40, 56})
      test(n, 1.f / std::sqrt(float(n)), 2, dtype);
  std::cout << "SUMMARY cases=" << cases << " passed=" << cases - failed_cases
            << " failed_cases=" << failed_cases << " checks=" << checks
            << " failures=" << failures << '\n';
  return failures ? 1 : 0;
}
