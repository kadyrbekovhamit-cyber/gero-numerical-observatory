#include <cmath>
#include <functional>
#include <iostream>
#include <optional>
#include <string>
#include <vector>
#include "mlx/mlx.h"

namespace mx = mlx::core;
using A = mx::array;
int scenarios = 0, failed_scenarios = 0, checks = 0, failures = 0;

std::vector<double> values(const A& x) {
  auto a = mx::contiguous(mx::astype(x, mx::float32));
  mx::eval(a);
  return {a.data<float>(), a.data<float>() + a.size()};
}

void check(const std::string& label, const A& actual,
           const std::vector<double>& expected, const mx::Shape& shape) {
  ++checks;
  auto a = values(actual);
  bool ok = actual.shape() == shape && a.size() == expected.size();
  double error = 0;
  for (size_t i = 0; i < a.size() && i < expected.size(); ++i) {
    error = std::max(error, std::abs(a[i] - expected[i]));
    ok &= std::isfinite(a[i]) &&
          std::abs(a[i] - expected[i]) <= 3e-5 * (1 + std::abs(expected[i]));
  }
  if (!ok) {
    ++failures;
    std::cout << "FAIL " << label << " max_error=" << error << '\n';
  }
}

void scenario(const std::string& label, const std::function<void()>& f) {
  ++scenarios;
  int before = failures;
  try { f(); } catch (const std::exception& e) {
    ++failures;
    std::cout << "FAIL " << label << " exception=" << e.what() << '\n';
  }
  if (failures != before) ++failed_scenarios;
}

A data(const mx::Shape& shape, int multiplier, int offset) {
  int size = 1;
  for (int n : shape) size *= n;
  std::vector<float> v(size);
  for (int i = 0; i < size; ++i) v[i] = ((i * multiplier + offset) % 11 - 5) / 8.f;
  return A(v.data(), shape, mx::float32);
}

std::optional<A> index(const std::vector<uint32_t>& v) {
  if (v.empty()) return std::nullopt;
  return A(v.data(), {static_cast<int>(v.size())}, mx::uint32);
}

struct Selection {
  std::string name;
  int ba, bw;
  std::vector<uint32_t> lhs, rhs;
  std::vector<int> il, ir;
  bool sorted;
};

void qmm(const Selection& c, int K, int N, bool trans, int bits,
         int group, bool strided = false) {
  int M = 2, Q = c.il.size(), rows = trans ? N : K, cols = trans ? K : N;
  int per_word = 32 / bits, words = cols / per_word, groups = cols / group;
  std::vector<uint32_t> codes(c.bw * rows * cols), packed(c.bw * rows * words);
  for (int e = 0; e < c.bw; ++e) for (int r = 0; r < rows; ++r)
    for (int col = 0; col < cols; ++col) {
      int i = (e * rows + r) * cols + col;
      codes[i] = (col + 3 * r + 2 * e) % (1 << bits);
      packed[(e * rows + r) * words + col / per_word] |=
          codes[i] << (bits * (col % per_word));
    }
  A w(packed.data(), {c.bw, rows, words}, mx::uint32);
  A x = data({c.ba, M, K}, 3, 2);
  A s = mx::add(mx::abs(data({c.bw, rows, groups}, 1, 0)), A(.125f));
  A b = data({c.bw, rows, groups}, 2, 1);
  A cot = data({Q, M, N}, 5, 1);
  if (strided) {
    x = mx::swapaxes(mx::contiguous(mx::swapaxes(x, -1, -2)), -1, -2);
    cot = mx::swapaxes(mx::contiguous(mx::swapaxes(cot, -1, -2)), -1, -2);
  }
  auto xv = values(x), sv = values(s), bv = values(b), cv = values(cot);
  std::vector<double> y(Q * M * N), dx(x.size()), ds(s.size()), db(b.size());
  for (int q = 0; q < Q; ++q) for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) for (int k = 0; k < K; ++k) {
      int r = trans ? n : k, col = trans ? k : n;
      int wi = (c.ir[q] * rows + r) * cols + col;
      int si = (c.ir[q] * rows + r) * groups + col / group;
      int xi = (c.il[q] * M + m) * K + k, oi = (q * M + m) * N + n;
      double weight = codes[wi] * sv[si] + bv[si];
      y[oi] += xv[xi] * weight;
      dx[xi] += cv[oi] * weight;
      ds[si] += cv[oi] * xv[xi] * codes[wi];
      db[si] += cv[oi] * xv[xi];
    }
  auto f = [&](const std::vector<A>& v) { return std::vector<A>{mx::gather_qmm(
      v[0], w, v[1], v[2], index(c.lhs), index(c.rhs), trans, group, bits,
      "affine", c.sorted)}; };
  std::string label = c.name + "/K=" + std::to_string(K) + "/N=" + std::to_string(N) +
      "/transpose=" + std::to_string(trans) + "/bits=" + std::to_string(bits) +
      "/group=" + std::to_string(group) + "/strided=" + std::to_string(strided);
  scenario(label + "/forward", [&] { check(label + "/y", f({x,s,b})[0], y, {Q,M,N}); });
  scenario(label + "/joint", [&] {
    auto g = mx::vjp(f, {x,s,b}, {cot}).second;
    check(label + "/dx", g[0], dx, x.shape());
    check(label + "/ds", g[1], ds, s.shape());
    check(label + "/db", g[2], db, b.shape());
  });
  for (int arg = 0; arg < 3; ++arg) scenario(label + "/only_" + std::to_string(arg), [&] {
    std::vector<A> primals = {x,s,b};
    auto g = mx::vjp([&](const std::vector<A>& v) {
      auto ps = primals; ps[arg] = v[0]; return f(ps);
    }, {primals[arg]}, {cot}).second[0];
    check(label + "/only_" + std::to_string(arg), g,
          arg == 0 ? dx : (arg == 1 ? ds : db), primals[arg].shape());
  });
  scenario(label + "/zero_negative", [&] {
    auto zero = mx::vjp(f, {x,s,b}, {mx::zeros_like(cot)}).second;
    check(label + "/zero_ds", zero[1], std::vector<double>(s.size()), s.shape());
    check(label + "/zero_db", zero[2], std::vector<double>(b.size()), b.shape());
    auto neg = mx::vjp(f, {x,s,b}, {mx::negative(cot)}).second;
    auto ns = ds, nb = db;
    for (auto& z : ns) z = -z;
    for (auto& z : nb) z = -z;
    check(label + "/negative_ds", neg[1], ns, s.shape());
    check(label + "/negative_db", neg[2], nb, b.shape());
  });
}

void loss_test() {
  scenario("one_hot_quadratic_loss", [] {
    std::vector<float> xs(32); xs[0] = 1;
    A x(xs.data(), {1,1,32}, mx::float32);
    A w = mx::full({1,32,4}, A(uint32_t(0x11111111)));
    A s = mx::ones({1,32,1}), b = mx::zeros({1,32,1});
    std::vector<float> targets(32); targets[0] = 2;
    A target(targets.data(), {1,1,32}, mx::float32);
    A ids({0}, {1}, mx::uint32);
    auto loss = [&](const A& bias) {
      auto y = mx::gather_qmm(x,w,s,bias,std::nullopt,ids,false,32,4);
      return mx::multiply(A(.5f), mx::sum(mx::square(mx::subtract(y,target))));
    };
    auto g = mx::grad(loss)(b);
    std::vector<double> expected(32); expected[0] = 30;
    check("loss/gradient", g, expected, b.shape());
    check("loss/initial", loss(b), {16.}, {});
    check("loss/after_step", loss(mx::subtract(b,mx::divide(g,A(32.f)))), {1.9375}, {});
    float eps = 1.f/1024;
    for (int coordinate : {0,1}) {
      std::vector<float> direction(32); direction[coordinate] = eps;
      A d(direction.data(), b.shape(), mx::float32);
      auto fd = mx::divide(mx::subtract(loss(mx::add(b,d)),loss(mx::subtract(b,d))),A(2*eps));
      check("loss/finite_difference_" + std::to_string(coordinate), fd,
            {coordinate == 0 ? 30. : 0.}, {});
    }
    auto actual = values(loss(mx::subtract(b,mx::divide(g,A(32.f)))));
    std::cout << "EVIDENCE initial_loss=16 next_loss=" << actual[0] << '\n';
  });
}

int main() {
  mx::set_default_device(mx::Device(mx::Device::cpu));
  std::vector<Selection> selections = {
    {"right_repeat_unsorted",3,3,{}, {0,0,2},{0,1,2},{0,0,2},false},
    {"right_repeat_sorted",3,3,{}, {0,0,2},{0,1,2},{0,0,2},true},
    {"both_explicit",3,3,{2,0,2},{1,2,1},{2,0,2},{1,2,1},false},
    {"left_broadcast",1,3,{}, {0,1,1},{0,0,0},{0,1,1},false},
    {"no_indices_fallback",3,3,{}, {},{0,1,2},{0,1,2},false}
  };
  for (const auto& c : selections) for (bool trans : {false,true})
    for (const auto& shape : std::vector<std::pair<int,int>>{{32,32},{32,64},{64,32}})
      qmm(c,shape.first,shape.second,trans,4,32);
  for (bool trans : {false,true}) {
    qmm(selections[0],64,64,trans,8,64);
    qmm(selections[1],64,64,trans,8,32,true);
  }
  loss_test();
  std::cout << "SUMMARY scenarios=" << scenarios << " failed_scenarios=" << failed_scenarios
            << " checks=" << checks << " failures=" << failures << '\n';
  return failures ? 1 : 0;
}
