#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <optional>
#include <string>
#include <vector>
#include "mlx/mlx.h"

namespace mx = mlx::core;
using A = mx::array;
int scenarios = 0, checks = 0, failures = 0, failed_scenarios = 0;

std::vector<double> values(const A& x) {
  auto y = mx::contiguous(mx::astype(x, mx::float32));
  mx::eval(y);
  return {y.data<float>(), y.data<float>() + y.size()};
}

void check(const std::string& label, const A& actual,
           const std::vector<double>& expected, const mx::Shape& shape) {
  ++checks;
  auto v = values(actual);
  bool ok = actual.shape() == shape && v.size() == expected.size();
  double error = 0;
  for (size_t i = 0; i < std::min(v.size(), expected.size()); ++i) {
    error = std::max(error, std::abs(v[i] - expected[i]));
    ok &= std::isfinite(v[i]) &&
        std::abs(v[i] - expected[i]) <= 1e-5 * (1 + std::abs(expected[i]));
  }
  if (!ok) {
    ++failures;
    std::cout << "FAIL " << label << " max_error=" << error << '\n';
  }
}

void scenario(const std::string& label, const std::function<void()>& f) {
  ++scenarios;
  int before = failures;
  try { f(); }
  catch (const std::exception& e) {
    ++failures;
    std::cout << "FAIL " << label << " exception=" << e.what() << '\n';
  }
  if (failures > before) ++failed_scenarios;
}

A data(mx::Shape shape, int seed) {
  int n = 1;
  for (int d : shape) n *= d;
  std::vector<float> v(n);
  for (int i = 0; i < n; ++i) v[i] = ((i * seed + 2) % 7 - 3) / 4.f;
  return A(v.data(), shape, mx::float32);
}

int index(const mx::Shape& s, int batch, int row, int col) {
  return ((s[0] == 1 ? 0 : batch) * s[1] + (s[1] == 1 ? 0 : row)) * s[2]
      + (s[2] == 1 ? 0 : col);
}

struct Case {
  std::string name;
  int M, N, K, bs, ba, bb, bo;
  bool ops = false, scalar_out = false, strided = false, out = true;
};

void run_case(const Case& c, float mask_value, bool patterned = false) {
  int B = std::max(c.ba, c.bb);
  int tm = (c.M + c.bs - 1) / c.bs, tn = (c.N + c.bs - 1) / c.bs;
  int tk = (c.K + c.bs - 1) / c.bs;
  mx::Shape os = c.scalar_out ? mx::Shape{1,1,1} : mx::Shape{c.bo,tm,tn};
  std::vector<A> p = {
      data({c.ba,c.M,c.K}, 3), data({c.bb,c.K,c.N}, 5),
      patterned ? data(os, 2) : mx::full(os, A(mask_value)),
      data({1,tm,tk}, 2), data({B,tk,tn}, 3)};
  A cot = data({B,c.M,c.N}, 4);
  if (c.strided) {
    p[0] = mx::swapaxes(mx::contiguous(mx::swapaxes(p[0], -1, -2)), -1, -2);
    p[1] = mx::swapaxes(mx::contiguous(mx::swapaxes(p[1], -1, -2)), -1, -2);
    cot = mx::swapaxes(mx::contiguous(mx::swapaxes(cot, -1, -2)), -1, -2);
  }
  std::vector<std::vector<double>> v, g;
  for (const auto& x : p) { v.push_back(values(x)); g.emplace_back(x.size(), 0); }
  auto cv = values(cot);
  std::vector<double> y(B*c.M*c.N);
  for (int batch = 0; batch < B; ++batch)
    for (int m = 0; m < c.M; ++m) for (int n = 0; n < c.N; ++n)
      for (int k = 0; k < c.K; ++k) {
        int ai = index(p[0].shape(), batch, m, k);
        int bi = index(p[1].shape(), batch, k, n);
        int oi = index(p[2].shape(), batch, m/c.bs, n/c.bs);
        int li = index(p[3].shape(), batch, m/c.bs, k/c.bs);
        int ri = index(p[4].shape(), batch, k/c.bs, n/c.bs);
        int yi = (batch*c.M + m)*c.N + n;
        double a = v[0][ai], b = v[1][bi], o = c.out ? v[2][oi] : 1;
        double l = c.ops ? v[3][li] : 1, r = c.ops ? v[4][ri] : 1;
        double d = cv[yi];
        y[yi] += a*b*o*l*r;
        g[0][ai] += d*b*o*l*r;
        g[1][bi] += d*a*o*l*r;
        if (c.out) g[2][oi] += d*a*b*l*r;
        if (c.ops) {
          g[3][li] += d*a*b*o*r;
          g[4][ri] += d*a*b*o*l;
        }
      }
  auto f = [&](const std::vector<A>& q) {
    return std::vector<A>{mx::block_masked_mm(q[0], q[1], c.bs,
        c.out ? std::make_optional(q[2]) : std::nullopt,
        c.ops ? std::make_optional(q[3]) : std::nullopt,
        c.ops ? std::make_optional(q[4]) : std::nullopt)};
  };
  std::string label = c.name + "/mask=" + std::to_string(mask_value)
      + "/pattern=" + std::to_string(patterned);
  scenario(label + "/forward", [&] { check(label, f(p)[0], y, {B,c.M,c.N}); });
  std::vector<int> active = {0,1};
  if (c.out) active.push_back(2);
  if (c.ops) { active.push_back(3); active.push_back(4); }
  auto derivatives = [&](const std::vector<int>& selected, const std::string& suffix) {
    scenario(label + suffix, [&] {
      std::vector<A> ps;
      for (int i : selected) ps.push_back(p[i]);
      auto grads = mx::vjp([&](const std::vector<A>& q) {
        auto input = p;
        for (int j = 0; j < selected.size(); ++j) input[selected[j]] = q[j];
        return f(input);
      }, ps, {cot}).second;
      for (int j = 0; j < selected.size(); ++j) {
        int arg = selected[j];
        check(label + suffix + "/arg=" + std::to_string(arg), grads[j], g[arg], p[arg].shape());
      }
    });
  };
  derivatives(active, "/joint");
  for (int arg : active) derivatives({arg}, "/only=" + std::to_string(arg));
  if (c.out) derivatives({2,0}, "/reordered");
}

void simple() {
  A a = mx::ones({2,3}), b = mx::ones({3,2});
  auto f = [&](const std::vector<A>& x) {
    return std::vector<A>{mx::sum(mx::block_masked_mm(a,b,32,x[0]))};
  };
  for (float m : {0.f,.5f,1.f,-2.f}) scenario("simple", [&] {
    A mask = mx::full({1,1}, A(m));
    auto grad = mx::vjp(f, {mask}, {A(1.f)}).second[0];
    float e = 1.f/1024;
    auto fd = mx::divide(mx::subtract(f({mx::add(mask,A(e))})[0],
                                     f({mx::subtract(mask,A(e))})[0]), A(2*e));
    check("simple/vjp", grad, {12}, {1,1});
    check("simple/fd", fd, {12}, {});
    std::cout << "EVIDENCE mask=" << m << " gradient=" << grad.item<float>()
              << " finite_difference=" << fd.item<float>() << '\n';
  });
  scenario("training", [&] {
    A mask = mx::zeros({1,1});
    auto loss = [&](const std::vector<A>& x) {
      auto y = mx::block_masked_mm(a,b,32,x[0]);
      return std::vector<A>{mx::multiply(A(.5f), mx::sum(mx::square(mx::subtract(y,A(3.f)))))};
    };
    auto grad = mx::vjp(loss,{mask},{A(1.f)}).second[0];
    auto next = loss({mx::subtract(mask,mx::divide(grad,A(36.f)))})[0];
    check("training/gradient", grad, {-36}, {1,1});
    check("training/next", next, {0}, {});
    std::cout << "EVIDENCE training initial=" << loss({mask})[0].item<float>()
              << " gradient=" << grad.item<float>() << " next=" << next.item<float>() << '\n';
  });
  for (bool m : {false,true}) scenario("boolean_output_control", [&] {
    A mask = mx::full({1,1}, A(m));
    auto g = mx::vjp([&](const std::vector<A>& p) {
      return std::vector<A>{mx::block_masked_mm(p[0],p[1],32,mask)};
    }, {a,b}, {mx::ones({2,2})}).second;
    check("boolean/da", g[0], std::vector<double>(6,m ? 2 : 0), a.shape());
    check("boolean/db", g[1], std::vector<double>(6,m ? 2 : 0), b.shape());
  });
}

int main() {
  mx::set_default_device(mx::Device(mx::Device::cpu));
  std::vector<Case> cases = {
      {"small",2,2,3,32,1,1,1},
      {"small_operands",2,2,3,32,1,1,1,true},
      {"partial_tiles",33,35,7,32,1,1,1},
      {"partial_all_masks",33,35,37,32,1,1,1,true},
      {"tile64",3,67,5,64,1,1,1,true},
      {"batch_lhs",3,35,7,32,1,2,1,true},
      {"batch_rhs",35,3,7,32,2,1,2,true},
      {"scalar_out",33,35,7,32,1,2,1,true,true},
      {"strided",7,33,5,32,1,1,1,true,false,true},
      {"no_output_mask",3,35,7,32,1,2,1,true,false,false,false}
  };
  for (const auto& c : cases) {
    for (float m : {0.f,.5f,1.f,-2.f}) run_case(c,m);
    run_case(c,0.f,true);
  }
  simple();
  std::cout << "SUMMARY scenarios=" << scenarios << " checks=" << checks
            << " failures=" << failures << " failed_scenarios=" << failed_scenarios << '\n';
  return failures ? 1 : 0;
}
