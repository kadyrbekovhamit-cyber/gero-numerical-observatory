// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <array>
#include <attention_layer.h>
#include <cmath>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <layer_context.h>
#include <memory>
#include <tuple>
#include <var_grad.h>
#include <weight.h>

namespace {
struct AttentionAuditResult {
  std::vector<float> output, dq, dv, dk;
};

AttentionAuditResult attentionAudit(
  const std::vector<float> &q, const std::vector<float> &v,
  const std::vector<float> &k, unsigned nq, unsigned nk,
  unsigned batch = 1, unsigned width = 1, bool causal = true,
  bool scaled = false, const std::vector<float> &dy = {},
  const std::vector<std::pair<unsigned, unsigned>> &chunks = {}) {
  const nntrainer::TensorDim qdim({batch, 1, nq, width});
  const nntrainer::TensorDim kdim({batch, 1, nk, width});
  nntrainer::AttentionLayer layer;
  layer.setProperty({std::string("causal_mask=") + (causal ? "true" : "false"),
                     std::string("scaled_dot_product=") + (scaled ? "true" : "false")});
  nntrainer::InitLayerContext init({qdim, kdim, kdim}, {true}, false, "attention_audit");
  layer.finalize(init);
  std::vector<std::unique_ptr<nntrainer::Var_Grad>> tensors;
  std::vector<nntrainer::Var_Grad *> tp;
  for (const auto &spec : init.getTensorsSpec()) {
    tensors.emplace_back(std::make_unique<nntrainer::Var_Grad>(spec, true));
    tp.push_back(tensors.back().get());
  }
  nntrainer::Var_Grad query(qdim, nntrainer::Initializer::NONE, true, true, "q");
  nntrainer::Var_Grad value(kdim, nntrainer::Initializer::NONE, true, true, "v");
  nntrainer::Var_Grad key(kdim, nntrainer::Initializer::NONE, true, true, "k");
  nntrainer::Var_Grad output(qdim, nntrainer::Initializer::NONE, true, true, "out");
  nntrainer::RunLayerContext context("attention_audit", true, 0.0f, false, 1.0f,
                                     nullptr, false, {}, {&query, &value, &key},
                                     {&output}, tp);
  std::copy(q.begin(), q.end(), query.getVariableRef().getData());
  std::copy(v.begin(), v.end(), value.getVariableRef().getData());
  std::copy(k.begin(), k.end(), key.getVariableRef().getData());
  output.getVariableRef().setZero();
  if (chunks.empty())
    layer.forwarding(context, true);
  else
    for (const auto &chunk : chunks)
      layer.incremental_forwarding(context, chunk.first, chunk.second, false);
  if (!dy.empty()) {
    std::copy(dy.begin(), dy.end(), output.getGradientRef().getData());
    layer.calcDerivative(context);
  }
  auto copy = [](const nntrainer::Tensor &x) {
    return std::vector<float>(x.getData(), x.getData() + x.size());
  };
  return {copy(output.getVariableRef()),
          dy.empty() ? std::vector<float>{} : copy(query.getGradientRef()),
          dy.empty() ? std::vector<float>{} : copy(value.getGradientRef()),
          dy.empty() ? std::vector<float>{} : copy(key.getGradientRef())};
}

void expectAttention(const std::vector<float> &actual,
                     const std::vector<float> &expected) {
  ASSERT_EQ(actual.size(), expected.size());
  std::cout << "ATTENTION_OUTPUT" << std::setprecision(10);
  for (float x : actual)
    std::cout << " " << x;
  std::cout << std::endl;
  for (size_t i = 0; i < actual.size(); ++i)
    EXPECT_NEAR(actual[i], expected[i], 3e-5) << "index " << i;
}
} // namespace

TEST(AttentionNumericalAudit, SquareControl) {
  expectAttention(attentionAudit({0, 0}, {2, 10}, {0, 0}, 2, 2).output, {2, 6});
}

TEST(AttentionNumericalAudit, RectangularFewerQueries) {
  expectAttention(attentionAudit({0, 0}, {2, 10, 50}, {0, 0, 0}, 2, 3).output, {2, 6});
}

TEST(AttentionNumericalAudit, RectangularMoreQueries) {
  expectAttention(attentionAudit({0, 0, 0}, {2, 10}, {0, 0}, 3, 2).output, {2, 6, 6});
}

TEST(AttentionNumericalAudit, FutureValueHasZeroDerivative) {
  auto result = attentionAudit({0, 0}, {2, 10, 50}, {0, 0, 0}, 2, 3,
                               1, 1, true, false, {1, 0});
  expectAttention(result.dv, {1, 0, 0});
}

TEST(AttentionNumericalAudit, LargeFiniteFutureLogitIsExcluded) {
  expectAttention(attentionAudit({1, 1}, {2, 10}, {0, 2e10f}, 2, 2).output, {2, 10});
}

TEST(AttentionNumericalAudit, IncrementalChunksMatchFull) {
  auto full = attentionAudit({0, 0, 0}, {2, 10, 50}, {0, 0, 0}, 3, 3);
  auto chunked = attentionAudit({0, 0, 0}, {2, 10, 50}, {0, 0, 0}, 3, 3,
                                1, 1, true, false, {}, {{0, 1}, {1, 3}});
  expectAttention(chunked.output, full.output);
}

TEST(AttentionNumericalAudit, UnmaskedControl) {
  expectAttention(attentionAudit({0, 0}, {2, 10, 50}, {0, 0, 0}, 2, 3,
                                 1, 1, false).output, {62.0f / 3, 62.0f / 3});
}

namespace {
struct AttentionCase {
  unsigned nq, nk, batch, width;
  bool scaled;
  std::vector<float> q, v, k, dy;
};

AttentionCase makeAttentionCase(unsigned nq, unsigned nk, unsigned batch,
                                unsigned width, bool scaled) {
  AttentionCase c{nq, nk, batch, width, scaled, {}, {}, {}, {}};
  for (unsigned i = 0; i < batch * nq * width; ++i) {
    c.q.push_back((int(i * 7 % 13) - 6) / 8.0f);
    c.dy.push_back((int(i * 5 % 11) - 5) / 4.0f);
  }
  for (unsigned i = 0; i < batch * nk * width; ++i) {
    c.k.push_back((int(i * 3 % 17) - 8) / 8.0f);
    c.v.push_back((int(i * 11 % 19) - 9) / 4.0f);
  }
  return c;
}

AttentionAuditResult evaluateCase(const AttentionCase &c, bool backward = true) {
  return attentionAudit(c.q, c.v, c.k, c.nq, c.nk, c.batch, c.width,
                        true, c.scaled, backward ? c.dy : std::vector<float>{});
}

AttentionAuditResult attentionReference(const AttentionCase &c) {
  AttentionAuditResult result;
  result.output.resize(c.q.size());
  std::vector<double> dq(c.q.size()), dv(c.v.size()), dk(c.k.size());
  const double scale = c.scaled ? 1.0 / std::sqrt(double(c.width)) : 1.0;
  for (unsigned b = 0; b < c.batch; ++b) {
    for (unsigned i = 0; i < c.nq; ++i) {
      const unsigned allowed = std::min(i + 1, c.nk);
      const unsigned qi = (b * c.nq + i) * c.width;
      std::vector<double> p(allowed), dp(allowed);
      for (unsigned j = 0; j < allowed; ++j) {
        for (unsigned d = 0; d < c.width; ++d)
          p[j] += double(c.q[qi + d]) * c.k[(b * c.nk + j) * c.width + d] * scale;
      }
      const double maximum = *std::max_element(p.begin(), p.end());
      double total = 0;
      for (double &x : p) {
        x = std::exp(x - maximum);
        total += x;
      }
      for (double &x : p)
        x /= total;
      for (unsigned d = 0; d < c.width; ++d) {
        double sum = 0;
        for (unsigned j = 0; j < allowed; ++j) {
          const unsigned kj = (b * c.nk + j) * c.width + d;
          sum += p[j] * c.v[kj];
          dp[j] += double(c.dy[qi + d]) * c.v[kj];
          dv[kj] += p[j] * c.dy[qi + d];
        }
        result.output[qi + d] = float(sum);
      }
      double center = 0;
      for (unsigned j = 0; j < allowed; ++j)
        center += p[j] * dp[j];
      for (unsigned j = 0; j < allowed; ++j) {
        const double ds = p[j] * (dp[j] - center) * scale;
        for (unsigned d = 0; d < c.width; ++d) {
          const unsigned kj = (b * c.nk + j) * c.width + d;
          dq[qi + d] += ds * c.k[kj];
          dk[kj] += ds * c.q[qi + d];
        }
      }
    }
  }
  result.dq.assign(dq.begin(), dq.end());
  result.dv.assign(dv.begin(), dv.end());
  result.dk.assign(dk.begin(), dk.end());
  return result;
}

using AttentionShape = std::tuple<unsigned, unsigned, unsigned, bool>;
class AttentionShapes : public ::testing::TestWithParam<AttentionShape> {};
} // namespace

TEST_P(AttentionShapes, ForwardAndBackwardMatchReference) {
  const auto [nq, nk, batch, scaled] = GetParam();
  const auto c = makeAttentionCase(nq, nk, batch, 3, scaled);
  const auto actual = evaluateCase(c), expected = attentionReference(c);
  expectAttention(actual.output, expected.output);
  expectAttention(actual.dq, expected.dq);
  expectAttention(actual.dv, expected.dv);
  expectAttention(actual.dk, expected.dk);
}

INSTANTIATE_TEST_SUITE_P(
  CausalShapes, AttentionShapes,
  ::testing::Values(AttentionShape{1, 3, 1, false}, AttentionShape{1, 3, 2, false},
                    AttentionShape{2, 3, 1, false}, AttentionShape{2, 3, 2, false},
                    AttentionShape{3, 2, 1, false}, AttentionShape{3, 2, 2, false},
                    AttentionShape{3, 3, 1, false}, AttentionShape{3, 3, 2, false},
                    AttentionShape{1, 3, 1, true}, AttentionShape{1, 3, 2, true},
                    AttentionShape{2, 3, 1, true}, AttentionShape{2, 3, 2, true},
                    AttentionShape{3, 2, 1, true}, AttentionShape{3, 2, 2, true},
                    AttentionShape{3, 3, 1, true}, AttentionShape{3, 3, 2, true}));

TEST(AttentionNumericalAudit, ProbabilitiesNormalizeOnlyOverAllowedKeys) {
  const unsigned nq = 2, nk = 3;
  auto result = attentionAudit(std::vector<float>(nq * nk, 0),
                               {1, 0, 0, 0, 1, 0, 0, 0, 1},
                               std::vector<float>(nk * nk, 0), nq, nk, 1, nk);
  expectAttention(result.output, {1, 0, 0, 0.5f, 0.5f, 0});
  for (unsigned i = 0; i < nq; ++i) {
    double total = 0;
    for (unsigned j = 0; j < nk; ++j) {
      const float p = result.output[i * nk + j];
      EXPECT_GE(p, 0);
      EXPECT_LE(p, 1);
      if (j > i)
        EXPECT_EQ(p, 0);
      total += p;
    }
    EXPECT_NEAR(total, 1, 2e-6);
  }
}

TEST(AttentionNumericalAudit, PrefixUnaffectedByFutureValues) {
  auto a = attentionAudit({0, 0}, {2, 10, 50}, {0, 0, 0}, 2, 3);
  auto b = attentionAudit({0, 0}, {2, 10, -100}, {0, 0, 0}, 2, 3);
  expectAttention(a.output, b.output);
}

TEST(AttentionNumericalAudit, BatchedEvaluationMatchesIndependentExamples) {
  const auto c = makeAttentionCase(2, 3, 2, 3, true);
  auto batched = evaluateCase(c);
  for (unsigned b = 0; b < 2; ++b) {
    auto single = c;
    single.batch = 1;
    single.q = {c.q.begin() + b * 6, c.q.begin() + (b + 1) * 6};
    single.dy = {c.dy.begin() + b * 6, c.dy.begin() + (b + 1) * 6};
    single.k = {c.k.begin() + b * 9, c.k.begin() + (b + 1) * 9};
    single.v = {c.v.begin() + b * 9, c.v.begin() + (b + 1) * 9};
    const auto actual = evaluateCase(single);
    expectAttention(actual.output, {batched.output.begin() + b * 6, batched.output.begin() + (b + 1) * 6});
    expectAttention(actual.dq, {batched.dq.begin() + b * 6, batched.dq.begin() + (b + 1) * 6});
    expectAttention(actual.dv, {batched.dv.begin() + b * 9, batched.dv.begin() + (b + 1) * 9});
    expectAttention(actual.dk, {batched.dk.begin() + b * 9, batched.dk.begin() + (b + 1) * 9});
  }
}

TEST(AttentionNumericalAudit, NativeForwardFiniteDifferences) {
  const auto original = makeAttentionCase(2, 3, 1, 2, true);
  const auto actual = evaluateCase(original);
  const auto reference = attentionReference(original);
  const double step = 1.0 / 256;
  auto objective = [](const AttentionCase &c) {
    const auto output = evaluateCase(c, false).output;
    double sum = 0;
    for (size_t i = 0; i < output.size(); ++i)
      sum += double(output[i]) * c.dy[i];
    return sum;
  };
  for (unsigned input = 0; input < 3; ++input) {
    const auto &gradient = input == 0 ? actual.dq : (input == 1 ? actual.dv : actual.dk);
    const auto &target = input == 0 ? reference.dq : (input == 1 ? reference.dv : reference.dk);
    for (size_t i = 0; i < gradient.size(); ++i) {
      auto plus = original, minus = original;
      auto &p = input == 0 ? plus.q : (input == 1 ? plus.v : plus.k);
      auto &m = input == 0 ? minus.q : (input == 1 ? minus.v : minus.k);
      p[i] += step;
      m[i] -= step;
      const double fd = (objective(plus) - objective(minus)) / (2 * step);
      EXPECT_NEAR(gradient[i], fd, 2e-4) << input << ":" << i;
      EXPECT_NEAR(target[i], fd, 2e-4) << input << ":" << i;
    }
  }
}

TEST(AttentionNumericalAudit, IncrementalPartitionsMatchReference) {
  const auto c = makeAttentionCase(5, 5, 1, 3, true);
  const auto expected = attentionReference(c);
  const std::vector<std::vector<unsigned>> partitions = {
    {5}, {1, 4}, {2, 3}, {3, 2}, {4, 1}, {1, 2, 2}, {2, 1, 2}, {1, 1, 1, 1, 1}};
  for (const auto &partition : partitions) {
    std::vector<std::pair<unsigned, unsigned>> chunks;
    unsigned start = 0;
    for (auto length : partition) {
      chunks.emplace_back(start, start + length);
      start += length;
    }
    auto actual = attentionAudit(c.q, c.v, c.k, c.nq, c.nk, c.batch,
                                 c.width, true, c.scaled, {}, chunks);
    expectAttention(actual.output, expected.output);
  }
}
