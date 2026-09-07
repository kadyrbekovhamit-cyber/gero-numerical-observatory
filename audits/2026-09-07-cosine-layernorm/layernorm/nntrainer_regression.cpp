// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <layer_context.h>
#include <layer_normalization_layer.h>
#include <memory>
#include <numeric>
#include <tuple>
#include <var_grad.h>
#include <weight.h>

namespace {
struct LayerNormAuditResult {
  std::vector<float> output, dx, dgamma, dbeta;
};

LayerNormAuditResult evaluateLayerNormAudit(
  const std::vector<float> &x, const std::vector<float> &gamma,
  const std::vector<float> &dy, bool trainable = true,
  const std::array<unsigned int, 4> &shape = {1, 1, 1, 4},
  const std::vector<unsigned int> &axes = {3}, float epsilon = 1e-5f,
  bool backward = true, const std::vector<float> &beta = {}) {
  const nntrainer::TensorDim dim({shape[0], shape[1], shape[2], shape[3]});
  nntrainer::LayerNormalizationLayer layer;
  std::string axis_property = "axis=";
  for (unsigned int axis : axes)
    axis_property += std::to_string(axis) + ",";
  axis_property.pop_back();
  layer.setProperty({axis_property, "epsilon=" + std::to_string(epsilon)});
  nntrainer::InitLayerContext init({dim}, {true}, false, "ln_audit");
  layer.finalize(init);
  std::vector<std::unique_ptr<nntrainer::Weight>> weights;
  std::vector<std::unique_ptr<nntrainer::Var_Grad>> tensors;
  std::vector<nntrainer::Weight *> wp;
  std::vector<nntrainer::Var_Grad *> tp;
  for (const auto &spec : init.getWeightsSpec()) {
    weights.emplace_back(std::make_unique<nntrainer::Weight>(spec, true));
    wp.push_back(weights.back().get());
  }
  for (const auto &spec : init.getTensorsSpec()) {
    tensors.emplace_back(std::make_unique<nntrainer::Var_Grad>(spec, true));
    tp.push_back(tensors.back().get());
  }
  nntrainer::Var_Grad input(dim, nntrainer::Initializer::NONE, true, true,
                            "input");
  nntrainer::Var_Grad output(dim, nntrainer::Initializer::NONE, true, true,
                             "output");
  nntrainer::RunLayerContext context("ln_audit", trainable, 0.0f, false, 1.0f,
                                     nullptr, false, wp, {&input}, {&output},
                                     tp);
  std::copy(x.begin(), x.end(), input.getVariableRef().getData());
  std::copy(gamma.begin(), gamma.end(), context.getWeight(0).getData());
  context.getWeight(1).setZero();
  if (!beta.empty())
    std::copy(beta.begin(), beta.end(), context.getWeight(1).getData());
  std::copy(dy.begin(), dy.end(), output.getGradientRef().getData());
  layer.forwarding(context, true);
  if (backward) {
    layer.calcDerivative(context);
    if (trainable)
      layer.calcGradient(context);
  }
  for (size_t i = 0; i < x.size(); ++i) {
    EXPECT_EQ(input.getVariableRef().getData()[i], x[i]);
    EXPECT_EQ(output.getGradientRef().getData()[i], dy[i]);
  }
  auto copy = [](const nntrainer::Tensor &tensor) {
    const float *p = tensor.getData();
    return std::vector<float>(p, p + tensor.size());
  };
  return {copy(output.getVariableRef()),
          backward ? copy(input.getGradientRef()) : std::vector<float>{},
          backward && trainable ? copy(context.getWeightGrad(0))
                                : std::vector<float>{},
          backward && trainable ? copy(context.getWeightGrad(1))
                                : std::vector<float>{}};
}
} // namespace

TEST(LayerNormNumericalAudit, OrdinaryNonuniformGamma) {
  const std::vector<float> x{-1, 0, 1, 2}, gamma{1, 2, 4, 8}, dy{1, 1, 1, 1};
  const auto result = evaluateLayerNormAudit(x, gamma, dy);
  std::cout << "AUDIT_DX" << std::setprecision(10);
  for (float value : result.dx)
    std::cout << " " << value;
  std::cout << std::endl;
  const double mean = 0.5, variance = 1.25 + 1e-5;
  for (size_t i = 0; i < x.size(); ++i) {
    const double expected =
      ((gamma[i] - 3.75) - (x[i] - mean) * 2.875 / variance) /
      std::sqrt(variance);
    EXPECT_NEAR(result.dx[i], expected, 2e-6);
  }
}

namespace {
struct LayerNormAuditCase {
  std::array<unsigned int, 4> shape{2, 3, 2, 4};
  std::vector<unsigned int> axes{3};
  std::vector<float> x, gamma, dy, beta;
  float epsilon = 1e-5f;
};

// Map each NCHW element to its independent normalization group and affine
// weight.
std::pair<size_t, size_t> auditIndices(size_t index,
                                       const LayerNormAuditCase &c) {
  std::array<unsigned int, 4> coordinates{};
  for (int axis = 3; axis >= 0; --axis) {
    coordinates[axis] = index % c.shape[axis];
    index /= c.shape[axis];
  }
  size_t group = 0, weight = 0;
  for (unsigned int axis = 0; axis < 4; ++axis) {
    if (std::find(c.axes.begin(), c.axes.end(), axis) != c.axes.end())
      weight = weight * c.shape[axis] + coordinates[axis];
    else
      group = group * c.shape[axis] + coordinates[axis];
  }
  return {group, weight};
}

LayerNormAuditCase makeAuditCase(unsigned int mask) {
  LayerNormAuditCase c;
  c.axes.clear();
  size_t weights = 1;
  for (unsigned int axis = 1; axis < 4; ++axis) {
    if (mask & (1u << (axis - 1))) {
      c.axes.push_back(axis);
      weights *= c.shape[axis];
    }
  }
  for (int i = 0; i < 48; ++i) {
    c.x.push_back(((i * 7) % 19 - 9) * 0.25f);
    c.dy.push_back(((i * 11) % 13 - 6) * 0.5f);
  }
  for (size_t i = 0; i < weights; ++i) {
    c.gamma.push_back((int((i * 3) % 11) - 4) * 0.5f);
    c.beta.push_back((int(i % 5) - 2) * 0.25f);
  }
  return c;
}

LayerNormAuditResult evaluateAuditCase(const LayerNormAuditCase &c,
                                       bool trainable = true,
                                       bool backward = true) {
  return evaluateLayerNormAudit(c.x, c.gamma, c.dy, trainable, c.shape, c.axes,
                                c.epsilon, backward, c.beta);
}

struct LayerNormAuditReference {
  std::vector<double> output, dx, dgamma, dbeta;
};

// Independent FP64 equations evaluated from the original, rounded FP32 inputs.
LayerNormAuditReference referenceAuditCase(const LayerNormAuditCase &c) {
  const size_t n = c.x.size(), width = c.gamma.size(), groups = n / width;
  std::vector<double> mean(groups), variance(groups, c.epsilon);
  std::vector<double> mean_g(groups), mean_g_deviation(groups);
  for (size_t i = 0; i < n; ++i) {
    const auto [group, weight] = auditIndices(i, c);
    mean[group] += double(c.x[i]) / width;
  }
  for (size_t i = 0; i < n; ++i) {
    const auto [group, weight] = auditIndices(i, c);
    const double d = c.x[i] - mean[group],
                 g = double(c.dy[i]) * c.gamma[weight];
    variance[group] += d * d / width;
    mean_g[group] += g / width;
    mean_g_deviation[group] += g * d / width;
  }
  LayerNormAuditReference out{std::vector<double>(n), std::vector<double>(n),
                              std::vector<double>(width),
                              std::vector<double>(width)};
  for (size_t i = 0; i < n; ++i) {
    const auto [group, weight] = auditIndices(i, c);
    const double d = c.x[i] - mean[group], inv = 1 / std::sqrt(variance[group]);
    out.output[i] =
      c.gamma[weight] * d * inv + (c.beta.empty() ? 0 : c.beta[weight]);
    out.dx[i] = inv * (double(c.dy[i]) * c.gamma[weight] - mean_g[group] -
                       d * mean_g_deviation[group] / variance[group]);
    out.dgamma[weight] += c.dy[i] * d * inv;
    out.dbeta[weight] += c.dy[i];
  }
  return out;
}

void expectAuditVector(const std::vector<float> &actual,
                       const std::vector<double> &expected,
                       double atol = 3e-5) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_TRUE(std::isfinite(actual[i]));
    EXPECT_NEAR(actual[i], expected[i], atol + 3e-5 * std::abs(expected[i]));
  }
}

double auditObjective(const LayerNormAuditCase &c) {
  const auto result = evaluateAuditCase(c, true, false);
  double value = 0;
  for (size_t i = 0; i < c.x.size(); ++i)
    value += double(result.output[i]) * c.dy[i];
  return value;
}

class LayerNormAxesAudit
  : public ::testing::TestWithParam<std::tuple<int, bool>> {};
} // namespace

TEST_P(LayerNormAxesAudit, AffineOutputAndAllGradients) {
  const auto [mask, trainable] = GetParam();
  const auto c = makeAuditCase(mask);
  const auto actual = evaluateAuditCase(c, trainable);
  const auto expected = referenceAuditCase(c);
  expectAuditVector(actual.output, expected.output);
  expectAuditVector(actual.dx, expected.dx);
  if (trainable) {
    expectAuditVector(actual.dgamma, expected.dgamma);
    expectAuditVector(actual.dbeta, expected.dbeta);
  }
}

INSTANTIATE_TEST_SUITE_P(EveryNonBatchAxis, LayerNormAxesAudit,
                         ::testing::Combine(::testing::Range(1, 8),
                                            ::testing::Bool()));

TEST(LayerNormNumericalAudit,
     InputGradientMatchesActualForwardFiniteDifference) {
  LayerNormAuditCase c;
  c.shape = {1, 1, 1, 4};
  c.x = {-1, 0, 1, 2};
  c.gamma = {1, 2, 4, 8};
  c.dy = {1, 1, 1, 1};
  const auto result = evaluateAuditCase(c);
  constexpr float step = 1.0f / 256;
  for (size_t i = 0; i < c.x.size(); ++i) {
    auto plus = c, minus = c;
    plus.x[i] += step;
    minus.x[i] -= step;
    const double finite_difference =
      (auditObjective(plus) - auditObjective(minus)) / (2 * step);
    std::cout << "AUDIT_FINITE_DIFFERENCE " << i << " " << finite_difference
              << std::endl;
    EXPECT_NEAR(result.dx[i], finite_difference, 5e-4);
  }
}

TEST(LayerNormNumericalAudit, MultiAxisInputAndWeightsMatchFiniteDifference) {
  auto c = makeAuditCase(5);
  const auto actual = evaluateAuditCase(c);
  constexpr float step = 1.0f / 128;
  for (size_t i : {size_t(0), size_t(5), size_t(23), size_t(47)}) {
    auto plus = c, minus = c;
    plus.x[i] += step;
    minus.x[i] -= step;
    EXPECT_NEAR(actual.dx[i],
                (auditObjective(plus) - auditObjective(minus)) / (2 * step),
                1e-3);
  }
  for (size_t i = 0; i < c.gamma.size(); ++i) {
    auto plus = c, minus = c;
    plus.gamma[i] += step;
    minus.gamma[i] -= step;
    EXPECT_NEAR(actual.dgamma[i],
                (auditObjective(plus) - auditObjective(minus)) / (2 * step),
                1e-3);
    plus = c;
    minus = c;
    plus.beta[i] += step;
    minus.beta[i] -= step;
    EXPECT_NEAR(actual.dbeta[i],
                (auditObjective(plus) - auditObjective(minus)) / (2 * step),
                1e-3);
  }
}

TEST(LayerNormNumericalAudit, UnitGammaControl) {
  for (unsigned int mask = 1; mask < 8; ++mask) {
    auto c = makeAuditCase(mask);
    std::fill(c.gamma.begin(), c.gamma.end(), 1.0f);
    expectAuditVector(evaluateAuditCase(c).dx, referenceAuditCase(c).dx);
  }
}

TEST(LayerNormNumericalAudit, ZeroGammaMakesInputGradientZero) {
  auto c = makeAuditCase(4);
  std::fill(c.gamma.begin(), c.gamma.end(), 0.0f);
  expectAuditVector(evaluateAuditCase(c).dx,
                    std::vector<double>(c.x.size(), 0));
}

TEST(LayerNormNumericalAudit, ConstantGammaScalesInputGradient) {
  auto c = makeAuditCase(4);
  for (float gamma : {-2.0f, 0.5f, 3.0f}) {
    std::fill(c.gamma.begin(), c.gamma.end(), gamma);
    expectAuditVector(evaluateAuditCase(c).dx, referenceAuditCase(c).dx);
  }
}

TEST(LayerNormNumericalAudit, ConstantInputUsesEpsilon) {
  auto c = makeAuditCase(4);
  std::fill(c.x.begin(), c.x.end(), 2.0f);
  c.epsilon = 0.25f;
  const auto result = evaluateAuditCase(c);
  const auto expected = referenceAuditCase(c);
  expectAuditVector(result.output, expected.output);
  expectAuditVector(result.dx, expected.dx);
  expectAuditVector(result.dgamma, expected.dgamma);
}

TEST(LayerNormNumericalAudit, IndependentBatchPartition) {
  auto c = makeAuditCase(4);
  const auto together = evaluateAuditCase(c);
  auto one = c;
  one.shape[0] = 1;
  std::vector<double> weight_sum(c.gamma.size()), bias_sum(c.gamma.size());
  for (size_t batch = 0; batch < 2; ++batch) {
    one.x = {c.x.begin() + batch * 24, c.x.begin() + (batch + 1) * 24};
    one.dy = {c.dy.begin() + batch * 24, c.dy.begin() + (batch + 1) * 24};
    const auto apart = evaluateAuditCase(one);
    for (size_t i = 0; i < 24; ++i) {
      EXPECT_NEAR(apart.output[i], together.output[batch * 24 + i], 3e-6);
      EXPECT_NEAR(apart.dx[i], together.dx[batch * 24 + i], 3e-6);
    }
    for (size_t i = 0; i < weight_sum.size(); ++i) {
      weight_sum[i] += apart.dgamma[i];
      bias_sum[i] += apart.dbeta[i];
    }
  }
  expectAuditVector(together.dgamma, weight_sum);
  expectAuditVector(together.dbeta, bias_sum);
}

TEST(LayerNormNumericalAudit, EquivalentIndependentBatchShape) {
  auto c = makeAuditCase(4);
  const auto first = evaluateAuditCase(c);
  c.shape = {4, 1, 3, 4};
  const auto second = evaluateAuditCase(c);
  for (size_t i = 0; i < c.x.size(); ++i) {
    EXPECT_NEAR(first.output[i], second.output[i], 3e-6);
    EXPECT_NEAR(first.dx[i], second.dx[i], 3e-6);
  }
  for (size_t i = 0; i < c.gamma.size(); ++i)
    EXPECT_NEAR(first.dgamma[i], second.dgamma[i], 3e-6);
}

TEST(LayerNormNumericalAudit, ShiftInvarianceAndZeroSumInputGradient) {
  auto c = makeAuditCase(5);
  const auto initial = evaluateAuditCase(c);
  for (float &value : c.x)
    value += 4;
  const auto shifted = evaluateAuditCase(c);
  std::vector<double> gradient_sum(c.x.size() / c.gamma.size());
  for (size_t i = 0; i < c.x.size(); ++i) {
    EXPECT_NEAR(initial.output[i], shifted.output[i], 5e-6);
    EXPECT_NEAR(initial.dx[i], shifted.dx[i], 5e-6);
    gradient_sum[auditIndices(i, c).first] += initial.dx[i];
  }
  for (double sum : gradient_sum)
    EXPECT_NEAR(sum, 0, 3e-5);
}
