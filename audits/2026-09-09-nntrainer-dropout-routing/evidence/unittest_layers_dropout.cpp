// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_dropout.cpp
 * @date 15 October 2021
 * @brief Dropout Layer Test
 * @see	https://github.com/nntrainer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <algorithm>
#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include <dropout.h>
#include <layer_context.h>
#include <layers_common_tests.h>
#include <var_grad.h>

auto semantic_dropout = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::DropOutLayer>,
  nntrainer::DropOutLayer::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(Dropout, LayerSemantics,
                     ::testing::Values(semantic_dropout));

auto dropout_inference_option =
  LayerGoldenTestParamOptions::SKIP_CALC_GRAD |
  LayerGoldenTestParamOptions::SKIP_CALC_DERIV |
  LayerGoldenTestParamOptions::FORWARD_MODE_INFERENCE;

auto dropout_20_training = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::DropOutLayer>, {"dropout_rate=0.2"},
  "2:3:2:3", "dropout_20_training.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT |
    LayerGoldenTestParamOptions::DROPOUT_MATCH_60_PERCENT,
  "nchw", "fp32", "fp32");

auto dropout_20_inference = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::DropOutLayer>, {"dropout_rate=0.2"},
  "2:3:2:3", "dropout_20_inference.nnlayergolden", dropout_inference_option,
  "nchw", "fp32", "fp32");

auto dropout_0_training = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::DropOutLayer>, {"dropout_rate=0.0"},
  "2:3:2:3", "dropout_0_training.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto dropout_100_training = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::DropOutLayer>, {"dropout_rate=1.0"},
  "2:3:2:3", "dropout_100_training.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

GTEST_PARAMETER_TEST(Dropout, LayerGoldenTest,
                     ::testing::Values(dropout_20_training, dropout_0_training,
                                       dropout_100_training,
                                       dropout_20_inference));

#ifdef ENABLE_FP16
auto dropout_20_training_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::DropOutLayer>, {"dropout_rate=0.2"},
  "2:3:2:3", "dropout_20_training_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT |
    LayerGoldenTestParamOptions::DROPOUT_MATCH_60_PERCENT,
  "nchw", "fp16", "fp16");

auto dropout_20_inference_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::DropOutLayer>, {"dropout_rate=0.2"},
  "2:3:2:3", "dropout_20_inference_w16a16.nnlayergolden",
  dropout_inference_option, "nchw", "fp16", "fp16");

auto dropout_0_training_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::DropOutLayer>, {"dropout_rate=0.0"},
  "2:3:2:3", "dropout_0_training_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto dropout_100_training_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::DropOutLayer>, {"dropout_rate=1.0"},
  "2:3:2:3", "dropout_100_training_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

GTEST_PARAMETER_TEST(Dropout16, LayerGoldenTest,
                     ::testing::Values(dropout_20_training_w16a16,
                                       dropout_0_training_w16a16,
                                       dropout_100_training_w16a16,
                                       dropout_20_inference_w16a16));
#endif

namespace {
using DropoutVectors = std::vector<std::vector<float>>;

struct DropoutRoutingResult {
  DropoutVectors output, dx;
};

DropoutRoutingResult runDropoutRouting(const DropoutVectors &x,
                                       const DropoutVectors &dy,
                                       const DropoutVectors &replay_masks = {},
                                       bool backward = true) {
  if (x.empty() || x.size() != dy.size() ||
      (!replay_masks.empty() && replay_masks.size() != x.size()))
    throw std::invalid_argument("dropout audit input count mismatch");
  nntrainer::DropOutLayer layer;
  layer.setProperty(
    {replay_masks.empty() ? "dropout_rate=0.0" : "dropout_rate=0.5"});
  std::vector<nntrainer::TensorDim> dims;
  for (size_t i = 0; i < x.size(); ++i) {
    if (x[i].empty() || x[i].size() != dy[i].size() ||
        (!replay_masks.empty() && replay_masks[i].size() != x[i].size()))
      throw std::invalid_argument("dropout audit shape mismatch");
    dims.emplace_back(1, 1, 1, static_cast<unsigned>(x[i].size()));
  }
  nntrainer::InitLayerContext init(dims, std::vector<bool>(x.size(), true),
                                   false, "dropout_routing_audit");
  layer.finalize(init);
  std::vector<std::unique_ptr<nntrainer::Var_Grad>> inputs, outputs, tensors;
  std::vector<nntrainer::Var_Grad *> ip, op, tp;
  for (const auto &spec : init.getTensorsSpec()) {
    tensors.emplace_back(std::make_unique<nntrainer::Var_Grad>(spec, true));
    tp.push_back(tensors.back().get());
  }
  for (size_t i = 0; i < x.size(); ++i) {
    inputs.emplace_back(std::make_unique<nntrainer::Var_Grad>(
      dims[i], nntrainer::Initializer::NONE, true, true, "input"));
    outputs.emplace_back(std::make_unique<nntrainer::Var_Grad>(
      init.getOutSpecs()[i].variable_spec.dim, nntrainer::Initializer::NONE,
      true, true, "output"));
    ip.push_back(inputs.back().get());
    op.push_back(outputs.back().get());
    std::copy(x[i].begin(), x[i].end(),
              ip[i]->getVariableRef().getData<float>());
    std::fill_n(ip[i]->getGradientRef().getData<float>(), x[i].size(), 0.0f);
    std::copy(dy[i].begin(), dy[i].end(),
              op[i]->getGradientRef().getData<float>());
  }
  nntrainer::RunLayerContext context("dropout_routing_audit", true, 0.0f, false,
                                     1.0f, nullptr, false, {}, ip, op, tp);
  // Replay is an existing runtime path: each input must reuse its own mask.
  if (!replay_masks.empty()) {
    context.reStoreData(true);
    for (size_t i = 0; i < replay_masks.size(); ++i)
      std::copy(replay_masks[i].begin(), replay_masks[i].end(),
                context.getTensor(i).getData<float>());
  }
  layer.forwarding(context, true);
  if (backward)
    layer.calcDerivative(context);
  DropoutRoutingResult result;
  for (size_t i = 0; i < x.size(); ++i) {
    EXPECT_TRUE(std::equal(x[i].begin(), x[i].end(),
                           ip[i]->getVariableRef().getData<float>()));
    EXPECT_TRUE(std::equal(dy[i].begin(), dy[i].end(),
                           op[i]->getGradientRef().getData<float>()));
    const auto *y = op[i]->getVariableRef().getData<float>();
    result.output.emplace_back(y, y + x[i].size());
    if (backward) {
      EXPECT_EQ(ip[i]->getGradientRef().size(), x[i].size());
      const auto *dx = ip[i]->getGradientRef().getData<float>();
      result.dx.emplace_back(dx, dx + x[i].size());
    }
  }
  return result;
}
} // namespace

TEST(DropoutRoutingAudit, ZeroRateSingleInputControl) {
  const DropoutVectors x{{1, 2, 3}}, dy{{-1, 2, 4}};
  const auto result = runDropoutRouting(x, dy);
  EXPECT_EQ(result.output, x);
  EXPECT_EQ(result.dx, dy);
}

TEST(DropoutRoutingAudit, ZeroRateRoutesEachIncomingGradientToItsInput) {
  const DropoutVectors x{{1, 2, 3}, {4, 5, 6}}, dy{{1, 2, 3}, {10, 20, 30}};
  const auto result = runDropoutRouting(x, dy);
  EXPECT_EQ(result.output, x);
  for (size_t i = 0; i < result.dx.size(); ++i) {
    std::cout << "DROPOUT_AUDIT_DX[" << i << "]";
    for (float v : result.dx[i])
      std::cout << " " << v;
    std::cout << std::endl;
    EXPECT_EQ(result.dx[i], dy[i]) << "input " << i;
  }
}

TEST(DropoutRoutingAudit, ZeroRateMatchesRuntimeFiniteDifferences) {
  const DropoutVectors x{{1, 2, 3}, {4, 5, 6}}, dy{{1, -2, 3}, {-4, 5, -6}};
  const auto result = runDropoutRouting(x, dy);
  const float step = 1.0f / 32;
  for (size_t i = 0; i < x.size(); ++i) {
    for (size_t j = 0; j < x[i].size(); ++j) {
      auto plus = x, minus = x;
      plus[i][j] += step;
      minus[i][j] -= step;
      const auto yp = runDropoutRouting(plus, dy, {}, false).output;
      const auto ym = runDropoutRouting(minus, dy, {}, false).output;
      double numeric = 0;
      for (size_t k = 0; k < dy.size(); ++k)
        for (size_t l = 0; l < dy[k].size(); ++l)
          numeric +=
            double(dy[k][l]) * (double(yp[k][l]) - ym[k][l]) / (2 * step);
      EXPECT_NEAR(result.dx[i][j], numeric, 1e-6)
        << "input " << i << " coordinate " << j;
    }
  }
}

TEST(DropoutRoutingAudit, ReplayUsesEachInputsOwnGradientAndMask) {
  const DropoutVectors x{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
    dy{{1, -2, 3}, {4, 5, -6}, {-7, 8, 9}},
    masks{{2, 0, 2}, {0, 2, 2}, {2, 2, 0}};
  const auto result = runDropoutRouting(x, dy, masks);
  for (size_t i = 0; i < x.size(); ++i)
    for (size_t j = 0; j < x[i].size(); ++j) {
      EXPECT_FLOAT_EQ(result.output[i][j], x[i][j] * masks[i][j]);
      EXPECT_FLOAT_EQ(result.dx[i][j], dy[i][j] * masks[i][j])
        << "input " << i << " coordinate " << j;
    }
}
