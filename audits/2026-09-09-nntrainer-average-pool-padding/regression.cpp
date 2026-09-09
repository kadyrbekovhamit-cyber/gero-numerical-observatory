// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_pooling.cpp
 * @date 6 July 2021
 * @brief Pooling2d Layer Test
 * @see	https://github.com/nntrainer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <algorithm>
#include <array>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <tuple>

#include <gtest/gtest.h>

#include <layer_context.h>
#include <layers_common_tests.h>
#include <pooling2d_layer.h>
#include <var_grad.h>

namespace {
struct PoolingPaddingResult {
  std::vector<float> output, dx;
};

PoolingPaddingResult
runPoolingPadding(const nntrainer::TensorDim &dim,
                  const std::vector<std::string> &properties,
                  const std::vector<float> &x, const std::vector<float> &dy,
                  bool backward = true) {
  nntrainer::Pooling2DLayer layer;
  layer.setProperty(properties);
  nntrainer::InitLayerContext init({dim}, {true}, false, "pool_padding_audit");
  layer.finalize(init);
  const auto out_dim = init.getOutSpecs()[0].variable_spec.dim;
  if (x.size() != dim.getDataLen() || dy.size() != out_dim.getDataLen())
    throw std::invalid_argument("pool padding audit shape mismatch");
  std::vector<std::unique_ptr<nntrainer::Var_Grad>> tensors;
  std::vector<nntrainer::Var_Grad *> tp;
  for (const auto &spec : init.getTensorsSpec()) {
    tensors.emplace_back(std::make_unique<nntrainer::Var_Grad>(spec, true));
    tp.push_back(tensors.back().get());
  }
  nntrainer::Var_Grad input(dim, nntrainer::Initializer::NONE, true, true,
                            "input");
  nntrainer::Var_Grad output(out_dim, nntrainer::Initializer::NONE, true, true,
                             "output");
  nntrainer::RunLayerContext context("pool_padding_audit", true, 0.0f, false,
                                     1.0f, nullptr, false, {}, {&input},
                                     {&output}, tp);
  std::copy(x.begin(), x.end(), input.getVariableRef().getData<float>());
  std::copy(dy.begin(), dy.end(), output.getGradientRef().getData<float>());
  layer.forwarding(context, true);
  if (backward)
    layer.calcDerivative(context);
  EXPECT_TRUE(
    std::equal(x.begin(), x.end(), input.getVariableRef().getData<float>()));
  EXPECT_TRUE(
    std::equal(dy.begin(), dy.end(), output.getGradientRef().getData<float>()));
  auto copy = [](const nntrainer::Tensor &t) {
    const float *data = t.getData<float>();
    return std::vector<float>(data, data + t.size());
  };
  return {copy(output.getVariableRef()),
          backward ? copy(input.getGradientRef()) : std::vector<float>{}};
}
} // namespace

TEST(PoolingPaddingAudit, SameEvenKernelConservesIncomingGradient) {
  const nntrainer::TensorDim dim({1, 1, 2, 2});
  const std::vector<float> x{1, 2, 3, 4}, dy{1, 1, 1, 1};
  const auto result = runPoolingPadding(
    dim, {"pooling=average", "pool_size=2,2", "stride=1,1", "padding=same"}, x,
    dy);
  EXPECT_EQ(result.output, (std::vector<float>{2.5f, 3.0f, 3.5f, 4.0f}));
  std::cout << "POOL_AUDIT_DX" << std::setprecision(10);
  for (float v : result.dx)
    std::cout << " " << v;
  std::cout << std::endl;
  const std::vector<float> expected{0.25f, 0.75f, 0.75f, 2.25f};
  for (size_t i = 0; i < expected.size(); ++i)
    EXPECT_FLOAT_EQ(result.dx[i], expected[i]);
  EXPECT_FLOAT_EQ(std::accumulate(result.dx.begin(), result.dx.end(), 0.0f),
                  4.0f);
}

TEST(PoolingPaddingAudit, SameEvenKernelMatchesRuntimeFiniteDifferences) {
  const nntrainer::TensorDim dim({1, 1, 2, 2});
  const std::vector<std::string> props{"pooling=average", "pool_size=2,2",
                                       "stride=1,1", "padding=same"};
  const std::vector<float> x{1, 2, 3, 4}, dy{1, -2, 3, 4};
  const auto result = runPoolingPadding(dim, props, x, dy);
  const float step = 1.0f / 32;
  for (size_t i = 0; i < x.size(); ++i) {
    auto plus = x, minus = x;
    plus[i] += step;
    minus[i] -= step;
    const auto yp = runPoolingPadding(dim, props, plus, dy, false).output;
    const auto ym = runPoolingPadding(dim, props, minus, dy, false).output;
    double numeric = 0;
    for (size_t j = 0; j < dy.size(); ++j)
      numeric += double(dy[j]) * (double(yp[j]) - ym[j]) / (2 * step);
    EXPECT_NEAR(result.dx[i], numeric, 1e-5) << "input index " << i;
  }
}

namespace {
struct PoolingPaddingCase {
  std::string name, pooling, padding_property;
  std::array<unsigned int, 4> shape;
  std::array<unsigned int, 2> kernel, stride;
  std::array<unsigned int, 4> padding;
};

class PoolingPaddingMatrix
  : public ::testing::TestWithParam<PoolingPaddingCase> {};

TEST_P(PoolingPaddingMatrix, MatchesIndependentWindowJacobian) {
  const auto &c = GetParam();
  const auto [batch, channels, height, width] = c.shape;
  const auto [pt, pb, pl, pr] = c.padding;
  const unsigned oh = (height + pt + pb - c.kernel[0]) / c.stride[0] + 1;
  const unsigned ow = (width + pl + pr - c.kernel[1]) / c.stride[1] + 1;
  const nntrainer::TensorDim dim({batch, channels, height, width});
  std::vector<float> x(batch * channels * height * width);
  std::vector<float> dy(batch * channels * oh * ow);
  for (size_t i = 0; i < x.size(); ++i)
    x[i] = (int((i * 7) % 29) - 14) * 0.25f;
  for (size_t i = 0; i < dy.size(); ++i)
    dy[i] = (int((i * 3) % 11) - 5) * 0.5f;
  std::vector<std::string> properties{"pooling=" + c.pooling,
                                      "padding=" + c.padding_property,
                                      "stride=" + std::to_string(c.stride[0]) +
                                        "," + std::to_string(c.stride[1])};
  if (c.pooling != "global_average")
    properties.push_back("pool_size=" + std::to_string(c.kernel[0]) + "," +
                         std::to_string(c.kernel[1]));

  // Build each output's incidence set from the finite input coordinates.
  // This reference does not read nntrainer's saved counters or loop bounds.
  std::vector<double> expected_y(dy.size(), 0), expected_dx(x.size(), 0);
  for (unsigned bc = 0; bc < batch * channels; ++bc) {
    for (unsigned a = 0; a < oh; ++a) {
      for (unsigned b = 0; b < ow; ++b) {
        const size_t o = (bc * oh + a) * ow + b;
        std::vector<size_t> members;
        for (unsigned i = 0; i < height; ++i) {
          for (unsigned j = 0; j < width; ++j) {
            const int local_h = int(i + pt) - int(a * c.stride[0]);
            const int local_w = int(j + pl) - int(b * c.stride[1]);
            if (local_h >= 0 && local_h < int(c.kernel[0]) && local_w >= 0 &&
                local_w < int(c.kernel[1]))
              members.push_back((bc * height + i) * width + j);
          }
        }
        ASSERT_FALSE(members.empty());
        if (c.pooling == "max") {
          size_t winner = members[0];
          for (size_t i : members)
            if (x[i] > x[winner])
              winner = i;
          expected_y[o] = x[winner];
          expected_dx[winner] += dy[o];
        } else {
          for (size_t i : members) {
            expected_y[o] += double(x[i]) / members.size();
            expected_dx[i] += double(dy[o]) / members.size();
          }
        }
      }
    }
  }
  const auto result = runPoolingPadding(dim, properties, x, dy);
  for (size_t i = 0; i < x.size(); ++i)
    EXPECT_NEAR(result.dx[i], expected_dx[i], 2e-6) << "dx " << i;
  for (size_t i = 0; i < dy.size(); ++i)
    EXPECT_NEAR(result.output[i], expected_y[i], 2e-6) << "output " << i;

  std::vector<float> scaled_dy = dy;
  for (float &v : scaled_dy)
    v *= -2;
  const auto scaled = runPoolingPadding(dim, properties, x, scaled_dy);
  const auto zero =
    runPoolingPadding(dim, properties, x, std::vector<float>(dy.size(), 0));
  for (size_t i = 0; i < x.size(); ++i) {
    EXPECT_NEAR(scaled.dx[i], -2 * result.dx[i], 2e-6);
    EXPECT_FLOAT_EQ(zero.dx[i], 0);
  }
  EXPECT_NEAR(std::accumulate(result.dx.begin(), result.dx.end(), 0.0),
              std::accumulate(dy.begin(), dy.end(), 0.0), 2e-5);
}

INSTANTIATE_TEST_SUITE_P(
  PaddingCases, PoolingPaddingMatrix,
  ::testing::Values(PoolingPaddingCase{"BothSame",
                                       "average",
                                       "same",
                                       {1, 1, 2, 2},
                                       {2, 2},
                                       {1, 1},
                                       {0, 1, 0, 1}},
                    PoolingPaddingCase{"BottomOnly",
                                       "average",
                                       "0,1,0,0",
                                       {1, 1, 2, 3},
                                       {2, 3},
                                       {1, 1},
                                       {0, 1, 0, 0}},
                    PoolingPaddingCase{"RightOnly",
                                       "average",
                                       "0,0,0,1",
                                       {1, 1, 3, 2},
                                       {3, 2},
                                       {1, 1},
                                       {0, 0, 0, 1}},
                    PoolingPaddingCase{"StrideTwo",
                                       "average",
                                       "same",
                                       {1, 1, 3, 5},
                                       {2, 2},
                                       {2, 2},
                                       {0, 1, 0, 1}},
                    PoolingPaddingCase{"BatchChannels",
                                       "average",
                                       "same",
                                       {2, 2, 2, 2},
                                       {2, 2},
                                       {1, 1},
                                       {0, 1, 0, 1}},
                    PoolingPaddingCase{"ExplicitBottomRight",
                                       "average",
                                       "0,1,0,2",
                                       {1, 1, 3, 4},
                                       {3, 3},
                                       {1, 1},
                                       {0, 1, 0, 2}},
                    PoolingPaddingCase{"SymmetricControl",
                                       "average",
                                       "1,1,1,1",
                                       {1, 1, 3, 3},
                                       {3, 3},
                                       {1, 1},
                                       {1, 1, 1, 1}},
                    PoolingPaddingCase{"ValidControl",
                                       "average",
                                       "valid",
                                       {2, 2, 4, 5},
                                       {2, 3},
                                       {1, 2},
                                       {0, 0, 0, 0}},
                    PoolingPaddingCase{"GlobalAverageControl",
                                       "global_average",
                                       "valid",
                                       {2, 2, 3, 4},
                                       {3, 4},
                                       {1, 1},
                                       {0, 0, 0, 0}},
                    PoolingPaddingCase{"MaxSameControl",
                                       "max",
                                       "same",
                                       {1, 2, 2, 2},
                                       {2, 2},
                                       {1, 1},
                                       {0, 1, 0, 1}}),
  [](const ::testing::TestParamInfo<PoolingPaddingCase> &info) {
    return info.param.name;
  });
} // namespace

