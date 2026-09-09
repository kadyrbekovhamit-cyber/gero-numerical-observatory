// Independent dense interpolation-matrix checks against the native layer.
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <layer_context.h>
#include <upsample2d_layer.h>
#include <var_grad.h>

static long double axisWeight(int out, int in, int length, int scale,
                              bool nearest) {
  if (nearest)
    return in == out / scale ? 1.0L : 0.0L;
  const long double coordinate = std::clamp(
    (out + 0.5L) / scale - 0.5L, 0.0L, static_cast<long double>(length - 1));
  return std::max(0.0L, 1.0L - std::abs(coordinate - in));
}

int main() {
  using namespace nntrainer;
  const std::vector<std::pair<int, int>> shapes = {{1, 1}, {1, 3}, {3, 1},
                                                   {2, 3}, {3, 2}, {5, 7}};
  const std::vector<std::pair<int, int>> scales = {{1, 1}, {2, 3}, {3, 2},
                                                   {1, 4}, {4, 1}, {7, 5}};
  int cases = 0;
  long double maxForward = 0, maxBackward = 0, maxMass = 0;
  std::cout << std::setprecision(17);
  for (const bool nearest : {true, false}) {
    for (const auto [height, width] : shapes) {
      for (const auto [sh, sw] : scales) {
        const std::string mode = nearest ? "nearest" : "bilinear";
        const int oh = height * sh, ow = width * sw;
        const int ni = height * width, no = oh * ow, groups = 4;
        std::vector<long double> matrix(no * ni, 0);
        for (int u = 0; u < no; ++u)
          for (int t = 0; t < ni; ++t)
            matrix[u * ni + t] =
              axisWeight(u / ow, t / width, height, sh, nearest) *
              axisWeight(u % ow, t % width, width, sw, nearest);
        Upsample2dLayer layer;
        layer.setProperty(
          {"upsample=" + mode,
           "kernel_size=" + std::to_string(sh) + "," + std::to_string(sw)});
        TensorDim dim(
          {2, 2, static_cast<unsigned>(height), static_cast<unsigned>(width)});
        InitLayerContext init({dim}, {true}, false, "upsample_matrix_review");
        layer.finalize(init);
        const auto odim = init.getOutSpecs()[0].variable_spec.dim;
        Var_Grad input(dim, Initializer::NONE, true, true, "input");
        Var_Grad output(odim, Initializer::NONE, true, true, "output");
        RunLayerContext context("upsample_matrix_review", true, 0.0f, false,
                                1.0f, nullptr, false, {}, {&input}, {&output},
                                {});
        auto *xp = input.getVariableRef().getData<float>();
        auto *yp = output.getVariableRef().getData<float>();
        auto *dxp = input.getGradientRef().getData<float>();
        auto *dyp = output.getGradientRef().getData<float>();
        std::vector<float> x(groups * ni), dy(groups * no);
        for (size_t i = 0; i < x.size(); ++i)
          x[i] = (int((i * 7) % 23) - 11) / 8.0f;
        for (size_t i = 0; i < dy.size(); ++i)
          dy[i] = (int((i * 11) % 29) - 14) / 16.0f;
        std::copy(x.begin(), x.end(), xp);
        std::copy(dy.begin(), dy.end(), dyp);
        std::fill_n(dxp, x.size(), 17.0f);
        layer.forwarding(context, true);
        layer.calcDerivative(context);
        bool ok = std::equal(x.begin(), x.end(), xp) &&
                  std::equal(dy.begin(), dy.end(), dyp);
        long double fw = 0, bw = 0, mass = 0;
        for (int g = 0; g < groups; ++g) {
          for (int u = 0; u < no; ++u) {
            long double ref = 0, weightSum = 0;
            for (int t = 0; t < ni; ++t) {
              ref += matrix[u * ni + t] * x[g * ni + t];
              weightSum += matrix[u * ni + t];
            }
            const long double err = std::abs(ref - yp[g * no + u]);
            fw = std::max(fw, err);
            ok &= std::isfinite(yp[g * no + u]) &&
                  err <= 5e-5L + 3e-6L * std::abs(ref) &&
                  std::abs(weightSum - 1) <= 1e-15L;
          }
          for (int t = 0; t < ni; ++t) {
            long double ref = 0;
            for (int u = 0; u < no; ++u)
              ref += matrix[u * ni + t] * dy[g * no + u];
            const long double err = std::abs(ref - dxp[g * ni + t]);
            bw = std::max(bw, err);
            ok &= std::isfinite(dxp[g * ni + t]) &&
                  err <= 5e-5L + 3e-6L * std::abs(ref);
          }
        }
        const double sumDx = std::accumulate(dxp, dxp + x.size(), 0.0);
        const double sumDy = std::accumulate(dy.begin(), dy.end(), 0.0);
        mass = std::abs(sumDx - sumDy);
        ok &= mass <= 3e-4L + 3e-6L * std::abs(sumDy);
        std::fill_n(xp, x.size(), 7.25f);
        layer.forwarding(context, true);
        for (size_t i = 0; i < dy.size(); ++i)
          ok &= std::abs(yp[i] - 7.25f) <= 1e-5f;
        std::fill_n(dyp, dy.size(), 0.0f);
        std::fill_n(dxp, x.size(), 17.0f);
        layer.calcDerivative(context);
        for (size_t i = 0; i < x.size(); ++i)
          ok &= dxp[i] == 0;
        maxForward = std::max(maxForward, fw);
        maxBackward = std::max(maxBackward, bw);
        maxMass = std::max(maxMass, mass);
        std::cout << "{\"mode\":\"" << mode << "\",\"shape\":[2,2," << height
                  << "," << width << "],\"scale\":[" << sh << "," << sw
                  << "],\"max_forward_error\":" << fw
                  << ",\"max_backward_error\":" << bw
                  << ",\"gradient_mass_error\":" << mass
                  << ",\"passed\":" << (ok ? "true" : "false") << "}\n";
        if (!ok)
          return 1;
        ++cases;
      }
    }
  }
  std::cout << "{\"summary\":true,\"passed_cases\":" << cases
            << ",\"max_forward_error\":" << maxForward
            << ",\"max_backward_error\":" << maxBackward
            << ",\"gradient_mass_error\":" << maxMass << "}\n";
  return 0;
}
