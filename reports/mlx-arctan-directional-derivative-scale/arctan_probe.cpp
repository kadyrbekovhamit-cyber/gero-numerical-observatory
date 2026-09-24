#include "mlx/mlx.h"

#include <bit>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace mx = mlx::core;

static uint32_t bits(float value) {
  return std::bit_cast<uint32_t>(value);
}

int main(int argc, char** argv) {
  if (argc != 3) {
    return 2;
  }
  mx::set_default_device(mx::Device(mx::Device::cpu));
  std::ifstream input(argv[1]);
  std::vector<int> ids;
  std::vector<float> xs;
  std::vector<float> vs;
  int id;
  uint32_t xb;
  uint32_t vb;
  while (input >> id >> xb >> vb) {
    ids.push_back(id);
    xs.push_back(std::bit_cast<float>(xb));
    vs.push_back(std::bit_cast<float>(vb));
  }
  int n = static_cast<int>(xs.size());
  mx::Shape shape{n};
  std::string layout(argv[2]);
  if (layout == "row") {
    shape = {1, n};
  } else if (layout == "column") {
    shape = {n, 1};
  }
  mx::array x(xs.data(), shape, mx::float32);
  mx::array v(vs.data(), shape, mx::float32);
  auto f = [](const mx::array& a) { return mx::arctan(a); };
  auto j = mx::jvp(f, x, v);
  auto w = mx::vjp(f, x, v);
  mx::eval(j.first, j.second, w.second);
  std::cout << "id,x_bits,v_bits,forward_bits,jvp_bits,vjp_bits\n";
  for (int i = 0; i < n; ++i) {
    std::cout << ids[i] << ',' << bits(xs[i]) << ',' << bits(vs[i]) << ','
              << bits(j.first.data<float>()[i]) << ','
              << bits(j.second.data<float>()[i]) << ','
              << bits(w.second.data<float>()[i]) << '\n';
  }
}
