#include "mlx/mlx.h"
#include <bit>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
namespace mx = mlx::core;
uint32_t bits(float v) { return std::bit_cast<uint32_t>(v); }
int main(int argc, char** argv) {
  if (argc != 3) return 2;
  mx::set_default_device(mx::Device(mx::Device::cpu));
  std::ifstream input(argv[1]);
  std::vector<float> xs, vs;
  std::vector<int> ids;
  int id; uint32_t xb, vb;
  while (input >> id >> xb >> vb) {
    ids.push_back(id); xs.push_back(std::bit_cast<float>(xb));
    vs.push_back(std::bit_cast<float>(vb));
  }
  int n = xs.size();
  mx::Shape shape{n};
  std::string layout(argv[2]);
  if (layout == "row") shape = {1, n};
  if (layout == "column") shape = {n, 1};
  mx::array x(xs.data(), shape, mx::float32);
  mx::array v(vs.data(), shape, mx::float32);
  auto f = [](const mx::array& a) { return mx::rsqrt(a); };
  auto j = mx::jvp(f, x, v);
  auto w = mx::vjp(f, x, v);
  mx::eval(j.first, j.second, w.second);
  std::cout << "id,x_bits,v_bits,forward_bits,jvp_bits,vjp_bits\n";
  for (int i=0; i<n; ++i) {
    std::cout << ids[i] << ',' << bits(xs[i]) << ',' << bits(vs[i])
              << ',' << bits(j.first.data<float>()[i])
              << ',' << bits(j.second.data<float>()[i])
              << ',' << bits(w.second.data<float>()[i]) << '\n';
  }
}
