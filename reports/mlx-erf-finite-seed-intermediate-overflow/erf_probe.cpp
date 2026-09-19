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
  std::vector<float> xs, vs; std::vector<int> ids;
  int id; uint32_t xb, vb;
  while (input >> id >> xb >> vb) {
    ids.push_back(id); xs.push_back(std::bit_cast<float>(xb));
    vs.push_back(std::bit_cast<float>(vb));
  }
  int n = xs.size(); mx::Shape shape{n}; std::string layout(argv[2]);
  if (layout == "row") shape = {1, n};
  if (layout == "column") shape = {n, 1};
  auto f = [](const mx::array& a) { return mx::erf(a); };
  mx::array x(xs.data(), shape, mx::float32), v(vs.data(), shape, mx::float32);
  auto j = mx::jvp(f, x, v); auto w = mx::vjp(f, x, v);
  auto u = mx::vjp(f, x, mx::ones_like(v)); auto scaled = mx::multiply(v,u.second);
  auto loss = [&v](const mx::array& a) { return mx::sum(mx::multiply(v, mx::erf(a))); };
  auto g = mx::grad(loss)(x);
  auto h = mx::grad([&loss](const mx::array& a) { return mx::sum(mx::grad(loss)(a)); })(x);
  auto step = mx::subtract(x, mx::multiply(mx::array(0x1p-126f),g));
  auto lv=mx::multiply(v,j.first);
  auto base_step=mx::subtract(x,mx::multiply(mx::array(0.1f),u.second));
  auto normalized_step=mx::subtract(x,mx::multiply(mx::array(0.1f),mx::divide(g,v)));
  mx::eval(j.first,j.second,w.second,scaled,g,h,step,lv,base_step,normalized_step);
  std::cout << "id,x_bits,v_bits,forward_bits,jvp_bits,vjp_bits,scaled_unit_vjp_bits,loss_bits,grad_bits,second_bits,step_bits,base_step_bits,normalized_step_bits\n";
  for (int i=0; i<n; ++i) {
    std::cout << ids[i] << ',' << bits(xs[i]) << ',' << bits(vs[i]);
    for(auto a : {j.first,j.second,w.second,scaled,lv,g,h,step,base_step,normalized_step})
      std::cout << ',' << bits(a.data<float>()[i]);
    std::cout << '\n';
  }
}
