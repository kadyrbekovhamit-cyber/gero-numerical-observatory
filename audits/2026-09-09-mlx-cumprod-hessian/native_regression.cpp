#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include "mlx/mlx.h"

namespace mx = mlx::core;
using A = mx::array;
using Fn = std::function<A(const A&)>;

A component(const A& x, int i) {
  return mx::sum(mx::slice(mx::reshape(x, {-1}), {i}, {i+1}));
}
Fn partial(Fn f, int i) {
  return [f,i](const A& x) { return component(mx::grad(f)(x),i); };
}
std::vector<double> values(const A& x) {
  mx::eval(x);
  std::vector<double> out(x.size());
  for (int i=0;i<x.size();++i) out[i]=x.data<float>()[i];
  return out;
}

int checks=0, failures=0, cases=0, failed_cases=0;
void check(std::string label, const A& actual, const std::vector<double>& expected, double tol=2e-5) {
  ++checks;
  auto got=values(actual);
  bool ok=got.size()==expected.size();
  for (int i=0;ok && i<got.size();++i)
    ok=std::isfinite(got[i]) && std::abs(got[i]-expected[i]) <= tol*(1+std::abs(expected[i]));
  if (!ok) {
    ++failures;
    std::cout << "FAIL " << label << " got=";
    for (double v:got) std::cout << v << ',';
    std::cout << " expected=";
    for (double v:expected) std::cout << v << ',';
    std::cout << '\n';
  }
}

void test(std::vector<float> data, mx::Shape shape, int axis, bool reverse, bool inclusive, std::string label) {
  ++cases;
  int before=failures;
  int N=data.size(), ax=axis<0 ? axis+shape.size() : axis, n=shape[ax], stride=1;
  for (int k=ax+1;k<shape.size();++k) stride*=shape[k];
  std::vector<float> weights(N);
  for (int i=0;i<N;++i) weights[i]=(i%3==0) ? 1.f : ((i%3==1) ? -0.5f : 2.f);
  A x(data.data(),shape,mx::float32), cot(weights.data(),shape,mx::float32);
  Fn f=[=](const A& v){return mx::sum(mx::multiply(mx::cumprod(v,axis,reverse,inclusive),cot));};
  std::vector<double> expected_y(N), expected_g(N), expected_h(N*N);
  for (int out=0;out<N;++out) {
    int pos=(out/stride)%n, line=out-pos*stride;
    std::vector<int> term;
    for (int k=0;k<n;++k) {
      bool include=reverse ? (inclusive ? k>=pos : k>pos) : (inclusive ? k<=pos : k<pos);
      if (include) term.push_back(line+k*stride);
    }
    auto product_except=[&](int skip1,int skip2) {
      double p=1;
      for (int id:term) if (id!=skip1 && id!=skip2) p*=data[id];
      return p;
    };
    expected_y[out]=product_except(-1,-1);
    for (int i:term) {
      expected_g[i]+=weights[out]*product_except(i,-1);
      for (int j:term) if (i!=j) expected_h[i*N+j]+=weights[out]*product_except(i,j);
    }
  }
  check(label+" forward",mx::cumprod(x,axis,reverse,inclusive),expected_y);
  check(label+" gradient",mx::grad(f)(x),expected_g);
  for (int i=0;i<N;++i) {
    std::vector<double> row(expected_h.begin()+i*N,expected_h.begin()+(i+1)*N);
    check(label+" Hessian_row_"+std::to_string(i),mx::grad(partial(f,i))(x),row);
  }
  failed_cases += failures!=before;
}

int main() {
  mx::set_default_device(mx::Device::cpu);
  std::cout << std::setprecision(10);
  std::vector<std::vector<float>> points{{1,2,3},{0,2,3},{1,0,3},{1,2,0},{0,0,3},{0,0,0}};
  for (bool reverse:{false,true}) for (bool inclusive:{false,true}) {
    for (int i=0;i<points.size();++i)
      test(points[i],{3},0,reverse,inclusive,"vector_"+std::to_string(i)+"_"+std::to_string(reverse)+std::to_string(inclusive));
    test({0,2,3,1,0,2},{2,3},1,reverse,inclusive,"matrix_axis1_"+std::to_string(reverse)+std::to_string(inclusive));
    test({0,1,2,0,3,2},{3,2},-2,reverse,inclusive,"matrix_axis-2_"+std::to_string(reverse)+std::to_string(inclusive));
    test({0},{1},0,reverse,inclusive,"singleton_"+std::to_string(reverse)+std::to_string(inclusive));
    test({1,.5f,-1,2,.25f},{5},0,reverse,inclusive,"length5_nonzero_"+std::to_string(reverse)+std::to_string(inclusive));
    test({1,.5f,0,2,-1},{5},0,reverse,inclusive,"length5_zero_"+std::to_string(reverse)+std::to_string(inclusive));
    test({1,0,2,1,0,.5f,1,-1},{8},0,reverse,inclusive,"length8_zeros_"+std::to_string(reverse)+std::to_string(inclusive));
    test({},{0},0,reverse,inclusive,"empty_"+std::to_string(reverse)+std::to_string(inclusive));
  }
  // A cubic polynomial has a constant mixed third derivative and zero fourth derivative.
  for (bool reverse:{false,true}) for (bool inclusive:{false,true}) {
    ++cases;
    int before=failures;
    Fn f=[=](const A& x){return mx::sum(mx::cumprod(x,0,reverse,inclusive));};
    A x({0.f,0.f,0.f});
    Fn third=partial(partial(partial(f,0),1),2);
    check("third_derivative",third(x),{inclusive ? 1.0 : 0.0});
    check("fourth_derivative",partial(third,0)(x),{0.0});
    failed_cases+=failures!=before;
  }
  // Finite differences of the library's first gradient independently recover the Hessian.
  ++cases;
  int before=failures;
  Fn f=[](const A& x){return mx::sum(mx::cumprod(x));};
  A x({0.f,2.f,3.f});
  std::vector<std::vector<double>> columns{{0,4,2},{4,0,0},{2,0,0}};
  for (int j=0;j<3;++j) {
    std::vector<float> delta(3,0); delta[j]=1.f/256;
    A d(delta.data(),{3},mx::float32);
    A fd=mx::divide(mx::subtract(mx::grad(f)(mx::add(x,d)),mx::grad(f)(mx::subtract(x,d))),A(2.f/256));
    check("finite_difference_column_"+std::to_string(j),fd,columns[j],1e-3);
  }
  failed_cases+=failures!=before;
  std::cout << "SUMMARY cases=" << cases << " passed_cases=" << cases-failed_cases
            << " failed_cases=" << failed_cases << " checks=" << checks << " failures=" << failures << '\n';
  return failures ? 1 : 0;
}
