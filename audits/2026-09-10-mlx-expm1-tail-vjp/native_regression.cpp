#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "mlx/mlx.h"
namespace mx=mlx::core;
using A=mx::array;
int checks=0,failures=0;
std::vector<double> values(const A& input) {
  auto x=mx::contiguous(mx::astype(input,mx::float64));
  mx::eval(x);
  return {x.data<double>(),x.data<double>()+x.size()};
}
void numbers(const std::vector<double>& v) {
  std::cout << '[';
  for (int i=0;i<v.size();++i) {
    if(i)std::cout << ',';
    if(std::isfinite(v[i]))std::cout << std::setprecision(17) << v[i];
    else std::cout << (std::isnan(v[i]) ? "\"NaN\"" : v[i]>0 ? "\"Infinity\"" : "\"-Infinity\"");
  }
  std::cout << ']';
}
void check(const std::string& name,const A& actual,const std::vector<double>& expected,mx::Dtype dtype) {
  ++checks;
  auto got=values(actual);
  bool ok=got.size()==expected.size() && actual.dtype()==dtype;
  double tol=dtype==mx::float64 ? 2e-14 : 4e-6;
  for(int i=0;i<got.size() && i<expected.size();++i)
    ok &= std::isfinite(got[i]) && (expected[i]==0 ? got[i]==0 :
          std::abs(got[i]/expected[i]-1)<=tol);
  failures+=!ok;
  std::cout << "CASE {\"label\":\"" << name << "\",\"dtype\":\""
            << (dtype==mx::float64 ? "float64" : "float32") << "\",\"actual\":";
  numbers(got);
  std::cout << ",\"expected\":";numbers(expected);
  std::cout << ",\"passed\":" << (ok ? "true" : "false") << "}\n";
}
std::function<A(const A&)> fun=[](const A& x){return mx::expm1(x);};
void scalar(double value,double weight,mx::Dtype dtype) {
  A x=mx::astype(A(value),dtype),w=mx::astype(A(weight),dtype);
  double q=values(x)[0],c=values(w)[0],expected=c*std::exp(q);
  std::string tag="x="+std::to_string(value)+"/w="+std::to_string(weight);
  check(tag+"/VJP",mx::vjp(fun,x,w).second,{expected},dtype);
  check(tag+"/JVP",mx::jvp(fun,x,w).second,{expected},dtype);
}
void higher(double value,mx::Dtype dtype) {
  A x=mx::astype(A(value),dtype);
  double expected=std::exp(values(x)[0]);
  auto g=mx::grad(fun),g2=mx::grad(g),g3=mx::grad(g2);
  std::string tag="higher/x="+std::to_string(value);
  check(tag+"/2",g2(x),{expected},dtype);
  check(tag+"/3",g3(x),{expected},dtype);
}
void matrix(mx::Dtype dtype,bool strided) {
  std::vector<double> vx{-40,-20,-10,0,.25,1},vw{0,-.5,1,1e8,-1,2},expected;
  A x=mx::astype(A(vx.data(),{2,3},mx::float64),dtype);
  A w=mx::astype(A(vw.data(),{2,3},mx::float64),dtype);
  if(strided) x=mx::swapaxes(mx::contiguous(mx::swapaxes(x,0,1)),0,1);
  auto q=values(x),c=values(w);
  for(int i=0;i<q.size();++i)expected.push_back(c[i]*std::exp(q[i]));
  std::string tag=strided ? "strided" : "matrix";
  check(tag+"/VJP",mx::vjp(fun,x,w).second,expected,dtype);
  check(tag+"/JVP",mx::jvp(fun,x,w).second,expected,dtype);
}
int main() {
  mx::set_default_device(mx::Device::cpu);
  for(auto dtype:{mx::float32,mx::float64}) {
    for(double x:{-80.,-40.,-20.,-18.,-10.,-1.,0.,.25,1.,20.})
      for(double w:{0.,1.,-.5,2.})scalar(x,w,dtype);
    for(double x:{-40.,-20.,-10.,0.,1.})higher(x,dtype);
    matrix(dtype,false);matrix(dtype,true);
  }
  for(double x:{-700.,-100.,100.,700.})
    for(double w:{0.,1.,-.5})scalar(x,w,mx::float64);
  scalar(-20.,1e8,mx::float32);
  // A scalar broadcast tests accumulated gradients through the composition.
  for(auto dtype:{mx::float32,mx::float64}) {
    auto f=std::function<A(const A&)>([=](const A& x) {
      return mx::sum(mx::expm1(mx::broadcast_to(x,{2,3})));
    });
    A x=mx::astype(A(-20.),dtype);
    check("broadcast",mx::grad(f)(x),{6*std::exp(-20.)},dtype);
  }
  std::cout << "SUMMARY checks=" << checks << " failures=" << failures << '\n';
  return failures ? 1 : 0;
}
