#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "mlx/mlx.h"
namespace mx=mlx::core;
using A=mx::array;
using VF=std::function<std::vector<A>(const std::vector<A>&)>;
using SF=std::function<A(const A&)>;
using MF=std::function<A(const std::vector<A>&)>;
int checks=0,failures=0;
VF fun=[](const std::vector<A>& z){return std::vector<A>{mx::arctan2(z[0],z[1])};};
std::string dtype_name(mx::Dtype dtype) {
  if(dtype==mx::float64)return "float64";
  if(dtype==mx::float16)return "float16";
  if(dtype==mx::bfloat16)return "bfloat16";
  return "float32";
}
std::vector<double> values(const A& input) {
  auto a=mx::contiguous(mx::astype(input,mx::float64));
  mx::eval(a);
  return {a.data<double>(),a.data<double>()+a.size()};
}
A pack(const std::vector<A>& arrays) {
  std::vector<A> flat;
  for(const auto& a:arrays)flat.push_back(mx::reshape(a,{-1}));
  return mx::concatenate(flat);
}
void numbers(const std::vector<double>& v) {
  std::cout << '[';
  for(size_t i=0;i<v.size();++i) {
    if(i)std::cout << ',';
    if(std::isfinite(v[i]))std::cout << std::setprecision(17) << v[i];
    else std::cout << (std::isnan(v[i]) ? "\"NaN\"" : v[i]>0 ? "\"Infinity\"" : "\"-Infinity\"");
  }
  std::cout << ']';
}
void check(std::string label,const A& actual,const std::vector<double>& expected,mx::Dtype dtype,double tol=0) {
  ++checks;
  if(!tol)tol=dtype==mx::float64 ? 5e-13 :
      dtype==mx::float16 ? 3e-3 : dtype==mx::bfloat16 ? 2e-2 : 2e-5;
  auto got=values(actual);
  bool ok=got.size()==expected.size() && actual.dtype()==dtype;
  for(size_t i=0;i<got.size() && i<expected.size();++i)
    ok &= std::isfinite(got[i]) && std::isfinite(expected[i]) &&
      (expected[i]==0 ? std::abs(got[i])<=tol : std::abs(got[i]/expected[i]-1)<=tol);
  failures+=!ok;
  std::cout << "CASE {\"label\":\"" << label << "\",\"dtype\":\""
            << dtype_name(dtype) << "\",\"actual\":";
  numbers(got);
  std::cout << ",\"expected\":";numbers(expected);
  std::cout << ",\"tolerance\":" << tol << ",\"passed\":" << (ok ? "true" : "false") << "}\n";
}
std::pair<double,double> reference(double y,double x) {
  double radius=std::hypot(y,x);
  return {(x/radius)/radius,-(y/radius)/radius};
}
std::string tag(int power,double a,double b) {
  return "p="+std::to_string(power)+"/a="+std::to_string(a)+"/b="+std::to_string(b);
}
void first(int power,double a,double b,mx::Dtype dtype) {
  double c=std::ldexp(1.,power);
  A y(c*a,dtype),x(c*b,dtype);
  auto [gy,gx]=reference(values(y)[0],values(x)[0]);
  std::string label=tag(power,a,b);
  for(double weight:{0.,1.,-.5}) {
    A w(weight,dtype);
    std::string name=label+"/w="+std::to_string(weight);
    check(name+"/VJP",pack(mx::vjp(fun,{y,x},{w}).second),{weight*gy,weight*gx},dtype);
    SF fy=[=](const A& z){return mx::arctan2(z,x);};
    SF fx=[=](const A& z){return mx::arctan2(y,z);};
    check(name+"/JVP-y",mx::jvp(fy,y,w).second,{weight*gy},dtype);
    check(name+"/JVP-x",mx::jvp(fx,x,w).second,{weight*gx},dtype);
    check(name+"/JVP-both",mx::jvp(fun,{y,x},{w,mx::negative(w)}).second[0],
          {weight*(gy-gx)},dtype);
  }
  check(label+"/radial-JVP",mx::jvp(fun,{y,x},{y,x}).second[0],{0.},dtype);
  check(label+"/angular-JVP",mx::jvp(fun,{y,x},{x,mx::negative(y)}).second[0],{1.},dtype);
}
void higher(int power,double tv,mx::Dtype dtype) {
  A c(std::ldexp(1.,power),dtype),t(tv,dtype);
  SF f=[=](const A& z){return mx::arctan2(mx::multiply(c,z),c);};
  double d=1+tv*tv;
  std::string label="composite/p="+std::to_string(power)+"/t="+std::to_string(tv);
  check(label+"/1",mx::grad(f)(t),{1/d},dtype);
  check(label+"/2",mx::grad(mx::grad(f))(t),{-2*tv/(d*d)},dtype);
  check(label+"/3",mx::grad(mx::grad(mx::grad(f)))(t),{(6*tv*tv-2)/(d*d*d)},dtype);
  if(tv==1) {
    double h=std::ldexp(1.,dtype==mx::float64 ? -10 : -6);
    auto fd=mx::divide(mx::subtract(f(A(tv+h,dtype)),f(A(tv-h,dtype))),A(2*h,dtype));
    check(label+"/finite-difference",fd,{1/d},dtype,dtype==mx::float64 ? 2e-6 : 5e-4);
  }
}
void mixed(int power,mx::Dtype dtype) {
  A c(std::ldexp(1.,power),dtype);
  MF f=[=](const std::vector<A>& z){return mx::arctan2(mx::multiply(c,z[0]),mx::multiply(c,z[1]));};
  auto gradient=mx::grad(f,std::vector<int>{0,1});
  std::vector<A> h;
  for(int i=0;i<2;++i) {
    MF part=[=](const std::vector<A>& z){return gradient(z)[i];};
    auto row=mx::grad(part,std::vector<int>{0,1})({A(1.,dtype),A(2.,dtype)});
    h.insert(h.end(),row.begin(),row.end());
  }
  check("mixed-Hessian/p="+std::to_string(power),pack(h),{-.16,-.12,-.12,.16},dtype);
}
void broadcast(int power,mx::Dtype dtype) {
  double c=std::ldexp(1.,power);
  double ys[]={c,2*c},xs[]={c,-c,2*c},ws[]={0,1,-.5,2,-1,1};
  A y=mx::astype(A(ys,{2,1},mx::float64),dtype);
  A x=mx::astype(A(xs,{1,3},mx::float64),dtype);
  A w=mx::astype(A(ws,{2,3},mx::float64),dtype);
  std::vector<double> v(5,0.),j;
  for(int i=0;i<2;++i)for(int k=0;k<3;++k) {
    auto [gy,gx]=reference(ys[i],xs[k]);
    v[i]+=ws[3*i+k]*gy;v[2+k]+=ws[3*i+k]*gx;
    j.push_back(c*gy);
  }
  std::string label="broadcast/p="+std::to_string(power);
  check(label+"/VJP",pack(mx::vjp(fun,{y,x},{w}).second),v,dtype);
  check(label+"/JVP",mx::jvp(fun,{y,x},{mx::full_like(y,c),mx::zeros_like(x)}).second[0],j,dtype);
}
void matrix(int power,mx::Dtype dtype,bool strided) {
  double c=std::ldexp(1.,power);
  double ys[]={c,2*c,0,-c,-2*c,c},xs[]={2*c,c,c,-2*c,-c,0},ws[]={0,1,-.5,2,-1,1};
  A y=mx::astype(A(ys,{2,3},mx::float64),dtype);
  A x=mx::astype(A(xs,{2,3},mx::float64),dtype);
  A w=mx::astype(A(ws,{2,3},mx::float64),dtype);
  if(strided)y=mx::swapaxes(mx::contiguous(mx::swapaxes(y,0,1)),0,1);
  std::vector<double> gyv,gxv,j;
  for(int i=0;i<6;++i) {
    auto [gy,gx]=reference(ys[i],xs[i]);
    gyv.push_back(ws[i]*gy);gxv.push_back(ws[i]*gx);
    j.push_back(c*(gy-gx));
  }
  gyv.insert(gyv.end(),gxv.begin(),gxv.end());
  std::string label=(strided ? "strided/p=" : "matrix/p=")+std::to_string(power);
  check(label+"/VJP",pack(mx::vjp(fun,{y,x},{w}).second),gyv,dtype);
  check(label+"/JVP",mx::jvp(fun,{y,x},{mx::full_like(y,c),mx::full_like(x,-c)}).second[0],j,dtype);
}
int main() {
  mx::set_default_device(mx::Device::cpu);
  for(auto dtype:{mx::float16,mx::bfloat16,mx::float32,mx::float64}) {
    int low=dtype==mx::float16 ? -14 : dtype==mx::float64 ? -600 : -80;
    int high=dtype==mx::float16 ? 10 : dtype==mx::float64 ? 600 : 80;
    for(int power:{0,low,high}) {
      for(auto [a,b]:std::vector<std::pair<double,double>>{
            {1,1},{2,1},{-1,2},{1,-2},{-1,-2},{0,1},{1,0}})
        first(power,a,b,dtype);
      if(dtype!=mx::float32 && dtype!=mx::float64)continue;
      for(double t:{-2.,-1.,-.5,0.,.5,1.,2.})higher(power,t,dtype);
      mixed(power,dtype);broadcast(power,dtype);
      matrix(power,dtype,false);matrix(power,dtype,true);
    }
  }
  std::cout << "SUMMARY checks=" << checks << " failures=" << failures << '\n';
  return failures ? 1 : 0;
}
