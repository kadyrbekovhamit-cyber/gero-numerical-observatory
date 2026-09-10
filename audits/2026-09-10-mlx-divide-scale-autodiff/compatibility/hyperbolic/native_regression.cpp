#include <cmath>
#include <complex>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include "mlx/mlx.h"
namespace mx=mlx::core;
using A=mx::array;
int checks=0,failures=0;
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
void numbers(const std::vector<double>& v) {
  std::cout << '[';
  for(int i=0;i<v.size();++i) {
    if(i)std::cout << ',';
    if(std::isfinite(v[i]))std::cout << std::setprecision(17) << v[i];
    else std::cout << (std::isnan(v[i]) ? "\"NaN\"" : v[i]>0 ? "\"Infinity\"" : "\"-Infinity\"");
  }
  std::cout << ']';
}
void check(std::string name,const A& actual,const std::vector<double>& expected,mx::Dtype dtype,double tol=0) {
  ++checks;
  if(!tol)tol=dtype==mx::float64 ? 3e-13 :
              dtype==mx::float16 ? 3e-3 : dtype==mx::bfloat16 ? 2e-2 : 8e-6;
  auto got=values(actual);
  bool ok=got.size()==expected.size() && actual.dtype()==dtype;
  for(int i=0;i<got.size() && i<expected.size();++i)
    ok &= std::isfinite(got[i]) && (expected[i]==0 ? std::abs(got[i])<=tol :
                                  std::abs(got[i]/expected[i]-1)<=tol);
  failures+=!ok;
  std::cout << "CASE {\"label\":\"" << name << "\",\"dtype\":\""
            << dtype_name(dtype) << "\",\"actual\":";
  numbers(got);
  std::cout << ",\"expected\":";numbers(expected);
  std::cout << ",\"tolerance\":" << tol << ",\"passed\":" << (ok ? "true" : "false") << "}\n";
}
double slope(double x,bool acosh) {
  return acosh ? (1/std::sqrt(x-1))/std::sqrt(x+1) : 1/std::hypot(x,1.);
}
std::function<A(const A&)> operation(bool acosh) {
  return [=](const A& x){return acosh ? mx::arccosh(x) : mx::arcsinh(x);};
}
void first(double input,double weight,bool acosh,mx::Dtype dtype) {
  auto f=operation(acosh);
  A x(input,dtype),w(weight,dtype);
  double q=values(x)[0],c=values(w)[0],g=slope(q,acosh);
  std::string label=(acosh ? "acosh/" : "asinh/")+std::to_string(input)+"/g="+std::to_string(weight);
  check(label+"/JVP",mx::jvp(f,x,w).second,{c*g},dtype);
  check(label+"/VJP",mx::vjp(f,x,w).second,{c*g},dtype);
}
void high_order(double input,bool acosh,mx::Dtype dtype) {
  auto f=operation(acosh);
  A x(input,dtype);
  double q=values(x)[0],d=acosh ? (q-1)*(q+1) : q*q+1,g=slope(q,acosh);
  double second=-q*g/d,third=(2*q*q+(acosh ? 1 : -1))*g/(d*d);
  std::string label=(acosh ? "acosh-higher/" : "asinh-higher/")+std::to_string(input);
  check(label+"/2",mx::grad(mx::grad(f))(x),{second},dtype);
  check(label+"/3",mx::grad(mx::grad(mx::grad(f)))(x),{third},dtype);
}
void composite(bool acosh,mx::Dtype dtype,double tvalue) {
  A c(std::ldexp(1.,dtype==mx::float64 ? 600 : 80),dtype);
  A t(tvalue,dtype);
  auto op=operation(acosh);
  auto f=std::function<A(const A&)>([=](const A& z){return op(mx::multiply(c,z));});
  double tv=values(t)[0],cv=values(c)[0],q=(1/cv)/tv;
  double d=1+(acosh ? -1 : 1)*q*q;
  double g1=1/(std::abs(tv)*std::sqrt(d));
  double g2=-std::copysign(1.,tv)/(tv*tv*std::pow(d,1.5));
  double g3=(2+(acosh ? 1 : -1)*q*q)/(std::pow(std::abs(tv),3)*std::pow(d,2.5));
  std::string label=(acosh ? "acosh-composite/" : "asinh-composite/")+std::to_string(tv);
  check(label+"/1",mx::grad(f)(t),{g1},dtype);
  check(label+"/2",mx::grad(mx::grad(f))(t),{g2},dtype);
  check(label+"/3",mx::grad(mx::grad(mx::grad(f)))(t),{g3},dtype);
  if(tv==1) {
    double step=std::ldexp(1.,dtype==mx::float64 ? -10 : -6);
    A plus(tv+step,dtype),minus(tv-step,dtype);
    auto fd=mx::divide(mx::subtract(f(plus),f(minus)),A(2*step,dtype));
    check(label+"/finite-difference",fd,{g1},dtype,dtype==mx::float64 ? 2e-6 : 5e-4);
  }
}
void matrix(bool acosh,mx::Dtype dtype,bool strided) {
  double big=dtype==mx::float64 ? 1e200 : 1e20;
  std::vector<double> xval=acosh ? std::vector<double>{1.0001,2,3,big,2,.25+1} :
                                 std::vector<double>{0,-1,1,big,-big,.25};
  std::vector<double> wval{0,1,-.5,2,-1,1},expected;
  A x=mx::astype(A(xval.data(),{2,3},mx::float64),dtype);
  A w=mx::astype(A(wval.data(),{2,3},mx::float64),dtype);
  if(strided)x=mx::swapaxes(mx::contiguous(mx::swapaxes(x,0,1)),0,1);
  auto q=values(x),c=values(w);
  for(int i=0;i<q.size();++i)expected.push_back(c[i]*slope(q[i],acosh));
  std::string label=(acosh ? "acosh-matrix" : "asinh-matrix");
  if(strided)label+="-strided";
  auto f=operation(acosh);
  check(label+"/JVP",mx::jvp(f,x,w).second,expected,dtype);
  check(label+"/VJP",mx::vjp(f,x,w).second,expected,dtype);
}
int main() {
  mx::set_default_device(mx::Device::cpu);
  for(auto dtype:{mx::float32,mx::float64}) {
    double big=dtype==mx::float64 ? 1e200 : 1e20;
    double bigger=dtype==mx::float64 ? 1e300 : 1e30;
    double nearest=dtype==mx::float64 ? std::nextafter(1.,2.) : std::nextafter(1.f,2.f);
    for(double x:{0.,-1e-6,1e-6,-1.,1.,2.,10.,big,-big,bigger,-bigger})
      for(double g:{0.,1.,-.5,2.})first(x,g,false,dtype);
    for(double x:{nearest,1.0001,1.25,2.,10.,big,bigger})
      for(double g:{0.,1.,-.5,2.})first(x,g,true,dtype);
    for(double x:{-2.,-1.,0.,.25,1.,2.})high_order(x,false,dtype);
    for(double x:{1.0001,1.25,2.,10.})high_order(x,true,dtype);
    for(bool acosh:{false,true}) {
      for(double t:{.5,1.,2.})composite(acosh,dtype,t);
      if(!acosh)composite(acosh,dtype,-1.);
      matrix(acosh,dtype,false);matrix(acosh,dtype,true);
    }
  }
  for(auto dtype:{mx::float16,mx::bfloat16}) {
    double big=dtype==mx::float16 ? 1000. : 1e20;
    double bigger=dtype==mx::float16 ? 2048. : 2e20;
    for(double x:{0.,1.,big,-big})
      for(double g:{0.,1.,-.5,2.})first(x,g,false,dtype);
    for(double x:{1.125,2.,big,bigger})
      for(double g:{0.,1.,-.5,2.})first(x,g,true,dtype);
  }
  std::cout << "SUMMARY checks=" << checks << " failures=" << failures << '\n';
  return failures ? 1 : 0;
}
