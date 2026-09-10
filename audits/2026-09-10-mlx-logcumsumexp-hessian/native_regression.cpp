#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>
#include "mlx/mlx.h"
namespace mx = mlx::core;
using A=mx::array;
int checks=0, failures=0, scenarios=0, failed_scenarios=0;

std::vector<double> values(const A& a) {
  A x=mx::contiguous(mx::astype(a,mx::float32));
  mx::eval(x);
  return {x.data<float>(),x.data<float>()+x.size()};
}
void check(const std::string& label,const A& actual,
           const std::vector<double>& expected,const mx::Shape& shape) {
  ++checks;
  auto v=values(actual);
  bool ok=actual.shape()==shape && v.size()==expected.size();
  double error=0;
  for(size_t i=0;i<std::min(v.size(),expected.size());++i) {
    error=std::max(error,std::abs(v[i]-expected[i]));
    ok &= std::isfinite(v[i]) && std::abs(v[i]-expected[i]) <= 2e-4*(1+std::abs(expected[i]));
  }
  if(!ok) { ++failures; std::cout<<"FAIL "<<label<<" max_error="<<error<<'\n'; }
}
void scenario(const std::string& label,const std::function<void()>& f) {
  ++scenarios; int before=failures;
  try { f(); } catch(const std::exception& e) {
    ++failures; std::cout<<"FAIL "<<label<<" exception="<<e.what()<<'\n';
  }
  if(failures>before) ++failed_scenarios;
}
A from(const std::vector<float>& v,const mx::Shape& shape) {
  return A(v.data(),shape,mx::float32);
}
A basis(int n,int i,const mx::Shape& shape) {
  std::vector<float> v(n); v[i]=1; return from(v,shape);
}

void test(mx::Shape shape,int axis,bool reverse,bool inclusive,int mode,bool strided=false) {
  int n=1; for(int d:shape)n*=d;
  int length=shape[axis], stride=1;
  for(int d=axis+1;d<shape.size();++d)stride*=shape[d];
  std::vector<float> input(n);
  for(int i=0;i<n;++i) input[i]=mode==3 ? (i%2 ? 20.f : -20.f) : (i%5-2)*.25f;
  A x=from(input,shape);
  if(strided) x=mx::swapaxes(mx::contiguous(mx::swapaxes(x,-1,-2)),-1,-2);
  auto scan=[&](const A& a){return mx::logcumsumexp(a,axis,reverse,inclusive);};
  auto safe=[&](const A& a){auto y=scan(a);return mx::where(mx::isfinite(y),y,A(0.f));};
  A y=safe(x);
  auto yv=values(y);
  std::vector<float> tv(n);
  for(int i=0;i<n;++i) {
    int row=(i/stride)%length;
    bool empty=!inclusive && row==(reverse ? length-1 : 0);
    float r=empty || mode==0 ? 0.f : (mode==1 ? (i%3==0 ? 0.f : (i%2 ? -.5f : .25f)) : .5f);
    tv[i]=static_cast<float>(yv[i])-r;
  }
  A target=from(tv,shape);
  std::vector<double> J(n*n), grad(n), hess(n*n), residual(n);
  for(int i=0;i<n;++i) {
    int pos=(i/stride)%length;
    int base=i-pos*stride;
    std::vector<int> ids;
    for(int k=0;k<length;++k) {
      bool included=reverse ? (inclusive ? k>=pos : k>pos) : (inclusive ? k<=pos : k<pos);
      if(included)ids.push_back(base+k*stride);
    }
    if(ids.empty())continue;
    double max=-INFINITY;
    for(int k:ids)max=std::max(max,double(input[k]));
    double den=0; for(int k:ids)den+=std::exp(input[k]-max);
    for(int k:ids)J[i*n+k]=std::exp(input[k]-max)/den;
    residual[i]=yv[i]-tv[i];
    for(int k:ids) {
      grad[k]+=residual[i]*J[i*n+k];
      for(int l:ids)hess[k*n+l]+=J[i*n+k]*J[i*n+l]+residual[i]*
          ((k==l ? J[i*n+k] : 0)-J[i*n+k]*J[i*n+l]);
    }
  }
  std::function<A(const A&)> loss=[&](const A& a) {
    return mx::multiply(A(.5f),mx::sum(mx::square(mx::subtract(safe(a),target))));
  };
  auto g=mx::grad(loss);
  std::string name="n="+std::to_string(n)+"/axis="+std::to_string(axis)+
      "/reverse="+std::to_string(reverse)+"/inclusive="+std::to_string(inclusive)+
      "/mode="+std::to_string(mode)+"/strided="+std::to_string(strided);
  scenario(name+"/gradient",[&]{check(name+"/gradient",g(x),grad,shape);});
  scenario(name+"/hessian",[&]{
    for(int k=0;k<n;++k) {
      A unit=basis(n,k,shape);
      auto row=mx::grad(std::function<A(const A&)>([&](const A& a) {
        return mx::sum(mx::multiply(g(a),unit));
      }))(x);
      std::vector<double> expected(hess.begin()+k*n,hess.begin()+(k+1)*n);
      check(name+"/hessian/row="+std::to_string(k),row,expected,shape);
    }
  });
  if(mode==0)scenario(name+"/cotangent_linearity",[&]{
    for(int k=0;k<n;++k) {
      A unit=basis(n,k,shape);
      auto gg=mx::grad(std::function<A(const A&)>([&](const A& cot) {
        auto v=mx::vjp([&](const std::vector<A>& p){return std::vector<A>{scan(p[0])};},
                      {x},{cot}).second[0];
        return mx::sum(mx::multiply(v,unit));
      }))(mx::zeros_like(x));
      std::vector<double> expected(n);
      for(int i=0;i<n;++i)expected[i]=J[i*n+k];
      check(name+"/cotangent/row="+std::to_string(k),gg,expected,shape);
    }
  });
}

void simple() {
  scenario("singleton",[&] {
    std::function<A(const A&)> f=[](const A& a) {
      return mx::multiply(A(.5f),mx::sum(mx::square(mx::logcumsumexp(mx::reshape(a,{1})))));
    };
    A x(0.f); float e=1.f/1024;
    auto h=mx::grad(mx::grad(f))(x);
    auto fd=mx::divide(mx::add(f(A(e)),f(A(-e))),A(e*e));
    check("singleton/hessian",h,{1},{});
    check("singleton/fd",fd,{1},{});
    std::cout<<"EVIDENCE singleton second="<<h.item<float>()<<" finite="<<fd.item<float>()<<'\n';
  });
  for(int n:{1,2})scenario("third_order",[&] {
    A x=mx::zeros({n}), y0=mx::logcumsumexp(x);
    mx::eval(y0);
    std::function<A(const A&)> f=[&](const A& t) {
      auto r=mx::subtract(mx::logcumsumexp(mx::add(x,t)),y0);
      return mx::divide(mx::sum(mx::multiply(mx::square(r),r)),A(6.f));
    };
    auto third=mx::grad(mx::grad(mx::grad(f)))(A(0.f));
    check("third_order/n="+std::to_string(n),third,{double(n)},{});
    std::cout<<"EVIDENCE third n="<<n<<" actual="<<third.item<float>()<<" expected="<<n<<'\n';
  });
}
int main() {
  mx::set_default_device(mx::Device(mx::Device::cpu));
  for(auto shape:{mx::Shape{1},mx::Shape{2},mx::Shape{4},mx::Shape{2,3}})
    for(int axis=0;axis<shape.size();++axis)
      for(bool rev:{false,true})for(bool inc:{false,true})for(int mode:{0,1,2,3})
        test(shape,axis,rev,inc,mode);
  test({2,3},1,false,true,0,true);
  simple();
  std::cout<<"SUMMARY scenarios="<<scenarios<<" checks="<<checks<<" failures="<<failures
           <<" failed_scenarios="<<failed_scenarios<<'\n';
  return failures ? 1:0;
}
