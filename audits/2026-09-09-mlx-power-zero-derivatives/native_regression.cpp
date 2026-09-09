#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "mlx/mlx.h"

namespace mx=mlx::core;
using A=mx::array;
using Fn=std::function<A(const A&)>;
int cases=0,failed_cases=0,checks=0,failures=0;

std::vector<double> values(const A& input) {
  auto x=mx::contiguous(input);mx::eval(x);
  std::vector<double> result(x.size());
  for(int i=0;i<x.size();++i)result[i]=x.data<float>()[i];
  return result;
}
void check(const std::string& label,const A& actual,std::vector<double> expected,double tol=3e-5) {
  ++checks;auto got=values(actual);bool ok=got.size()==expected.size();
  for(int i=0;ok && i<got.size();++i)
    ok=std::isfinite(got[i]) && std::abs(got[i]-expected[i])<=tol*(1+std::abs(expected[i]));
  if(!ok){++failures;std::cout<<"FAIL "<<label<<" got=";
    for(auto x:got)std::cout<<x<<',';
    std::cout<<" expected=";for(auto x:expected)std::cout<<x<<',';std::cout<<'\n';}
}
void scenario(const std::string& label,const std::function<void()>& run) {
  ++cases;int before=failures;
  try{run();}catch(const std::exception& e){++failures;++checks;std::cout<<"FAIL "<<label<<" exception="<<e.what()<<'\n';}
  failed_cases+=failures!=before;
}
double derivative(float x,int n,int order) {
  if(order>n)return 0;
  double factor=1;
  for(int i=0;i<order;++i)factor*=n-i;
  return factor*std::pow(double(x),n-order);
}
int main() {
  mx::set_default_device(mx::Device::cpu);std::cout<<std::setprecision(10);
  for(int n=0;n<=4;++n) {
    int point=0;
    for(float x:{0.f,-0.f,-2.f,0.5f,2.f}) {
      std::string label="polynomial_"+std::to_string(n)+"_point_"+std::to_string(point++);
      scenario(label,[=](){
        Fn f=[=](const A& a){return mx::power(a,A(float(n)));};
        Fn df=f;
        for(int order=0;order<=5;++order){
          check(label+" derivative_"+std::to_string(order),df(A(x)),{derivative(x,n,order)});
          if(order<5)df=mx::grad(df);
        }
        auto [y,j]=mx::jvp(f,A(x),A(1.5f));
        auto [z,g]=mx::vjp(f,A(x),A(-2.5f));
        double d=derivative(x,n,1);
        check(label+" jvp",j,{1.5*d});check(label+" vjp",g,{-2.5*d});
        auto [o,zero]=mx::vjp(f,A(x),A(0.f));check(label+" zero_upstream",zero,{0});
        check(label+" fd",mx::divide(mx::subtract(f(A(x+1.f/1024)),f(A(x-1.f/1024))),A(2.f/1024)),{d},2e-4);
      });
    }
  }
  for(float a:{0.5f,2.f,3.f})for(float b:{-2.f,0.f,1.f,2.5f}) {
    std::string label="mixed_"+std::to_string(a)+"_"+std::to_string(b);
    scenario(label,[=](){
      std::function<A(const A&,const A&)> f=[](const A& x,const A& y){return mx::power(x,y);};
      auto ga=[=](const A& x,const A& y){return mx::grad([=](const A& v){return f(v,y);})(x);};
      auto gb=[=](const A& x,const A& y){return mx::grad([=](const A& v){return f(x,v);})(y);};
      double out=std::pow(double(a),b),ln=std::log(double(a));
      double da=b*std::pow(double(a),b-1),db=out*ln;
      double dab=std::pow(double(a),b-1)*(1+b*ln);
      check(label+" forward",f(A(a),A(b)),{out});
      check(label+" da",ga(A(a),A(b)),{da});check(label+" db",gb(A(a),A(b)),{db});
      check(label+" daa",mx::grad([=](const A& x){return ga(x,A(b));})(A(a)),{b*(b-1)*std::pow(double(a),b-2)});
      check(label+" dbb",mx::grad([=](const A& y){return gb(A(a),y);})(A(b)),{out*ln*ln});
      check(label+" dab",mx::grad([=](const A& y){return ga(A(a),y);})(A(b)),{dab});
      check(label+" dba",mx::grad([=](const A& x){return gb(x,A(b));})(A(a)),{dab});
      Fn first_b=[=](const A& y){return mx::grad([=](const A& v){return ga(A(a),v);})(y);};
      check(label+" dabb",mx::grad(first_b)(A(b)),{std::pow(double(a),b-1)*(2*ln+b*ln*ln)});
      auto fn=[=](const std::vector<A>& v){return std::vector<A>{f(v[0],v[1])};};
      auto [y,t]=mx::jvp(fn,{A(a),A(b)},{A(0.5f),A(-0.25f)});
      auto [z,g]=mx::vjp(fn,{A(a),A(b)},{A(1.25f)});
      check(label+" jvp_both",t[0],{0.5*da-0.25*db});
      check(label+" vjp_base",g[0],{1.25*da});check(label+" vjp_exponent",g[1],{1.25*db});
    });
  }
  scenario("fractional_first_at_zero",[](){
    Fn f=[](const A& a){return mx::power(a,A(1.5f));};
    check("fractional forward",f(A(0.f)),{0});check("fractional first",mx::grad(f)(A(0.f)),{0});
  });
  scenario("polynomial_features_loss",[](){
    Fn model=[](const A& x){return mx::sum(mx::multiply(A({1.f,2.f,3.f}),mx::power(x,A({0.f,1.f,2.f}))));};
    Fn loss=[=](const A& x){auto d=mx::subtract(model(x),A(4.f));return mx::multiply(A(0.5f),mx::multiply(d,d));};
    A x(0.f);auto g=mx::grad(loss)(x);auto next=loss(mx::subtract(x,mx::multiply(A(0.125f),g)));
    check("features forward",model(x),{1});check("features first",mx::grad(model)(x),{2});
    check("features second",mx::grad(mx::grad(model))(x),{6});
    check("features initial_loss",loss(x),{4.5});check("features loss_gradient",g,{-6});
    check("features next_loss",next,{0.017578125});
    std::cout<<"EVIDENCE loss initial=4.5 next="<<values(next)[0]<<'\n';
  });
  scenario("broadcast_base_at_zero",[](){
    A x=mx::reshape(A({0.f,2.f}),{2,1});A b=mx::reshape(A({0.f,1.f,2.f}),{1,3});
    A cot=mx::reshape(A({1.f,2.f,3.f,-1.f,4.f,-5.f}),{2,3});
    Fn f=[=](const A& a){return mx::power(a,b);};
    auto [y,g]=mx::vjp(f,x,cot);
    check("broadcast forward",y,{1,0,0,1,2,4});check("broadcast gradient",g,{2,-16});
    auto [z,j]=mx::jvp(f,x,mx::reshape(A({1.f,-2.f}),{2,1}));
    check("broadcast jvp",j,{0,1,0,0,-2,-8});
  });
  scenario("broadcast_exponent_at_zero",[](){
    A a({0.5f,2.f,3.f});Fn f=[=](const A& b){return mx::sum(mx::power(a,b));};
    check("exponent forward",f(A(0.f)),{3});
    check("exponent first",mx::grad(f)(A(0.f)),{std::log(3.)});
    check("exponent second",mx::grad(mx::grad(f))(A(0.f)),{2*std::log(2.)*std::log(2.)+std::log(3.)*std::log(3.)});
  });
  std::cout<<"SUMMARY cases="<<cases<<" passed_cases="<<cases-failed_cases<<" failed_cases="<<failed_cases
           <<" checks="<<checks<<" failures="<<failures<<'\n';
  return failures?1:0;
}
