#include <cmath>
#include <fstream>
#include <iostream>
#include <functional>
#include <sstream>
#include "mlx/mlx.h"
#include "nlohmann/json.hpp"
namespace mx=mlx::core;
using J=nlohmann::json;
using A=mx::array;
std::string dtype_name(mx::Dtype d){std::ostringstream s;s<<d;return s.str();}
int scenarios=0,failures=0,comparisons=0;
std::vector<double> vals(A a){a=mx::contiguous(mx::astype(a,mx::float64));mx::eval(a);return a.size()?std::vector<double>(a.data<double>(),a.data<double>()+a.size()):std::vector<double>{};}
A make(const J& v,const J& shape,mx::Dtype dt){auto x=v.get<std::vector<double>>();return A(x.begin(),shape.get<mx::Shape>(),dt);}
void check(std::string name,std::function<A()> fn,const std::vector<double>& want,mx::Shape shape,mx::Dtype dt,bool control=false){
  ++scenarios;J r={{"name",name},{"control",control}};
  try{auto y=fn();auto got=vals(y);bool ok=y.shape()==shape&&y.dtype()==dt&&got.size()==want.size();r["shape"]=y.shape();r["dtype"]=dtype_name(y.dtype());
    for(size_t i=0;i<got.size()&&i<want.size();++i){++comparisons;ok &= std::isfinite(got[i])&&std::abs(got[i]-want[i])<=1e-6*std::max(1.,std::abs(want[i]));}
    r["actual"]=got;r["expected"]=want;r["passed"]=ok;failures+=!ok;
  }catch(const std::exception&e){r["passed"]=false;r["exception"]=e.what();++failures;}
  std::cout<<r.dump()<<'\n';
}
int main(int argc,char**argv){
  if(argc!=2)return 2;mx::set_default_device(mx::Device::cpu);std::ifstream f(argv[1]);J cases;f>>cases;
  for(auto&c:cases){
    auto want=c["expected"].get<std::vector<double>>();auto shape=c["shape"].get<mx::Shape>();int axis=c["axis"];std::string n=c["name"];bool ctrl=c["control"];
    for(auto dt:{mx::float16,mx::bfloat16,mx::float32,mx::float64,mx::int8,mx::int32}){
      auto a=make(c["a"],c["ashape"],dt),b=make(c["b"],c["bshape"],dt);auto prefix=n+"/dtype"+dtype_name(dt);
      check(prefix,[&]{return mx::linalg::cross(a,b,axis);},want,shape,dt,ctrl);
      if(dt==mx::float32){
        auto view=[](A x){return mx::transpose(mx::contiguous(mx::transpose(x)));};auto av=view(a),bv=view(b);
        check(prefix+"/strided",[&]{return mx::linalg::cross(av,bv,axis);},want,shape,dt,ctrl);
        auto ib=make(c["b"],c["bshape"],mx::int32);
        check(prefix+"/mixed",[&]{return mx::linalg::cross(a,ib,axis);},want,shape,dt,ctrl);
        auto ca=mx::multiply(a,A(mx::complex64_t(1,2))),cb=mx::multiply(b,A(mx::complex64_t(2,-1)));
        std::vector<double> re=want,im=want;for(auto&x:re)x*=4;for(auto&x:im)x*=3;
        check(prefix+"/complex-real",[&]{return mx::real(mx::linalg::cross(ca,cb,axis));},re,shape,dt,ctrl);
        check(prefix+"/complex-imag",[&]{return mx::imag(mx::linalg::cross(ca,cb,axis));},im,shape,dt,ctrl);
      }
    }
    if(c["autodiff"].get<bool>()){
      auto a=make(c["a"],c["ashape"],mx::float64),b=make(c["b"],c["bshape"],mx::float64);
      auto fn=[axis](const std::vector<A>&x){return std::vector<A>{mx::linalg::cross(x[0],x[1],axis)};};
      auto w=make(c["w"],c["shape"],mx::float64);auto ta=make(c["ta"],c["ashape"],mx::float64),tb=make(c["tb"],c["bshape"],mx::float64);
      check(n+"/vjp-a",[&]{return mx::vjp(fn,{a,b},{w}).second[0];},c["ga"],c["ashape"],mx::float64,ctrl);
      check(n+"/vjp-b",[&]{return mx::vjp(fn,{a,b},{w}).second[1];},c["gb"],c["bshape"],mx::float64,ctrl);
      check(n+"/jvp",[&]{return mx::jvp(fn,{a,b},{ta,tb}).second[0];},c["jvp"],shape,mx::float64,ctrl);
    }
  }
  auto a=A({1.f,2.f,3.f});auto parallel=mx::broadcast_to(mx::reshape(a,{3,1}),{3,3});
  check("parallel_vectors_must_give_zero",[&]{return mx::linalg::cross(a,parallel,0);},std::vector<double>(9,0),{3,3},mx::float32);
  auto loss=[](const A&t){auto a=mx::stack({A(1.,mx::float64),A(2.,mx::float64),t});auto y=mx::linalg::cross(a,mx::eye(3,mx::float64),0);auto z=mx::sum(mx::multiply(y,A({0.,1.,0.,0.,0.,0.,0.,0.,0.},{3,3},mx::float64)));return mx::multiply(A(.5,mx::float64),mx::square(z));};
  auto g=mx::grad(loss);auto h=mx::grad(g);auto t=A(3.,mx::float64);
  check("selected_component_loss",[&]{return loss(t);},{4.5},{},mx::float64);
  check("selected_component_gradient",[&]{return g(t);},{3.},{},mx::float64);
  check("selected_component_hessian",[&]{return h(t);},{1.},{},mx::float64);
  for(int kind=0;kind<5;++kind){++scenarios;bool threw=false;try{A x=A({1.f,2.f,3.f}),y=x;if(kind==0)x=A({1.f});if(kind==1)x=A(1.f);if(kind==4){x=mx::ones({2,3});y=mx::ones({4,3});}mx::eval(mx::linalg::cross(x,y,kind==2?1:kind==3?-2:-1));}catch(const std::invalid_argument&){threw=true;}failures+=!threw;std::cout<<J{{"name","invalid_input_"+std::to_string(kind)},{"control",true},{"passed",threw}}.dump()<<'\n';}
  std::cout<<J{{"summary",true},{"scenarios",scenarios},{"numerical_comparisons",comparisons},{"failures",failures}}.dump()<<'\n';return failures?1:0;
}
