#include <iostream>
#include <limits>
#include "mlx/mlx.h"
namespace mx=mlx::core;
int main(){
 mx::set_default_device(mx::Device::cpu);
 auto p=[](std::string label,mx::array a,mx::array b,double r,double t){
 try{auto y=mx::isclose(a,b,r,t);auto z=mx::allclose(a,b,r,t);mx::eval(y,z);std::cout<<label<<" a="<<a<<" b="<<b<<" rtol="<<r<<" atol="<<t<<" isclose="<<y<<" allclose="<<z<<"\n";}
 catch(const std::exception&e){std::cout<<label<<" ERROR "<<e.what()<<"\n";}};
 p("uint8 small gap",mx::array({0},mx::uint8),mx::array({1},mx::uint8),0,1);
 p("uint8 wrapped gap",mx::array({0},mx::uint8),mx::array({255},mx::uint8),0,1);
 p("int8 minimum",mx::array({-128},mx::int8),mx::array({0},mx::int8),0,0);
 p("int32 minimum",mx::array({std::numeric_limits<int32_t>::min()}),mx::array({0}),0,0);
 p("int8 near min",mx::array({-127},mx::int8),mx::array({-128},mx::int8),0.01,0);
 p("bool",mx::array({false}),mx::array({true}),0,1);
 try{auto a=mx::broadcast_to(mx::eye(2),{3,2,2});auto b=mx::array({1.f,2.f}); std::cout<<"batched solve="<<mx::linalg::solve(a,b)<<"\n";}catch(const std::exception&e){std::cout<<"batched solve ERROR "<<e.what()<<"\n";}
}
