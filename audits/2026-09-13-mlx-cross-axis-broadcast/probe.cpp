#include <iostream>
#include "mlx/mlx.h"
namespace mx=mlx::core;
int main(){mx::set_default_device(mx::Device::cpu);auto a=mx::array({1.f,2.f,3.f});auto b=mx::eye(3);for(int ax:{0,-1}){try{auto y=mx::linalg::cross(a,b,ax);mx::eval(y);std::cout<<"axis="<<ax<<" result="<<y<<"\n";}catch(const std::exception&e){std::cout<<e.what()<<"\n";}}auto ac=mx::reshape(a,{3,1});std::cout<<"explicit axis0="<<mx::linalg::cross(ac,b,0)<<"\n";auto parallel=mx::broadcast_to(ac,{3,3});std::cout<<"parallel="<<mx::linalg::cross(a,parallel,0)<<"\n";std::cout<<"explicit parallel="<<mx::linalg::cross(ac,parallel,0)<<"\n";}
