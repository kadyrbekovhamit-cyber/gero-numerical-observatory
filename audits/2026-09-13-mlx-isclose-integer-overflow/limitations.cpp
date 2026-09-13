#include <iostream>
#include <cstdint>
#include "mlx/mlx.h"
namespace mx=mlx::core;
int main(){mx::set_default_device(mx::Device::cpu);
 auto a=mx::array(int64_t(9007199254740993LL),mx::int64),b=mx::array(int64_t(9007199254740992LL),mx::int64);
 std::cout<<"int64 adjacent exact original="<<mx::isclose(a,b,0,0)<<" naive_float64="<<mx::isclose(mx::astype(a,mx::float64),mx::astype(b,mx::float64),0,0)<<" expected=false\n";
 auto c=mx::array(int32_t(16777217),mx::int32),d=mx::array(int32_t(16777216),mx::int32);
 std::cout<<"int32 adjacent original="<<mx::isclose(c,d,0,0)<<" naive_float32="<<mx::isclose(mx::astype(c,mx::float32),mx::astype(d,mx::float32),0,0)<<" expected=false\n";
 auto u=mx::array(uint64_t(0),mx::uint64),v=mx::array(UINT64_MAX,mx::uint64);
 std::cout<<"uint64 wrap original="<<mx::isclose(u,v,0,1)<<" expected=false (outside prototype correction)\n";
}
