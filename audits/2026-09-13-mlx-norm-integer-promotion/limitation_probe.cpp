#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include "mlx/mlx.h"
namespace mx=mlx::core;
void show(const char* name,mx::array x,double ord,double expected){
 auto y=mx::linalg::norm(x,ord);mx::eval(y);
 double got=mx::astype(y,mx::float64).item<double>();
 std::cout<<"{\"name\":\""<<name<<"\",\"actual\":";
 if(std::isfinite(got))std::cout<<std::setprecision(17)<<got;
 else std::cout<<'"'<<(std::isnan(got)?"NaN":got>0?"Infinity":"-Infinity")<<'"';
 std::cout<<",\"mathematical_reference\":"<<std::setprecision(17)<<expected<<"}\n";
}
int main(){
 mx::set_default_device(mx::Device::cpu);
 show("uint64_max_l2",mx::reshape(mx::array(UINT64_MAX,mx::uint64),{1}),2.,(double)UINT64_MAX);
 show("int64_max_p3",mx::reshape(mx::array(INT64_MAX,mx::int64),{1}),3.,(double)INT64_MAX);
 show("float32_1e20_l2",mx::array({1e20f}),2.,(double)1e20f);
}
