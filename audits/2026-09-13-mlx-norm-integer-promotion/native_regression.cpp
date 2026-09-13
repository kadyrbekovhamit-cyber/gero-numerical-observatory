#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include "mlx/mlx.h"
namespace mx = mlx::core;
using A = mx::array;
int cases = 0, failures = 0, values_checked = 0;
void num(double x) {
  if (std::isfinite(x)) std::cout << std::setprecision(17) << x;
  else std::cout << '"' << (std::isnan(x)?"NaN":x>0?"Infinity":"-Infinity") << '"';
}
std::vector<double> values(A x) {
  x=mx::contiguous(mx::astype(x,mx::float64)); mx::eval(x);
  return x.size()?std::vector<double>(x.data<double>(),x.data<double>()+x.size()):std::vector<double>{};
}
void check(const std::string& name, std::function<A()> fn,
           const std::vector<double>& want, const mx::Shape& shape,
           mx::Dtype dtype=mx::float32, double tolerance=3e-6) {
  ++cases;
  try {
    A a=fn(); auto got=values(a);
    bool metadata=a.shape()==shape && a.dtype()==dtype;
    bool ok=metadata && got.size()==want.size();
    for(size_t i=0;i<want.size() && i<got.size();++i) {
      ++values_checked;
      ok &= got[i]==want[i] || (std::isfinite(got[i]) &&
            std::abs(got[i]-want[i])<=tolerance*std::max(1.,std::abs(want[i])));
    }
    failures+=!ok;
    std::cout<<"{\"name\":\""<<name<<"\",\"actual\":[";
    for(size_t i=0;i<got.size();++i){if(i)std::cout<<",";num(got[i]);}
    std::cout<<"],\"expected\":[";
    for(size_t i=0;i<want.size();++i){if(i)std::cout<<",";num(want[i]);}
    std::cout<<"],\"metadata_passed\":"<<(metadata?"true":"false")
             <<",\"passed\":"<<(ok?"true":"false")<<"}\n";
  } catch(const std::exception& e) {
    ++failures;
    std::cout<<"{\"name\":\""<<name<<"\",\"exception\":true,\"passed\":false}\n";
    std::cerr<<name<<": "<<e.what()<<"\n";
  }
}
double reference(const std::vector<long double>& x,double p) {
  long double s=0;
  if(p==0){for(auto z:x)s+=z!=0;return s;}
  if(p==std::numeric_limits<double>::infinity()){for(auto z:x)s=std::max(s,std::abs(z));return s;}
  if(p==-std::numeric_limits<double>::infinity()){s=std::numeric_limits<double>::infinity();for(auto z:x)s=std::min(s,std::abs(z));return s;}
  if(p==2){for(auto z:x)s=std::hypot(s,z);return s;}
  for(auto z:x)s+=std::pow(std::abs(z),(long double)p);
  return std::pow(s,1.L/p);
}
std::string type_name(mx::Dtype dt){std::ostringstream s;s<<dt;return s.str();}
int main(){
  mx::set_default_device(mx::Device::cpu);
  struct Set{mx::Dtype dt;std::vector<int64_t> x;};
  std::vector<Set> sets={
    {mx::int8,{-128,-100,-16,-12,0,12,16,100,127}},
    {mx::uint8,{0,15,16,100,200,255}},
    {mx::int16,{-32768,-30000,-182,-181,0,181,182,30000,32767}},
    {mx::uint16,{0,255,256,50000,65535}},
    {mx::int32,{-2147483648LL,-100000,-46341,-46340,0,46340,46341,65536,2147483647}},
    {mx::uint32,{0,65535,65536,100000,4294967295LL}},
    {mx::int64,{INT64_MIN,-(1LL<<40),-65536,0,65536,1LL<<40,INT64_MAX}},
    {mx::uint64,{0,65535,65536,1LL<<32,1LL<<40,INT64_MAX}},
    {mx::bool_,{0,1}}
  };
  for(const auto& set:sets){
    auto dn=type_name(set.dt);
    for(auto x:set.x){
      A a=mx::reshape(A(x,set.dt),{1});
      for(bool keep:{false,true}){
        auto shape=keep?mx::Shape{1}:mx::Shape{};
        auto prefix=dn+"/"+std::to_string(x)+"/keep"+std::to_string(keep);
        check(prefix+"/default",[&]{return mx::linalg::norm(a,std::nullopt,keep);},{(double)std::abs((long double)x)},shape);
        for(double p:{0.,1.,2.,std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()})
          check(prefix+"/p"+std::to_string(p),[&]{return mx::linalg::norm(a,p,std::nullopt,keep);},{reference({(long double)x},p)},shape);
        if(std::abs((long double)x)<=100000)
          for(double p:{0.5,3.,-1.,-2.})
            check(prefix+"/p"+std::to_string(p),[&]{return mx::linalg::norm(a,p,std::nullopt,keep);},{reference({(long double)x},p)},shape, mx::float32,1e-5);
      }
    }
    // Six exactly represented source values with an independent host reference.
    int64_t q=set.dt==mx::bool_?1:set.dt==mx::int8||set.dt==mx::uint8?100:
      set.dt==mx::int16||set.dt==mx::uint16?30000:65536;
    std::vector<int64_t> v={q,0,1,1,q,0};A m(v.begin(),{2,3},set.dt);
    for(bool keep:{false,true}){
      for(bool transposed:{false,true}){
        A a=transposed?mx::transpose(m):m;
        auto prefix=dn+"/matrix/keep"+std::to_string(keep)+"/transpose"+std::to_string(transposed);
        std::vector<long double> all={ (long double)q,0,1,1,(long double)q,0};
        check(prefix+"/fro",[&]{return mx::linalg::norm(a,std::string("fro"),std::nullopt,keep);},{reference(all,2)},keep?mx::Shape{1,1}:mx::Shape{});
        for(double p:{-1.,1.,std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()}){
          std::vector<long double> rows=transposed?std::vector<long double>{(long double)q+1,(long double)q,1}:std::vector<long double>{(long double)q+1,(long double)q+1};
          std::vector<long double> cols=transposed?std::vector<long double>{(long double)q+1,(long double)q+1}:std::vector<long double>{(long double)q+1,(long double)q,1};
          auto sums=std::abs(p)==1?cols:rows;
          auto ref=p>0?*std::max_element(sums.begin(),sums.end()):*std::min_element(sums.begin(),sums.end());
          check(prefix+"/p"+std::to_string(p),[&]{return mx::linalg::norm(a,p,std::vector<int>{-2,-1},keep);},{(double)ref},keep?mx::Shape{1,1}:mx::Shape{});
        }
      }
      for(int axis:{0,1,-1}){
        std::vector<double> refs=axis==0?std::vector<double>{std::hypot((double)q,1.),(double)q,1.}:std::vector<double>{std::hypot((double)q,1.),std::hypot((double)q,1.)};
        auto shape=keep?(axis==0?mx::Shape{1,3}:mx::Shape{2,1}):(axis==0?mx::Shape{3}:mx::Shape{2});
        check(dn+"/axis"+std::to_string(axis)+"/keep"+std::to_string(keep),[&]{return mx::linalg::norm(m,std::vector<int>{axis},keep);},refs,shape);
      }
    }
    check(dn+"/empty",[&]{return mx::linalg::norm(mx::zeros({0},set.dt));},{0},{});
  }
  for(auto dt:{mx::float16,mx::bfloat16,mx::float32,mx::float64}){
    auto dn=type_name(dt); double tol=dt==mx::float64?1e-12:dt==mx::float32?3e-6:0.01;
    for(double scale:{0.,.5,1.,4.}){
      A a=mx::multiply(A({3.,4.},dt),A(scale,dt));
      check("control/"+dn+"/scale"+std::to_string(scale),[&]{return mx::linalg::norm(a);},{5*scale},{},dt,tol);
    }
    if(dt==mx::float32||dt==mx::float64){
      auto f=[dt](const A& t){return mx::linalg::norm(mx::multiply(A({3.,4.},dt),t));};
      auto g=mx::grad(f);auto h=mx::grad(g);
      for(double t:{-2.,-.5,.5,2.}){
        check("control/"+dn+"/gradient/"+std::to_string(t),[&]{return g(A(t,dt));},{t>0?5.:-5.},{},dt,tol*10);
        check("control/"+dn+"/hessian/"+std::to_string(t),[&]{return h(A(t,dt));},{0},{},dt,tol*10);
      }
    }
  }
  A complex(std::complex<float>(3,4));complex=mx::reshape(complex,{1});
  check("control/complex",[&]{return mx::linalg::norm(complex);},{5},{});
  std::cout<<"{\"summary\":true,\"scenarios\":"<<cases<<",\"numeric_values\":"<<values_checked<<",\"failures\":"<<failures<<"}\n";
  return failures?1:0;
}
