#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "mlx/mlx.h"
namespace mx = mlx::core;
using A = mx::array;
int checks = 0, failures = 0;
bool widen_f32 = false;
std::pair<A,A> slogdet(const A& a) {
  if (widen_f32 && a.dtype()==mx::float32 && a.shape(-1)<=3) {
    auto result=mx::linalg::slogdet(mx::astype(a,mx::float64));
    return {mx::astype(result.first,mx::float32),mx::astype(result.second,mx::float32)};
  }
  return mx::linalg::slogdet(a);
}
void number(double x) {
  if (std::isfinite(x)) std::cout << std::setprecision(17) << x;
  else std::cout << '"' << (std::isnan(x) ? "NaN" : x > 0 ? "Infinity" : "-Infinity") << '"';
}
void check(std::string name, double got, double want, double tol) {
  bool ok = got == want || (std::isfinite(got) && std::isfinite(want) && std::abs(got-want) <= tol*(1+std::abs(want)));
  checks++; failures += !ok;
  std::cout << "{\"name\":\"" << name << "\",\"actual\":"; number(got);
  std::cout << ",\"expected\":"; number(want);
  std::cout << ",\"tolerance\":" << tol << ",\"passed\":" << (ok ? "true" : "false") << "}\n";
}
std::vector<double> vals(A x) {
  x=mx::contiguous(mx::astype(x,mx::float64)); mx::eval(x);
  return x.size() ? std::vector<double>(x.data<double>(),x.data<double>()+x.size()) : std::vector<double>{};
}
void run(std::string name, A x, const std::vector<double>& signs, const std::vector<double>& logs, double tol) {
  auto [s,l] = slogdet(x);
  auto sv=vals(s), lv=vals(l);
  check(name+"/size",sv.size(),signs.size(),0);
  check(name+"/dtype",l.dtype()==x.dtype(),1,0);
  for (size_t i=0;i<signs.size();i++) {
    check(name+"/sign/"+std::to_string(i),sv.at(i),signs[i],0);
    check(name+"/log/"+std::to_string(i),lv.at(i),logs[i],tol);
  }
}
int main(int argc, char** argv) {
  bool skip_autodiff=argc>1 && std::string(argv[1])=="--skip-autodiff";
  widen_f32=argc>1 && std::string(argv[1])=="--widen-f32";
  mx::set_default_device(mx::Device::cpu);
  for(auto dt:{mx::float32,mx::float64}) {
    const bool f32=dt==mx::float32; const std::string d=f32?"f32":"f64";
    const double tol=f32?2e-6:2e-13;
    const auto es=f32?std::vector<int>{-120,-80,-40,0,40,80,120}:std::vector<int>{-900,-600,-300,0,300,600,900};
    for(int n:{2,3,4}) for(int e:es) for(int variant:{0,1,2,3}) {
      std::vector<double> x(n*n,0); double base_det=1;
      // Integer triangular factors have a known determinant; row powers of
      // two give independent log-domain references without forming det(A).
      int exponent_sum=0;
      for(int i=0;i<n;i++) {
        int ei=variant==3 ? (i%2?-e:e) : e;
        exponent_sum+=ei;
        x[i*n+i]=std::ldexp(i==0 && variant==1?-1.:1.,ei);
        if(variant>=2 && i+1<n) x[i*n+i+1]=std::ldexp(.5,ei);
      }
      if(variant==1) base_det=-1;
      A a=mx::astype(A(x.data(),{n,n},mx::float64),dt);
      if(variant==2) a=mx::swapaxes(a,-1,-2);
      run(d+"/n"+std::to_string(n)+"/e"+std::to_string(e)+"/v"+std::to_string(variant),a,{base_det},{exponent_sum*std::log(2.)},tol);
    }
    for(int n:{2,3}) {
      // Dense, non-singular integer matrices: det([[2,1],[1,2]])=3;
      // det of the 3x3 matrix with diagonal 2 and off-diagonal 1 is 4.
      std::vector<double> packed; std::vector<double> signs,logs;
      for(int e:es) {
        for(int i=0;i<n;i++) for(int j=0;j<n;j++) packed.push_back(std::ldexp(i==j?2.:1.,e));
        signs.push_back(1); logs.push_back(std::log(n+1.)+n*e*std::log(2.));
      }
      run(d+"/dense-batch/n"+std::to_string(n),mx::astype(A(packed.data(),{int(es.size()),n,n},mx::float64),dt),signs,logs,tol);
      for(int e:{es.front(),0,es.back()}) {
        A a=mx::full({n,n},A(std::ldexp(1.,e),dt),dt);
        run(d+"/singular/n"+std::to_string(n)+"/e"+std::to_string(e),a,{0},{-INFINITY},tol);
      }
      if (skip_autodiff) continue;
      auto f=[n,dt](const A& t){return slogdet(mx::multiply(mx::eye(n,dt),t)).second;};
      auto g=mx::grad(f); auto h=mx::grad(g);
      for(double t:{-.5,.5,2.,8.}) {
        for(bool second:{false,true}) {
          auto name=d+(second?"/hessian/n":"/grad/n")+std::to_string(n)+"/t"+std::to_string(t);
          try {
            check(name,vals(second?h(A(t,dt)):g(A(t,dt)))[0],second?-n/(t*t):n/t,tol*10);
          } catch(const std::exception& error) {
            checks++;failures++;
            std::cout<<"{\"name\":\""<<name<<"\",\"error\":\""<<error.what()<<"\",\"passed\":false}\n";
          }
        }
      }
    }
    // This exactly represented matrix has determinant 1. Row normalization
    // can erase its second column even though the direct formula is finite.
    double hi=std::ldexp(1.,f32?120:600),lo=1./hi;
    double wide[]{hi,lo,hi,2*lo};
    run(d+"/wide-row-counterexample",mx::astype(A(wide,{2,2},mx::float64),dt),{1},{0},tol);
    run(d+"/empty",mx::zeros({0,0},dt),{1},{0},tol);
    for(double t:{-2.,0.,.5}) run(d+"/scalar"+std::to_string(t),mx::full({1,1},A(t,dt),dt),{t==0?0.:t>0?1.:-1.},{std::log(std::abs(t))},tol);
  }
  std::cout<<"{\"summary\":true,\"checks\":"<<checks<<",\"failures\":"<<failures<<"}\n";
  return failures?1:0;
}
