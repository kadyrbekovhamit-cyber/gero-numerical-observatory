#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <optional>
#include <string>
#include <vector>
#include "mlx/mlx.h"

namespace mx = mlx::core;
using A = mx::array;
int scenarios = 0, failed_scenarios = 0, checks = 0, failures = 0;

std::vector<double> values(const A& x) {
  auto a = mx::contiguous(mx::astype(x, mx::float32));
  mx::eval(a);
  return {a.data<float>(), a.data<float>() + a.size()};
}

void check(const std::string& label, const A& actual,
           const std::vector<double>& expected, const mx::Shape& shape,
           double tol = 3e-5) {
  ++checks;
  auto a = values(actual);
  bool ok = actual.shape() == shape && a.size() == expected.size();
  double error = 0;
  for (size_t i = 0; i < a.size() && i < expected.size(); ++i) {
    error = std::max(error, std::abs(a[i] - expected[i]));
    ok &= std::isfinite(a[i]) &&
        std::abs(a[i] - expected[i]) <= tol * (1 + std::abs(expected[i]));
  }
  if (!ok) {
    ++failures;
    std::cout << "FAIL " << label << " error=" << error << '\n';
  }
}

void scenario(const std::string& label, const std::function<void()>& f) {
  ++scenarios;
  auto before = failures;
  try { f(); } catch (const std::exception& e) {
    ++failures;
    std::cout << "FAIL " << label << " exception=" << e.what() << '\n';
  }
  if (failures != before) ++failed_scenarios;
}

struct Case {
  std::string name;
  int ba, bb;
  std::vector<uint32_t> lhs, rhs;
  std::vector<int> il, ir;
};

std::optional<A> index(const std::vector<uint32_t>& v) {
  if (v.empty()) return std::nullopt;
  return A(v.data(), {static_cast<int>(v.size())}, mx::uint32);
}

std::vector<Case> cases() {
  return {
    {"left_repeat", 3, 3, {0,0,2}, {}, {0,0,2}, {0,1,2}},
    {"left_subset", 4, 3, {0,1,1}, {}, {0,1,1}, {0,1,2}},
    {"left_identity", 3, 3, {0,1,2}, {}, {0,1,2}, {0,1,2}},
    {"left_scalar_broadcast", 2, 3, {1}, {}, {1,1,1}, {0,1,2}},
    {"right_repeat", 3, 4, {}, {0,0,2}, {0,1,2}, {0,0,2}},
    {"right_left_broadcast", 1, 3, {}, {0,1,1}, {0,0,0}, {0,1,1}},
    {"both_explicit", 3, 3, {0,0,2}, {1,2,2}, {0,0,2}, {1,2,2}},
    {"right_scalar_broadcast", 3, 2, {}, {1}, {0,1,2}, {1,1,1}}
  };
}

A data(int count, const mx::Shape& shape, mx::Dtype dtype, int shift) {
  std::vector<float> v(count);
  for (int i=0;i<count;++i) v[i] = ((i*3+shift)%11-5)/8.f;
  return mx::astype(A(v.data(),shape,mx::float32),dtype);
}

void mm(const Case& c, bool sorted, mx::Dtype dtype, bool strided) {
  int M=2,K=3,N=2,Q=c.il.size();
  auto a=data(c.ba*M*K,{c.ba,M,K},dtype,0);
  auto b=data(c.bb*K*N,{c.bb,K,N},dtype,2);
  if(strided) {
    a=mx::swapaxes(mx::contiguous(mx::swapaxes(a,-1,-2)),-1,-2);
    b=mx::swapaxes(mx::contiguous(mx::swapaxes(b,-1,-2)),-1,-2);
  }
  if(c.ba==1) {
    auto backing=data(Q*M*K,{Q,M,K},dtype,0);
    mx::eval(backing);
    a=mx::slice(backing,{0,0,0},{1,M,K});
  }
  auto cot=data(Q*M*N,{Q,M,N},dtype,4);
  auto av=values(a),bv=values(b),cv=values(cot);
  std::vector<double> y(Q*M*N),da(a.size()),db(b.size());
  for(int q=0;q<Q;++q) for(int i=0;i<M;++i) for(int j=0;j<N;++j)
    for(int k=0;k<K;++k) {
      int oi=(q*M+i)*N+j,ai=(c.il[q]*M+i)*K+k,bi=(c.ir[q]*K+k)*N+j;
      y[oi]+=av[ai]*bv[bi]; da[ai]+=cv[oi]*bv[bi]; db[bi]+=av[ai]*cv[oi];
    }
  std::string label="mm/"+c.name+"/sorted="+std::to_string(sorted)+
      "/dtype="+std::to_string(static_cast<int>(dtype.val()))+"/strided="+std::to_string(strided);
  auto f=[&](const std::vector<A>& v) {return std::vector<A>{
      mx::gather_mm(v[0],v[1],index(c.lhs),index(c.rhs),sorted)};};
  double tol=dtype==mx::float32 ? 3e-5 : .02;
  scenario(label,[&] {
    auto [out,g]=mx::vjp(f,{a,b},{cot});
    check(label+"/forward",out[0],y,{Q,M,N},tol);
    check(label+"/da",g[0],da,a.shape(),tol);
    check(label+"/db",g[1],db,b.shape(),tol);
    auto onea=mx::vjp([&](const std::vector<A>& v){return f({v[0],b});},{a},{cot});
    auto oneb=mx::vjp([&](const std::vector<A>& v){return f({a,v[0]});},{b},{cot});
    check(label+"/only_da",onea.second[0],da,a.shape(),tol);
    check(label+"/only_db",oneb.second[0],db,b.shape(),tol);
    auto zero=mx::vjp(f,{a,b},{mx::zeros_like(cot)}).second;
    check(label+"/zero_da",zero[0],std::vector<double>(a.size()),a.shape(),tol);
    check(label+"/zero_db",zero[1],std::vector<double>(b.size()),b.shape(),tol);
    auto neg=mx::vjp(f,{a,b},{mx::negative(cot)}).second;
    for(auto& x:da)x=-x;
    for(auto& x:db)x=-x;
    check(label+"/negative_da",neg[0],da,a.shape(),tol);
    check(label+"/negative_db",neg[1],db,b.shape(),tol);
  });
}

void qmm(const Case& c, bool sorted, int bits, mx::Dtype dtype) {
  int M=1,K=32,N=32,Q=c.il.size(),per_word=32/bits,words=K/per_word;
  std::vector<uint32_t> codes(c.bb*N*K),packed(c.bb*N*words);
  for(int e=0;e<c.bb;++e) for(int n=0;n<N;++n) for(int k=0;k<K;++k) {
    int i=(e*N+n)*K+k;
    codes[i]=(k+n+2*e)%((1<<bits)-1);
    packed[(e*N+n)*words+k/per_word] |= codes[i] << (bits*(k%per_word));
  }
  A w(packed.data(),{c.bb,N,words},mx::uint32);
  A x=data(c.ba*M*K,{c.ba,M,K},dtype,0);
  if(c.ba==1) {
    auto backing=data(Q*M*K,{Q,M,K},dtype,0);
    mx::eval(backing);
    x=mx::slice(backing,{0,0,0},{1,M,K});
  }
  A s=mx::full({c.bb,N,1},.125f,dtype);
  A bias=data(c.bb*N,{c.bb,N,1},dtype,2);
  A cot=data(Q*M*N,{Q,M,N},dtype,4);
  auto xv=values(x),sv=values(s),bv=values(bias),cv=values(cot);
  std::vector<double> y(Q*M*N),dx(x.size()),ds(s.size()),db(bias.size());
  for(int q=0;q<Q;++q) for(int m=0;m<M;++m) for(int n=0;n<N;++n)
    for(int k=0;k<K;++k) {
      int xi=(c.il[q]*M+m)*K+k,si=c.ir[q]*N+n,wi=si*K+k,oi=(q*M+m)*N+n;
      double weight=codes[wi]*sv[si]+bv[si];
      y[oi]+=xv[xi]*weight; dx[xi]+=cv[oi]*weight;
      ds[si]+=cv[oi]*xv[xi]*codes[wi]; db[si]+=cv[oi]*xv[xi];
    }
  auto f=[&](const std::vector<A>& v){return std::vector<A>{mx::gather_qmm(
      v[0],w,v[1],v[2],index(c.lhs),index(c.rhs),true,32,bits,"affine",sorted)};};
  std::string label="qmm/"+c.name+"/sorted="+std::to_string(sorted)+
      "/bits="+std::to_string(bits)+"/dtype="+std::to_string(static_cast<int>(dtype.val()));
  double tol=dtype==mx::float32 ? 3e-5 : .03;
  scenario(label,[&] {
    auto [out,g]=mx::vjp(f,{x,s,bias},{cot});
    check(label+"/forward",out[0],y,{Q,M,N},tol);
    check(label+"/dx",g[0],dx,x.shape(),tol);
    check(label+"/ds",g[1],ds,s.shape(),tol);
    check(label+"/db",g[2],db,bias.shape(),tol);
    for(int arg=0;arg<3;++arg) {
      std::vector<A> primals={x,s,bias};
      auto partial=mx::vjp([&](const std::vector<A>& v){auto p=primals;p[arg]=v[0];return f(p);},
                           {primals[arg]},{cot}).second[0];
      check(label+"/only_"+std::to_string(arg),partial,
            arg==0 ? dx : (arg==1 ? ds : db),primals[arg].shape(),tol);
    }
    auto zero=mx::vjp(f,{x,s,bias},{mx::zeros_like(cot)}).second;
    check(label+"/zero_dx",zero[0],std::vector<double>(x.size()),x.shape(),tol);
    check(label+"/zero_ds",zero[1],std::vector<double>(s.size()),s.shape(),tol);
    check(label+"/zero_db",zero[2],std::vector<double>(bias.size()),bias.shape(),tol);
  });
}

void loss_test() {
  scenario("loss_and_second_derivative",[] {
    A a({2.f,7.f},{2,1,1}), b({3.f,5.f},{2,1,1});
    A ids({0,0},{2},mx::uint32);
    auto loss=[&](const A& a){
      auto y=mx::reshape(mx::gather_mm(a,b,ids,std::nullopt,true),{2});
      auto diff=mx::subtract(mx::take(y,A(0)),mx::take(y,A(1)));
      return mx::multiply(A(.5f),mx::square(diff));
    };
    auto g=mx::grad(loss);
    check("loss/initial",loss(a),{8.},{});
    check("loss/gradient",g(a),{8.,0.},a.shape());
    auto next=mx::subtract(a,mx::divide(g(a),A(32.f)));
    check("loss/step",loss(next),{6.125},{});
    float eps=1.f/1024;
    for(int i=0;i<2;++i) {
      std::vector<float> h(2);h[i]=eps;
      A d(h.data(),a.shape(),mx::float32);
      auto fd=mx::divide(mx::subtract(loss(mx::add(a,d)),loss(mx::subtract(a,d))),A(2*eps));
      check("loss/fd"+std::to_string(i),fd,{i==0 ? 8. : 0.},{});
    }
    auto h=mx::grad([&](const A& v){return mx::sum(g(v));});
    check("loss/hessian_sum",h(a),{4.,0.},a.shape());
  });
}

void view_isolation() {
  for(const auto& tail:std::vector<std::vector<float>>{{101.f,202.f},{-31.f,47.f}}) {
    for(bool sorted:{false,true}) scenario("view_isolation",[&] {
      A backing({2.f,tail[0],tail[1]},{3,1,1});
      mx::eval(backing);
      auto a=mx::slice(backing,{0,0,0},{1,1,1});
      A b({3.f,5.f,7.f},{3,1,1}), ids({0,1,2},{3},mx::uint32);
      auto loss=[&](const A& b){return mx::sum(mx::gather_mm(a,b,std::nullopt,ids,sorted));};
      check("view/visible",a,{2.},a.shape());
      check("view/loss",loss(b),{30.},{});
      check("view/db",mx::grad(loss)(b),{2.,2.,2.},b.shape());
    });
  }
}

int main() {
  mx::set_default_device(mx::Device(mx::Device::cpu));
  for(const auto& c:cases()) for(bool sorted:{false,true}) {
    mm(c,sorted,mx::float32,false);
    mm(c,sorted,mx::float32,true);
    for(int bits:{4,8}) qmm(c,sorted,bits,mx::float32);
  }
  loss_test();
  view_isolation();
  std::cout<<"SUMMARY scenarios="<<scenarios<<" failed_scenarios="<<failed_scenarios
           <<" checks="<<checks<<" failures="<<failures<<'\n';
  return failures ? 1 : 0;
}
