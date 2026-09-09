#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include "mlx/mlx.h"

namespace mx = mlx::core;
using A = mx::array;
using Op = std::function<A(const A&, const A&)>;
int cases=0, failed_cases=0, checks=0, failures=0;

std::vector<double> values(const A& x) {
  auto dense=mx::contiguous(x);
  mx::eval(dense);
  std::vector<double> r(dense.size());
  for (int i=0;i<dense.size();++i) r[i]=dense.data<float>()[i];
  return r;
}
void check(std::string label,const A& a,const std::vector<double>& expected,double tol=2e-6) {
  ++checks;
  auto got=values(a);
  bool ok=got.size()==expected.size();
  for (int i=0;ok && i<got.size();++i)
    ok=std::isfinite(got[i]) && std::abs(got[i]-expected[i])<=tol*(1+std::abs(expected[i]));
  if (!ok) {
    ++failures;
    std::cout<<"FAIL "<<label<<" got=";
    for(auto x:got) std::cout<<x<<',';
    std::cout<<" expected=";
    for(auto x:expected) std::cout<<x<<',';
    std::cout<<'\n';
  }
}

void test(std::string label,bool is_max,Op op,std::vector<float> source,mx::Shape source_shape,
          std::vector<float> updates,mx::Shape update_shape,std::vector<int> targets,
          bool coordinate_fd=false) {
  ++cases; int before=failures;
  try {
  A src(source.data(),source_shape,mx::float32),upd(updates.data(),update_shape,mx::float32);
  std::vector<float> cot(source.size());
  for(int i=0;i<cot.size();++i) cot[i]=(i%3==0)?2.f:((i%3==1)?-3.f:4.f);
  A c(cot.data(),source_shape,mx::float32);
  std::vector<double> out(source.begin(),source.end()),ds(source.size()),du(updates.size());
  std::vector<int> counts(source.size());
  for(int i=0;i<updates.size();++i) {
    int j=targets[i];
    out[j]=is_max?std::max(out[j],double(updates[i])):std::min(out[j],double(updates[i]));
  }
  for(int i=0;i<updates.size();++i) if(updates[i]==out[targets[i]]) ++counts[targets[i]];
  for(int j=0;j<source.size();++j) if(counts[j]==0) ds[j]=cot[j];
  for(int i=0;i<updates.size();++i) {
    int j=targets[i];
    if(updates[i]==out[j]) du[i]=cot[j]/double(counts[j]);
  }
  auto fn=[op](const std::vector<A>& v){return std::vector<A>{op(v[0],v[1])};};
  auto [outputs,grads]=mx::vjp(fn,{src,upd},{c});
  check(label+" forward",outputs[0],out);
  check(label+" source_vjp",grads[0],ds);
  check(label+" updates_vjp",grads[1],du);
  check(label+" source_only",mx::grad([&](const A& x){return mx::sum(mx::multiply(op(x,upd),c));})(src),ds);
  check(label+" updates_only",mx::grad([&](const A& x){return mx::sum(mx::multiply(op(src,x),c));})(upd),du);
  double total=std::accumulate(cot.begin(),cot.end(),0.0);
  check(label+" cotangent_conservation",mx::add(mx::sum(grads[0]),mx::sum(grads[1])),{total});
  auto shift=[=](const A& t){return mx::sum(mx::multiply(op(mx::add(src,t),mx::add(upd,t)),c));};
  A zero(0.f),eps(1.f/256);
  check(label+" shift_derivative",mx::grad(shift)(zero),{total});
  check(label+" shift_second",mx::grad(mx::grad(shift))(zero),{0});
  check(label+" shift_finite_difference",mx::divide(mx::subtract(shift(eps),shift(mx::negative(eps))),A(2.f/256)),{total},1e-4);
  if(coordinate_fd) {
    std::vector<double> fd_source(source.size()),fd_updates(updates.size());
    auto loss=[&](const std::vector<float>& s,const std::vector<float>& u) {
      return values(mx::sum(mx::multiply(op(A(s.data(),source_shape,mx::float32),
          A(u.data(),update_shape,mx::float32)),c)))[0];
    };
    for(int i=0;i<source.size();++i) {
      auto hi=source,lo=source;
      hi[i]+=1.f/256;lo[i]-=1.f/256;
      fd_source[i]=(loss(hi,updates)-loss(lo,updates))/(2.f/256);
    }
    for(int i=0;i<updates.size();++i) {
      auto hi=updates,lo=updates;
      hi[i]+=1.f/256;lo[i]-=1.f/256;
      fd_updates[i]=(loss(source,hi)-loss(source,lo))/(2.f/256);
    }
    check(label+" source_coordinate_fd",grads[0],fd_source,1e-4);
    check(label+" updates_coordinate_fd",grads[1],fd_updates,1e-4);
    std::cout<<"EVIDENCE "<<label<<" update_gradient=";
    for(auto v:values(grads[1])) std::cout<<v<<',';
    std::cout<<" coordinate_fd=";
    for(auto v:fd_updates) std::cout<<v<<',';
    std::cout<<'\n';
  }
  auto [out2,grad2]=mx::vjp(fn,{src,upd},{mx::multiply(c,A(-2.f))});
  for(auto& d:ds)d*=-2;
  for(auto& d:du)d*=-2;
  check(label+" scaled_source",grad2[0],ds);
  check(label+" scaled_updates",grad2[1],du);
  } catch(const std::exception& e) {
    ++checks; ++failures;
    std::cout<<"FAIL "<<label<<" exception="<<e.what()<<'\n';
  }
  failed_cases+=failures!=before;
}

Op scatter_op(bool is_max,std::vector<std::vector<int>> index_values,std::vector<int> axes) {
  std::vector<A> idx;
  for(auto v:index_values) idx.emplace_back(v.data(),mx::Shape{int(v.size())},mx::int32);
  return [=](const A& a,const A& b){return is_max?mx::scatter_max(a,idx,b,axes):mx::scatter_min(a,idx,b,axes);};
}
Op slice_op(bool is_max,mx::Shape start,mx::Shape end,mx::Shape stride) {
  return [=](const A& a,const A& b){return is_max?mx::slice_update_max(a,b,start,end,stride):mx::slice_update_min(a,b,start,end,stride);};
}

int main(int argc,char** argv) {
  mx::set_default_device(mx::Device::cpu);
  std::cout<<std::setprecision(10);
  for(bool is_max:{true,false}) {
    std::string m=is_max?"max":"min";
    auto sc=[&](std::vector<int> idx){return scatter_op(is_max,{idx},{0});};
    if(argc==2 && std::string(argv[1])=="--strict-blocks") {
      float sign=is_max?1.f:-1.f;
      test(m+"_strict_overlapping_blocks",is_max,sc({0,1}),{0,0,0,0},{4},
           {3*sign,4*sign,5*sign,6*sign},{2,2},{0,1,1,2},true);
      continue;
    }
    test(m+"_strict_case_a",is_max,sc({0,2}),{5,1,8},{3},{-2,-3},{2,1},{0,2});
    test(m+"_strict_case_b",is_max,sc({0,2}),{-5,1,-8},{3},{2,3},{2,1},{0,2});
    test(m+"_unique_ties",is_max,sc({0,2}),{1,5,0},{3},{1,0},{2,1},{0,2});
    test(m+"_duplicate_source_ties",is_max,sc({0,0,1,2}),{1,5,0},{3},{1,1,4,0},{4,1},{0,0,1,2});
    test(m+"_duplicate_update_ties",is_max,sc({0,0,2,2}),{0,3,2},{3},{5,5,-1,-1},{4,1},{0,0,2,2});
    test(m+"_unequal_duplicates",is_max,sc({0,0,0,2}),{0,3,2},{3},{5,4,-1,-3},{4,1},{0,0,0,2});
    test(m+"_zero_ties",is_max,sc({0,0,2}),{0,0,0},{3},{0,-0.f,0},{3,1},{0,0,2});
    test(m+"_empty",is_max,sc({}),{1,2,3},{3},{},{0,1},{});
    test(m+"_row_blocks",is_max,sc({0,0,2}),{1,2,3,4,5,6},{3,2},{1,2,1,7,5,6},{3,1,2},{0,1,0,1,4,5});
    test(m+"_column_blocks",is_max,scatter_op(is_max,{{0,0,2}},{1}),{1,2,3,4,5,6},{2,3},{1,4,7,4,3,6},{3,2,1},{0,3,0,3,2,5});
    test(m+"_two_axes",is_max,scatter_op(is_max,{{0,0,1},{1,1,2}},{0,1}),{1,2,3,4,5,6},{2,3},{2,2,6},{3,1,1},{1,1,5});
    test(m+"_overlapping_blocks",is_max,sc({0,1}),{1,2,3,4},{4},{1,2,2,3},{2,2},{0,1,1,2});
    float sign=is_max?1.f:-1.f;
    test(m+"_strict_overlapping_blocks",is_max,sc({0,1}),{0,0,0,0},{4},
         {3*sign,4*sign,5*sign,6*sign},{2,2},{0,1,1,2},true);
    test(m+"_slice_partial",is_max,slice_op(is_max,{1},{3},{1}),{1,2,3,4},{4},{2,3},{2},{1,2});
    test(m+"_slice_strided",is_max,slice_op(is_max,{0},{6},{2}),{1,2,3,4,5,6},{6},{1,7,5},{3},{0,2,4});
    test(m+"_slice_matrix",is_max,slice_op(is_max,{0,1},{2,3},{1,1}),{1,2,3,4,5,6},{2,3},{2,9,5,-3},{2,2},{1,2,4,5});
    for(int n:{1,2,4}) {
      ++cases; int before=failures;
      Op op=sc(std::vector<int>(n,0));
      auto identity=[=](const A& t){return mx::sum(op(mx::broadcast_to(t,{1}),mx::broadcast_to(t,{n,1})));};
      auto square=[=](const A& t){auto y=identity(t); return mx::divide(mx::multiply(y,y),A(2.f));};
      A x(2.f);
      std::string label=m+"_shared_"+std::to_string(n);
      check(label+" forward",identity(x),{2});
      check(label+" gradient",mx::grad(identity)(x),{1});
      check(label+" second",mx::grad(mx::grad(identity))(x),{0});
      check(label+" square_gradient",mx::grad(square)(x),{2});
      check(label+" square_second",mx::grad(mx::grad(square))(x),{1});
      failed_cases+=failures!=before;
    }
    ++cases; int before=failures;
    auto control=[=](const A& t){return is_max?mx::maximum(t,t):mx::minimum(t,t);};
    check(m+"_ordinary_control",mx::grad(control)(A(2.f)),{1});
    failed_cases+=failures!=before;
  }
  std::cout<<"SUMMARY cases="<<cases<<" passed_cases="<<cases-failed_cases
           <<" failed_cases="<<failed_cases<<" checks="<<checks<<" failures="<<failures<<'\n';
  return failures?1:0;
}
