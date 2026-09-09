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
using Fn = std::function<std::vector<A>(const std::vector<A>&)>;
using Op = std::function<A(const A&,const A&)>;
int cases=0, failed_cases=0, checks=0, failures=0;

int size(const mx::Shape& shape) {
  return std::accumulate(shape.begin(),shape.end(),1,std::multiplies<int>());
}
std::vector<double> values(const A& input) {
  auto x=mx::contiguous(input);
  mx::eval(x);
  std::vector<double> result(x.size());
  for(int i=0;i<x.size();++i) result[i]=x.data<float>()[i];
  return result;
}
void check(const std::string& name,const A& actual,const std::vector<double>& wanted,double tolerance=2e-6) {
  ++checks;
  auto got=values(actual);
  bool ok=got.size()==wanted.size();
  for(int i=0;ok && i<got.size();++i)
    ok=std::isfinite(got[i]) && std::abs(got[i]-wanted[i])<=tolerance*(1+std::abs(wanted[i]));
  if(!ok) {
    ++failures;
    std::cout<<"FAIL "<<name<<" got=";
    for(auto x:got) std::cout<<x<<',';
    std::cout<<" expected=";
    for(auto x:wanted) std::cout<<x<<',';
    std::cout<<'\n';
  }
}

void test(std::string name,mx::Shape dst_shape,mx::Shape src_shape,
          mx::Shape mask_shape,std::vector<int> mask_values,
          std::vector<int> source_at_output,std::vector<int> dst_at_output,
          mx::Shape out_shape,std::vector<int> axes={},int out_axis=0,int depth=1) {
  ++cases; int before=failures;
  try {
    std::vector<float> d(size(dst_shape)),s(size(src_shape)),c(size(out_shape));
    for(int i=0;i<d.size();++i)d[i]=float(i+1)/2;
    for(int i=0;i<s.size();++i)s[i]=float(i+3);
    for(int i=0;i<c.size();++i)c[i]=(i%3==1?-1.f:1.f)*float(i+1);
    A dst(d.data(),dst_shape,mx::float32),src(s.data(),src_shape,mx::float32);
    A cot(c.data(),out_shape,mx::float32);
    A mask=mx::astype(A(mask_values.data(),mask_shape,mx::int32),mx::bool_);
    Fn fn=[](const std::vector<A>& v){return std::vector<A>{mx::masked_scatter(v[0],v[2],v[1])};};
    if(!axes.empty()) {
      for(int k=0;k<depth;++k)fn=mx::vmap(fn,axes,{out_axis});
    }
    Op op=[=](const A& a,const A& b){return fn({a,b,mask})[0];};
    std::vector<double> out(c.size()),gd(d.size()),gs(s.size()),counts(s.size());
    for(int i=0;i<c.size();++i) {
      int j=source_at_output[i];
      if(j>=0) {out[i]=s[j];gs[j]+=c[i];counts[j]+=1;}
      else {int k=dst_at_output[i];out[i]=d[k];gd[k]+=c[i];}
    }
    auto vf=[=](const std::vector<A>& v){return std::vector<A>{op(v[0],v[1])};};
    auto [outputs,grads]=mx::vjp(vf,{dst,src},{cot});
    check(name+" forward",outputs[0],out);
    check(name+" destination_vjp",grads[0],gd);
    check(name+" source_vjp",grads[1],gs);
    check(name+" destination_only",mx::grad([&](const A& a){return mx::sum(mx::multiply(op(a,src),cot));})(dst),gd);
    check(name+" source_only",mx::grad([&](const A& b){return mx::sum(mx::multiply(op(dst,b),cot));})(src),gs);
    auto [scaled_out,scaled_grad]=mx::vjp(vf,{dst,src},{mx::multiply(cot,A(-2.f))});
    auto gd2=gd,gs2=gs;
    for(auto& x:gd2)x*=-2;
    for(auto& x:gs2)x*=-2;
    check(name+" upstream_destination",scaled_grad[0],gd2);
    check(name+" upstream_source",scaled_grad[1],gs2);
    std::vector<float> direction(s.size());
    for(int i=0;i<direction.size();++i)direction[i]=float(i+1);
    A tangent(direction.data(),src_shape,mx::float32);
    auto sf=[=](const A& b){return op(dst,b);};
    auto [jout,jtan]=mx::jvp(sf,src,tangent);
    std::vector<double> jt(c.size());
    double inner=0;
    for(int i=0;i<c.size();++i) {
      if(source_at_output[i]>=0)jt[i]=direction[source_at_output[i]];
      inner+=jt[i]*c[i];
    }
    check(name+" jvp",jtan,jt);
    check(name+" adjoint_lhs",mx::sum(mx::multiply(jtan,cot)),{inner});
    check(name+" adjoint_rhs",mx::sum(mx::multiply(grads[1],tangent)),{inner});
    auto loss=[&](const std::vector<float>& a,const std::vector<float>& b){
      return values(mx::sum(mx::multiply(op(A(a.data(),dst_shape,mx::float32),
           A(b.data(),src_shape,mx::float32)),cot)))[0];
    };
    std::vector<double> fds(s.size()),fdd(d.size());
    for(int i=0;i<s.size();++i) {
      auto hi=s,lo=s;hi[i]+=1.f/256;lo[i]-=1.f/256;
      fds[i]=(loss(d,hi)-loss(d,lo))/(2.f/256);
    }
    for(int i=0;i<d.size();++i) {
      auto hi=d,lo=d;hi[i]+=1.f/256;lo[i]-=1.f/256;
      fdd[i]=(loss(hi,s)-loss(lo,s))/(2.f/256);
    }
    check(name+" source_coordinate_fd",grads[1],fds,1e-4);
    check(name+" destination_coordinate_fd",grads[0],fdd,1e-4);
    auto quadratic=[=](const A& b){auto y=op(dst,b);return mx::divide(mx::sum(mx::multiply(y,y)),A(2.f));};
    std::vector<double> square_grad(s.size()),hvp(s.size());
    for(int i=0;i<s.size();++i){square_grad[i]=counts[i]*s[i];hvp[i]=counts[i]*direction[i];}
    auto qgrad=mx::grad(quadratic);
    check(name+" quadratic_gradient",qgrad(src),square_grad);
    check(name+" quadratic_hvp",mx::grad([=](const A& b){return mx::sum(mx::multiply(qgrad(b),tangent));})(src),hvp);
  } catch(const std::exception& e) {
    ++checks;++failures;std::cout<<"FAIL "<<name<<" exception="<<e.what()<<'\n';
  }
  failed_cases+=failures!=before;
}

std::vector<int> identity(int n){std::vector<int> x(n);std::iota(x.begin(),x.end(),0);return x;}
int main() {
  mx::set_default_device(mx::Device::cpu);
  std::cout<<std::setprecision(10);
  const std::vector<int> mask={1,0,1,0,0,1,0,1};
  const std::vector<int> target={0,-1,1,-1,-1,3,-1,4};
  const std::vector<int> axes={0,0,0};
  test("unbatched_padded",{4},{3},{4},{1,0,1,0},{0,-1,1,-1},identity(4),{4});
  test("unbatched_leading_mask",{2,2},{2,2},{2},{1,0},{0,1,-1,-1},identity(4),{2,2});
  test("batch_padded",{2,4},{2,3},{2,4},mask,target,identity(8),{2,4},axes);
  test("batch_unequal_populations",{2,4},{2,4},{2,4},{1,0,0,0,0,1,1,1},
       {0,-1,-1,-1,-1,4,5,6},identity(8),{2,4},axes);
  test("batch_first_empty",{2,4},{2,3},{2,4},{0,0,0,0,0,1,0,1},
       {-1,-1,-1,-1,-1,3,-1,4},identity(8),{2,4},axes);
  test("batch_last_empty",{2,4},{2,3},{2,4},{1,0,1,0,0,0,0,0},
       {0,-1,1,-1,-1,-1,-1,-1},identity(8),{2,4},axes);
  test("batch_all_false",{2,4},{2,1},{2,4},std::vector<int>(8,0),
       std::vector<int>(8,-1),identity(8),{2,4},axes);
  test("batch_all_true_exact",{2,4},{2,4},{2,4},std::vector<int>(8,1),
       identity(8),identity(8),{2,4},axes);
  test("shared_source",{2,4},{3},{2,4},mask,{0,-1,1,-1,-1,0,-1,1},identity(8),{2,4},{0,-1,0});
  test("shared_destination",{4},{2,3},{2,4},mask,target,{0,1,2,3,0,1,2,3},{2,4},{-1,0,0});
  test("shared_mask",{2,4},{2,3},{4},{1,0,1,0},
       {0,-1,1,-1,3,-1,4,-1},identity(8),{2,4},{0,0,-1});
  test("shared_scalar",{2,4},{},{2,4},mask,{0,-1,0,-1,-1,0,-1,0},identity(8),{2,4},{0,-1,0});
  test("batched_scalars",{2,4},{2},{2,4},mask,{0,-1,0,-1,-1,1,-1,1},identity(8),{2,4},axes);
  test("leading_mask_blocks",{2,2,2},{2,2,2},{2,2},{1,0,0,1},
       {0,1,-1,-1,-1,-1,4,5},identity(8),{2,2,2},axes);
  test("nonleading_batch_axis",{4,2},{3,2},{4,2},{1,0,0,1,1,0,0,1},
       {0,-1,-1,1,2,-1,-1,3},identity(8),{4,2},{1,1,1},1);
  test("nested_vmap",{2,2,4},{2,2,3},{2,2,4},
       {1,0,1,0,0,1,0,1,1,1,0,0,0,0,1,1},
       {0,-1,1,-1,-1,3,-1,4,6,7,-1,-1,-1,-1,9,10},identity(16),{2,2,4},axes,0,2);
  test("empty_mask",{2,0},{2,0},{2,0},{},{},{},{2,0},axes);
  test("empty_source_all_false",{2,4},{2,0},{2,4},std::vector<int>(8,0),
       std::vector<int>(8,-1),identity(8),{2,4},axes);
  {
    ++cases;int before=failures;
    A src=mx::reshape(A({10.f,20.f,99.f,30.f,40.f,88.f}),{2,3});
    A m=mx::reshape(A({true,false,true,false,false,true,false,true}),{2,4});
    A cot=mx::reshape(A({1.f,2.f,3.f,4.f,10.f,-20.f,30.f,40.f}),{2,4});
    Fn fn=[](const std::vector<A>& v){return std::vector<A>{mx::masked_scatter(v[0],v[2],v[1])};};
    auto mapped=mx::vmap(fn);
    auto loss=[=](const A& b){return mx::sum(mx::multiply(mapped({mx::zeros({2,4}),b,m})[0],cot));};
    auto g=mx::grad(loss)(src);
    auto next=loss(mx::subtract(src,mx::multiply(g,A(0.125f))));
    check("descent initial_loss",loss(src),{1070});
    check("descent gradient",g,{1,3,0,-20,40,0});
    check("descent next_loss",next,{818.75});
    std::cout<<"EVIDENCE descent initial=1070 next="<<values(next)[0]<<'\n';
    failed_cases+=failures!=before;
  }
  std::cout<<"SUMMARY cases="<<cases<<" passed_cases="<<cases-failed_cases
           <<" failed_cases="<<failed_cases<<" checks="<<checks<<" failures="<<failures<<'\n';
  return failures?1:0;
}
