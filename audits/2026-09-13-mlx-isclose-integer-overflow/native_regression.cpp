#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "mlx/mlx.h"
namespace mx=mlx::core;
using json=nlohmann::json;
mx::Dtype dtype(const std::string&s){static std::map<std::string,mx::Dtype> m={{"bool",mx::bool_},{"int8",mx::int8},{"uint8",mx::uint8},{"int16",mx::int16},{"uint16",mx::uint16},{"int32",mx::int32},{"uint32",mx::uint32},{"float16",mx::float16},{"bfloat16",mx::bfloat16},{"float32",mx::float32},{"float64",mx::float64}};return m.at(s);}
mx::array input(const json& vals,const std::string& type){
 if(type=="bool" || type.find("int")!=std::string::npos){std::vector<int64_t> v;for(auto&x:vals)v.push_back(x.get<int64_t>());return mx::array(v.begin(),{int(v.size())},dtype(type));}
 std::vector<double> v;for(auto&x:vals){if(x.is_string())v.push_back(std::stod(x.get<std::string>()));else v.push_back(x.get<double>());}return mx::array(v.begin(),{int(v.size())},dtype(type));
}
int main(int argc,char**argv){
 mx::set_default_device(mx::Device::cpu);std::ifstream f(argv[1]);json cases;f>>cases;int failed=0,total=0;size_t comparisons=0,mismatches=0;json counts=json::object();
 for(const auto&c:cases){json row={{"name",c["name"]},{"group",c["group"]}};size_t errors=0;try{
 auto a=input(c["a"],c["da"]),b=input(c["b"],c["db"]);int na=a.size(),nb=b.size();
 a=mx::reshape(a,{na,1});b=mx::reshape(b,{1,nb});
 if(c["transpose"].get<bool>()){auto ap=mx::stack({mx::reshape(a,{na}),mx::reshape(a,{na})},1);auto bp=mx::stack({mx::reshape(b,{nb}),mx::reshape(b,{nb})},1);mx::eval(ap,bp);a=mx::slice(ap,{0,0},{na,1});b=mx::transpose(mx::slice(bp,{0,0},{nb,1}));}
 auto y=mx::isclose(a,b,c["rtol"],c["atol"],c["equal_nan"]);auto all=mx::allclose(a,b,c["rtol"],c["atol"],c["equal_nan"]);auto flat=mx::reshape(y,{-1});mx::eval(flat,all);
 bool expected_all=true;uint64_t bits_hash=1469598103934665603ull;json examples=json::array();
 for(size_t i=0;i<c["expected"].size();++i){bool ex=c["expected"][i];bool act=flat.data<bool>()[i];expected_all &= ex;bits_hash=(bits_hash^uint64_t(act))*1099511628211ull;if(ex!=act){++errors;if(examples.size()<3)examples.push_back({{"index",i},{"expected",ex},{"actual",act}});}}
 errors += all.item<bool>()!=expected_all;errors += y.shape()!=mx::Shape{na,nb};errors += y.dtype()!=mx::bool_;
 comparisons += c["expected"].size();row["elements"]=c["expected"].size();row["allclose"]=all.item<bool>();row["expected_allclose"]=expected_all;row["actual_hash"]=bits_hash;row["examples"]=examples;
 }catch(const std::exception&e){row["exception"]=e.what();errors++;}
 row["mismatches"]=errors;row["pass"]=errors==0;mismatches+=errors;failed+=errors>0;++total;std::cout<<row.dump()<<"\n";
 }
 std::cout<<json({{"summary",true},{"scenarios",total},{"failed_scenarios",failed},{"element_comparisons",comparisons},{"mismatches",mismatches}}).dump()<<"\n";
 return failed?1:0;
}
