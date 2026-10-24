#pragma once

#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/storage_owner.h"
#include "hyle/graph/executable.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace static_fp8_reference_test {
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace m=hyle::mem;
using Type=c::DataType;
using F16=hyle::simd::float16_t;
using BF16=hyle::simd::bfloat16_t;
constexpr c::RoutePolicyId kPackedPolicy{0x48594c4553465038ULL,1,2};

template<class T> T take(hyle::Result<T> r,std::string_view where) {
  if(!r)throw std::runtime_error(std::string(where)+": "+std::string(r.error().message()));
  return std::move(*r);
}
inline void check(hyle::Status s,std::string_view where) {
  if(!s.ok())throw std::runtime_error(std::string(where)+": "+std::string(s.message()));
}
inline c::TensorDesc dense(Type type,std::initializer_list<std::int64_t> dims) {
  return take(c::TensorDesc::dense(type,{dims.begin(),dims.size()}),"dense");
}
template<class T> std::vector<std::byte> bytes(const std::vector<T> &v) {
  const auto b=std::as_bytes(std::span(v));return {b.begin(),b.end()};
}
inline double round(Type type,double value) {
  if(type==Type::kFloat16)return half_test::decode(half_test::encode<F16>(value));
  if(type==Type::kBFloat16)return half_test::decode(half_test::encode<BF16>(value));
  return static_cast<float>(value);
}
inline std::vector<std::byte> floating(Type type,std::span<const double> values) {
  if(type==Type::kFloat32) {std::vector<float> v;for(auto x:values)v.push_back(static_cast<float>(x));return bytes(v);}
  std::vector<std::uint16_t> v;
  for(auto x:values)v.push_back(type==Type::kFloat16 ? half_test::encode<F16>(x).bits : half_test::encode<BF16>(x).bits);
  return bytes(v);
}
// Independent bit writer. The mathematical oracle below consumes separate
// scalar codes/zero values and never invokes a product unpack/round helper.
// Independent scalar FP8 table and nearest-neighbor RNE; never the product codec.
inline float e4m3(std::uint8_t code) {
  const auto magnitude=unsigned(code&127u), exponent=magnitude>>3, fraction=magnitude&7u;
  if(magnitude==127u)return std::numeric_limits<float>::quiet_NaN();
  const auto value=exponent==0 ? std::ldexp(float(fraction),-9)
      : std::ldexp(1.F+float(fraction)/8.F,int(exponent)-7);
  return code&128u ? -value : value;
}
inline std::uint8_t encode_e4m3(float value) {
  const auto sign=std::signbit(value)?128u:0u;
  if(std::isnan(value))return static_cast<std::uint8_t>(sign|127u);
  const auto magnitude=std::abs(value);
  if(magnitude>=448.F)return static_cast<std::uint8_t>(sign|126u);
  unsigned below=0;
  while(below<126u&&e4m3(static_cast<std::uint8_t>(below+1))<=magnitude)++below;
  if(below<126u) {
    const double low=double(magnitude)-double(e4m3(static_cast<std::uint8_t>(below)));
    const double high=double(e4m3(static_cast<std::uint8_t>(below+1)))-double(magnitude);
    if(high<low||(high==low&&(below&1u)))++below;
  }
  return static_cast<std::uint8_t>(sign|below);
}
inline float multiply(float a,float b) {volatile float x=a*b;return x;}
inline float reciprocal(float a) {volatile float x=1.F/a;return x;}
struct Profile final {Type input,output;bool vector_scales=false;};
inline constexpr std::array profiles{Profile{Type::kBFloat16,Type::kBFloat16},
    Profile{Type::kFloat16,Type::kFloat16,true},Profile{Type::kFloat32,Type::kFloat32},
    Profile{Type::kBFloat16,Type::kFloat32,true}};
struct Data final {std::vector<double> a;std::vector<std::uint8_t> w;float sa=1.F,sw=1.F;};
struct Case final {
  static constexpr std::int64_t rows=2,columns=3,depth=32;
  Profile profile;
  std::array<c::TensorDesc,4> descriptions;
  c::TensorDesc output,encoded;
  explicit Case(Profile p=profiles[0]):profile(p) {
    const auto scalar=p.vector_scales?dense(Type::kFloat32,{1}):dense(Type::kFloat32,{});
    descriptions={dense(p.input,{rows,depth}),dense(Type::kFloat8E4M3FN,{columns,depth}),scalar,scalar};
    output=dense(p.output,{rows,columns});encoded=dense(Type::kFloat8E4M3FN,{rows,depth});
  }
  Data data(unsigned phase=0) const {
    Data d;d.a.resize(std::size_t(rows*depth));d.w.resize(std::size_t(columns*depth));
    constexpr std::array<float,4> sa{1.F,1.00390625F,.5F,2.F},sw{1.F,1.00390625F,1.125F,-2.F};
    d.sa=sa[phase%4];d.sw=sw[phase%4];
    constexpr std::array values{1.0625,1.1875,-1024.,0.0009765625,-0.0009765625,-0.0,448.,449.};
    for(std::int64_t r=0;r<rows;++r)for(std::int64_t k=0;k<8;++k)
      d.a[std::size_t(r*depth+k)]=round(profile.input,values[std::size_t((k+r+phase)%8)]);
    for(std::int64_t n=0;n<columns;++n)d.w[std::size_t(n*depth+n)]=n==1?0xb8u:0x38u;
    return d;
  }
  std::array<std::vector<std::byte>,4> payloads(const Data &d) const {
    return {floating(profile.input,d.a),bytes(d.w),bytes(std::vector<float>{d.sa}),bytes(std::vector<float>{d.sw})};
  }
  std::vector<std::uint8_t> codes(const Data &d) const {
    std::vector<std::uint8_t> q(d.a.size());const auto inverse=reciprocal(d.sa);
    for(std::size_t i=0;i<q.size();++i)q[i]=encode_e4m3(multiply(static_cast<float>(round(profile.input,d.a[i])),inverse));
    return q;
  }
  std::vector<double> oracle(const Data &d) const {
    const auto q=codes(d);std::vector<double> y(std::size_t(rows*columns));
    for(std::int64_t r=0;r<rows;++r)for(std::int64_t n=0;n<columns;++n) {
      float sum=0;
      for(std::int64_t k=0;k<depth;++k) {volatile float term=e4m3(q[std::size_t(r*depth+k)])*e4m3(d.w[std::size_t(n*depth+k)]);volatile float next=sum+term;sum=next;}
      y[std::size_t(r*columns+n)]=round(profile.output,multiply(multiply(sum,d.sa),d.sw));
    }
    return y;
  }
};
struct Logical final {
  std::array<g::ValueId,4> ports{};g::ValueId output{},encoded{};g::Graph graph;
  Logical(const Case &f,bool constant_weight=true,bool escape=false) {
    auto b=take(g::GraphBuilder::create(),"static FP8 graph");
    for(std::size_t i=0;i<4;++i)ports[i]=take(constant_weight&&i==1?b.constant(f.descriptions[i]):b.input(f.descriptions[i]),"static FP8 original port");
    const auto a=take(b.add(c::Cast{Type::kFloat32},ports[0]),"original activation to F32");
    const auto one=take(b.add(c::Fill{.output_shape=take(c::TensorShape::create(f.descriptions[2].dimensions()),"SA scalar shape"),.output_type=Type::kFloat32,.value=c::ScalarValue(1.F)}),"literal one matching SA");
    const auto inverse=take(b.add(c::Binary{c::BinaryKind::kDivide},one,ports[2]),"current F32 reciprocal");
    const auto normalized=take(b.add(c::Binary{c::BinaryKind::kMultiply},a,inverse),"F32 activation normalize");
    encoded=take(b.add(c::Cast{Type::kFloat8E4M3FN},normalized),"SATFINITE nearest-even FP8");
    const auto q=take(b.add(c::Cast{Type::kFloat32},encoded),"FP8 code values to F32");
    auto w=take(b.add(c::Cast{Type::kFloat32},ports[1]),"original FP8 weight values");
    w=take(b.add(take(c::View::permute(std::array<std::int32_t,2>{1,0}),"weight permutation"),w),"exact weight transpose");
    auto y=take(b.add(c::MatMul{.accumulation_type=Type::kFloat32},q,w),"F32 code-domain dot");
    y=take(b.add(c::Binary{c::BinaryKind::kMultiply},y,ports[2]),"F32 dot times SA");
    y=take(b.add(c::Binary{c::BinaryKind::kMultiply},y,ports[3]),"F32 separate SW product");
    output=take(b.add(c::Cast{f.profile.output},y),"final declared output Cast");
    if(escape)check(b.publish(encoded),"publish real quantized activation");
    check(b.publish(output),"publish final output");graph=take(std::move(b).finish(),"finish static FP8 graph");
  }
};
inline void selected(const g::CompileReport &report,bool expected) {
  std::size_t count=0;for(std::size_t i=0;i<report.step_count();++i)if(report.step(i).value().policy==kPackedPolicy)++count;
  EXPECT_EQ(count,expected?1u:0u);
}
template<class Read>void expect_result(const g::GraphOutput &result,Type type,std::span<const double> expected,Read read) {
  const auto view=take(result.tensor(),"static FP8 result view");std::vector<std::byte> raw(view.storage().size());check(read(view.storage(),raw),"read static result");
  ASSERT_EQ(raw.size(),expected.size()*c::data_type_size(type));
  for(std::size_t i=0;i<expected.size();++i) {
    double actual;
    if(type==Type::kFloat32) {float v;std::memcpy(&v,raw.data()+4*i,4);actual=v;}
    else {std::uint16_t v;std::memcpy(&v,raw.data()+2*i,2);actual=type==Type::kFloat16?half_test::decode_bits<F16>(v):half_test::decode_bits<BF16>(v);}
    if(std::isnan(expected[i])) {EXPECT_TRUE(std::isnan(actual))<<i;}
    else {EXPECT_EQ(actual,expected[i])<<i;}
  }
}

template<class Compile,class Write,class Read,class Prepare>
void run_lifetimes(m::MemoryResourceRef resource,std::span<const g::QueueBinding> queues,
    Compile compile,Write write,Read read,Prepare prepare,c::StorageOwner owner={}) {
  unsigned submits=0;
  for(const auto profile:profiles) {
    SCOPED_TRACE(::testing::Message()<<"A="<<int(profile.input)<<" Y="<<int(profile.output));
    Case f(profile);Logical logical(f);const auto original=f.data();const auto payload=f.payloads(original);
    auto bank=take(m::Buffer::create(resource,payload[1].size(),256),"original static FP8 bank");check(write(bank.view(),payload[1]),"upload original bank");
    auto ready=owner?take(owner.adopt_buffer(std::move(bank)),"owned bank"):take(c::ReadLease::adopt(std::move(bank)),"ready bank");
    std::array constants{g::ConstantBinding{logical.ports[1],take(ready.tensor(f.descriptions[1]),"typed original FP8 bank"),g::ConstantPolicy::kMaterialize,owner,owner?ready:c::ReadLease{}}};
    auto compiled=compile(logical.graph,g::CompileOptions{.constants=constants});ASSERT_TRUE(compiled)<<compiled.error().message();selected(compiled->report,true);
    EXPECT_EQ(compiled->executable.input_count(),3u);
    const auto ownership=take(compiled->executable.output_ownership(),"output ownership");const bool caller=ownership==g::OutputOwnership::kCaller;
    if(!caller) {ASSERT_EQ(ownership,g::OutputOwnership::kInvocation);}
    std::array runs{take(compiled->executable.create_invocation({.resources=g::ResourceOptions{}}),"slot0"),take(compiled->executable.create_invocation({.resources=g::ResourceOptions{}}),"slot1")};
    constants={};ready={};compiled=g::CompiledGraph{};logical.graph={};
    const auto output_bytes=f.output.required_bytes().value();c::LeasePool pool;
    if(caller)pool=take(c::LeasePool::create(resource,2*((output_bytes+255)/256*256),{.max_blocks=2}),"independent outputs");
    constexpr std::array<std::size_t,3> dynamic{0,2,3};std::array<std::array<m::Buffer,3>,2> inputs;
    for(std::size_t slot=0;slot<2;++slot) {
      for(std::size_t i=0;i<3;++i) {const auto port=dynamic[i];inputs[slot][i]=take(m::Buffer::create(resource,payload[port].size(),256),"dynamic original bytes");check(write(inputs[slot][i].view(),payload[port]),"upload initial dynamic");check(runs[slot].bind(logical.ports[port],take(c::ConstTensorView::create(f.descriptions[port],inputs[slot][i].view()),"dynamic view")),"bind once");}
      if(caller)check(runs[slot].bind(logical.output,f.output,take(pool.allocate(output_bytes),"output writer")),"bind output");
      check(prepare(runs[slot],queues),"prepare static replay");
    }
    std::array<g::GraphOutput,2> retained;std::array<std::vector<double>,2> old;
    for(unsigned repeat=0;repeat<2;++repeat)for(std::size_t slot=0;slot<2;++slot) {
      if(repeat) {expect_result(retained[slot],profile.output,old[slot],read);EXPECT_FALSE(runs[slot].submit(queues));retained[slot].reset();}
      const auto data=f.data(repeat*2+static_cast<unsigned>(slot));const auto current=f.payloads(data);
      for(std::size_t i=0;i<3;++i)check(write(inputs[slot][i].view(),current[dynamic[i]]),"update current A/scales");
      auto done=take(runs[slot].submit(queues),"submit static FP8");check(done.wait(),"static completion");++submits;
      retained[slot]=take(runs[slot].result(logical.output),"retain result");old[slot]=f.oracle(data);expect_result(retained[slot],profile.output,old[slot],read);
      if(retained[1-slot].valid())expect_result(retained[1-slot],profile.output,old[1-slot],read);
    }
    runs={};for(std::size_t slot=0;slot<2;++slot)expect_result(retained[slot],profile.output,old[slot],read);retained={};
    if(caller) {EXPECT_EQ(pool.stats().live_blocks,0u);}
  }
  ::testing::Test::RecordProperty("actual_submissions",submits);
}
template<class Compile,class Write,class Read,class Prepare>
void run_runtime_domain(m::MemoryResourceRef resource,std::span<const g::QueueBinding> queues,
    Compile compile,Write write,Read read,Prepare prepare) {
  Case f;Logical logical(f,false);auto compiled=compile(logical.graph,g::CompileOptions{});ASSERT_TRUE(compiled)<<compiled.error().message();selected(compiled->report,true);
  auto run=take(compiled->executable.create_invocation({.resources=g::ResourceOptions{}}),"runtime slot");
  const auto ownership=take(compiled->executable.output_ownership(),"runtime ownership");const bool caller=ownership==g::OutputOwnership::kCaller;
  if(!caller) {ASSERT_EQ(ownership,g::OutputOwnership::kInvocation);}
  const auto initial=f.payloads(f.data());std::array<m::Buffer,4> buffers;
  for(std::size_t i=0;i<4;++i) {buffers[i]=take(m::Buffer::create(resource,initial[i].size(),256),"runtime payload");check(write(buffers[i].view(),initial[i]),"upload runtime bytes");check(run.bind(logical.ports[i],take(c::ConstTensorView::create(f.descriptions[i],buffers[i].view()),"runtime typed port")),"bind runtime once");}
  const auto count=f.output.required_bytes().value();c::LeasePool pool;
  if(caller) {pool=take(c::LeasePool::create(resource,(count+255)/256*256,{.max_blocks=1}),"runtime output");check(run.bind(logical.output,f.output,take(pool.allocate(count),"runtime writer")),"bind runtime output");}
  check(prepare(run,queues),"prepare runtime");compiled=g::CompiledGraph{};logical.graph={};unsigned successes=0;
  const auto success=[&](const Data &data) {const auto payload=f.payloads(data);for(std::size_t i=0;i<4;++i)check(write(buffers[i].view(),payload[i]),"current IEEE input");auto done=take(run.submit(queues),"IEEE submit");check(done.wait(),"IEEE completion");auto result=take(run.result(logical.output),"IEEE result");expect_result(result,f.profile.output,f.oracle(data),read);++successes;};
  auto data=f.data();success(data);
  for(const auto sa:std::array<float,8>{0.F,-0.F,-1.F,std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN(),std::bit_cast<float>(1u),std::numeric_limits<float>::min()}) {data=f.data();data.sa=sa;success(data);}
  for(const auto sw:std::array<float,7>{0.F,-0.F,-1.F,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN(),std::bit_cast<float>(1u),std::bit_cast<float>(0x00080000u)}) {data=f.data();data.sw=sw;success(data);}
  // A zero dot must remain finite when SA*SW would overflow before the dot.
  data=f.data();std::fill(data.w.begin(),data.w.end(),0u);data.sa=2.F;data.sw=std::ldexp(1.F,127);success(data);
  // This rejects an operand-BF16 QDQ substitute: codes1 with these scales
  // give BF16 0x3f81 after the code dot, versus 0x3f80 if operands narrow first.
  data=f.data();std::fill(data.a.begin(),data.a.end(),0.);std::fill(data.w.begin(),data.w.end(),0u);
  data.a[0]=1.;data.w[0]=0x38u;data.sa=data.sw=1.00390625F;
  EXPECT_EQ(half_test::encode<BF16>(f.oracle(data)[0]).bits,0x3f81u);success(data);
  for(const auto value:{std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {data=f.data();data.a[0]=value;success(data);}
  data=f.data();data.w[0]=0x7fu;success(data);
  ::testing::Test::RecordProperty("successful_submissions",successes);
  ::testing::Test::RecordProperty("failed_completions",0);
}
template<class Compile,class Write,class Read,class Prepare>
void run_escape(m::MemoryResourceRef resource,std::span<const g::QueueBinding> queues,
    Compile compile,Write write,Read read,Prepare prepare) {
  Case f;Logical logical(f,false,true);auto compiled=compile(logical.graph,g::CompileOptions{});ASSERT_TRUE(compiled)<<compiled.error().message();selected(compiled->report,false);
  auto run=take(compiled->executable.create_invocation({.resources=g::ResourceOptions{}}),"ordinary escape slot");
  const auto ownership=take(compiled->executable.output_ownership(),"escape ownership");const bool caller=ownership==g::OutputOwnership::kCaller;
  if(!caller) {ASSERT_EQ(ownership,g::OutputOwnership::kInvocation);}
  const auto first=f.payloads(f.data());std::array<m::Buffer,4> inputs;std::array<c::LeasePool,2> pools;
  for(std::size_t i=0;i<4;++i) {inputs[i]=take(m::Buffer::create(resource,first[i].size(),256),"escape input");check(write(inputs[i].view(),first[i]),"escape initial bytes");check(run.bind(logical.ports[i],take(c::ConstTensorView::create(f.descriptions[i],inputs[i].view()),"escape input view")),"bind escape once");}
  if(caller)for(std::size_t i=0;i<2;++i) {const auto description=i?f.encoded:f.output;const auto size=description.required_bytes().value();pools[i]=take(c::LeasePool::create(resource,(size+255)/256*256,{.max_blocks=1}),"escape output pool");check(run.bind(i?logical.encoded:logical.output,description,take(pools[i].allocate(size),"escape writer")),"bind escape output");}
  check(prepare(run,queues),"prepare original static graph");compiled=g::CompiledGraph{};logical.graph={};
  std::vector<Data> cases{f.data(),f.data(1)};
  auto special=f.data();std::fill(special.w.begin(),special.w.end(),0u);special.sa=2.F;special.sw=std::ldexp(1.F,127);cases.push_back(special);
  special=f.data();std::fill(special.a.begin(),special.a.end(),0.);std::fill(special.w.begin(),special.w.end(),0u);special.a[0]=1.;special.w[0]=0x38u;special.sa=1.F;special.sw=std::bit_cast<float>(0x00080000u);cases.push_back(special);
  for(const auto sa:{0.F,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {special=f.data();special.sa=sa;cases.push_back(special);}
  for(const auto &data:cases) {
    const auto payload=f.payloads(data);for(std::size_t i=0;i<4;++i)check(write(inputs[i].view(),payload[i]),"update original path");
    auto done=take(run.submit(queues),"ordinary submit");check(done.wait(),"ordinary completion");auto y=take(run.result(logical.output),"ordinary result");expect_result(y,f.profile.output,f.oracle(data),read);
    auto q=take(run.result(logical.encoded),"published code result");const auto tensor=take(q.tensor(),"published code view");std::vector<std::uint8_t> actual(std::size_t(Case::rows*Case::depth));check(read(tensor.storage(),std::as_writable_bytes(std::span(actual))),"read exact activation codes");const auto expected=f.codes(data);for(std::size_t i=0;i<actual.size();++i) {if((expected[i]&127u)==127u) {EXPECT_EQ(actual[i]&127u,127u)<<i;} else {EXPECT_EQ(actual[i],expected[i])<<i;}}
  }
  ::testing::Test::RecordProperty("actual_submissions",static_cast<unsigned>(cases.size()));
}
} // namespace static_fp8_reference_test
