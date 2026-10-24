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

namespace gptq_reference_test {
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace m=hyle::mem;
using Type=c::DataType;
using F16=hyle::simd::float16_t;
using BF16=hyle::simd::bfloat16_t;
constexpr c::RoutePolicyId kPackedPolicy{0x48594c454354504cULL,1,1};

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
inline void put(std::int32_t &word,unsigned lane,unsigned value) {
  auto b=std::bit_cast<std::uint32_t>(word);
  for(unsigned i=0;i<4;++i) {const auto mask=std::uint32_t{1}<<(lane*4+i);b=(b&~mask)|((value&(1u<<i))?mask:0);}
  word=std::bit_cast<std::int32_t>(b);
}
struct Data final {
  std::vector<double> a;
  std::vector<std::int32_t> q,z,ids;
  std::vector<std::uint16_t> scales;
  std::vector<int> logical_q,logical_z;
};
struct Profile final {
  std::int64_t group,depth,columns;
  Type decode,operand;
  bool cast_before=true;
};
inline constexpr std::array profiles{
    Profile{32,64,16,Type::kFloat32,Type::kFloat32},
    Profile{32,64,16,Type::kBFloat16,Type::kFloat32,false},
    Profile{64,128,8,Type::kFloat16,Type::kBFloat16}};
struct Case final {
  static constexpr std::int64_t rows=2;
  Profile profile;
  std::array<c::TensorDesc,5> descriptions;
  c::TensorDesc decoded,output;
  explicit Case(Profile p=profiles[0]):profile(p) {
    descriptions={dense(p.operand,{rows,p.depth}),dense(Type::kInt32,{p.depth/8,p.columns}),
        dense(Type::kInt32,{p.depth/p.group,p.columns/8}),dense(Type::kFloat16,{p.depth/p.group,p.columns}),
        dense(Type::kInt32,{p.depth})};
    decoded=dense(p.decode,{p.columns,p.depth});output=dense(p.operand,{rows,p.columns});
  }
  Data data(unsigned phase=0) const {
    const auto k=profile.depth,n=profile.columns,groups=k/profile.group;
    Data d;d.q.resize(std::size_t(k*n/8));d.z.resize(std::size_t(groups*n/8));
    d.logical_q.resize(std::size_t(k*n));d.logical_z.resize(std::size_t(groups*n));
    d.scales.resize(std::size_t(groups*n));d.ids.resize(std::size_t(k));d.a.resize(std::size_t(rows*k));
    constexpr std::array values{.10009765625,.0999755859375,-.10009765625,.03125,1.00390625,-0.0,0.0,.5};
    for(std::int64_t g0=0;g0<groups;++g0)for(std::int64_t n0=0;n0<n;++n0) {
      const auto j=std::size_t(g0*n+n0);const unsigned raw=n0==0?15u:n0==1?0u:n0==2?7u:unsigned((n0*7+g0*3)%16);
      d.logical_z[j]=int(raw)+1;put(d.z[std::size_t(g0*(n/8)+n0/8)],unsigned(n0%8),raw);
      d.scales[j]=half_test::encode<F16>(values[std::size_t((g0*3+n0)%std::int64_t(values.size()))]).bits;
    }
    for(std::int64_t k0=0;k0<k;++k0) {
      d.ids[std::size_t(k0)]=static_cast<std::int32_t>((k0/3+k0%7+phase)%groups);
      for(std::int64_t n0=0;n0<n;++n0) {
        const auto q=k0==0&&n0==0?0u:unsigned((k0*5+n0*13)%16);
        d.logical_q[std::size_t(k0*n+n0)]=int(q);put(d.q[std::size_t((k0/8)*n+n0)],unsigned(k0%8),q);
      }
    }
    for(std::int64_t row=0;row<rows;++row) {
      const auto selected=(row*37+std::int64_t(phase)*3)%k;
      d.a[std::size_t(row*k+selected)]=round(profile.operand,phase%2 ? -2.0 : 1.0);
    }
    return d;
  }
  std::array<std::vector<std::byte>,5> payloads(const Data &d) const {
    return {floating(profile.operand,d.a),bytes(d.q),bytes(d.z),bytes(d.scales),bytes(d.ids)};
  }
  std::vector<double> weights(const Data &d,bool omit_source_round=false,bool natural_indices=false) const {
    const auto k=profile.depth,n=profile.columns;std::vector<double> w(std::size_t(n*k));
    for(std::int64_t n0=0;n0<n;++n0)for(std::int64_t k0=0;k0<k;++k0) {
      const auto group=natural_indices ? k0/profile.group : d.ids[std::size_t(k0)];
      const auto j=std::size_t(group*n+n0);
      const auto difference=d.logical_q[std::size_t(k0*n+n0)]-d.logical_z[j];
      const auto product=double(difference)*half_test::decode_bits<F16>(d.scales[j]);
      const auto source=omit_source_round ? product : round(Type::kFloat16,product);
      w[std::size_t(n0*k+k0)]=round(profile.operand,round(profile.decode,source));
    }
    return w;
  }
  std::vector<double> decoded_values(const Data &d) const {
    auto p=profile;p.operand=p.decode;return Case(p).weights(d);
  }
  std::vector<double> oracle(const Data &d) const {
    const auto w=weights(d);const auto k=profile.depth,n=profile.columns;std::vector<double> out(std::size_t(rows*n));
    for(std::int64_t r=0;r<rows;++r)for(std::int64_t n0=0;n0<n;++n0) {
      float sum=0;
      for(std::int64_t k0=0;k0<k;++k0)sum=static_cast<float>(double(sum)+d.a[std::size_t(r*k+k0)]*w[std::size_t(n0*k+k0)]);
      out[std::size_t(r*n+n0)]=round(profile.operand,sum);
    }
    return out;
  }
};
struct Logical final {
  std::array<g::ValueId,5> ports{};
  g::ValueId output{},decoded{};
  g::Graph graph;
  Logical(const Case &f,bool constants=true,bool escape=false) {
    auto b=take(g::GraphBuilder::create(),"GPTQ GraphBuilder");
    for(std::size_t i=0;i<ports.size();++i)ports[i]=take(constants&&i>0&&i<4
        ?b.constant(f.descriptions[i]):b.input(f.descriptions[i]),"original GPTQ port");
    decoded=take(b.add(c::model_ops::artifact_weight_decode(c::model_ops::WeightArtifactFormat::kGptqV1Int4,
        static_cast<std::uint32_t>(f.profile.group),f.profile.decode),ports[1],ports[2],ports[3],ports[4]),"GPTQ decode");
    auto w=decoded;
    if(f.profile.decode!=f.profile.operand&&f.profile.cast_before)w=take(b.add(c::Cast{f.profile.operand},w),"typed weight Cast");
    w=take(b.add(take(c::View::permute(std::array<std::int32_t,2>{1,0}),"exact transpose"),w),"transpose original weight");
    if(f.profile.decode!=f.profile.operand&&!f.profile.cast_before)w=take(b.add(c::Cast{f.profile.operand},w),"typed weight Cast after View");
    output=take(b.add(c::MatMul{.accumulation_type=Type::kFloat32},ports[0],w),"ordinary GPTQ projection");
    if(escape)check(b.publish(decoded),"publish original decoded weight");
    check(b.publish(output),"publish only final GPTQ result");graph=take(std::move(b).finish(),"GPTQ graph");
  }
};
inline void selected(const g::CompileReport &report,bool expected) {
  std::size_t count=0;
  for(std::size_t i=0;i<report.step_count();++i)if(report.step(i).value().policy==kPackedPolicy)++count;
  EXPECT_EQ(count,expected?1u:0u);
}
template<class Read> void expect_result(const g::GraphOutput &result,Type type,std::span<const double> expected,Read read,bool signed_zero=false) {
  auto view=take(result.tensor(),"GPTQ result view");std::vector<std::byte> payload(view.storage().size());check(read(view.storage(),payload),"read GPTQ result");
  ASSERT_EQ(payload.size(),expected.size()*c::data_type_size(type));
  for(std::size_t i=0;i<expected.size();++i) {
    double actual;
    if(type==Type::kFloat32) {float v;std::memcpy(&v,payload.data()+i*4,4);actual=v;}
    else {std::uint16_t v;std::memcpy(&v,payload.data()+i*2,2);actual=type==Type::kFloat16?half_test::decode_bits<F16>(v):half_test::decode_bits<BF16>(v);}
    if(std::isnan(expected[i])) {EXPECT_TRUE(std::isnan(actual))<<i;}
    else {EXPECT_EQ(actual,expected[i])<<i;if(signed_zero&&expected[i]==0) {EXPECT_EQ(std::signbit(actual),std::signbit(expected[i]))<<i;}}
  }
}
template<class Compile,class Write,class Read,class Prepare>
void run_lifetimes(m::MemoryResourceRef resource,std::span<const g::QueueBinding> queues,
    Compile compile,Write write,Read read,Prepare prepare,c::StorageOwner owner={}) {
  unsigned submits=0;
  for(const auto profile:profiles) {
    SCOPED_TRACE(::testing::Message()<<"g="<<profile.group<<" decode="<<int(profile.decode)<<" operand="<<int(profile.operand));
    Case f(profile);Logical logical(f);const auto original=f.data();const auto payload=f.payloads(original);
    ASSERT_NE(f.weights(original),f.weights(original,false,true));
    if(profile.decode==Type::kFloat32) {ASSERT_NE(f.weights(original),f.weights(original,true));}
    EXPECT_EQ(original.logical_q[0]-original.logical_z[0],-16);
    EXPECT_TRUE(std::any_of(original.q.begin(),original.q.end(),[](auto x){return x<0;}));
    std::array<c::ReadLease,3> ready;std::array<g::ConstantBinding,3> constants;
    for(std::size_t i=0;i<ready.size();++i) {
      auto b=take(m::Buffer::create(resource,payload[i+1].size(),256),"original GPTQ bank");check(write(b.view(),payload[i+1]),"upload original GPTQ bank");
      ready[i]=owner?take(owner.adopt_buffer(std::move(b)),"GPTQ owned bank"):take(c::ReadLease::adopt(std::move(b)),"GPTQ bank lease");
      constants[i]={logical.ports[i+1],take(ready[i].tensor(f.descriptions[i+1]),"GPTQ typed bank"),g::ConstantPolicy::kMaterialize,owner,owner?ready[i]:c::ReadLease{}};
    }
    auto compiled=compile(logical.graph,g::CompileOptions{.constants=constants});ASSERT_TRUE(compiled)<<compiled.error().message();selected(compiled->report,true);
    EXPECT_EQ(compiled->executable.input_count(),2u);
    const auto ownership=take(compiled->executable.output_ownership(),"GPTQ ownership");const bool caller=ownership==g::OutputOwnership::kCaller;
    if(!caller) {ASSERT_EQ(ownership,g::OutputOwnership::kInvocation);}
    std::array<g::Invocation,2> runs{take(compiled->executable.create_invocation({.resources=g::ResourceOptions{}}),"GPTQ slot0"),take(compiled->executable.create_invocation({.resources=g::ResourceOptions{}}),"GPTQ slot1")};
    constants={};ready={};compiled=g::CompiledGraph{};logical.graph={};
    c::LeasePool pool;const auto output_bytes=f.output.required_bytes().value();
    if(caller)pool=take(c::LeasePool::create(resource,2*((output_bytes+255)/256*256),{.max_blocks=2}),"GPTQ outputs");
    std::array<std::array<m::Buffer,2>,2> inputs;
    for(std::size_t slot=0;slot<2;++slot) {
      for(std::size_t i=0;i<2;++i) {const auto p=i?4u:0u;inputs[slot][i]=take(m::Buffer::create(resource,payload[p].size(),256),"dynamic A/index");check(write(inputs[slot][i].view(),payload[p]),"initial A/index");check(runs[slot].bind(logical.ports[p],take(c::ConstTensorView::create(f.descriptions[p],inputs[slot][i].view()),"GPTQ dynamic view")),"bind GPTQ once");}
      if(caller)check(runs[slot].bind(logical.output,f.output,take(pool.allocate(output_bytes),"GPTQ output writer")),"bind GPTQ result");
      check(prepare(runs[slot],queues),"prepare GPTQ replay");
    }
    std::array<g::GraphOutput,2> retained;std::array<std::vector<double>,2> old;
    for(unsigned repeat=0;repeat<2;++repeat)for(std::size_t slot=0;slot<2;++slot) {
      if(repeat) {expect_result(retained[slot],profile.operand,old[slot],read);EXPECT_FALSE(runs[slot].submit(queues));retained[slot].reset();}
      const auto data=f.data(repeat*2+static_cast<unsigned>(slot));const auto current=f.payloads(data);
      check(write(inputs[slot][0].view(),current[0]),"update A");check(write(inputs[slot][1].view(),current[4]),"update g_idx");
      auto done=take(runs[slot].submit(queues),"submit GPTQ");check(done.wait(),"GPTQ completion");++submits;
      retained[slot]=take(runs[slot].result(logical.output),"GPTQ output");old[slot]=f.oracle(data);expect_result(retained[slot],profile.operand,old[slot],read);
      if(retained[1-slot].valid())expect_result(retained[1-slot],profile.operand,old[1-slot],read);
    }
    runs={};for(std::size_t slot=0;slot<2;++slot)expect_result(retained[slot],profile.operand,old[slot],read);retained={};
    if(caller) {EXPECT_EQ(pool.stats().live_blocks,0u);}
  }
  ::testing::Test::RecordProperty("actual_submissions",submits);
}
template<class Compile,class Write,class Read,class Prepare>
void run_runtime_domain(m::MemoryResourceRef resource,std::span<const g::QueueBinding> queues,
    Compile compile,Write write,Read read,Prepare prepare) {
  Case f;Logical logical(f,false);auto compiled=compile(logical.graph,g::CompileOptions{});ASSERT_TRUE(compiled)<<compiled.error().message();selected(compiled->report,true);
  auto executable=compiled->executable;auto run=take(executable.create_invocation({.resources=g::ResourceOptions{}}),"GPTQ runtime slot");
  const auto ownership=take(executable.output_ownership(),"GPTQ runtime ownership");const bool caller=ownership==g::OutputOwnership::kCaller;if(!caller) {ASSERT_EQ(ownership,g::OutputOwnership::kInvocation);}
  auto valid=f.data();const auto initial=f.payloads(valid);std::array<m::Buffer,5> buffers;
  for(std::size_t i=0;i<buffers.size();++i)buffers[i]=take(m::Buffer::create(resource,initial[i].size(),256),"GPTQ runtime payload");
  c::LeasePool pool;const auto size=f.output.required_bytes().value();if(caller)pool=take(c::LeasePool::create(resource,(size+255)/256*256,{.max_blocks=1}),"GPTQ runtime outputs");
  const auto upload=[&](const Data &data) {const auto payload=f.payloads(data);for(std::size_t i=0;i<buffers.size();++i)check(write(buffers[i].view(),payload[i]),"update GPTQ runtime bytes");};
  const auto bind=[&] {for(std::size_t i=0;i<buffers.size();++i)check(run.bind(logical.ports[i],take(c::ConstTensorView::create(f.descriptions[i],buffers[i].view()),"GPTQ current view")),"bind GPTQ runtime port");if(caller)check(run.bind(logical.output,f.output,take(pool.allocate(size),"GPTQ runtime writer")),"bind runtime output");check(prepare(run,queues),"prepare runtime replay");};
  upload(valid);bind();compiled=g::CompiledGraph{};logical.graph={};unsigned successes=0,failures=0;
  const auto success=[&](const Data &data) {upload(data);auto done=take(run.submit(queues),"GPTQ valid submit");check(done.wait(),"GPTQ valid completion");auto result=take(run.result(logical.output),"GPTQ runtime result");expect_result(result,f.profile.operand,f.oracle(data),read);++successes;};
  success(valid);
  for(const auto position:{std::size_t{0},std::size_t(f.profile.depth-1)})for(const auto id:{-1,std::int32_t(f.profile.depth/f.profile.group)}) {
    auto data=valid;data.ids[position]=id;upload(data);
    {auto done=take(run.submit(queues),"GPTQ invalid index accepted work");EXPECT_FALSE(done.wait().ok());EXPECT_FALSE(run.result(logical.output));}
    EXPECT_FALSE(run.submit(queues));++failures;run={};run=take(executable.create_invocation({.resources=g::ResourceOptions{}}),"fresh GPTQ recovery slot");upload(valid);bind();success(valid);
  }
  for(const auto scale:std::array<std::uint16_t,5>{0x7e01,0x7c00,0xfc00,0,0x8000}) {auto data=valid;data.scales[0]=scale;success(data);}
  ::testing::Test::RecordProperty("successful_submissions",successes);::testing::Test::RecordProperty("failed_completions",failures);
}
template<class Compile,class Write,class Read,class Prepare>
void run_escape(m::MemoryResourceRef resource,std::span<const g::QueueBinding> queues,
    Compile compile,Write write,Read read,Prepare prepare) {
  Case f;Logical logical(f,false,true);auto compiled=compile(logical.graph,g::CompileOptions{});ASSERT_TRUE(compiled)<<compiled.error().message();selected(compiled->report,false);
  auto run=take(compiled->executable.create_invocation({.resources=g::ResourceOptions{}}),"escaped GPTQ slot");const auto ownership=take(compiled->executable.output_ownership(),"escaped GPTQ ownership");const bool caller=ownership==g::OutputOwnership::kCaller;if(!caller) {ASSERT_EQ(ownership,g::OutputOwnership::kInvocation);}
  const auto initial=f.payloads(f.data());std::array<m::Buffer,5> buffers;std::array<c::LeasePool,2> outputs;
  for(std::size_t i=0;i<buffers.size();++i) {buffers[i]=take(m::Buffer::create(resource,initial[i].size(),256),"escaped original input");check(write(buffers[i].view(),initial[i]),"upload escaped input");check(run.bind(logical.ports[i],take(c::ConstTensorView::create(f.descriptions[i],buffers[i].view()),"escaped input view")),"bind escaped input once");}
  if(caller)for(std::size_t i=0;i<2;++i) {const auto desc=i?f.decoded:f.output;const auto bytes=desc.required_bytes().value();outputs[i]=take(c::LeasePool::create(resource,bytes,{.max_blocks=1}),"escaped output pool");check(run.bind(i?logical.decoded:logical.output,desc,take(outputs[i].allocate(bytes),"escaped output writer")),"bind escaped output");}
  check(prepare(run,queues),"prepare original GPTQ path");compiled=g::CompiledGraph{};logical.graph={};
  for(unsigned phase=0;phase<2;++phase) {const auto data=f.data(phase);const auto payload=f.payloads(data);for(std::size_t i=0;i<buffers.size();++i)check(write(buffers[i].view(),payload[i]),"update original escaped data");auto done=take(run.submit(queues),"escaped submit");check(done.wait(),"escaped completion");auto y=take(run.result(logical.output),"escaped Y");auto d=take(run.result(logical.decoded),"escaped decoded W");expect_result(y,f.profile.operand,f.oracle(data),read);expect_result(d,f.profile.decode,f.decoded_values(data),read,true);}
  ::testing::Test::RecordProperty("actual_submissions",2);
}
} // namespace gptq_reference_test
