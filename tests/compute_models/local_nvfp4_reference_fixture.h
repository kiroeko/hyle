#pragma once

#include "nvfp4_reference_fixture.h"

namespace local_nvfp4_reference_test {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
namespace storage = nvfp4_reference_test;
using Type = c::DataType;
using Rounding = c::model_ops::Fp4InputRounding;
using storage::take;
using storage::check;
using storage::dense;
using storage::bytes;

// Independent enumerated scalar oracle. Only the old fixture's storage,
// BF16 bit conversion and lease helpers are reused; no product quantizer.
inline float f32(double value) { volatile float rounded = static_cast<float>(value); return rounded; }
inline float divide(float x, float y) { volatile float a=x,b=y; volatile float v=a/b; return v; }
inline float multiply(float x, float y) { return f32(double(x)*double(y)); }
inline float bf(float value) { return storage::wide(storage::bf16(value)); }
inline constexpr std::array<float,8> magnitudes{0.F,.5F,1.F,1.5F,2.F,3.F,4.F,6.F};
inline float e4(std::uint8_t code) {
  const auto v=unsigned(code&127u), exponent=v>>3, fraction=v&7u;
  if(v==127u) return std::numeric_limits<float>::quiet_NaN();
  const auto value=exponent ? std::ldexp(1.F+float(fraction)/8.F,int(exponent)-7)
                            : std::ldexp(float(fraction),-9);
  return code&128u ? -value : value;
}
inline float scale_round(float value) {
  const double bounded=std::min(double(value),448.);
  unsigned chosen=0;double distance=bounded;
  for(unsigned code=1;code<127;++code) {
    const auto next=std::abs(bounded-double(e4(static_cast<std::uint8_t>(code))));
    if(next<distance || (next==distance && !(code&1u))) {chosen=code;distance=next;}
  }
  return e4(static_cast<std::uint8_t>(chosen));
}
inline float nearest(float value) {
  unsigned chosen=0;double distance=std::abs(double(value));
  for(unsigned code=1;code<magnitudes.size();++code) {
    const auto next=std::abs(std::abs(double(value))-double(magnitudes[code]));
    if(next<distance || (next==distance && !(code&1u))) {chosen=code;distance=next;}
  }
  return value<0.F ? -magnitudes[chosen] : magnitudes[chosen];
}
inline float activation(float value,float maximum,float global,Rounding rounding) {
  const auto base=bf(divide(maximum,6.F));
  auto local=scale_round(multiply(global,base));
  if(local==0.F) local=.125F;
  const auto effective=divide(local,global);
  auto scaled=std::clamp(divide(value,effective),-6.F,6.F);
  if(rounding==Rounding::kBFloat16) scaled=bf(scaled);
  return bf(multiply(nearest(scaled),effective));
}
inline float weight(unsigned code,std::uint8_t scale,float global) {
  const auto value=code&8u ? -magnitudes[code&7u] : magnitudes[code&7u];
  return bf(multiply(value,divide(e4(scale),global)));
}
inline void oracle_checks() {
  const auto ig=std::bit_cast<float>(std::uint32_t{0x3f8020c5});
  EXPECT_EQ(activation(1.25F,6.F,ig,Rounding::kFloat32),1.5F);
  EXPECT_EQ(activation(1.25F,6.F,ig,Rounding::kBFloat16),1.F);
  const auto wg=std::bit_cast<float>(std::uint32_t{0x453f40be});
  EXPECT_EQ(storage::bf16(weight(2,0x3c,wg)),0x3a01u);
  EXPECT_EQ(storage::bf16(bf(multiply(1.5F,divide(1.F,wg)))),0x3a00u);
  EXPECT_EQ(std::bit_cast<std::uint32_t>(weight(8,0x38,1.F)),0x80000000u);
  EXPECT_EQ(storage::bf16(activation(-0.F,0.F,1.F,Rounding::kFloat32)),0u);
  EXPECT_EQ(storage::bf16(activation(-1.e-6F,1.e-6F,1.F,Rounding::kFloat32)),0x8000u);
}

enum class Layout { kDense, kShared, kPerSlot, kSharedView };
struct Data final {
  std::vector<std::uint16_t> a;
  std::vector<std::uint8_t> w,scales;
  std::vector<float> wg,ig;
  std::vector<std::int32_t> ids;
  std::array<std::vector<std::byte>,6> payloads() const {
    return {bytes<std::uint16_t>(a),bytes<std::uint8_t>(w),bytes<std::uint8_t>(scales),
        bytes<float>(wg),bytes<float>(ig),bytes<std::int32_t>(ids)};
  }
};
struct Case final {
  static constexpr std::int64_t tokens=2;
  Layout layout;
  Rounding rounding;
  std::int64_t slots,experts,columns=3,depth;
  c::TensorDesc activation_backing,output_description;
  std::array<c::TensorDesc,6> descriptions;
  Case(Layout l=Layout::kDense,Rounding r=Rounding::kFloat32,std::int64_t e=3,std::int64_t k=32)
      :layout(l),rounding(r),slots(l==Layout::kDense?1:3),experts(l==Layout::kDense?1:e),depth(k) {
    const auto physical_depth=depth+(layout==Layout::kSharedView?16:0);
    activation_backing=layout==Layout::kDense ? dense(Type::kBFloat16,{tokens,depth})
        :dense(Type::kBFloat16,{tokens,layout==Layout::kPerSlot?slots:1,physical_depth});
    descriptions[0]=layout==Layout::kSharedView
        ?take(c::TensorDesc::strided_bytes(Type::kBFloat16,std::array{tokens,slots,depth},
            std::array<std::int64_t,3>{physical_depth*2,0,2}),"shared CT view")
        :activation_backing;
    descriptions[1]=grouped()?dense(Type::kUInt8,{experts,columns,depth/2}):dense(Type::kUInt8,{columns,depth/2});
    descriptions[2]=grouped()?dense(Type::kFloat8E4M3FN,{experts,columns,depth/16}):dense(Type::kFloat8E4M3FN,{columns,depth/16});
    descriptions[3]=descriptions[4]=grouped()?dense(Type::kFloat32,{experts}):dense(Type::kFloat32,{1});
    if(grouped()) descriptions[5]=dense(Type::kInt32,{tokens,slots});
    output_description=grouped()?dense(Type::kBFloat16,{tokens,slots,columns}):dense(Type::kBFloat16,{tokens,columns});
  }
  bool grouped() const {return layout!=Layout::kDense;}
  std::size_t port_count() const {return grouped()?6:5;}
  std::size_t a_index(std::int64_t t,std::int64_t s,std::int64_t k) const {
    return std::size_t(t*(activation_backing.byte_strides()[0]/2)+
        (layout==Layout::kPerSlot?s*(activation_backing.byte_strides()[1]/2):0)+k);
  }
  Data data(unsigned phase=0) const {
    Data d;d.a.resize(activation_backing.required_bytes().value()/2,0x7fc1);
    d.w.resize(std::size_t(experts*columns*(depth/2)));
    d.scales.resize(std::size_t(experts*columns*(depth/16)),0x7f);
    d.wg.resize(std::size_t(experts),std::numeric_limits<float>::quiet_NaN());d.ig=d.wg;
    d.wg[0]=2.F;d.ig[0]=std::bit_cast<float>(std::uint32_t{0x3f8020c5});
    if(grouped()) {d.wg[1]=1.00390625F;d.ig[1]=.3F;}
    constexpr std::array<float,4> changes{1.25F,.75F,2.5F,-1.25F};
    for(std::int64_t t=0;t<tokens;++t) for(std::int64_t s=0;s<(layout==Layout::kPerSlot?slots:1);++s) {
      for(std::int64_t k=0;k<depth;++k)d.a[a_index(t,s,k)]=0;
      d.a[a_index(t,s,0)]=storage::bf16(changes[(std::size_t(t+s)+phase)%changes.size()]);
      d.a[a_index(t,s,1)]=storage::bf16(6.F);
      d.a[a_index(t,s,2)]=storage::bf16(-.25F);
      d.a[a_index(t,s,16)]=storage::bf16(1.e-6F);
      d.a[a_index(t,s,17)]=storage::bf16(-1.e-6F);
      d.a[a_index(t,s,18)]=0x8000;
    }
    for(std::int64_t e=0;e<(grouped()?2:1);++e)for(std::int64_t n=0;n<columns;++n) {
      for(std::int64_t group=0;group<depth/16;++group)
        d.scales[std::size_t((e*columns+n)*(depth/16)+group)]=0x38;
      const auto k=n==2?17:n;
      d.w[std::size_t((e*columns+n)*(depth/2)+k/2)]=static_cast<std::uint8_t>(2u<<(4*(k%2)));
    }
    if(grouped())for(std::int64_t t=0;t<tokens;++t)for(std::int64_t s=0;s<slots;++s)
      d.ids.push_back(static_cast<std::int32_t>((t+s+phase)%2));
    return d;
  }
  std::vector<std::uint16_t> oracle(const Data &d) const {
    std::vector<std::uint16_t> result(std::size_t(tokens*slots*columns));
    for(std::int64_t t=0;t<tokens;++t)for(std::int64_t s=0;s<slots;++s) {
      const auto e=grouped()?std::size_t(d.ids[std::size_t(t*slots+s)]):0;
      std::vector<float> a(std::size_t(depth),0.F);
      for(std::int64_t group=0;group<depth/16;++group) {
        float maximum=0;
        for(std::int64_t lane=0;lane<16;++lane)maximum=std::max(maximum,std::abs(storage::wide(d.a[a_index(t,s,group*16+lane)])));
        for(std::int64_t lane=0;lane<16;++lane)a[std::size_t(group*16+lane)]=activation(storage::wide(d.a[a_index(t,s,group*16+lane)]),maximum,d.ig[e],rounding);
      }
      for(std::int64_t n=0;n<columns;++n) {
        float sum=0;
        for(std::int64_t k=0;k<depth;++k) {
          const auto byte=d.w[(e*std::size_t(columns)+std::size_t(n))*std::size_t(depth/2)+std::size_t(k/2)];
          const auto code=(byte>>(4*(k%2)))&15u;
          const auto sf=d.scales[(e*std::size_t(columns)+std::size_t(n))*std::size_t(depth/16)+std::size_t(k/16)];
          sum=f32(double(sum)+double(a[std::size_t(k)])*double(weight(code,sf,d.wg[e])));
        }
        result[std::size_t((t*slots+s)*columns+n)]=storage::bf16(sum);
      }
    }
    return result;
  }
};
struct Logical final {
  std::array<g::ValueId,6> ports{};g::ValueId output{};g::Graph graph;
  Logical(const Case &f,bool dynamic=false) {
    auto b=take(g::GraphBuilder::create(),"local NVFP4 GraphBuilder");
    ports[0]=take(b.input(f.activation_backing),"original dense A backing");auto a=ports[0];
    if(f.layout==Layout::kSharedView) {
      a=take(b.add(take(g::ViewSpec::slice(2,g::Dimension::fixed(0),g::Dimension::fixed(f.depth)),"slice token padding"),a),"sliced actual A");
      const std::array shape{g::Dimension::fixed(f.tokens),g::Dimension::fixed(f.slots),g::Dimension::fixed(f.depth)};
      a=take(b.add(take(g::ViewSpec::broadcast_to(shape),"zero slot stride"),a),"shared actual A view");
    }
    for(std::size_t i=1;i<f.port_count();++i)ports[i]=take(dynamic||i==5?b.input(f.descriptions[i]):b.constant(f.descriptions[i]),"original local NVFP4 port");
    output=f.grouped()?take(b.add(c::model_ops::grouped_local_nvfp4_linear(f.rounding),a,ports[1],ports[2],ports[3],ports[4],ports[5]),"grouped local reference")
        :take(b.add(c::model_ops::local_nvfp4_linear(f.rounding),a,ports[1],ports[2],ports[3],ports[4]),"dense local reference");
    check(b.publish(output),"publish only local NVFP4 output");graph=take(std::move(b).finish(),"finish local reference graph");
  }
};

template<class Compile,class Write,class Read,class Prepare>
void run_lifetimes(m::MemoryResourceRef resource,std::span<const g::QueueBinding> queues,
    Compile compile,Write write,Read read,Prepare prepare,c::StorageOwner owner={}) {
  oracle_checks();unsigned submissions=0;
  for(const auto rounding:{Rounding::kFloat32,Rounding::kBFloat16})
    for(const auto layout:{Layout::kDense,Layout::kShared,Layout::kPerSlot,Layout::kSharedView}) {
      SCOPED_TRACE(::testing::Message()<<"round="<<int(rounding)<<" layout="<<int(layout));
      Case f(layout,rounding,layout==Layout::kSharedView?17:3,
             layout==Layout::kDense?2048:32);Logical logical(f);
      const auto original=f.data();const auto payload=original.payloads();
      std::array<c::ReadLease,4> ready;std::array<g::ConstantBinding,4> constants;
      for(std::size_t i=0;i<4;++i) {
        auto buffer=take(m::Buffer::create(resource,payload[i+1].size(),256),"original local bank");check(write(buffer.view(),payload[i+1]),"upload original local bank");
        ready[i]=owner?take(owner.adopt_buffer(std::move(buffer)),"owned local bank"):take(c::ReadLease::adopt(std::move(buffer)),"local bank lease");
        constants[i]={logical.ports[i+1],take(ready[i].tensor(f.descriptions[i+1]),"typed original bank"),g::ConstantPolicy::kMaterialize,owner,owner?ready[i]:c::ReadLease{}};
      }
      auto compiled=compile(logical.graph,g::CompileOptions{.constants=constants});ASSERT_TRUE(compiled)<<compiled.error().message();
      const auto ownership=take(compiled->executable.output_ownership(),"local output ownership");
      if(ownership!=g::OutputOwnership::kCaller) {ASSERT_EQ(ownership,g::OutputOwnership::kInvocation);}
      std::array runs{take(compiled->executable.create_invocation({.resources=g::ResourceOptions{}}),"local slot0"),take(compiled->executable.create_invocation({.resources=g::ResourceOptions{}}),"local slot1")};
      constants={};ready={};compiled=g::CompiledGraph{};logical.graph={};
      const auto output_bytes=f.output_description.required_bytes().value();c::LeasePool pool;
      if(ownership==g::OutputOwnership::kCaller)pool=take(c::LeasePool::create(resource,2*((output_bytes+255)/256*256),{.max_blocks=2}),"two output writers");
      std::array<std::array<m::Buffer,2>,2> inputs;
      for(std::size_t slot=0;slot<2;++slot) {
        for(std::size_t i=0;i<(f.grouped()?2u:1u);++i) {
          const auto port=i?5u:0u;inputs[slot][i]=take(m::Buffer::create(resource,payload[port].size(),256),"dynamic A or IDs");check(write(inputs[slot][i].view(),payload[port]),"initial A or IDs");
          check(runs[slot].bind(logical.ports[port],take(c::ConstTensorView::create(i?f.descriptions[5]:f.activation_backing,inputs[slot][i].view()),"current view")),"bind once");
        }
        if(ownership==g::OutputOwnership::kCaller)check(runs[slot].bind(logical.output,f.output_description,take(pool.allocate(output_bytes),"output writer")),"bind owned output");
        check(prepare(runs[slot],queues),"prepare local replay");
      }
      std::array<g::GraphOutput,2> old;std::array<std::vector<std::uint16_t>,2> expected;
      for(unsigned round=0;round<2;++round)for(std::size_t slot=0;slot<2;++slot) {
        if(round) {storage::expect_result(old[slot],expected[slot],read);EXPECT_FALSE(runs[slot].submit(queues));old[slot].reset();}
        const auto data=f.data(round*2+static_cast<unsigned>(slot));check(write(inputs[slot][0].view(),std::as_bytes(std::span(data.a))),"update current A");
        if(f.grouped())check(write(inputs[slot][1].view(),std::as_bytes(std::span(data.ids))),"update current IDs");
        auto done=take(runs[slot].submit(queues),"local submit");check(done.wait(),"local completion");++submissions;
        old[slot]=take(runs[slot].result(logical.output),"local result");expected[slot]=f.oracle(data);storage::expect_result(old[slot],expected[slot],read);
        if(old[1-slot].valid())storage::expect_result(old[1-slot],expected[1-slot],read);
      }
      EXPECT_NE(f.oracle(f.data()),f.oracle(f.data(2)));runs={};
      for(std::size_t slot=0;slot<2;++slot)storage::expect_result(old[slot],expected[slot],read);
      old={};if(ownership==g::OutputOwnership::kCaller) {EXPECT_EQ(pool.stats().live_blocks,0u);}
    }
  ::testing::Test::RecordProperty("actual_submissions",submissions);
}

template<class Compile,class Write,class Read,class Prepare>
void run_runtime_domain(m::MemoryResourceRef resource,std::span<const g::QueueBinding> queues,
    Compile compile,Write write,Read read,Prepare prepare) {
  unsigned successes=0,failures=0;
  for(const auto rounding:{Rounding::kFloat32,Rounding::kBFloat16})for(const auto layout:{Layout::kDense,Layout::kSharedView}) {
    SCOPED_TRACE(::testing::Message()<<"round="<<int(rounding)<<" layout="<<int(layout));
    Case f(layout,rounding);Logical logical(f,true);auto compiled=compile(logical.graph,g::CompileOptions{});ASSERT_TRUE(compiled)<<compiled.error().message();
    auto executable=compiled->executable;auto run=take(executable.create_invocation({.resources=g::ResourceOptions{}}),"dynamic local slot");
    const auto ownership=take(executable.output_ownership(),"runtime local ownership");
    if(ownership!=g::OutputOwnership::kCaller) {ASSERT_EQ(ownership,g::OutputOwnership::kInvocation);}
    const auto valid=f.data();const auto initial=valid.payloads();std::array<m::Buffer,6> inputs;
    for(std::size_t i=0;i<f.port_count();++i)inputs[i]=take(m::Buffer::create(resource,initial[i].size(),256),"dynamic original input");
    const auto count=f.output_description.required_bytes().value();c::LeasePool pool;
    if(ownership==g::OutputOwnership::kCaller)pool=take(c::LeasePool::create(resource,(count+255)/256*256,{.max_blocks=1}),"runtime writer");
    const auto upload=[&](const Data &d) {const auto p=d.payloads();for(std::size_t i=0;i<f.port_count();++i)check(write(inputs[i].view(),p[i]),"update actual runtime bytes");};
    const auto bind=[&] {
      for(std::size_t i=0;i<f.port_count();++i)check(run.bind(logical.ports[i],take(c::ConstTensorView::create(i?f.descriptions[i]:f.activation_backing,inputs[i].view()),"runtime view")),"bind runtime once");
      if(ownership==g::OutputOwnership::kCaller)check(run.bind(logical.output,f.output_description,take(pool.allocate(count),"runtime owned output")),"bind runtime result");
      check(prepare(run,queues),"prepare current local slot");
    };
    upload(valid);bind();compiled=g::CompiledGraph{};logical.graph={};
    const auto success=[&](const Data &d) {upload(d);auto done=take(run.submit(queues),"accepted local submit");check(done.wait(),"valid local completion");auto result=take(run.result(logical.output),"current local result");storage::expect_result(result,f.oracle(d),read);++successes;};
    success(valid); // Unselected expert2 remains invalid; NaN token padding is unread.
    std::vector<std::pair<std::string,Data>> invalid;
    auto d=valid;d.a[0]=0x7f80;invalid.emplace_back("nonfinite A",d);
    d=valid;d.scales[0]=0x7f;invalid.emplace_back("NaN SF",d);
    d=valid;d.scales[0]=0xb8;invalid.emplace_back("negative SF",d);
    d=valid;d.wg[0]=0.F;invalid.emplace_back("nonpositive WG",d);
    d=valid;d.ig[0]=std::numeric_limits<float>::quiet_NaN();invalid.emplace_back("nonfinite IG",d);
    if(f.grouped())for(const auto id:{-1,static_cast<int>(f.experts)}) {d=valid;d.ids.back()=id;invalid.emplace_back("invalid last ID",d);}
    for(const auto &[label,bad]:invalid) {
      SCOPED_TRACE(label);upload(bad);
      {auto done=run.submit(queues);ASSERT_TRUE(done)<<done.error().message();EXPECT_FALSE(done->wait().ok());EXPECT_FALSE(run.result(logical.output));}
      EXPECT_FALSE(run.submit(queues))<<"failed Invocation is terminal";++failures;
      run={};run=take(executable.create_invocation({.resources=g::ResourceOptions{}}),"fresh recovery slot");upload(valid);bind();success(valid);
    }
    // Direct division has a BF16-visible result that reciprocal multiplication loses.
    d=valid;std::fill(d.a.begin(),d.a.end(),0);std::fill(d.w.begin(),d.w.end(),0);std::fill(d.scales.begin(),d.scales.end(),0x3c);
    for(std::int64_t t=0;t<f.tokens;++t) {d.a[f.a_index(t,0,0)]=storage::bf16(1.F);d.a[f.a_index(t,0,1)]=storage::bf16(6.F);}
    for(std::int64_t e=0;e<(f.grouped()?2:1);++e) {d.wg[std::size_t(e)]=std::bit_cast<float>(std::uint32_t{0x453f40be});d.ig[std::size_t(e)]=1.F;for(std::int64_t n=0;n<f.columns;++n)d.w[std::size_t((e*f.columns+n)*(f.depth/2))]=2;}
    for(const auto expected:f.oracle(d)) {ASSERT_EQ(expected,0x3a01u);}
    success(d);
    d=valid;d.ig[0]=std::numeric_limits<float>::denorm_min();success(d);
    d=valid;d.wg[0]=std::numeric_limits<float>::denorm_min();success(d);
    d=valid;for(std::size_t i=0;i<d.scales.size();++i)d.scales[i]=i%2?0x80:0;success(d);
  }
  ::testing::Test::RecordProperty("successful_submissions",successes);
  ::testing::Test::RecordProperty("failed_completions",failures);
}
} // namespace local_nvfp4_reference_test
