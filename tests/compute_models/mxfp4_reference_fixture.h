#pragma once

#include "nvfp4_reference_fixture.h"
#include "hyle/compute/weight_import.h"

namespace mxfp4_reference_test {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
namespace storage = nvfp4_reference_test;
using Type = c::DataType;
using storage::take;
using storage::check;
using storage::dense;
using storage::bytes;
using ActivationMode = c::model_ops::Mxfp4ActivationMode;

// Independent scalar oracle: a numerical binade/threshold scale calculation
// and enumeration of the eight magnitudes, rather than the product bit masks.
inline float f32(double value) { volatile float v = static_cast<float>(value); return v; }
inline float bf(float value) { return storage::wide(storage::bf16(value)); }
inline constexpr std::array<float,8> magnitudes{0.F,.5F,1.F,1.5F,2.F,3.F,4.F,6.F};
inline float scale(float maximum) {
  if (maximum == 0.F) return std::ldexp(1.F,-127);
  if (double(maximum) >= std::ldexp(1.75,127)) return std::ldexp(1.F,127);
  int exponent = 0;
  const auto fraction = std::frexp(double(maximum), &exponent);
  return std::ldexp(1.F, std::clamp(exponent - (fraction >= .875 ? 2 : 3), -127, 127));
}
inline float nearest(float value) {
  unsigned best = 0;
  double distance = std::abs(double(value));
  for (unsigned code=1; code<magnitudes.size(); ++code) {
    const auto next = std::abs(std::abs(double(value))-double(magnitudes[code]));
    if (next < distance || (next == distance && !(code&1u))) { best=code; distance=next; }
  }
  return std::copysign(magnitudes[best],value);
}
inline float activation(float value,float maximum) {
  const auto s = scale(maximum);
  const auto normalized = bf(f32(double(value)/double(s)));
  return bf(f32(double(nearest(normalized))*double(s)));
}
inline float weight(unsigned code,std::uint8_t sf) {
  const auto value = code&8u ? -magnitudes[code&7u] : magnitudes[code&7u];
  return bf(f32(double(value)*std::ldexp(1.,int(sf)-127)));
}
inline void oracle_checks() {
  EXPECT_EQ(scale(6.5F),1.F); // ceil(max/6) would choose2.
  EXPECT_EQ(activation(1.25F,6.F),1.F); // optional Triton ties-up would choose1.5.
  EXPECT_EQ(activation(1.75F,6.F),2.F);
  EXPECT_EQ(storage::bf16(activation(-0.F,0.F)),0x8000u);
  EXPECT_EQ(storage::bf16(weight(8,127)),0x8000u);
  EXPECT_EQ(storage::bf16(weight(2,0)),0x0040u); // scale0 is not zero.
  EXPECT_EQ(activation(storage::wide(0x0040),storage::wide(0x0040)),storage::wide(0x0040));
  EXPECT_TRUE(std::isinf(activation(storage::wide(0x7f7f),storage::wide(0x7f7f))));
}

enum class Layout { kDense, kShared, kPerSlot, kSharedView };
struct Data final {
  std::vector<std::uint16_t> a;
  std::vector<std::uint8_t> w,scales;
  std::vector<std::int32_t> ids;
  std::array<std::vector<std::byte>,4> payloads() const {
    return {bytes<std::uint16_t>(a),bytes<std::uint8_t>(w),bytes<std::uint8_t>(scales),bytes<std::int32_t>(ids)};
  }
};
struct Case final {
  static constexpr std::int64_t tokens=2;
  Layout layout;
  ActivationMode mode;
  std::int64_t slots,experts,columns=3,depth;
  c::TensorDesc activation_backing,output_description;
  std::array<c::TensorDesc,4> descriptions;
  Case(Layout l=Layout::kDense,std::int64_t e=3,std::int64_t k=64,
       ActivationMode activation_mode=ActivationMode::kFloat4Group32)
      :layout(l),mode(activation_mode),slots(l==Layout::kDense?1:3),experts(l==Layout::kDense?1:e),depth(k) {
    const auto physical_depth=depth+(layout==Layout::kSharedView?32:0);
    activation_backing=layout==Layout::kDense?dense(Type::kBFloat16,{tokens,depth})
        :dense(Type::kBFloat16,{tokens,layout==Layout::kPerSlot?slots:1,physical_depth});
    descriptions[0]=layout==Layout::kSharedView
        ?take(c::TensorDesc::strided_bytes(Type::kBFloat16,std::array{tokens,slots,depth},
            std::array<std::int64_t,3>{physical_depth*2,0,2}),"shared MX activation view")
        :activation_backing;
    descriptions[1]=grouped()?dense(Type::kUInt8,{experts,columns,depth/2}):dense(Type::kUInt8,{columns,depth/2});
    descriptions[2]=grouped()?dense(Type::kUInt8,{experts,columns,depth/32}):dense(Type::kUInt8,{columns,depth/32});
    if(grouped()) descriptions[3]=dense(Type::kInt32,{tokens,slots});
    output_description=grouped()?dense(Type::kBFloat16,{tokens,slots,columns}):dense(Type::kBFloat16,{tokens,columns});
  }
  bool grouped() const {return layout!=Layout::kDense;}
  std::size_t port_count() const {return grouped()?4:3;}
  std::size_t a_index(std::int64_t t,std::int64_t s,std::int64_t k) const {
    return std::size_t(t*(activation_backing.byte_strides()[0]/2)+
        (layout==Layout::kPerSlot?s*(activation_backing.byte_strides()[1]/2):0)+k);
  }
  Data data(unsigned phase=0) const {
    Data d; d.a.resize(activation_backing.required_bytes().value()/2,0x7fc1);
    d.w.resize(std::size_t(experts*columns*(depth/2)));
    d.scales.resize(std::size_t(experts*columns*(depth/32)),255);
    constexpr std::array<float,4> changes{1.25F,.75F,2.5F,-1.25F};
    for(std::int64_t t=0;t<tokens;++t)for(std::int64_t s=0;s<(layout==Layout::kPerSlot?slots:1);++s) {
      for(std::int64_t k=0;k<depth;++k)d.a[a_index(t,s,k)]=0;
      d.a[a_index(t,s,0)]=storage::bf16(changes[(std::size_t(t+s)+phase)%changes.size()]);
      d.a[a_index(t,s,1)]=storage::bf16(6.5F);
      d.a[a_index(t,s,2)]=0x8000;
      d.a[a_index(t,s,3)]=storage::bf16(-1.25F);
      d.a[a_index(t,s,32)]=storage::bf16(1.e-6F);
      d.a[a_index(t,s,33)]=storage::bf16(-1.e-6F);
    }
    for(std::int64_t e=0;e<(grouped()?2:1);++e)for(std::int64_t n=0;n<columns;++n) {
      for(std::int64_t group=0;group<depth/32;++group)
        d.scales[std::size_t((e*columns+n)*(depth/32)+group)]=static_cast<std::uint8_t>(127-e-(n==2?10:0));
      const auto k=n==2?33:n;
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
      for(std::int64_t group=0;group<depth/32;++group) {
        float maximum=0.F;
        for(std::int64_t lane=0;lane<32;++lane)maximum=std::max(maximum,std::abs(storage::wide(d.a[a_index(t,s,group*32+lane)])));
        for(std::int64_t lane=0;lane<32;++lane) {
          const auto value=storage::wide(d.a[a_index(t,s,group*32+lane)]);
          a[std::size_t(group*32+lane)]=mode==ActivationMode::kBFloat16?value:activation(value,maximum);
        }
      }
      for(std::int64_t n=0;n<columns;++n) {
        float sum=0;
        for(std::int64_t k=0;k<depth;++k) {
          const auto byte=d.w[(e*std::size_t(columns)+std::size_t(n))*std::size_t(depth/2)+std::size_t(k/2)];
          const auto code=(byte>>(4*(k%2)))&15u;
          const auto sf=d.scales[(e*std::size_t(columns)+std::size_t(n))*std::size_t(depth/32)+std::size_t(k/32)];
          sum=f32(double(sum)+double(a[std::size_t(k)])*double(weight(code,sf)));
        }
        result[std::size_t((t*slots+s)*columns+n)]=storage::bf16(sum);
      }
    }
    return result;
  }
};
struct Logical final {
  std::array<g::ValueId,4> ports{};g::ValueId output{};g::Graph graph;
  Logical(const Case &f,bool dynamic=false) {
    auto b=take(g::GraphBuilder::create(),"MXFP4 GraphBuilder");
    ports[0]=take(b.input(f.activation_backing),"original A backing");auto a=ports[0];
    if(f.layout==Layout::kSharedView) {
      a=take(b.add(take(g::ViewSpec::slice(2,g::Dimension::fixed(0),g::Dimension::fixed(f.depth)),"slice token padding"),a),"sliced actual A");
      const std::array shape{g::Dimension::fixed(f.tokens),g::Dimension::fixed(f.slots),g::Dimension::fixed(f.depth)};
      a=take(b.add(take(g::ViewSpec::broadcast_to(shape),"zero slot stride"),a),"shared actual A view");
    }
    for(std::size_t i=1;i<f.port_count();++i)ports[i]=take(dynamic||i==3?b.input(f.descriptions[i]):b.constant(f.descriptions[i]),"original MXFP4 port");
    output=f.grouped()?take(b.add(c::model_ops::grouped_mxfp4_linear(f.mode),a,ports[1],ports[2],ports[3]),"grouped MX reference")
        :take(b.add(c::model_ops::mxfp4_linear(f.mode),a,ports[1],ports[2]),"dense MX reference");
    check(b.publish(output),"publish MXFP4 output");graph=take(std::move(b).finish(),"finish MXFP4 graph");
  }
};

template<class Compile,class Write,class Read,class Prepare>
void run_lifetimes(m::MemoryResourceRef resource,std::span<const g::QueueBinding> queues,
    Compile compile,Write write,Read read,Prepare prepare,c::StorageOwner owner={},
    ActivationMode mode=ActivationMode::kFloat4Group32) {
  oracle_checks();unsigned submissions=0;
  for(const auto layout:{Layout::kDense,Layout::kShared,Layout::kPerSlot,Layout::kSharedView}) {
      SCOPED_TRACE(::testing::Message()<<"layout="<<int(layout));
      Case f(layout,layout==Layout::kSharedView?17:3,
             layout==Layout::kDense?2048:64,mode);Logical logical(f);
      const auto original=f.data();const auto payload=original.payloads();
      if(mode==ActivationMode::kBFloat16) {
        auto a4=f;a4.mode=ActivationMode::kFloat4Group32;
        EXPECT_NE(f.oracle(original),a4.oracle(original));
      }
      std::array<c::ReadLease,2> ready;std::array<g::ConstantBinding,2> constants;
      for(std::size_t i=0;i<2;++i) {
        auto buffer=take(m::Buffer::create(resource,payload[i+1].size(),256),"original MX bank");check(write(buffer.view(),payload[i+1]),"upload original MX bank");
        ready[i]=owner?take(owner.adopt_buffer(std::move(buffer)),"owned MX bank"):take(c::ReadLease::adopt(std::move(buffer)),"MX bank lease");
        constants[i]={logical.ports[i+1],take(ready[i].tensor(f.descriptions[i+1]),"typed original bank"),g::ConstantPolicy::kMaterialize,owner,owner?ready[i]:c::ReadLease{}};
      }
      auto compiled=compile(logical.graph,g::CompileOptions{.constants=constants});ASSERT_TRUE(compiled)<<compiled.error().message();
      const auto ownership=take(compiled->executable.output_ownership(),"MX output ownership");
      if(ownership!=g::OutputOwnership::kCaller) {ASSERT_EQ(ownership,g::OutputOwnership::kInvocation);}
      std::array runs{take(compiled->executable.create_invocation({.resources=g::ResourceOptions{}}),"MX slot0"),take(compiled->executable.create_invocation({.resources=g::ResourceOptions{}}),"MX slot1")};
      constants={};ready={};compiled=g::CompiledGraph{};logical.graph={};
      const auto output_bytes=f.output_description.required_bytes().value();c::LeasePool pool;
      if(ownership==g::OutputOwnership::kCaller)pool=take(c::LeasePool::create(resource,2*((output_bytes+255)/256*256),{.max_blocks=2}),"two output writers");
      std::array<std::array<m::Buffer,2>,2> inputs;
      for(std::size_t slot=0;slot<2;++slot) {
        for(std::size_t i=0;i<(f.grouped()?2u:1u);++i) {
          const auto port=i?3u:0u;inputs[slot][i]=take(m::Buffer::create(resource,payload[port].size(),256),"dynamic A or IDs");check(write(inputs[slot][i].view(),payload[port]),"initial A or IDs");
          check(runs[slot].bind(logical.ports[port],take(c::ConstTensorView::create(i?f.descriptions[3]:f.activation_backing,inputs[slot][i].view()),"current view")),"bind once");
        }
        if(ownership==g::OutputOwnership::kCaller)check(runs[slot].bind(logical.output,f.output_description,take(pool.allocate(output_bytes),"output writer")),"bind owned output");
        check(prepare(runs[slot],queues),"prepare MX replay");
      }
      std::array<g::GraphOutput,2> old;std::array<std::vector<std::uint16_t>,2> expected;
      for(unsigned round=0;round<2;++round)for(std::size_t slot=0;slot<2;++slot) {
        if(round) {storage::expect_result(old[slot],expected[slot],read);EXPECT_FALSE(runs[slot].submit(queues));old[slot].reset();}
        const auto data=f.data(round*2+static_cast<unsigned>(slot));check(write(inputs[slot][0].view(),std::as_bytes(std::span(data.a))),"update current A");
        if(f.grouped())check(write(inputs[slot][1].view(),std::as_bytes(std::span(data.ids))),"update current IDs");
        auto done=take(runs[slot].submit(queues),"MX submit");check(done.wait(),"MX completion");++submissions;
        old[slot]=take(runs[slot].result(logical.output),"MX result");expected[slot]=f.oracle(data);storage::expect_result(old[slot],expected[slot],read);
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
    Compile compile,Write write,Read read,Prepare prepare,
    ActivationMode mode=ActivationMode::kFloat4Group32) {
  unsigned successes=0,failures=0;
  for(const auto layout:{Layout::kDense,Layout::kSharedView}) {
    SCOPED_TRACE(::testing::Message()<<"layout="<<int(layout));
    Case f(layout,3,64,mode);Logical logical(f,true);
    auto compiled=compile(logical.graph,g::CompileOptions{});ASSERT_TRUE(compiled)<<compiled.error().message();
    auto executable=compiled->executable;
    auto run=take(executable.create_invocation({.resources=g::ResourceOptions{}}),"dynamic MX slot");
    const auto ownership=take(executable.output_ownership(),"MX result ownership");
    if(ownership!=g::OutputOwnership::kCaller) {ASSERT_EQ(ownership,g::OutputOwnership::kInvocation);}
    const auto valid=f.data();const auto initial=valid.payloads();std::array<m::Buffer,4> inputs;
    for(std::size_t i=0;i<f.port_count();++i)inputs[i]=take(m::Buffer::create(resource,initial[i].size(),256),"original dynamic MX input");
    const auto count=f.output_description.required_bytes().value();c::LeasePool pool;
    if(ownership==g::OutputOwnership::kCaller)pool=take(c::LeasePool::create(resource,(count+255)/256*256,{.max_blocks=1}),"MX output writer");
    const auto upload=[&](const Data &d) {const auto p=d.payloads();for(std::size_t i=0;i<f.port_count();++i)check(write(inputs[i].view(),p[i]),"update runtime original bytes");};
    const auto bind=[&] {
      for(std::size_t i=0;i<f.port_count();++i)check(run.bind(logical.ports[i],take(c::ConstTensorView::create(i?f.descriptions[i]:f.activation_backing,inputs[i].view()),"MX runtime view")),"bind runtime once");
      if(ownership==g::OutputOwnership::kCaller)check(run.bind(logical.output,f.output_description,take(pool.allocate(count),"MX result lease")),"bind runtime result");
      check(prepare(run,queues),"prepare MX replay");
    };
    upload(valid);bind();compiled=g::CompiledGraph{};logical.graph={};
    const auto success=[&](const Data &d) {
      upload(d);auto done=take(run.submit(queues),"MX submit");check(done.wait(),"valid MX completion");
      auto result=take(run.result(logical.output),"MX result");storage::expect_result(result,f.oracle(d),read);++successes;
    };
    success(valid); // Unselected SF255 and NaN token padding must remain unread.
    std::vector<std::pair<std::string,Data>> invalid;
    for(const auto bits:std::array<std::uint16_t,3>{0x7f80,0xff80,0x7fc1}) {
      auto d=valid;d.a[0]=bits;invalid.emplace_back("nonfinite source A",d);
    }
    auto bad_scale=valid;bad_scale.scales[0]=255;invalid.emplace_back("selected scale255",bad_scale);
    if(f.grouped())for(const auto id:{-1,static_cast<int>(f.experts)}) {
      auto d=valid;d.ids.back()=id;invalid.emplace_back("invalid last ID",d);
    }
    for(const auto &[label,bad]:invalid) {
      SCOPED_TRACE(label);upload(bad);
      {auto done=run.submit(queues);ASSERT_TRUE(done)<<done.error().message();EXPECT_FALSE(done->wait().ok());EXPECT_FALSE(run.result(logical.output));}
      EXPECT_FALSE(run.submit(queues))<<"failed Invocation is terminal";++failures;
      run={};run=take(executable.create_invocation({.resources=g::ResourceOptions{}}),"fresh recovery slot");upload(valid);bind();success(valid);
    }
    // Publish the same real buffers with changed W/SF/A under the same replay.
    // A tiny BF16 operand times a large finite W exposes flush-to-zero and a
    // scale0 decoder error, while using only one nonzero term avoids ambiguous
    // dot-reduction comparisons.
    for(const auto bits:std::array<std::uint16_t,6>{0,0x8000,0x0001,0x0010,0x0030,0x0040}) {
      auto d=valid;std::fill(d.a.begin(),d.a.end(),0);std::fill(d.w.begin(),d.w.end(),0);
      for(std::int64_t e=0;e<(f.grouped()?2:1);++e)for(std::int64_t n=0;n<f.columns;++n) {
        const auto row=std::size_t(e*f.columns+n);d.w[row*std::size_t(f.depth/2)]=2;
        for(std::int64_t g=0;g<f.depth/32;++g)d.scales[row*std::size_t(f.depth/32)+std::size_t(g)]=254;
      }
      for(std::int64_t t=0;t<f.tokens;++t)d.a[f.a_index(t,0,0)]=bits;
      const auto expected=mode==ActivationMode::kFloat4Group32
          ? ((bits==0x0030||bits==0x0040)?0x3f80u:0u)
          : ((bits&0x7fffu)==0?0u:storage::bf16(std::ldexp(storage::wide(bits),127)));
      for(const auto result:f.oracle(d)) {ASSERT_EQ(result,expected);}
      success(d);
    }
    // E8M0 code0 reconstructs a nonzero BF16 W operand, not a zero scale.
    auto d=valid;std::fill(d.a.begin(),d.a.end(),0);std::fill(d.w.begin(),d.w.end(),0);
    for(std::int64_t t=0;t<f.tokens;++t)d.a[f.a_index(t,0,0)]=storage::bf16(6.F);
    for(std::int64_t e=0;e<(f.grouped()?2:1);++e)for(std::int64_t n=0;n<f.columns;++n) {
      const auto row=std::size_t(e*f.columns+n);d.w[row*std::size_t(f.depth/2)]=2;
      for(std::int64_t g=0;g<f.depth/32;++g)d.scales[row*std::size_t(f.depth/32)+std::size_t(g)]=0;
    }
    for(const auto result:f.oracle(d)) {ASSERT_EQ(result,0x0140u);}
    success(d);
    if(mode==ActivationMode::kBFloat16) {
      // W must be scaled before the dot. An unscaled code-dot overflows here,
      // while the decoded BF16 W and final product are both finite.
      d=valid;std::fill(d.a.begin(),d.a.end(),0);std::fill(d.w.begin(),d.w.end(),0);
      for(std::int64_t t=0;t<f.tokens;++t)d.a[f.a_index(t,0,0)]=0x7f7f;
      for(std::int64_t e=0;e<(f.grouped()?2:1);++e)for(std::int64_t n=0;n<f.columns;++n) {
        const auto row=std::size_t(e*f.columns+n);d.w[row*std::size_t(f.depth/2)]=7;
        for(std::int64_t group=0;group<f.depth/32;++group)
          d.scales[row*std::size_t(f.depth/32)+std::size_t(group)]=117;
      }
      for(const auto result:f.oracle(d)) {ASSERT_TRUE(std::isfinite(storage::wide(result)));}
      success(d);
    }
    // Finite A can overflow during QDQ; finite W scales can also overflow
    // decoded BF16 operands. Such results remain successful IEEE arithmetic.
    d=valid;for(std::int64_t t=0;t<f.tokens;++t)d.a[f.a_index(t,0,0)]=0x7f7f;success(d);
    d=valid;
    for(std::int64_t e=0;e<(f.grouped()?2:1);++e)for(std::int64_t n=0;n<f.columns;++n) {
      d.scales[std::size_t((e*f.columns+n)*(f.depth/32))]=254;
      d.w[std::size_t((e*f.columns+n)*(f.depth/2))]=4;
    }
    success(d);
  }
  ::testing::Test::RecordProperty("successful_submissions",successes);
  ::testing::Test::RecordProperty("failed_completions",failures);
}
} // namespace mxfp4_reference_test
