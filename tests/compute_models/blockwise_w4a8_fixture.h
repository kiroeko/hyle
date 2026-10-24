#pragma once

#include "nvfp4_reference_fixture.h"

namespace blockwise_w4a8_test {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
namespace storage = nvfp4_reference_test;
using Type = c::DataType;
using storage::take;
using storage::check;
using storage::dense;
using storage::bytes;
inline float bf(double v) { return storage::wide(storage::bf16(storage::f32(v))); }
enum class Layout { kDense, kShared, kPerSlot, kSharedView };
struct Data final {
  std::vector<std::uint16_t> a;
  std::vector<std::uint32_t> w;
  std::vector<float> scales;
  std::vector<std::int32_t> ids;
  std::array<std::vector<std::byte>,4> payloads() const {
    return {bytes<std::uint16_t>(a),bytes<std::uint32_t>(w),bytes<float>(scales),bytes<std::int32_t>(ids)};
  }
};
struct Case final {
  static constexpr std::int64_t tokens = 2;
  Layout layout;
  std::int64_t slots, experts, columns = 3, depth;
  c::TensorDesc activation_backing, output_description;
  std::array<c::TensorDesc,4> descriptions;
  Case(Layout l = Layout::kDense, std::int64_t e = 3, std::int64_t k = 1536)
      : layout(l), slots(l == Layout::kDense ? 1 : 3), experts(l == Layout::kDense ? 1 : e), depth(k) {
    const auto physical = depth + (layout == Layout::kSharedView ? 8 : 0);
    activation_backing = !grouped() ? dense(Type::kBFloat16,{tokens,depth})
        : dense(Type::kBFloat16,{tokens,layout == Layout::kPerSlot ? slots : 1,physical});
    descriptions[0] = layout == Layout::kSharedView
        ? take(c::TensorDesc::strided_bytes(Type::kBFloat16,std::array{tokens,slots,depth},
            std::array<std::int64_t,3>{physical*2,0,2}),"shared group512 A") : activation_backing;
    descriptions[1] = grouped() ? dense(Type::kInt32,{experts,columns,depth/8}) : dense(Type::kInt32,{columns,depth/8});
    descriptions[2] = grouped() ? dense(Type::kFloat32,{experts,columns,depth/512}) : dense(Type::kFloat32,{columns,depth/512});
    if (grouped()) descriptions[3] = dense(Type::kInt32,{tokens,slots});
    output_description = grouped() ? dense(Type::kBFloat16,{tokens,slots,columns}) : dense(Type::kBFloat16,{tokens,columns});
  }
  bool grouped() const { return layout != Layout::kDense; }
  std::size_t port_count() const { return grouped() ? 4 : 3; }
  std::size_t a_index(std::int64_t t,std::int64_t s,std::int64_t k) const {
    return std::size_t(t*(activation_backing.byte_strides()[0]/2) +
        (layout == Layout::kPerSlot ? s*(activation_backing.byte_strides()[1]/2) : 0) + k);
  }
  Data data(unsigned phase = 0) const {
    Data d; d.a.resize(activation_backing.required_bytes().value()/2,0x7fc1);
    d.w.resize(std::size_t(experts*columns*(depth/8)),0x88888888u);
    d.scales.resize(std::size_t(experts*columns*(depth/512)),std::numeric_limits<float>::quiet_NaN());
    for (std::int64_t t=0;t<tokens;++t) for (std::int64_t s=0;s<(layout==Layout::kPerSlot?slots:1);++s) {
      for(std::int64_t k=0;k<depth;++k) d.a[a_index(t,s,k)]=0;
      for(std::int64_t g=0;g<depth/512;++g) {
        const float factor=float(1+((std::size_t(t+s)+phase)%4));
        d.a[a_index(t,s,g*512)] = storage::bf16(factor*(g%2?-1.F:1.F));
        d.a[a_index(t,s,g*512+1)] = storage::bf16(2.F*factor);
        d.a[a_index(t,s,g*512+7)] = storage::bf16(-.75F*factor);
        d.a[a_index(t,s,g*512+8)] = 0x8000;
      }
    }
    for(std::int64_t e=0;e<(grouped()?2:1);++e) for(std::int64_t n=0;n<columns;++n) {
      const auto row=std::size_t(e*columns+n);
      for(std::int64_t g=0;g<depth/512;++g) {
        d.scales[row*std::size_t(depth/512)+std::size_t(g)] = (g==1?0.003921509F:1.00390625F)*float(e+1);
        for(const auto lane:{0,1,7,8}) {
          const auto k=std::size_t(g*512+lane);
          const auto shift=unsigned(k%8)*4;
          const auto code=lane==0?int(n)+1:lane==1?0:lane==7?-int(n)-1:-8;
          auto &word=d.w[row*std::size_t(depth/8)+k/8];
          word=(word&~(15u<<shift))|(unsigned(code+8)<<shift);
        }
      }
    }
    if(grouped()) for(std::int64_t t=0;t<tokens;++t) for(std::int64_t s=0;s<slots;++s)
      d.ids.push_back(std::int32_t((t+s+phase)%2));
    return d;
  }
  // Scalar oracle uses double arithmetic, enumerated logical nibble offsets,
  // and a separate storage RNE conversion; no product codec or native kernel.
  std::vector<std::uint16_t> oracle(const Data &d) const {
    std::vector<std::uint16_t> y(std::size_t(tokens*slots*columns));
    for(std::int64_t t=0;t<tokens;++t) for(std::int64_t s=0;s<slots;++s) {
      const auto e=grouped()?std::size_t(d.ids[std::size_t(t*slots+s)]):0;
      for(std::int64_t n=0;n<columns;++n) {
        float sum=0.F;
        for(std::int64_t g=0;g<depth/512;++g) {
          float maximum=0;
          for(std::int64_t i=0;i<512;++i) maximum=std::max(maximum,std::abs(storage::wide(d.a[a_index(t,s,g*512+i)])));
          const auto scale=bf(double(maximum)*0.00787353515625);
          const auto inverse=scale==0.F?0.F:bf(1./double(scale));
          std::int64_t total=0;
          for(std::int64_t i=0;i<512;++i) {
            const auto k=std::size_t(g*512+i);
            const auto v=bf(double(storage::wide(d.a[a_index(t,s,std::int64_t(k))]))*double(inverse));
            if(!std::isfinite(v)||std::trunc(v)<-128.F||std::trunc(v)>127.F) throw std::runtime_error("undefined source I8 conversion in oracle");
            const auto word=d.w[(e*std::size_t(columns)+std::size_t(n))*std::size_t(depth/8)+k/8];
            const auto code=int((word>>(unsigned(k%8)*4))&15u)-8;
            total+=std::int64_t(std::trunc(v))*code;
          }
          const auto dot=bf(double(total));
          const auto first=bf(double(dot)*double(scale));
          const auto sf=bf(d.scales[(e*std::size_t(columns)+std::size_t(n))*std::size_t(depth/512)+std::size_t(g)]);
          const auto value=bf(double(first)*double(sf));
          sum=bf(double(sum)+double(value));
        }
        y[std::size_t((t*slots+s)*columns+n)]=storage::bf16(sum);
      }
    }
    return y;
  }
};
inline void oracle_checks() {
  Case f(Layout::kDense,1,512);auto d=f.data();
  std::fill(d.a.begin(),d.a.end(),0);std::fill(d.w.begin(),d.w.end(),0x88888888u);
  std::fill(d.scales.begin(),d.scales.end(),1.F);
  for(std::int64_t t=0;t<f.tokens;++t) {d.a[f.a_index(t,0,0)]=0x3f80;d.a[f.a_index(t,0,1)]=0x4000;}
  for(std::int64_t n=0;n<f.columns;++n) d.w[std::size_t(n*f.depth/8)]=0x88888889u;
  for(const auto value:f.oracle(d)) EXPECT_EQ(value,0x3f7eu); // W4A16 would be1.
}
struct Logical final {
  std::array<g::ValueId,4> ports{};g::ValueId output{};g::Graph graph;
  Logical(const Case &f,bool dynamic=false) {
    auto b=take(g::GraphBuilder::create(),"W4A8 builder");
    ports[0]=take(b.input(f.activation_backing),"original A");auto a=ports[0];
    if(f.layout==Layout::kSharedView) {
      a=take(b.add(take(g::ViewSpec::slice(2,g::Dimension::fixed(0),g::Dimension::fixed(f.depth)),"slice padding"),a),"sliced A");
      const std::array shape{g::Dimension::fixed(f.tokens),g::Dimension::fixed(f.slots),g::Dimension::fixed(f.depth)};
      a=take(b.add(take(g::ViewSpec::broadcast_to(shape),"shared A"),a),"zero-slot A");
    }
    for(std::size_t i=1;i<f.port_count();++i) ports[i]=take(dynamic||i==3?b.input(f.descriptions[i]):b.constant(f.descriptions[i]),"original W4A8 port");
    output=f.grouped()?take(b.add(c::model_ops::grouped_blockwise_w4a8_linear(),a,ports[1],ports[2],ports[3]),"grouped W4A8")
        :take(b.add(c::model_ops::blockwise_w4a8_linear(),a,ports[1],ports[2]),"dense W4A8");
    check(b.publish(output),"publish W4A8");graph=take(std::move(b).finish(),"finish W4A8");
  }
};
template<class Compile,class Write,class Read,class Prepare>
void run_lifetimes(m::MemoryResourceRef resource,std::span<const g::QueueBinding> queues,
    Compile compile,Write write,Read read,Prepare prepare,c::StorageOwner owner={}) {
  oracle_checks();unsigned submissions=0;
  for(const auto layout:{Layout::kDense,Layout::kShared,Layout::kPerSlot,Layout::kSharedView}) {
      SCOPED_TRACE(::testing::Message()<<"layout="<<int(layout));
      Case f(layout,layout==Layout::kSharedView?17:3,
             1536);Logical logical(f);
      const auto original=f.data();const auto payload=original.payloads();
      std::array<c::ReadLease,2> ready;std::array<g::ConstantBinding,2> constants;
      for(std::size_t i=0;i<2;++i) {
        auto buffer=take(m::Buffer::create(resource,payload[i+1].size(),256),"original W4A8 bank");check(write(buffer.view(),payload[i+1]),"upload original W4A8 bank");
        ready[i]=owner?take(owner.adopt_buffer(std::move(buffer)),"owned W4A8 bank"):take(c::ReadLease::adopt(std::move(buffer)),"W4A8 bank lease");
        constants[i]={logical.ports[i+1],take(ready[i].tensor(f.descriptions[i+1]),"typed original bank"),g::ConstantPolicy::kMaterialize,owner,owner?ready[i]:c::ReadLease{}};
      }
      auto compiled=compile(logical.graph,g::CompileOptions{.constants=constants});ASSERT_TRUE(compiled)<<compiled.error().message();
      const auto ownership=take(compiled->executable.output_ownership(),"W4A8 output ownership");
      if(ownership!=g::OutputOwnership::kCaller) {ASSERT_EQ(ownership,g::OutputOwnership::kInvocation);}
      std::array runs{take(compiled->executable.create_invocation({.resources=g::ResourceOptions{}}),"W4A8 slot0"),take(compiled->executable.create_invocation({.resources=g::ResourceOptions{}}),"W4A8 slot1")};
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
        check(prepare(runs[slot],queues),"prepare W4A8 replay");
      }
      std::array<g::GraphOutput,2> old;std::array<std::vector<std::uint16_t>,2> expected;
      for(unsigned round=0;round<2;++round)for(std::size_t slot=0;slot<2;++slot) {
        if(round) {storage::expect_result(old[slot],expected[slot],read);EXPECT_FALSE(runs[slot].submit(queues));old[slot].reset();}
        const auto data=f.data(round*2+static_cast<unsigned>(slot));check(write(inputs[slot][0].view(),std::as_bytes(std::span(data.a))),"update current A");
        if(f.grouped())check(write(inputs[slot][1].view(),std::as_bytes(std::span(data.ids))),"update current IDs");
        auto done=take(runs[slot].submit(queues),"W4A8 submit");check(done.wait(),"W4A8 completion");++submissions;
        old[slot]=take(runs[slot].result(logical.output),"W4A8 result");expected[slot]=f.oracle(data);storage::expect_result(old[slot],expected[slot],read);
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
  for(const auto layout:{Layout::kDense,Layout::kSharedView}) {
    SCOPED_TRACE(::testing::Message()<<"layout="<<int(layout));
    Case f(layout);Logical logical(f,true);
    auto compiled=compile(logical.graph,g::CompileOptions{});ASSERT_TRUE(compiled)<<compiled.error().message();
    auto executable=compiled->executable;
    auto run=take(executable.create_invocation({.resources=g::ResourceOptions{}}),"dynamic W4A8 slot");
    const auto ownership=take(executable.output_ownership(),"W4A8 ownership");
    if(ownership!=g::OutputOwnership::kCaller) ASSERT_EQ(ownership,g::OutputOwnership::kInvocation);
    const auto valid=f.data();const auto initial=valid.payloads();std::array<m::Buffer,4> inputs;
    for(std::size_t i=0;i<f.port_count();++i) inputs[i]=take(m::Buffer::create(resource,initial[i].size(),256),"original dynamic W4A8 input");
    const auto count=f.output_description.required_bytes().value();c::LeasePool pool;
    if(ownership==g::OutputOwnership::kCaller) pool=take(c::LeasePool::create(resource,(count+255)/256*256,{.max_blocks=1}),"W4A8 writer");
    const auto upload=[&](const Data &d) {const auto p=d.payloads();for(std::size_t i=0;i<f.port_count();++i)check(write(inputs[i].view(),p[i]),"update original bytes");};
    const auto bind=[&] {
      for(std::size_t i=0;i<f.port_count();++i) check(run.bind(logical.ports[i],take(c::ConstTensorView::create(i?f.descriptions[i]:f.activation_backing,inputs[i].view()),"W4A8 current view")),"bind original buffers");
      if(ownership==g::OutputOwnership::kCaller) check(run.bind(logical.output,f.output_description,take(pool.allocate(count),"W4A8 output lease")),"bind output");
      check(prepare(run,queues),"prepare W4A8 replay");
    };
    upload(valid);bind();compiled=g::CompiledGraph{};logical.graph={};
    const auto success=[&](const Data &d) {
      upload(d);auto done=take(run.submit(queues),"W4A8 submit");check(done.wait(),"valid W4A8 completion");
      auto result=take(run.result(logical.output),"W4A8 result");storage::expect_result(result,f.oracle(d),read);++successes;
    };
    success(valid); // Unselected NaN SF and NaN token padding stay unread.
    std::vector<std::pair<std::string,Data>> invalid;
    for(const auto bits:std::array<std::uint16_t,3>{0x7f80,0xff80,0x7fc1}) {
      auto d=valid;d.a[0]=bits;invalid.emplace_back("nonfinite source A",d);
    }
    for(const auto sf:{0.F,-1.F,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
      auto d=valid;d.scales[0]=sf;invalid.emplace_back("invalid original selected SF",d);
    }
    for(const auto bits:std::array<std::uint16_t,2>{0x0080,0x0304}) {
      auto d=valid;std::fill(d.a.begin(),d.a.end(),0);d.a[0]=bits;
      invalid.emplace_back("undefined derived I8 cast",d);
    }
    if(f.grouped()) for(const auto id:{-1,static_cast<int>(f.experts)}) {
      auto d=valid;d.ids.back()=id;invalid.emplace_back("invalid last ID",d);
    }
    for(const auto &[label,bad]:invalid) {
      SCOPED_TRACE(label);upload(bad);
      {auto done=run.submit(queues);ASSERT_TRUE(done)<<done.error().message();EXPECT_FALSE(done->wait().ok());EXPECT_FALSE(run.result(logical.output));}
      EXPECT_FALSE(run.submit(queues))<<"failed Invocation is terminal";++failures;
      run={};run=take(executable.create_invocation({.resources=g::ResourceOptions{}}),"fresh recovery slot");upload(valid);bind();success(valid);
    }
    // Published normal witness distinguishes INT8 truncation from W4A16.
    auto d=valid;std::fill(d.a.begin(),d.a.end(),0);std::fill(d.w.begin(),d.w.end(),0x88888888u);
    for(std::int64_t t=0;t<f.tokens;++t) {d.a[f.a_index(t,0,0)]=0x3f80;d.a[f.a_index(t,0,1)]=0x4000;}
    for(std::int64_t e=0;e<(f.grouped()?2:1);++e)for(std::int64_t n=0;n<f.columns;++n) {
      const auto row=std::size_t(e*f.columns+n);d.w[row*std::size_t(f.depth/8)]=0x88888889u;
      for(std::int64_t g=0;g<f.depth/512;++g)d.scales[row*std::size_t(f.depth/512)+std::size_t(g)]=1.F;
    }
    for(const auto v:f.oracle(d))ASSERT_EQ(v,0x3f7eu);success(d);
    // Zero scale, subnormal BF16 scale, and 127.5 ->127 are defined cases.
    for(const auto bits:std::array<std::uint16_t,8>{0,0x8000,0x0001,0x8001,0x0381,0x8380,0x8304,0x7f7f}) {
      auto next=d;std::fill(next.a.begin(),next.a.end(),0);
      for(std::int64_t t=0;t<f.tokens;++t)next.a[f.a_index(t,0,0)]=bits;
      success(next);
    }
    // SF itself is finite positive; only its BF16 conversion over/underflows.
    // The smallest positive BF16 SF also probes subnormal output arithmetic.
    for(const auto sf:{std::numeric_limits<float>::denorm_min(),storage::wide(1),std::numeric_limits<float>::max()}) {
      auto next=d;
      for(std::int64_t e=0;e<(f.grouped()?2:1);++e)for(std::int64_t n=0;n<f.columns;++n)
        next.scales[std::size_t((e*f.columns+n)*(f.depth/512))]=sf;
      success(next);
    }
  }
  ::testing::Test::RecordProperty("successful_submissions",successes);
  ::testing::Test::RecordProperty("failed_completions",failures);
}
} // namespace blockwise_w4a8_test
