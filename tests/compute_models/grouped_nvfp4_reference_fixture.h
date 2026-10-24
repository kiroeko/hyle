#pragma once

#include "nvfp4_reference_fixture.h"

namespace grouped_nvfp4_reference_test {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
namespace scalar = nvfp4_reference_test;
using Type = c::DataType;
using scalar::take;
using scalar::check;
using scalar::dense;
using scalar::bytes;

enum class Activation { kShared, kPerSlot, kSharedView };
struct Data final {
  std::vector<std::uint16_t> a;
  std::vector<std::uint8_t> w, scales;
  std::vector<float> wg, ig;
  std::vector<std::int32_t> ids;
  std::array<std::vector<std::byte>,6> payloads() const {
    return {bytes<std::uint16_t>(a), bytes<std::uint8_t>(w), bytes<std::uint8_t>(scales),
        bytes<float>(wg), bytes<float>(ig), bytes<std::int32_t>(ids)};
  }
};

// Geometry and independent scalar oracle only: Direct tests need no Graph
// object or Graph target dependency. Scalar math reuses the existing test's
// enumerated E2M1/E4M3 table and explicit F32/BF16 rounding, never product code.
struct Case final {
  static constexpr std::int64_t tokens = 2, slots = 3;
  Activation layout;
  std::int64_t experts, columns = 5, depth = 32;
  std::array<c::TensorDesc,6> descriptions;
  c::TensorDesc activation_backing, output_description;
  explicit Case(Activation selected = Activation::kShared, std::int64_t e = 3)
      : layout(selected), experts(e) {
    const bool per_slot = layout == Activation::kPerSlot;
    activation_backing = dense(Type::kBFloat16,
        {tokens, per_slot ? slots : 1, depth + (layout == Activation::kSharedView ? 16 : 0)});
    descriptions = {layout == Activation::kSharedView
        ? take(c::TensorDesc::strided_bytes(Type::kBFloat16, std::array{tokens,slots,depth},
            std::array<std::int64_t,3>{(depth+16)*2,0,2}), "shared physical A")
        : activation_backing,
        dense(Type::kUInt8,{experts,columns,depth/2}),
        dense(Type::kFloat8E4M3FN,{experts,columns,depth/16}),
        dense(Type::kFloat32,{experts}), dense(Type::kFloat32,{experts}),
        dense(Type::kInt32,{tokens,slots})};
    output_description = dense(Type::kBFloat16,{tokens,slots,columns});
  }
  std::size_t a_index(std::int64_t token, std::int64_t slot, std::int64_t k) const {
    return std::size_t(token * (descriptions[0].byte_strides()[0]/2) +
        (descriptions[0].dimensions()[1] == 1 ? 0 : slot * (descriptions[0].byte_strides()[1]/2)) + k);
  }
  Data data(unsigned phase = 0) const {
    Data d;
    d.a.resize(activation_backing.required_bytes().value()/2, 0x7fc1); // NaN padding is never an operand.
    d.w.resize(std::size_t(experts*columns*(depth/2)));
    d.scales.resize(std::size_t(experts*columns*(depth/16)), 0x7f);
    d.wg.resize(std::size_t(experts), std::numeric_limits<float>::quiet_NaN());
    d.ig = d.wg;
    d.wg[0] = d.wg[1] = 1.00390625F;
    d.ig[0] = .1F; d.ig[1] = .3F;
    constexpr std::array<float,16> values{.25F,.75F,1.25F,1.75F,2.5F,3.5F,5.F,6.F,
        -.25F,-.75F,-1.25F,-1.75F,-2.5F,-3.5F,-5.F,-6.F};
    for (std::int64_t t = 0; t < tokens; ++t) {
      for (std::int64_t s = 0; s < (layout == Activation::kPerSlot ? slots : 1); ++s) {
        for (std::int64_t k = 0; k < depth; ++k)
          d.a[a_index(t,s,k)] = k < 16
              ? scalar::bf16(values[(std::size_t(k+t*3+s*2)+phase*3)%values.size()])
              : scalar::bf16(k == 16 ? 1.e-6F : k == 17 ? -1.e-6F : 0.F);
      }
    }
    for (std::int64_t e = 0; e < 2; ++e) for (std::int64_t n = 0; n < columns; ++n) {
      for (std::int64_t group = 0; group < depth/16; ++group)
        d.scales[std::size_t((e*columns+n)*(depth/16)+group)] = static_cast<std::uint8_t>(0x39+(n+group)%3);
      const auto k = n == 3 ? 17 : n;
      const unsigned code = n == 0 ? 1u : n == 1 ? 7u : 3u;
      d.w[std::size_t((e*columns+n)*(depth/2)+k/2)] = static_cast<std::uint8_t>(code << (4*(k%2)));
    }
    for (std::int64_t t = 0; t < tokens; ++t) for (std::int64_t s = 0; s < slots; ++s)
      d.ids.push_back(static_cast<std::int32_t>((t+s+phase)%2));
    return d;
  }
  std::vector<std::uint16_t> oracle(const Data &d, bool incorrectly_share_ig = false) const {
    std::vector<std::uint16_t> result(std::size_t(tokens*slots*columns));
    for (std::int64_t t = 0; t < tokens; ++t) for (std::int64_t s = 0; s < slots; ++s) {
      const auto selected = std::size_t(d.ids[std::size_t(t*slots+s)]);
      std::vector<float> a(std::size_t(depth), 0.F);
      for (std::int64_t group = 0; group < depth/16; ++group) {
        float maximum = 0.F;
        for (std::int64_t lane = 0; lane < 16; ++lane)
          maximum = std::max(maximum,std::abs(scalar::wide(d.a[a_index(t,s,group*16+lane)])));
        for (std::int64_t lane = 0; lane < 16; ++lane) {
          const auto k = group*16+lane;
          a[std::size_t(k)] = scalar::activation_value(scalar::wide(d.a[a_index(t,s,k)]),maximum,
              d.ig[incorrectly_share_ig ? 0 : selected]);
        }
      }
      for (std::int64_t n = 0; n < columns; ++n) {
        float sum = 0.F;
        for (std::int64_t k = 0; k < depth; ++k) {
          const auto packed = d.w[(selected*std::size_t(columns)+std::size_t(n))*std::size_t(depth/2)+std::size_t(k/2)];
          const auto code = (packed >> (4*(k%2))) & 15u;
          const auto sf = d.scales[(selected*std::size_t(columns)+std::size_t(n))*std::size_t(depth/16)+std::size_t(k/16)];
          const auto w = scalar::weight_value(code,sf,d.wg[selected]);
          sum = scalar::f32(double(sum)+double(a[std::size_t(k)])*double(w));
        }
        result[std::size_t((t*slots+s)*columns+n)] = scalar::bf16(sum);
      }
    }
    return result;
  }
};

struct Logical final {
  std::array<g::ValueId,6> ports{};
  g::ValueId output{};
  g::Graph graph;
  Logical(const Case &f, bool dynamic_banks = false) {
    auto b = take(g::GraphBuilder::create(), "grouped GraphBuilder");
    ports[0] = take(b.input(f.activation_backing), "original dense activation backing");
    auto a = ports[0];
    if (f.layout == Activation::kSharedView) {
      a = take(b.add(take(g::ViewSpec::slice(2,g::Dimension::fixed(0),g::Dimension::fixed(f.depth)),"slice actual token padding"),a),"shared sliced activation");
      const std::array shape{g::Dimension::fixed(f.tokens),g::Dimension::fixed(f.slots),g::Dimension::fixed(f.depth)};
      a = take(b.add(take(g::ViewSpec::broadcast_to(shape),"zero-slot broadcast"),a),"actual shared view");
    }
    for (std::size_t i = 1; i < ports.size(); ++i)
      ports[i] = take(i == 5 || dynamic_banks ? b.input(f.descriptions[i]) : b.constant(f.descriptions[i]), "grouped original port");
    output = take(b.add(c::model_ops::grouped_nvfp4_linear(),a,ports[1],ports[2],ports[3],ports[4],ports[5]),"grouped reference operation");
    check(b.publish(output),"publish only grouped result");
    graph = take(std::move(b).finish(),"finish grouped reference graph");
  }
};

template<class Compile,class Write,class Read,class Prepare>
void run_lifetimes(m::MemoryResourceRef resource,std::span<const g::QueueBinding> queues,
    Compile compile,Write write,Read read,Prepare prepare,c::StorageOwner owner = {}) {
  scalar::oracle_checks();
  unsigned submissions = 0;
  for (const auto layout : {Activation::kShared,Activation::kPerSlot,Activation::kSharedView}) {
    for (const auto experts : {3LL,17LL}) {
      SCOPED_TRACE(::testing::Message() << "layout=" << static_cast<int>(layout) << " E=" << experts);
      Case f(layout,experts); Logical logical(f);
      const auto original = f.data(); const auto payload = original.payloads();
      ASSERT_NE(f.oracle(original),f.oracle(original,true)) << "different IG must change slot QDQ";
      std::array<c::ReadLease,4> ready;
      std::array<g::ConstantBinding,4> constants;
      for (std::size_t i = 0; i < ready.size(); ++i) {
        auto buffer = take(m::Buffer::create(resource,payload[i+1].size(),256),"grouped original bank");
        check(write(buffer.view(),payload[i+1]),"write grouped original bank");
        ready[i] = owner ? take(owner.adopt_buffer(std::move(buffer)),"owned grouped bank")
            : take(c::ReadLease::adopt(std::move(buffer)),"grouped bank lease");
        constants[i] = {logical.ports[i+1],take(ready[i].tensor(f.descriptions[i+1]),"typed grouped constant"),
            g::ConstantPolicy::kMaterialize,owner,owner ? ready[i] : c::ReadLease{}};
      }
      auto compiled = compile(logical.graph,g::CompileOptions{.constants=constants});
      ASSERT_TRUE(compiled) << compiled.error().message();
      EXPECT_EQ(compiled->executable.input_count(),2u);
      const auto ownership = take(compiled->executable.output_ownership(),"grouped output ownership");
      const bool caller = ownership == g::OutputOwnership::kCaller;
      if (!caller) { ASSERT_EQ(ownership,g::OutputOwnership::kInvocation); }
      std::array<g::Invocation,2> runs{
          take(compiled->executable.create_invocation({.resources=g::ResourceOptions{}}),"first grouped slot"),
          take(compiled->executable.create_invocation({.resources=g::ResourceOptions{}}),"second grouped slot")};
      constants={};ready={};compiled=g::CompiledGraph{};logical.graph=g::Graph{};
      const auto output_bytes=f.output_description.required_bytes().value();
      c::LeasePool output_pool;
      if (caller) output_pool=take(c::LeasePool::create(resource,2*((output_bytes+255)/256*256),{.max_blocks=2}),"grouped output pool");
      std::array<std::array<m::Buffer,2>,2> inputs;
      for (std::size_t slot=0;slot<runs.size();++slot) {
        for (std::size_t i=0;i<2;++i) {
          const auto port=i ? 5u : 0u;
          inputs[slot][i]=take(m::Buffer::create(resource,payload[port].size(),256),"dynamic grouped payload");
          check(write(inputs[slot][i].view(),payload[port]),"initial grouped input");
          const auto desc=i ? f.descriptions[5] : f.activation_backing;
          check(runs[slot].bind(logical.ports[port],take(c::ConstTensorView::create(desc,inputs[slot][i].view()),"grouped input view")),"bind grouped input once");
        }
        if (caller) check(runs[slot].bind(logical.output,f.output_description,take(output_pool.allocate(output_bytes),"grouped output writer")),"bind grouped result");
        check(prepare(runs[slot],queues),"prepare original grouped slot");
      }
      std::array<g::GraphOutput,2> retained;
      std::array<std::vector<std::uint16_t>,2> old;
      for (unsigned repeat=0;repeat<2;++repeat) for (std::size_t slot=0;slot<runs.size();++slot) {
        if (repeat) {
          scalar::expect_result(retained[slot],old[slot],read);
          EXPECT_FALSE(runs[slot].submit(queues));
          retained[slot].reset();
        }
        const auto data=f.data(repeat*2+static_cast<unsigned>(slot));
        check(write(inputs[slot][0].view(),std::as_bytes(std::span(data.a))),"update original A");
        check(write(inputs[slot][1].view(),std::as_bytes(std::span(data.ids))),"update checked IDs");
        auto done=runs[slot].submit(queues);ASSERT_TRUE(done)<<done.error().message();
        const auto status=done->wait();ASSERT_TRUE(status.ok())<<status.message();++submissions;
        retained[slot]=take(runs[slot].result(logical.output),"grouped result");
        old[slot]=f.oracle(data);scalar::expect_result(retained[slot],old[slot],read);
        if (retained[1-slot].valid()) scalar::expect_result(retained[1-slot],old[1-slot],read);
      }
      EXPECT_NE(f.oracle(f.data(0)),f.oracle(f.data(2)));
      runs={};
      for (std::size_t slot=0;slot<retained.size();++slot) scalar::expect_result(retained[slot],old[slot],read);
      retained={};
      if (caller) { EXPECT_EQ(output_pool.stats().live_blocks,0u); }
    }
  }
  ::testing::Test::RecordProperty("actual_submissions",submissions);
}

template<class Compile,class Write,class Read,class Prepare>
void run_runtime_domain(m::MemoryResourceRef resource,std::span<const g::QueueBinding> queues,
    Compile compile,Write write,Read read,Prepare prepare) {
  Case f(Activation::kSharedView);Logical logical(f,true);
  auto compiled=compile(logical.graph,g::CompileOptions{});ASSERT_TRUE(compiled)<<compiled.error().message();
  EXPECT_EQ(compiled->executable.input_count(),6u);
  auto executable=compiled->executable;
  auto invocation=take(executable.create_invocation({.resources=g::ResourceOptions{}}),"dynamic grouped slot");
  const auto ownership=take(compiled->executable.output_ownership(),"dynamic grouped ownership");
  if (ownership!=g::OutputOwnership::kCaller) { ASSERT_EQ(ownership,g::OutputOwnership::kInvocation); }
  const auto valid=f.data();const auto initial=valid.payloads();
  std::array<m::Buffer,6> inputs;
  for(std::size_t i=0;i<inputs.size();++i) inputs[i]=take(m::Buffer::create(resource,initial[i].size(),256),"dynamic grouped input");
  c::LeasePool outputs;
  if(ownership==g::OutputOwnership::kCaller) {
    const auto bytes=f.output_description.required_bytes().value();
    outputs=take(c::LeasePool::create(resource,(bytes+255)/256*256,{.max_blocks=1}),"dynamic grouped output");
  }
  const auto upload=[&](const Data &d) {
    const auto payload=d.payloads();
    for(std::size_t i=0;i<inputs.size();++i) check(write(inputs[i].view(),payload[i]),"update grouped runtime port");
  };
  const auto bind=[&] {
    for(std::size_t i=0;i<inputs.size();++i) check(invocation.bind(logical.ports[i],take(c::ConstTensorView::create(i ? f.descriptions[i] : f.activation_backing,inputs[i].view()),"runtime grouped view")),"bind dynamic grouped port");
    if(ownership==g::OutputOwnership::kCaller) check(invocation.bind(logical.output,f.output_description,take(outputs.allocate(f.output_description.required_bytes().value()),"runtime output writer")),"bind grouped output");
    check(prepare(invocation,queues),"prepare grouped runtime slot");
  };
  upload(valid);bind();compiled=g::CompiledGraph{};logical.graph=g::Graph{};
  unsigned successes=0,failures=0;
  const auto success=[&](const Data &d) {
    upload(d);auto done=take(invocation.submit(queues),"grouped accepted submit");
    check(done.wait(),"grouped runtime Completion");
    auto result=take(invocation.result(logical.output),"successful grouped output");
    scalar::expect_result(result,f.oracle(d),read);++successes;
  };
  success(valid); // Unselected experts already contain invalid WG/IG/SF.
  std::vector<std::pair<std::string,Data>> invalid;
  for(const auto id:std::array<std::int32_t,2>{-1,static_cast<std::int32_t>(f.experts)}) {
    auto d=valid;d.ids.back()=id;invalid.emplace_back("invalid final expert ID",std::move(d));
  }
  for(const auto value:std::array<std::uint16_t,3>{0x7fc1,0x7f80,0xff80}) {
    auto d=valid;d.a[0]=value;invalid.emplace_back("nonfinite selected A",std::move(d));
  }
  for(const auto value:std::array<std::uint8_t,3>{0x7f,0xff,0xb8}) {
    auto d=valid;d.scales[0]=value;invalid.emplace_back("invalid selected SF",std::move(d));
  }
  for(const auto value:std::array<float,6>{0.F,-0.F,-1.F,std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
    auto d=valid;d.wg[0]=value;invalid.emplace_back("invalid selected WG",std::move(d));
    d=valid;d.ig[0]=value;invalid.emplace_back("invalid selected IG",std::move(d));
  }
  for(const auto &[label,d]:invalid) {
    SCOPED_TRACE(label);upload(d);
    { auto done=invocation.submit(queues);ASSERT_TRUE(done)<<done.error().message();
      EXPECT_FALSE(done->wait().ok());EXPECT_FALSE(invocation.result(logical.output)); }
    EXPECT_FALSE(invocation.submit(queues))<<"failed Graph invocation is terminal";++failures;
    invocation={};invocation=take(executable.create_invocation({.resources=g::ResourceOptions{}}),"fresh grouped recovery slot");
    upload(valid);bind();success(valid);
  }
  {
    auto d=valid;d.ig[0]=std::numeric_limits<float>::denorm_min();success(d);
    d=valid;d.wg[0]=std::numeric_limits<float>::denorm_min();success(d);
  }
  {
    auto d=valid;std::fill(d.a.begin(),d.a.end(),0);
    for(std::int64_t t=0;t<f.tokens;++t)d.a[f.a_index(t,0,0)]=0x4380;
    std::fill(d.w.begin(),d.w.end(),0);
    for(std::int64_t e=0;e<2;++e) for(std::int64_t n=0;n<f.columns;++n)
      d.w[std::size_t((e*f.columns+n)*(f.depth/2))]=2;
    std::fill(d.scales.begin(),d.scales.end(),0x38);
    for(std::size_t e=0;e<2;++e) {d.wg[e]=std::bit_cast<float>(std::uint32_t{0x00080000});d.ig[e]=1.F;}
    for(const auto v:f.oracle(d)) { ASSERT_EQ(v,0x0284u); }
    success(d);
  }
  {
    auto d=valid;for(std::size_t i=0;i<d.scales.size();++i)d.scales[i]=i%2 ? 0x80 : 0;
    success(d);
  }
  {
    auto d=valid;std::fill(d.w.begin(),d.w.end(),0);std::fill(d.scales.begin(),d.scales.end(),0x40);
    d.wg[0]=d.wg[1]=std::numeric_limits<float>::max();
    for(const auto v:f.oracle(d)) { ASSERT_TRUE(std::isnan(scalar::wide(v))); }
    success(d);
  }
  ::testing::Test::RecordProperty("successful_submissions",successes);
  ::testing::Test::RecordProperty("failed_completions",failures);
}
} // namespace grouped_nvfp4_reference_test
