#pragma once

#include <cstring>
#include "nvfp4_reference_fixture.h"

namespace grouped_fp8_power2_test {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
namespace scalar = nvfp4_reference_test;
using scalar::take;
using scalar::check;
using scalar::dense;
using Type = c::DataType;
using Policy = c::model_ops::Float8ScalePolicy;
static_assert(sizeof(c::model_ops::GroupedFloat8Linear) == 8);

struct Data final {
  std::vector<std::uint16_t> a;
  std::vector<std::int32_t> ids;
};

struct Case final {
  static constexpr std::int64_t tokens=2, slots=3, depth=256, columns=129, experts=3;
  bool shared;
  Policy scale_policy;
  float floor;
  std::array<c::TensorDesc,3> descriptions;
  c::TensorDesc output_description;
  std::vector<std::byte> weights;
  g::ValueId weight_id{}, activation_id{}, ids_id{}, output_id{};
  g::Graph graph;

  explicit Case(bool share=true, Policy policy=Policy::kPowerOfTwo)
      : shared(share), scale_policy(policy), floor(policy==Policy::kPowerOfTwo?1.e-4F:1.e-10F) {
    descriptions={take(c::float8_grouped_weight_description(experts,columns,depth),"FP8 bank"),
      dense(Type::kBFloat16,{tokens,shared?1:slots,depth}),dense(Type::kInt32,{tokens,slots})};
    output_description=dense(Type::kBFloat16,{tokens,slots,columns});
    weights.resize(descriptions[0].required_bytes().value(),std::byte{0});
    const auto payload=std::size_t(experts*columns*depth);
    for (std::int64_t e=0;e<experts;++e) {
      for (std::int64_t n=0;n<columns;++n) {
        if (e==2) { // A valid but unselected expert is never decoded/read.
          std::fill_n(weights.begin()+std::ptrdiff_t((e*columns+n)*depth),std::size_t(depth),std::byte{0x7f});
        } else for (std::int64_t group=0;group<2;++group) {
          if ((n%3==0 && group==1) || (n%3==1 && group==0)) continue;
          weights[std::size_t((e*columns+n)*depth+group*128+1)]=
              std::byte(e==0?0x38:0xb0); // +1 or -0.5, only the secondary A lane.
        }
      }
      for (std::int64_t block=0;block<2;++block) for (std::int64_t group=0;group<2;++group) {
        constexpr std::array<float,4> values{1.F,2.F,.5F,4.F};
        const auto value=e==2?std::numeric_limits<float>::quiet_NaN():values[std::size_t((e*2+block+group)%4)];
        std::memcpy(weights.data()+payload+std::size_t((e*2+block)*2+group)*4,&value,4);
      }
    }
    auto b=take(g::GraphBuilder::create(),"grouped power builder");
    weight_id=take(b.constant(descriptions[0]),"original encoded bank");
    activation_id=take(b.input(descriptions[1]),"BF16 activation");
    ids_id=take(b.input(descriptions[2]),"current selected IDs");
    const auto operation=scale_policy==Policy::kDivide?c::model_ops::grouped_float8_linear()
        :c::model_ops::grouped_float8_linear(scale_policy,floor);
    output_id=take(b.add(operation,weight_id,activation_id,ids_id),"grouped FP8");
    check(b.publish(output_id),"publish grouped FP8");
    graph=take(std::move(b).finish(),"grouped power graph");
  }
  Data data(unsigned phase=0) const {
    const auto activation_slots=shared?1:slots;
    Data d;d.a.resize(std::size_t(tokens*activation_slots*depth),0);
    for (std::int64_t row=0;row<tokens*activation_slots;++row) for (std::int64_t group=0;group<2;++group) {
      const auto pattern=(std::size_t(row+group*2)+phase*2)%6;
      constexpr std::array<float,6> maxima{0.F,1.F,446.F,448.F,450.F,1.e-30F};
      constexpr std::array<float,6> secondary{0.F,.40625F,1.F/512.F,.40625F,1.F/512.F,5.e-31F};
      const auto base=std::size_t(row*depth+group*128);
      d.a[base]=scalar::bf16(maxima[pattern]);
      const auto entry=scale_policy==Policy::kMultiplyReciprocal&&pattern==4?
          .006866455078125F:secondary[pattern];
      d.a[base+1]=scalar::bf16(phase%2?-entry:entry);
      d.a[base+7]=0x8000; // Explicit negative zero remains a finite operand.
    }
    for (std::int64_t t=0;t<tokens;++t) for (std::int64_t s=0;s<slots;++s)
      d.ids.push_back(std::int32_t((std::size_t(t+s)+phase)%2));
    return d;
  }
  // Mathematical ceil via frexp is independent of the producer's bit ladder.
  static float scale(float maximum, float minimum, Policy policy, bool wrong_bfloat=false) {
    maximum=std::max(maximum,minimum);
    if (policy==Policy::kDivide) return scalar::f32(double(maximum)/448.);
    const auto reciprocal=scalar::f32(1./448.);
    // A BF16 scalar reciprocal plus BF16 arithmetic is a different quantizer:
    // at amax450 it can round the pre-ceil scale to1 instead of exceeding1.
    const auto raw=wrong_bfloat?scalar::rounded_bf16(scalar::multiply(
        scalar::rounded_bf16(maximum),scalar::rounded_bf16(reciprocal)))
        :scalar::multiply(maximum,reciprocal);
    if (policy==Policy::kMultiplyReciprocal) return raw;
    int exponent=0;const auto fraction=std::frexp(raw,&exponent);
    return std::ldexp(1.F,fraction==.5F?exponent-1:exponent);
  }
  std::vector<std::uint16_t> oracle(const Data &d, bool wrong_bfloat=false,
                                  bool wrong_none=false, bool wrong_floor=false) const {
    const auto rows=tokens*(shared?1:slots);
    std::vector<float> codes(d.a.size()), scales(std::size_t(rows*2),0.F);
    for (std::int64_t row=0;row<rows;++row) for (std::int64_t group=0;group<2;++group) {
      const auto base=std::size_t(row*depth+group*128);float maximum=0;
      for (std::size_t k=0;k<128;++k) maximum=std::max(maximum,std::abs(scalar::wide(d.a[base+k])));
      const auto sa=scale(maximum,wrong_floor?1.e-38F:floor,wrong_none?Policy::kDivide:scale_policy,wrong_bfloat);
      scales[std::size_t(row*2+group)]=sa;
      for (std::size_t k=0;k<128;++k) {
        const auto normalized=scale_policy==Policy::kMultiplyReciprocal&&!wrong_none
            ?scalar::multiply(scalar::wide(d.a[base+k]),scalar::f32(1./double(sa)))
            :scalar::f32(double(scalar::wide(d.a[base+k]))/double(sa));
        const auto magnitude=scalar::encode_scale(std::abs(normalized));
        codes[base+k]=scalar::e4m3(std::uint8_t(magnitude|(std::signbit(normalized)?0x80u:0)));
      }
    }
    const auto payload=std::size_t(experts*columns*depth);
    std::vector<std::uint16_t> y(std::size_t(tokens*slots*columns));
    for (std::int64_t t=0;t<tokens;++t) for (std::int64_t s=0;s<slots;++s) for (std::int64_t n=0;n<columns;++n) {
      const auto expert=d.ids[std::size_t(t*slots+s)];
      const auto row=t*(shared?1:slots)+(shared?0:s);float sum=0;
      for (std::int64_t group=0;group<2;++group) {
        float dot=0;
        for (std::int64_t k=0;k<128;++k) {
          const auto w=std::to_integer<std::uint8_t>(weights[std::size_t((expert*columns+n)*depth+group*128+k)]);
          dot=scalar::f32(double(dot)+double(codes[std::size_t(row*depth+group*128+k)])*double(scalar::e4m3(w)));
        }
        float sw=0;std::memcpy(&sw,weights.data()+payload+std::size_t((expert*2+n/128)*2+group)*4,4);
        const auto scaled=scalar::multiply(scalar::multiply(dot,scales[std::size_t(row*2+group)]),sw);
        sum=scalar::f32(double(sum)+double(scaled));
      }
      y[std::size_t((t*slots+s)*columns+n)]=scalar::bf16(sum);
    }
    return y;
  }
};

template<class Write>
std::pair<c::ReadLease,g::ConstantBinding> constant(const Case &f,m::MemoryResourceRef resource,
                                                  Write write,c::StorageOwner owner) {
  auto buffer=take(m::Buffer::create(resource,f.weights.size(),256),"encoded constant buffer");
  check(write(buffer.view(),f.weights),"upload original encoded bytes");
  auto lease=owner?take(owner.adopt_buffer(std::move(buffer)),"owned encoded bank")
                  :take(c::ReadLease::adopt(std::move(buffer)),"encoded bank lease");
  g::ConstantBinding binding{f.weight_id,take(lease.tensor(f.descriptions[0]),"encoded constant view"),
    g::ConstantPolicy::kMaterialize,owner,owner?lease:c::ReadLease{}};
  return {std::move(lease),std::move(binding)};
}

template<class Compile,class Write,class Read,class Prepare>
void run_with_policy(m::MemoryResourceRef resource,std::span<const g::QueueBinding> queues,
         Compile compile,Write write,Read read,Prepare prepare,c::StorageOwner owner,Policy policy) {
  unsigned submissions=0;
  for (const bool shared:{true,false}) {
    SCOPED_TRACE(::testing::Message()<<"shared="<<shared<<" policy="<<unsigned(policy));
    Case f(shared,policy);
    if (policy==Policy::kPowerOfTwo) {
      bool different_policy=false,bfloat=false,floor=false;
      for (unsigned phase=0;phase<4;++phase) {
        const auto d=f.data(phase);const auto y=f.oracle(d);
        different_policy|=y!=f.oracle(d,false,true);bfloat|=y!=f.oracle(d,true);floor|=y!=f.oracle(d,false,false,true);
      }
      ASSERT_TRUE(different_policy);ASSERT_TRUE(bfloat);ASSERT_TRUE(floor);
    }
    if (policy==Policy::kMultiplyReciprocal) {
      const auto d=f.data(2);const auto y=f.oracle(d);const auto wrong=f.oracle(d,false,true);
      ASSERT_NE(y,wrong);ASSERT_EQ(y[0],0x3bc1U);ASSERT_EQ(wrong[0],0x3c01U);
    }
    auto bound=constant(f,resource,write,owner);
    auto compiled=compile(f.graph,g::CompileOptions{.constants=std::span(&bound.second,1)});
    ASSERT_TRUE(compiled)<<compiled.error().message();
    const auto ownership=take(compiled->executable.output_ownership(),"grouped output ownership");
    ASSERT_TRUE(ownership==g::OutputOwnership::kCaller || ownership==g::OutputOwnership::kInvocation);
    std::array runs{take(compiled->executable.create_invocation({.resources=g::ResourceOptions{}}),"power slot0"),
                    take(compiled->executable.create_invocation({.resources=g::ResourceOptions{}}),"power slot1")};
    bound={};compiled=g::CompiledGraph{};
    const auto size=f.output_description.required_bytes().value();c::LeasePool pool;
    if (ownership==g::OutputOwnership::kCaller)
      pool=take(c::LeasePool::create(resource,2*((size+255)/256*256),{.max_blocks=2}),"two power outputs");
    std::array<std::array<m::Buffer,2>,2> inputs;
    for (std::size_t slot=0;slot<2;++slot) {
      for (std::size_t i=0;i<2;++i) inputs[slot][i]=take(m::Buffer::create(resource,
        f.descriptions[i+1].required_bytes().value(),256),"power original input");
      check(runs[slot].bind(f.activation_id,take(c::ConstTensorView::create(f.descriptions[1],inputs[slot][0].view()),"power A")),"bind power A");
      check(runs[slot].bind(f.ids_id,take(c::ConstTensorView::create(f.descriptions[2],inputs[slot][1].view()),"power IDs")),"bind power IDs");
      if (ownership==g::OutputOwnership::kCaller)
        check(runs[slot].bind(f.output_id,f.output_description,take(pool.allocate(size),"power writer")),"bind power output");
      check(prepare(runs[slot],queues),"prepare power replay");
    }
    std::array<g::GraphOutput,2> old;std::array<std::vector<std::uint16_t>,2> expected;
    for (unsigned round=0;round<2;++round) for (std::size_t slot=0;slot<2;++slot) {
      const auto d=f.data(round*2+unsigned(slot));
      if (round) {scalar::expect_result(old[slot],expected[slot],read);EXPECT_FALSE(runs[slot].submit(queues));old[slot].reset();}
      check(write(inputs[slot][0].view(),std::as_bytes(std::span(d.a))),"update BF16 power A");
      check(write(inputs[slot][1].view(),std::as_bytes(std::span(d.ids))),"update power IDs");
      auto done=take(runs[slot].submit(queues),"power submit");check(done.wait(),"power Completion");++submissions;
      old[slot]=take(runs[slot].result(f.output_id),"power result");expected[slot]=f.oracle(d);
      scalar::expect_result(old[slot],expected[slot],read);
      if (old[1-slot].valid()) scalar::expect_result(old[1-slot],expected[1-slot],read);
    }
    runs={};for (std::size_t slot=0;slot<2;++slot) scalar::expect_result(old[slot],expected[slot],read);
    old={};if (ownership==g::OutputOwnership::kCaller) EXPECT_EQ(pool.stats().live_blocks,0u);
  }
  ::testing::Test::RecordProperty("actual_submissions",submissions);
}

template<class Compile,class Write,class Read,class Prepare>
void run(m::MemoryResourceRef resource,std::span<const g::QueueBinding> queues,
         Compile compile,Write write,Read read,Prepare prepare,c::StorageOwner owner={},bool ordinary=false) {
  run_with_policy(resource,queues,compile,write,read,prepare,owner,
      ordinary?Policy::kDivide:Policy::kPowerOfTwo);
}

template<class Compile,class Write,class Read,class Prepare>
void run_checked_ids(m::MemoryResourceRef resource,std::span<const g::QueueBinding> queues,
                     Compile compile,Write write,Read read,Prepare prepare,c::StorageOwner owner={},
                     Policy policy=Policy::kPowerOfTwo) {
  Case f(false,policy);auto bound=constant(f,resource,write,owner);
  auto compiled=compile(f.graph,g::CompileOptions{.constants=std::span(&bound.second,1)});
  ASSERT_TRUE(compiled)<<compiled.error().message();
  auto executable=compiled->executable;const auto ownership=take(executable.output_ownership(),"checked power ownership");
  auto run=take(executable.create_invocation({.resources=g::ResourceOptions{}}),"checked power invocation");bound={};compiled=g::CompiledGraph{};
  auto a=take(m::Buffer::create(resource,f.descriptions[1].required_bytes().value(),256),"checked power A");
  auto ids=take(m::Buffer::create(resource,f.descriptions[2].required_bytes().value(),256),"checked power IDs");
  const auto size=f.output_description.required_bytes().value();c::LeasePool pool;
  if (ownership==g::OutputOwnership::kCaller)
    pool=take(c::LeasePool::create(resource,(size+255)/256*256,{.max_blocks=1}),"checked power output");
  const auto bind=[&] {
    check(run.bind(f.activation_id,take(c::ConstTensorView::create(f.descriptions[1],a.view()),"checked A")),"bind checked A");
    check(run.bind(f.ids_id,take(c::ConstTensorView::create(f.descriptions[2],ids.view()),"checked IDs")),"bind checked IDs");
    if (ownership==g::OutputOwnership::kCaller)
      check(run.bind(f.output_id,f.output_description,take(pool.allocate(size),"checked writer")),"bind checked output");
    check(prepare(run,queues),"prepare checked power replay");
  };
  const auto upload=[&](const Data &d) {
    check(write(a.view(),std::as_bytes(std::span(d.a))),"checked A upload");
    check(write(ids.view(),std::as_bytes(std::span(d.ids))),"checked IDs upload");
  };
  unsigned successes=0,failures=0;auto data=f.data();upload(data);bind();
  const auto success=[&] {
    upload(data);auto done=take(run.submit(queues),"checked power submit");check(done.wait(),"checked power Completion");
    auto result=take(run.result(f.output_id),"checked power result");scalar::expect_result(result,f.oracle(data),read);++successes;
  };
  success();
  for (const auto invalid:{-1,int(f.experts)}) {
    data.ids.back()=invalid;upload(data);
    {auto done=run.submit(queues);ASSERT_TRUE(done)<<done.error().message();EXPECT_FALSE(done->wait().ok());EXPECT_FALSE(run.result(f.output_id));}
    EXPECT_FALSE(run.submit(queues));++failures;run={};run=take(executable.create_invocation({.resources=g::ResourceOptions{}}),"fresh power recovery");
    data=f.data(failures);upload(data);bind();success();
  }
  ::testing::Test::RecordProperty("successful_submissions",successes);
  ::testing::Test::RecordProperty("expected_failed_completions",failures);
}
} // namespace grouped_fp8_power2_test
