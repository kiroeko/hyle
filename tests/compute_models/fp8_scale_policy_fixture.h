#pragma once

#include "grouped_fp8_power2_fixture.h"

namespace fp8_scale_policy_test {
namespace base=grouped_fp8_power2_test;
namespace scalar=base::scalar;
namespace c=base::c;
namespace g=base::g;
namespace m=base::m;
using base::Policy;
using base::Type;
using base::take;
using base::check;
using base::dense;

struct Geometry { std::int64_t rows=2,columns=129,depth=256; };
struct DenseCase {
  Geometry geometry;
  Policy policy;
  float floor;
  std::array<c::TensorDesc,2> descriptions;
  c::TensorDesc output;
  std::vector<std::byte> weights;
  g::ValueId a{},w{},y{};
  g::Graph graph;
  explicit DenseCase(Policy chosen,Geometry shape={}):geometry(shape),policy(chosen),
      floor(chosen==Policy::kPowerOfTwo?1.e-4F:1.e-10F),descriptions{
          dense(Type::kBFloat16,{shape.rows,shape.depth}),
          take(c::float8_weight_description(shape.depth,shape.columns,c::Float8ScaleMode::kFloat32Block128),"original block128 carrier")},
      output(dense(Type::kBFloat16,{shape.rows,shape.columns})){
    const auto groups=shape.depth/128,blocks=(shape.columns+127)/128;
    const auto payload=std::size_t(shape.columns*shape.depth);
    weights.resize(descriptions[1].required_bytes().value(),std::byte{0});
    // One nonzero E4 code per contributing group. The code-domain dot is then
    // exact independently of reduction order; scale/add boundaries stay F32.
    for(std::int64_t n=0;n<shape.columns;++n)for(std::int64_t group=0;group<groups;++group){
      if((n==0&&group!=0)||(n%3==1&&group!=groups-1))continue;
      weights[std::size_t(n*shape.depth+group*128+1)]=std::byte(n%2?0xb8:0x38);
    }
    for(std::int64_t block=0;block<blocks;++block)for(std::int64_t group=0;group<groups;++group){
      constexpr std::array<float,4> values{1.F,2.F,.5F,4.F};
      const float scale=values[std::size_t(block+group)%values.size()];
      std::memcpy(weights.data()+payload+std::size_t(block*groups+group)*4,&scale,4);
    }
    auto builder=take(g::GraphBuilder::create(),"scale policy dense graph");
    a=take(builder.input(descriptions[0]),"BF16 source A");
    w=take(builder.constant(descriptions[1]),"original E4/F32 block128 bank");
    y=take(builder.add(c::model_ops::float8_block_linear(policy,floor),a,w),"existing dynamic FP8 linear");
    check(builder.publish(y),"publish policy result");graph=take(std::move(builder).finish(),"policy dense graph");
  }
  std::vector<std::uint16_t> data(unsigned phase)const{
    const auto &s=geometry;std::vector<std::uint16_t> input(std::size_t(s.rows*s.depth),0);
    for(std::int64_t row=0;row<s.rows;++row)for(std::int64_t group=0;group<s.depth/128;++group){
      const auto first=std::size_t(row*s.depth+group*128);
      const float factor=(phase%2?2.F:1.F)*(phase>=2?2.F:1.F)*(row%2?.5F:1.F);
      const float maximum=group==0?450.F*factor:group==1?0.F:
          std::ldexp(448.F,int((row+group+std::int64_t(phase))%3)-4);
      const float entry=group==0?.006866455078125F*factor:group==1?0.F:maximum/512.F;
      input[first]=scalar::bf16(maximum);
      input[first+1]=scalar::bf16(phase%2?-entry:entry);
      input[first+7]=0x8000;
    }
    return input;
  }
  std::vector<std::uint16_t> oracle(std::span<const std::uint16_t> input,
                                    bool wrong_divide=false)const{
    const auto &s=geometry;const auto groups=s.depth/128;
    const auto selected=wrong_divide?Policy::kDivide:policy;
    std::vector<float> scales(std::size_t(s.rows*groups));
    std::vector<float> codes(input.size());
    for(std::int64_t row=0;row<s.rows;++row)for(std::int64_t group=0;group<groups;++group){
      const auto first=std::size_t(row*s.depth+group*128);float maximum=0;
      for(std::size_t k=0;k<128;++k)maximum=std::max(maximum,std::abs(scalar::wide(input[first+k])));
      const auto scale=base::Case::scale(maximum,floor,selected);
      scales[std::size_t(row*groups+group)]=scale;
      for(std::size_t k=0;k<128;++k){
        const auto value=scalar::wide(input[first+k]);
        const auto normalized=selected==Policy::kMultiplyReciprocal?
            scalar::multiply(value,scalar::f32(1./double(scale))):scalar::f32(double(value)/double(scale));
        const auto magnitude=scalar::encode_scale(std::abs(normalized));
        codes[first+k]=scalar::e4m3(std::uint8_t(magnitude|(std::signbit(normalized)?0x80u:0)));
      }
    }
    const auto payload=std::size_t(s.columns*s.depth);
    std::vector<std::uint16_t> output_values(std::size_t(s.rows*s.columns));
    for(std::int64_t row=0;row<s.rows;++row)for(std::int64_t n=0;n<s.columns;++n){
      float sum=0;
      for(std::int64_t group=0;group<groups;++group){
        const auto code=std::to_integer<std::uint8_t>(weights[std::size_t(n*s.depth+group*128+1)]);
        const auto dot=scalar::multiply(codes[std::size_t(row*s.depth+group*128+1)],scalar::e4m3(code));
        float sw=0;std::memcpy(&sw,weights.data()+payload+std::size_t((n/128)*groups+group)*4,4);
        const auto term=scalar::multiply(scalar::multiply(dot,scales[std::size_t(row*groups+group)]),sw);
        sum=scalar::f32(double(sum)+double(term));
      }
      output_values[std::size_t(row*s.columns+n)]=scalar::bf16(sum);
    }
    return output_values;
  }
};

template<class Compile,class Write,class Read,class Prepare>
void run_dense_case(m::MemoryResourceRef resource,std::span<const g::QueueBinding> queues,
    Compile compile,Write write,Read read,Prepare prepare,const DenseCase &test,c::StorageOwner owner){
  if(test.policy==Policy::kMultiplyReciprocal){
    const auto input=test.data(0);const auto correct=test.oracle(input);const auto wrong=test.oracle(input,true);
    ASSERT_NE(correct,wrong);ASSERT_EQ(correct[0],0x3bc1U);ASSERT_EQ(wrong[0],0x3c01U);
  }
  auto bank=take(m::Buffer::create(resource,test.weights.size(),256),"original policy bank");
  check(write(bank.view(),test.weights),"write original encoded bytes");
  auto ready=owner?take(owner.adopt_buffer(std::move(bank)),"owned policy bank"):
      take(c::ReadLease::adopt(std::move(bank)),"original policy bank lease");
  std::array constants{g::ConstantBinding{test.w,take(ready.tensor(test.descriptions[1]),"typed original policy bank"),
      g::ConstantPolicy::kMaterialize,owner,owner?ready:c::ReadLease{}}};
  auto compiled=take(compile(test.graph,g::CompileOptions{.constants=constants}),"compile existing policy linear");
  const auto ownership=take(compiled.executable.output_ownership(),"policy output ownership");
  ASSERT_TRUE(ownership==g::OutputOwnership::kCaller||ownership==g::OutputOwnership::kInvocation);
  std::array runs{take(compiled.executable.create_invocation({.resources=g::ResourceOptions{}}),"dense policy slot0"),
                  take(compiled.executable.create_invocation({.resources=g::ResourceOptions{}}),"dense policy slot1")};
  compiled={};constants={};ready={};
  const auto output_bytes=test.output.required_bytes().value();c::LeasePool pool;
  if(ownership==g::OutputOwnership::kCaller)
    pool=take(c::LeasePool::create(resource,2*((output_bytes+255)/256*256),{.max_blocks=2}),"two dense policy outputs");
  std::array<m::Buffer,2> inputs;
  for(std::size_t slot=0;slot<2;++slot){
    inputs[slot]=take(m::Buffer::create(resource,test.descriptions[0].required_bytes().value(),256),"current BF16 policy A");
    check(runs[slot].bind(test.a,take(c::ConstTensorView::create(test.descriptions[0],inputs[slot].view()),"policy A view")),"bind policy A once");
    if(ownership==g::OutputOwnership::kCaller)
      check(runs[slot].bind(test.y,test.output,take(pool.allocate(output_bytes),"policy writer")),"bind actual output");
    check(prepare(runs[slot],queues),"prepare original policy invocation");
  }
  std::array<g::GraphOutput,2> retained;std::array<std::vector<std::uint16_t>,2> expected;
  for(unsigned round=0;round<2;++round)for(std::size_t slot=0;slot<2;++slot){
    if(round){scalar::expect_result(retained[slot],expected[slot],read);EXPECT_FALSE(runs[slot].submit(queues));retained[slot].reset();}
    const auto input=test.data(round*2+static_cast<unsigned>(slot));
    check(write(inputs[slot].view(),std::as_bytes(std::span(input))),"current policy input bytes");
    auto done=take(runs[slot].submit(queues),"dense policy submit");check(done.wait(),"dense policy Completion");
    retained[slot]=take(runs[slot].result(test.y),"actual policy result");expected[slot]=test.oracle(input);
    scalar::expect_result(retained[slot],expected[slot],read);
    if(retained[1-slot].valid())scalar::expect_result(retained[1-slot],expected[1-slot],read);
  }
  runs={};pool={};for(std::size_t slot=0;slot<2;++slot)scalar::expect_result(retained[slot],expected[slot],read);
}

template<class Compile,class Write,class Read,class Prepare>
void run_dense(m::MemoryResourceRef resource,std::span<const g::QueueBinding> queues,
    Compile compile,Write write,Read read,Prepare prepare,c::StorageOwner owner={}){
  for(const auto policy:{Policy::kDivide,Policy::kPowerOfTwo,Policy::kMultiplyReciprocal}){
    SCOPED_TRACE(::testing::Message()<<"dense policy="<<unsigned(policy));
    run_dense_case(resource,queues,compile,write,read,prepare,DenseCase(policy),owner);
    if(::testing::Test::HasFatalFailure())return;
  }
  ::testing::Test::RecordProperty("successful_submissions",12);
  ::testing::Test::RecordProperty("policy_witness","450/tiny: Divide3c01,MultiplyReciprocal3bc1;Wcode1/SF1");
}

template<class Compile,class Write,class Read,class Prepare>
void run_source_role(m::MemoryResourceRef resource,std::span<const g::QueueBinding> queues,
    Compile compile,Write write,Read read,Prepare prepare,c::StorageOwner owner={}){
  // Actual fixed catalog30 header: layer45 expert0 gate_proj E4[2048,4096]
  // and F32SF[16,32]. These are finite artificial bytes at that geometry,
  // not downloaded checkpoint values or a complete routed expert layer.
  run_dense_case(resource,queues,compile,write,read,prepare,
      DenseCase(Policy::kMultiplyReciprocal,{1,2048,4096}),owner);
  if(::testing::Test::HasFatalFailure())return;
  ::testing::Test::RecordProperty("successful_submissions",4);
  ::testing::Test::RecordProperty("role_scope","catalog30 layer45 expert0 gate_proj:BF16A[1,4096],E4W[2048,4096],F32SF[16,32];artificial bytes;no payload/fullmodel/AMDrcpf claim");
}

template<class Compile,class Write,class Read,class Prepare>
void run_grouped(m::MemoryResourceRef resource,std::span<const g::QueueBinding> queues,
    Compile compile,Write write,Read read,Prepare prepare,c::StorageOwner owner={}){
  base::run_with_policy(resource,queues,compile,write,read,prepare,owner,Policy::kMultiplyReciprocal);
}

template<class Compile,class Write,class Read,class Prepare>
void run_checked_ids(m::MemoryResourceRef resource,std::span<const g::QueueBinding> queues,
    Compile compile,Write write,Read read,Prepare prepare,c::StorageOwner owner={}){
  base::run_checked_ids(resource,queues,compile,write,read,prepare,owner,Policy::kMultiplyReciprocal);
}
} // namespace fp8_scale_policy_test
