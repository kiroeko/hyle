#pragma once

#include "ct_block128_qdq_fixture.h"

namespace ct_block128_qdq_test {
// Independent published Fraction witnesses exercise scale underflow, a
// nonzero subnormal BF16 scale, and legal overflow after finite-source QDQ.
// Source NaN/Inf reduction policy is a separate domain, not silently assumed.
template<class Compile,class Write,class Read,class Prepare>
void run_activation_edges(m::MemoryResourceRef resource,std::span<const g::QueueBinding> queues,
    Compile compile,Write write,Read read,Prepare prepare,c::StorageOwner owner={}) {
  Case f(false,Output::kActivation);
  std::array<c::ReadLease,7> leases;std::array<g::ConstantBinding,7> constants;
  for(std::size_t i=0;i<constants.size();++i) {
    auto buffer=take(m::Buffer::create(resource,f.payloads[i].size(),256),"edge constant source");check(write(buffer.view(),f.payloads[i]),"upload edge constants");
    leases[i]=owner?take(owner.adopt_buffer(std::move(buffer)),"owned edge constants"):take(c::ReadLease::adopt(std::move(buffer)),"edge constants");
    constants[i]={f.constants[i],take(leases[i].tensor(f.descriptions[i]),"edge constant view"),g::ConstantPolicy::kMaterialize,owner,owner?leases[i]:c::ReadLease{}};
  }
  auto compiled=compile(f.graph,g::CompileOptions{.constants=constants});ASSERT_TRUE(compiled)<<compiled.error().message();
  auto run=take(compiled->executable.create_invocation({.resources=g::ResourceOptions{}}),"edge invocation");
  const auto ownership=take(compiled->executable.output_ownership(),"edge output ownership");
  ASSERT_TRUE(ownership==g::OutputOwnership::kCaller || ownership==g::OutputOwnership::kInvocation);
  auto input=take(m::Buffer::create(resource,f.activation_description.required_bytes().value(),256),"edge original A");
  check(run.bind(f.activation,take(c::ConstTensorView::create(f.activation_description,input.view()),"edge A view")),"bind edge A");
  c::LeasePool pool;const auto size=f.output_description.required_bytes().value();
  if(ownership==g::OutputOwnership::kCaller) {
    pool=take(c::LeasePool::create(resource,(size+255)/256*256,{.max_blocks=1}),"edge result pool");
    check(run.bind(f.output,f.output_description,take(pool.allocate(size),"edge result writer")),"bind edge result");
  }
  check(prepare(run,queues),"prepare edge replay");
  constants={};leases={};compiled=g::CompiledGraph{};
  constexpr std::array<std::uint16_t,6> inputs{0,0x8000,0x0001,0x0304,0x8304,0x7f7f};
  constexpr std::array<std::uint16_t,6> expected_bits{0,0,0,0x02fc,0x82fc,0x7f80};
  unsigned successes=0;
  for(std::size_t i=0;i<inputs.size();++i) {
    SCOPED_TRACE(::testing::Message()<<"BF16 input bits="<<inputs[i]);
    auto data=f.data();std::fill(data.a.begin(),data.a.end(),0);
    for(std::int64_t t=0;t<f.tokens;++t)for(std::int64_t group=0;group<f.groups;++group)
      data.a[std::size_t(t*f.depth+group*128)]=inputs[i];
    const auto expected=f.activation_oracle(data);
    ASSERT_EQ(expected[0],expected_bits[i]);
    check(write(input.view(),std::as_bytes(std::span(data.a))),"upload BF16 edge bits");
    auto done=take(run.submit(queues),"edge submit");check(done.wait(),"defined finite-source edge completion");
    auto result=take(run.result(f.output),"edge result");storage::expect_result(result,expected,read);++successes;
  }
  ::testing::Test::RecordProperty("successful_submissions",successes);
}
} // namespace ct_block128_qdq_test
