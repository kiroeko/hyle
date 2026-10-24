#include "../compute_models/static_fp8_reference_fixture.h"
#include "../compute/native_context_fixture.h"
#include "../../src/compute_packed_shared/static_fp8_linear.h"
#include "hyle/graph/cpu.h"

namespace {
template<class Run> void run_cpu(Run run,bool contracted) {
  namespace f=static_fp8_reference_test;
  auto context=f::take(f::g::cpu::create_context(),"CPU static FP8 context");
  auto queue=f::take(context.create_queue(),"CPU queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0},&queue}};
  run(f::m::system_resource(),queues,
      [&](const f::g::Graph &graph,f::g::CompileOptions options) {
        std::vector<f::g::NodeProviderConstraint> required;
        if(contracted)for(std::size_t i=0;i<graph.node_count();++i)
          required.push_back({f::take(graph.node_id(i),"constrained original node"),f::c::native::kProviderId,{}});
        options.node_providers=required;
        options.provider_replay=contracted?f::g::ProviderReplayPolicy::kRequirePrepared:f::g::ProviderReplayPolicy::kAllowAny;
        return f::g::cpu::compile(graph,context,options);
      },[](f::m::BufferView d,std::span<const std::byte> s) {return d.write(s);},
      [](f::m::ConstBufferView s,std::span<std::byte> d) {return s.read(d);},
      [](f::g::Invocation &i,std::span<const f::g::QueueBinding> q) {return i.prepare_replay(q);});
}
}
TEST(CpuStaticFloat8ReferenceGraph, OriginalBanksAndCurrentScalarsSurviveIndependentSlots) {
  run_cpu([](auto r,auto q,auto compile,auto write,auto read,auto prepare) {
    static_fp8_reference_test::run_lifetimes(r,q,compile,write,read,prepare);
  },true);
}
TEST(CpuStaticFloat8ReferenceGraph, ExceptionalScalesKeepTheOrdinaryIEEEOperationOrder) {
  run_cpu([](auto r,auto q,auto compile,auto write,auto read,auto prepare) {
    static_fp8_reference_test::run_runtime_domain(r,q,compile,write,read,prepare);
  },true);
}
TEST(CpuStaticFloat8ReferenceGraph, PublishedCodesExecuteTheOriginalDecomposition) {
  run_cpu([](auto r,auto q,auto compile,auto write,auto read,auto prepare) {
    static_fp8_reference_test::run_escape(r,q,compile,write,read,prepare);
  },false);
}
TEST(CpuStaticFloat8ReferenceDirect, DescriptorWorkspaceAndRejectedAliasPreserveOriginalBytes) {
  namespace f=static_fp8_reference_test;namespace c=f::c;namespace m=f::m;
  using Op=c::packed::internal::StaticFloat8Linear;
  auto context=f::take(c::test::native_context(),"native context");auto queue=f::take(context.create_queue(),"native queue");
  const std::array large{f::dense(f::Type::kBFloat16,{1,5120}),f::dense(f::Type::kFloat8E4M3FN,{10240,5120}),f::dense(f::Type::kFloat32,{}),f::dense(f::Type::kFloat32,{1})};
  const std::array large_output{f::dense(f::Type::kBFloat16,{1,10240})};
  auto geometry=f::take(c::prepare(context,c::Operation::custom(Op{}),large,large_output,m::system_resource()),"large prepare");
  EXPECT_EQ(geometry.workspace_requirement().size,5120u);EXPECT_EQ(geometry.invocation_private_bytes(),0u);
  f::Case test;const auto data=test.data();const auto payload=test.payloads(data);std::array<m::Buffer,4> buffers;std::array<c::ConstTensorView,4> views;
  for(std::size_t i=0;i<4;++i) {buffers[i]=f::take(m::Buffer::create(m::system_resource(),payload[i].size()+16,64),"direct input");f::check(buffers[i].view().fill(std::byte{0xa5}),"guard");f::check(buffers[i].view().subview(0,payload[i].size()).value().write(payload[i]),"input bytes");views[i]=f::take(c::ConstTensorView::create(test.descriptions[i],buffers[i].view()),"view");}
  const std::array descriptions{test.output};auto prepared=f::take(c::prepare(context,c::Operation::custom(Op{}),test.descriptions,descriptions,m::system_resource()),"direct prepare");
  auto output=f::take(m::Buffer::create(m::system_resource(),test.output.required_bytes().value()+16,64),"direct output");f::check(output.view().fill(std::byte{0xa5}),"output poison");const std::array outputs{f::take(c::TensorView::create(test.output,output.view()),"output view")};
  auto slot=f::take(prepared.create_invocation(buffers[0].view()),"deliberately aliased workspace");
  auto done=f::take(slot.submit(queue,views,outputs),"accepted rejected-alias work");EXPECT_TRUE(done.wait().IsInvalidArgument());
  for(std::size_t i=0;i<4;++i) {std::vector<std::byte> bytes(payload[i].size()+16);f::check(buffers[i].view().read(bytes),"read protected input");auto expected=payload[i];expected.resize(expected.size()+16,std::byte{0xa5});EXPECT_EQ(bytes,expected);}
  std::vector<std::byte> poison(output.size());f::check(output.view().read(poison),"read untouched output");EXPECT_EQ(poison,std::vector<std::byte>(output.size(),std::byte{0xa5}));
}
