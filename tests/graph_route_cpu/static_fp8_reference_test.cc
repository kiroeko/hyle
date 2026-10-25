#include "../compute_models/q24_modelopt_mixed_source_case.h"
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

TEST(CpuStaticFloat8ReferenceGraph, ModelOptMixedSourceRolesConnectOrderedMoeTailToNextGdnProjection) {
  namespace s=q24_modelopt_mixed_source_case;
  namespace f=static_fp8_reference_test;
  struct SystemResource final {f::m::MemoryResourceRef ref() noexcept{return f::m::system_resource();}};
  auto context=f::take(f::g::cpu::create_context(),"mixed ordinary CPU context");
  const auto owner=f::take(s::c::StorageOwner::adopt(context,SystemResource{}),"mixed original resource owner");
  auto queue=f::take(context.create_queue(),"mixed CPU queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0},&queue}};
  std::array<std::array<f::g::GraphOutput,5>,2> held;
  std::array<s::Fixture,2> fixtures{s::Fixture(0),s::Fixture(1)};
  std::array<s::c::LeasePool,2> pools;
  const auto read=[](const auto &results) {
    std::array<std::vector<std::byte>,5> raw;
    for(std::size_t i=0;i<raw.size();++i) {
      const auto tensor=f::take(results[i].tensor(),"mixed result tensor");
      raw[i].resize(tensor.storage().size());f::check(tensor.storage().read(raw[i]),"mixed output bytes");
    }
    return raw;
  };
  const auto expected=s::descriptions();
  for(std::size_t phase=0;phase<fixtures.size();++phase) {
    const auto &fixture=fixtures[phase];
    std::array<s::source::ReadonlyRole,s::roles> parameters;
    for(std::size_t i=0;i<s::roles;++i)
      parameters[i]=s::source::detail::bind(s::names[i],expected[i],owner,
          fixture.parameters[i],&s::Fixture::identity);
    // The scalars share their dtype/extent, but retain distinct logical roles.
    auto swapped=parameters;std::swap(swapped[s::kSa],swapped[s::kSw]);
    EXPECT_THROW((void)s::Case(swapped),std::runtime_error);
    s::Case logical(parameters);
    auto compiled=f::take(f::g::cpu::compile(logical.graph,context,
        f::g::CompileOptions{.constants=logical.constants}),"compile connected mixed roles");
    auto run=f::take(compiled.executable.create_invocation({.resources=f::g::ResourceOptions{}}),"mixed ordinary invocation");
    const auto ownership=f::take(compiled.executable.output_ownership(),"mixed output ownership");
    const bool caller=ownership==f::g::OutputOwnership::kCaller;
    if(!caller)ASSERT_EQ(ownership,f::g::OutputOwnership::kInvocation);
    std::size_t capacity=0;
    for(const auto &d:logical.output_descriptions)capacity+=(f::take(d.required_bytes(),"mixed output extent")+255)/256*256;
    if(caller)pools[phase]=f::take(s::c::LeasePool::create(f::m::system_resource(),capacity,{.max_blocks=5}),"mixed output pool");
    std::array<f::m::Buffer,s::context_inputs> inputs;
    for(std::size_t i=0;i<inputs.size();++i) {
      inputs[i]=f::take(f::m::Buffer::create(f::m::system_resource(),fixture.inputs[i].size(),256),"mixed boundary input");
      f::check(inputs[i].view().write(fixture.inputs[i]),"mixed boundary bytes");
      f::check(run.bind(logical.inputs[i],f::take(s::c::ConstTensorView::create(logical.input_descriptions[i],inputs[i].view()),"mixed boundary view")),"mixed boundary bind");
    }
    if(caller)for(std::size_t i=0;i<logical.outputs.size();++i)
      f::check(run.bind(logical.outputs[i],logical.output_descriptions[i],f::take(pools[phase].allocate(f::take(logical.output_descriptions[i].required_bytes(),"mixed output bytes")),"mixed output writer")),"bind mixed result");
    f::check(run.prepare_replay(queues),"prepare connected mixed graph");
    parameters={};swapped={};logical.constants={};logical.graph={};compiled=f::g::CompiledGraph{};
    auto done=f::take(run.submit(queues),"submit connected mixed graph");f::check(done.wait(),"mixed ordinary completion");
    for(std::size_t i=0;i<held[phase].size();++i)held[phase][i]=f::take(run.result(logical.outputs[i]),"retain mixed output");
    ASSERT_NO_FATAL_FAILURE(fixtures[phase].check(read(held[phase])));
    if(phase)ASSERT_NO_FATAL_FAILURE(fixtures[0].check(read(held[0])));
    // Invocation/compile objects and input owners leave scope before the
    // final read. Old outputs and both source generations remain independent.
  }
  for(std::size_t phase=0;phase<fixtures.size();++phase)ASSERT_NO_FATAL_FAILURE(fixtures[phase].check(read(held[phase])));
  held={};for(const auto &pool:pools)if(pool.valid())EXPECT_EQ(pool.stats().live_blocks,0u);
  ::testing::Test::RecordProperty("source_value_scope","generated fixture, not original publisher payload");
  ::testing::Test::RecordProperty("original_scale_span_status","layer41 SA shard137 header/offset still unknown");
  ::testing::Test::RecordProperty("connected_source_roles",7);
  ::testing::Test::RecordProperty("actual_submissions",2);
}
