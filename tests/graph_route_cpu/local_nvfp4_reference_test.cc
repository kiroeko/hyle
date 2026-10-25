#include "../compute_models/local_nvfp4_reference_fixture.h"
#include "../compute_models/ct_block128_mixed_source_fixture.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/native.h"
#include "hyle/graph/cpu.h"

namespace {
template<class Run> void run_cpu(Run run) {
  namespace f=local_nvfp4_reference_test;
  auto context=f::take(f::g::cpu::create_context(),"CPU local NVFP4 context");
  auto queue=f::take(context.create_queue(),"CPU local NVFP4 queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0},&queue}};
  run(f::m::system_resource(),queues,
      [&](const f::g::Graph &graph,f::g::CompileOptions options) {return f::g::cpu::compile(graph,context,options);},
      [](f::m::BufferView destination,std::span<const std::byte> source) {return destination.write(source);},
      [](f::m::ConstBufferView source,std::span<std::byte> destination) {return source.read(destination);},
      [](f::g::Invocation &invocation,std::span<const f::g::QueueBinding> bindings) {return invocation.prepare_replay(bindings);});
}
}
TEST(CpuLocalNvfp4ReferenceGraph, OriginalDivisorBanksAndBothRoundingModesSurviveSlots) {
  run_cpu([](auto resource,auto queues,auto compile,auto write,auto read,auto prepare) {
    local_nvfp4_reference_test::run_lifetimes(resource,queues,compile,write,read,prepare);
  });
}
TEST(CpuLocalNvfp4ReferenceGraph, DynamicSelectedRuntimeDomainAndFailureRecovery) {
  run_cpu([](auto resource,auto queues,auto compile,auto write,auto read,auto prepare) {
    local_nvfp4_reference_test::run_runtime_domain(resource,queues,compile,write,read,prepare);
  });
}
TEST(CpuLocalNvfp4ReferenceDirect, CallerWorkspaceCannotOverwriteOriginalInputs) {
  namespace f=local_nvfp4_reference_test;
  auto context=f::take(f::g::cpu::create_context(),"CPU local direct context");
  auto queue=f::take(context.create_queue(),"CPU local direct queue");
  for(const auto rounding:{f::Rounding::kFloat32,f::Rounding::kBFloat16}) {
    f::Case sample(f::Layout::kDense,rounding);
    const auto data=sample.data();const auto payload=data.payloads();
    const std::array output_desc{sample.output_description};
    auto prepared=f::take(f::c::prepare(context,f::c::model_ops::local_nvfp4_linear(rounding),
        std::span(sample.descriptions).first(5),output_desc,f::m::system_resource()),"prepare local direct");
    EXPECT_EQ(prepared.provider().id,f::c::native::kProviderId);
    const auto requirement=prepared.workspace_requirement();
    EXPECT_EQ(requirement.size,std::size_t(2*sample.tokens*sample.depth));
    std::array<f::m::Buffer,5> buffers;
    std::array<f::c::ConstTensorView,5> inputs;
    for(std::size_t i=0;i<5;++i) {
      buffers[i]=f::take(f::m::Buffer::create(f::m::system_resource(),payload[i].size(),64),"direct original bank");
      f::check(buffers[i].view().write(payload[i]),"upload direct original bytes");
      inputs[i]=f::take(f::c::ConstTensorView::create(sample.descriptions[i],buffers[i].view()),"direct original view");
    }
    const auto size=sample.output_description.required_bytes().value();
    auto destination=f::take(f::m::Buffer::create(f::m::system_resource(),size,64),"direct output");
    const std::vector<std::byte> poison(size,std::byte{0x5a});
    f::check(destination.view().write(poison),"poison direct output");
    const std::array outputs{f::take(f::c::TensorView::create(sample.output_description,destination.view()),"direct output view")};
    auto aliased=f::take(prepared.create_invocation(buffers[0].view()),"aliased caller workspace");
    auto done=aliased.submit(queue,inputs,outputs);ASSERT_TRUE(done)<<done.error().message();
    const auto rejected=done->wait();EXPECT_TRUE(rejected.IsInvalidArgument())<<rejected.message();
    for(std::size_t i=0;i<5;++i) {
      std::vector<std::byte> actual(payload[i].size());f::check(buffers[i].view().read(actual),"read unchanged direct input");EXPECT_EQ(actual,payload[i]);
    }
    std::vector<std::byte> actual(size);f::check(destination.view().read(actual),"read unchanged direct output");EXPECT_EQ(actual,poison);
  }
}


TEST(CpuLocalNvfp4ReferenceGraph, OriginalCtMixedRolesKeepBfloatPreRoundAndSelectedSourceGenerations) {
  namespace s=ct18_selected_source_test;namespace c=s::c;namespace g=s::g;namespace m=s::m;
  auto context=s::take(g::cpu::create_context(),"CT18 mixed CPU context");
  auto queue=s::take(context.create_queue(),"CT18 mixed queue");
  struct Drain{c::Context &context;~Drain(){context.drain().IgnoreError();}}drain{context};
  struct SystemResource{m::MemoryResourceRef ref()const noexcept{return m::system_resource();}};
  const auto owner=s::take(c::StorageOwner::adopt(context,SystemResource{}),"CT18 actual source owner");
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&queue}};
  const auto layout=s::take(s::supply::Layout::create(s::plane_descriptions(),s::domain,s::capacity),"CT18 original quartet/map layout");
  EXPECT_EQ(layout.source[2],s::expert_specs[0].roles[2].description());
  EXPECT_EQ(layout.source[3],s::expert_specs[0].roles[3].description());
  EXPECT_EQ(layout.resident[2],s::dense(s::Type::kFloat32,{s::capacity,1}));
  EXPECT_EQ(layout.resident[3],s::dense(s::Type::kFloat32,{s::capacity,1}));
  auto supply=s::take(s::supply::CpuSupply::create(owner,layout),"existing two-generation source supply");
  auto source_ct=s::make_ct(owner);auto source_experts=s::make_experts(s::selection(0),owner);
  {auto wrong=source_ct;wrong[1].begin+=2;EXPECT_THROW(s::Case rejected(wrong,layout),std::runtime_error);}
  {auto wrong=source_experts;wrong[1].roles[3].begin+=4;EXPECT_FALSE(s::Source::accept(wrong));}
  auto first=s::take(s::Source::accept(source_experts),"original-shape CT18 local source admission");source_experts={};
  s::Case logical(source_ct,layout);source_ct={};
  ASSERT_TRUE(logical.constants[0].lease.valid()&&logical.constants[1].lease.valid());
  EXPECT_EQ(logical.constants[0].tensor.storage().resource(),owner.ref());
  EXPECT_EQ(logical.constants[1].tensor.storage().resource(),owner.ref());
  auto compiled=s::take(g::cpu::compile(logical.graph,context,g::CompileOptions{.constants=logical.constants}),"actual shared CT and local16 graph");
  auto pool=s::take(c::LeasePool::create(owner.ref(),256u<<10,{.max_blocks=16}),"finite CT18 original IO");
  const auto upload=[&](std::span<const std::byte> bytes) {
    auto writer=s::take(pool.allocate(bytes.size(),64),"CT18 input writer");s::check(writer.view()->write(bytes),"CT18 actual input bytes");
    return s::take(writer.publish(),"CT18 ready original input");
  };
  const auto prepare=[&](const s::supply::CpuBatch &batch,std::span<const std::uint16_t> attention,
      std::span<const std::uint16_t> mlp,const std::array<std::int32_t,2> &route) {
    auto run=s::take(compiled.executable.create_invocation({.resources=g::ResourceOptions{}}),"CT18 actual Invocation");
    for(std::size_t p=0;p<5;++p)s::check(run.bind(logical.banks[p],p<4?layout.resident[p]:layout.map_description(),batch.banks[p]),"same-generation quartet/map leases");
    s::check(run.bind(logical.attention,logical.attention_desc,upload(std::as_bytes(attention))),"original GDN input boundary");
    s::check(run.bind(logical.mlp,logical.mlp_desc,upload(std::as_bytes(mlp))),"separate original MLP input boundary");
    s::check(run.bind(logical.ids,logical.ids_desc,upload(std::as_bytes(std::span(route)))),"original global expert IDs");
    const std::array outputs{logical.ct_output,logical.expert_output};const std::array desc{logical.ct_output_desc,logical.expert_output_desc};
    for(std::size_t i=0;i<outputs.size();++i)if(s::take(compiled.executable.output_ownership(i),"actual CT18 output authority")==g::OutputOwnership::kCaller)
      s::check(run.bind(outputs[i],desc[i],s::take(pool.allocate(s::take(desc[i].required_bytes(),"CT18 output extent"),64),"actual CT18 output writer")),"original output binding");
    s::check(run.prepare_replay(queues),"original CT18 replay");return run;
  };
  const auto read=[&](const c::ReadLease &lease) {
    std::vector<std::uint16_t> values(lease.size()/2);s::check(lease.view()->read(std::as_writable_bytes(std::span(values))),"read actual CT18 BF16 result");return values;
  };
  std::array<s::supply::CpuBatch,2> batches;std::array<g::Invocation,2> runs;
  std::array<std::array<c::ReadLease,2>,2> results;
  std::array<std::array<std::vector<std::uint16_t>,2>,2> saved;
  bool bfloat_preround_matters=false,independent_ig_matters=false;
  for(unsigned phase=0;phase<2;++phase) {
    const auto selected=s::selection(phase);
    if(!phase){batches[phase]=s::take(supply.upload(selected,first),"first actual quartet/map generation");s::check(first.close(),"unmap first original source");}
    else {auto external=s::make_experts(selected,owner);auto next=s::take(s::Source::accept(external),"second original selected source");external={};
      batches[phase]=s::take(supply.upload(selected,next),"second actual quartet/map generation");s::check(next.close(),"unmap second original source");}
    const auto attention=s::attention_input(phase),mlp=s::mlp_input(phase);const std::array route{selected.globals[1],selected.globals[0]};
    EXPECT_NE(s::activation_reference(attention),s::activation_reference(attention,true));
    const auto expert=s::expert_oracle(mlp,route);
    bfloat_preround_matters|=expert!=s::expert_oracle(mlp,route,c::model_ops::Fp4InputRounding::kFloat32);
    independent_ig_matters|=expert!=s::expert_oracle(mlp,route,c::model_ops::Fp4InputRounding::kBFloat16,true);
    runs[phase]=prepare(batches[phase],attention,mlp,route);
    {auto done=s::take(runs[phase].submit(queues),"ordinary mixed CT18 submit");s::check(done.wait(),"actual mixed CT18 Completion");}
    const std::array ids{logical.ct_output,logical.expert_output};
    for(std::size_t i=0;i<ids.size();++i)results[phase][i]=s::take(s::take(runs[phase].result(ids[i]),"actual CT18 result").storage_lease(),"escaped original result lease");
    saved[phase]={read(results[phase][0]),read(results[phase][1])};
    ASSERT_TRUE(s::expect_ct(saved[phase][0],attention));EXPECT_EQ(saved[phase][1],expert);
    if(phase)for(std::size_t i=0;i<2;++i)EXPECT_EQ(read(results[0][i]),saved[0][i]);
  }
  EXPECT_TRUE(bfloat_preround_matters);EXPECT_TRUE(independent_ig_matters);
  EXPECT_EQ(supply.stats().live_blocks,10u);batches={};EXPECT_EQ(supply.stats().live_blocks,10u);
  runs={};EXPECT_EQ(supply.stats().live_blocks,0u);logical.constants={};compiled={};
  for(std::size_t phase=0;phase<2;++phase)for(std::size_t i=0;i<2;++i)EXPECT_EQ(read(results[phase][i]),saved[phase][i]);
  ::testing::Test::RecordProperty("source_value_scope","generated fixture; original cached config/header only");
  ::testing::Test::RecordProperty("source_repository",std::string(s::repository));
  ::testing::Test::RecordProperty("source_revision",std::string(s::revision));
  ::testing::Test::RecordProperty("upstream_boundaries","independent GDN8192 and post-attention-MLP8192");
  ::testing::Test::RecordProperty("successful_graph_submissions",2);
  ::testing::Test::RecordProperty("checked_BF16_endpoints",268);
}
