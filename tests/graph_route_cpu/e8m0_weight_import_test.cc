#include "../compute_models/e8m0_weight_import_fixture.h"
#include "../compute_models/kimi_modelopt_mixed_source_fixture.h"
#include "hyle/graph/cpu.h"

TEST(CpuE8M0WeightImport, ColdCanonicalBankUsesOriginalLinearAndResultLifetime) {
  namespace f=e8m0_weight_import_test;
  auto context=f::take(f::g::cpu::create_context(), "CPU import consumer");
  auto queue=f::take(context.create_queue(), "CPU queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0},&queue}};
  f::run(f::m::system_resource(),queues,
      [&](const f::g::Graph &graph,f::g::CompileOptions options) {
        return f::g::cpu::compile(graph,context,options);
      },
      [](f::m::BufferView to,std::span<const std::byte> from) {return to.write(from);},
      [](f::m::ConstBufferView from,std::span<std::byte> to) {return from.read(to);},
      [](f::g::Invocation &invocation,std::span<const f::g::QueueBinding> bindings) {
        return invocation.prepare_replay(bindings);
      });
}

TEST(CpuE8M0WeightImport, StoredFloat32ScalesKeepDynamicActivationQuantization) {
  namespace f=e8m0_weight_import_test;
  auto context=f::take(f::g::cpu::create_context(), "CPU stored-scale consumer");
  auto queue=f::take(context.create_queue(), "CPU queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0},&queue}};
  f::run(f::m::system_resource(),queues,
      [&](const f::g::Graph &graph,f::g::CompileOptions options) {
        return f::g::cpu::compile(graph,context,options);
      },
      [](f::m::BufferView to,std::span<const std::byte> from) {return to.write(from);},
      [](f::m::ConstBufferView from,std::span<std::byte> to) {return from.read(to);},
      [](f::g::Invocation &invocation,std::span<const f::g::QueueBinding> bindings) {
        return invocation.prepare_replay(bindings);
      },{},true);
}


TEST(CpuE8M0WeightImport, KimiMixedOriginalSourceRolesKeepDistinctReadersAndSelectedGenerations) {
  namespace s=kimi20_mixed_source_test;namespace c=s::c;namespace g=s::g;namespace m=s::m;
  auto context=s::take(g::cpu::create_context(),"Kimi20 mixed CPU context");
  auto queue=s::take(context.create_queue(),"Kimi20 mixed queue");
  struct Drain{c::Context &context;~Drain(){context.drain().IgnoreError();}}drain{context};
  struct SystemResource{m::MemoryResourceRef ref()const noexcept{return m::system_resource();}};
  const auto owner=s::take(c::StorageOwner::adopt(context,SystemResource{}),"actual Kimi20 source owner");
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&queue}};
  const auto layout=s::take(s::supply::Layout::create(s::plane_descriptions(),s::domain,s::capacity),"Kimi20 existing quartet/map layout");
  EXPECT_EQ(layout.source[2],s::expert_specs[0].roles[2].description());
  EXPECT_EQ(layout.source[3],s::expert_specs[0].roles[3].description());
  EXPECT_EQ(layout.resident[2],s::dense(s::Type::kFloat32,{s::capacity}));
  EXPECT_EQ(layout.resident[3],s::dense(s::Type::kFloat32,{s::capacity}));
  auto supply=s::take(s::supply::CpuSupply::create(owner,layout),"existing bounded two-generation supply");
  auto original_pb=s::make_pb(owner);auto first_experts=s::make_experts(s::selection(0),owner);
  {auto wrong=original_pb;wrong[1].begin+=4;EXPECT_FALSE(s::prepare_pb(wrong));}
  {auto wrong=first_experts;wrong[1].roles[3].begin+=4;EXPECT_FALSE(s::Source::accept(wrong));}
  auto prepared_pb=s::take(s::prepare_pb(original_pb),"cold original PB import once");original_pb={};
  auto first=s::take(s::Source::accept(first_experts),"original-shape Kimi20 source admission");first_experts={};
  {
    std::vector<std::byte> bytes(prepared_pb.lease.size());
    s::check(prepared_pb.lease.view()->read(bytes),"verify original PB E4/F32 bit import");
    const auto payload=std::size_t(s::pb_columns*s::hidden);
    for(std::int64_t n=0;n<s::pb_columns;++n)for(std::int64_t k=0;k<s::hidden;++k) {
      const auto group=k/128;const auto expected=k%128==1&&(group==0||group==s::hidden/128-1)?s::pb_code(n,group):std::uint8_t{0};
      ASSERT_EQ(std::to_integer<std::uint8_t>(bytes[std::size_t(n*s::hidden+k)]),expected);
    }
    for(std::size_t group=0;group<std::size_t(s::hidden/128);++group) {
      std::uint32_t actual=0;std::memcpy(&actual,bytes.data()+payload+group*4,4);
      EXPECT_EQ(actual,std::bit_cast<std::uint32_t>(s::pb_sf(group)));
    }
  }
  s::Case logical(prepared_pb,layout);prepared_pb={};
  auto compiled=s::take(g::cpu::compile(logical.graph,context,g::CompileOptions{.constants=std::span(&logical.constant,1)}),"existing mixed PB/NVFP4 graph");
  auto pool=s::take(c::LeasePool::create(owner.ref(),256u<<10,{.max_blocks=16}),"finite Kimi20 dynamic IO");
  const auto upload=[&](std::span<const std::byte> bytes) {
    auto writer=s::take(pool.allocate(bytes.size(),64),"Kimi20 actual input writer");
    s::check(writer.view()->write(bytes),"Kimi20 actual input write");return s::take(writer.publish(),"actual ready input");
  };
  const auto prepare=[&](const s::supply::CpuBatch &batch,std::span<const std::uint16_t> attention,
      std::span<const std::uint16_t> routed,const std::array<std::int32_t,2> &ids) {
    auto run=s::take(compiled.executable.create_invocation({.resources=g::ResourceOptions{}}),"mixed actual Invocation");
    for(std::size_t p=0;p<5;++p)s::check(run.bind(logical.banks[p],p<4?layout.resident[p]:layout.map_description(),batch.banks[p]),"same-generation original quartet/map");
    s::check(run.bind(logical.attention,logical.attention_desc,upload(std::as_bytes(attention))),"actual attention boundary");
    s::check(run.bind(logical.routed,logical.routed_desc,upload(std::as_bytes(routed))),"distinct actual routed boundary");
    s::check(run.bind(logical.ids,logical.ids_desc,upload(std::as_bytes(std::span(ids)))),"actual global expert IDs");
    const std::array outputs{logical.pb_output,logical.expert_output};
    const std::array desc{logical.pb_output_desc,logical.expert_output_desc};
    for(std::size_t i=0;i<outputs.size();++i)if(s::take(compiled.executable.output_ownership(i),"Kimi20 output authority")==g::OutputOwnership::kCaller)
      s::check(run.bind(outputs[i],desc[i],s::take(pool.allocate(s::take(desc[i].required_bytes(),"mixed output extent"),64),"actual mixed output writer")),"original output binding");
    s::check(run.prepare_replay(queues),"original mixed replay");return run;
  };
  const auto read=[&](const c::ReadLease &lease) {
    std::vector<std::uint16_t> values(lease.size()/2);
    s::check(lease.view()->read(std::as_writable_bytes(std::span(values))),"read actual mixed BF16 result");return values;
  };
  std::array<s::supply::CpuBatch,2> batches;std::array<g::Invocation,2> runs;
  std::array<std::array<c::ReadLease,2>,2> outputs;
  std::array<std::array<std::vector<std::uint16_t>,2>,2> saved;
  bool ig_matters=false,a_round_matters=false,w_round_matters=false;
  for(unsigned phase=0;phase<2;++phase) {
    const auto selected=s::selection(phase);
    if(!phase) {
      batches[phase]=s::take(supply.upload(selected,first),"original first quartet/map generation");
      s::check(first.close(),"close first original source mappings after copy");
    } else {
      auto experts=s::make_experts(selected,owner);
      auto next=s::take(s::Source::accept(experts),"original second selected source");experts={};
      batches[phase]=s::take(supply.upload(selected,next),"original second quartet/map generation");
      s::check(next.close(),"close second original source mappings after copy");
    }
    const auto a=s::attention_input(phase),routed=s::routed_input(phase);
    const std::array route{selected.globals[1],selected.globals[0]};
    const auto pb_expected=s::pb_oracle(a),expert_expected=s::expert_oracle(routed,route);
    if(!phase){EXPECT_NE(pb_expected,s::pb_oracle(a,true));EXPECT_NE(pb_expected,s::pb_oracle(a,false,true));}
    ig_matters|=expert_expected!=s::expert_oracle(routed,route,true);
    a_round_matters|=expert_expected!=s::expert_oracle(routed,route,false,true);
    w_round_matters|=expert_expected!=s::expert_oracle(routed,route,false,false,true);
    runs[phase]=prepare(batches[phase],a,routed,route);
    {auto done=s::take(runs[phase].submit(queues),"mixed PB and NVFP4 submit");s::check(done.wait(),"real mixed Completion");}
    const std::array ids{logical.pb_output,logical.expert_output};
    for(std::size_t i=0;i<ids.size();++i)outputs[phase][i]=s::take(s::take(runs[phase].result(ids[i]),"mixed completed result").storage_lease(),"independent actual old result lease");
    saved[phase]={read(outputs[phase][0]),read(outputs[phase][1])};
    EXPECT_EQ(saved[phase][0],pb_expected);EXPECT_EQ(saved[phase][1],expert_expected);
    if(phase)for(std::size_t i=0;i<2;++i)EXPECT_EQ(read(outputs[0][i]),saved[0][i]);
  }
  EXPECT_TRUE(ig_matters);EXPECT_TRUE(a_round_matters);EXPECT_TRUE(w_round_matters);
  EXPECT_EQ(supply.stats().live_blocks,10u);batches={};EXPECT_EQ(supply.stats().live_blocks,10u);
  runs={};EXPECT_EQ(supply.stats().live_blocks,0u);logical.constant={};compiled={};
  for(std::size_t phase=0;phase<2;++phase)for(std::size_t i=0;i<2;++i)EXPECT_EQ(read(outputs[phase][i]),saved[phase][i]);
  ::testing::Test::RecordProperty("source_value_scope","generated fixture; original header spans only");
  ::testing::Test::RecordProperty("PB_policy","stored F32 SF, dynamic group128 divide floor1e-4; no BF16 scale pre-round");
  ::testing::Test::RecordProperty("upstream_boundaries","separate attention7168 and post-routed-down/norm3584; no fabricated connection");
  ::testing::Test::RecordProperty("successful_graph_submissions",2);
  ::testing::Test::RecordProperty("checked_BF16_endpoints",268);
}
