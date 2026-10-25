#include "../compute_models/ct_block128_qdq_fixture.h"
#include "../compute_models/ct_block128_source_role_fixture.h"
#include "../compute_models/ct_block128_selected_source_fixture.h"
#include "hyle/graph/cpu.h"

namespace {
void run_cpu(bool probes,bool checked=false) {
  namespace f=ct_block128_qdq_test;
  auto context=f::take(f::g::cpu::create_context(),"CPU CT block context");
  auto queue=f::take(context.create_queue(),"CPU CT block queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0},&queue}};
  const auto exercise=[&](auto resource,auto bindings,auto compile,auto write,auto read,auto prepare) {
    if(checked)f::run_checked_ids(resource,bindings,compile,write,read,prepare);
    else f::run(resource,bindings,compile,write,read,prepare,{},probes);
  };
  exercise(f::m::system_resource(),queues,
    [&](const f::g::Graph &graph,f::g::CompileOptions options) {return f::g::cpu::compile(graph,context,options);},
    [](f::m::BufferView destination,std::span<const std::byte> source) {return destination.write(source);},
    [](f::m::ConstBufferView source,std::span<std::byte> destination) {return source.read(destination);},
    [](f::g::Invocation &run,std::span<const f::g::QueueBinding> bindings) {return run.prepare_replay(bindings);});
}
} // namespace
TEST(CpuCtBlock128QdqGraph, DenseAndSelectedExpertsKeepBfloatMathAndResultLifetimes) {run_cpu(false);}
TEST(CpuCtBlock128QdqGraph, PublishedQdqAndSelectedWeightKeepEveryBfloatBoundary) {run_cpu(true);}
TEST(CpuCtBlock128QdqGraph, CompleteExpertIdsFailBeforeSelectionAndFreshSlotRecovers) {run_cpu(false,true);}

TEST(CpuCtBlock128QdqGraph, OriginalRoleSpanAndProvidedReadonlyBindingsStayExplicit) {
  namespace f = ct_block128_qdq_test;
  namespace s = ct_block128_source_role_test;
  s::Case declared(2);
  EXPECT_EQ(s::SourceRole::weight_begin, 905172320u);
  EXPECT_EQ(s::SourceRole::weight_bytes, 458752u);
  EXPECT_EQ(s::SourceRole::scale_begin, 1438176u);
  EXPECT_EQ(s::SourceRole::scale_bytes, 56u);
  EXPECT_EQ(declared.output_description, f::dense(f::Type::kBFloat16, {2, 128}));
  EXPECT_FALSE(declared.constants[0].tensor.valid());
  EXPECT_FALSE(declared.constants[1].tensor.valid());
  EXPECT_FALSE(declared.bind_source({}, {}).ok());
  EXPECT_FALSE(declared.constants[0].lease.valid());
  EXPECT_FALSE(declared.constants[1].lease.valid());

  // Generated nonzero bytes check only the real lease/binding entry and SF
  // preservation. They are NOT this publisher's payload or a numeric oracle.
  const std::vector<std::byte> codes(s::SourceRole::weight_bytes, std::byte{0x38});
  std::vector<std::uint16_t> scales(s::SourceRole::scale_bytes / 2);
  for (std::size_t i = 0; i < scales.size(); ++i) scales[i] = std::uint16_t(0x3f81 + i);
  const auto adopt = [](std::span<const std::byte> bytes) {
    auto buffer = f::take(f::m::Buffer::create(f::m::system_resource(), bytes.size(), 256), "source binding bytes");
    f::check(buffer.view().write(bytes), "source binding write");
    return f::take(f::c::ReadLease::adopt(std::move(buffer)), "source binding lease");
  };
  auto weight = adopt(codes), scale = adopt(std::as_bytes(std::span(scales)));
  const auto w_generation = weight.generation(), sf_generation = scale.generation();
  // A failed second extent cannot install only the first original binding.
  auto partial = f::take(scale.subview(0, scale.size() - 2), "short SF witness");
  EXPECT_FALSE(declared.bind_source(weight, partial).ok());
  EXPECT_FALSE(declared.constants[0].lease.valid());
  f::check(declared.bind_source(weight, scale), "bind provided source tile");
  EXPECT_EQ(declared.constants[0].lease.generation(), w_generation);
  EXPECT_EQ(declared.constants[1].lease.generation(), sf_generation);
  EXPECT_EQ(declared.constants[1].tensor.description(), f::dense(f::Type::kBFloat16, {1, 28}));
  EXPECT_FALSE(declared.bind_source(weight, scale).ok());
  weight.reset(); scale.reset(); partial.reset();
  std::vector<std::uint16_t> retained(scales.size());
  f::check(declared.constants[1].tensor.storage().read(std::as_writable_bytes(std::span(retained))), "read retained actual SF bits");
  EXPECT_EQ(retained, scales);
  EXPECT_TRUE(declared.constants[0].lease.valid());

  // A partial final N block still owns one complete SF row. This checks only
  // cold metadata assembly; no generated/original matrix is submitted here.
  namespace math = hyle::inference::ct_block128;
  auto b = f::take(f::g::GraphBuilder::create(), "partial N block builder");
  const auto w_desc = f::dense(f::Type::kFloat8E4M3FN, {129, 256});
  const auto sf_desc = f::dense(f::Type::kBFloat16, {2, 2});
  const auto short_desc = f::dense(f::Type::kBFloat16, {1, 2});
  const auto row_desc = f::dense(f::Type::kInt32, {129});
  const math::Operand w{f::take(b.input(w_desc), "N129 W"), w_desc};
  const math::Operand sf{f::take(b.input(sf_desc), "ceil-N SF"), sf_desc};
  const math::Operand short_sf{f::take(b.input(short_desc), "short SF"), short_desc};
  const math::Operand rows{f::take(b.input(row_desc), "actual block row mapping"), row_desc};
  EXPECT_FALSE(math::append_weight(b, w, short_sf, rows));
  const auto tail = f::take(math::append_weight(b, w, sf, rows), "N129 uses two SF rows");
  EXPECT_EQ(f::take(tail.description.concrete_description(), "N129 output"), f::dense(f::Type::kBFloat16, {129, 256}));
  f::check(b.publish(tail.value), "publish N129 metadata result");
  const auto graph = f::take(std::move(b).finish(), "finish N129 metadata graph");
  EXPECT_EQ(f::take(graph.output_description(), "N129 inferred output"), f::take(tail.description.concrete_description(), "N129 output"));
}


TEST(CpuCtBlock128QdqGraph, IndependentKimiSourceBanksMapBeforeSharedProjectionAndKeepOldResults) {
  namespace s=ct21_selected_source_test;namespace f=ct_block128_qdq_test;
  namespace c=s::c;namespace g=s::g;namespace m=s::m;
  auto context=f::take(g::cpu::create_context(),"CT21 selected CPU context");
  auto queue=f::take(context.create_queue(),"CT21 selected queue");
  struct Drain{c::Context &context;~Drain(){context.drain().IgnoreError();}}drain{context};
  struct SystemResource{m::MemoryResourceRef ref()const noexcept{return m::system_resource();}};
  const auto owner=f::take(c::StorageOwner::adopt(context,SystemResource{}),"CT21 source owner");
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&queue}};
  const auto layout=f::take(s::supply::Layout::create(s::descriptions(),s::domain,s::capacity),"CT21 original plane layout");
  auto supply=f::take(s::supply::CpuSupply::create(owner,layout),"existing CT21 bounded selected bank supply");
  s::Case logical(layout);auto compiled=f::take(g::cpu::compile(logical.graph,context,
      g::CompileOptions{.constants=logical.constants}),"compile shared actual CT21 projection");
  ASSERT_EQ(logical.output_desc,f::dense(f::Type::kBFloat16,{1,2,s::n}));
  auto pool=f::take(c::LeasePool::create(owner.ref(),128u<<10,{.max_blocks=10}),"CT21 finite dynamic IO");
  const auto upload=[&](std::span<const std::byte> bytes) {
    auto writer=f::take(pool.allocate(bytes.size(),64),"CT21 original dynamic source");
    f::check(writer.view()->write(bytes),"CT21 dynamic write");return f::take(writer.publish(),"CT21 ready dynamic lease");
  };
  const auto execute=[&](const s::supply::CpuBatch &batch,std::span<const std::uint16_t> a,
      const std::array<std::int32_t,2> &route) {
    auto run=f::take(compiled.executable.create_invocation({.resources=g::ResourceOptions{}}),"CT21 original slot");
    for(std::size_t p=0;p<3;++p)f::check(run.bind(logical.banks[p],p<2?layout.resident[p]:layout.map_description(),batch.banks[p]),"CT21 same-generation W/SF/map");
    f::check(run.bind(logical.activation,logical.a_desc,upload(std::as_bytes(a))),"CT21 actual BF16 A");
    f::check(run.bind(logical.ids,logical.ids_desc,upload(std::as_bytes(std::span(route)))),"CT21 actual global route IDs");
    if(f::take(compiled.executable.output_ownership(),"CT21 output ownership")==g::OutputOwnership::kCaller)
      f::check(run.bind(logical.output,logical.output_desc,f::take(pool.allocate(f::take(logical.output_desc.required_bytes(),"CT21 output bytes"),64),"CT21 output writer")),"CT21 output bind");
    f::check(run.prepare_replay(queues),"CT21 original replay");return run;
  };
  std::array<s::supply::CpuBatch,2> batches;std::array<g::Invocation,2> runs;
  std::array<c::ReadLease,2> results;std::array<std::vector<std::uint16_t>,2> saved;
  const auto read_result=[&](const c::ReadLease &lease) {
    std::vector<std::uint16_t> values(lease.size()/2);
    f::check(lease.view()->read(std::as_writable_bytes(std::span(values))),"read CT21 BF16 output");return values;
  };
  for(unsigned phase=0;phase<2;++phase) {
    const auto selected=s::selection(phase);auto external=s::make_source(selected,owner);
    auto accepted=f::take(s::Source::accept(external),"accept separate cached-shape W/SF leases");external={};
    std::array<std::vector<std::byte>,2> original;
    for(std::size_t p=0;p<2;++p)for(const auto global:selected.globals) {
      const auto bytes=accepted.plane_bytes(p,global);original[p].insert(original[p].end(),bytes.begin(),bytes.end());
    }
    batches[phase]=f::take(supply.upload(selected,accepted),"existing CT21 compressed planes/map generation");
    f::check(accepted.close(),"close original CT21 host mappings before Graph");
    for(std::size_t p=0;p<2;++p) {
      std::vector<std::byte> actual(batches[phase].banks[p].size());f::check(batches[phase].banks[p].view()->read(actual),"verify original copied bank bytes");
      EXPECT_EQ(actual,original[p]);
    }
    original={};
    const auto a=s::activation(phase);const std::array route{selected.globals[1],selected.globals[0]};
    runs[phase]=execute(batches[phase],a,route);
    {auto done=f::take(runs[phase].submit(queues),"CT21 actual selected submit");f::check(done.wait(),"CT21 selected Completion");}
    results[phase]=f::take(f::take(runs[phase].result(logical.output),"CT21 selected result").storage_lease(),"independent CT21 old output");
    saved[phase]=read_result(results[phase]);ASSERT_TRUE(s::expect(saved[phase],a,route));
    if(phase)EXPECT_EQ(read_result(results[0]),saved[0]);
  }
  {
    const auto a=s::activation(1);const std::array<std::int32_t,2> absent{2,0}; // Global0 is valid but absent from generation1.
    auto bad=execute(batches[1],a,absent);
    {auto done=f::take(bad.submit(queues),"CT21 accepted missing-map entry");EXPECT_FALSE(done.wait().ok());}
    EXPECT_FALSE(bad.result(logical.output));EXPECT_FALSE(bad.submit(queues));
    EXPECT_EQ(read_result(results[0]),saved[0]);EXPECT_EQ(read_result(results[1]),saved[1]);
  }
  EXPECT_EQ(supply.stats().live_blocks,6u);
  batches={};EXPECT_EQ(supply.stats().live_blocks,6u); // Original Invocations still retain W/SF/map readers.
  runs={};EXPECT_EQ(supply.stats().live_blocks,0u);
  logical.constants={};compiled={};
  for(std::size_t phase=0;phase<2;++phase)EXPECT_EQ(read_result(results[phase]),saved[phase]);
  ::testing::Test::RecordProperty("source_value_scope","generated fixture only; no original CT21 W/SF payload");
  ::testing::Test::RecordProperty("source_repository",std::string(s::repository));
  ::testing::Test::RecordProperty("source_revision",std::string(s::revision));
  ::testing::Test::RecordProperty("projection_n",3072);::testing::Test::RecordProperty("projection_k",3584);
  ::testing::Test::RecordProperty("successful_graph_submissions",2);
  ::testing::Test::RecordProperty("expected_missing_global_map_completion_failures",1);
}
