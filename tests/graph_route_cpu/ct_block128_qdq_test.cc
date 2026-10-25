#include "../compute_models/ct_block128_qdq_fixture.h"
#include "../compute_models/ct_block128_source_role_fixture.h"
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
  EXPECT_EQ(tail.description, f::dense(f::Type::kBFloat16, {129, 256}));
  f::check(b.publish(tail.value), "publish N129 metadata result");
  const auto graph = f::take(std::move(b).finish(), "finish N129 metadata graph");
  EXPECT_EQ(f::take(graph.output_description(), "N129 inferred output"), tail.description);
}
