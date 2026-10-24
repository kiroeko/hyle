#include "../compute_models/row_cache_paged_indexer_fixture.h"
#if defined(HYLE_ROW_CACHE_GRAPH_CUDA)
#include "hyle/graph/cuda.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/detail/builtin_provider_ids.h"
#include "../graph/weight_alias_allocator.h"
#include <cstdio>
TEST(CudaPagedRowCacheIndexerGraph, FirstPrefillTopKAndRawWindowPaddingUseExactPages){
  auto context=hyle::graph::cuda::create_context().value();
  auto resource=hyle::compute::cuda::MemoryResource::create(context,hyle::compute::cuda::MemoryKind::kDevice).value();
  row_cache_paged_indexer_test::exercise(context,resource.ref(),[&](const hyle::graph::Graph&g){return hyle::graph::cuda::compile(g,context,resource.ref());});
}
TEST(CudaPagedRowCacheIndexerGraph, RequestLocalRangesCrossTheBoundedKeyTile){
  auto context=hyle::graph::cuda::create_context().value();
  auto resource=hyle::compute::cuda::MemoryResource::create(context,hyle::compute::cuda::MemoryKind::kDevice).value();
  row_cache_paged_indexer_test::crosses_tile(context,resource.ref(),[&](const hyle::graph::Graph&g){return hyle::graph::cuda::compile(g,context,resource.ref());});
}
TEST(CudaPagedRowCacheIndexerGraph, InvalidPaddingAndPrivateCapacityCannotEscape){
  auto context=hyle::graph::cuda::create_context().value();
  auto resource=hyle::compute::cuda::MemoryResource::create(context,hyle::compute::cuda::MemoryKind::kDevice).value();
  const auto compile=[&](const hyle::graph::Graph&g){return hyle::graph::cuda::compile(g,context,resource.ref());};
  row_cache_paged_indexer_test::rejected_private_readers(context,compile);
  row_cache_paged_indexer_test::invalid_padded(context,resource.ref(),compile);
}
TEST(CudaPagedRowCacheIndexerGraph, ColdSpanFailureInvalidatesTheBatchAndSameSlotCanReplan){
  namespace c=hyle::compute;namespace g=hyle::graph;namespace gb=g::batch;
  namespace f=row_cache_paged_indexer_test;
  hyle::graph::test::WeightAliasAllocator allocator;
  auto context=g::cuda::create_context({.metadata_allocator=allocator.get()}).value();
  auto resource=c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice).value();
  const auto schema=c::RowCacheBatchSchema::create(f::capacity(c::RowCacheEncoding::float4_block32())).value();
  row_cache_indexer_test::Fixture data(8,3);data.ranges={0,8,0,8,0,8};const auto reference=data.expected();
  auto builder=g::GraphBuilder::create().value();
  const auto q=builder.input(data.descriptions[0]).value(),w=builder.input(data.descriptions[2]).value(),r=builder.input(data.descriptions[3]).value();
  const auto scorer=gb::add_row_cache_indexer(builder,schema,q,w,r,4).value();
  ASSERT_TRUE(builder.publish(scorer.output()).ok());const auto graph=std::move(builder).finish().value();
  auto compiled=g::cuda::compile(graph,context,resource.ref()).value();
  ASSERT_EQ(compiled.report.step_count(),1u);const auto report=compiled.report.step(0).value();
  EXPECT_EQ(report.provider.id,c::detail::kCudaNativeProviderId);EXPECT_FALSE(report.runtime.capture_compatible);
  EXPECT_GE(report.workspace.size,32u<<20);
  std::printf("paged_indexer_workspace=%zu capture_compatible=%d\n",report.workspace.size,int(report.runtime.capture_compatible));
  auto queue=context.create_queue().value();const std::array lanes{g::QueueBinding{.lane=g::ExecutionLaneId{0},.queue=&queue}};
  auto slot=compiled.executable.create_invocation({.resources = g::ResourceOptions{}}).value();
  const std::array<std::int32_t,3> qi_a{0,2,3},qi_b{0,1,3},ki{0,2,3};
  const std::array<std::int32_t,3> pi{3,1,0};const std::array<std::int32_t,2> tail{1,1};
  const auto a=schema.version({qi_a,ki,pi,tail}).value(),b=schema.version({qi_b,ki,pi,tail}).value();
  auto arena=c::PagedArena::create(schema,resource.ref(),{.max_backing_bytes=8*68+1}).value();
  const auto pages=f::upload(arena,data.bank,68);auto pool=c::LeasePool::create(resource.ref(),131072).value();
  const auto output=graph.value_description(scorer.output()).value();
  const auto prepare=[&](c::RowCacheLayoutVersion v){return gb::prepare(slot, graph, lanes, {}, std::array{gb::RowCacheBinding{scorer.reader(),v}});};
  const auto execute=[&](c::RowCacheLayoutVersion version,bool second){
    ASSERT_TRUE(prepare(version).ok());ASSERT_TRUE(gb::bind(slot, scorer.bank_input(), f::frame(arena,version,pages,false)).ok());
    const std::array<std::int32_t,6> ranges=second?std::array<std::int32_t,6>{0,3,0,1,0,1}:std::array<std::int32_t,6>{0,3,1,3,0,1};
    ASSERT_TRUE(slot.bind(q, data.descriptions[0], f::f::load(pool,std::as_bytes(std::span(data.query)))).ok());
    ASSERT_TRUE(slot.bind(w, data.descriptions[2], f::f::load(pool,std::as_bytes(std::span(data.weights)))).ok());
    ASSERT_TRUE(slot.bind(r, data.descriptions[3], f::f::load(pool,std::as_bytes(std::span(ranges)))).ok());
    auto writer=pool.allocate(*output.required_bytes()).value();ASSERT_TRUE(slot.bind(scorer.output(), output, std::move(writer)).ok());
    ASSERT_TRUE(slot.prepare_replay(lanes).ok());auto done=slot.submit(lanes);ASSERT_TRUE(done);ASSERT_TRUE(done->wait().ok());
    auto held=slot.result(scorer.output()).value();const auto actual=f::read<float>(held);
    for(unsigned t=0;t<3;++t)for(int key=0;key<4;++key){
      const auto request=t<unsigned(second?1:2)?0u:1u;
      const bool visible=key>=ranges[t*2]&&key<ranges[t*2+1];
      const auto physical=visible?pi[std::size_t(ki[request])+unsigned(key)/2]*2+key%2:0;
      const auto expected=visible?reference[t*8+physical]:-std::numeric_limits<float>::infinity();
      EXPECT_FLOAT_EQ(actual[t*4+key],expected)<<t<<':'<<key;
    }
    EXPECT_TRUE(prepare(second?a:b).IsUnavailable());EXPECT_EQ(f::read<float>(held),actual);
  };
  execute(a,false);if(HasFatalFailure())return;
  // A schema-valid layout with the wrong active T fails in idle preflight.
  const std::array<std::int32_t,3> qi_short{0,1,2};const auto wrong=schema.version({qi_short,ki,pi,tail}).value();
  EXPECT_TRUE(prepare(wrong).IsInvalidArgument());auto rejected=slot.submit(lanes);ASSERT_FALSE(rejected);EXPECT_TRUE(rejected.error().IsFailedPrecondition());
  ASSERT_TRUE(prepare(a).ok());
  allocator.deny=true;const auto denied=prepare(b);allocator.deny=false;
  EXPECT_TRUE(denied.IsResourceExhausted());EXPECT_GT(allocator.rejected.load(),0u);
  rejected=slot.submit(lanes);ASSERT_FALSE(rejected);EXPECT_TRUE(rejected.error().IsFailedPrecondition());
  execute(b,true);if(HasFatalFailure())return;execute(a,false);
}
#else
#include "hyle/graph/cpu.h"
TEST(CpuPagedRowCacheIndexerGraph, FirstPrefillTopKAndRawWindowPaddingUseExactPages){
  auto context=hyle::graph::cpu::create_context().value();
  row_cache_paged_indexer_test::exercise(context,hyle::mem::system_resource(),[&](const hyle::graph::Graph&g){return hyle::graph::cpu::compile(g,context);});
}
#endif

#if !defined(HYLE_ROW_CACHE_GRAPH_CUDA)
TEST(CpuPagedRowCacheIndexerGraph, InvalidPaddingAndPrivateCapacityCannotEscape){
  auto context=hyle::graph::cpu::create_context().value();
  const auto compile=[&](const hyle::graph::Graph&g){return hyle::graph::cpu::compile(g,context);};
  row_cache_paged_indexer_test::rejected_private_readers(context,compile);
  row_cache_paged_indexer_test::invalid_padded(context,hyle::mem::system_resource(),compile);
}
#endif

#if !defined(HYLE_ROW_CACHE_GRAPH_CUDA)
TEST(CpuPagedRowCacheIndexerGraph, RequestLocalRangesCrossTheBoundedKeyTile){
  auto context=hyle::graph::cpu::create_context().value();
  row_cache_paged_indexer_test::crosses_tile(context,hyle::mem::system_resource(),[&](const hyle::graph::Graph&g){return hyle::graph::cpu::compile(g,context);});
}
#endif

#if !defined(HYLE_ROW_CACHE_GRAPH_CUDA)
#include "../../src/compute/provider_internal.h"
#include "../../src/compute_batch/row_cache_planning.h"
TEST(CpuPagedRowCacheIndexerGraph, ProjectionIsDescribedByTheSelectedImplementation){
  namespace c=hyle::compute;
  auto context=hyle::graph::cpu::create_context().value();
  const auto encoding=c::RowCacheEncoding::float4_block32();
  const auto bank=c::row_cache_carrier_description(encoding,5).value();
  const auto ids=c::TensorDesc::dense(c::DataType::kInt32,std::array<std::int64_t,1>{2}).value();
  const auto inspect=[&](c::Operation op){
    const auto inputs=std::array{bank,ids};
    const auto output=c::infer_single_output(op,inputs).value();
    auto prepared=c::prepare(context,op,inputs,std::array{output},hyle::mem::system_resource()).value();
    return c::detail::OperationPreparationAccess::inspect_access(prepared,c::detail::kRowCacheColdPlanningId);
  };
  auto checked=inspect(c::model_ops::row_cache_gather_decode(encoding));
  ASSERT_TRUE(checked);EXPECT_EQ(checked->projected_input,0u);
  EXPECT_FALSE(checked->allows_empty_footprint);EXPECT_FALSE(checked->requires_configuration);
  auto padded=inspect(c::model_ops::row_cache_gather_decode_padded(encoding));
  ASSERT_TRUE(padded);EXPECT_TRUE(padded->allows_empty_footprint);
  // The sibling unchecked decoder is selected from the same provider family;
  // schema acceptance/provider identity cannot grant its physical permission.
  auto unchecked=inspect(c::model_ops::row_cache_gather_decode(encoding,c::IndexBoundsPolicy::kUnchecked));
  ASSERT_FALSE(unchecked);EXPECT_TRUE(unchecked.error().IsUnimplemented());
}
#endif
