#include "../compute_models/row_cache_graph_fixture.h"
#if defined(HYLE_ROW_CACHE_GRAPH_CUDA)
#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/cuda.h"
#include <chrono>
#include <condition_variable>
#include <mutex>
#else
#include "hyle/graph/cpu.h"
#endif

namespace {
namespace c=hyle::compute;namespace g=hyle::graph;namespace gb=g::batch;
namespace f=row_cache_paged_test;
#if defined(HYLE_ROW_CACHE_GRAPH_CUDA)
struct Gate final {
  std::mutex mutex;std::condition_variable condition;bool entered=false,released=false;
  static void CUDART_CB wait(void*raw){auto&g=*static_cast<Gate*>(raw);std::unique_lock lock(g.mutex);
    g.entered=true;g.condition.notify_all();g.condition.wait(lock,[&]{return g.released;});}
  bool await(){std::unique_lock lock(mutex);return condition.wait_for(lock,std::chrono::seconds(3),[&]{return entered;});}
  void release(){std::lock_guard lock(mutex);released=true;condition.notify_all();}
};
#endif
template<class Compile>void exercise(c::Context&context,hyle::mem::MemoryResourceRef resource,Compile compile){
  for(const auto encoding:{c::RowCacheEncoding::float8_block64_tail64(),c::RowCacheEncoding::float4_block32()})for(unsigned mutable_page:{4u,1u}){
    SCOPED_TRACE(encoding.depth());
    SCOPED_TRACE(mutable_page);auto queue=context.create_queue().value();
    auto schema=c::RowCacheBatchSchema::create(f::capacity(encoding)).value();const auto version=f::layout(schema);
    const auto ids_desc=c::TensorDesc::dense(c::DataType::kInt32,std::array<std::int64_t,2>{4,3}).value();
    auto builder=g::GraphBuilder::create().value();const auto ids_port=builder.input(ids_desc).value();
    const auto recipe=gb::add_row_cache_gather(builder,schema,ids_port).value();
    const auto floats=builder.add(c::Cast{c::DataType::kFloat32},recipe.output()).value();
    const auto negative=builder.add(c::Unary{c::UnaryKind::kNeg},floats).value();
    ASSERT_TRUE(builder.publish(recipe.output()).ok());ASSERT_TRUE(builder.publish(floats).ok());ASSERT_TRUE(builder.publish(negative).ok());
    const auto graph=std::move(builder).finish().value();auto compiled=f::take(compile(graph),"pending row compile");
    const std::array lanes{g::QueueBinding{.lane=g::ExecutionLaneId{0},.queue=&queue}};
    auto receiver=f::take(compiled.executable.create_invocation({.resources=g::ResourceOptions{}}),"pending row owner");
    auto pool=c::LeasePool::create(resource,65536).value();const auto ids=f::ids();
    const auto selected_ids=f::load(pool,std::as_bytes(std::span(ids)));
    const auto reference=f::data(encoding);const auto output_desc=graph.value_description(recipe.output()).value();
    const auto negative_desc=graph.value_description(negative).value();
    const std::array versions{gb::RowCacheBinding{recipe,version}};
    const auto prepare=[&](){return gb::prepare(receiver,graph,lanes,{},versions,{.max_source_completions=3});};
    const auto bind_outputs=[&](auto&slot){
      auto output=pool.allocate(*output_desc.required_bytes()).value();auto neg=pool.allocate(*negative_desc.required_bytes()).value();
      auto cast=pool.allocate(*negative_desc.required_bytes()).value();
      ASSERT_TRUE(slot.bind(recipe.output(),output_desc,std::move(output)).ok());
      ASSERT_TRUE(slot.bind(floats,negative_desc,std::move(cast)).ok());
      ASSERT_TRUE(slot.bind(negative,negative_desc,std::move(neg)).ok());
    };
    // Warm the exact original Graph/native kernels before a real CUDA gate.
    {
      auto arena=c::PagedArena::create(schema,resource,{.max_backing_bytes=10*encoding.row_bytes()+1}).value();
      auto ready=f::pages(arena,encoding,reference);ASSERT_TRUE(prepare().ok());
      ASSERT_TRUE(receiver.bind(ids_port,ids_desc,selected_ids).ok());
      ASSERT_TRUE(gb::bind(receiver,recipe.bank_input(),f::frame(arena,version,ready)).ok());bind_outputs(receiver);
      ASSERT_TRUE(receiver.prepare_replay(lanes).ok());
      auto done=receiver.submit(lanes);ASSERT_TRUE(done)<<done.error().message();ASSERT_TRUE(done->wait().ok());
      EXPECT_EQ(row_cache_graph_test::read<std::uint16_t>(receiver.result(recipe.output()).value()),f::expected(encoding,reference,ids));
    }
    ASSERT_TRUE(prepare().ok());
    ASSERT_TRUE(receiver.bind(ids_port,ids_desc,selected_ids).ok());bind_outputs(receiver);
    auto arena=c::PagedArena::create(schema,resource,{.max_backing_bytes=10*encoding.row_bytes()+1}).value();
    std::vector<c::ReadLease> ready;
    for(unsigned p:{0u,1u,3u,4u}){if(p==mutable_page)continue;auto w=arena.acquire(p).value();
      ASSERT_TRUE(w.lease().view()->write(std::as_bytes(std::span(reference.bytes).subspan(2*p*encoding.row_bytes(),2*encoding.row_bytes()))).ok());
      ready.push_back(w.publish(2).value());}
    auto page=arena.acquire(mutable_page).value();const auto bytes=arena.page_bytes();
    const unsigned required_rows=mutable_page==4?2u:1u;const auto copy_bytes=required_rows*encoding.row_bytes();
    // Full page or partial tail: authorize exactly the new initialized prefix.
    auto changed=reference;std::copy_n(reference.bytes.begin(),copy_bytes,changed.bytes.begin()+mutable_page*bytes);
    std::copy_n(reference.decoded.begin(),required_rows*encoding.depth(),changed.decoded.begin()+mutable_page*2*encoding.depth());
    const auto payload=std::as_bytes(std::span(reference.bytes)).first(copy_bytes);
    const std::array specs{c::CopyResourceSpec{resource,copy_bytes},c::CopyResourceSpec{resource,bytes}};
    auto copies=c::CopyBatchProgram::prepare(context,specs).value();auto copy=c::CopyBatchInvocation::create(copies,queue).value();
#if defined(HYLE_ROW_CACHE_GRAPH_CUDA)
    namespace n=c::cuda;auto source=pool.allocate(copy_bytes).value();auto target=n::pointer<std::byte>(*source.view()).value();
    auto host=n::MemoryResource::create(context,n::MemoryKind::kPinnedHost).value();auto host_pool=c::LeasePool::create(host.ref(),copy_bytes+64).value();
    const auto staging=f::load(host_pool,payload);const auto host_pointer=n::pointer<std::byte>(*staging.view()).value();
    auto uses=c::LeaseUsePool::create().value();auto writing=uses.begin().value();
    ASSERT_TRUE(writing.write(source).ok());ASSERT_TRUE(writing.read(staging).ok());
    ASSERT_TRUE(copy.bind_mutable({0},std::move(source)).ok());
#else
    const auto source=f::load(pool,payload);ASSERT_TRUE(copy.bind_source({0},source).ok());
#endif
    ASSERT_TRUE(copy.bind_mutable({1},std::move(page.lease())).ok());
    const std::array<c::CopyPhase,2> phases{{{}, {.dependency_mask=1}}};
    const std::array<c::CopyRange,1> ranges{{{{0},{1},{1},0,0,copy_bytes}}};
    ASSERT_TRUE(copy.prepare_replay(phases,ranges).ok());const std::array<c::PagedPageUpdate,1> updates{{{&page,{1}}}};
    auto frame=f::take(arena.pending_read(version,ready,updates),"pending typed row frame");
    auto wrong=arena.pending_read(f::layout(schema),ready,updates).value();
    EXPECT_TRUE(gb::bind(receiver,recipe.bank_input(),wrong).IsInvalidArgument());
    EXPECT_TRUE(gb::bind(receiver,ids_port,frame).IsInvalidArgument());
    EXPECT_TRUE(gb::bind(receiver,recipe.bank_input(),c::PagedRowCachePendingRead{}).IsInvalidArgument());
    ASSERT_TRUE(gb::bind(receiver,recipe.bank_input(),frame).ok());
    EXPECT_TRUE(gb::bind(receiver,recipe.bank_input(),frame).IsFailedPrecondition());
    ASSERT_TRUE(receiver.prepare_replay(lanes,{.mode=g::ReplayMode::kCommands,.host_wait=g::HostWaitPolicy::kReject}).ok());
#if !defined(HYLE_ROW_CACHE_GRAPH_CUDA)
    // An accepted copy of an incomplete new row cannot authorize its decoder.
    auto short_ranges=ranges;short_ranges[0].bytes=copy_bytes-1;ASSERT_TRUE(copy.prepare_replay(phases,short_ranges).ok());
    auto partial=copy.submit().value();ASSERT_TRUE(partial.wait().ok());
    auto short_update=c::CopyBatchUpdate::take(std::move(copy),std::move(partial)).value();
    auto rejected=gb::submit(receiver,lanes,std::move(short_update));ASSERT_FALSE(rejected);
    EXPECT_TRUE(rejected.error().IsOutOfRange())<<rejected.error().message();EXPECT_TRUE(short_update.valid());
    page.lease()=short_update.take_mutable({1}).value();copy=short_update.release_invocation().value();ASSERT_TRUE(copy.reset().ok());
    ASSERT_TRUE(copy.bind_source({0},source).ok());ASSERT_TRUE(copy.bind_mutable({1},std::move(page.lease())).ok());
    ASSERT_TRUE(copy.prepare_replay(phases,ranges).ok());
#endif
#if defined(HYLE_ROW_CACHE_GRAPH_CUDA)
    Gate gate;struct Drain{Gate&gate;c::Context&context;~Drain(){gate.release();context.drain().IgnoreError();}}drain{gate,context};
    auto producer=n::submit(queue,[&](cudaStream_t stream)noexcept{
      auto status=n::status(cudaLaunchHostFunc(stream,&Gate::wait,&gate));if(!status.ok())return status;
      return n::status(cudaMemcpyAsync(target,host_pointer,copy_bytes,cudaMemcpyHostToDevice,stream));}).value();
    ASSERT_TRUE(writing.attach(producer).ok());ASSERT_TRUE(gate.await());
    ASSERT_TRUE(copy.bind_producer({0},std::move(producer)).ok());ASSERT_TRUE(copy.prepare_replay(phases,ranges).ok());
#endif
    auto copied=copy.submit().value();auto update=c::CopyBatchUpdate::take(std::move(copy),std::move(copied)).value();
    auto done=gb::submit(receiver,lanes,std::move(update));ASSERT_TRUE(done)<<done.error().message();EXPECT_FALSE(update.valid());
#if defined(HYLE_ROW_CACHE_GRAPH_CUDA)
    ASSERT_TRUE(done->poll());EXPECT_EQ(*done->poll(),c::CompletionPoll::kPending);
    EXPECT_FALSE(receiver.result(recipe.output()));EXPECT_FALSE(gb::take_update(receiver));
    EXPECT_FALSE(arena.acquire(mutable_page));EXPECT_TRUE(arena.acquire(2));
    EXPECT_TRUE(prepare().IsUnavailable());
    gate.release();
#endif
    ASSERT_TRUE(done->wait().ok());auto output=receiver.result(recipe.output()).value();auto retained=output;
    const auto expected=f::expected(encoding,changed,ids);
    EXPECT_EQ(row_cache_graph_test::read<std::uint16_t>(output),expected);
    {const auto cast=row_cache_graph_test::read<std::uint32_t>(receiver.result(floats).value());
      const auto neg=row_cache_graph_test::read<std::uint32_t>(receiver.result(negative).value());
      std::size_t bad_cast=0,bad_neg=0;for(std::size_t i=0;i<cast.size();++i){const auto bits=std::uint32_t(expected[i])<<16;
        bad_cast+=cast[i]!=bits;bad_neg+=neg[i]!=(bits^0x80000000u);}
      EXPECT_EQ(bad_cast,0u);EXPECT_EQ(bad_neg,0u);}
    *done=c::Completion{};EXPECT_FALSE(gb::take_update(receiver));output.reset();EXPECT_FALSE(gb::take_update(receiver));retained.reset();
    update=f::take(gb::take_update(receiver),"return pending row update");page.lease()=update.take_mutable({1}).value();
    auto published=page.publish(required_rows).value();ready.push_back(published);EXPECT_TRUE(arena.read(version,ready));
    copy=update.release_invocation().value();ASSERT_TRUE(copy.reset().ok());frame={};wrong={};
    ASSERT_TRUE(prepare().ok());ASSERT_TRUE(receiver.drain().ok());
#if defined(HYLE_ROW_CACHE_GRAPH_CUDA)
    EXPECT_EQ(uses.active_batches(),0u);
#endif
  }
}
}
#if defined(HYLE_ROW_CACHE_GRAPH_CUDA)
TEST(CudaPagedRowCacheGraph, PendingProducerCopyKeepsTypedPagePermissionsAndPublishesSelectedRows){
  auto context=g::cuda::create_context().value();auto resource=c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice).value();
  exercise(context,resource.ref(),[&](const g::Graph&graph){return g::cuda::compile(graph,context,resource.ref());});
}
#else
TEST(CpuPagedRowCacheGraph, PendingCopyKeepsTypedPagePermissionsAndPublishesSelectedRows){
  auto context=g::cpu::create_context().value();exercise(context,hyle::mem::system_resource(),[&](const g::Graph&graph){return g::compile(graph,context);});
}
#endif
