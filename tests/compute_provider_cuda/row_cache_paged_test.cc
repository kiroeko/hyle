#include "hyle/compute/cuda.h"
#include "hyle/compute/backends/cuda.h"
#include "../compute_models/row_cache_paged_fixture.h"
#include <chrono>
#include <condition_variable>
#include <mutex>
TEST(CudaPagedRowCache, ReadyFramesMapRequestLogicalIdsAndDecodeOnlySelectedRows){
  auto context=hyle::compute::cuda::create_context().value();
  auto resource=hyle::compute::cuda::MemoryResource::create(context,hyle::compute::cuda::MemoryKind::kDevice).value();
  row_cache_paged_test::exercise(context,resource.ref());
}
namespace {
struct RowGate final {
  std::mutex mutex;std::condition_variable condition;bool entered=false,released=false;
  static void CUDART_CB wait(void*raw){auto&g=*static_cast<RowGate*>(raw);std::unique_lock lock(g.mutex);g.entered=true;g.condition.notify_all();g.condition.wait(lock,[&]{return g.released;});}
  bool await(){std::unique_lock lock(mutex);return condition.wait_for(lock,std::chrono::seconds(3),[&]{return entered;});}
  void release(){std::lock_guard lock(mutex);released=true;condition.notify_all();}
};
}
TEST(CudaPagedRowCache, PendingAndAcceptedPrefixKeepExactPagesAndNativeOwnersAlive){
  namespace c=hyle::compute;namespace n=c::cuda;namespace m=hyle::mem;namespace f=row_cache_paged_test;
  auto context=n::create_context().value();auto resource=n::MemoryResource::create(context,n::MemoryKind::kDevice).value();
  const auto encoding=c::RowCacheEncoding::float4_block32();
  for(bool reject_decode:{false,true}){
    SCOPED_TRACE(reject_decode);auto queue=context.create_queue().value();
    auto schema=c::RowCacheBatchSchema::create(f::capacity(encoding)).value();auto version=f::layout(schema);
    auto arena=c::PagedArena::create(schema,resource.ref(),{.max_backing_bytes=10*encoding.row_bytes()+1}).value();
    const auto reference=f::data(encoding);auto ready=f::pages(arena,encoding,reference);auto frame=f::frame(arena,version,ready);
    const auto desc=c::TensorDesc::dense(c::DataType::kInt32,std::array<std::int64_t,2>{4,3}).value();
    auto slot=c::PagedRowCacheGatherInvocation::create(context,queue,schema,desc,{.resource=resource.ref()}).value();ASSERT_TRUE(slot.prepare_layout(version).ok());
    auto pool=c::LeasePool::create(resource.ref(),16384).value();auto ids_writer=pool.allocate(48).value();auto output=pool.allocate(12*encoding.depth()*2).value();
    auto host=c::LeasePool::create(m::system_resource(),8192).value();auto rejected_output=host.allocate(12*encoding.depth()*2).value();
    const std::vector<std::uint16_t> sentinel(12*encoding.depth(),0x5a5a);auto rejected_raw=rejected_output.view().value();ASSERT_TRUE(rejected_raw.write(std::as_bytes(std::span(sentinel))).ok());
    // Actual device work creates the IDs. Publish only after its original
    // write use retires; the ready-reader never invents pending permission.
    const auto pointer=n::pointer<std::int32_t>(ids_writer.view().value()).value();
    auto uses=c::LeaseUsePool::create().value();auto writing=uses.begin().value();ASSERT_TRUE(writing.write(ids_writer).ok());
    {auto generated=n::submit(queue,[&](cudaStream_t stream)noexcept{return n::status(cudaMemsetAsync(pointer,0,48,stream));}).value();
      ASSERT_TRUE(writing.attach(generated).ok());ASSERT_TRUE(generated.wait().ok());}
    auto ids=ids_writer.publish().value();
    // Warm the exact map/checked-codec kernels and Queue status storage before
    // blocking this stream, so cold CUDA module setup cannot hide the proof.
    {auto warm=slot.submit(frame,ids,output);ASSERT_TRUE(warm);ASSERT_TRUE(warm->wait().ok());}
    RowGate gate;struct Drain{RowGate&gate;c::Context&context;~Drain(){gate.release();context.drain().IgnoreError();}}drain{gate,context};
    auto producer=n::submit(queue,[&](cudaStream_t stream)noexcept{return n::status(cudaLaunchHostFunc(stream,&RowGate::wait,&gate));}).value();
    ASSERT_TRUE(gate.await());
    auto done=slot.submit(frame,ids,reject_decode?rejected_output:output);ASSERT_TRUE(done)<<done.error().message();
    const auto poll=done->poll();if(reject_decode){if(poll)EXPECT_EQ(*poll,c::CompletionPoll::kPending);else EXPECT_TRUE(poll.error().IsInvalidArgument());}
    else {ASSERT_TRUE(poll);EXPECT_EQ(*poll,c::CompletionPoll::kPending);}
    frame={};ready.clear();ids.reset();
    EXPECT_FALSE(arena.acquire(0));EXPECT_FALSE(arena.acquire(4));EXPECT_TRUE(arena.acquire(2));
    EXPECT_TRUE(slot.close().IsUnavailable());
    // No caller frame/arena/reader survives the checkpoint. Accepted original
    // page uses and native invocation retention must preserve their backing.
    slot={};arena={};schema={};version={};pool={};producer={};
    gate.release();const auto terminal=done->wait();
    if(reject_decode){EXPECT_TRUE(terminal.IsInvalidArgument());std::vector<std::uint16_t> actual(sentinel.size());
      ASSERT_TRUE(rejected_raw.read(std::as_writable_bytes(std::span(actual))).ok());EXPECT_EQ(actual,sentinel);EXPECT_FALSE(rejected_output.publish());}
    else {ASSERT_TRUE(terminal.ok());const std::array<std::int32_t,12> selected{};EXPECT_EQ(f::read(output,12*encoding.depth()),f::expected(encoding,reference,selected));}
  }
}
