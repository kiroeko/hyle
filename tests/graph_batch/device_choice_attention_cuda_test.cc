// Native math and captured Source dependencies for finite device layout choices.
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <span>
#include <thread>
#include <vector>
#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/batch.h"
#include "hyle/graph/cuda.h"
#include "hyle/graph/source.h"
#include "hyle/graph/detail/recording_access.h"

namespace {
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace gb=g::batch;
using hyle::Result;
using hyle::Status;
constexpr std::size_t depth=64,page_rows=16;
std::uint16_t word(float value){return std::uint16_t(std::bit_cast<std::uint32_t>(value)>>16);}
float number(std::uint16_t value){return std::bit_cast<float>(std::uint32_t(value)<<16);}

struct AttentionFixture final {
  c::Context context;
  c::Queue queue;
  c::StorageOwner storage;
  c::cuda::MemoryResource staging;
  c::BatchSchema schema;
  std::array<c::BatchLayoutVersion,2> layouts;
  c::PagedArena keys,values;
  std::vector<c::ReadLease> key_guards,value_guards;
  c::PagedArenaRead key_frame,value_frame;
  c::LeasePool inputs,outputs;
  c::ReadLease query,zero,one;
  c::TensorDesc query_description,selector_description,output_description;
  g::Graph graph;
  g::CompiledGraph compiled;
  std::array<g::ValueId,4> ports;
  g::NodeId reader;
  g::ValueId output;
  std::uint32_t heads=1;
  std::size_t page_size=page_rows,first_kv=3;
};

Result<c::BatchLayoutVersion> layout(AttentionFixture&f,std::size_t queries,std::size_t kv){
  const auto pages=(kv+f.page_size-1)/f.page_size;
  const std::array<std::int32_t,2> q{0,std::int32_t(queries)},ptr{0,std::int32_t(pages)};
  std::vector<std::int32_t> ids(pages);for(std::size_t i=0;i<pages;++i)ids[i]=std::int32_t(i);
  const std::array<std::int32_t,1> tail{std::int32_t((kv-1)%f.page_size+1)};
  return f.schema.version({q,ptr,ids,tail});
}

Result<AttentionFixture> fixture(bool draft,std::uint32_t heads=1,std::size_t page_size=page_rows,
    std::size_t first_kv=3){
  AttentionFixture f;f.heads=heads;f.page_size=page_size;f.first_kv=first_kv;
  HYLE_ASSIGN_OR_RETURN(f.context,g::cuda::create_context());
  HYLE_ASSIGN_OR_RETURN(f.queue,f.context.create_queue());
  HYLE_RETURN_IF_ERROR(f.queue.reserve_completions(16));
  HYLE_ASSIGN_OR_RETURN(auto device,c::cuda::MemoryResource::create(f.context,c::cuda::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(f.storage,c::StorageOwner::adopt(f.context,std::move(device)));
  HYLE_ASSIGN_OR_RETURN(f.staging,c::cuda::MemoryResource::create(f.context,c::cuda::MemoryKind::kPinnedHost));
  HYLE_ASSIGN_OR_RETURN(auto plain,c::CacheStorage::plain(c::DataType::kBFloat16));
  const c::BatchCapacity cap{.query_type=c::DataType::kBFloat16,.output_type=c::DataType::kBFloat16,
      .key_storage=plain,.value_storage=plain,.max_requests=1,.max_query_rows=2,.max_kv_rows=1024,
      .max_pages=64,.max_page_refs=64,.query_heads=heads,.kv_heads=1,
      .qk_head_dim=depth,.value_head_dim=depth,.page_size=std::uint32_t(page_size)};
  HYLE_ASSIGN_OR_RETURN(f.schema,c::BatchSchema::create(cap));
  HYLE_ASSIGN_OR_RETURN(f.layouts[0],layout(f,draft?1:2,first_kv));
  HYLE_ASSIGN_OR_RETURN(f.layouts[1],layout(f,2,first_kv+1));
  const auto arena_bytes=64*page_size*depth*sizeof(std::uint16_t)+1;
  HYLE_ASSIGN_OR_RETURN(f.keys,c::PagedArena::create(f.schema,c::BatchCacheRole::kKey,f.storage.ref(),{.max_backing_bytes=arena_bytes}));
  HYLE_ASSIGN_OR_RETURN(f.values,c::PagedArena::create(f.schema,c::BatchCacheRole::kValue,f.storage.ref(),{.max_backing_bytes=arena_bytes}));
  const auto initialized_rows=first_kv+3;
  for(std::size_t page=0;page*page_size<initialized_rows;++page){
    HYLE_ASSIGN_OR_RETURN(auto k,f.keys.acquire(std::uint32_t(page)));
    HYLE_ASSIGN_OR_RETURN(auto v,f.values.acquire(std::uint32_t(page)));
    std::vector<std::uint16_t> key_words(page_size*depth,0x7fc1),value_words(page_size*depth,0x7fc1);
    const auto rows=std::min(page_size,initialized_rows-page*page_size);
    for(std::size_t row=0;row<rows;++row)for(std::size_t d=0;d<depth;++d){
      key_words[row*depth+d]=0;value_words[row*depth+d]=word(float(page*page_size+row+1));
    }
    HYLE_RETURN_IF_ERROR(k.lease().view()->write(std::as_bytes(std::span(key_words))));
    HYLE_RETURN_IF_ERROR(v.lease().view()->write(std::as_bytes(std::span(value_words))));
    HYLE_ASSIGN_OR_RETURN(auto kr,k.publish(rows));HYLE_ASSIGN_OR_RETURN(auto vr,v.publish(rows));
    f.key_guards.push_back(std::move(kr));f.value_guards.push_back(std::move(vr));
  }
  const auto initial_pages=(first_kv+page_size)/page_size;
  HYLE_ASSIGN_OR_RETURN(f.key_frame,f.keys.read(f.layouts[1],std::span(f.key_guards).first(initial_pages)));
  HYLE_ASSIGN_OR_RETURN(f.value_frame,f.values.read(f.layouts[1],std::span(f.value_guards).first(initial_pages)));
  HYLE_ASSIGN_OR_RETURN(f.query_description,c::batch_query_description(cap,2));
  HYLE_ASSIGN_OR_RETURN(f.output_description,c::batch_output_description(cap,2));
  HYLE_ASSIGN_OR_RETURN(f.selector_description,c::TensorDesc::dense(c::DataType::kInt32,std::array<std::int64_t,1>{1}));
  HYLE_ASSIGN_OR_RETURN(auto qbytes,f.query_description.required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto obytes,f.output_description.required_bytes());
  HYLE_ASSIGN_OR_RETURN(f.inputs,c::LeasePool::create(f.storage.ref(),qbytes+512,{.max_blocks=3,.backing_alignment=256}));
  HYLE_ASSIGN_OR_RETURN(f.outputs,c::LeasePool::create(f.storage.ref(),obytes,{.max_blocks=1,.backing_alignment=256}));
  HYLE_ASSIGN_OR_RETURN(auto qw,f.inputs.allocate(qbytes,256));HYLE_RETURN_IF_ERROR(qw.view()->fill(std::byte{0}));
  HYLE_ASSIGN_OR_RETURN(f.query,qw.publish());
  for(std::int32_t choice=0;choice<2;++choice){
    HYLE_ASSIGN_OR_RETURN(auto sw,f.inputs.allocate(sizeof(choice),256));
    HYLE_RETURN_IF_ERROR(sw.view()->write(std::as_bytes(std::span(&choice,1))));
    HYLE_ASSIGN_OR_RETURN(auto ready,sw.publish());(choice?f.one:f.zero)=std::move(ready);
  }
  HYLE_ASSIGN_OR_RETURN(auto builder,g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(f.ports[0],builder.input(f.query_description));
  HYLE_ASSIGN_OR_RETURN(auto kd,c::batch_cache_description(f.schema,c::BatchCacheRole::kKey));
  HYLE_ASSIGN_OR_RETURN(auto vd,c::batch_cache_description(f.schema,c::BatchCacheRole::kValue));
  HYLE_ASSIGN_OR_RETURN(f.ports[1],builder.input(kd));HYLE_ASSIGN_OR_RETURN(f.ports[2],builder.input(vd));
  HYLE_ASSIGN_OR_RETURN(f.ports[3],builder.input(f.selector_description));
  HYLE_ASSIGN_OR_RETURN(auto op,gb::device_choice_paged_attention(f.schema,{.scale=.125f,.causal=true}));
  HYLE_ASSIGN_OR_RETURN(f.reader,builder.add_node(c::Operation::custom(op),f.ports));
  HYLE_ASSIGN_OR_RETURN(f.output,builder.result(f.reader));HYLE_RETURN_IF_ERROR(builder.publish(f.output));
  HYLE_ASSIGN_OR_RETURN(f.graph,std::move(builder).finish());
  const std::array domains{g::StorageDomain{f.storage.ref()}};
  HYLE_ASSIGN_OR_RETURN(auto source,g::ExecutionSource::acquire(f.context));
  const std::array targets{g::ExecutionTarget{source,g::StorageDomainId{0}}};
  HYLE_ASSIGN_OR_RETURN(f.compiled,gb::compile(f.graph,{.storage_domains=domains,.targets=targets}));
  return f;
}

Result<g::Invocation> prepared(AttentionFixture&f,std::int32_t selector){
  HYLE_ASSIGN_OR_RETURN(auto invocation,f.compiled.executable.create_invocation());
  HYLE_RETURN_IF_ERROR(invocation.prepare_resources());
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&f.queue}};
  const std::array choices{gb::NodeLayoutChoice{f.reader,f.layouts}};
  HYLE_RETURN_IF_ERROR(gb::prepare_choices(invocation,queues,choices,{.host_workspace=f.staging.ref()}));
  HYLE_RETURN_IF_ERROR(invocation.bind(f.ports[0],f.query_description,f.query));
  HYLE_RETURN_IF_ERROR(gb::bind(invocation,f.ports[1],f.key_frame));
  HYLE_RETURN_IF_ERROR(gb::bind(invocation,f.ports[2],f.value_frame));
  HYLE_RETURN_IF_ERROR(invocation.bind(f.ports[3],f.selector_description,selector?f.one:f.zero));
  HYLE_ASSIGN_OR_RETURN(auto bytes,f.output_description.required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto writer,f.outputs.allocate(bytes,256));
  // If active-Q1 leaves its second physical row uninitialized, this NaN
  // sentinel will survive and make the check fail.
  std::vector<std::uint16_t> poison(bytes/2,0x7fc1);
  HYLE_RETURN_IF_ERROR(writer.view()->write(std::as_bytes(std::span(poison))));
  HYLE_RETURN_IF_ERROR(invocation.bind(f.output,f.output_description,std::move(writer)));
  HYLE_RETURN_IF_ERROR(invocation.prepare_replay(queues,{.mode=g::ReplayMode::kRequireCapture,.host_wait=g::HostWaitPolicy::kReject}));
  return invocation;
}

void check_values(AttentionFixture&f,g::Invocation&invocation,bool draft,std::int32_t selector){
  std::vector<std::uint16_t> actual(2*f.heads*depth);
  auto tensor=invocation.output(f.output);ASSERT_TRUE(tensor)<<tensor.error().message();
  ASSERT_TRUE(tensor->storage().read(std::as_writable_bytes(std::span(actual))).ok());
  const auto length=float(f.first_kv+(selector?1:0));
  const std::array<float,2> expected=draft&&!selector?std::array<float,2>{(length+1)/2,0}:
      std::array<float,2>{length/2,(length+1)/2};
  for(std::size_t row=0;row<2;++row)for(std::size_t i=0;i<f.heads*depth;++i){
    const auto value=number(actual[row*f.heads*depth+i]);
    EXPECT_FLOAT_EQ(value,expected[row])<<"physical row="<<row<<" element="<<i;
  }
}
void run_and_check(AttentionFixture&f,g::Invocation&invocation,bool draft,std::int32_t selector){
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&f.queue}};
  auto submitted=invocation.submit(queues);ASSERT_TRUE(submitted)<<submitted.error().message();
  const auto status=submitted->wait();ASSERT_TRUE(status.ok())<<status.message();
  *submitted={};check_values(f,invocation,draft,selector);
}

void native_case(bool draft,std::int32_t selector){
  auto made=fixture(draft);ASSERT_TRUE(made)<<made.error().message();auto&f=*made;
  auto ready=prepared(f,selector);ASSERT_TRUE(ready)<<ready.error().message();
  run_and_check(f,*ready,draft,selector);
}
TEST(DeviceChoicePagedAttentionCuda, TargetQ2RejectLayoutHasShorterKvMean){native_case(false,0);}
TEST(DeviceChoicePagedAttentionCuda, TargetQ2AcceptLayoutHasLongerKvMean){native_case(false,1);}
TEST(DeviceChoicePagedAttentionCuda, DraftQ1InitializesInactivePhysicalOutput){native_case(true,0);}
TEST(DeviceChoicePagedAttentionCuda, DraftQ2UsesBothActiveRows){native_case(true,1);}

TEST(DeviceChoicePagedAttentionCuda, ForeignSchemaRejectsBeforeReplacingPreparedCapture){
  auto made=fixture(false);ASSERT_TRUE(made)<<made.error().message();auto&f=*made;
  auto ready=prepared(f,0);ASSERT_TRUE(ready)<<ready.error().message();
  auto foreign=c::BatchSchema::create(f.schema.capacity());ASSERT_TRUE(foreign);
  std::array<c::BatchLayoutVersion,2> layouts;
  for(std::size_t i=0;i<2;++i){auto layout=foreign->version(f.layouts[i].host());ASSERT_TRUE(layout);layouts[i]=std::move(*layout);}
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&f.queue}};
  const std::array bad{gb::NodeLayoutChoice{f.reader,layouts}};
  const auto rejected=gb::prepare_choices(*ready,queues,bad,{.host_workspace=f.staging.ref()});
  EXPECT_TRUE(rejected.IsInvalidArgument())<<rejected.message();
  ASSERT_EQ(ready->replay_route(),g::ReplayRoute::kCaptured);
  // Submit directly: no repair/reprepare is allowed to hide a lost old binding.
  run_and_check(f,*ready,false,0);
}

TEST(DeviceChoicePagedAttentionCuda, UnsupportedMultiQueryTileReturnsTypedRefusal){
  // The current provider has no qualified plan for this multi-query-tile
  // domain. Report the initial typed refusal without claiming a capture.
  auto made=fixture(false,256);ASSERT_TRUE(made)<<made.error().message();auto&f=*made;
  auto ready=prepared(f,0);ASSERT_FALSE(ready);
  EXPECT_TRUE(ready.error().IsUnimplemented())<<ready.error().message();
}

TEST(DeviceChoicePagedAttentionCuda, DifferentChunkCountsReturnFalseAndPreserveOldCapture){
  auto made=fixture(false,1,128,127);ASSERT_TRUE(made)<<made.error().message();auto&f=*made;
  auto ready=prepared(f,0);ASSERT_TRUE(ready)<<ready.error().message();
  auto a=layout(f,2,128),b=layout(f,2,129);ASSERT_TRUE(a);ASSERT_TRUE(b);
  const std::array choices{gb::NodeLayoutChoice{f.reader,{*a,*b}}};
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&f.queue}};
  const auto refused=gb::try_prepare_choices(*ready,queues,choices,{.host_workspace=f.staging.ref()});
  ASSERT_TRUE(refused)<<refused.error().message();EXPECT_FALSE(*refused);
  EXPECT_EQ(ready->replay_route(),g::ReplayRoute::kCaptured);
  run_and_check(f,*ready,false,0);
}

struct AttentionGate final {
  std::atomic<bool> open{false},timed_out{false};
  std::mutex mutex;std::condition_variable changed;std::thread watchdog;
  cudaStream_t stream=nullptr;
  static void CUDART_CB wait(void*opaque){static_cast<AttentionGate*>(opaque)->open.wait(false,std::memory_order_acquire);}
  void arm(cudaStream_t value){
    stream=value;watchdog=std::thread([this]{
      std::unique_lock lock(mutex);
      if(!changed.wait_for(lock,std::chrono::seconds(10),[&]{return open.load();})){
        timed_out.store(true);release();
      }
    });
  }
  void release()noexcept{open.store(true,std::memory_order_release);open.notify_all();changed.notify_all();}
  ~AttentionGate(){release();if(watchdog.joinable())watchdog.join();if(stream)EXPECT_EQ(cudaStreamSynchronize(stream),cudaSuccess);}
};

void retarget_pending(bool accepted){
  auto made=fixture(false,1,128,127);ASSERT_TRUE(made)<<made.error().message();auto&f=*made;
  const auto scalar_f=c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,1>{1}).value();
  auto apool=c::LeasePool::create(f.storage.ref(),512,{.max_blocks=2}).value();
  auto source_pool=c::LeasePool::create(f.storage.ref(),256,{.max_blocks=1}).value();
  const auto input=[&](float value){
    auto w=apool.allocate(sizeof(value),256).value();
    EXPECT_TRUE(w.view()->write(std::as_bytes(std::span(&value,1))).ok());return w.publish().value();
  };
  auto draft=input(accepted?5.f:6.f),target=input(5.f);
  auto builder=g::GraphBuilder::create().value();
  const auto d=builder.input(scalar_f).value(),t=builder.input(scalar_f).value();
  const auto z=builder.input(f.selector_description).value(),o=builder.input(f.selector_description).value();
  const auto difference=builder.add(c::Binary{c::BinaryKind::kSubtract},d,t).value();
  const auto mismatch=builder.add(c::Cast{c::DataType::kBool},difference).value();
  const auto prefix=builder.add(c::Select{},mismatch,z,o).value();
  ASSERT_TRUE(builder.publish(prefix).ok());auto graph=std::move(builder).finish().value();
  auto compiled=g::cuda::compile(graph,f.context,f.storage);ASSERT_TRUE(compiled)<<compiled.error().message();
  auto a=compiled->executable.create_invocation().value();ASSERT_TRUE(a.prepare_resources().ok());
  ASSERT_TRUE(a.bind(d,scalar_f,draft).ok());ASSERT_TRUE(a.bind(t,scalar_f,target).ok());
  ASSERT_TRUE(a.bind(z,f.selector_description,f.zero).ok());ASSERT_TRUE(a.bind(o,f.selector_description,f.one).ok());
  auto writer=source_pool.allocate(sizeof(std::int32_t),256).value();
  const auto coordinates=c::ConstTensorView::create(f.selector_description,writer.view().value()).value();
  ASSERT_TRUE(a.bind(prefix,f.selector_description,std::move(writer)).ok());
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&f.queue}};
  ASSERT_TRUE(a.prepare_replay(queues,{.mode=g::ReplayMode::kRequireCapture,.host_wait=g::HostWaitPolicy::kAllow}).ok());

  auto b=f.compiled.executable.create_invocation().value();ASSERT_TRUE(b.prepare_resources().ok());
  const std::array old{gb::NodeLayoutChoice{f.reader,f.layouts}};
  const auto old_prepared=gb::prepare_choices(b,queues,old,{.host_workspace=f.staging.ref()});
  ASSERT_TRUE(old_prepared.ok())<<old_prepared.message();
  ASSERT_TRUE(b.bind(f.ports[0],f.query_description,f.query).ok());
  ASSERT_TRUE(gb::bind(b,f.ports[1],f.key_frame).ok());ASSERT_TRUE(gb::bind(b,f.ports[2],f.value_frame).ok());
  auto output_writer=f.outputs.allocate(f.output_description.required_bytes().value(),256).value();
  ASSERT_TRUE(b.bind(f.output,f.output_description,std::move(output_writer)).ok());
  ASSERT_TRUE(g::detail::RecordingAccess::begin(b).ok());
  ASSERT_TRUE(g::detail::RecordingAccess::bind_input(b,f.ports[3],coordinates).ok());
  const auto recorded=b.prepare_replay(queues,{.mode=g::ReplayMode::kRequireCapture,.host_wait=g::HostWaitPolicy::kAllow});
  ASSERT_TRUE(recorded.ok())<<recorded.message();
  ASSERT_EQ(b.replay_route(),g::ReplayRoute::kCaptured);
  const auto ranges=b.replay_report()->recorded_capture_range_count;ASSERT_GT(ranges,0u);
  ASSERT_TRUE(g::detail::RecordingAccess::end(b).ok());

  f.layouts[0]=layout(f,2,129).value();f.layouts[1]=layout(f,2,130).value();f.first_kv=129;
  const std::array next{gb::NodeLayoutChoice{f.reader,f.layouts}};
  const auto refreshed=gb::try_prepare_choices(b,queues,next,{.host_workspace=f.staging.ref()});
  ASSERT_TRUE(refreshed)<<refreshed.error().message();ASSERT_TRUE(*refreshed);
  const auto report=gb::report(b,f.reader);ASSERT_TRUE(report)<<report.error().message();
  EXPECT_TRUE(report->reused_native_plan); // The real planner refreshed its integer schedule.
  f.key_frame=f.keys.read(f.layouts[1],f.key_guards).value();
  f.value_frame=f.values.read(f.layouts[1],f.value_guards).value();
  ASSERT_TRUE(gb::bind(b,f.ports[1],f.key_frame).ok());ASSERT_TRUE(gb::bind(b,f.ports[2],f.value_frame).ok());

  AttentionGate gate;gate.arm(c::cuda::stream(f.queue).value());
  ASSERT_EQ(cudaLaunchHostFunc(gate.stream,&AttentionGate::wait,&gate),cudaSuccess);
  auto first=a.submit(queues);ASSERT_TRUE(first)<<first.error().message();
  auto common=g::SubmissionSource::create(a,std::move(*first),f.storage);ASSERT_TRUE(common)<<common.error().message();
  auto selected=g::OutputSource::create(*common,prefix,f.storage,source_pool);ASSERT_TRUE(selected)<<selected.error().message();
  ASSERT_EQ(common->poll().value(),c::CompletionPoll::kPending);
  const std::array bindings{selected->reader_binding(f.ports[3])};
  const auto bound=g::bind_sources(b,bindings);ASSERT_TRUE(bound.ok())<<bound.message();
  const auto prepared=g::prepare_sources(b,queues);ASSERT_TRUE(prepared.ok())<<prepared.message();
  ASSERT_EQ(b.replay_route(),g::ReplayRoute::kCaptured);
  EXPECT_EQ(b.replay_report()->recorded_capture_range_count,0u);
  EXPECT_EQ(b.replay_report()->reused_capture_range_count,ranges);
  const auto sources=b.input_source_report().value();
  EXPECT_EQ(sources.host_wait_count,0u);EXPECT_EQ(sources.device_wait_count,1u);
  auto finished=g::submit_sources(b,queues);ASSERT_TRUE(finished)<<finished.error().message();
  EXPECT_FALSE(gate.open.load());ASSERT_EQ(common->poll().value(),c::CompletionPoll::kPending);
  gate.release();const auto status=finished->wait();ASSERT_TRUE(status.ok())<<status.message();
  *finished={};check_values(f,b,false,accepted?1:0);
  EXPECT_FALSE(gate.timed_out.load());ASSERT_TRUE(g::release_sources(b).ok());
}
TEST(DeviceChoicePagedAttentionCuda, CommonKvGrowthRetainsCaptureRejectWhileProducerPending){retarget_pending(false);}
TEST(DeviceChoicePagedAttentionCuda, CommonKvGrowthRetainsCaptureAcceptWhileProducerPending){retarget_pending(true);}
} // namespace
