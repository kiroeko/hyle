// Migrated Source/delta/retirement regression cases; physical Q2 and variable I32 lengths.
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
#include "hyle/compute/detail/lease_access.h"
#include "hyle/compute/paged_frame.h"
#include "hyle/graph/batch.h"
#include "hyle/graph/cuda.h"
#include "hyle/graph/detail/recording_access.h"
#include "hyle/graph/paged_append_source.h"
#include "hyle/graph/source.h"

namespace {
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace gb=g::batch;

struct AppendLengthsGate final {
  cudaStream_t stream=nullptr;
  std::atomic<bool> open{false},timed_out{false};
  std::mutex mutex;std::condition_variable changed;std::thread watchdog;
  static void CUDART_CB wait(void*opaque){static_cast<AppendLengthsGate*>(opaque)->open.wait(false,std::memory_order_acquire);}
  void arm(cudaStream_t value){stream=value;watchdog=std::thread([this]{
    std::unique_lock lock(mutex);
    if(!changed.wait_for(lock,std::chrono::seconds(10),[&]{return open.load();})){
      timed_out.store(true);release();
    }
  });}
  void release()noexcept{open.store(true,std::memory_order_release);open.notify_all();changed.notify_all();}
  ~AppendLengthsGate(){release();if(watchdog.joinable())watchdog.join();if(stream)EXPECT_EQ(cudaStreamSynchronize(stream),cudaSuccess);}
};

std::uint16_t bf16(float value){return static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(value)>>16);}

void raw_prefix_journey(std::int32_t selector,bool delta_only_predecessor=false){
  constexpr std::size_t logical_rows=1;
  constexpr bool cancel=false;
  SCOPED_TRACE(selector);
  SCOPED_TRACE(logical_rows);
  SCOPED_TRACE(cancel);
  auto context_result=g::cuda::create_context();
  ASSERT_TRUE(context_result)<<context_result.error().message();
  auto context=std::move(*context_result);auto queue=context.create_queue().value();
  ASSERT_TRUE(queue.reserve_completions(16).ok());
  auto device=c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice).value();
  auto storage=c::StorageOwner::adopt(context,std::move(device)).value();
  auto host=c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kPinnedHost).value();
  const auto plain=c::CacheStorage::plain(c::DataType::kBFloat16).value();
  const c::BatchCapacity capacity{.storage=c::BatchStorage::kPaged,
    .query_type=c::DataType::kBFloat16,.output_type=c::DataType::kBFloat16,
    .key_storage=plain,.value_storage=plain,.max_requests=1,.max_query_rows=8,.max_kv_rows=8,
    .max_pages=6,.max_page_refs=2,.query_heads=1,.kv_heads=1,.qk_head_dim=64,.value_head_dim=64,.page_size=4};
  auto schema=c::BatchSchema::create(capacity).value();
  const std::array<std::int32_t,2> q0{0,0},q2{0,2},ptr0{0,0},ptr1{0,1},ptr2{0,2};
  const std::array<std::int32_t,1> old_id{0},a_id{1},b_first_id{2};
  const std::array<std::int32_t,2> b_ids{2,3};
  const auto version=[&](auto query,auto ptr,auto ids,std::int32_t tail){
    const std::array<std::int32_t,1> last{tail};return schema.version({query,ptr,ids,last}).value();
  };
  const auto initial_layout=version(q0,ptr1,old_id,2);
  const auto a_layout=version(q2,ptr1,a_id,4);
  const auto empty_layout=version(q0,ptr0,std::span<const std::int32_t>{},0);
  const std::array<std::int32_t,2> b_query{0,static_cast<std::int32_t>(logical_rows)};
  const auto before0=version(q0,ptr1,a_id,3),before1=version(q0,ptr1,a_id,4);
  const auto after0=logical_rows==1?version(b_query,ptr1,b_first_id,4):version(b_query,ptr2,b_ids,1);
  const auto after1=version(b_query,ptr2,b_ids,static_cast<std::int32_t>(logical_rows));
  const std::array previous_prefixes{before0,before1},accepted_prefixes{after0,after1};
  const auto bounded_next=version(q2,ptr2,b_ids,1);
  // Device allocation/free must stay outside every original-Queue host gate.
  // Keep this unrelated real writer/backing through C's actual retirement.
  c::PagedArena other_arena;c::PagedPageWriter replacement;
  if(delta_only_predecessor){
    other_arena=c::PagedArena::create(schema,c::BatchCacheRole::kKey,storage.ref(),{.max_backing_bytes=8192}).value();
    replacement=other_arena.acquire(static_cast<std::uint32_t>(b_first_id[0])).value();
  }
  auto arena=c::PagedArena::create(schema,c::BatchCacheRole::kKey,storage.ref(),{.max_backing_bytes=8192}).value();
  auto initial_writer=arena.acquire(0).value();
  std::array<std::uint16_t,256> original{};
  std::fill_n(original.begin(),64,bf16(10));std::fill_n(original.begin()+64,64,bf16(20));
  std::fill(original.begin()+128,original.end(),std::uint16_t{0x7fc1});
  ASSERT_TRUE(initial_writer.lease().view()->write(std::as_bytes(std::span(original))).ok());
  auto initial_guard=initial_writer.publish(2).value();
  auto initial=arena.read(initial_layout,std::span(&initial_guard,1)).value();
  auto empty=arena.read(empty_layout,{}).value();
  auto payload=c::LeasePool::create(storage.ref(),6*256,{.max_blocks=6}).value();
  auto scalar_pool=c::LeasePool::create(storage.ref(),256,{.max_blocks=1}).value();
  const auto upload=[&](auto values){
    auto writer=payload.allocate(sizeof(values),256).value();
    EXPECT_TRUE(writer.view()->write(std::as_bytes(std::span(values))).ok());
    return writer.publish().value();
  };
  std::array<std::uint16_t,128> a_words{},b_words{};
  std::fill_n(a_words.begin(),64,bf16(30));std::fill(a_words.begin()+64,a_words.end(),bf16(40));
  std::fill_n(b_words.begin(),64,bf16(50));std::fill(b_words.begin()+64,b_words.end(),bf16(60));
  auto a_rows=upload(a_words),b_rows=upload(b_words);
  auto scalar_input=upload(std::array<std::int32_t,1>{4+selector});
  auto query_count=upload(std::array<std::int32_t,1>{1});
  auto final_length=upload(std::array<std::int32_t,1>{5+selector});
  const auto rows=c::TensorDesc::dense(c::DataType::kBFloat16,std::array<std::int64_t,3>{2,1,64}).value();
  const auto scalar=c::TensorDesc::dense(c::DataType::kInt32,std::array<std::int64_t,1>{1}).value();
  auto a_builder=g::GraphBuilder::create().value();
  const auto a_input=a_builder.input(rows).value(),a_scalar=a_builder.input(scalar).value();
  const auto a_append=a_builder.add_node(c::Operation::custom(gb::paged_append(schema,c::BatchCacheRole::kKey).value()),std::array{a_input}).value();
  const auto selected=a_builder.add(c::Copy{},a_scalar).value();
  ASSERT_TRUE(a_builder.publish(selected).ok());
  const auto a_graph=std::move(a_builder).finish().value();
  auto a_plan=g::cuda::compile(a_graph,context,storage);
  ASSERT_TRUE(a_plan)<<a_plan.error().message();
  auto a=a_plan->executable.create_invocation().value();
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&queue}};
  const gb::PrepareOptions options{.host_workspace=host.ref()};
  ASSERT_TRUE(gb::prepare(a,queues,{},options).ok());
  std::array<c::PagedPageWriter,1> a_dirty{arena.acquire(1).value()};
  auto a_pages=gb::PagedAppendPages::create(arena,a_layout,initial,a_dirty).value();
  ASSERT_TRUE(gb::bind_append(a,a_graph,a_append,std::move(a_pages)).ok());
  ASSERT_TRUE(a.bind(a_input,rows,a_rows).ok());ASSERT_TRUE(a.bind(a_scalar,scalar,scalar_input).ok());
  auto selected_writer=scalar_pool.allocate(sizeof(std::int32_t),256).value();
  const auto coordinates=c::ConstTensorView::create(scalar,c::detail::LeaseAccess::backing_span(scalar_pool).value().subview(0,sizeof(std::int32_t)).value()).value();
  ASSERT_TRUE(a.bind(selected,scalar,std::move(selected_writer)).ok());
  auto page_source=gb::PagedAppendSource::prepare(a,a_append,storage).value();

  auto b_builder=g::GraphBuilder::create().value();
  const auto b_input=b_builder.input(rows).value(),b_kv=b_builder.input(scalar).value(),b_queries=b_builder.input(scalar).value();
  const auto b_append=b_builder.add_node(c::Operation::custom(gb::device_lengths_paged_append(schema,c::BatchCacheRole::kKey).value()),std::array{b_input,b_queries,b_kv}).value();
  const auto b_graph=std::move(b_builder).finish().value();
  auto b_plan=g::cuda::compile(b_graph,context,storage);
  ASSERT_TRUE(b_plan)<<b_plan.error().message();
  auto b=b_plan->executable.create_invocation().value();
  ASSERT_TRUE(gb::prepare(b,queues,{},options).ok());
  ASSERT_TRUE(b.bind(b_input,rows,b_rows).ok());
  ASSERT_TRUE(b.bind(b_queries,scalar,query_count).ok());
  // Record with genuine empty ready prefixes, never a fake pending Source.
  // Cold has one page; runtime adds prefix copies and another COW page.
  const auto cold_next=version(q2,ptr1,b_first_id,2);
  const std::array cold_sources{c::PagedArenaFootprint::from_read(empty).value()};
  std::array<c::PagedPageWriter,1> cold_dirty{arena.acquire(2).value()};
  auto invalid=gb::PagedAppendPages::create_lengths(arena,cold_next,initial_layout,empty_layout,cold_sources,cold_dirty);
  EXPECT_FALSE(invalid);EXPECT_TRUE(cold_dirty[0].valid() && cold_dirty[0].lease().valid());
  auto cold_pages=gb::PagedAppendPages::create_lengths(arena,cold_next,empty_layout,empty_layout,cold_sources,cold_dirty);
  ASSERT_TRUE(cold_pages)<<cold_pages.error().message();
  EXPECT_FALSE(cold_pages->select_lengths(std::array<std::int32_t,1>{1},std::array<std::int32_t,1>{4+selector}).ok());
  ASSERT_TRUE(gb::bind_append(b,b_graph,b_append,std::move(*cold_pages)).ok());
  ASSERT_TRUE(g::detail::RecordingAccess::begin(b).ok());
  ASSERT_TRUE(g::detail::RecordingAccess::bind_input(b,b_kv,coordinates).ok());
  const auto recorded=b.prepare_replay(queues,{.mode=g::ReplayMode::kRequireCapture,.host_wait=g::HostWaitPolicy::kAllow});
  ASSERT_TRUE(recorded.ok())<<recorded.message();
  auto cold_returned=gb::take_append(b,b_append);
  ASSERT_TRUE(cold_returned)<<cold_returned.error().message();
  *cold_returned={};
  ASSERT_TRUE(g::detail::RecordingAccess::end(b).ok());
  ASSERT_TRUE(a.prepare_replay(queues,{.mode=g::ReplayMode::kRequireCapture,.host_wait=g::HostWaitPolicy::kReject}).ok());

  std::array<std::uint16_t,128> c_words{};
  std::fill_n(c_words.begin(),64,bf16(70));std::fill(c_words.begin()+64,c_words.end(),bf16(80));
  auto c_rows=upload(c_words);
  auto c_builder=g::GraphBuilder::create().value();
  const auto c_input=c_builder.input(rows).value(),c_kv=c_builder.input(scalar).value(),c_query=c_builder.input(scalar).value();
  const auto c_append=c_builder.add_node(c::Operation::custom(gb::device_lengths_paged_append(schema,c::BatchCacheRole::kKey).value()),std::array{c_input,c_query,c_kv}).value();
  const auto c_graph=std::move(c_builder).finish().value();
  auto c_plan=g::cuda::compile(c_graph,context,storage);ASSERT_TRUE(c_plan)<<c_plan.error().message();
  auto consumer=c_plan->executable.create_invocation().value();
  ASSERT_TRUE(gb::prepare(consumer,queues,{},options).ok());
  ASSERT_TRUE(consumer.bind(c_input,rows,c_rows).ok());
  ASSERT_TRUE(consumer.bind(c_kv,scalar,final_length).ok());
  ASSERT_TRUE(consumer.bind(c_query,scalar,query_count).ok());
  const std::array<std::int32_t,1> c_id{4};
  const auto cold_c_next=version(q2,ptr1,c_id,2);
  std::array<c::PagedPageWriter,1> cold_c_dirty{arena.acquire(4).value()};
  auto cold_c_pages=gb::PagedAppendPages::create_lengths(arena,cold_c_next,empty_layout,empty_layout,cold_sources,cold_c_dirty);
  ASSERT_TRUE(cold_c_pages)<<cold_c_pages.error().message();
  ASSERT_TRUE(gb::bind_append(consumer,c_graph,c_append,std::move(*cold_c_pages)).ok());
  auto cold_source=gb::PagedAppendSource::prepare(consumer,c_append,storage);
  ASSERT_TRUE(cold_source);*cold_source={};
  ASSERT_TRUE(g::detail::RecordingAccess::begin(consumer).ok());
  ASSERT_TRUE(consumer.prepare_replay(queues,{.mode=g::ReplayMode::kRequireCapture,.host_wait=g::HostWaitPolicy::kAllow}).ok());
  auto cold_c_returned=gb::take_append(consumer,c_append);ASSERT_TRUE(cold_c_returned);
  *cold_c_returned={};ASSERT_TRUE(g::detail::RecordingAccess::end(consumer).ok());
  // Poison raw backing before admitting A. Only the missing envelope row may
  // become zero; bytes beyond the advertised maximum remain untouched.
  std::array<c::PagedPageWriter,2> dirty{arena.acquire(2).value(),arena.acquire(3).value()};
  std::array<std::uint16_t,256> poison;poison.fill(0x7fc1);
  for(auto&writer:dirty)ASSERT_TRUE(writer.lease().view()->write(std::as_bytes(std::span(poison))).ok());

  AppendLengthsGate a_gate;a_gate.arm(c::cuda::stream(queue).value());
  ASSERT_EQ(cudaLaunchHostFunc(a_gate.stream,&AppendLengthsGate::wait,&a_gate),cudaSuccess);
  auto submitted=a.submit(queues);ASSERT_TRUE(submitted)<<submitted.error().message();
  auto common_result=g::SubmissionSource::create(a,std::move(*submitted),storage);ASSERT_TRUE(common_result);
  auto common=std::move(*common_result);
  auto scalar_result=g::OutputSource::create(common,selected,storage,scalar_pool);ASSERT_TRUE(scalar_result);
  auto scalar_source=std::move(*scalar_result);
  ASSERT_TRUE(page_source.capture(common).ok());
  ASSERT_EQ(common.poll().value(),c::CompletionPoll::kPending);
  auto refused=gb::PagedAppendPages::create_lengths(arena,bounded_next,initial_layout,page_source,dirty);
  ASSERT_FALSE(refused);EXPECT_TRUE(refused.error().IsInvalidArgument());
  EXPECT_TRUE(dirty[0].valid());EXPECT_TRUE(dirty[1].valid());
  auto pending=gb::PagedAppendPages::create_lengths(arena,bounded_next,before0,page_source,dirty);
  ASSERT_TRUE(pending)<<pending.error().message();
  EXPECT_FALSE(pending->select_lengths(std::array<std::int32_t,1>{1},std::array<std::int32_t,1>{4+selector}).ok());
  ASSERT_TRUE(gb::bind_append(b,b_graph,b_append,std::move(*pending)).ok());
  auto b_source_result=gb::PagedAppendSource::prepare(b,b_append,storage);
  ASSERT_TRUE(b_source_result)<<b_source_result.error().message();
  auto b_source=std::move(*b_source_result);
  auto effect=gb::append_binding(b,b_append,page_source);ASSERT_TRUE(effect);
  const std::array bindings{scalar_source.reader_binding(b_kv),*effect};
  ASSERT_TRUE(g::bind_sources(b,bindings).ok());
  ASSERT_TRUE(g::prepare_sources(b,queues).ok());
  EXPECT_EQ(b.replay_report()->recorded_capture_range_count,0u);
  EXPECT_EQ(b.replay_report()->reused_capture_range_count,1u);
  EXPECT_EQ(b.input_source_report()->host_wait_count,0u);
  EXPECT_EQ(b.input_source_report()->device_wait_count,1u);
  AppendLengthsGate b_gate;b_gate.arm(a_gate.stream);
  ASSERT_EQ(cudaLaunchHostFunc(b_gate.stream,&AppendLengthsGate::wait,&b_gate),cudaSuccess);
  auto b_submitted=g::submit_sources(b,queues);ASSERT_TRUE(b_submitted)<<b_submitted.error().message();
  auto b_common_result=g::SubmissionSource::create(b,std::move(*b_submitted),storage);ASSERT_TRUE(b_common_result);
  auto b_common=std::move(*b_common_result);
  // B's Source is captured before A's length has been observed on the host.
  ASSERT_TRUE(b_source.capture(b_common).ok());
  EXPECT_EQ(common.poll().value(),c::CompletionPoll::kPending);
  EXPECT_EQ(b_common.poll().value(),c::CompletionPoll::kPending);
  EXPECT_FALSE(a_gate.open.load());
  a_gate.release();ASSERT_TRUE(common.wait().ok());
  ASSERT_TRUE(scalar_source.detach().ok());ASSERT_TRUE(common.detach().ok());
  std::array<std::int32_t,1> observed{};
  ASSERT_TRUE(a.output(selected)->storage().read(std::as_writable_bytes(std::span(observed))).ok());
  ASSERT_EQ(observed[0],4+selector);
  const auto chosen=observed[0]==5?1u:0u;
  auto returned_a=gb::take_append(a,a_append);ASSERT_TRUE(returned_a)<<returned_a.error().message();
  hyle::Result<c::PagedArenaRead> full_a=initial;
  if(delta_only_predecessor){
    auto frame=c::PagedArenaFrame::fork(c::PagedArenaFootprint::from_read(initial).value()).value();
    auto delta=page_source.ready_delta(*returned_a,frame.view().value(),a_layout);
    ASSERT_TRUE(delta)<<delta.error().message();
    auto commit=frame.prepare_commit(std::move(*delta));ASSERT_TRUE(commit)<<commit.error().message();
    ASSERT_TRUE(frame.commit(std::move(*commit)).ok());ASSERT_TRUE(frame.seal().ok());
    full_a=frame.snapshot();
  }else full_a=page_source.ready(*returned_a);
  ASSERT_TRUE(full_a)<<full_a.error().message();
  ASSERT_EQ(b_common.poll().value(),c::CompletionPoll::kPending);
  const std::array<std::int32_t,2> c_ids{4,5};
  const auto b_actual=chosen?version(q0,ptr2,b_ids,1):version(q0,ptr1,b_first_id,4);
  const auto c_next=version(b_query,ptr2,c_ids,chosen?2:1);
  const auto c_capacity=version(q2,ptr2,c_ids,chosen?3:2);
  std::array<c::PagedPageWriter,2> c_dirty{arena.acquire(4).value(),arena.acquire(5).value()};
  auto c_pages=gb::PagedAppendPages::create_lengths(arena,c_capacity,b_actual,b_source,c_dirty);
  ASSERT_TRUE(c_pages)<<c_pages.error().message();
  ASSERT_TRUE(gb::bind_append(consumer,c_graph,c_append,std::move(*c_pages)).ok());
  auto c_effect=gb::append_binding(consumer,c_append,b_source);ASSERT_TRUE(c_effect)<<c_effect.error().message();
  ASSERT_TRUE(g::bind_sources(consumer,std::array{*c_effect}).ok());
  ASSERT_TRUE(g::prepare_sources(consumer,queues).ok());
  EXPECT_EQ(consumer.replay_report()->recorded_capture_range_count,0u);
  EXPECT_EQ(consumer.replay_report()->reused_capture_range_count,1u);
  EXPECT_EQ(consumer.input_source_report()->host_wait_count,0u);
  EXPECT_EQ(consumer.input_source_report()->device_wait_count,1u);
  AppendLengthsGate c_gate;c_gate.arm(a_gate.stream);
  ASSERT_EQ(cudaLaunchHostFunc(c_gate.stream,&AppendLengthsGate::wait,&c_gate),cudaSuccess);
  auto last=g::submit_sources(consumer,queues);ASSERT_TRUE(last)<<last.error().message();
  struct RetireConsumer final {
    AppendLengthsGate&gate;c::Completion&completion;
    ~RetireConsumer(){
      gate.release();
      if(completion.valid()){
        const auto status=completion.wait();EXPECT_TRUE(status.ok())<<status.message();
        completion={};
      }
    }
  }retire_consumer{c_gate,*last};
  EXPECT_EQ(b_common.poll().value(),c::CompletionPoll::kPending);
  EXPECT_FALSE(b_gate.open.load());
  auto early=gb::take_append(b,b_append);EXPECT_FALSE(early);
  b_gate.release();ASSERT_TRUE(b_common.wait().ok());
  ASSERT_TRUE(b_common.detach().ok());
  auto returned_b=gb::take_append(b,b_append);ASSERT_TRUE(returned_b)<<returned_b.error().message();
  EXPECT_FALSE(b_source.ready(*returned_b));
  ASSERT_TRUE(returned_b->select_lengths(std::array<std::int32_t,1>{1},observed).ok());
  c::PagedArenaFrame working;
  {
    const std::array before_sources{*full_a};
    auto before=arena.reframe(previous_prefixes[chosen],before_sources).value();
    working=c::PagedArenaFrame::fork(c::PagedArenaFootprint::from_read(before).value()).value();
  }
  const auto original_epoch=working.epoch();
  if(!chosen){
    auto padding_is_not_canonical=b_source.ready_delta(*returned_b,working.view().value(),accepted_prefixes[1]);
    ASSERT_FALSE(padding_is_not_canonical);EXPECT_TRUE(padding_is_not_canonical.error().IsInvalidArgument());
    EXPECT_EQ(working.epoch(),original_epoch);
  }
  // Missing expected authority must reject before publication; the same
  // genuine returned transaction remains available for the valid retry below.
  ASSERT_GT(returned_b->page_count(),0u);
  ASSERT_TRUE(returned_b->page(0)->lease().valid());
  const auto writer_generation=returned_b->page(0)->lease().generation();
  auto missing_expected=b_source.ready_delta(*returned_b,{},accepted_prefixes[chosen]);
  ASSERT_FALSE(missing_expected);EXPECT_TRUE(missing_expected.error().IsInvalidArgument());
  EXPECT_EQ(working.epoch(),original_epoch);
  EXPECT_TRUE(returned_b->page(0)->lease().valid());
  EXPECT_EQ(returned_b->page(0)->lease().generation(),writer_generation);
  auto delta=b_source.ready_delta(*returned_b,working.view().value(),accepted_prefixes[chosen]);
  ASSERT_TRUE(delta)<<delta.error().message();
  auto commit=working.prepare_commit(std::move(*delta));ASSERT_TRUE(commit)<<commit.error().message();
  ASSERT_TRUE(working.commit(std::move(*commit)).ok());ASSERT_TRUE(working.seal().ok());
  auto canonical_b=working.snapshot().value();
  EXPECT_EQ(canonical_b.layout().facts().kv_rows,chosen?5u:4u);
  EXPECT_EQ(canonical_b.page_count(),chosen?2u:1u);
  EXPECT_EQ(returned_b->appender_count(),0u);
  if(delta_only_predecessor){
    // A same-ordinal writer from another real arena is not this Source's
    // original returned authority, even after its own publication is cached.
    auto original_writer=std::move(returned_b->page(0).value());
    ASSERT_EQ(replacement.page(),original_writer.page());
    returned_b->page(0).value()=std::move(replacement);
    const auto generation=returned_b->page(0)->lease().generation();
    const auto rejected=b_source.ready_delta(*returned_b,working.view().value(),accepted_prefixes[chosen]);
    EXPECT_FALSE(rejected);
    EXPECT_TRUE(returned_b->page(0)->lease().valid());
    EXPECT_EQ(returned_b->page(0)->lease().generation(),generation);
    replacement=std::move(returned_b->page(0).value());
    returned_b->page(0).value()=std::move(original_writer);
    // Neither A's nor B's Source has needed a full cache on its Delta path.
    // A genuine whole-footprint consumer must materialize them lazily here.
    auto materialized=b_source.ready(*returned_b);ASSERT_TRUE(materialized)<<materialized.error().message();
    EXPECT_EQ(materialized->layout().facts().kv_rows,chosen?5u:4u);
    EXPECT_EQ(materialized->page_count(),chosen?2u:1u);
  }
  EXPECT_EQ(last->poll().value(),c::CompletionPoll::kPending);
  EXPECT_FALSE(c_gate.open.load());
  // Release every external B guard while C is still a real native reader.
  // C's original transaction/Source permissions must keep the raw envelope.
  canonical_b={};working={};*returned_b={};b_source={};b_common={};
  ASSERT_TRUE(g::release_sources(b).ok());
  EXPECT_FALSE(arena.acquire(3));
  c_gate.release();ASSERT_TRUE(last->wait().ok());*last={};
  if(delta_only_predecessor){replacement.lease().reset();ASSERT_TRUE(replacement.discard().ok());}
  auto returned_c=gb::take_append(consumer,c_append);ASSERT_TRUE(returned_c)<<returned_c.error().message();
  const auto released=g::release_sources(consumer);ASSERT_TRUE(released.ok())<<released.message();
  ASSERT_TRUE(returned_c->select_lengths(std::array<std::int32_t,1>{1},std::array<std::int32_t,1>{5+selector}).ok());
  auto final_result=returned_c->ready();ASSERT_TRUE(final_result)<<final_result.error().message();
  auto final=std::move(*final_result);
  const std::vector<float> expected=chosen?std::vector<float>{10,20,30,40,50,70}:std::vector<float>{10,20,30,50,70};
  EXPECT_EQ(final.layout().facts().kv_rows,expected.size());
  const auto final_host=final.layout().host();std::size_t logical=0;
  for(std::size_t ordinal=0;ordinal<final_host.page_indices.size();++ordinal){
    const auto count=ordinal+1==final_host.page_indices.size()?static_cast<std::size_t>(final_host.last_page_lengths[0]):4u;
    auto guard=final.page_prefix(static_cast<std::uint32_t>(final_host.page_indices[ordinal])).value();
    std::array<std::uint16_t,256> actual{};
    ASSERT_TRUE(guard.view()->read(std::as_writable_bytes(std::span(actual).first(count*64))).ok());
    for(std::size_t row=0;row<count;++row,++logical)
      for(std::size_t column=0;column<64;++column)EXPECT_EQ(actual[row*64+column],bf16(expected.at(logical)));
  }
  EXPECT_EQ(logical,expected.size());
  final={};*returned_c={};*returned_a={};*full_a={};page_source={};scalar_source={};common={};
  consumer={};b={};a={};initial={};empty={};initial_guard={};a_rows={};b_rows={};c_rows={};scalar_input={};query_count={};final_length={};
  EXPECT_EQ(scalar_pool.stats().active_uses,0u);EXPECT_EQ(scalar_pool.stats().live_blocks,0u);
  EXPECT_EQ(payload.stats().active_uses,0u);EXPECT_EQ(payload.stats().live_blocks,0u);
  // Reacquire original writer authority before inspecting otherwise hidden raw
  // bytes. Padding was initialized, but never became canonical KV or a tail.
  for(std::uint32_t page=0;page<6;++page){
    auto owner=arena.acquire(page);ASSERT_TRUE(owner)<<page<<": "<<owner.error().message();
    if(page==3){
      std::array<std::uint16_t,256> actual{};
      ASSERT_TRUE(owner->lease().view()->read(std::as_writable_bytes(std::span(actual))).ok());
      for(std::size_t column=0;column<64;++column)EXPECT_EQ(actual[column],chosen?bf16(50):std::uint16_t{0});
      for(std::size_t column=64;column<256;++column)EXPECT_EQ(actual[column],std::uint16_t{0x7fc1});
    }
  }
  EXPECT_FALSE(a_gate.timed_out.load());EXPECT_FALSE(b_gate.timed_out.load());EXPECT_FALSE(c_gate.timed_out.load());
}

TEST(DeviceLengthsSourceCuda, RejectedPrefixExportsZeroRawPageButCommitsOnlyRealPrefix){raw_prefix_journey(0);}
TEST(DeviceLengthsSourceCuda, AcceptedPrefixExportsRealRawRowAndFeedsPendingSuccessor){raw_prefix_journey(1);}
TEST(DeviceLengthsSourceCuda, RejectedRawSourceDeltaFeedsPendingCAndLazilyMaterializes){raw_prefix_journey(0,true);}
TEST(DeviceLengthsSourceCuda, AcceptedRawSourceDeltaFeedsPendingCAndLazilyMaterializes){raw_prefix_journey(1,true);}
TEST(DeviceLengthsSourceCuda, SourceDeltaRejectsUnsubmittedNewAppendEpochAfterOwnPublication) {
  auto context=g::cuda::create_context().value();auto queue=context.create_queue().value();
  auto device=c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice).value();
  auto storage=c::StorageOwner::adopt(context,std::move(device)).value();
  auto host=c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kPinnedHost).value();
  const auto plain=c::CacheStorage::plain(c::DataType::kBFloat16).value();
  const c::BatchCapacity capacity{.storage=c::BatchStorage::kPaged,
    .query_type=c::DataType::kBFloat16,.output_type=c::DataType::kBFloat16,
    .key_storage=plain,.value_storage=plain,.max_requests=1,.max_query_rows=2,.max_kv_rows=4,
    .max_pages=2,.max_page_refs=1,.query_heads=1,.kv_heads=1,.qk_head_dim=64,.value_head_dim=64,.page_size=4};
  auto schema=c::BatchSchema::create(capacity).value();
  const std::array<std::int32_t,2> q0{0,0},q1{0,1},ptr{0,1};
  const std::array<std::int32_t,1> id{0},last1{1},last2{2};
  const auto before=schema.version({q0,ptr,id,last1}).value();
  const auto after=schema.version({q1,ptr,id,last2}).value();
  auto arena=c::PagedArena::create(schema,c::BatchCacheRole::kKey,storage.ref(),{.max_backing_bytes=2048}).value();
  auto original=arena.acquire(0).value();
  std::array<std::uint16_t,256> raw{};raw.fill(std::uint16_t{0x7fc1});
  std::fill_n(raw.begin(),64,bf16(10));
  ASSERT_TRUE(original.lease().view()->write(std::as_bytes(std::span(raw))).ok());
  auto publication=original.publish_appendable(1).value();
  auto prefix=std::move(publication.prefix);
  auto initial=arena.read(before,std::span(&prefix,1)).value();
  std::array<c::PagedPageAppender,1> tails{std::move(publication.appender)};
  ASSERT_TRUE(tails[0].reserve_rows(1).ok());
  auto payload=c::LeasePool::create(storage.ref(),256,{.max_blocks=1}).value();
  auto input_writer=payload.allocate(128,256).value();
  std::array<std::uint16_t,64> words{};words.fill(bf16(20));
  ASSERT_TRUE(input_writer.view()->write(std::as_bytes(std::span(words))).ok());
  auto input_guard=input_writer.publish().value();
  const auto rows=c::TensorDesc::dense(c::DataType::kBFloat16,std::array<std::int64_t,3>{1,1,64}).value();
  auto builder=g::GraphBuilder::create().value();const auto input=builder.input(rows).value();
  const auto node=builder.add_node(c::Operation::custom(gb::paged_append(schema,c::BatchCacheRole::kKey).value()),std::array{input}).value();
  const auto graph=std::move(builder).finish().value();
  auto compiled=g::cuda::compile(graph,context,storage);ASSERT_TRUE(compiled)<<compiled.error().message();
  auto run=compiled->executable.create_invocation().value();
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&queue}};
  ASSERT_TRUE(gb::prepare(run,queues,{}, {.host_workspace=host.ref()}).ok());
  auto pages=gb::PagedAppendPages::create(arena,after,initial,std::span<c::PagedPageWriter>{},tails);
  ASSERT_TRUE(pages)<<pages.error().message();
  ASSERT_TRUE(gb::bind_append(run,graph,node,std::move(*pages)).ok());
  ASSERT_TRUE(run.bind(input,rows,input_guard).ok());
  auto source=gb::PagedAppendSource::prepare(run,node,storage).value();
  ASSERT_TRUE(run.prepare_replay(queues,{.mode=g::ReplayMode::kCommands}).ok());
  auto submitted=run.submit(queues);ASSERT_TRUE(submitted)<<submitted.error().message();
  auto common_result=g::SubmissionSource::create(run,std::move(*submitted),storage);ASSERT_TRUE(common_result);
  auto common=std::move(*common_result);ASSERT_TRUE(source.capture(common).ok());
  ASSERT_TRUE(common.wait().ok());ASSERT_TRUE(common.detach().ok());
  auto returned=gb::take_append(run,node);ASSERT_TRUE(returned)<<returned.error().message();
  auto working=c::PagedArenaFrame::fork(c::PagedArenaFootprint::from_read(initial).value()).value();
  const auto epoch=working.epoch();
  auto delta=source.ready_delta(*returned,working.view().value(),after);ASSERT_TRUE(delta)<<delta.error().message();
  ASSERT_EQ(returned->appender_count(),1u);
  auto&tail=returned->appender(0).value();
  ASSERT_EQ(tail.initialized_rows(),2u);ASSERT_EQ(tail.reserved_rows(),0u);
  ASSERT_TRUE(tail.reserve_rows(1).ok());
  // No new native operation is submitted. Cached own publication must not
  // let this new epoch publish its uninitialized reserved suffix.
  const auto repeated=source.ready_delta(*returned,working.view().value(),after);
  EXPECT_FALSE(repeated);
  EXPECT_FALSE(source.ready(*returned));
  EXPECT_EQ(tail.reserved_rows(),1u);EXPECT_EQ(tail.initialized_rows(),2u);
  EXPECT_EQ(working.epoch(),epoch);
  ASSERT_TRUE(tail.rollback().ok());
  auto commit=working.prepare_commit(std::move(*delta));ASSERT_TRUE(commit)<<commit.error().message();
  ASSERT_TRUE(working.commit(std::move(*commit)).ok());ASSERT_TRUE(working.seal().ok());
  {
    auto snapshot=working.snapshot().value();ASSERT_EQ(snapshot.layout().facts().kv_rows,2u);
    auto guard=snapshot.page_prefix(0).value();std::array<std::uint16_t,128> actual{};
    ASSERT_TRUE(guard.view()->read(std::as_writable_bytes(std::span(actual))).ok());
    for(std::size_t i=0;i<actual.size();++i)EXPECT_EQ(actual[i],bf16(i<64?10.F:20.F));
  }
  source={};common={};*returned={};working={};initial={};prefix={};run={};input_guard={};
  EXPECT_EQ(payload.stats().active_uses,0u);EXPECT_EQ(payload.stats().live_blocks,0u);
  auto reclaimed=arena.acquire(0);ASSERT_TRUE(reclaimed)<<reclaimed.error().message();
  ASSERT_TRUE(reclaimed->lease().view()->read(std::as_writable_bytes(std::span(raw))).ok());
  for(std::size_t i=128;i<raw.size();++i)EXPECT_EQ(raw[i],std::uint16_t{0x7fc1});
  reclaimed->lease().reset();ASSERT_TRUE(reclaimed->discard().ok());
}

} // namespace
