// P=4, A base=2 is the exact page-boundary analogue of P=128/base=126.
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

struct AppendChoiceGate final {
  cudaStream_t stream=nullptr;
  std::atomic<bool> open{false},timed_out{false};
  std::mutex mutex;std::condition_variable changed;std::thread watchdog;
  static void CUDART_CB wait(void*opaque){static_cast<AppendChoiceGate*>(opaque)->open.wait(false,std::memory_order_acquire);}
  void arm(cudaStream_t value){stream=value;watchdog=std::thread([this]{
    std::unique_lock lock(mutex);
    if(!changed.wait_for(lock,std::chrono::seconds(10),[&]{return open.load();})){
      timed_out.store(true);release();
    }
  });}
  void release()noexcept{open.store(true,std::memory_order_release);open.notify_all();changed.notify_all();}
  ~AppendChoiceGate(){release();if(watchdog.joinable())watchdog.join();if(stream)EXPECT_EQ(cudaStreamSynchronize(stream),cudaSuccess);}
};

std::uint16_t bf16(float value){return static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(value)>>16);}

void append_choice_journey(std::int32_t selector,std::size_t logical_rows,bool cancel){
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
    .key_storage=plain,.value_storage=plain,.max_requests=1,.max_query_rows=2,.max_kv_rows=8,
    .max_pages=4,.max_page_refs=2,.query_heads=1,.kv_heads=1,.qk_head_dim=64,.value_head_dim=64,.page_size=4};
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
  const std::array choices{gb::PagedAppendChoice{before0,after0},gb::PagedAppendChoice{before1,after1}};
  auto arena=c::PagedArena::create(schema,c::BatchCacheRole::kKey,storage.ref(),{.max_backing_bytes=8192}).value();
  auto initial_writer=arena.acquire(0).value();
  std::array<std::uint16_t,256> original{};
  std::fill_n(original.begin(),64,bf16(10));std::fill_n(original.begin()+64,64,bf16(20));
  std::fill(original.begin()+128,original.end(),std::uint16_t{0x7fc1});
  ASSERT_TRUE(initial_writer.lease().view()->write(std::as_bytes(std::span(original))).ok());
  auto initial_guard=initial_writer.publish(2).value();
  auto initial=arena.read(initial_layout,std::span(&initial_guard,1)).value();
  auto empty=arena.read(empty_layout,{}).value();
  auto payload=c::LeasePool::create(storage.ref(),4*256,{.max_blocks=4}).value();
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
  auto scalar_input=upload(std::array<std::int32_t,1>{selector});
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
  const auto b_input=b_builder.input(rows).value(),b_selector=b_builder.input(scalar).value();
  const auto b_append=b_builder.add_node(c::Operation::custom(gb::device_choice_paged_append(schema,c::BatchCacheRole::kKey).value()),std::array{b_input,b_selector}).value();
  const auto b_graph=std::move(b_builder).finish().value();
  auto b_plan=g::cuda::compile(b_graph,context,storage);
  ASSERT_TRUE(b_plan)<<b_plan.error().message();
  auto b=b_plan->executable.create_invocation().value();
  ASSERT_TRUE(gb::prepare(b,queues,{},options).ok());
  ASSERT_TRUE(b.bind(b_input,rows,b_rows).ok());
  // Record with genuine empty ready prefixes, never a fake pending Source.
  // Cold has one range/branch; runtime has prefix copies and two/three ranges.
  const auto cold_next=version(b_query,ptr1,b_first_id,static_cast<std::int32_t>(logical_rows));
  const std::array cold_choices{gb::PagedAppendChoice{empty_layout,cold_next},gb::PagedAppendChoice{empty_layout,cold_next}};
  const std::array cold_sources{c::PagedArenaFootprint::from_read(empty).value()};
  std::array<c::PagedPageWriter,1> cold_dirty{arena.acquire(2).value()};
  auto invalid_choices=cold_choices;invalid_choices[1].previous=initial_layout;
  auto invalid=gb::PagedAppendPages::create_choices(arena,invalid_choices,cold_sources,cold_dirty);
  EXPECT_FALSE(invalid);
  EXPECT_TRUE(cold_dirty[0].valid() && cold_dirty[0].lease().valid());
  auto cold_pages=gb::PagedAppendPages::create_choices(arena,cold_choices,cold_sources,cold_dirty);
  ASSERT_TRUE(cold_pages)<<cold_pages.error().message();
  EXPECT_FALSE(cold_pages->select_choice(selector).ok());
  ASSERT_TRUE(gb::bind_append(b,b_graph,b_append,std::move(*cold_pages)).ok());
  ASSERT_TRUE(g::detail::RecordingAccess::begin(b).ok());
  ASSERT_TRUE(g::detail::RecordingAccess::bind_input(b,b_selector,coordinates).ok());
  const auto recorded=b.prepare_replay(queues,{.mode=g::ReplayMode::kRequireCapture,.host_wait=g::HostWaitPolicy::kAllow});
  ASSERT_TRUE(recorded.ok())<<recorded.message();
  auto cold_returned=gb::take_append(b,b_append);
  ASSERT_TRUE(cold_returned)<<cold_returned.error().message();
  *cold_returned={};
  ASSERT_TRUE(g::detail::RecordingAccess::end(b).ok());
  ASSERT_TRUE(a.prepare_replay(queues,{.mode=g::ReplayMode::kRequireCapture,.host_wait=g::HostWaitPolicy::kReject}).ok());

  AppendChoiceGate gate;gate.arm(c::cuda::stream(queue).value());
  ASSERT_EQ(cudaLaunchHostFunc(gate.stream,&AppendChoiceGate::wait,&gate),cudaSuccess);
  auto submitted=a.submit(queues);ASSERT_TRUE(submitted)<<submitted.error().message();
  auto common_result=g::SubmissionSource::create(a,std::move(*submitted),storage);
  ASSERT_TRUE(common_result)<<common_result.error().message();auto common=std::move(*common_result);
  auto scalar_result=g::OutputSource::create(common,selected,storage,scalar_pool);
  ASSERT_TRUE(scalar_result)<<scalar_result.error().message();auto scalar_source=std::move(*scalar_result);
  ASSERT_TRUE(page_source.capture(common).ok());
  ASSERT_EQ(common.poll().value(),c::CompletionPoll::kPending);
  std::array<c::PagedPageWriter,2> dirty{arena.acquire(2).value(),arena.acquire(3).value()};
  auto pending=gb::PagedAppendPages::create_choices(arena,choices,page_source,dirty);
  ASSERT_TRUE(pending)<<pending.error().message();
  EXPECT_FALSE(pending->select_choice(selector).ok());
  ASSERT_TRUE(gb::bind_append(b,b_graph,b_append,std::move(*pending)).ok());
  auto effect=gb::append_binding(b,b_append,page_source);ASSERT_TRUE(effect)<<effect.error().message();
  const std::array bindings{scalar_source.reader_binding(b_selector),*effect};
  ASSERT_TRUE(g::bind_sources(b,bindings).ok());
  const auto prepared=g::prepare_sources(b,queues);ASSERT_TRUE(prepared.ok())<<prepared.message();
  ASSERT_EQ(b.replay_route(),g::ReplayRoute::kCaptured);
  EXPECT_EQ(b.replay_report()->recorded_capture_range_count,0u);
  EXPECT_EQ(b.replay_report()->reused_capture_range_count,1u);
  EXPECT_EQ(b.input_source_report()->host_wait_count,0u);
  EXPECT_EQ(b.input_source_report()->device_wait_count,1u);
  auto finished=g::submit_sources(b,queues);ASSERT_TRUE(finished)<<finished.error().message();
  EXPECT_FALSE(gate.open.load());
  auto early=gb::take_append(b,b_append);EXPECT_FALSE(early);
  if(!early)EXPECT_TRUE(early.error().IsUnavailable() || early.error().IsFailedPrecondition());
  if(cancel){
    // CUDA cannot preempt these already admitted native commands. An owner
    // cancellation must still retire them, then discard without publication.
    const auto requested=finished->request_cancel();
    EXPECT_TRUE(requested.IsUnimplemented())<<requested.message();
    ASSERT_EQ(finished->poll().value(),c::CompletionPoll::kPending);
  }
  gate.release();const auto terminal=finished->wait();
  ASSERT_TRUE(terminal.ok())<<terminal.message();
  const auto a_terminal=common.wait();
  ASSERT_TRUE(a_terminal.ok())<<a_terminal.message();
  *finished={};
  ASSERT_TRUE(scalar_source.detach().ok());
  ASSERT_TRUE(common.detach().ok());
  auto returned_a=gb::take_append(a,a_append);ASSERT_TRUE(returned_a)<<returned_a.error().message();
  auto returned=gb::take_append(b,b_append);ASSERT_TRUE(returned)<<returned.error().message();
  // A succeeded even when the owner abandons B. Publish A's own sealed
  // original writers before releasing its Source; cancellation intent for B
  // cannot skip this source handoff or turn a sealed writer into a fresh one.
  auto full_a=page_source.ready(*returned_a);ASSERT_TRUE(full_a)<<full_a.error().message();
  if(cancel){
    // The caller declines the successful payload; no native cancellation or
    // failed mathematical verdict is invented, and no branch is published.
    EXPECT_FALSE(returned->ready());
  }else{
    const auto chosen=selector?1u:0u;
    const std::array frames{*full_a};
    auto before_result=arena.reframe(choices[chosen].previous,frames);
    ASSERT_TRUE(before_result)<<before_result.error().message();
    auto before=std::move(*before_result);
    auto before_view=c::PagedArenaFootprint::from_read(before);
    ASSERT_TRUE(before_view)<<before_view.error().message();
    auto working_result=c::PagedArenaFrame::fork(*before_view);
    ASSERT_TRUE(working_result)<<working_result.error().message();
    auto working=std::move(*working_result);
    ASSERT_FALSE(working.sealed());
    auto unselected=returned->ready();ASSERT_FALSE(unselected);
    EXPECT_TRUE(unselected.error().IsFailedPrecondition());
    EXPECT_NE(unselected.error().message().find("select"),std::string_view::npos);
    ASSERT_TRUE(returned->select_choice(selector).ok());
    ASSERT_TRUE(returned->select_choice(selector).ok());
    EXPECT_FALSE(returned->select_choice(selector?0:1).ok());
    auto delta=returned->ready_delta(working.view().value(),choices[chosen].next);
    ASSERT_TRUE(delta)<<delta.error().message();
    auto commit=working.prepare_commit(std::move(*delta));ASSERT_TRUE(commit)<<commit.error().message();
    ASSERT_TRUE(working.commit(std::move(*commit)).ok());
    ASSERT_TRUE(working.seal().ok());
    auto final_result=working.snapshot();ASSERT_TRUE(final_result)<<final_result.error().message();
    auto final=std::move(*final_result);
    const std::vector<float> expected=selector?(logical_rows==1?std::vector<float>{10,20,30,40,50}:std::vector<float>{10,20,30,40,50,60}):
        (logical_rows==1?std::vector<float>{10,20,30,50}:std::vector<float>{10,20,30,50,60});
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
    if(!selector && logical_rows==1){
      EXPECT_EQ(final.page_count(),1u);
      auto unused=arena.acquire(3);ASSERT_TRUE(unused)<<unused.error().message();
      EXPECT_EQ(unused->initialized_rows(),0u); // not a fabricated ready prefix
    }
  }
  EXPECT_FALSE(gate.timed_out.load());
  *finished={};ASSERT_TRUE(g::release_sources(b).ok());
  *returned={};*returned_a={};*full_a={};page_source={};scalar_source={};common={};b={};a={};
  initial={};empty={};initial_guard={};a_rows={};b_rows={};scalar_input={};
  EXPECT_EQ(scalar_pool.stats().active_uses,0u);EXPECT_EQ(scalar_pool.stats().live_blocks,0u);
  EXPECT_EQ(payload.stats().active_uses,0u);EXPECT_EQ(payload.stats().live_blocks,0u);
  // Reacquisition is an actual original page-owner test, not a counter guess.
  for(std::uint32_t page=0;page<4;++page){auto owner=arena.acquire(page);EXPECT_TRUE(owner)<<page<<": "<<owner.error().message();}
}

TEST(DeviceChoicePagedAppendCuda, ZeroSelectsRejectedPrefixAcrossPage){append_choice_journey(0,2,false);}
TEST(DeviceChoicePagedAppendCuda, OneSelectsAcceptedCopyOnlyFullPrefix){append_choice_journey(1,2,false);}
TEST(DeviceChoicePagedAppendCuda, RejectedOneRowDoesNotPublishUnusedUnionWriter){append_choice_journey(0,1,false);}
TEST(DeviceChoicePagedAppendCuda, AcceptedOneRowWritesNextPageAfterCopyOnlyPrefix){append_choice_journey(1,1,false);}
TEST(DeviceChoicePagedAppendCuda, CancellationIntentRetiresNativeWorkBeforeDiscard){append_choice_journey(1,2,true);}

// A real pending A -> B -> C chain keeps the zero-filled raw envelope separate
// from the selected canonical prefix and its original page permissions.
void append_source_journey(std::int32_t selector){
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
    .key_storage=plain,.value_storage=plain,.max_requests=2,.max_query_rows=8,.max_kv_rows=8,
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
  const std::array choices{gb::PagedAppendChoice{before0,after0},gb::PagedAppendChoice{before1,after1}};
  auto arena=c::PagedArena::create(schema,c::BatchCacheRole::kKey,storage.ref(),{.max_backing_bytes=8192}).value();
  auto initial_writer=arena.acquire(0).value();
  std::array<std::uint16_t,256> original{};
  std::fill_n(original.begin(),64,bf16(10));std::fill_n(original.begin()+64,64,bf16(20));
  std::fill(original.begin()+128,original.end(),std::uint16_t{0x7fc1});
  ASSERT_TRUE(initial_writer.lease().view()->write(std::as_bytes(std::span(original))).ok());
  auto initial_guard=initial_writer.publish(2).value();
  auto initial=arena.read(initial_layout,std::span(&initial_guard,1)).value();
  auto empty=arena.read(empty_layout,{}).value();
  auto payload=c::LeasePool::create(storage.ref(),4*256,{.max_blocks=4}).value();
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
  auto scalar_input=upload(std::array<std::int32_t,1>{selector});
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
  const auto b_input=b_builder.input(rows).value(),b_selector=b_builder.input(scalar).value();
  const auto b_append=b_builder.add_node(c::Operation::custom(gb::device_choice_paged_append(schema,c::BatchCacheRole::kKey,true).value()),std::array{b_input,b_selector}).value();
  const auto b_graph=std::move(b_builder).finish().value();
  auto b_plan=g::cuda::compile(b_graph,context,storage);
  ASSERT_TRUE(b_plan)<<b_plan.error().message();
  auto b=b_plan->executable.create_invocation().value();
  ASSERT_TRUE(gb::prepare(b,queues,{},options).ok());
  ASSERT_TRUE(b.bind(b_input,rows,b_rows).ok());
  // Record with genuine empty ready prefixes, never a fake pending Source.
  // Cold has one range/branch; runtime has prefix copies and two/three ranges.
  const auto cold_next=version(b_query,ptr1,b_first_id,static_cast<std::int32_t>(logical_rows));
  const std::array cold_choices{gb::PagedAppendChoice{empty_layout,cold_next},gb::PagedAppendChoice{empty_layout,cold_next}};
  const std::array cold_sources{c::PagedArenaFootprint::from_read(empty).value()};
  std::array<c::PagedPageWriter,1> cold_dirty{arena.acquire(2).value()};
  auto invalid_choices=cold_choices;invalid_choices[1].previous=initial_layout;
  auto invalid=gb::PagedAppendPages::create_choices(arena,invalid_choices,cold_sources,cold_dirty,true);
  EXPECT_FALSE(invalid);
  EXPECT_TRUE(cold_dirty[0].valid() && cold_dirty[0].lease().valid());
  auto cold_pages=gb::PagedAppendPages::create_choices(arena,cold_choices,cold_sources,cold_dirty,true);
  ASSERT_TRUE(cold_pages)<<cold_pages.error().message();
  EXPECT_FALSE(cold_pages->select_choice(selector).ok());
  ASSERT_TRUE(gb::bind_append(b,b_graph,b_append,std::move(*cold_pages)).ok());
  ASSERT_TRUE(g::detail::RecordingAccess::begin(b).ok());
  ASSERT_TRUE(g::detail::RecordingAccess::bind_input(b,b_selector,coordinates).ok());
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
  const auto c_input=c_builder.input(rows).value(),c_selector=c_builder.input(scalar).value();
  const auto c_append=c_builder.add_node(c::Operation::custom(gb::device_choice_paged_append(schema,c::BatchCacheRole::kKey).value()),std::array{c_input,c_selector}).value();
  const auto c_graph=std::move(c_builder).finish().value();
  auto c_plan=g::cuda::compile(c_graph,context,storage);ASSERT_TRUE(c_plan)<<c_plan.error().message();
  auto consumer=c_plan->executable.create_invocation().value();
  ASSERT_TRUE(gb::prepare(consumer,queues,{},options).ok());
  ASSERT_TRUE(consumer.bind(c_input,rows,c_rows).ok());
  ASSERT_TRUE(consumer.bind(c_selector,scalar,scalar_input).ok());
  const std::array<std::int32_t,1> c_id{4};
  const auto cold_c_next=version(b_query,ptr1,c_id,1);
  const std::array cold_c_choices{gb::PagedAppendChoice{empty_layout,cold_c_next},gb::PagedAppendChoice{empty_layout,cold_c_next}};
  std::array<c::PagedPageWriter,1> cold_c_dirty{arena.acquire(4).value()};
  auto cold_c_pages=gb::PagedAppendPages::create_choices(arena,cold_c_choices,cold_sources,cold_c_dirty);
  ASSERT_TRUE(cold_c_pages)<<cold_c_pages.error().message();
  ASSERT_TRUE(gb::bind_append(consumer,c_graph,c_append,std::move(*cold_c_pages)).ok());
  auto unsupported=gb::PagedAppendSource::prepare(consumer,c_append,storage);
  ASSERT_FALSE(unsupported);EXPECT_TRUE(unsupported.error().IsUnimplemented());
  ASSERT_TRUE(g::detail::RecordingAccess::begin(consumer).ok());
  ASSERT_TRUE(consumer.prepare_replay(queues,{.mode=g::ReplayMode::kRequireCapture,.host_wait=g::HostWaitPolicy::kAllow}).ok());
  auto cold_c_returned=gb::take_append(consumer,c_append);ASSERT_TRUE(cold_c_returned);
  *cold_c_returned={};ASSERT_TRUE(g::detail::RecordingAccess::end(consumer).ok());
  // Poison raw backing before admitting A. Only the missing envelope row may
  // become zero; bytes beyond the advertised maximum remain untouched.
  std::array<c::PagedPageWriter,2> dirty{arena.acquire(2).value(),arena.acquire(3).value()};
  std::array<std::uint16_t,256> poison;poison.fill(0x7fc1);
  for(auto&writer:dirty)ASSERT_TRUE(writer.lease().view()->write(std::as_bytes(std::span(poison))).ok());

  AppendChoiceGate a_gate;a_gate.arm(c::cuda::stream(queue).value());
  ASSERT_EQ(cudaLaunchHostFunc(a_gate.stream,&AppendChoiceGate::wait,&a_gate),cudaSuccess);
  auto submitted=a.submit(queues);ASSERT_TRUE(submitted)<<submitted.error().message();
  auto common_result=g::SubmissionSource::create(a,std::move(*submitted),storage);ASSERT_TRUE(common_result);
  auto common=std::move(*common_result);
  auto scalar_result=g::OutputSource::create(common,selected,storage,scalar_pool);ASSERT_TRUE(scalar_result);
  auto scalar_source=std::move(*scalar_result);
  ASSERT_TRUE(page_source.capture(common).ok());
  ASSERT_EQ(common.poll().value(),c::CompletionPoll::kPending);
  auto excessive_gap=choices;
  excessive_gap[1].next=version(q2,ptr2,b_ids,2);
  auto refused_gap=gb::PagedAppendPages::create_choices(arena,excessive_gap,page_source,dirty,true);
  ASSERT_FALSE(refused_gap);EXPECT_TRUE(refused_gap.error().IsInvalidArgument());
  EXPECT_TRUE(dirty[0].valid());EXPECT_TRUE(dirty[1].valid());
  auto pending=gb::PagedAppendPages::create_choices(arena,choices,page_source,dirty,true);
  ASSERT_TRUE(pending)<<pending.error().message();
  EXPECT_FALSE(pending->select_choice(selector).ok());
  ASSERT_TRUE(gb::bind_append(b,b_graph,b_append,std::move(*pending)).ok());
  auto b_source_result=gb::PagedAppendSource::prepare(b,b_append,storage);
  ASSERT_TRUE(b_source_result)<<b_source_result.error().message();
  auto b_source=std::move(*b_source_result);
  auto effect=gb::append_binding(b,b_append,page_source);ASSERT_TRUE(effect);
  const std::array bindings{scalar_source.reader_binding(b_selector),*effect};
  ASSERT_TRUE(g::bind_sources(b,bindings).ok());
  ASSERT_TRUE(g::prepare_sources(b,queues).ok());
  EXPECT_EQ(b.replay_report()->recorded_capture_range_count,0u);
  EXPECT_EQ(b.replay_report()->reused_capture_range_count,1u);
  EXPECT_EQ(b.input_source_report()->host_wait_count,0u);
  EXPECT_EQ(b.input_source_report()->device_wait_count,1u);
  AppendChoiceGate b_gate;b_gate.arm(a_gate.stream);
  ASSERT_EQ(cudaLaunchHostFunc(b_gate.stream,&AppendChoiceGate::wait,&b_gate),cudaSuccess);
  auto b_submitted=g::submit_sources(b,queues);ASSERT_TRUE(b_submitted)<<b_submitted.error().message();
  auto b_common_result=g::SubmissionSource::create(b,std::move(*b_submitted),storage);ASSERT_TRUE(b_common_result);
  auto b_common=std::move(*b_common_result);
  // B's Source is captured before A's selector has been observed on the host.
  ASSERT_TRUE(b_source.capture(b_common).ok());
  EXPECT_EQ(common.poll().value(),c::CompletionPoll::kPending);
  EXPECT_EQ(b_common.poll().value(),c::CompletionPoll::kPending);
  EXPECT_FALSE(a_gate.open.load());
  a_gate.release();ASSERT_TRUE(common.wait().ok());
  ASSERT_TRUE(scalar_source.detach().ok());ASSERT_TRUE(common.detach().ok());
  std::array<std::int32_t,1> observed{};
  ASSERT_TRUE(a.output(selected)->storage().read(std::as_writable_bytes(std::span(observed))).ok());
  ASSERT_EQ(observed[0],selector);
  const auto chosen=observed[0]?1u:0u;
  auto returned_a=gb::take_append(a,a_append);ASSERT_TRUE(returned_a)<<returned_a.error().message();
  auto full_a=page_source.ready(*returned_a);ASSERT_TRUE(full_a)<<full_a.error().message();
  ASSERT_EQ(b_common.poll().value(),c::CompletionPoll::kPending);
  const std::array<std::int32_t,2> c_ids{4,5};
  const auto b_actual=chosen?version(q0,ptr2,b_ids,1):version(q0,ptr1,b_first_id,4);
  const auto c_next=version(b_query,ptr2,c_ids,chosen?2:1);
  const std::array c_choices{gb::PagedAppendChoice{b_actual,c_next},gb::PagedAppendChoice{b_actual,c_next}};
  std::array<c::PagedPageWriter,2> c_dirty{arena.acquire(4).value(),arena.acquire(5).value()};
  auto c_pages=gb::PagedAppendPages::create_choices(arena,c_choices,b_source,c_dirty);
  ASSERT_TRUE(c_pages)<<c_pages.error().message();
  ASSERT_TRUE(gb::bind_append(consumer,c_graph,c_append,std::move(*c_pages)).ok());
  auto c_effect=gb::append_binding(consumer,c_append,b_source);ASSERT_TRUE(c_effect)<<c_effect.error().message();
  ASSERT_TRUE(g::bind_sources(consumer,std::array{*c_effect}).ok());
  ASSERT_TRUE(g::prepare_sources(consumer,queues).ok());
  EXPECT_EQ(consumer.replay_report()->recorded_capture_range_count,0u);
  EXPECT_EQ(consumer.replay_report()->reused_capture_range_count,1u);
  EXPECT_EQ(consumer.input_source_report()->host_wait_count,0u);
  EXPECT_EQ(consumer.input_source_report()->device_wait_count,1u);
  AppendChoiceGate c_gate;c_gate.arm(a_gate.stream);
  ASSERT_EQ(cudaLaunchHostFunc(c_gate.stream,&AppendChoiceGate::wait,&c_gate),cudaSuccess);
  auto last=g::submit_sources(consumer,queues);ASSERT_TRUE(last)<<last.error().message();
  struct RetireConsumer final {
    AppendChoiceGate&gate;c::Completion&completion;
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
  ASSERT_TRUE(returned_b->select_choice(observed[0]).ok());
  c::PagedArenaFrame working;
  {
    const std::array before_sources{*full_a};
    auto before=arena.reframe(choices[chosen].previous,before_sources).value();
    working=c::PagedArenaFrame::fork(c::PagedArenaFootprint::from_read(before).value()).value();
  }
  const auto original_epoch=working.epoch();
  if(!chosen){
    auto padding_is_not_canonical=b_source.ready_delta(*returned_b,working.view().value(),choices[1].next);
    ASSERT_FALSE(padding_is_not_canonical);EXPECT_TRUE(padding_is_not_canonical.error().IsInvalidArgument());
    EXPECT_EQ(working.epoch(),original_epoch);
  }
  auto delta=b_source.ready_delta(*returned_b,working.view().value(),choices[chosen].next);
  ASSERT_TRUE(delta)<<delta.error().message();
  auto commit=working.prepare_commit(std::move(*delta));ASSERT_TRUE(commit)<<commit.error().message();
  ASSERT_TRUE(working.commit(std::move(*commit)).ok());ASSERT_TRUE(working.seal().ok());
  auto canonical_b=working.snapshot().value();
  EXPECT_EQ(canonical_b.layout().facts().kv_rows,chosen?5u:4u);
  EXPECT_EQ(canonical_b.page_count(),chosen?2u:1u);
  EXPECT_EQ(returned_b->appender_count(),0u);
  EXPECT_EQ(last->poll().value(),c::CompletionPoll::kPending);
  EXPECT_FALSE(c_gate.open.load());
  // Release every external B guard while C is still a real native reader.
  // C's original transaction/Source permissions must keep the raw envelope.
  canonical_b={};working={};*returned_b={};b_source={};b_common={};
  ASSERT_TRUE(g::release_sources(b).ok());
  EXPECT_FALSE(arena.acquire(3));
  c_gate.release();ASSERT_TRUE(last->wait().ok());*last={};
  auto returned_c=gb::take_append(consumer,c_append);ASSERT_TRUE(returned_c)<<returned_c.error().message();
  const auto released=g::release_sources(consumer);ASSERT_TRUE(released.ok())<<released.message();
  ASSERT_TRUE(returned_c->select_choice(0).ok());
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
  consumer={};b={};a={};initial={};empty={};initial_guard={};a_rows={};b_rows={};c_rows={};scalar_input={};
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

TEST(DeviceChoicePagedAppendCuda, RejectedPrefixExportsZeroRawPageButCommitsOnlyRealPrefix){append_source_journey(0);}
TEST(DeviceChoicePagedAppendCuda, AcceptedPrefixExportsRealRawRowAndFeedsPendingSuccessor){append_source_journey(1);}
} // namespace
