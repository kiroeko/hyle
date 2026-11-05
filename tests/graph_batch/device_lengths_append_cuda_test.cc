#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <numeric>
#include <mutex>
#include <source_location>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "hyle/compute/detail/paged_arena_access.h"
#include "hyle/compute/detail/batch_provider.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/batch.h"
#include "hyle/graph/cuda.h"
#include "hyle/graph/paged_append_source.h"
#include "hyle/graph/detail/recording_access.h"

namespace {
namespace c=hyle::compute;namespace g=hyle::graph;namespace b=g::batch;
template<class T>T take(hyle::Result<T>r,std::source_location at=std::source_location::current()){
  if(!r)throw std::runtime_error(std::to_string(at.line())+": "+std::string(r.error().message()));return std::move(*r);
}
void check(hyle::Status s,std::source_location at=std::source_location::current()){
  if(!s.ok())throw std::runtime_error(std::to_string(at.line())+": "+std::string(s.message()));
}
constexpr std::uint32_t page_rows=16,depth=64,row_bytes=depth*2;
struct PendingGate final{
  std::atomic<bool>open{false},timed_out{false};std::mutex mutex;std::condition_variable changed;std::thread watchdog;
  static void CUDART_CB wait(void*opaque){static_cast<PendingGate*>(opaque)->open.wait(false,std::memory_order_acquire);}
  void arm(){watchdog=std::thread([this]{std::unique_lock lock(mutex);
    if(!changed.wait_for(lock,std::chrono::seconds(10),[&]{return open.load();})){timed_out=true;release();}});}
  void release(){open.store(true,std::memory_order_release);open.notify_all();changed.notify_all();}
  ~PendingGate(){release();if(watchdog.joinable())watchdog.join();}
};
struct AppendCell final{
  enum class QueryMode{kAppendRows,kLast,kLastForeignKv};
  c::Context context=take(g::cuda::create_context());
  c::Queue queue=take(context.create_queue());
  c::cuda::MemoryResource device=take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice));
  c::StorageOwner storage=take(c::StorageOwner::adopt(context,std::move(device)));
  c::cuda::MemoryResource host=take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kPinnedHost));
  std::uint32_t requests,query_capacity,minimum,prior,maximum,old_pages,new_pages,dirty_begin,physical_pages;
  c::BatchSchema schema;c::PagedArena arena;c::BatchLayoutVersion previous,lower,capacity;
  c::PagedArena values;c::PagedArenaRead value_frame;std::vector<c::ReadLease>value_guards;
  c::PagedArenaRead frame;std::vector<c::ReadLease>old_guards;
  c::LeasePool payload,output_pool;c::ReadLease rows,query;c::TensorDesc row_desc,reader_desc,control_desc;
  c::BatchLayoutVersion reader_capacity;g::ValueId reader_query_port;bool last_query=false;
  g::Graph graph;g::CompiledGraph compiled;std::array<g::ValueId,3>inputs;g::NodeId append;
  bool with_attention=false;g::ValueId key_port,value_port,query_port,output;g::NodeId reader;
  AppendCell(std::uint32_t r,std::uint32_t q,std::uint32_t base,bool attention=false,bool pending=false,
      bool larger_previous=false,QueryMode query_mode=QueryMode::kAppendRows):requests(r),query_capacity(q),minimum(base),
      prior(base+(larger_previous?2*q:q)),maximum(base+(larger_previous?q:2*q)),old_pages((prior+15)/16),new_pages((maximum+15)/16),
      dirty_begin(base/16),physical_pages(r*(old_pages+new_pages-dirty_begin+
          (pending?(base+3*q+15)/16-dirty_begin:0u))+1),last_query(query_mode!=QueryMode::kAppendRows),with_attention(attention){
    const auto plain=take(c::CacheStorage::plain(c::DataType::kBFloat16));
    schema=take(c::BatchSchema::create({.query_type=c::DataType::kBFloat16,.output_type=c::DataType::kBFloat16,
        .key_storage=plain,.value_storage=plain,.max_requests=r,.max_query_rows=r*q,.max_kv_rows=r*std::max(prior,maximum+(pending?q:0u)),
        .max_pages=physical_pages,.max_page_refs=r*((std::max(prior,maximum+(pending?q:0u))+15)/16),.query_heads=1,.kv_heads=1,
        .qk_head_dim=depth,.value_head_dim=depth,.page_size=page_rows}));
    previous=layout(std::vector<std::int32_t>(r,0),std::vector<std::int32_t>(r,std::int32_t(prior)),false);
    lower=layout(std::vector<std::int32_t>(r,0),std::vector<std::int32_t>(r,std::int32_t(minimum)),false);
    capacity=layout(std::vector<std::int32_t>(r,std::int32_t(q)),std::vector<std::int32_t>(r,std::int32_t(maximum)),true);
    reader_capacity=last_query?layout(std::vector<std::int32_t>(r,1),std::vector<std::int32_t>(r,std::int32_t(maximum)),true):capacity;
    arena=take(c::PagedArena::create(schema,c::BatchCacheRole::kKey,storage.ref(),
        {.max_backing_bytes=std::size_t(physical_pages)*page_rows*row_bytes+1}));
    for(std::uint32_t p=0;p<r*old_pages;++p){auto writer=take(arena.acquire(p));
      std::vector<std::uint16_t>words(page_rows*depth);
      for(std::uint32_t row=0;row<page_rows;++row)std::fill_n(words.begin()+row*depth,depth,old_value(p/old_pages,(p%old_pages)*page_rows+row));
      check(take(writer.lease().view()).write(std::as_bytes(std::span(words))));old_guards.push_back(take(writer.publish(page_rows)));
    }
    frame=take(arena.read(previous,old_guards));
    row_desc=take(c::TensorDesc::dense(c::DataType::kBFloat16,std::array<std::int64_t,3>{r*q,1,depth}));
    reader_desc=last_query?take(c::TensorDesc::dense(c::DataType::kBFloat16,std::array<std::int64_t,3>{r,1,depth})):row_desc;
    control_desc=take(c::TensorDesc::dense(c::DataType::kInt32,std::array<std::int64_t,1>{r}));
    payload=take(c::LeasePool::create(storage.ref(),take(row_desc.required_bytes())*(with_attention?2u:1u)+4096,{.max_blocks=last_query?5u:4u,.backing_alignment=256}));
    auto writer=take(payload.allocate(take(row_desc.required_bytes()),256));std::vector<std::uint16_t>words(r*q*depth);
    for(std::uint32_t row=0;row<r*q;++row)std::fill_n(words.begin()+row*depth,depth,input_value(row));
    check(take(writer.view()).write(std::as_bytes(std::span(words))));rows=take(writer.publish());
    auto builder=take(g::GraphBuilder::create());inputs={take(builder.input(row_desc)),take(builder.input(control_desc)),take(builder.input(control_desc))};
    const auto qcontrol=take(builder.add(c::Copy{},inputs[1])),kvcontrol=take(builder.add(c::Copy{},inputs[2]));
    append=take(builder.add_node(c::Operation::custom(take(b::device_lengths_paged_append(schema,c::BatchCacheRole::kKey))),
        std::array{inputs[0],qcontrol,kvcontrol}));
    if(pending){output=kvcontrol;check(builder.publish(output));}
    if(with_attention){
      values=take(c::PagedArena::create(schema,c::BatchCacheRole::kValue,storage.ref(),
          {.max_backing_bytes=std::size_t(physical_pages)*page_rows*row_bytes+1}));
      for(std::uint32_t p=0;p<physical_pages;++p){auto value_writer=take(values.acquire(p));
        std::vector<std::uint16_t>ones(page_rows*depth,0x3f80);check(take(value_writer.lease().view()).write(std::as_bytes(std::span(ones))));
        value_guards.push_back(take(value_writer.publish(page_rows)));
      }
      const auto ids=take(c::detail::BatchLayoutAccess::page_footprint(reader_capacity));std::vector<c::ReadLease>used;
      for(const auto id:ids)used.push_back(value_guards[std::size_t(id)]);value_frame=take(values.read(reader_capacity,used));
      auto query_writer=take(payload.allocate(take(reader_desc.required_bytes()),256));check(take(query_writer.view()).fill(std::byte{0}));query=take(query_writer.publish());
      output_pool=take(c::LeasePool::create(storage.ref(),take(reader_desc.required_bytes()),{.max_blocks=1,.backing_alignment=256}));
      key_port=take(builder.input(take(c::batch_cache_description(schema,c::BatchCacheRole::kKey))));
      value_port=take(builder.input(take(c::batch_cache_description(schema,c::BatchCacheRole::kValue))));query_port=take(builder.input(reader_desc));
      auto reader_queries=qcontrol;if(last_query){reader_query_port=take(builder.input(control_desc));reader_queries=take(builder.add(c::Copy{},reader_query_port));}
      const auto reader_kv=query_mode==QueryMode::kLastForeignKv?take(builder.add(c::Copy{},kvcontrol)):kvcontrol;
      reader=take(builder.add_node(c::Operation::custom(take(b::device_lengths_paged_attention(schema,last_query?r:r*q,{.scale=.125F,.causal=true}))),
          std::array{query_port,key_port,value_port,reader_queries,reader_kv}));
      check(builder.depends_on(reader,append));output=take(builder.result(reader));check(builder.publish(output));
    }
    graph=take(std::move(builder).finish());
    compiled=take(g::cuda::compile(graph,context,storage));
  }
  static std::uint16_t old_value(std::uint32_t r,std::uint32_t row){return std::uint16_t(0x3f00u+(r*13+row%101)%128);}
  static std::uint16_t input_value(std::uint32_t row){return std::uint16_t(0x4000u+row);}
  std::uint32_t destination(std::uint32_t r,std::uint32_t logical)const{
    return logical<dirty_begin?r*old_pages+logical:requests*old_pages+r*(new_pages-dirty_begin)+logical-dirty_begin;
  }
  c::BatchLayoutVersion layout(const std::vector<std::int32_t>&q,const std::vector<std::int32_t>&kv,bool next){
    std::vector<std::int32_t>queries(requests+1),pointers(requests+1),ids,tails;
    for(std::uint32_t r=0;r<requests;++r){const auto count=(std::uint32_t(kv[r])+15)/16;
      queries[r+1]=queries[r]+q[r];pointers[r+1]=pointers[r]+std::int32_t(count);
      for(std::uint32_t p=0;p<count;++p)ids.push_back(std::int32_t(next?destination(r,p):r*old_pages+p));
      tails.push_back(kv[r]?(kv[r]-1)%16+1:0);
    }
    return take(schema.version({queries,pointers,ids,tails}));
  }
  void run(const std::vector<std::int32_t>&q,const std::vector<std::int32_t>&kv,bool invalid=false,bool discard=false,
      const std::vector<std::int32_t>&selected_queries={}){
    auto invocation=take(compiled.executable.create_invocation());const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&queue}};
    if(with_attention){const std::array envelopes{b::NodeLayout{reader,reader_capacity}};
      check(b::prepare_lengths(invocation,queues,envelopes,{.host_workspace=host.ref()}));
      check(invocation.bind(query_port,reader_desc,query));check(b::bind(invocation,value_port,value_frame));
      auto out=take(output_pool.allocate(take(reader_desc.required_bytes()),256));check(invocation.bind(output,reader_desc,std::move(out)));
    }else check(b::prepare(invocation,queues,{}, {.host_workspace=host.ref()}));
    check(invocation.bind(inputs[0],row_desc,rows));
    std::array<c::ReadLease,3>controls;
    for(std::size_t i=0;i<2;++i){const auto&v=i?kv:q;auto writer=take(payload.allocate(v.size()*4,256));
      check(take(writer.view()).write(std::as_bytes(std::span(v))));controls[i]=take(writer.publish());check(invocation.bind(inputs[i+1],control_desc,controls[i]));}
    const auto reader_queries=last_query?(selected_queries.empty()?std::vector<std::int32_t>(requests,1):selected_queries):q;
    if(last_query){auto writer=take(payload.allocate(requests*4,256));check(take(writer.view()).write(std::as_bytes(std::span(reader_queries))));
      controls[2]=take(writer.publish());check(invocation.bind(reader_query_port,control_desc,controls[2]));}
    std::vector<c::PagedPageWriter>writers;
    for(std::uint32_t p=requests*old_pages;p<physical_pages;++p){auto writer=take(arena.acquire(p));
      check(take(writer.lease().view()).fill(std::byte{0x5a}));if(p+1!=physical_pages)writers.push_back(std::move(writer));}
    const auto footprint=take(c::PagedArenaFootprint::from_read(frame));const std::array sources{footprint};
    auto pages=take(b::PagedAppendPages::create_lengths(arena,capacity,lower,previous,sources,writers));
    EXPECT_FALSE(pages.select_lengths(q,kv).ok());
    if(with_attention)check(b::bind_append(invocation,graph,append,key_port,std::move(pages)));
    else check(b::bind_append(invocation,graph,append,std::move(pages)));
    auto source=take(b::PagedAppendSource::prepare(invocation,append,storage));
    check(invocation.prepare_replay(queues,{.mode=g::ReplayMode::kRequireCapture,.host_wait=g::HostWaitPolicy::kReject}));
    PendingGate gate;
    if(discard){gate.arm();ASSERT_EQ(cudaLaunchHostFunc(take(c::cuda::stream(queue)),&PendingGate::wait,&gate),cudaSuccess);}
    auto done=take(invocation.submit(queues));auto common=take(g::SubmissionSource::create(invocation,std::move(done),storage));
    check(source.capture(common));
    if(discard){EXPECT_EQ(take(common.poll()),c::CompletionPoll::kPending);
      EXPECT_FALSE(b::take_append(invocation,append));EXPECT_FALSE(arena.acquire(requests*old_pages));gate.release();}
    const auto status=common.wait();check(common.detach());
    if(with_attention && !invalid){check(status);auto result=take(invocation.output(output));std::vector<std::uint16_t>got_output(take(reader_desc.required_bytes())/2);
      check(result.storage().read(std::as_writable_bytes(std::span(got_output))));const auto active=std::accumulate(reader_queries.begin(),reader_queries.end(),std::size_t{0});
      for(std::size_t i=0;i<got_output.size();++i)EXPECT_EQ(got_output[i],i<active*depth?0x3f80:0u)<<i;
    }
    auto returned=take(b::take_append(invocation,append));
    auto tensor=take(c::detail::PagedArenaAccess::tensor(arena));std::vector<std::uint16_t>got(take(tensor.description().required_bytes())/2);
    check(tensor.storage().read(std::as_writable_bytes(std::span(got))));
    if(invalid){EXPECT_TRUE(status.IsInvalidArgument())<<status.message();EXPECT_FALSE(returned.select_lengths(q,kv).ok());EXPECT_FALSE(source.ready(returned));}
    else if(discard){check(status);EXPECT_FALSE(returned.ready());EXPECT_FALSE(source.ready(returned));}
    else{check(status);EXPECT_FALSE(returned.ready());check(returned.select_lengths(q,kv));check(returned.select_lengths(q,kv));
      auto changed=kv;++changed[0];EXPECT_FALSE(returned.select_lengths(q,changed).ok());
      if(std::any_of(kv.begin(),kv.end(),[&](auto rows){return rows<std::int32_t(maximum);})){auto padding=source.ready_delta(returned,footprint,capacity);
        EXPECT_FALSE(padding);if(!padding)EXPECT_TRUE(padding.error().IsInvalidArgument());}
      auto published=take(source.ready(returned));EXPECT_EQ(published.layout(),returned.layout());
      auto canonical=take(returned.ready());EXPECT_EQ(canonical.layout().facts().query_rows,std::uint32_t(std::accumulate(q.begin(),q.end(),0)));
      const auto actual=layout(q,kv,true);EXPECT_TRUE(c::detail::BatchLayoutAccess::logical_prefix_covers(canonical.layout(),actual).value());
    }
    std::uint32_t packed=0;
    for(std::uint32_t r=0;r<requests;++r){
      for(std::uint32_t p=0;p<old_pages;++p)for(std::uint32_t row=0;row<page_rows;++row)for(std::uint32_t d=0;d<depth;++d)
        EXPECT_EQ(got[((r*old_pages+p)*page_rows+row)*depth+d],old_value(r,p*page_rows+row));
      for(std::uint32_t p=dirty_begin;p<new_pages;++p)for(std::uint32_t row=0;row<page_rows;++row){
        const auto token=p*page_rows+row;std::uint16_t expected=0x5a5a;
        if(!invalid && token<maximum){expected=0;const auto old=std::uint32_t(kv[r]-q[r]);
          if(token<old)expected=old_value(r,token);else if(token<std::uint32_t(kv[r]))expected=input_value(packed+token-old);}
        for(std::uint32_t d=0;d<depth;++d)EXPECT_EQ(got[(destination(r,p)*page_rows+row)*depth+d],expected)<<r<<','<<token<<','<<d;
      }
      if(!invalid)packed+=std::uint32_t(q[r]);
    }
    for(std::size_t i=std::size_t(physical_pages-1)*page_rows*depth;i<got.size();++i)EXPECT_EQ(got[i],0x5a5a);
    if(discard){
      source={};common={};invocation={};
      for(std::size_t i=0;i<returned.page_count();++i){auto item=returned.page(i);ASSERT_TRUE(item);auto&writer=*item;writer.lease().reset();check(writer.discard());}
      returned={};
      for(auto page=requests*old_pages;page+1<physical_pages;++page){auto owner=take(arena.acquire(page));EXPECT_EQ(owner.initialized_rows(),0u);}
      EXPECT_FALSE(gate.timed_out.load());
    }
    if(invalid){
      // A failed Source seals its original generation. Actual retirement and
      // withdrawal precede dropping every ticket, then explicit page discard
      // obtains a fresh generation; observing an error alone cannot unseal it.
      source={};common={};invocation={};
      for(std::size_t i=0;i<returned.page_count();++i){auto item=returned.page(i);ASSERT_TRUE(item);auto&writer=*item;writer.lease().reset();check(writer.discard());}
    }
  }
};
TEST(DeviceLengthsAppendCuda, SeparateQueryAndRawKvCapacitiesPublishOnlyExecutedPrefix){
  for(const auto r:{1u,2u,4u})for(const auto q:{2u,4u,8u})for(const auto base:{15u,31u,1023u}){
    SCOPED_TRACE(::testing::Message()<<"R="<<r<<" Q="<<q<<" base="<<base);AppendCell cell(r,q,base);
    for(std::uint32_t n=0;n<=q;++n){std::vector<std::int32_t>queries(r),kv(r);
      for(std::uint32_t request=0;request<r;++request){queries[request]=std::int32_t((n+request)%(q+1));
        kv[request]=std::int32_t(base+(n*3+request)%(q+1))+queries[request];}
      cell.run(queries,kv);
    }
  }
}
TEST(DeviceLengthsAppendCuda, AbandonedPendingPayloadRetiresBeforeWritersCanBeReacquired){
  for(const auto r:{1u,2u,4u})for(const auto q:{2u,4u,8u}){
    SCOPED_TRACE(::testing::Message()<<"R="<<r<<" Q="<<q);
    AppendCell cell(r,q,31);cell.run(std::vector<std::int32_t>(r,1),std::vector<std::int32_t>(r,32),false,true);
    cell.run(std::vector<std::int32_t>(r,1),std::vector<std::int32_t>(r,32));
  }
}
TEST(DeviceLengthsAppendCuda, InvalidControlsRetireWithOriginalErrorAndLeaveEveryWriterUntouched){
  const std::array<std::pair<std::vector<std::int32_t>,std::vector<std::int32_t>>,8>bad{{
    {{-1,0},{31,31}},{{5,0},{36,31}},{{0,0},{30,31}},{{0,0},{36,31}},
    {{0,INT32_MAX},{31,INT32_MAX}},{{0,0},{INT32_MIN,31}},{{4,0},{40,31}},{{4,0},{31,31}}}};
  AppendCell cell(2,4,31);for(const auto&[q,kv]:bad)cell.run(q,kv,true);cell.run({1,3},{34,38});
}
TEST(DeviceLengthsAppendCuda, SmallerNextCapacityRetainsOnlyTheActualPredecessorPrefix){
  for(const auto r:{1u,2u,4u})for(const auto q:{2u,4u,8u}){
    SCOPED_TRACE(::testing::Message()<<"R="<<r<<" Q="<<q);AppendCell cell(r,q,31,false,false,true);
    for(std::uint32_t n=0;n<=q;++n){std::vector<std::int32_t>queries(r),kv(r);
      for(std::uint32_t request=0;request<r;++request){queries[request]=std::int32_t((n+request)%(q+1));
        kv[request]=std::int32_t(31+(q-std::uint32_t(queries[request]))/2)+queries[request];}
      cell.run(queries,kv);
    }
    cell.run(std::vector<std::int32_t>(r,0),std::vector<std::int32_t>(r,32+q),true);
    cell.run(std::vector<std::int32_t>(r,1),std::vector<std::int32_t>(r,32));
  }
}
TEST(DeviceLengthsAppendCuda, SameGraphAttentionReadsOnlyDeviceSelectedWrittenRows){
  for(const auto r:{1u,2u,4u})for(const auto q:{2u,4u,8u}){
    SCOPED_TRACE(::testing::Message()<<"R="<<r<<" Q="<<q);
    AppendCell cell(r,q,31,true);std::vector<std::int32_t>queries(r),kv(r);
    for(std::uint32_t request=0;request<r;++request){queries[request]=std::int32_t((request+1)%(q+1));kv[request]=std::int32_t(31+q/2)+queries[request];}
    cell.run(queries,kv);cell.run(std::vector<std::int32_t>(r,0),std::vector<std::int32_t>(r,31));
  }
}
TEST(DeviceLengthsAppendCuda, EmptyActualPrefixKeepsReservedCapacityPrivateAndZerosInactiveQueries){
  for(const auto r:{1u,2u,4u})for(const auto q:{2u,4u,8u}){
    SCOPED_TRACE(::testing::Message()<<"R="<<r<<" Q="<<q);
    const std::vector<std::int32_t>zero(r,0),one(r,1);
    AppendCell packed(r,q,0,true);packed.run(zero,zero);packed.run(one,one);
    AppendCell last(r,q,0,true,false,false,AppendCell::QueryMode::kLast);
    last.run(zero,zero,false,false,zero);last.run(one,one);
  }
}
TEST(DeviceLengthsAppendCuda, LastQueryCanReadTheSameKvPrefixAfterMultipleRowsWereAppended){
  for(const auto r:{1u,2u,4u})for(const auto q:{2u,4u,8u}){
    SCOPED_TRACE(::testing::Message()<<"R="<<r<<" appendQ="<<q<<" readerQ=1");
    AppendCell cell(r,q,31,true,false,false,AppendCell::QueryMode::kLast);
    for(std::uint32_t n=0;n<=q;++n){std::vector<std::int32_t>queries(r),kv(r);
      for(std::uint32_t request=0;request<r;++request){queries[request]=std::int32_t((n+request)%(q+1));
        kv[request]=std::int32_t(31+(n+request)%(q+1))+queries[request];}
      cell.run(queries,kv);
    }
    std::vector<std::int32_t>inactive(r);for(std::uint32_t request=0;request<r;++request)inactive[request]=std::int32_t(request%2);
    cell.run(std::vector<std::int32_t>(r,0),std::vector<std::int32_t>(r,31),false,false,inactive);
    auto bad=std::vector<std::int32_t>(r,0),kv=std::vector<std::int32_t>(r,31);bad[0]=std::int32_t(q+1);kv[0]+=bad[0];
    cell.run(bad,kv,true);cell.run(std::vector<std::int32_t>(r,1),std::vector<std::int32_t>(r,32));
  }
}
TEST(DeviceLengthsAppendCuda, IndependentQueryControlsStillRequireTheOriginalKvControlIdentity){
  AppendCell cell(2,8,31,true,false,false,AppendCell::QueryMode::kLastForeignKv);
  auto invocation=take(cell.compiled.executable.create_invocation());const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&cell.queue}};
  const std::array envelopes{b::NodeLayout{cell.reader,cell.reader_capacity}};
  check(b::prepare_lengths(invocation,queues,envelopes,{.host_workspace=cell.host.ref()}));
  std::vector<c::PagedPageWriter>writers;for(auto p=cell.requests*cell.old_pages;p+1<cell.physical_pages;++p)
    writers.push_back(take(cell.arena.acquire(p)));
  const std::array sources{take(c::PagedArenaFootprint::from_read(cell.frame))};
  auto pages=take(b::PagedAppendPages::create_lengths(cell.arena,cell.capacity,cell.lower,cell.previous,sources,writers));
  EXPECT_TRUE(b::bind_append(invocation,cell.graph,cell.append,cell.key_port,std::move(pages)).IsInvalidArgument());
  EXPECT_TRUE(pages.valid());for(std::size_t i=0;i<pages.page_count();++i){auto writer=pages.page(i);ASSERT_TRUE(writer);
    EXPECT_TRUE(writer->valid() && writer->lease().valid());}
}
void pending_chain(std::uint32_t requests,std::uint32_t qcap,bool smaller_capacity=false){
  AppendCell a(requests,qcap,31,false,true);const auto r=requests,q=qcap;
  const auto bmax=smaller_capacity?a.minimum+q:a.maximum+q,bpages=(bmax+15)/16;
  const auto first_b=r*(a.old_pages+a.new_pages-a.dirty_begin);
  const auto end_b=first_b+r*(bpages-a.dirty_begin);
  const auto destination=[&](std::uint32_t request,std::uint32_t logical){return logical<a.dirty_begin?
      request*a.old_pages+logical:first_b+request*(bpages-a.dirty_begin)+logical-a.dirty_begin;};
  const auto version=[&](std::uint32_t rows,bool next){
    std::vector<std::int32_t>queries(r+1),ptr(r+1),ids,tails(r);const auto pages=(rows+15)/16;
    for(std::uint32_t request=0;request<r;++request){queries[request+1]=queries[request]+(next?std::int32_t(q):0);
      ptr[request+1]=ptr[request]+std::int32_t(pages);tails[request]=std::int32_t((rows-1)%16+1);
      for(std::uint32_t p=0;p<pages;++p)ids.push_back(std::int32_t(next?destination(request,p):a.destination(request,p)));
    }return take(a.schema.version({queries,ptr,ids,tails}));
  };
  const auto bcapacity=version(bmax,true),minimum=version(a.minimum,false);
  auto builder=take(g::GraphBuilder::create());const auto rows=take(builder.input(a.row_desc)),
      qinput=take(builder.input(a.control_desc)),previous=take(builder.input(a.control_desc));
  const auto qlengths=take(builder.add(c::Copy{},qinput));
  // The producer emits B's final KV length. B's new query count selects an
  // inherited subprefix of A's executed rows, as speculative acceptance does.
  const auto kvlengths=take(builder.add(c::Copy{},previous));
  const auto append=take(builder.add_node(c::Operation::custom(take(b::device_lengths_paged_append(a.schema,c::BatchCacheRole::kKey))),
      std::array{rows,qlengths,kvlengths}));const auto graph=take(std::move(builder).finish());
  auto plan=take(g::cuda::compile(graph,a.context,a.storage));auto b=take(plan.executable.create_invocation());
  auto first=take(a.compiled.executable.create_invocation());const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&a.queue}};
  const b::PrepareOptions options{.host_workspace=a.host.ref()};check(b::prepare(first,queues,{},options));check(b::prepare(b,queues,{},options));
  std::vector<std::int32_t>aq(r),ak(r),bq(r),bk(r);
  for(std::uint32_t request=0;request<r;++request){aq[request]=std::int32_t((request+1)%(q+1));
    ak[request]=std::int32_t(a.minimum+q)+(smaller_capacity?0:aq[request]);bq[request]=std::int32_t(q-request%q);bk[request]=ak[request];}
  const auto upload=[&](c::LeasePool&pool,std::span<const std::byte>bytes){auto writer=take(pool.allocate(bytes.size(),256));
    check(take(writer.view()).write(bytes));return take(writer.publish());};
  const auto aqread=upload(a.payload,std::as_bytes(std::span(aq))),akread=upload(a.payload,std::as_bytes(std::span(ak)));
  check(first.bind(a.inputs[0],a.row_desc,a.rows));check(first.bind(a.inputs[1],a.control_desc,aqread));check(first.bind(a.inputs[2],a.control_desc,akread));
  auto awriter=take(a.payload.allocate(r*4,256));const auto coordinates=take(c::ConstTensorView::create(a.control_desc,take(awriter.view())));
  check(first.bind(a.output,a.control_desc,std::move(awriter)));
  auto bpool=take(c::LeasePool::create(a.storage.ref(),take(a.row_desc.required_bytes())+4096,{.max_blocks=2,.backing_alignment=256}));
  std::vector<std::uint16_t>bwords(r*q*depth);for(std::uint32_t row=0;row<r*q;++row)
    std::fill_n(bwords.begin()+row*depth,depth,std::uint16_t(0x4100+row));
  const auto brows=upload(bpool,std::as_bytes(std::span(bwords))),bqread=upload(bpool,std::as_bytes(std::span(bq)));
  check(b.bind(rows,a.row_desc,brows));check(b.bind(qinput,a.control_desc,bqread));
  const auto fresh=[&](std::uint32_t begin,std::uint32_t end){std::vector<c::PagedPageWriter>writers;
    for(auto p=begin;p<end;++p){auto writer=take(a.arena.acquire(p));check(take(writer.lease().view()).fill(std::byte{0x5a}));writers.push_back(std::move(writer));}
    return writers;};
  // Capture B against genuine ready old pages. Only the cold checked records
  // change later; the graph retains the same native program and addresses.
  const std::array ready{take(c::PagedArenaFootprint::from_read(a.frame))};
  auto cold_writers=fresh(first_b,end_b);
  auto cold=take(b::PagedAppendPages::create_lengths(a.arena,bcapacity,a.lower,a.previous,ready,cold_writers));
  check(b::bind_append(b,graph,append,std::move(cold)));check(g::detail::RecordingAccess::begin(b));
  check(g::detail::RecordingAccess::bind_input(b,previous,coordinates));
  check(b.prepare_replay(queues,{.mode=g::ReplayMode::kRequireCapture,.host_wait=g::HostWaitPolicy::kAllow}));
  auto unused=take(b::take_append(b,append));unused={};check(g::detail::RecordingAccess::end(b));
  auto a_writers=fresh(r*a.old_pages,first_b);
  auto apages=take(b::PagedAppendPages::create_lengths(a.arena,a.capacity,a.lower,a.previous,ready,a_writers));
  check(b::bind_append(first,a.graph,a.append,std::move(apages)));auto asource=take(b::PagedAppendSource::prepare(first,a.append,a.storage));
  check(first.prepare_replay(queues,{.mode=g::ReplayMode::kRequireCapture,.host_wait=g::HostWaitPolicy::kReject}));
  PendingGate gate;gate.arm();ASSERT_EQ(cudaLaunchHostFunc(take(c::cuda::stream(a.queue)),&PendingGate::wait,&gate),cudaSuccess);
  auto submitted=take(first.submit(queues));auto common=take(g::SubmissionSource::create(first,std::move(submitted),a.storage));
  auto lengths=take(g::OutputSource::create(common,a.output,a.storage,a.payload));check(asource.capture(common));
  ASSERT_EQ(take(common.poll()),c::CompletionPoll::kPending);
  auto dirty=fresh(first_b,end_b);
  auto pages=take(b::PagedAppendPages::create_lengths(a.arena,bcapacity,minimum,asource,dirty));
  check(b::bind_append(b,graph,append,std::move(pages)));const auto effect=take(b::append_binding(b,append,asource));
  const std::array bindings{lengths.reader_binding(previous),effect};check(g::bind_sources(b,bindings));check(g::prepare_sources(b,queues));
  EXPECT_EQ(b.replay_route(),g::ReplayRoute::kCaptured);EXPECT_EQ(b.replay_report()->recorded_capture_range_count,0u);
  EXPECT_GT(b.replay_report()->reused_capture_range_count,0u);EXPECT_EQ(b.input_source_report()->host_wait_count,0u);
  EXPECT_EQ(b.input_source_report()->device_wait_count,1u);
  auto bsource=take(b::PagedAppendSource::prepare(b,append,a.storage));auto bdone=take(g::submit_sources(b,queues));
  auto bcommon=take(g::SubmissionSource::create(b,std::move(bdone),a.storage));check(bsource.capture(bcommon));
  EXPECT_FALSE(gate.open.load());EXPECT_FALSE(b::take_append(b,append));gate.release();check(bcommon.wait());check(common.wait());
  check(lengths.detach());check(common.detach());check(bcommon.detach());
  auto returned_a=take(b::take_append(first,a.append)),returned_b=take(b::take_append(b,append));
  check(returned_a.select_lengths(aq,ak));check(returned_b.select_lengths(bq,bk));
  EXPECT_FALSE(bsource.ready(returned_b));auto canonical_a=take(asource.ready(returned_a));
  auto canonical_b=take(bsource.ready(returned_b));
  EXPECT_EQ(canonical_a.layout().facts().kv_rows,std::uint32_t(std::accumulate(ak.begin(),ak.end(),0)));
  EXPECT_EQ(canonical_b.layout().facts().kv_rows,std::uint32_t(std::accumulate(bk.begin(),bk.end(),0)));
  const auto tensor=take(c::detail::PagedArenaAccess::tensor(a.arena));std::vector<std::uint16_t>got(take(tensor.description().required_bytes())/2);
  check(tensor.storage().read(std::as_writable_bytes(std::span(got))));std::uint32_t abase=0,bbase=0;
  for(std::uint32_t request=0;request<r;++request){
    for(std::uint32_t p=a.dirty_begin;p<bpages;++p)for(std::uint32_t row=0;row<page_rows;++row){
      const auto token=p*page_rows+row;std::uint16_t expected=0x5a5a;
      if(token<bmax){expected=0;
        if(token<std::uint32_t(bk[request]-bq[request])){
          if(token<std::uint32_t(ak[request]-aq[request]))expected=AppendCell::old_value(request,token);
          else expected=AppendCell::input_value(abase+token-std::uint32_t(ak[request]-aq[request]));
        }else if(token<std::uint32_t(bk[request]))expected=std::uint16_t(0x4100+bbase+token-std::uint32_t(bk[request]-bq[request]));}
      for(std::uint32_t d=0;d<depth;++d)EXPECT_EQ(got[(destination(request,p)*page_rows+row)*depth+d],expected)<<request<<','<<token;
    }abase+=std::uint32_t(aq[request]);bbase+=std::uint32_t(bq[request]);
  }
  EXPECT_FALSE(gate.timed_out.load());
}
TEST(DeviceLengthsAppendCuda, UnobservedSuccessorUsesOriginalSourceAndReusesCapturedCapacity){
  for(const auto r:{1u,2u,4u})for(const auto q:{2u,4u,8u}){SCOPED_TRACE(::testing::Message()<<"R="<<r<<" Q="<<q);pending_chain(r,q);}
}
TEST(DeviceLengthsAppendCuda, UnobservedSuccessorCanUseSmallerCapacityThanTheRawSource){
  for(const auto r:{1u,2u,4u})for(const auto q:{2u,4u,8u}){SCOPED_TRACE(::testing::Message()<<"R="<<r<<" Q="<<q);pending_chain(r,q,true);}
}
TEST(DeviceLengthsAppendCuda, ColdFailurePreservesWritersAndBindingRejectsReplacedAuthority){
  AppendCell cell(2,4,31);std::vector<c::PagedPageWriter>writers;
  for(auto p=cell.requests*cell.old_pages;p+1<cell.physical_pages;++p)writers.push_back(take(cell.arena.acquire(p)));
  const std::array sources{take(c::PagedArenaFootprint::from_read(cell.frame))};
  auto invalid=b::PagedAppendPages::create_lengths(cell.arena,cell.capacity,cell.capacity,cell.previous,sources,writers);
  EXPECT_FALSE(invalid);for(auto&writer:writers)EXPECT_TRUE(writer.valid() && writer.lease().valid());
  auto pages=take(b::PagedAppendPages::create_lengths(cell.arena,cell.capacity,cell.lower,cell.previous,sources,writers));
  auto replacement=take(cell.arena.acquire(cell.physical_pages-1));auto first=pages.page(0);ASSERT_TRUE(first);std::swap(*first,replacement);
  auto invocation=take(cell.compiled.executable.create_invocation());const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&cell.queue}};
  check(b::prepare(invocation,queues,{}, {.host_workspace=cell.host.ref()}));
  EXPECT_TRUE(b::bind_append(invocation,cell.graph,cell.append,std::move(pages)).IsInvalidArgument());EXPECT_TRUE(pages.valid());
  std::swap(*first,replacement);check(b::bind_append(invocation,cell.graph,cell.append,std::move(pages)));
  auto returned=take(b::take_append(invocation,cell.append));EXPECT_FALSE(returned.ready());
}
} // namespace
