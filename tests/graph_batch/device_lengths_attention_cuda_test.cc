#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <span>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include "hyle/graph/batch.h"
#include "hyle/graph/cuda.h"

namespace {
namespace c=hyle::compute;namespace g=hyle::graph;namespace b=g::batch;
template<class T>T take(hyle::Result<T> r){if(!r)throw std::runtime_error(std::string(r.error().message()));return std::move(*r);}
void check(hyle::Status s){if(!s.ok())throw std::runtime_error(std::string(s.message()));}
std::uint16_t bf16(float f){auto u=std::bit_cast<std::uint32_t>(f);return std::uint16_t((u+0x7fff+((u>>16)&1))>>16);}
float value(std::uint16_t b){return std::bit_cast<float>(std::uint32_t(b)<<16);}
constexpr std::uint32_t kDepth=256,kHeads=8,kKvHeads=2,kPage=16;
struct Cell final{
  c::Context context=take(g::cuda::create_context());
  c::Queue queue=take(context.create_queue());
  c::cuda::MemoryResource device=take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice));
  c::cuda::MemoryResource host=take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kPinnedHost));
  c::BatchSchema schema;
  c::BatchLayoutVersion anchor;
  c::PagedArena keys,values;
  std::vector<c::ReadLease> key_pages,value_pages;
  c::PagedArenaRead key_frame,value_frame;
  c::LeasePool inputs,outputs;
  c::ReadLease query;
  c::TensorDesc query_desc,index_desc,output_desc;
  g::Graph graph;g::CompiledGraph compiled;
  std::array<g::ValueId,5> ports;
  g::NodeId reader;g::ValueId output;
  std::uint32_t requests,rows,context_rows,pages;
  bool weighted=false;
  Cell(std::uint32_t r,std::uint32_t q,std::uint32_t ctx,bool use_weights=false):requests(r),rows(q),context_rows(ctx),pages((ctx+kPage-1)/kPage),weighted(use_weights){
    const auto plain=take(c::CacheStorage::plain(c::DataType::kBFloat16));
    const c::BatchCapacity cap{.query_type=c::DataType::kBFloat16,.output_type=c::DataType::kBFloat16,
      .key_storage=plain,.value_storage=plain,.max_requests=r,.max_query_rows=r*q,.max_kv_rows=r*ctx,
      .max_pages=r*pages,.max_page_refs=r*pages,.query_heads=kHeads,.kv_heads=kKvHeads,
      .qk_head_dim=kDepth,.value_head_dim=kDepth,.page_size=kPage};
    schema=take(c::BatchSchema::create(cap));anchor=layout(std::vector<std::uint32_t>(r,q));
    const auto arena_bytes=std::size_t(r)*pages*kPage*kKvHeads*kDepth*2+1;
    keys=take(c::PagedArena::create(schema,c::BatchCacheRole::kKey,device.ref(),{.max_backing_bytes=arena_bytes}));
    values=take(c::PagedArena::create(schema,c::BatchCacheRole::kValue,device.ref(),{.max_backing_bytes=arena_bytes}));
    for(std::uint32_t page=0;page<r*pages;++page){
      auto k=take(keys.acquire(page)),v=take(values.acquire(page));
      std::vector<std::uint16_t> payload(kPage*kKvHeads*kDepth);
      for(std::uint32_t row=0;row<kPage;++row)for(std::uint32_t d=0;d<kKvHeads*kDepth;++d)
        payload[row*kKvHeads*kDepth+d]=bf16(sample(page/pages,(page%pages)*kPage+row,d%kDepth));
      check(take(k.lease().view()).fill(std::byte{0}));
      if(weighted){
        std::vector<std::uint16_t>key_payload(payload.size(),0);
        for(std::uint32_t row=0;row<kPage;++row)for(std::uint32_t h=0;h<kKvHeads;++h)
          key_payload[(row*kKvHeads+h)*kDepth]=bf16(key_weight(page/pages,(page%pages)*kPage+row,h));
        check(take(k.lease().view()).write(std::as_bytes(std::span(key_payload))));
      }
      check(take(v.lease().view()).write(std::as_bytes(std::span(payload))));
      key_pages.push_back(take(k.publish(kPage)));value_pages.push_back(take(v.publish(kPage)));
    }
    key_frame=take(keys.read(anchor,key_pages));value_frame=take(values.read(anchor,value_pages));
    query_desc=take(c::batch_query_description(cap,r*q));output_desc=take(c::batch_output_description(cap,r*q));
    index_desc=take(c::TensorDesc::dense(c::DataType::kInt32,std::array<std::int64_t,1>{r}));
    inputs=take(c::LeasePool::create(device.ref(),take(query_desc.required_bytes())+4096,{.max_blocks=4,.backing_alignment=256}));
    outputs=take(c::LeasePool::create(device.ref(),take(output_desc.required_bytes()),{.max_blocks=1,.backing_alignment=256}));
    auto qwrite=take(inputs.allocate(take(query_desc.required_bytes()),256));check(take(qwrite.view()).fill(std::byte{0}));
    if(weighted){
      std::vector<std::uint16_t>payload(std::size_t(r)*q*kHeads*kDepth,0);
      for(std::uint32_t row=0;row<r*q;++row)for(std::uint32_t h=0;h<kHeads;++h)
        payload[(row*kHeads+h)*kDepth]=bf16(query_weight(row,h));
      check(take(qwrite.view()).write(std::as_bytes(std::span(payload))));
    }
    query=take(qwrite.publish());
    auto builder=take(g::GraphBuilder::create());ports[0]=take(builder.input(query_desc));
    ports[1]=take(builder.input(take(c::batch_cache_description(schema,c::BatchCacheRole::kKey))));
    ports[2]=take(builder.input(take(c::batch_cache_description(schema,c::BatchCacheRole::kValue))));ports[3]=take(builder.input(index_desc));ports[4]=take(builder.input(index_desc));
    // Produce both length vectors on device in this same Graph. The native
    // batch region must preserve the real data dependencies into metadata.
    const auto qlengths=take(builder.add(c::Copy{},ports[3]));
    const auto kvlengths=take(builder.add(c::Copy{},ports[4]));
    const auto op=take(b::device_lengths_paged_attention(schema,r*q,{.scale=.0625f,.causal=true}));
    reader=take(builder.add_node(c::Operation::custom(op),std::array{ports[0],ports[1],ports[2],qlengths,kvlengths}));
    output=take(builder.result(reader));check(builder.publish(output));graph=take(std::move(builder).finish());
    const std::array domains{g::StorageDomain{device.ref()}};
    auto source=take(g::ExecutionSource::acquire(context));const std::array targets{g::ExecutionTarget{source,g::StorageDomainId{0}}};
    compiled=take(b::compile(graph,{.storage_domains=domains,.targets=targets}));
  }
  static float sample(std::uint32_t r,std::uint32_t row,std::uint32_t d){return float(r)+float(row%61)/64.F+float(d%7)/32.F;}
  static float query_weight(std::size_t row,std::uint32_t head){return float(4+(row+head)%5);}
  static float key_weight(std::uint32_t request,std::uint32_t row,std::uint32_t head){return float(int((row*13+request*17+head*7)%23)-11)/4.F;}
  c::BatchLayoutVersion layout(const std::vector<std::uint32_t>&qrows,const std::vector<std::uint32_t>&lengths={}){
    std::vector<std::int32_t>q(requests+1),k(requests+1),ids,tails(requests);
    for(std::uint32_t r=0;r<requests;++r){
      const auto length=lengths.empty()?context_rows:lengths[r],used=(length+kPage-1)/kPage;
      q[r+1]=q[r]+std::int32_t(qrows[r]);k[r+1]=k[r]+std::int32_t(used);
      for(std::uint32_t page=0;page<used;++page)ids.push_back(std::int32_t(r*pages+page));
      tails[r]=std::int32_t((length-1)%kPage+1);
    }
    return take(schema.version({q,k,ids,tails}));
  }
  g::Invocation prepare(){
    auto invocation=take(compiled.executable.create_invocation());check(invocation.prepare_resources());
    const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&queue}};
    const std::array envelopes{b::NodeLayout{reader,anchor}};
    check(b::prepare_lengths(invocation,queues,envelopes,{.host_workspace=host.ref()}));return invocation;
  }
  void run(g::Invocation&invocation,const std::vector<std::int32_t>&active,
      const std::vector<std::int32_t>&lengths,bool invalid=false){
    const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&queue}};
    check(invocation.bind(ports[0],query_desc,query));check(b::bind(invocation,ports[1],key_frame));check(b::bind(invocation,ports[2],value_frame));
    std::array<c::ReadLease,2>controls;
    for(std::size_t p=0;p<2;++p){const auto&control=p?lengths:active;
      auto writer=take(inputs.allocate(control.size()*4,256));check(take(writer.view()).write(std::as_bytes(std::span(control))));
      controls[p]=take(writer.publish());check(invocation.bind(ports[3+p],index_desc,controls[p]));
    }
    auto out=take(outputs.allocate(take(output_desc.required_bytes()),256));check(take(out.view()).fill(std::byte{0x7f}));
    check(invocation.bind(output,output_desc,std::move(out)));
    check(invocation.prepare_replay(queues,{.mode=g::ReplayMode::kRequireCapture,.host_wait=g::HostWaitPolicy::kReject}));
    auto done=take(invocation.submit(queues));const auto status=done.wait();done={};
    if(invalid){EXPECT_TRUE(status.IsInvalidArgument())<<status.message();EXPECT_FALSE(invocation.output(output));return;}
    check(status);auto result=take(invocation.output(output));
    std::vector<std::uint16_t>got(take(output_desc.required_bytes())/2);check(result.storage().read(std::as_writable_bytes(std::span(got))));
    verify(got,active,lengths);
  }
  void verify(std::span<const std::uint16_t>got,const std::vector<std::int32_t>&active,
      const std::vector<std::int32_t>&lengths){
    std::size_t row=0;
    for(std::uint32_t r=0;r<requests;++r)for(std::uint32_t q=0;q<std::uint32_t(active[r]);++q,++row){
      const auto length=std::uint32_t(lengths[r]-active[r])+q+1;
      std::array<std::array<float,7>,kHeads>means{};
      for(std::uint32_t h=0;h<(weighted?kHeads:1);++h)for(std::uint32_t d=0;d<7;++d){
        double sum=0,denominator=0;
        for(std::uint32_t k=0;k<length;++k){
          const double weight=weighted?std::exp(double(query_weight(row,h))*key_weight(r,k,h/(kHeads/kKvHeads))*.0625):1.;
          sum+=weight*value(bf16(sample(r,k,d)));denominator+=weight;
        }
        means[h][d]=value(bf16(float(sum/denominator)));
      }
      for(std::uint32_t d=0;d<kDepth;++d)for(std::uint32_t h=0;h<kHeads;++h)
        EXPECT_NEAR(value(got[(row*kHeads+h)*kDepth+d]),means[weighted?h:0][d%7],.016F)<<r<<','<<q<<','<<d;
    }
    for(std::size_t i=row*kHeads*kDepth;i<got.size();++i)EXPECT_EQ(got[i],0u)<<i;
  }
};
TEST(DeviceLengthsCuda, MultipleRequestQueryAndContextCapacitiesUseOriginalCapturedGraph){
  for(const auto r:{1u,2u,4u})for(const auto q:{2u,4u,8u})for(const auto ctx:{1024u,4096u,32768u}){
    SCOPED_TRACE(::testing::Message()<<"R="<<r<<" Q="<<q<<" C="<<ctx);Cell cell(r,q,ctx);
    for(std::uint32_t n=0;n<=q;++n){
      std::vector<std::int32_t>active(r),lengths(r);
      for(std::uint32_t request=0;request<r;++request){active[request]=std::int32_t((n+request)%(q+1));lengths[request]=std::int32_t(ctx-20+n-request);}
      auto invocation=cell.prepare();cell.run(invocation,active,lengths);
    }
    auto empty=cell.prepare();cell.run(empty,std::vector<std::int32_t>(r,0),std::vector<std::int32_t>(r,0));
  }
}
TEST(DeviceLengthsCuda, EachRequestIsBoundedIndependentlyAndErrorsUseOriginalCompletion){
  Cell cell(2,4,1024);
  const std::array<std::pair<std::vector<std::int32_t>,std::vector<std::int32_t>>,8>bad{{
    {{-1,0},{100,100}},{{5,3},{100,100}},{{0,INT32_MAX},{100,100}},{{0,0},{INT32_MIN,0}},
    {{0,0},{INT32_MAX,0}},{{1,3},{100,1025}},{{1,0},{0,0}},{{0,0},{-1,100}}}};
  for(const auto&[q,k]:bad){auto invocation=cell.prepare();cell.run(invocation,q,k,true);}
  auto valid=cell.prepare();cell.run(valid,{1,3},{1001,1015});
}
TEST(DeviceLengthsCuda, NonzeroQueriesAndKeysMatchFp64SoftmaxWithNonuniformRows){
  Cell cell(2,4,31,true);
  {auto invocation=cell.prepare();cell.run(invocation,{1,3},{15,31});}
  {auto invocation=cell.prepare();cell.run(invocation,{4,1},{16,25});}
  {auto invocation=cell.prepare();cell.run(invocation,{0,4},{0,17});}
  // The final physical page has 32 initialized rows. The admitted logical
  // envelope ends at31, so initialized padding cannot grant another KV row.
  {auto invocation=cell.prepare();cell.run(invocation,{1,0},{32,0},true);}
}
TEST(DeviceLengthsCuda, BadPhysicalEnvelopeDeclinesAndValidPreparationCanRetry){
  Cell cell(2,4,1024);auto invocation=take(cell.compiled.executable.create_invocation());check(invocation.prepare_resources());
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&cell.queue}};
  const std::array bad{b::NodeLayout{cell.reader,cell.layout({1,3})}};
  EXPECT_TRUE(b::prepare_lengths(invocation,queues,bad,{.host_workspace=cell.host.ref()}).IsInvalidArgument());
  const std::array good{b::NodeLayout{cell.reader,cell.anchor}};
  check(b::prepare_lengths(invocation,queues,good,{.host_workspace=cell.host.ref()}));cell.run(invocation,{1,3},{1001,1015});
}
TEST(DeviceLengthsCuda, BackendRequestCapacityLimitIsAnExplicitTypedRefusal){
  Cell cell(33,2,32);auto invocation=take(cell.compiled.executable.create_invocation());
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&cell.queue}};
  const std::array layouts{b::NodeLayout{cell.reader,cell.anchor}};
  const auto status=b::prepare_lengths(invocation,queues,layouts,{.host_workspace=cell.host.ref()});
  EXPECT_TRUE(status.IsUnimplemented())<<status.message();
  EXPECT_NE(invocation.replay_route(),g::ReplayRoute::kCaptured);
}
TEST(DeviceLengthsCuda, ForeignSchemaRefusalPreservesOriginalPreparedEpisodeAndCanRetry){
  Cell cell(2,4,1024);auto invocation=cell.prepare();
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&cell.queue}};
  std::array<hyle::mem::Buffer,2>controls{take(hyle::mem::Buffer::create(cell.device.ref(),8,256)),
      take(hyle::mem::Buffer::create(cell.device.ref(),8,256))};
  auto output=take(hyle::mem::Buffer::create(cell.device.ref(),take(cell.output_desc.required_bytes()),256));
  const std::vector<std::int32_t>q{1,3},kv{1001,1015};
  check(controls[0].view().write(std::as_bytes(std::span(q))));check(controls[1].view().write(std::as_bytes(std::span(kv))));
  check(invocation.bind(cell.ports[0],cell.query_desc,cell.query));
  check(b::bind(invocation,cell.ports[1],cell.key_frame));check(b::bind(invocation,cell.ports[2],cell.value_frame));
  for(std::size_t p=0;p<2;++p)check(invocation.bind(cell.ports[3+p],take(c::ConstTensorView::create(cell.index_desc,controls[p].view()))));
  check(invocation.bind(cell.output,take(c::TensorView::create(cell.output_desc,output.view()))));
  check(invocation.prepare_replay(queues,{.mode=g::ReplayMode::kRequireCapture,.host_wait=g::HostWaitPolicy::kReject}));
  const auto captured=invocation.replay_report()->recorded_capture_range_count;
  const auto foreign=take(c::BatchSchema::create(cell.schema.capacity()));
  const auto other=take(foreign.version(cell.anchor.host()));
  const auto original=cell.anchor.host();const std::array<std::int32_t,2>one_request_queries{0,8};
  const auto fewer=take(cell.schema.version({one_request_queries,original.kv_indptr.first(2),
      original.page_indices.first(std::size_t(original.kv_indptr[1])),original.last_page_lengths.first(1)}));
  for(const auto&bad:{other,cell.layout({1,3}),fewer}){
    const std::array layouts{b::NodeLayout{cell.reader,bad}};
    EXPECT_TRUE(b::prepare_lengths(invocation,queues,layouts,{.host_workspace=cell.host.ref()}).IsInvalidArgument());
    EXPECT_EQ(invocation.replay_route(),g::ReplayRoute::kCaptured);
  }
  const std::array duplicate{b::NodeLayout{cell.reader,cell.anchor},b::NodeLayout{cell.reader,cell.anchor}};
  EXPECT_TRUE(b::prepare_lengths(invocation,queues,duplicate,{.host_workspace=cell.host.ref()}).IsInvalidArgument());
  auto foreign_context=take(g::cuda::create_context());auto foreign_queue=take(foreign_context.create_queue());
  const std::array other_queues{g::QueueBinding{g::ExecutionLaneId{0},&foreign_queue}};
  const std::array good{b::NodeLayout{cell.reader,cell.anchor}};
  EXPECT_TRUE(b::prepare_lengths(invocation,other_queues,good,{.host_workspace=cell.host.ref()}).IsInvalidArgument());
  EXPECT_EQ(invocation.replay_route(),g::ReplayRoute::kCaptured);
  EXPECT_EQ(invocation.replay_report()->recorded_capture_range_count,captured);
  auto done=take(invocation.submit(queues));check(done.wait());done={};
  std::vector<std::uint16_t>actual(output.size()/2);check(output.view().read(std::as_writable_bytes(std::span(actual))));
  cell.verify(actual,q,kv);
  check(b::prepare_lengths(invocation,queues,good,{.host_workspace=cell.host.ref()}));
}
TEST(DeviceLengthsCuda, OneCapturedGraphReplaysDeviceLengthsAtTheSameAddresses){
  Cell cell(2,8,1024);auto invocation=cell.prepare();
  std::array<hyle::mem::Buffer,2>controls{take(hyle::mem::Buffer::create(cell.device.ref(),8,256)),
      take(hyle::mem::Buffer::create(cell.device.ref(),8,256))};
  auto output=take(hyle::mem::Buffer::create(cell.device.ref(),take(cell.output_desc.required_bytes()),256));
  check(invocation.bind(cell.ports[0],cell.query_desc,cell.query));
  check(b::bind(invocation,cell.ports[1],cell.key_frame));check(b::bind(invocation,cell.ports[2],cell.value_frame));
  for(std::size_t p=0;p<2;++p)check(invocation.bind(cell.ports[3+p],take(c::ConstTensorView::create(cell.index_desc,controls[p].view()))));
  check(invocation.bind(cell.output,take(c::TensorView::create(cell.output_desc,output.view()))));
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&cell.queue}};
  check(invocation.prepare_replay(queues,{.mode=g::ReplayMode::kRequireCapture,.host_wait=g::HostWaitPolicy::kReject}));
  ASSERT_EQ(invocation.replay_route(),g::ReplayRoute::kCaptured);
  const auto ranges=invocation.replay_report()->recorded_capture_range_count;
  for(const std::int32_t n:{8,0,3,1,7,2}){
    const std::array<std::int32_t,2>q{n,8-n},kv{n?1001:0,n==8?0:1015};
    check(controls[0].view().write(std::as_bytes(std::span(q))));check(controls[1].view().write(std::as_bytes(std::span(kv))));
    check(output.view().fill(std::byte{0x7f}));auto done=take(invocation.submit(queues));check(done.wait());done={};
    std::vector<std::uint16_t>actual(output.size()/2);check(output.view().read(std::as_writable_bytes(std::span(actual))));
    const auto active=std::size_t(8)*kHeads*kDepth;
    for(std::size_t i=0;i<active;++i)EXPECT_TRUE(std::isfinite(value(actual[i])))<<i;
    for(std::size_t i=active;i<actual.size();++i)EXPECT_EQ(actual[i],0u)<<i;
    EXPECT_EQ(invocation.replay_route(),g::ReplayRoute::kCaptured);
    EXPECT_EQ(invocation.replay_report()->recorded_capture_range_count,ranges);
  }
}
TEST(DeviceLengthsCuda, IndependentQueuesKeepCapacityMetadataAndErrorsWithTheirOriginalInvocation){
  Cell cell(2,4,1024);auto other=take(cell.context.create_queue());
  const auto run_pair=[&](bool invalid){
    auto a=take(cell.compiled.executable.create_invocation()),b=take(cell.compiled.executable.create_invocation());
    const std::array qa{g::QueueBinding{g::ExecutionLaneId{0},&cell.queue}},
        qb{g::QueueBinding{g::ExecutionLaneId{0},&other}};
    const std::array acap{b::NodeLayout{cell.reader,cell.anchor}},
        bcap{b::NodeLayout{cell.reader,cell.layout({4,4},{31,31})}};
    std::vector<c::ReadLease>bkeys,bvalues;
    for(const auto page:bcap[0].layout.host().page_indices){bkeys.push_back(cell.key_pages[std::size_t(page)]);
      bvalues.push_back(cell.value_pages[std::size_t(page)]);}
    auto key_frame=take(cell.keys.read(bcap[0].layout,bkeys)),value_frame=take(cell.values.read(bcap[0].layout,bvalues));
    check(b::prepare_lengths(a,qa,acap,{.host_workspace=cell.host.ref()}));
    check(b::prepare_lengths(b,qb,bcap,{.host_workspace=cell.host.ref()}));
    std::array<hyle::mem::Buffer,6>buffers;
    for(std::size_t i=0;i<buffers.size();++i)buffers[i]=take(hyle::mem::Buffer::create(cell.device.ref(),
        i%3==2?take(cell.output_desc.required_bytes()):8,256));
    const std::vector<std::int32_t>aq=invalid?std::vector<std::int32_t>{5,0}:std::vector<std::int32_t>{4,1},
        ak{1001,1015},bq{1,3},bk{15,31};
    const auto bind=[&](g::Invocation&invocation,std::size_t first,const auto&q,const auto&kv){
      check(invocation.bind(cell.ports[0],cell.query_desc,cell.query));
      check(b::bind(invocation,cell.ports[1],first?key_frame:cell.key_frame));
      check(b::bind(invocation,cell.ports[2],first?value_frame:cell.value_frame));
      check(buffers[first].view().write(std::as_bytes(std::span(q))));
      check(buffers[first+1].view().write(std::as_bytes(std::span(kv))));
      for(std::size_t p=0;p<2;++p)check(invocation.bind(cell.ports[3+p],
          take(c::ConstTensorView::create(cell.index_desc,buffers[first+p].view()))));
      check(invocation.bind(cell.output,take(c::TensorView::create(cell.output_desc,buffers[first+2].view()))));
    };
    bind(a,0,aq,ak);bind(b,3,bq,bk);
    check(a.prepare_replay(qa,{.mode=g::ReplayMode::kRequireCapture,.host_wait=g::HostWaitPolicy::kReject}));
    check(b.prepare_replay(qb,{.mode=g::ReplayMode::kRequireCapture,.host_wait=g::HostWaitPolicy::kReject}));
    struct Gate final{
      std::atomic<bool>open{false},timed_out{false};std::mutex mutex;std::condition_variable changed;std::thread watchdog;
      static void CUDART_CB wait(void*opaque){static_cast<Gate*>(opaque)->open.wait(false,std::memory_order_acquire);}
      void release(){open.store(true,std::memory_order_release);open.notify_all();changed.notify_all();}
      ~Gate(){release();if(watchdog.joinable())watchdog.join();}
    }gate;
    gate.watchdog=std::thread([&]{std::unique_lock lock(gate.mutex);
      if(!gate.changed.wait_for(lock,std::chrono::seconds(10),[&]{return gate.open.load();})){gate.timed_out=true;gate.release();}});
    ASSERT_EQ(cudaLaunchHostFunc(take(c::cuda::stream(cell.queue)),&Gate::wait,&gate),cudaSuccess);
    auto adone=take(a.submit(qa)),bdone=take(b.submit(qb));check(bdone.wait());
    EXPECT_EQ(take(adone.poll()),c::CompletionPoll::kPending);EXPECT_FALSE(gate.open.load());
    gate.release();const auto status=adone.wait();adone={};bdone={};
    if(invalid){EXPECT_TRUE(status.IsInvalidArgument())<<status.message();EXPECT_FALSE(a.output(cell.output));}
    else check(status);
    const auto verify=[&](std::size_t at,const auto&q,const auto&kv){std::vector<std::uint16_t>got(buffers[at].size()/2);
      check(buffers[at].view().read(std::as_writable_bytes(std::span(got))));cell.verify(got,q,kv);};
    if(!invalid)verify(2,aq,ak);verify(5,bq,bk);EXPECT_FALSE(gate.timed_out.load());
    EXPECT_EQ(a.replay_route(),g::ReplayRoute::kCaptured);EXPECT_EQ(b.replay_route(),g::ReplayRoute::kCaptured);
  };
  run_pair(false);run_pair(true);run_pair(false);
}
} // namespace
