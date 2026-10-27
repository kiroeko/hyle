#pragma once

#include <hyle/graph/batch.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <span>
#include <utility>
#include <vector>

namespace hyle_test::paged_append_active {
namespace c=hyle::compute;namespace g=hyle::graph;namespace gb=g::batch;
using hyle::Status;
inline Status require(bool value,const char*message){return value?Status::Ok():Status::Internal(message);}

// One physical R=8 carrier, an independently bounded schema C=16, and only A
// active new rows. All words are initialized; padding is an observable sentinel.
inline Status run(const c::Context&context,hyle::mem::MemoryResourceRef memory,
    hyle::mem::MemoryResourceRef host,c::Queue&queue,bool capture,std::uint32_t active,bool two_requests){
  constexpr std::size_t depth=64,page_words=2*depth;
  constexpr std::uint16_t sentinel=0x7fc1;
  HYLE_ASSIGN_OR_RETURN(auto plain,c::CacheStorage::plain(c::DataType::kBFloat16));
  const c::BatchCapacity cap{.query_type=c::DataType::kBFloat16,.output_type=c::DataType::kBFloat16,
      .key_storage=plain,.value_storage=plain,.max_requests=2,.max_query_rows=16,.max_kv_rows=32,
      .max_pages=20,.max_page_refs=18,.query_heads=1,.kv_heads=1,.qk_head_dim=64,.value_head_dim=64,.page_size=2};
  HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create(cap));
  const auto requests=two_requests?2u:1u;
  std::vector<std::int32_t> old_q{0},old_ptr{0},old_ids,old_tails(requests,1);
  for(std::uint32_t j=0;j<requests;++j){old_q.push_back(std::int32_t(j+1));old_ptr.push_back(std::int32_t(j+1));old_ids.push_back(std::int32_t(j));}
  HYLE_ASSIGN_OR_RETURN(auto before,schema.version({old_q,old_ptr,old_ids,old_tails}));
  HYLE_ASSIGN_OR_RETURN(auto arena,c::PagedArena::create(schema,c::BatchCacheRole::kValue,memory,{.max_backing_bytes=8192}));
  std::vector<c::ReadLease> old_guards(requests);
  std::vector<std::array<std::uint16_t,page_words>> old_words(requests);
  for(std::uint32_t j=0;j<requests;++j){
    HYLE_ASSIGN_OR_RETURN(auto writer,arena.acquire(j));
    old_words[j].fill(sentinel);std::fill_n(old_words[j].begin(),depth,std::uint16_t(0x3e00+j));
    HYLE_ASSIGN_OR_RETURN(auto bytes,writer.lease().view());
    HYLE_RETURN_IF_ERROR(bytes.write(std::as_bytes(std::span(old_words[j]))));
    // Both physical rows were initialized. The old logical layout still grants
    // exactly one row; retain the full initialized guard only for this oracle.
    HYLE_ASSIGN_OR_RETURN(old_guards[j],writer.publish(2));
  }
  HYLE_ASSIGN_OR_RETURN(auto previous,arena.read(before,old_guards));
  HYLE_ASSIGN_OR_RETURN(auto rows,c::TensorDesc::dense(c::DataType::kBFloat16,std::array<std::int64_t,3>{8,1,64}));
  HYLE_ASSIGN_OR_RETURN(auto op,gb::paged_append(schema,c::BatchCacheRole::kValue));
  HYLE_ASSIGN_OR_RETURN(auto wrong_encoding,c::TensorDesc::dense(c::DataType::kBFloat16,std::array<std::int64_t,3>{8,2,32}));
  const std::array wrong_input{wrong_encoding};
  HYLE_RETURN_IF_ERROR(require(op.infer_output_descriptions(wrong_input,{}).IsInvalidArgument(),"append accepted equal-byte but differently encoded rows"));
  HYLE_ASSIGN_OR_RETURN(auto builder,g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto input,builder.input(rows));
  HYLE_ASSIGN_OR_RETURN(auto node,builder.add_node(c::Operation::custom(op),std::array{input}));
  HYLE_ASSIGN_OR_RETURN(auto graph,std::move(builder).finish());
  const std::array domains{g::StorageDomain{memory}};
  HYLE_ASSIGN_OR_RETURN(auto source,g::ExecutionSource::acquire(context));
  const std::array targets{g::ExecutionTarget{source,g::StorageDomainId{0}}};
  HYLE_ASSIGN_OR_RETURN(auto compiled,gb::compile(graph,{.storage_domains=domains,.targets=targets}));
  HYLE_ASSIGN_OR_RETURN(auto invocation,compiled.executable.create_invocation({.resources=g::ResourceOptions{}}));
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&queue}};
  HYLE_RETURN_IF_ERROR(gb::prepare(invocation,queues,std::span<const gb::NodeLayout>{},{.host_workspace=host}));
  std::array<std::uint16_t,page_words> padding;padding.fill(sentinel);
  // Invalid A=9 is legal under C=16, but cannot consume an R=8 carrier.
  // The cold rejection must retain every original writer and leave its bytes.
  if(!two_requests && active==1){
    const std::array<std::int32_t,2> q{0,9},ptr{0,5};
    const std::array<std::int32_t,5> ids{12,13,14,15,16};const std::array<std::int32_t,1> tail{2};
    HYLE_ASSIGN_OR_RETURN(auto invalid_layout,schema.version({q,ptr,ids,tail}));
    std::array<c::PagedPageWriter,5> dirty;
    for(std::size_t i=0;i<dirty.size();++i){
      HYLE_ASSIGN_OR_RETURN(dirty[i],arena.acquire(std::uint32_t(ids[i])));
      HYLE_ASSIGN_OR_RETURN(auto bytes,dirty[i].lease().view());HYLE_RETURN_IF_ERROR(bytes.write(std::as_bytes(std::span(padding))));
    }
    HYLE_ASSIGN_OR_RETURN(auto invalid,gb::PagedAppendPages::create(arena,invalid_layout,previous,dirty));
    const auto status=gb::bind_append(invocation,graph,node,std::move(invalid));
    HYLE_RETURN_IF_ERROR(require(status.IsInvalidArgument() && invalid.valid(),"A>R rejection consumed the original transaction"));
    for(std::size_t i=0;i<invalid.page_count();++i){
      HYLE_ASSIGN_OR_RETURN(auto&writer,invalid.page(i));HYLE_ASSIGN_OR_RETURN(auto bytes,writer.lease().view());
      std::array<std::uint16_t,page_words> actual{};HYLE_RETURN_IF_ERROR(bytes.read(std::as_writable_bytes(std::span(actual))));
      HYLE_RETURN_IF_ERROR(require(actual==padding,"invalid active extent changed a destination"));
    }
  }
  std::vector<std::int32_t> q{0},ptr{0},ids,tails;
  std::vector<std::uint32_t> counts,first_new;
  std::uint32_t packed=0,next_page=requests;
  for(std::uint32_t j=0;j<requests;++j){
    const auto count=two_requests?(j==0?1u:active-1):active;
    counts.push_back(count);first_new.push_back(packed);packed+=count;q.push_back(std::int32_t(packed));
    const auto total=1+count,pages=(total+1)/2;
    for(std::uint32_t page=0;page<pages;++page)ids.push_back(std::int32_t(next_page++));
    ptr.push_back(std::int32_t(ids.size()));tails.push_back(std::int32_t(total%2?1:2));
  }
  HYLE_ASSIGN_OR_RETURN(auto after,schema.version({q,ptr,ids,tails}));
  std::vector<c::PagedPageWriter> dirty(ids.size());
  for(std::size_t i=0;i<dirty.size();++i){
    HYLE_ASSIGN_OR_RETURN(dirty[i],arena.acquire(std::uint32_t(ids[i])));
    HYLE_ASSIGN_OR_RETURN(auto bytes,dirty[i].lease().view());HYLE_RETURN_IF_ERROR(bytes.write(std::as_bytes(std::span(padding))));
  }
  HYLE_ASSIGN_OR_RETURN(auto transaction,gb::PagedAppendPages::create(arena,after,previous,dirty));
  HYLE_RETURN_IF_ERROR(gb::bind_append(invocation,graph,node,std::move(transaction)));
  HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(memory,2048,{.max_blocks=1}));
  std::array<std::uint16_t,8*depth> incoming;incoming.fill(sentinel);
  for(std::uint32_t row=0;row<active;++row)std::fill_n(incoming.begin()+row*depth,depth,std::uint16_t(0x3f00+row));
  HYLE_ASSIGN_OR_RETURN(auto writer,pool.allocate(sizeof(incoming),256));HYLE_ASSIGN_OR_RETURN(auto input_bytes,writer.view());
  HYLE_RETURN_IF_ERROR(input_bytes.write(std::as_bytes(std::span(incoming))));
  HYLE_ASSIGN_OR_RETURN(auto guard,writer.publish());HYLE_RETURN_IF_ERROR(invocation.bind(input,rows,guard));
  HYLE_RETURN_IF_ERROR(invocation.prepare_replay(queues,{.mode=capture?g::ReplayMode::kRequireCapture:g::ReplayMode::kCommands}));
  if(capture){
    HYLE_ASSIGN_OR_RETURN(auto report,invocation.replay_report());
    HYLE_RETURN_IF_ERROR(require(report.route==g::ReplayRoute::kCaptured && report.recorded_capture_range_count>0,
        "active append did not record the original native captured route"));
  }
  HYLE_ASSIGN_OR_RETURN(auto completion,invocation.submit(queues));HYLE_RETURN_IF_ERROR(completion.wait());completion={};
  std::array<std::uint16_t,8*depth> unchanged{};
  HYLE_ASSIGN_OR_RETURN(auto source_bytes,guard.view());
  HYLE_RETURN_IF_ERROR(source_bytes.read(std::as_writable_bytes(std::span(unchanged))));
  HYLE_RETURN_IF_ERROR(require(unchanged==incoming,"active append changed its carrier or padded rows"));
  HYLE_ASSIGN_OR_RETURN(auto returned,gb::take_append(invocation,node));
  for(std::uint32_t j=0;j<requests;++j)for(auto page=ptr[j];page<ptr[j+1];++page){
    HYLE_ASSIGN_OR_RETURN(auto&target,returned.page(std::size_t(page)));HYLE_ASSIGN_OR_RETURN(auto bytes,target.lease().view());
    std::array<std::uint16_t,page_words> actual{};HYLE_RETURN_IF_ERROR(bytes.read(std::as_writable_bytes(std::span(actual))));
    for(std::size_t index=0;index<actual.size();++index){
      const auto row=std::size_t(page-ptr[j])*2+index/depth;
      const auto expected=row==0?std::uint16_t(0x3e00+j):row<=counts[j]?std::uint16_t(0x3f00+first_new[j]+row-1):sentinel;
      HYLE_RETURN_IF_ERROR(require(actual[index]==expected,"active append copied padding or lost a COW/new row"));
    }
  }
  HYLE_ASSIGN_OR_RETURN(auto ready,returned.ready());
  HYLE_RETURN_IF_ERROR(require(ready.layout()==after && previous.layout()==before,"active append changed an original layout"));
  for(std::uint32_t j=0;j<requests;++j){
    HYLE_ASSIGN_OR_RETURN(auto bytes,old_guards[j].view());std::array<std::uint16_t,page_words> actual{};
    HYLE_RETURN_IF_ERROR(bytes.read(std::as_writable_bytes(std::span(actual))));
    HYLE_RETURN_IF_ERROR(require(actual==old_words[j],"active append changed an old snapshot or its initialized padding"));
  }
  std::printf("paged append R8/A%u/B%u exact active rows and padding PASS\n",active,requests);
  return Status::Ok();
}
inline Status matrix(const c::Context&context,hyle::mem::MemoryResourceRef memory,
    hyle::mem::MemoryResourceRef host,c::Queue&queue,bool capture){
  for(const auto active:{1u,3u,8u})HYLE_RETURN_IF_ERROR(run(context,memory,host,queue,capture,active,false));
  return run(context,memory,host,queue,capture,4,true);
}
} // namespace hyle_test::paged_append_active
