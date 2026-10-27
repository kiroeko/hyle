#pragma once

#include <hyle/graph/batch.h>
#include <array>
#include <cstdint>
#include <span>
#include <utility>

namespace hyle_test::query_offsets {
namespace c=hyle::compute;namespace g=hyle::graph;namespace gb=g::batch;namespace m=hyle::mem;
using hyle::Status;
inline Status require(bool value,const char*message){return value?Status::Ok():Status::Internal(message);}

// The ordinary I32 Copy really consumes the public projection, while the same
// original Graph's attention consumes its owned opaque metadata separately.
inline Status exercise(const c::Context&context,m::MemoryResourceRef memory,
    m::MemoryResourceRef host,c::Queue&queue,bool owned_layout){
  constexpr std::uint32_t depth=64;
  HYLE_ASSIGN_OR_RETURN(auto plain,c::CacheStorage::plain(c::DataType::kFloat16));
  const c::BatchCapacity cap{.query_type=c::DataType::kFloat16,.output_type=c::DataType::kFloat16,
      .key_storage=plain,.value_storage=plain,.max_requests=2,.max_query_rows=8,.max_kv_rows=8,
      .max_pages=2,.max_page_refs=2,.query_heads=1,.kv_heads=1,.qk_head_dim=depth,.value_head_dim=depth,.page_size=4};
  HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create(cap));
  const std::array<std::int32_t,3> first{0,1,4},second{0,2,4},ptr{0,1,2};
  const std::array<std::int32_t,2> ids{0,1},tails{4,4};
  HYLE_ASSIGN_OR_RETURN(auto before,schema.version({first,ptr,ids,tails}));
  HYLE_ASSIGN_OR_RETURN(auto after,schema.version({second,ptr,ids,tails}));
  HYLE_ASSIGN_OR_RETURN(auto query_desc,c::batch_query_description(cap,4));
  HYLE_ASSIGN_OR_RETURN(auto bank_desc,c::batch_cache_description(schema,c::BatchCacheRole::kKey));
  HYLE_ASSIGN_OR_RETURN(auto offset_desc,c::TensorDesc::dense(c::DataType::kInt32,std::array<std::int64_t,1>{3}));
  HYLE_ASSIGN_OR_RETURN(auto builder,g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto query,builder.input(query_desc));
  HYLE_ASSIGN_OR_RETURN(auto key,builder.input(bank_desc));
  HYLE_ASSIGN_OR_RETURN(auto value,builder.input(bank_desc));
  HYLE_ASSIGN_OR_RETURN(auto offsets,builder.input(offset_desc));
  g::NodeId attention;
  if(owned_layout){
    HYLE_ASSIGN_OR_RETURN(auto op,gb::paged_attention(before,{.scale=.125F,.causal=false}));
    HYLE_ASSIGN_OR_RETURN(attention,builder.add_node(c::Operation::custom(op),std::array{query,key,value}));
  }else{
    HYLE_ASSIGN_OR_RETURN(auto metadata_desc,schema.metadata_description());
    HYLE_ASSIGN_OR_RETURN(auto metadata,builder.input(metadata_desc));
    HYLE_ASSIGN_OR_RETURN(attention,builder.add_node(c::Operation::custom(c::PagedAttention{
        .capacity=cap,.scale=.125F,.causal=false}),std::array{query,key,value,metadata}));
  }
  HYLE_ASSIGN_OR_RETURN(auto attended,builder.result(attention));HYLE_RETURN_IF_ERROR(builder.publish(attended));
  HYLE_ASSIGN_OR_RETURN(auto copied,builder.add(c::Copy{},offsets));HYLE_RETURN_IF_ERROR(builder.publish(copied));
  HYLE_ASSIGN_OR_RETURN(auto graph,std::move(builder).finish());
  const std::array domains{g::StorageDomain{memory}};
  HYLE_ASSIGN_OR_RETURN(auto source,g::ExecutionSource::acquire(context));
  const std::array targets{g::ExecutionTarget{source,g::StorageDomainId{0}}};
  HYLE_ASSIGN_OR_RETURN(auto compiled,gb::compile(graph,{.storage_domains=domains,.targets=targets}));
  HYLE_ASSIGN_OR_RETURN(auto run,compiled.executable.create_invocation({.resources=g::ResourceOptions{}}));
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&queue}};
  const auto unprepared=gb::query_offsets(run,attention);
  HYLE_RETURN_IF_ERROR(require(!unprepared && unprepared.error().IsFailedPrecondition() &&
      run.replay_route()==g::ReplayRoute::kUnprepared,"unprepared offset query changed preparation or exported permission"));
  HYLE_ASSIGN_OR_RETURN(auto foreign_builder,g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto foreign_input,foreign_builder.input(offset_desc));
  HYLE_ASSIGN_OR_RETURN(auto foreign,foreign_builder.add_node(c::Operation(c::Copy{}),std::array{foreign_input}));
  HYLE_ASSIGN_OR_RETURN(auto query_buffer,m::Buffer::create(memory,query_desc.required_bytes().value(),256));
  HYLE_ASSIGN_OR_RETURN(auto attention_output,m::Buffer::create(memory,query_desc.required_bytes().value(),256));
  HYLE_ASSIGN_OR_RETURN(auto offset_output,m::Buffer::create(memory,offset_desc.required_bytes().value(),256));
  HYLE_RETURN_IF_ERROR(query_buffer.view().fill(std::byte{0}));
  HYLE_RETURN_IF_ERROR(run.bind(query,c::ConstTensorView::create(query_desc,query_buffer.view()).value()));
  HYLE_RETURN_IF_ERROR(run.bind(attended,c::TensorView::create(query_desc,attention_output.view()).value()));
  HYLE_RETURN_IF_ERROR(run.bind(copied,c::TensorView::create(offset_desc,offset_output.view()).value()));
  HYLE_ASSIGN_OR_RETURN(auto keys,c::PagedArena::create(schema,c::BatchCacheRole::kKey,memory,{.max_backing_bytes=2048}));
  HYLE_ASSIGN_OR_RETURN(auto values,c::PagedArena::create(schema,c::BatchCacheRole::kValue,memory,{.max_backing_bytes=2048}));
  std::array<c::ReadLease,2> key_guards,value_guards;
  std::array<std::uint16_t,4*depth> words{};
  for(std::uint32_t page=0;page<2;++page){
    HYLE_ASSIGN_OR_RETURN(auto kw,keys.acquire(page));HYLE_ASSIGN_OR_RETURN(auto vw,values.acquire(page));
    HYLE_RETURN_IF_ERROR(kw.lease().view()->fill(std::byte{0}));
    words.fill(std::uint16_t(page?0x4200:0x3c00)); // Exact F16 constants 3 and 1.
    HYLE_RETURN_IF_ERROR(vw.lease().view()->write(std::as_bytes(std::span(words))));
    HYLE_ASSIGN_OR_RETURN(key_guards[page],kw.publish(4));HYLE_ASSIGN_OR_RETURN(value_guards[page],vw.publish(4));
  }
  const gb::PrepareOptions options{.host_workspace=host};
  const auto prepare=[&](c::BatchLayoutVersion layout){return gb::prepare(run,queues,std::array{gb::NodeLayout{attention,layout}},options);};
  const auto bind_pages=[&](c::BatchLayoutVersion layout)->Status{
    HYLE_ASSIGN_OR_RETURN(auto k,keys.read(layout,key_guards));HYLE_ASSIGN_OR_RETURN(auto v,values.read(layout,value_guards));
    HYLE_RETURN_IF_ERROR(gb::bind(run,key,std::move(k)));return gb::bind(run,value,std::move(v));
  };
  HYLE_RETURN_IF_ERROR(prepare(before));HYLE_RETURN_IF_ERROR(bind_pages(before));
  HYLE_ASSIGN_OR_RETURN(auto projection,gb::query_offsets(run,attention));
  HYLE_ASSIGN_OR_RETURN(auto again,gb::query_offsets(run,attention));
  HYLE_ASSIGN_OR_RETURN(auto a,projection.view());HYLE_ASSIGN_OR_RETURN(auto b,again.view());
  HYLE_RETURN_IF_ERROR(require(projection.size()==sizeof(first) && projection.generation()==again.generation() &&
      a.resource()==b.resource() && a.native_handle()==b.native_handle() && a.offset()==b.offset() && a.size()==b.size(),
      "offset projection copied storage or exposed more than B+1 I32 entries"));
  HYLE_RETURN_IF_ERROR(run.bind(offsets,offset_desc,projection));again={};
  HYLE_RETURN_IF_ERROR(run.prepare_replay(queues));
  const auto route=run.replay_route();
  const auto rejected=gb::query_offsets(run,foreign);
  HYLE_RETURN_IF_ERROR(require(!rejected && rejected.error().IsInvalidArgument() && run.replay_route()==route,
      "foreign node query changed existing binding or preparation"));
  HYLE_ASSIGN_OR_RETURN(auto completion,run.submit(queues));
  {
    const bool pending=run.in_flight();
    const auto unobserved=gb::query_offsets(run,attention);
    if(pending)HYLE_RETURN_IF_ERROR(require(!unobserved && unobserved.error().IsUnavailable(),
        "offset query bypassed the actual in-flight/unobserved Completion gate"));
  }
  HYLE_RETURN_IF_ERROR(completion.wait());
  {
    // wait() settles/releases the original observation proxy, even though the
    // value remains valid. Only owned-layout execution keeps its prepared token.
    const auto settled=gb::query_offsets(run,attention);
    if(owned_layout){
      HYLE_RETURN_IF_ERROR(require(bool(settled) && settled->generation()==projection.generation() &&
          settled->size()==projection.size(),"settled owned-layout offsets lost their actual prepared mirror"));
      HYLE_ASSIGN_OR_RETURN(auto current,settled->view());
      HYLE_RETURN_IF_ERROR(require(current.resource()==a.resource() && current.native_handle()==a.native_handle() &&
          current.offset()==a.offset() && current.size()==a.size(),"settled owned-layout offsets replaced the mirror carrier"));
    }else HYLE_RETURN_IF_ERROR(require(!settled && settled.error().IsFailedPrecondition(),
        "consumed explicit-metadata token still exported query offsets"));
  }
  completion={};
  std::array<std::int32_t,3> actual{};
  HYLE_RETURN_IF_ERROR(offset_output.view().read(std::as_writable_bytes(std::span(actual))));
  HYLE_RETURN_IF_ERROR(require(actual==first,"ordinary I32 consumer did not read the prepared offset projection"));
  // The original caller binding and local alias retain this exact mirror.
  // No replacement carrier or hidden copy may evade its normal unique gate.
  const auto pinned=prepare(after);
  HYLE_RETURN_IF_ERROR(require(pinned.IsUnavailable(),"held query-offset alias did not prevent mirror refresh"));
  HYLE_RETURN_IF_ERROR(projection.view()->read(std::as_writable_bytes(std::span(actual))));
  HYLE_RETURN_IF_ERROR(require(actual==first,"rejected mirror refresh changed retained old offsets"));
  const std::array detach{offsets};HYLE_RETURN_IF_ERROR(run.unbind(detach));projection={};
  HYLE_RETURN_IF_ERROR(prepare(after));HYLE_RETURN_IF_ERROR(bind_pages(after));
  HYLE_ASSIGN_OR_RETURN(auto next,gb::query_offsets(run,attention));
  HYLE_RETURN_IF_ERROR(run.bind(offsets,offset_desc,next));HYLE_RETURN_IF_ERROR(run.prepare_replay(queues));
  HYLE_ASSIGN_OR_RETURN(completion,run.submit(queues));HYLE_RETURN_IF_ERROR(completion.wait());completion={};
  HYLE_RETURN_IF_ERROR(offset_output.view().read(std::as_writable_bytes(std::span(actual))));
  HYLE_RETURN_IF_ERROR(require(actual==second,"released offset aliases did not permit the real metadata update"));
  run={};
  HYLE_RETURN_IF_ERROR(next.view()->read(std::as_writable_bytes(std::span(actual))));
  return require(actual==second,"query-offset lease did not retain its original allocation after Invocation destruction");
}
} // namespace hyle_test::query_offsets
