#include "row_cache_node.h"
#include "hyle/compute/model_ops.h"
#include "hyle/graph/detail/region_access.h"
#include "../compute_batch/row_cache_planning.h"
#include <array>
namespace hyle::graph::batch {
namespace c=compute;
Result<PagedRowCacheGather> add_row_cache_gather(GraphBuilder&builder,c::RowCacheBatchSchema schema,ValueId ids)noexcept{
  if(!schema.valid())return std::unexpected(Status::InvalidArgument("row Graph schema is empty"));
  const auto cap=schema.capacity();const auto rows=std::uint64_t(cap.max_pages)*cap.page_size;
  if(rows>std::uint64_t(INT32_MAX)+1)return std::unexpected(Status::OutOfRange("row Graph physical I32 row capacity"));
  PagedRowCacheGather result;result.schema_=std::move(schema);result.ids_=ids;
  HYLE_ASSIGN_OR_RETURN(auto bank,c::row_cache_carrier_description(cap.encoding,std::int64_t(rows)));
  HYLE_ASSIGN_OR_RETURN(result.bank_,builder.input(bank));
  HYLE_ASSIGN_OR_RETURN(auto metadata,c::TensorDesc::dense(c::DataType::kUInt8,std::array<std::int64_t,1>{std::int64_t(result.schema_.metadata_offsets().bytes)}));
  HYLE_ASSIGN_OR_RETURN(result.metadata_,builder.input(metadata));
  HYLE_ASSIGN_OR_RETURN(result.map_,builder.add_node(c::Operation::custom(detail::RowCacheAccess::operation(result)),std::array{ids,result.metadata_}));
  HYLE_ASSIGN_OR_RETURN(result.mapped_,builder.result(result.map_));
  HYLE_ASSIGN_OR_RETURN(result.decode_,builder.add_node(c::model_ops::row_cache_gather_decode(cap.encoding),std::array{result.bank_,result.mapped_}));
  HYLE_ASSIGN_OR_RETURN(result.output_,builder.result(result.decode_));return result;
}
Result<PagedRowCacheGatherPadded> add_row_cache_gather_padded(GraphBuilder&builder,c::RowCacheBatchSchema schema,
    ValueId ids,ValueId ranges,std::uint32_t columns)noexcept{
  if(!schema.valid())return std::unexpected(Status::InvalidArgument("padded row schema is empty"));
  const auto cap=schema.capacity();const auto rows=std::uint64_t(cap.max_pages)*cap.page_size;
  if(rows>std::uint64_t(INT32_MAX)+1)return std::unexpected(Status::OutOfRange("padded row physical I32 capacity"));
  HYLE_ASSIGN_OR_RETURN(auto bankdesc,c::row_cache_carrier_description(cap.encoding,std::int64_t(rows)));
  HYLE_ASSIGN_OR_RETURN(auto bank,builder.input(bankdesc));
  HYLE_ASSIGN_OR_RETURN(auto metadesc,c::TensorDesc::dense(c::DataType::kUInt8,std::array{std::int64_t(schema.metadata_offsets().bytes)}));
  HYLE_ASSIGN_OR_RETURN(auto meta,builder.input(metadesc));
  auto reader=detail::RowCacheAccess::make(schema,{},{},bank,meta,ids,{},{});
  HYLE_ASSIGN_OR_RETURN(auto map,builder.add_node(c::Operation::custom(c::detail::RowCachePaddedMap{detail::RowCacheAccess::operation(reader),columns}),std::array{ids,ranges,meta}));
  HYLE_ASSIGN_OR_RETURN(auto mapped,builder.result(map,0));HYLE_ASSIGN_OR_RETURN(auto mask,builder.result(map,1));
  HYLE_ASSIGN_OR_RETURN(auto decoder,builder.add_node(c::model_ops::row_cache_gather_decode_padded(cap.encoding),std::array{bank,mapped}));
  HYLE_ASSIGN_OR_RETURN(auto out,builder.result(decoder));
  PagedRowCacheGatherPadded result;result.reader_=detail::RowCacheAccess::make(std::move(schema),map,decoder,bank,meta,ids,mapped,out);result.mask_=mask;return result;
}
Result<PagedRowCacheIndexer> add_row_cache_indexer(GraphBuilder&builder,c::RowCacheBatchSchema schema,
    ValueId query,ValueId weights,ValueId ranges,std::uint32_t columns)noexcept{
  if(!schema.valid() || schema.capacity().encoding!=c::RowCacheEncoding::float4_block32())
    return std::unexpected(Status::InvalidArgument("paged indexer requires its FP4 row schema"));
  const auto cap=schema.capacity();const auto rows=std::uint64_t(cap.max_pages)*cap.page_size;
  if(rows>std::uint64_t(INT32_MAX)+1)return std::unexpected(Status::OutOfRange("paged indexer physical I32 capacity"));
  HYLE_ASSIGN_OR_RETURN(auto bankdesc,c::row_cache_carrier_description(cap.encoding,std::int64_t(rows)));
  HYLE_ASSIGN_OR_RETURN(auto bank,builder.input(bankdesc));
  HYLE_ASSIGN_OR_RETURN(auto metadesc,c::TensorDesc::dense(c::DataType::kUInt8,std::array{std::int64_t(schema.metadata_offsets().bytes)}));
  HYLE_ASSIGN_OR_RETURN(auto meta,builder.input(metadesc));
  auto reader=detail::RowCacheAccess::make(schema,{},{},bank,meta,query,{},{});
  HYLE_ASSIGN_OR_RETURN(auto node,builder.add_node(c::Operation::custom(c::detail::RowCachePagedIndexerScores{detail::RowCacheAccess::operation(reader),columns}),std::array{query,bank,weights,ranges,meta}));
  HYLE_ASSIGN_OR_RETURN(auto out,builder.result(node));
  PagedRowCacheIndexer result;result.reader_=detail::RowCacheAccess::make(std::move(schema),node,node,bank,meta,query,{},out);return result;
}
namespace {
bool same_geometry(const c::detail::RowCacheMap&a,const c::detail::RowCacheMap&b)noexcept{
  return a.query_offset==b.query_offset && a.kv_offset==b.kv_offset && a.page_offset==b.page_offset &&
    a.tail_offset==b.tail_offset && a.metadata_bytes==b.metadata_bytes && a.page_size==b.page_size && a.max_pages==b.max_pages;
}
}
bool detail::RowCacheAccess::empty_safe(const Graph&graph,const PagedRowCacheGather&r)noexcept{
  const auto op=graph.operation(r.decode_);return op && (op->custom_as<c::model_ops::RowCacheGatherDecodePadded>() || op->custom_as<c::detail::RowCachePagedIndexerScores>());
}
Result<mem::MemoryResourceRef> detail::RowCacheAccess::resource(
    const Executable&executable,const PagedRowCacheGather&r,ExecutionLaneId*lane)noexcept{
  const std::array nodes{r.map_,r.decode_};
  return graph::detail::RegionAccess::projection_resource(executable,
      {r.bank_,r.metadata_,r.decode_,nodes,c::detail::kRowCacheColdPlanningId},lane);
}
Status detail::RowCacheAccess::bind(graph::Invocation&invocation,const PagedRowCacheGather&r,
    c::ConstTensorView tensor,std::span<const c::ReadLease>pages,bool pending)noexcept{
  const std::array nodes{r.map_,r.decode_};
  return graph::detail::RegionAccess::bind_projection(invocation,
      {r.bank_,r.metadata_,r.decode_,nodes,c::detail::kRowCacheColdPlanningId},tensor,pages,pending);
}
Status detail::RowCacheAccess::configure(graph::Invocation&invocation,const PagedRowCacheGather&r,
    const c::RowCacheLayoutVersion&layout)noexcept{
  const auto host=layout.host();
  const c::detail::RowCacheColdLayout config{host.query_indptr,host.kv_indptr,host.last_page_lengths,layout.facts().requests};
  for(const auto node:std::array{r.map_,r.decode_}){
    HYLE_ASSIGN_OR_RETURN(auto configured,graph::detail::RegionAccess::configure_node(
        invocation,node,c::detail::kRowCacheColdPlanningId,&config));
    (void)configured;
    if(r.map_==r.decode_)break;
  }
  return Status::Ok();
}
Result<detail::RowCacheFacts> detail::RowCacheAccess::facts(const Graph&graph,const PagedRowCacheGather&r)noexcept{
  HYLE_RETURN_IF_ERROR(validate(graph,r));RowCacheFacts result;
  HYLE_ASSIGN_OR_RETURN(result.metadata,graph.value_description(r.metadata_));
  HYLE_ASSIGN_OR_RETURN(auto input,graph.node_input(r.map_,0));
  HYLE_ASSIGN_OR_RETURN(auto description,graph.value_description(input));
  result.query_rows=std::uint32_t(description.dimensions()[0]);
  HYLE_ASSIGN_OR_RETURN(auto output,graph.value_description(r.output_));
  HYLE_ASSIGN_OR_RETURN(auto count,output.element_count());result.nonempty_output=count!=0;
  result.empty_safe=empty_safe(graph,r);
  HYLE_ASSIGN_OR_RETURN(auto operation,graph.operation(r.node()));
  if(const auto scorer=operation.custom_as<c::detail::RowCachePagedIndexerScores>()){
    result.scorer=true;result.score_columns=scorer->score_columns;
  }
  return result;
}
Status detail::RowCacheAccess::validate_layout(const PagedRowCacheGather&r,
    const c::RowCacheLayoutVersion&layout,const RowCacheFacts&facts)noexcept{
  if(!r.schema().owns(layout) || layout.facts().query_rows!=facts.query_rows)
    return Status::InvalidArgument("row Graph selection rows or owned schema differ from layout");
  if(facts.nonempty_output && !layout.facts().kv_rows && !facts.empty_safe)
    return Status::OutOfRange("row Graph nonempty selections cannot read empty histories");
  if(facts.scorer){
    const auto host=layout.host();const auto page_size=r.schema().capacity().page_size;
    for(std::size_t request=0;request<layout.facts().requests;++request){
      const auto refs=host.kv_indptr[request+1]-host.kv_indptr[request];
      const auto length=refs?std::uint64_t(refs-1)*page_size+std::uint32_t(host.last_page_lengths[request]):0;
      if(length>facts.score_columns)return Status::OutOfRange("row Graph history exceeds logical score columns");
    }
  }
  return Status::Ok();
}
Status detail::RowCacheAccess::validate(const Graph&graph,const PagedRowCacheGather&r)noexcept{
  if(!graph.valid() || !r.valid())return Status::InvalidArgument("row Graph macro is empty");
  HYLE_ASSIGN_OR_RETURN(auto map,graph.operation(r.map_));HYLE_ASSIGN_OR_RETURN(auto decode,graph.operation(r.decode_));
  const auto ordinary=map.custom_as<c::detail::RowCacheMap>();
  const auto padded=map.custom_as<c::detail::RowCachePaddedMap>();
  const auto scorer=map.custom_as<c::detail::RowCachePagedIndexerScores>();
  const auto codec=decode.custom_as<c::model_ops::RowCacheGatherDecode>();
  const auto padded_codec=decode.custom_as<c::model_ops::RowCacheGatherDecodePadded>();
  const auto expected=operation(r);const c::detail::RowCacheMap*actual=nullptr;
  std::size_t metadata_port=1,bank_port=0,map_inputs=2,map_outputs=1,decode_inputs=2;
  if(scorer){actual=&scorer->mapping;metadata_port=4;bank_port=1;map_inputs=decode_inputs=5;
    if(r.map_!=r.decode_ || r.mapped_.valid() || r.schema_.capacity().encoding!=c::RowCacheEncoding::float4_block32())
      return Status::InvalidArgument("paged indexer macro role differs");
  }else if(padded){actual=&padded->mapping;metadata_port=2;map_inputs=3;map_outputs=2;
    if(!padded_codec || padded_codec->encoding!=r.schema_.capacity().encoding)return Status::InvalidArgument("padded row macro codec differs");
  }else if(ordinary){actual=&*ordinary;
    if(!codec || codec->encoding!=r.schema_.capacity().encoding || codec->bounds!=c::IndexBoundsPolicy::kChecked)
      return Status::InvalidArgument("row macro checked codec differs");
  }
  if(!actual || !same_geometry(*actual,expected))return Status::InvalidArgument("row Graph geometry differs from its owned schema");
  HYLE_ASSIGN_OR_RETURN(auto ni,graph.node_input_count(r.map_));HYLE_ASSIGN_OR_RETURN(auto no,graph.node_output_count(r.map_));
  HYLE_ASSIGN_OR_RETURN(auto di,graph.node_input_count(r.decode_));HYLE_ASSIGN_OR_RETURN(auto dout,graph.node_output_count(r.decode_));
  HYLE_ASSIGN_OR_RETURN(auto a,graph.node_input(r.map_,0));HYLE_ASSIGN_OR_RETURN(auto b,graph.node_input(r.map_,metadata_port));
  HYLE_ASSIGN_OR_RETURN(auto d,graph.node_input(r.decode_,bank_port));HYLE_ASSIGN_OR_RETURN(auto o,graph.result(r.decode_));
  HYLE_ASSIGN_OR_RETURN(auto id_desc,graph.value_description(a));
  if(ni!=map_inputs || no!=map_outputs || di!=decode_inputs || dout!=1 || a!=r.ids_ || b!=r.metadata_ || d!=r.bank_ || o!=r.output_ ||
      r.bank_.kind()!=ValueKind::kInput || r.metadata_.kind()!=ValueKind::kInput ||
      std::uint64_t(id_desc.dimensions()[0])>r.schema_.capacity().max_query_rows)
    return Status::InvalidArgument("row Graph macro ports/selection rows differ");
  if(!scorer){HYLE_ASSIGN_OR_RETURN(auto e,graph.node_input(r.decode_,1));HYLE_ASSIGN_OR_RETURN(auto m,graph.result(r.map_,0));
    if(e!=r.mapped_ || m!=r.mapped_)return Status::InvalidArgument("row Graph mapped producer port differs");}
  HYLE_ASSIGN_OR_RETURN(auto bank_desc,graph.value_description(r.bank_));
  HYLE_ASSIGN_OR_RETURN(auto required,c::row_cache_carrier_description(r.schema_.capacity().encoding,
    std::int64_t(std::uint64_t(r.schema_.capacity().max_pages)*r.schema_.capacity().page_size)));
  if(bank_desc!=required)return Status::InvalidArgument("row Graph physical capacity differs");
  for(std::size_t i=0;i<graph.output_count();++i){HYLE_ASSIGN_OR_RETURN(auto v,graph.output_id(i));
    if(v==r.bank_ || v==r.metadata_ || (r.mapped_.valid() && v==r.mapped_))return Status::InvalidArgument("row Graph private capacity/metadata/map cannot be published");}
  for(std::size_t i=0;i<graph.node_count();++i){HYLE_ASSIGN_OR_RETURN(auto node,graph.node_id(i));HYLE_ASSIGN_OR_RETURN(auto count,graph.node_input_count(node));
    for(std::size_t port=0;port<count;++port){HYLE_ASSIGN_OR_RETURN(auto v,graph.node_input(node,port));
      if((v==r.bank_ && (node!=r.decode_ || port!=bank_port)) || (v==r.metadata_ && (node!=r.map_ || port!=metadata_port)) ||
          (r.mapped_.valid() && v==r.mapped_ && (node!=r.decode_ || port!=1)))return Status::InvalidArgument("row Graph private input has an extra reader");}}
  return Status::Ok();
}
}
