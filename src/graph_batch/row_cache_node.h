#pragma once
#include "hyle/graph/batch.h"
#include "../compute_batch/row_cache_map.h"
namespace hyle::graph::batch::detail {
struct RowCacheFacts final {
  compute::TensorDesc metadata;
  std::uint32_t query_rows=0,score_columns=0;
  bool nonempty_output=false,empty_safe=false,scorer=false;
};
struct RowCacheAccess final {
  static ValueId metadata(const PagedRowCacheGather&v)noexcept{return v.metadata_;}
  static NodeId map(const PagedRowCacheGather&v)noexcept{return v.map_;}
  static compute::detail::RowCacheMap operation(const PagedRowCacheGather&v)noexcept{
    const auto offsets=v.schema_.metadata_offsets();const auto cap=v.schema_.capacity();
    return {offsets.query_indptr,offsets.kv_indptr,offsets.page_indices,offsets.last_page_lengths,offsets.bytes,cap.page_size,cap.max_pages};
  }
  static PagedRowCacheGather make(compute::RowCacheBatchSchema schema,NodeId map,NodeId decode,
      ValueId bank,ValueId metadata,ValueId ids,ValueId mapped,ValueId output)noexcept{
    PagedRowCacheGather r;r.schema_=std::move(schema);r.map_=map;r.decode_=decode;
    r.bank_=bank;r.metadata_=metadata;r.ids_=ids;r.mapped_=mapped;r.output_=output;return r;
  }
  static const PagedRowCacheGather& reader(const PagedRowCacheIndexer&v)noexcept{return v.reader_;}
  static bool empty_safe(const Graph&,const PagedRowCacheGather&)noexcept;
  static Status validate(const Graph&,const PagedRowCacheGather&)noexcept;
  static Result<RowCacheFacts> facts(const Graph&,const PagedRowCacheGather&)noexcept;
  static Result<mem::MemoryResourceRef> resource(const Executable&,
      const PagedRowCacheGather&,ExecutionLaneId*)noexcept;
  static Status bind(graph::Invocation&,const PagedRowCacheGather&,
      compute::ConstTensorView,std::span<const compute::ReadLease>,bool pending=false)noexcept;
  static Status configure(graph::Invocation&,const PagedRowCacheGather&,
      const compute::RowCacheLayoutVersion&)noexcept;
  static Status validate_layout(const PagedRowCacheGather&,
      const compute::RowCacheLayoutVersion&,const RowCacheFacts&)noexcept;
};
}
