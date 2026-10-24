#pragma once

#include "hyle/compute/model_ops.h"
#include <array>
#include <limits>

namespace hyle::compute::detail {
// Private address translation, not a new public tensor/catalog operation.
// Only the typed reader supplies owned, validated layout bytes. Invalid
// logical IDs become -1 for the original checked codec to reject atomically.
struct RowCacheMap final {
  std::uint64_t query_offset=0,kv_offset=0,page_offset=0,tail_offset=0,metadata_bytes=0;
  std::uint32_t page_size=0,max_pages=0;
  static constexpr OperationId operation_id{0x48594c45524f5743ULL,0x504147454d415001ULL};
  static constexpr std::string_view operation_name="private-row-cache-page-map";
  constexpr OperationTraits operation_traits()const noexcept{
    return {{2,2},1,AliasPolicy::kNone,EffectKind::kPure,PrecisionPolicy::kExact};
  }
  std::uint32_t max_requests()const noexcept{return std::uint32_t((kv_offset-query_offset)/4-1);}
  std::uint32_t max_refs()const noexcept{return std::uint32_t((tail_offset-page_offset)/4);}
  Status infer_output_descriptions(std::span<const TensorDesc> in,std::span<TensorDesc> out)const noexcept{
    if(in.size()!=2 || out.size()!=1 || !in[0].valid() || !in[1].valid() ||
       !in[0].is_plain() || !in[0].is_dense() || in[0].rank()!=2 || in[0].data_type()!=DataType::kInt32 ||
       !in[1].is_plain() || !in[1].is_dense() || in[1].rank()!=1 || in[1].data_type()!=DataType::kUInt8 ||
       !page_size || !max_pages || std::uint64_t(max_pages)*page_size>std::uint64_t(INT32_MAX)+1 ||
       query_offset!=32 || kv_offset<=query_offset+4 || (kv_offset-query_offset)%4 ||
       (kv_offset-query_offset)/4-1>std::uint32_t(INT32_MAX) ||
       page_offset<kv_offset || page_offset-kv_offset!=kv_offset-query_offset ||
       tail_offset<page_offset || (tail_offset-page_offset)%4 || (tail_offset-page_offset)/4>std::uint32_t(INT32_MAX) ||
       tail_offset>metadata_bytes || std::uint64_t(max_requests())*4!=metadata_bytes-tail_offset ||
       metadata_bytes>std::uint64_t(INT64_MAX) || in[1].dimensions()[0]!=std::int64_t(metadata_bytes))
      return Status::InvalidArgument("private row cache mapping geometry differs");
    out[0]=in[0];return Status::Ok();
  }
};
static_assert(InlineCustomOperationDescriptor<RowCacheMap>);
// A separate producer chain for padded TopK. A valid score column outside
// its query range is padding; malformed columns/ranges/metadata become -2,
// which the checked padded decoder must reject. Result1 is 0/-Inf additive
// attention mask, so zero-filled decoded padding cannot affect normalization.
struct RowCachePaddedMap final {
  RowCacheMap mapping{};
  std::uint32_t score_columns=0;
  static constexpr OperationId operation_id{0x48594c45524f5743ULL,0x5041474550410001ULL};
  static constexpr std::string_view operation_name="private-row-cache-padded-map";
  constexpr OperationTraits operation_traits()const noexcept{
    return {{3,3},2,AliasPolicy::kNone,EffectKind::kPure,PrecisionPolicy::kExact};
  }
  Status infer_output_descriptions(std::span<const TensorDesc> in,std::span<TensorDesc> out)const noexcept{
    if(in.size()!=3 || out.size()!=2 || (in[0].valid() && in[0].rank()==2 && in[0].dimensions()[0] && !in[0].dimensions()[1]) || score_columns>std::uint32_t(INT32_MAX))
      return Status::InvalidArgument("padded row map input/output count or columns differ");
    std::array<TensorDesc,1> inferred;
    HYLE_RETURN_IF_ERROR(mapping.infer_output_descriptions(std::array{in[0],in[2]},inferred));
    if(!in[1].valid() || !in[1].is_plain() || !in[1].is_dense() || in[1].data_type()!=DataType::kInt32 ||
       in[1].rank()!=2 || in[1].dimensions()[0]!=in[0].dimensions()[0] || in[1].dimensions()[1]!=2)
      return Status::InvalidArgument("padded row map needs one half-open range per query");
    out[0]=inferred[0];
    HYLE_ASSIGN_OR_RETURN(out[1],TensorDesc::dense(DataType::kFloat32,in[0].dimensions()));
    return Status::Ok();
  }
};
static_assert(InlineCustomOperationDescriptor<RowCachePaddedMap>);

// Typed-only five-port scorer. Columns are request-local logical row IDs;
// owned query/page metadata selects each query's request. The raw capacity is
// never an ordinary public bank input and requires exact page read guards.
struct RowCachePagedIndexerScores final {
  RowCacheMap mapping{};
  std::uint32_t score_columns=0;
  static constexpr OperationId operation_id{0x48594c45524f5743ULL,0x5041474549440001ULL};
  static constexpr std::string_view operation_name="private-row-cache-paged-indexer-scores";
  constexpr OperationTraits operation_traits()const noexcept{
    return {{5,5},1,AliasPolicy::kNone,EffectKind::kPure,PrecisionPolicy::kAccumulateInDeclaredType,DataType::kFloat32,true};
  }
  Status infer_output_descriptions(std::span<const TensorDesc> in,std::span<TensorDesc> out)const noexcept{
    if(in.size()!=5 || out.size()!=1 || score_columns>std::uint32_t(INT32_MAX))
      return Status::InvalidArgument("paged row indexer input/output count or columns differ");
    std::array<TensorDesc,1> inferred;
    HYLE_RETURN_IF_ERROR(model_ops::RowCacheIndexerScores{}.infer_output_descriptions(in.first(4),inferred));
    HYLE_RETURN_IF_ERROR(mapping.infer_output_descriptions(std::array{in[3],in[4]},inferred));
    if(std::uint64_t(in[1].dimensions()[0])!=std::uint64_t(mapping.max_pages)*mapping.page_size)
      return Status::InvalidArgument("paged row indexer physical bank differs from metadata capacity");
    HYLE_ASSIGN_OR_RETURN(out[0],TensorDesc::dense(DataType::kFloat32,std::array{in[0].dimensions()[0],std::int64_t(score_columns)}));
    return Status::Ok();
  }
};
static_assert(InlineCustomOperationDescriptor<RowCachePagedIndexerScores>);
} // namespace hyle::compute::detail
