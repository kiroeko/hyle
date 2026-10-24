#pragma once
#include "hyle/compute/batch_layout.h"
#include "hyle/compute/operation.h"
#include <cmath>

namespace hyle::compute {
// Four readonly ports: packed Q, K arena/packed K, V arena/packed V, the exact
// metadata encoding from BatchSchema. Native admission is separate from schema
// acceptance. Packed Q/output extent may be below max_query_rows; the owned
// layout's active query rows must fit that exact extent during preparation.
HYLE_COMPUTE_BATCH_API Status infer_batch_attention(BatchCapacity, BatchStorage,
    std::span<const TensorDesc>, std::span<TensorDesc>) noexcept;
struct PagedAttention final {
  BatchCapacity capacity{};
  float scale = 1.0F;
  bool causal = true;
  // Inclusive past count at the bottom-right aligned query position. -1 is
  // unbounded, 0 is self-only. This batch admits a window only with causal.
  std::int32_t window_left = -1;
  // Must match the batch provider's immutable program option. This permits
  // reduced intermediate storage, while accumulation remains F32.
  bool allow_bf16_partials = false;
  static constexpr OperationId operation_id{0x48594c4542415443ULL,0x5041474544410002ULL};
  static constexpr std::string_view operation_name="paged-attention";
  constexpr OperationTraits operation_traits() const noexcept {
    return {{4,4},1,AliasPolicy::kNone,EffectKind::kPure,
      allow_bf16_partials?PrecisionPolicy::kAccumulateInDeclaredType:PrecisionPolicy::kExact,DataType::kFloat32};
  }
  Status infer_output_descriptions(std::span<const TensorDesc> in,std::span<TensorDesc> out) const noexcept {
    if(!std::isfinite(scale))return Status::InvalidArgument("paged attention scale must be finite");
    HYLE_RETURN_IF_ERROR(validate_batch_attention_options(causal, window_left));
    return infer_batch_attention(capacity,BatchStorage::kPaged,in,out);
  }
};
struct RaggedAttention final {
  BatchCapacity capacity{.storage=BatchStorage::kRagged,.max_pages=0,.max_page_refs=0,.page_size=1};
  float scale = 1.0F;
  bool causal = true;
  std::int32_t window_left = -1;
  bool allow_bf16_partials = false;
  static constexpr OperationId operation_id{0x48594c4542415443ULL,0x5241474745440002ULL};
  static constexpr std::string_view operation_name="ragged-attention";
  constexpr OperationTraits operation_traits() const noexcept {
    return {{4,4},1,AliasPolicy::kNone,EffectKind::kPure,
      allow_bf16_partials?PrecisionPolicy::kAccumulateInDeclaredType:PrecisionPolicy::kExact,DataType::kFloat32};
  }
  Status infer_output_descriptions(std::span<const TensorDesc> in,std::span<TensorDesc> out) const noexcept {
    if(!std::isfinite(scale))return Status::InvalidArgument("ragged attention scale must be finite");
    HYLE_RETURN_IF_ERROR(validate_batch_attention_options(causal, window_left));
    return infer_batch_attention(capacity,BatchStorage::kRagged,in,out);
  }
};
static_assert(detail::InlineCustomOperationDescriptor<PagedAttention>);
static_assert(detail::InlineCustomOperationDescriptor<RaggedAttention>);
// Q, the single packed CKV/KPE cache, and schema metadata. The cache's CKV
// prefix is also V; this operation never binds an independent value tensor.
struct PagedMlaAttention final {
  MlaBatchCapacity capacity{};
  float scale=1.0F;
  bool causal=true;
  std::int32_t window_left=-1;
  static constexpr OperationId operation_id{0x48594c454d4c4142ULL,0x5041474544410003ULL};
  static constexpr std::string_view operation_name="paged-mla-attention";
  constexpr OperationTraits operation_traits() const noexcept {
    return {{3,3},1,AliasPolicy::kNone,EffectKind::kPure,PrecisionPolicy::kExact,DataType::kFloat32};
  }
  Status infer_output_descriptions(std::span<const TensorDesc>in,std::span<TensorDesc>out)const noexcept {
    if(!std::isfinite(scale) || capacity.storage!=BatchStorage::kPaged || in.size()!=3 || out.size()!=1)
      return Status::InvalidArgument("paged MLA options/ports are invalid");
    HYLE_RETURN_IF_ERROR(validate_batch_attention_options(causal,window_left));
    HYLE_ASSIGN_OR_RETURN(auto c,mla_batch_capacity(capacity));
    HYLE_ASSIGN_OR_RETURN(auto q,batch_query_description(c));
    HYLE_ASSIGN_OR_RETURN(auto cache,batch_cache_description(c,BatchCacheRole::kKey));
    HYLE_ASSIGN_OR_RETURN(auto metadata,batch_metadata_description(c));
    if(in[0]!=q || in[1]!=cache || in[2]!=metadata)return Status::InvalidArgument("paged MLA capacity differs from ports");
    HYLE_ASSIGN_OR_RETURN(out[0],batch_output_description(c,c.max_query_rows));return Status::Ok();
  }
};
static_assert(detail::InlineCustomOperationDescriptor<PagedMlaAttention>);
struct RaggedMlaAttention final {
  MlaBatchCapacity capacity{.storage=BatchStorage::kRagged,.max_pages=0,.max_page_refs=0,.page_size=1};
  float scale=1.0F;
  bool causal=true;
  std::int32_t window_left=-1;
  static constexpr OperationId operation_id{0x48594c454d4c4142ULL,0x5241474745440003ULL};
  static constexpr std::string_view operation_name="ragged-mla-attention";
  constexpr OperationTraits operation_traits() const noexcept {
    return {{3,3},1,AliasPolicy::kNone,EffectKind::kPure,PrecisionPolicy::kExact,DataType::kFloat32};
  }
  Status infer_output_descriptions(std::span<const TensorDesc>in,std::span<TensorDesc>out)const noexcept {
    if(!std::isfinite(scale) || capacity.storage!=BatchStorage::kRagged || in.size()!=3 || out.size()!=1)
      return Status::InvalidArgument("ragged MLA options/ports are invalid");
    HYLE_RETURN_IF_ERROR(validate_batch_attention_options(causal,window_left));
    HYLE_ASSIGN_OR_RETURN(auto c,mla_batch_capacity(capacity));
    HYLE_ASSIGN_OR_RETURN(auto q,batch_query_description(c));
    HYLE_ASSIGN_OR_RETURN(auto cache,batch_cache_description(c,BatchCacheRole::kKey));
    HYLE_ASSIGN_OR_RETURN(auto metadata,batch_metadata_description(c));
    if(in[0]!=q || in[1]!=cache || in[2]!=metadata)return Status::InvalidArgument("ragged MLA capacity differs from ports");
    HYLE_ASSIGN_OR_RETURN(out[0],batch_output_description(c,c.max_query_rows));return Status::Ok();
  }
};
static_assert(detail::InlineCustomOperationDescriptor<RaggedMlaAttention>);
} // namespace hyle::compute
