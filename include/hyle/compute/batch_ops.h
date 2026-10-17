#pragma once
#include "hyle/compute/batch_layout.h"
#include "hyle/compute/operation.h"
#include <cmath>

namespace hyle::compute {
// Four readonly ports: packed Q, K arena/packed K, V arena/packed V, the exact
// metadata encoding from BatchSchema. Native admission is separate from schema
// acceptance. Output capacity matches packed Q; layout owns effective rows.
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
} // namespace hyle::compute
