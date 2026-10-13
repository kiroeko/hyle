#pragma once
#include "hyle/compute/detail/batch_provider.h"

namespace hyle::compute::ggml_cpu::internal {
// Borrowed only from the native batch slot's immutable owned topology.
// Never exposed through the ordinary public operation factory.
struct PagedRead final {
  std::span<const std::int32_t> pages;
  std::uint32_t logical_rows=0,page_size=0;
  float scale=1.F;
  bool causal=true;
  static constexpr OperationId operation_id{0x48594c4543505546ULL,0x5041474544524401ULL};
  static constexpr std::string_view operation_name="ggml-cpu-paged-read";
  constexpr OperationTraits operation_traits()const noexcept {
    return {{3,3},1,AliasPolicy::kNone,EffectKind::kPure,PrecisionPolicy::kExact,DataType::kFloat32};
  }
  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
      std::span<TensorDesc> outputs)const noexcept {
    if(inputs.size()!=3 || outputs.size()!=1 || !logical_rows || !page_size ||
       pages.empty() || !std::isfinite(scale) || inputs[0].rank()!=4 ||
       inputs[1].rank()!=4 || inputs[2]!=inputs[1] ||
       (causal && inputs[0].dimensions()[1]>logical_rows))
      return Status::InvalidArgument("CPU paged read domain");
    return compute::infer_output_descriptions(Operation(ScaledDotProductAttention{
        .layout=AttentionLayout::kBatchSequenceHeadDepth,.scale=scale,.causal=false}),inputs,outputs);
  }
};
static_assert(detail::InlineCustomOperationDescriptor<PagedRead>);
const detail::BatchProviderVTable *native_batch_table() noexcept;
std::size_t native_invocation_bytes(void *, void *) noexcept;
bool native_invocation_reads_kv(void *) noexcept;
}
