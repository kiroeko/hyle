#pragma once
#include <array>
#include "hyle/compute/detail/batch_provider.h"

namespace hyle::compute::ggml_cpu::internal {
// Borrowed only from the native batch slot's immutable owned topology. Empty
// pages select contiguous ragged rows; both use the original flash operation.
// Never exposed through the ordinary public operation factory.
struct PagedRead final {
  std::span<const std::int32_t> pages;
  std::uint32_t logical_rows=0,page_size=0;
  float scale=1.F;
  bool causal=true;
  std::int32_t window_left=-1;
  static constexpr OperationId operation_id{0x48594c4543505546ULL,0x5041474544524401ULL};
  static constexpr std::string_view operation_name="ggml-cpu-paged-read";
  constexpr OperationTraits operation_traits()const noexcept {
    return {{3,3},1,AliasPolicy::kNone,EffectKind::kPure,PrecisionPolicy::kExact,DataType::kFloat32};
  }
  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
      std::span<TensorDesc> outputs)const noexcept {
    if(inputs.size()!=3 || outputs.size()!=1 || !logical_rows || (!pages.empty() && !page_size) ||
       !std::isfinite(scale) || inputs[0].rank()!=4 ||
       inputs[1].rank()!=4 || inputs[2].rank()!=4 ||
       (causal && inputs[0].dimensions()[1]>logical_rows))
      return Status::InvalidArgument("CPU paged read domain");
    HYLE_RETURN_IF_ERROR(validate_batch_attention_options(causal,window_left));
    const auto q=inputs[0].dimensions(),k=inputs[1].dimensions(),v=inputs[2].dimensions();
    if(q[0]!=1 || k[0]!=1 || v[0]!=1 || q[1]<=0 || k[1]<=0 ||
       k[1]!=v[1] || k[2]!=v[2] || q[2]<=0 || k[2]<=0 || q[2]%k[2] ||
       q[3]!=k[3] || v[3]<=0 || (!pages.empty() && k[1]%page_size) ||
       (pages.empty() && logical_rows>std::uint64_t(k[1])))
      return Status::InvalidArgument("CPU batch Q/K/V geometry differs");
    HYLE_ASSIGN_OR_RETURN(outputs[0],TensorDesc::dense(inputs[0].data_type(),
        std::array<std::int64_t,4>{1,q[1],q[2],v[3]}));
    return Status::Ok();
  }
};
static_assert(detail::InlineCustomOperationDescriptor<PagedRead>);
const detail::BatchProviderVTable *native_batch_table() noexcept;
std::size_t native_invocation_bytes(void *, void *) noexcept;
bool native_invocation_reads_kv(void *) noexcept;
bool native_invocation_reads_pages(void *,std::span<const std::int32_t>,
    std::uint32_t,std::uint32_t) noexcept;
bool native_invocation_can_update_q1_pages(void *) noexcept;
// Caller preflights every request and retains the new immutable layout.
void native_invocation_update_q1_pages(void *,std::span<const std::int32_t>,std::uint32_t) noexcept;
bool native_invocation_can_update_q1_ragged(void *) noexcept;
void native_invocation_update_q1_ragged(void *,std::uint32_t) noexcept;
}
