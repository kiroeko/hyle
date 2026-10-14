#include "hyle/graph/batch.h"

#include <array>
#include <cmath>

namespace hyle::graph::batch {
namespace c=compute;
Status NativePagedAttention::infer_output_descriptions(
    std::span<const c::TensorDesc> inputs,std::span<c::TensorDesc> outputs) const noexcept {
  HYLE_RETURN_IF_ERROR(c::validate_batch_capacity(capacity));
  if(capacity.storage!=c::BatchStorage::kPaged || capacity.page_size!=16 ||
     (capacity.query_type!=c::DataType::kFloat16 && capacity.query_type!=c::DataType::kBFloat16))
    return Status::Unimplemented("native page Graph schema requires F16/BF16 page16");
  if(inputs.size()!=3 || outputs.size()!=1 || active_query_rows>capacity.max_query_rows ||
     !std::isfinite(scale))
    return Status::InvalidArgument("native page Graph arity/active output/scale differs");
  HYLE_RETURN_IF_ERROR(c::validate_batch_attention_options(causal,window_left));
  HYLE_ASSIGN_OR_RETURN(auto query,c::batch_query_description(capacity));
  HYLE_ASSIGN_OR_RETURN(auto keys,c::batch_cache_description(capacity,c::BatchCacheRole::kKey));
  HYLE_ASSIGN_OR_RETURN(auto values,c::batch_cache_description(capacity,c::BatchCacheRole::kValue));
  if(inputs[0]!=query || inputs[1]!=keys || inputs[2]!=values)
    return Status::InvalidArgument("native page Graph needs truthful recipe Q/K/V descriptions");
  HYLE_ASSIGN_OR_RETURN(outputs[0],c::batch_output_description(capacity,active_query_rows));
  return Status::Ok();
}
Result<NativePagedAttention> native_paged_attention(
    const c::NativePagedBatchProgram& program,c::BatchLayoutVersion layout) noexcept {
  if(!program.valid() || !program.schema().owns(layout))
    return std::unexpected(Status::InvalidArgument("native page Graph recipe/layout identity differs"));
  HYLE_RETURN_IF_ERROR(c::validate_batch_attention_layout(layout,program.causal(),program.window_left()));
  NativePagedAttention result{.capacity=program.schema().capacity(),
      .active_query_rows=layout.facts().query_rows,.scale=program.scale(),.causal=program.causal(),.window_left=program.window_left()};
  std::array<c::TensorDesc,1> output;
  HYLE_RETURN_IF_ERROR(result.infer_output_descriptions(program.descriptions().first(3),output));
  return result;
}
} // namespace hyle::graph::batch
