#include "hyle/graph/batch.h"

#include <array>
#include <cmath>

namespace hyle::graph::batch {
namespace c = compute;
namespace {

Status infer(const c::BatchLayoutVersion &layout, const PagedAttentionOptions &options,
    bool mla, std::span<const c::TensorDesc> inputs, std::span<c::TensorDesc> outputs) noexcept {
  if (!layout.valid() ||
      (layout.origin_schema().mla_layout() != c::MlaCacheLayout::kNone) != mla)
    return Status::InvalidArgument("paged attention needs its owned layout and matching cache role");
  const auto schema = layout.origin_schema(); // Layout-only API uses its explicit origin interpretation.
  const auto capacity = schema.capacity();
  HYLE_RETURN_IF_ERROR(c::validate_batch_capacity(capacity));
  if (capacity.storage != c::BatchStorage::kPaged || inputs.size() != (mla ? 2u : 3u) ||
      outputs.size() != 1 || !std::isfinite(options.scale))
    return Status::InvalidArgument("paged attention ports/storage/scale differ");
  HYLE_RETURN_IF_ERROR(c::validate_batch_attention_layout(layout, options.causal, options.window_left));
  const auto rows = layout.facts().query_rows;
  HYLE_ASSIGN_OR_RETURN(auto query, c::TensorDesc::dense(capacity.query_type,
      std::array<std::int64_t, 3>{rows, capacity.query_heads, capacity.qk_head_dim}));
  HYLE_ASSIGN_OR_RETURN(auto keys, c::batch_cache_description(schema,
      mla ? c::BatchCacheRole::kMla : c::BatchCacheRole::kKey));
  if (inputs[0] != query || inputs[1] != keys)
    return Status::InvalidArgument("paged attention needs active Q and truthful cache capacity");
  if (!mla) {
    HYLE_ASSIGN_OR_RETURN(auto values, c::batch_cache_description(schema, c::BatchCacheRole::kValue));
    if (inputs[2] != values)
      return Status::InvalidArgument("paged attention value capacity differs");
  }
  HYLE_ASSIGN_OR_RETURN(outputs[0], c::batch_output_description(capacity, rows));
  return Status::Ok();
}

template<class Descriptor>
Result<Descriptor> descriptor(c::BatchLayoutVersion layout, PagedAttentionOptions options,
    bool mla) noexcept {
  if (!layout.valid())
    return std::unexpected(Status::InvalidArgument("paged attention layout is empty"));
  const auto schema = layout.origin_schema(); // Layout-only API uses its explicit origin interpretation.
  const auto capacity = schema.capacity();
  Descriptor result{std::move(layout), options};
  std::array<c::TensorDesc, 3> inputs;
  HYLE_ASSIGN_OR_RETURN(inputs[0], c::TensorDesc::dense(capacity.query_type,
      std::array<std::int64_t, 3>{result.layout.facts().query_rows,
          capacity.query_heads, capacity.qk_head_dim}));
  HYLE_ASSIGN_OR_RETURN(inputs[1], c::batch_cache_description(schema,
      mla ? c::BatchCacheRole::kMla : c::BatchCacheRole::kKey));
  if (!mla) {
    HYLE_ASSIGN_OR_RETURN(inputs[2], c::batch_cache_description(schema, c::BatchCacheRole::kValue));
  }
  std::array<c::TensorDesc, 1> output;
  HYLE_RETURN_IF_ERROR(result.infer_output_descriptions(std::span(inputs).first(mla ? 2u : 3u), output));
  return result;
}
} // namespace

Status PagedAttention::infer_output_descriptions(std::span<const c::TensorDesc> inputs,
    std::span<c::TensorDesc> outputs) const noexcept {
  return infer(layout, options, false, inputs, outputs);
}
Result<PagedAttention> paged_attention(c::BatchLayoutVersion layout,
    PagedAttentionOptions options) noexcept {
  return descriptor<PagedAttention>(std::move(layout), options, false);
}
Status PagedMlaAttention::infer_output_descriptions(std::span<const c::TensorDesc> inputs,
    std::span<c::TensorDesc> outputs) const noexcept {
  return infer(layout, options, true, inputs, outputs);
}
Result<PagedMlaAttention> paged_mla_attention(c::BatchLayoutVersion layout,
    PagedAttentionOptions options) noexcept {
  return descriptor<PagedMlaAttention>(std::move(layout), options, true);
}
Status DeviceLengthsPagedAttention::infer_output_descriptions(
    std::span<const c::TensorDesc>inputs,std::span<c::TensorDesc>outputs)const noexcept{
  if(!schema.valid() || schema.mla_layout()!=c::MlaCacheLayout::kNone ||
      schema.capacity().storage!=c::BatchStorage::kPaged || inputs.size()!=5 || outputs.size()!=1 ||
      !std::isfinite(options.scale))return Status::InvalidArgument("device lengths attention geometry/ports differ");
  const auto cap=schema.capacity();
  HYLE_RETURN_IF_ERROR(c::validate_batch_capacity(cap));
  HYLE_RETURN_IF_ERROR(c::validate_batch_attention_options(options.causal,options.window_left));
  if(!physical_query_rows || physical_query_rows>cap.max_query_rows || options.allow_bf16_partials)
    return Status::InvalidArgument("device lengths attention requires bounded physical Q and exact accumulation");
  HYLE_ASSIGN_OR_RETURN(auto q,c::batch_query_description(cap,physical_query_rows));
  HYLE_ASSIGN_OR_RETURN(auto k,c::batch_cache_description(schema,c::BatchCacheRole::kKey));
  HYLE_ASSIGN_OR_RETURN(auto v,c::batch_cache_description(schema,c::BatchCacheRole::kValue));
  HYLE_ASSIGN_OR_RETURN(auto s,c::TensorDesc::dense(c::DataType::kInt32,std::array<std::int64_t,1>{cap.max_requests}));
  if(inputs[0]!=q || inputs[1]!=k || inputs[2]!=v || inputs[3]!=s || inputs[4]!=s)
    return Status::InvalidArgument("device lengths attention needs physical Q, real cache capacities and per-request I32 query/KV lengths");
  HYLE_ASSIGN_OR_RETURN(outputs[0],c::batch_output_description(cap,physical_query_rows));return Status::Ok();
}
Result<DeviceLengthsPagedAttention> device_lengths_paged_attention(c::BatchSchema schema,
    std::uint32_t rows,PagedAttentionOptions options)noexcept{
  DeviceLengthsPagedAttention result{std::move(schema),options,rows};
  if(!result.schema.valid())return std::unexpected(Status::InvalidArgument("device lengths attention schema is empty"));
  const auto cap=result.schema.capacity();std::array<c::TensorDesc,5>in;
  HYLE_ASSIGN_OR_RETURN(in[0],c::batch_query_description(cap,rows));
  HYLE_ASSIGN_OR_RETURN(in[1],c::batch_cache_description(result.schema,c::BatchCacheRole::kKey));
  HYLE_ASSIGN_OR_RETURN(in[2],c::batch_cache_description(result.schema,c::BatchCacheRole::kValue));
  HYLE_ASSIGN_OR_RETURN(in[3],c::TensorDesc::dense(c::DataType::kInt32,std::array<std::int64_t,1>{cap.max_requests}));
  in[4]=in[3];
  std::array<c::TensorDesc,1>out;HYLE_RETURN_IF_ERROR(result.infer_output_descriptions(in,out));return result;
}
} // namespace hyle::graph::batch
