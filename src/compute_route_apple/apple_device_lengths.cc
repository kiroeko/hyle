#include "apple_device_lengths.h"

#include "apple_checks.h"
#include "apple_runtime_internal.h"
#include "hyle/compute/apple_batch.h"
#include "hyle/compute/detail/apple_plan.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <vector>

#include <mlx/fast.h>
#include <mlx/ops.h>

namespace hyle::compute::apple::detail {
namespace {
namespace mx = mlx::core;

constexpr auto written_guard = R"METAL(
  if (key_status[0] != 0 || value_status[0] != 0) {
    const uint lane = thread_index_in_threadgroup;
    const uint group = threadgroup_position_in_grid.x;
    const ulong base = ulong(group) * ulong(DV + 2);
    for (uint d = lane; d < uint(DV + 2); d += uint(THREADS)) partial[base+d] = 0.f;
    if (lane == 0) status[group] = 3;
    return;
  }
)METAL";

// Bounded split-key online softmax. As in the original native attention,
// partials stay F32 and only the final output is converted to its declared
// dtype. Each SIMD group owns one query/head/split, including its error word.
// The scalar controls are checked before any Q, page-table or cache access.
constexpr auto control_source = R"METAL(
  const uint lane = thread_index_in_simdgroup;
  const uint group = threadgroup_position_in_grid.x;
  const uint split = group % uint(SPLITS);
  const uint head = (group / uint(SPLITS)) % uint(HEADS);
  const uint row = group / (uint(SPLITS) * uint(HEADS));
  const ulong base = ulong(group) * ulong(DV + 2);
  int request = -1, local_row = 0, active_queries = 0, history = 0;
  uint total = 0;
  bool valid = true;
  for (uint r = 0; r < uint(REQUESTS); ++r) {
    const int nq = query_lengths[r], nk = kv_lengths[r];
    const int first_query = metadata[META_Q+r], last_query = metadata[META_Q+r+1];
    const int first_page = metadata[META_KV+r], last_page = metadata[META_KV+r+1];
    const int tail = metadata[META_TAIL+r];
    if (first_query < 0 || last_query < first_query || last_query > ROWS ||
        first_page < 0 || last_page < first_page || last_page > REFS ||
        (first_page == last_page ? tail != 0 : tail < 1 || tail > PAGE_ROWS)) {
      valid = false; continue;
    }
    const int qcap = last_query-first_query;
    const long history_cap = first_page == last_page ? 0 : long(last_page-first_page-1)*PAGE_ROWS+tail;
    if (history_cap > MAX_KV) { valid = false; continue; }
    const int kcap = int(history_cap);
    if (nq < 0 || nq > qcap || nk < 0 || nk > kcap || (nq && !nk) ||
        (CAUSAL && nq > nk) ||
        uint(nq) > uint(ROWS) - min(total, uint(ROWS))) {
      valid = false;
    } else {
      if (row >= total && row - total < uint(nq)) {
        request = int(r); local_row = int(row-total);
        active_queries = nq; history = nk;
      }
      total += uint(nq);
    }
  }
)METAL";

constexpr auto invalid_partial = R"METAL(
  if (!valid || request < 0) {
    for (uint d = thread_index_in_threadgroup; d < uint(DV + 2); d += uint(THREADS)) partial[base+d] = 0.f;
    if (thread_index_in_threadgroup == 0) status[group] = valid ? 0 : 3;
    return;
  }
)METAL";

// All locked native vector implementations use one pass below 1024 keys.
// Capacity is not active history: leave short rows to the final native vector
// group even when the immutable page envelope needs several long-history splits.
constexpr auto short_partial = R"METAL(
  if (history < 1024) {
    for (uint d=lane; d<uint(DV+2); d+=32) partial[base+d]=0.f;
    if (lane==0) status[group]=0;
    return;
  }
)METAL";

constexpr auto final_guard = R"METAL(
  if (!valid || request < 0) {
    for (uint d=thread_index_in_threadgroup; d<uint(DV); d+=1024)
      output[ulong(group)*DV+d]=0;
    if (thread_index_in_threadgroup==0) status[group]=valid ? 0 : 3;
    return;
  }
  if (thread_index_in_threadgroup==0) status[group]=0;
)METAL";

constexpr auto final_write_guard = R"METAL(
  {
    const uint lane=thread_index_in_simdgroup, group=threadgroup_position_in_grid.x;
    const int failure=simd_max(lane<uint(PART_SPLITS) ? partial_status[group*PART_SPLITS+lane] : 0);
    if (failure) {
      for(uint d=thread_index_in_threadgroup; d<uint(DV); d+=1024) output[ulong(group)*DV+d]=0;
      if(thread_index_in_threadgroup==0) status[group]=failure;
      return;
    }
  }
)METAL";

constexpr auto partial_source = R"METAL(
  const int end = CAUSAL ? history-active_queries+local_row+1 : history;
  const int begin = WINDOW_ENABLED ? int(max(0L, long(history)-active_queries+local_row-WINDOW)) : 0;
  const uint extent = uint(max(0, end-begin));
  const uint chunk = (extent + uint(SPLITS)-1) / uint(SPLITS);
  const uint first = uint(begin) + min(extent, split*chunk);
  const uint last = uint(begin) + min(extent, (split+1)*chunk);
  float q[QK/32];
  for (uint d = 0; d < uint(QK/32); ++d)
    q[d] = float(queries[(ulong(row)*HEADS+head)*QK+lane+32*d]);
  float accum[DV/32];
  for (uint d = 0; d < uint(DV/32); ++d) accum[d] = 0.f;
  float maximum = -INFINITY, denominator = 0.f;
  const uint kv_head = head / uint(HEADS/KV_HEADS);
  int error = 0;
  for (uint k = first; k < last; ++k) {
    const uint entry = uint(metadata[META_KV+request]) + k/uint(PAGE_ROWS);
    if (entry >= uint(REFS)) { error = 1; break; }
    const int page = metadata[META_PAGES+entry];
    if (page < 0 || uint(page) >= uint(PAGES)) { error = 1; break; }
    const ulong physical = (ulong(page)*PAGE_ROWS + k%uint(PAGE_ROWS))*KV_HEADS+kv_head;
    float score = 0.f;
    for (uint d = 0; d < uint(QK/32); ++d)
      score += q[d]*float(keys[physical*QK+lane+32*d]);
    score = simd_sum(score)*scale;
    const float next_max = max(maximum, score);
    const float previous = exp(maximum-next_max), weight = exp(score-next_max);
    denominator = denominator*previous+weight;
    for (uint d = 0; d < uint(DV/32); ++d)
      accum[d] = accum[d]*previous+weight*float(values[physical*DV+lane+32*d]);
    maximum = next_max;
  }
  for (uint d = 0; d < uint(DV/32); ++d)
    partial[base+lane+32*d] = error ? 0.f : accum[d];
  if (lane == 0) {
    partial[base+DV] = error || !denominator ? 0.f : maximum;
    partial[base+DV+1] = error ? 0.f : denominator;
    status[group] = error;
  }
)METAL";

// Adapted from MLX sdpa_vector, Copyright (c) 2024 Apple Inc. (MIT).
// Locked source and license: third_party/mlx_cast/NOTICE.md and LICENSE.
// Preserve its contiguous per-lane dot reduction, 32-way striped keys,
// pre-scaled F32 queries and original SIMD/threadgroup softmax merge.
constexpr auto vector_source = R"METAL(
  const uint simd = simdgroup_index_in_threadgroup;
  const uint kh = head / uint(HEADS/KV_HEADS);
  const int end = CAUSAL ? history-active_queries+local_row+1 : history;
  const int begin = WINDOW_ENABLED ? int(max(0L, long(history)-active_queries+local_row-WINDOW)) : 0;
  float q[QK/32], accum[DV/32];
  for (uint d=0; d<uint(QK/32); ++d)
    q[d] = float(scale)*float(queries[(ulong(row)*HEADS+head)*QK+lane*(QK/32)+d]);
  for (uint d=0; d<uint(DV/32); ++d) accum[d]=0.f;
  float maximum=-3.4028234663852886e38f, denominator=0.f;
  int error=0;
  for (uint k=simd; k<uint(history); k+=32) {
    if (int(k)<begin || int(k)>=end) continue;
    const uint entry=uint(metadata[META_KV+request])+k/uint(PAGE_ROWS);
    if (entry>=uint(REFS)) { error=1; break; }
    const int page=metadata[META_PAGES+entry];
    if (page<0 || uint(page)>=uint(PAGES)) { error=1; break; }
    const ulong physical=(ulong(page)*PAGE_ROWS+k%uint(PAGE_ROWS))*KV_HEADS+kh;
    float score=0.f;
    for (uint d=0; d<uint(QK/32); ++d)
      score += q[d]*float(keys[physical*QK+lane*(QK/32)+d]);
    score=simd_sum(score);
    const float next=max(maximum,score);
    const float factor=fast::exp(maximum-next),weight=fast::exp(score-next);
    maximum=next; denominator=denominator*factor+weight;
    for (uint d=0; d<uint(DV/32); ++d)
      accum[d]=accum[d]*factor+weight*float(values[physical*DV+lane*(DV/32)+d]);
  }
  threadgroup float shared_outputs[32*32], maxima[32], sums[32];
  threadgroup int errors[32];
  if (lane==0) { maxima[simd]=maximum; sums[simd]=denominator; errors[simd]=error; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  maximum=maxima[lane];
  const float next=simd_max(maximum),factor=fast::exp(maximum-next);
  denominator=simd_sum(sums[lane]*factor); error=simd_max(errors[lane]);
  for (uint d=0; d<uint(DV/32); ++d) {
    shared_outputs[lane*32+simd]=accum[d];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    accum[d]=simd_sum(shared_outputs[simd*32+lane]*factor);
    accum[d]=denominator==0.f ? accum[d] : accum[d]/denominator;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane==0) {
      const float value=error ? 0.f : accum[d];
      const uint bits=as_type<uint>(value);
      output[ulong(group)*DV+simd*(DV/32)+d]=BFLOAT ?
          ushort((bits & 0x7fffffffu)>0x7f800000u ? 0x7fc0u :
              (bits+0x7fffu+((bits>>16)&1u))>>16) : as_type<ushort>(half(value));
    }
  }
  if (thread_index_in_threadgroup==0) status[group]=error;
)METAL";

constexpr auto merge_source = R"METAL(
  if (thread_index_in_threadgroup>=32) return;
  float maximum = -INFINITY;
  for (uint split = 0; split < uint(PART_SPLITS); ++split) {
    const ulong base = (ulong(group)*PART_SPLITS+split)*(DV+2);
    if (partials[base+DV+1] > 0.f) maximum = max(maximum, partials[base+DV]);
  }
  float denominator = 0.f;
  float accum[DV/32];
  for (uint d = 0; d < uint(DV/32); ++d) accum[d] = 0.f;
  for (uint split = 0; split < uint(PART_SPLITS); ++split) {
    const ulong base = (ulong(group)*PART_SPLITS+split)*(DV+2);
    const float sum = partials[base+DV+1];
    if (sum > 0.f) {
      const float weight = exp(partials[base+DV]-maximum);
      denominator += sum*weight;
      for (uint d = 0; d < uint(DV/32); ++d)
        accum[d] += partials[base+lane+32*d]*weight;
    }
  }
  for (uint d = 0; d < uint(DV/32); ++d) {
    const float value = denominator ? accum[d]/denominator : 0.f;
    const uint bits = as_type<uint>(value);
    output[ulong(group)*DV+lane+32*d] = BFLOAT ?
        ushort((bits & 0x7fffffffu) > 0x7f800000u ? 0x7fc0u :
            (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16) : as_type<ushort>(half(value));
  }
)METAL";

struct Launch final { int splits; std::uint64_t groups; };
Result<Launch> launch_geometry(const BatchSchema &schema,std::uint32_t rows) noexcept {
  const auto c=schema.capacity();
  if(!rows || rows>c.max_query_rows)
    return std::unexpected(Status::InvalidArgument("Apple device lengths envelope differs from physical requests/queries"));
  const auto splits=static_cast<int>(std::clamp((std::uint64_t(c.max_kv_rows)+255)/256,
      std::uint64_t{1},std::uint64_t{32}));
  const auto groups=std::uint64_t(rows)*c.query_heads*splits;
  if(groups>std::uint64_t(INT32_MAX)/32 || groups/splits>std::uint64_t(INT32_MAX)/1024 ||
      groups*(c.value_head_dim+2)>INT32_MAX)
    return std::unexpected(Status::OutOfRange("Apple device lengths dispatch/partial extent exceeds native indexing"));
  return Launch{splits,groups};
}
} // namespace

Status qualify_device_lengths_paged_schema(const BatchSchema &schema, float scale,
    bool causal, std::int32_t window) noexcept {
  if (!schema.valid() || !std::isfinite(scale))
    return Status::InvalidArgument("Apple device lengths require original schema and finite scale");
  const auto c = schema.capacity();
  HYLE_RETURN_IF_ERROR(validate_batch_capacity(c));
  HYLE_RETURN_IF_ERROR(validate_batch_attention_options(causal, window));
  const auto plain = CacheStorage::plain(c.query_type);
  if (schema.mla_layout() != MlaCacheLayout::kNone || c.storage != BatchStorage::kPaged ||
      c.page_size != 16 || !plain || c.key_storage != *plain ||
      c.value_storage != *plain || c.output_type != c.query_type ||
      (c.query_type != DataType::kFloat16 && c.query_type != DataType::kBFloat16) ||
      (c.qk_head_dim != 64 && c.qk_head_dim != 128 && c.qk_head_dim != 256) ||
      (c.value_head_dim != 64 && c.value_head_dim != 128 && c.value_head_dim != 256))
    return Status::Unimplemented("Apple device lengths require plain F16/BF16 GQA and bounded native head depths");
  if (schema.metadata_offsets().bytes/4 > INT32_MAX)
    return Status::OutOfRange("Apple device lengths metadata exceeds native indexing");
  return Status::Ok();
}

Result<Program> prepare_device_lengths_paged_graph(std::shared_ptr<Runtime> runtime,
    BatchSchema schema, std::uint32_t rows, float scale, bool causal,
    std::int32_t window, const WorkloadContext &workload,bool guard_writes) noexcept {
  if (!runtime || !rows)
    return std::unexpected(Status::InvalidArgument("Apple device lengths require original Runtime and physical Q"));
  HYLE_RETURN_IF_ERROR(qualify_device_lengths_paged_schema(schema,scale,causal,window));
  const auto c = schema.capacity();
  HYLE_ASSIGN_OR_RETURN(auto launch,launch_geometry(schema,rows));
  try {
    const auto offsets = schema.metadata_offsets();
    if (offsets.bytes/4 > INT32_MAX)
      return std::unexpected(Status::OutOfRange("Apple device lengths metadata exceeds native indexing"));
    const auto [splits,groups]=launch;
    std::array<TensorDesc,8> descriptions;
    HYLE_ASSIGN_OR_RETURN(descriptions[0], batch_query_description(c, rows));
    HYLE_ASSIGN_OR_RETURN(descriptions[1], batch_cache_description(schema, BatchCacheRole::kKey));
    HYLE_ASSIGN_OR_RETURN(descriptions[2], batch_cache_description(schema, BatchCacheRole::kValue));
    HYLE_ASSIGN_OR_RETURN(descriptions[3], TensorDesc::dense(DataType::kInt32, std::array<std::int64_t,1>{c.max_requests}));
    descriptions[4] = descriptions[3];
    HYLE_ASSIGN_OR_RETURN(descriptions[5], TensorDesc::dense(DataType::kInt32,
        std::array<std::int64_t,1>{static_cast<std::int64_t>(offsets.bytes/4)}));
    HYLE_ASSIGN_OR_RETURN(descriptions[6],TensorDesc::dense(DataType::kInt32,std::array<std::int64_t,1>{1}));
    descriptions[7]=descriptions[6];
    std::vector<InputSpec> specs(guard_writes?8:6);
    for (std::size_t i = 0; i < specs.size(); ++i) { HYLE_ASSIGN_OR_RETURN(specs[i], input_spec(descriptions[i])); }
    const auto function = [offsets, c, rows, splits, groups, scale, causal, window,guard_writes](
        const std::vector<mx::array> &inputs, mx::Stream stream) -> Result<std::vector<mx::array>> {
      if (inputs.size() != (guard_writes?8u:6u)) return std::unexpected(Status::InvalidArgument("Apple device lengths physical input count differs"));
      static const auto partial = mx::fast::metal_kernel("hyle_device_lengths_paged_partials",
          {"queries", "keys", "values", "query_lengths", "kv_lengths", "metadata", "scale"},
          {"partial", "status"}, std::string(control_source)+invalid_partial+short_partial+partial_source, "", true, false, mx::CompileOptions{});
      static const auto guarded_partial=mx::fast::metal_kernel("hyle_device_lengths_paged_written_partials",
          {"queries","keys","values","query_lengths","kv_lengths","metadata","key_status","value_status","scale"},
          {"partial","status"},std::string(written_guard)+control_source+invalid_partial+short_partial+partial_source,"",true,false,mx::CompileOptions{});
      static const auto finish_source=std::string(final_write_guard)+control_source+final_guard+
          "if(history<1024) {"+vector_source+"} else {"+merge_source+"}";
      static const auto finish=mx::fast::metal_kernel("hyle_device_lengths_paged_finish",
          {"queries","keys","values","query_lengths","kv_lengths","metadata","scale","partials","partial_status"},
          {"output","status"},finish_source,"",true,false,mx::CompileOptions{});
      static const auto guarded_finish=mx::fast::metal_kernel("hyle_device_lengths_paged_written_finish",
          {"queries","keys","values","query_lengths","kv_lengths","metadata","key_status","value_status","scale","partials","partial_status"},
          {"output","status"},finish_source,"",true,false,mx::CompileOptions{});
      auto arguments = inputs;
      arguments.emplace_back(scale);
      std::vector<std::pair<std::string, mx::fast::TemplateArg>> constants{
          {"QK", static_cast<int>(c.qk_head_dim)}, {"DV", static_cast<int>(c.value_head_dim)},
          {"HEADS", static_cast<int>(c.query_heads)}, {"KV_HEADS", static_cast<int>(c.kv_heads)},
          {"PAGE_ROWS", static_cast<int>(c.page_size)}, {"PAGES", static_cast<int>(c.max_pages)},
          {"REQUESTS", static_cast<int>(c.max_requests)}, {"ROWS", static_cast<int>(rows)},
          {"REFS",static_cast<int>(c.max_page_refs)}, {"MAX_KV",static_cast<int>(c.max_kv_rows)},
          {"META_Q",static_cast<int>(offsets.query_indptr/4)}, {"META_KV",static_cast<int>(offsets.kv_indptr/4)},
          {"META_PAGES",static_cast<int>(offsets.page_indices/4)}, {"META_TAIL",static_cast<int>(offsets.last_page_lengths/4)},
          {"SPLITS", splits}, {"THREADS",32}, {"CAUSAL", causal}, {"WINDOW_ENABLED", window >= 0}, {"WINDOW", std::max(window,0)}};
      const auto &selected=guard_writes?guarded_partial:partial;
      auto parts = selected(arguments, {{static_cast<int>(groups), static_cast<int>(c.value_head_dim+2)},
          {static_cast<int>(groups)}}, {mx::float32, mx::int32},
          {static_cast<int>(groups*32),1,1}, {32,1,1}, constants, std::nullopt, false, stream);
      arguments.push_back(parts[0]); arguments.push_back(parts[1]);
      for(auto &[name,value]:constants){
        if(name=="SPLITS")value=1;
        if(name=="THREADS")value=1024;
      }
      constants.emplace_back("PART_SPLITS",splits);
      constants.emplace_back("BFLOAT",inputs[0].dtype()==mx::bfloat16);
      const auto &selected_finish=guard_writes?guarded_finish:finish;
      auto finished=selected_finish(arguments,{{static_cast<int>(rows),static_cast<int>(c.query_heads),static_cast<int>(c.value_head_dim)},
          {static_cast<int>(groups/splits)}},{mx::uint16,mx::int32},
          {static_cast<int>(groups/splits*1024),1,1},{1024,1,1},constants,std::nullopt,false,stream);
      auto output=finished[0];
      // The original 16-bit allocation is reinterpreted without a payload
      // cast/copy. Integer BF16 rounding preserves subnormal output values.
      output = mx::view(output, inputs[0].dtype(), stream);
      HYLE_ASSIGN_OR_RETURN(output, with_device_status(std::move(output), mx::max(finished[1], -1, false, stream), stream));
      return std::vector{std::move(output)};
    };
    return ContextAccess::compile_device_prefix(std::move(runtime), function, specs, workload);
  } catch (const std::bad_alloc &) {
    return std::unexpected(Status::ResourceExhausted("Apple device lengths cold metadata"));
  }
}
} // namespace hyle::compute::apple::detail

namespace hyle::compute::apple {
Result<bool> supports_device_lengths_paged(const Context &context,BatchCapacity capacity,
    const ResourceBudget &budget) noexcept {
  const auto runtime=detail::ContextAccess::runtime(context);
  if(!runtime)return std::unexpected(Status::FailedPrecondition("Apple device lengths query requires its original Context"));
  {
    std::lock_guard lock(runtime->queue_mutex);
    if(!runtime->accepting)return std::unexpected(Status::FailedPrecondition("Apple device lengths Context is closed"));
    if(!runtime->failure.ok())return std::unexpected(runtime->failure);
  }
  HYLE_RETURN_IF_ERROR(validate_batch_capacity(capacity));
  HYLE_ASSIGN_OR_RETURN(auto schema,BatchSchema::create(capacity,{.resource_budget=budget}));
  const auto domain=detail::qualify_device_lengths_paged_schema(schema,1.f,true,-1);
  if(domain.IsUnimplemented())return false;
  HYLE_RETURN_IF_ERROR(domain);
  HYLE_ASSIGN_OR_RETURN(auto launch,detail::launch_geometry(schema,capacity.max_query_rows));
  (void)launch;
  return !budget.valid() || budget.allows_unattributed();
}
} // namespace hyle::compute::apple
