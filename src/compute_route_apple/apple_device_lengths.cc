#include "apple_device_lengths.h"

#include "apple_checks.h"
#include "hyle/compute/detail/apple_plan.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <new>
#include <vector>

#include <mlx/fast.h>
#include <mlx/ops.h>

namespace hyle::compute::apple::detail {
namespace {
namespace mx = mlx::core;

// Bounded split-key online softmax. As in the original native attention,
// partials stay F32 and only the final output is converted to its declared
// dtype. Each SIMD group owns one query/head/split, including its error word.
// The scalar controls are checked before any Q, page-table or cache access.
constexpr auto partial_source = R"METAL(
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
  if (!valid || request < 0) {
    for (uint d = lane; d < uint(DV + 2); d += 32) partial[base+d] = 0.f;
    if (lane == 0) status[group] = valid ? 0 : 3;
    return;
  }
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

constexpr auto merge_source = R"METAL(
  const uint lane = thread_index_in_simdgroup;
  const uint group = threadgroup_position_in_grid.x;
  float maximum = -INFINITY;
  for (uint split = 0; split < uint(SPLITS); ++split) {
    const ulong base = (ulong(group)*SPLITS+split)*(DV+2);
    if (partial[base+DV+1] > 0.f) maximum = max(maximum, partial[base+DV]);
  }
  float denominator = 0.f;
  float accum[DV/32];
  for (uint d = 0; d < uint(DV/32); ++d) accum[d] = 0.f;
  for (uint split = 0; split < uint(SPLITS); ++split) {
    const ulong base = (ulong(group)*SPLITS+split)*(DV+2);
    const float sum = partial[base+DV+1];
    if (sum > 0.f) {
      const float weight = exp(partial[base+DV]-maximum);
      denominator += sum*weight;
      for (uint d = 0; d < uint(DV/32); ++d)
        accum[d] += partial[base+lane+32*d]*weight;
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
} // namespace

Result<Program> prepare_device_lengths_paged_graph(std::shared_ptr<Runtime> runtime,
    BatchLayoutVersion envelope, std::uint32_t rows, float scale, bool causal,
    std::int32_t window, const WorkloadContext &workload) noexcept {
  if (!runtime || !envelope.valid() || !rows || !std::isfinite(scale))
    return std::unexpected(Status::InvalidArgument("Apple device lengths require original Runtime/capacity/physical Q"));
  const auto schema = envelope.origin_schema();
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
    return std::unexpected(Status::Unimplemented("Apple device lengths require plain F16/BF16 GQA and bounded native head depths"));
  if (envelope.facts().requests != c.max_requests || envelope.facts().query_rows != rows ||
      rows > c.max_query_rows)
    return std::unexpected(Status::InvalidArgument("Apple device lengths envelope differs from physical requests/queries"));
  HYLE_RETURN_IF_ERROR(validate_batch_attention_layout(envelope, causal, window));
  try {
    const auto offsets = schema.metadata_offsets();
    if (offsets.bytes/4 > INT32_MAX)
      return std::unexpected(Status::OutOfRange("Apple device lengths metadata exceeds native indexing"));
    const auto splits = static_cast<int>(std::clamp((std::uint64_t(c.max_kv_rows)+255)/256, std::uint64_t{1}, std::uint64_t{32}));
    const auto groups = std::uint64_t(rows)*c.query_heads*splits;
    if (groups > INT32_MAX/32 || groups*(c.value_head_dim+2) > INT32_MAX)
      return std::unexpected(Status::OutOfRange("Apple device lengths dispatch/partial extent exceeds native indexing"));
    std::array<TensorDesc,6> descriptions;
    HYLE_ASSIGN_OR_RETURN(descriptions[0], batch_query_description(c, rows));
    HYLE_ASSIGN_OR_RETURN(descriptions[1], batch_cache_description(schema, BatchCacheRole::kKey));
    HYLE_ASSIGN_OR_RETURN(descriptions[2], batch_cache_description(schema, BatchCacheRole::kValue));
    HYLE_ASSIGN_OR_RETURN(descriptions[3], TensorDesc::dense(DataType::kInt32, std::array<std::int64_t,1>{c.max_requests}));
    descriptions[4] = descriptions[3];
    HYLE_ASSIGN_OR_RETURN(descriptions[5], TensorDesc::dense(DataType::kInt32,
        std::array<std::int64_t,1>{static_cast<std::int64_t>(offsets.bytes/4)}));
    std::array<InputSpec,6> specs;
    for (std::size_t i = 0; i < specs.size(); ++i) { HYLE_ASSIGN_OR_RETURN(specs[i], input_spec(descriptions[i])); }
    const auto function = [offsets, c, rows, splits, groups, scale, causal, window](
        const std::vector<mx::array> &inputs, mx::Stream stream) -> Result<std::vector<mx::array>> {
      if (inputs.size() != 6) return std::unexpected(Status::InvalidArgument("Apple device lengths physical input count differs"));
      static const auto partial = mx::fast::metal_kernel("hyle_device_lengths_paged_partials",
          {"queries", "keys", "values", "query_lengths", "kv_lengths", "metadata", "scale"},
          {"partial", "status"}, partial_source, "", true, false, mx::CompileOptions{});
      static const auto merge = mx::fast::metal_kernel("hyle_device_lengths_paged_merge",
          {"partial"}, {"output"}, merge_source, "", true, false, mx::CompileOptions{});
      auto arguments = inputs;
      arguments.emplace_back(scale);
      const std::vector<std::pair<std::string, mx::fast::TemplateArg>> constants{
          {"QK", static_cast<int>(c.qk_head_dim)}, {"DV", static_cast<int>(c.value_head_dim)},
          {"HEADS", static_cast<int>(c.query_heads)}, {"KV_HEADS", static_cast<int>(c.kv_heads)},
          {"PAGE_ROWS", static_cast<int>(c.page_size)}, {"PAGES", static_cast<int>(c.max_pages)},
          {"REQUESTS", static_cast<int>(c.max_requests)}, {"ROWS", static_cast<int>(rows)},
          {"REFS",static_cast<int>(c.max_page_refs)}, {"MAX_KV",static_cast<int>(c.max_kv_rows)},
          {"META_Q",static_cast<int>(offsets.query_indptr/4)}, {"META_KV",static_cast<int>(offsets.kv_indptr/4)},
          {"META_PAGES",static_cast<int>(offsets.page_indices/4)}, {"META_TAIL",static_cast<int>(offsets.last_page_lengths/4)},
          {"SPLITS", splits}, {"CAUSAL", causal}, {"WINDOW_ENABLED", window >= 0}, {"WINDOW", std::max(window,0)}};
      auto parts = partial(arguments, {{static_cast<int>(groups), static_cast<int>(c.value_head_dim+2)},
          {static_cast<int>(groups)}}, {mx::float32, mx::int32},
          {static_cast<int>(groups*32),1,1}, {32,1,1}, constants, std::nullopt, false, stream);
      auto output = merge({parts[0]}, {{static_cast<int>(rows), static_cast<int>(c.query_heads), static_cast<int>(c.value_head_dim)}},
          {mx::uint16}, {static_cast<int>(groups/splits*32),1,1}, {32,1,1},
          {{"DV",static_cast<int>(c.value_head_dim)}, {"SPLITS",splits}, {"BFLOAT",inputs[0].dtype()==mx::bfloat16}},
          std::nullopt, false, stream)[0];
      // The original 16-bit allocation is reinterpreted without a payload
      // cast/copy. Integer BF16 rounding preserves subnormal output values.
      output = mx::view(output, inputs[0].dtype(), stream);
      HYLE_ASSIGN_OR_RETURN(output, with_device_status(std::move(output), mx::max(parts[1], -1, false, stream), stream));
      return std::vector{std::move(output)};
    };
    return ContextAccess::compile(std::move(runtime), function, specs, workload);
  } catch (const std::bad_alloc &) {
    return std::unexpected(Status::ResourceExhausted("Apple device lengths cold metadata"));
  }
}
} // namespace hyle::compute::apple::detail
