#pragma once

#include <mlx/fast.h>

#include <array>
#include <bit>
#include <cstdint>

namespace hyle::compute::apple::detail {

inline constexpr int paged_vector_partitions = 64;

// Match the pinned native vector path's 64 interleaved partitions and
// reduction order. A SIMD group shares each K/V load across the three rows.
inline constexpr auto three_row_partials_source = R"METAL(
  const uint partition = threadgroup_position_in_grid.x;
  const uint head = threadgroup_position_in_grid.y;
  const uint lane = thread_index_in_simdgroup;
  const uint channel = lane * 8;
  const int length = config[0];
  const float scale = as_type<float>(config[2]);
  const uint kv_head = head / (HEADS / KV_HEADS);
  constexpr uint rows = 3, width = 256;
  float query[rows][8], accum[rows][8] = {};
  float largest[rows] = {-FLT_MAX, -FLT_MAX, -FLT_MAX}, sums[rows] = {};
  for (uint row = 0; row < rows; ++row)
    for (uint d = 0; d < 8; ++d)
      query[row][d] = scale * float(q[(row * HEADS + head) * width + channel + d]);

  for (int token = int(partition); token < length; token += PARTITIONS) {
    const uint physical = uint(pages[config[1] + token / 16]) * 16 + token % 16;
    const ulong base = (ulong(physical) * KV_HEADS + kv_head) * width + channel;
    float score[rows] = {}, value[8];
    for (uint d = 0; d < 8; ++d) {
      const float key = float(keys[base + d]);
      value[d] = float(values[base + d]);
      for (uint row = 0; row < rows; ++row) score[row] += query[row][d] * key;
    }
    for (uint row = 0; row < rows; ++row) {
      if (token > length - int(rows) + int(row)) continue;
      const float logit = simd_sum(score[row]);
      const float next = max(largest[row], logit);
      const float previous = fast::exp(largest[row] - next);
      const float weight = fast::exp(logit - next);
      largest[row] = next;
      sums[row] = sums[row] * previous + weight;
      for (uint d = 0; d < 8; ++d)
        accum[row][d] = accum[row][d] * previous + weight * value[d];
    }
  }
  for (uint row = 0; row < rows; ++row) {
    const uint slice = (row * HEADS + head) * PARTITIONS + partition;
    for (uint d = 0; d < 8; ++d) numerator[slice * width + channel + d] = accum[row][d];
    if (lane == 0) {
      stats[slice * 2] = largest[row];
      stats[slice * 2 + 1] = sums[row];
    }
  }
)METAL";

inline constexpr auto paged_vector_merge_source = R"METAL(
  const uint row = threadgroup_position_in_grid.x;
  const uint simd = simdgroup_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  constexpr uint lanes = 32, width = 256;
  const uint base = row * PARTITIONS;
  threadgroup float exchange[lanes * lanes];
  float largest = -FLT_MAX;
  for (uint part = lane; part < PARTITIONS; part += lanes)
    largest = max(largest, stats[(base + part) * 2]);
  largest = simd_max(largest);
  float total = 0.0f;
  for (uint part = lane; part < PARTITIONS; part += lanes) {
    const uint at = (base + part) * 2;
    total += fast::exp(stats[at] - largest) * stats[at + 1];
  }
  total = simd_sum(total);
  float local[8] = {};
  for (uint part = simd; part < PARTITIONS; part += lanes) {
    const uint at = base + part;
    const float weight = fast::exp(stats[at * 2] - largest);
    for (uint d = 0; d < 8; ++d)
      local[d] += weight * numerator[at * width + lane * 8 + d];
  }
  for (uint d = 0; d < 8; ++d) {
    exchange[lane * lanes + simd] = local[d];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float sum = simd_sum(exchange[simd * lanes + lane]);
    local[d] = total != 0.0f ? sum / total : sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0)
    for (uint d = 0; d < 8; ++d) output[row * width + simd * 8 + d] = bfloat16_t(local[d]);
)METAL";

inline mlx::core::array paged_three_row_attention(
    const mlx::core::array& q, const mlx::core::array& k, const mlx::core::array& v,
    const mlx::core::array& pages, int history, int first_page, float scale,
    int heads, int kv_heads, mlx::core::Stream stream) {
  namespace mx = mlx::core;
  constexpr int rows = 3, width = 256, lanes = 32, partitions = paged_vector_partitions;
  static const auto partial = mx::fast::metal_kernel("hyle_three_row_vector_partials",
      {"q", "keys", "values", "pages", "config"}, {"numerator", "stats"}, three_row_partials_source);
  static const auto merge = mx::fast::metal_kernel("hyle_three_row_vector_merge",
      {"numerator", "stats"}, {"output"}, paged_vector_merge_source);
  const std::array<std::int32_t, 3> config{history, first_page, std::bit_cast<std::int32_t>(scale)};
  auto parts = partial({q, k, v, pages, mx::array(config.data(), {3}, mx::int32)},
      {{rows * heads, partitions, width}, {rows * heads, partitions, 2}}, {mx::float32, mx::float32},
      {partitions * lanes, heads, 1}, {lanes, 1, 1},
      {{"HEADS", heads}, {"KV_HEADS", kv_heads}, {"PARTITIONS", partitions}}, std::nullopt, false, stream);
  return merge(parts, {{rows, heads, width}}, {mx::bfloat16}, {rows * heads * lanes, lanes, 1},
      {lanes, lanes, 1}, {{"PARTITIONS", partitions}}, std::nullopt, false, stream).front();
}

} // namespace hyle::compute::apple::detail
