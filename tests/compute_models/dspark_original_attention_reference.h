#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "dspark_original_shared_reference.h"

// Test-only layout adapter for the actual attention output-A GroupedMatMul.
// All Q8 decoding, native activation codecs and numerical bounds remain in the
// one shared independent reference. No expanded F32 bank or executor is added.
namespace dspark_original_attention_reference {
using ActivationCodec = dspark_original_shared_reference::ActivationCodec;
namespace detail {
struct Layout final {
  std::size_t bank_bytes = 0;
  std::vector<std::vector<std::size_t>> selected_rows;
};
inline Layout layout(std::span<const std::byte> packed, std::size_t groups,
    std::size_t outputs, std::size_t width, std::span<const float> input,
    std::span<const std::int32_t> ids) {
  if (!groups || !outputs || !width || width % 32 || ids.empty() ||
      input.size() % width || input.size() / width != ids.size() ||
      outputs > std::numeric_limits<std::size_t>::max() / (width / 32) / 34 ||
      ids.size() > std::numeric_limits<std::size_t>::max() / outputs)
    throw std::runtime_error("original attention grouped Q8 reference extent");
  Layout result; result.bank_bytes = outputs * (width / 32) * 34;
  if (groups > std::numeric_limits<std::size_t>::max() / result.bank_bytes ||
      packed.size() != groups * result.bank_bytes)
    throw std::runtime_error("original attention grouped Q8 bank extent");
  result.selected_rows.resize(groups);
  for (std::size_t row = 0; row < ids.size(); ++row) {
    if (ids[row] < 0 || std::uint64_t(ids[row]) >= groups)
      throw std::runtime_error("original attention grouped Q8 actual ID is out of range");
    result.selected_rows[std::size_t(ids[row])].push_back(row);
  }
  return result;
}
inline std::vector<float> gather(std::span<const float> source,
    std::span<const std::size_t> rows, std::size_t width) {
  std::vector<float> result(rows.size() * width);
  for (std::size_t row = 0; row < rows.size(); ++row)
    std::copy_n(source.data() + rows[row] * width, width, result.data() + row * width);
  return result;
}
} // namespace detail

// packed[G,N,K] stores G independent original Q8_0 banks. Input[M,S,K],
// ids[M,S] and output[M,S,N] are flattened in physical row/slot order. In the
// actual output-A graph S=G=8, M=5, N=1024, K=4096 and every row's IDs are0..7.
// The adapter also honors noncanonical actual IDs; it never treats the G banks
// as one dense G*N-output projection of a shared input.
inline std::vector<float> decoded_grouped_reference(std::span<const std::byte> packed,
    std::size_t groups, std::size_t outputs, std::size_t width,
    std::span<const float> input, std::span<const std::int32_t> ids) {
  const auto selected = detail::layout(packed, groups, outputs, width, input, ids);
  std::vector<float> result(ids.size() * outputs);
  for (std::size_t group = 0; group < groups; ++group) {
    const auto &rows = selected.selected_rows[group];
    if (rows.empty()) continue;
    const auto actual = detail::gather(input, rows, width);
    const auto projected = dspark_original_shared_reference::decoded_reference(
        packed.subspan(group * selected.bank_bytes, selected.bank_bytes), outputs, width, actual);
    for (std::size_t row = 0; row < rows.size(); ++row)
      std::copy_n(projected.data() + row * outputs, outputs, result.data() + rows[row] * outputs);
  }
  return result;
}

inline void expect_grouped_projection(std::span<const float> actual,
    std::span<const std::byte> packed, std::size_t groups, std::size_t outputs,
    std::size_t width, std::span<const float> actual_bfloat_input,
    std::span<const std::int32_t> actual_ids, ActivationCodec codec, std::string_view role) {
  const auto selected = detail::layout(packed, groups, outputs, width,
      actual_bfloat_input, actual_ids);
  ASSERT_EQ(actual.size(), actual_ids.size() * outputs) << role;
  for (std::size_t group = 0; group < groups; ++group) {
    const auto &rows = selected.selected_rows[group];
    if (rows.empty()) continue;
    const auto input_rows = detail::gather(actual_bfloat_input, rows, width);
    const auto output_rows = detail::gather(actual, rows, outputs);
    const auto group_name = std::string(role) + ".group" + std::to_string(group);
    dspark_original_shared_reference::expect_projection(output_rows,
        packed.subspan(group * selected.bank_bytes, selected.bank_bytes),
        outputs, width, input_rows, codec, group_name);
  }
}
} // namespace dspark_original_attention_reference
