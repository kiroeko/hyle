#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace hyle::inference {

struct PackedRowInput final {
  std::size_t base_position = 0, rows = 0;
};
struct PackedRowFacts final {
  std::size_t active_rows = 0, max_member_rows = 0;
};

namespace packed_rows_detail {
inline bool overlaps(const void *a, std::size_t a_bytes,
                     const void *b, std::size_t b_bytes) noexcept {
  if (!a_bytes || !b_bytes) return false;
  const auto first = reinterpret_cast<std::uintptr_t>(a);
  const auto second = reinterpret_cast<std::uintptr_t>(b);
  // Subtract ordered addresses; do not form a possibly overflowing end pointer.
  return first <= second ? second - first < a_bytes : first - second < b_bytes;
}
} // namespace packed_rows_detail

// Integer geometry for one original packed carrier, not execution authority.
// All members are nonempty. The caller supplies its existing cold-reserved
// buffers with exact B+1/B/physical_rows extents, disjoint from each other and
// from inputs. No allocation, payload access, cursor advance or native work is
// performed. Rejection leaves every output byte unchanged. The returned host
// facts do not replace the original layout, binding or retirement checks.
inline PackedRowFacts pack_sequence_rows(std::span<const PackedRowInput> inputs,
    std::size_t physical_rows, std::span<std::int32_t> query_indptr,
    std::span<std::int32_t> last_ids, std::span<std::int32_t> positions) {
  constexpr auto i32_max = static_cast<std::size_t>(
      (std::numeric_limits<std::int32_t>::max)());
  if (inputs.empty() || !physical_rows)
    throw std::invalid_argument("packed rows require nonempty members and carrier");
  if (inputs.size() > i32_max || physical_rows > i32_max)
    throw std::out_of_range("packed row metadata exceeds I32 coordinates");
  if (query_indptr.size() != inputs.size() + 1 || last_ids.size() != inputs.size() ||
      positions.size() != physical_rows)
    throw std::invalid_argument("packed row buffers differ from their cold extents");
  const auto aliases = [](auto a, auto b) noexcept {
    return packed_rows_detail::overlaps(a.data(), a.size_bytes(), b.data(), b.size_bytes());
  };
  if (aliases(query_indptr, last_ids) || aliases(query_indptr, positions) ||
      aliases(last_ids, positions) || aliases(inputs, query_indptr) ||
      aliases(inputs, last_ids) || aliases(inputs, positions))
    throw std::invalid_argument("packed row input and output buffers overlap");

  PackedRowFacts facts;
  // Prove every member before the first output write, including a bad final
  // member. Subtraction precedes sums so size_t and I32 cannot wrap.
  for (const auto &input : inputs) {
    if (!input.rows)
      throw std::invalid_argument("packed row member is empty");
    if (input.rows > physical_rows - facts.active_rows || input.base_position > i32_max ||
        input.rows - 1 > i32_max - input.base_position)
      throw std::out_of_range("packed row member exceeds carrier or position coordinates");
    facts.active_rows += input.rows;
    facts.max_member_rows = (std::max)(facts.max_member_rows, input.rows);
  }

  std::size_t row = 0;
  query_indptr[0] = 0;
  for (std::size_t member = 0; member < inputs.size(); ++member) {
    const auto input = inputs[member];
    for (std::size_t offset = 0; offset < input.rows; ++offset)
      positions[row++] = static_cast<std::int32_t>(input.base_position + offset);
    query_indptr[member + 1] = static_cast<std::int32_t>(row);
    last_ids[member] = static_cast<std::int32_t>(row - 1);
  }
  std::fill(positions.begin() + static_cast<std::ptrdiff_t>(row), positions.end(), 0);
  return facts;
}

} // namespace hyle::inference
