#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace mlx_affine_test {
// Independent test writer: scatter individual bits, rather than reproduce
// the product's one/two-word extraction. Oracles use the original code array.
inline std::vector<std::uint32_t> pack(std::span<const std::uint32_t> codes,
    std::size_t rows, std::size_t depth, unsigned bits) {
  assert((bits == 4 || bits == 5 || bits == 6 || bits == 8) && depth % 32 == 0);
  assert(codes.size() == rows * depth);
  const auto words = (depth / 32) * bits;
  std::vector<std::uint32_t> result(rows * words);
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t column = 0; column < depth; ++column) {
      const auto code = codes[row * depth + column];
      assert(code < (1u << bits));
      for (unsigned bit = 0; bit < bits; ++bit) {
        const auto position = column * bits + bit;
        result[row * words + position / 32] |=
            ((code >> bit) & 1u) << (position % 32);
      }
    }
  return result;
}
} // namespace mlx_affine_test
