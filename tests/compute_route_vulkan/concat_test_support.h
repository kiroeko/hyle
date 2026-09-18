#pragma once

#include "tensor_test_support.h"

namespace vulkan_concat_test {
using namespace vulkan_tensor_test;

inline std::uint64_t raw_value(std::size_t index, std::size_t port,
                               c::DataType type) {
  const auto size = c::data_type_size(type);
  const auto mask =
      size == 8 ? UINT64_MAX : (std::uint64_t{1} << (size * 8)) - 1;
  return (0xfffa123480017d01ULL ^ (static_cast<std::uint64_t>(port + 1) << 48) ^
          (index * 0x0103070bULL + port * 0x10231ULL)) &
         mask;
}

// Literal per-row concatenation; no packed-word schedule or shader addressing.
inline std::vector<std::uint64_t>
concatenate(std::span<const std::vector<std::uint64_t>> parts,
            std::span<const std::int64_t> widths, std::size_t outer,
            std::size_t inner) {
  std::vector<std::uint64_t> result;
  for (std::size_t row = 0; row < outer; ++row)
    for (std::size_t port = 0; port < parts.size(); ++port) {
      const auto size = static_cast<std::size_t>(widths[port]) * inner;
      for (std::size_t index = 0; index < size; ++index)
        result.push_back(parts[port][row * size + index]);
    }
  return result;
}
} // namespace vulkan_concat_test
