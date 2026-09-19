#pragma once

#include "tensor_test_support.h"

namespace vulkan_gather_test {
using namespace vulkan_tensor_test;

inline std::vector<std::uint64_t> raw_pattern(const c::TensorDesc &desc) {
  const std::array<std::uint64_t, 8> special{0,
                                             0x80000000ULL,
                                             0x7fc00013ULL,
                                             0xffc12345ULL,
                                             0x000000017fa1ff03ULL,
                                             0x8000000000000001ULL,
                                             0xffff0000ffff8000ULL,
                                             0x123456789abcdef0ULL};
  const auto size = c::data_type_size(desc.data_type());
  const auto mask =
      size == 8 ? UINT64_MAX : (std::uint64_t{1} << (size * 8)) - 1;
  std::vector<std::uint64_t> values(*desc.element_count());
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] =
        desc.data_type() == c::DataType::kBool
            ? i % 2
            : (special[i % special.size()] + (i / special.size()) * 17) & mask;
  return values;
}

inline std::vector<std::uint64_t>
reference(const c::TensorDesc &source, std::span<const std::byte> host,
          std::span<const std::uint64_t> indices, std::size_t axis) {
  std::size_t outer = 1, inner = 1;
  for (std::size_t dim = 0; dim < axis; ++dim)
    outer *= static_cast<std::size_t>(source.dimensions()[dim]);
  for (auto dim = axis + 1; dim < source.rank(); ++dim)
    inner *= static_cast<std::size_t>(source.dimensions()[dim]);
  std::vector<std::uint64_t> result;
  for (std::size_t row = 0; row < outer; ++row)
    for (const auto index : indices)
      for (std::size_t column = 0; column < inner; ++column) {
        const auto logical =
            (row * static_cast<std::size_t>(source.dimensions()[axis]) +
             static_cast<std::size_t>(index)) *
                inner +
            column;
        std::uint64_t value = 0;
        std::memcpy(&value, host.data() + offset(source, logical),
                    c::data_type_size(source.data_type()));
        result.push_back(value);
      }
  return result;
}
} // namespace vulkan_gather_test
