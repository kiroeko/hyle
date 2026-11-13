#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace hyle::inference {
// Philox addresses: 64-bit seed key, 32-bit logical stream, 32-bit draw.
// The stream is a caller-selected namespace, never a physical execution slot.
// Keeping coordinates separate is injective over this declared domain; hashing
// or XOR-folding two 64-bit values into a key would lose that property.
// A candidate reserves its draw range; only a successful Snapshot swap advances it.
struct SamplingCursor final {
  static constexpr std::uint64_t maximum_coordinate = UINT32_MAX;
  std::uint64_t seed = 0, stream = 0, draw = 0;
  friend bool operator==(const SamplingCursor &, const SamplingCursor &) = default;
  std::array<std::uint32_t, 4> identity(std::uint64_t offset = 0) const {
    if (stream > maximum_coordinate || draw > maximum_coordinate || offset > maximum_coordinate - draw)
      throw std::runtime_error("sampling draw address overflows");
    return {static_cast<std::uint32_t>(seed), static_cast<std::uint32_t>(seed >> 32),
        static_cast<std::uint32_t>(draw + offset), static_cast<std::uint32_t>(stream)};
  }
  SamplingCursor advanced(std::uint64_t count) const {
    if (stream > maximum_coordinate || draw > maximum_coordinate || count > maximum_coordinate - draw)
      throw std::runtime_error("sampling draw cursor is exhausted");
    auto result = *this; result.draw += count; return result;
  }
};
} // namespace hyle::inference
