#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace hyle::inference {
// Philox addresses, not mutable RNG state. A prepared candidate reserves its
// own draw range; only the original successful Snapshot swap advances it.
struct SamplingCursor final {
  std::uint64_t seed = 0, stream = 0, draw = 0;
  friend bool operator==(const SamplingCursor &, const SamplingCursor &) = default;
  std::array<std::uint32_t, 4> identity(std::uint64_t offset = 0) const {
    if (offset > (std::numeric_limits<std::uint64_t>::max)() - draw)
      throw std::runtime_error("sampling draw address overflows");
    const auto key = seed ^ stream, address = draw + offset;
    return {static_cast<std::uint32_t>(key), static_cast<std::uint32_t>(key >> 32),
        static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32)};
  }
  SamplingCursor advanced(std::uint64_t count) const {
    if (count > (std::numeric_limits<std::uint64_t>::max)() - draw)
      throw std::runtime_error("sampling draw cursor is exhausted");
    auto result = *this; result.draw += count; return result;
  }
};
} // namespace hyle::inference
