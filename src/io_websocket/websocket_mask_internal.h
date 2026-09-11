#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

namespace hyle::io::detail {

// RFC 6455's four-byte key repeats over the payload. Build one native 64-bit
// pattern per caller chunk, then let ordinary scalar code lower to wide loads
// and XORs on every supported target. The short tail retains the exact byte
// offset required when a frame arrives across multiple transport reads.
inline void websocket_apply_mask(std::span<std::byte> payload,
                                 const std::array<std::byte, 4> &key,
                                 std::size_t offset = 0) noexcept {
  std::array<std::byte, 8> repeated{};
  for (std::size_t i = 0; i != repeated.size(); ++i)
    repeated[i] = key[(offset + i) & 3u];

  std::uint64_t mask_word = 0;
  std::memcpy(&mask_word, repeated.data(), sizeof(mask_word));

  std::size_t position = 0;
  while (payload.size() - position >= sizeof(std::uint64_t)) {
    std::uint64_t word = 0;
    std::memcpy(&word, payload.data() + position, sizeof(word));
    word ^= mask_word;
    std::memcpy(payload.data() + position, &word, sizeof(word));
    position += sizeof(word);
  }
  for (; position != payload.size(); ++position)
    payload[position] ^= key[(offset + position) & 3u];
}

} // namespace hyle::io::detail
