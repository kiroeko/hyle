#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

// Pure cold layout transforms for the
// fixed original Kimi KDA0/NoPE MLA3 roles. Authentication, mapped-file owner,
// upload/ReadLease and Graph compilation remain with the existing consumer.
// These functions never generate coefficients or change their numeric bits.
namespace kimi_original_attention_layout {
inline constexpr std::size_t heads = 96, depth = 128, latent = 512;
inline constexpr std::size_t width = heads * depth, taps = 4;

inline std::vector<std::byte> qkv_convolution(
    std::span<const std::byte> query,
    std::span<const std::byte> key,
    std::span<const std::byte> value) {
  constexpr auto role_bytes = width * taps * sizeof(float);
  for (const auto role : {query, key, value})
    if (role.size() != role_bytes)
      throw std::runtime_error("original Kimi convolution role extent differs");
  std::vector<std::byte> result(3 * role_bytes);
  // Original [channel,1,tap] already has tap as the fastest dimension.
  // The append joins projected channels as Q, K, V. Do not flip the taps.
  std::memcpy(result.data(), query.data(), role_bytes);
  std::memcpy(result.data() + role_bytes, key.data(), role_bytes);
  std::memcpy(result.data() + 2 * role_bytes, value.data(), role_bytes);
  return result;
}

struct AbsorbedKv final {
  std::vector<std::byte> key;   // BF16 [96,512,128]
  std::vector<std::byte> value; // BF16 [96,128,512]
};

inline AbsorbedKv split_absorbed_kv(std::span<const std::byte> source) {
  constexpr auto item = sizeof(std::uint16_t);
  constexpr auto half_elements = heads * depth * latent;
  if (source.size() != 2 * half_elements * item)
    throw std::runtime_error("original Kimi KV-b role extent differs");
  AbsorbedKv result{std::vector<std::byte>(half_elements * item),
                    std::vector<std::byte>(half_elements * item)};
  // Original [96,256,512]: each head first has128 key rows, then128 value
  // rows. K is transposed within each head; V preserves original row order.
  for (std::size_t head = 0; head < heads; ++head) {
    for (std::size_t d = 0; d < depth; ++d)
      for (std::size_t column = 0; column < latent; ++column) {
        const auto input = (head * 2 * depth + d) * latent + column;
        const auto output = (head * latent + column) * depth + d;
        std::memcpy(result.key.data() + output * item,
                    source.data() + input * item, item);
      }
    const auto input = (head * 2 * depth + depth) * latent;
    const auto output = head * depth * latent;
    std::memcpy(result.value.data() + output * item,
                source.data() + input * item, depth * latent * item);
  }
  return result;
}
} // namespace kimi_original_attention_layout
