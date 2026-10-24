#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "../../engine/inference/artifact_digest.h"

// Test input only. The caller supplies the fixed artifact identity and logical
// element count. There is no model registry, network access or product loader.
namespace original_role_input {
using Bytes = std::vector<std::byte>;

using hyle::inference::artifact::sha256;


inline float widen(std::uint16_t value) {
  return std::bit_cast<float>(std::uint32_t{value} << 16u);
}

inline Bytes read_bytes(
    const std::filesystem::path &path, std::size_t size,
    std::string_view expected_sha256) {
  if (size > std::size_t(std::numeric_limits<std::streamoff>::max()) ||
      size > std::size_t(std::numeric_limits<std::streamsize>::max()))
    throw std::runtime_error("original role size overflow");
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file || file.tellg() != std::streamoff(size))
    throw std::runtime_error("original role has an incorrect byte length: " + path.string());
  file.seekg(0);
  Bytes payload(size);
  file.read(reinterpret_cast<char *>(payload.data()), static_cast<std::streamsize>(size));
  if (!file || sha256(payload) != expected_sha256)
    throw std::runtime_error("original role SHA256 differs: " + path.string());
  return payload;
}

inline std::vector<std::uint16_t> read_bfloat(
    const std::filesystem::path &path, std::size_t elements,
    std::string_view expected_sha256) {
  if (elements > std::size_t(std::numeric_limits<std::streamoff>::max()) / 2)
    throw std::runtime_error("original BF16 role size overflow");
  const auto payload = read_bytes(path, elements * 2, expected_sha256);
  std::vector<std::uint16_t> result(elements);
  for (std::size_t i = 0; i < elements; ++i) {
    result[i] = std::uint16_t(std::to_integer<unsigned>(payload[2 * i]) |
        (std::to_integer<unsigned>(payload[2 * i + 1]) << 8u));
    if (!std::isfinite(widen(result[i])))
      throw std::runtime_error("original BF16 role is nonfinite: " + path.string());
  }
  return result;
}

inline std::vector<float> read_float(
    const std::filesystem::path &path, std::size_t elements,
    std::string_view expected_sha256) {
  if (elements > std::size_t(std::numeric_limits<std::streamoff>::max()) / 4)
    throw std::runtime_error("original F32 role size overflow");
  const auto payload = read_bytes(path, elements * 4, expected_sha256);
  std::vector<float> result(elements);
  for (std::size_t i = 0; i < elements; ++i) {
    std::uint32_t bits = 0;
    for (std::size_t byte = 0; byte < 4; ++byte)
      bits |= std::to_integer<std::uint32_t>(payload[4 * i + byte]) << (8 * byte);
    result[i] = std::bit_cast<float>(bits);
    if (!std::isfinite(result[i]))
      throw std::runtime_error("original F32 role is nonfinite: " + path.string());
  }
  return result;
}

inline std::vector<float> widen(std::span<const std::uint16_t> values) {
  std::vector<float> result;
  result.reserve(values.size());
  for (const auto value : values) result.push_back(widen(value));
  return result;
}
} // namespace original_role_input
