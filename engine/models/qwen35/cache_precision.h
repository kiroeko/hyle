#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "prepared_roles.h"
#include "hyle/compute/low_precision.h"

namespace qwen35 {
using CacheStoragePair = std::array<hyle::compute::CacheStorage, 2>;

// Original calibration is a cold storage interpretation, not another weight
// tensor or a change to the model's activation/recurrent arithmetic.
inline std::vector<CacheStoragePair> read_cache_calibration(const prepared::Roles &roles) {
  namespace c = hyle::compute;
  if (roles.cache_policy != prepared::CachePolicy::kFloat8OriginalCalibration) return {};
  const auto count = roles.geometry.attention_layers();
  if (!count || roles.cache_calibration.size() != count * 2)
    throw std::runtime_error("E4 KV needs the complete original K/V calibration pairs");
  struct Source {
    std::uintmax_t bytes = 0;
    std::ifstream input;
  };
  std::map<std::filesystem::path, Source> sources;
  // Preflight every original source extent before reading any calibration.
  for (const auto &role : roles.cache_calibration) {
    if (role.dtype != "BF16" || role.shape != std::vector<std::int64_t>{1} || role.bytes != 2 ||
        role.offset > static_cast<std::uint64_t>((std::numeric_limits<std::streamoff>::max)()))
      throw std::runtime_error("invalid original BF16 cache calibration: " + role.name);
    auto [found, inserted] = sources.try_emplace(role.source);
    auto &source = found->second;
    if (inserted) {
      std::error_code error;
      source.bytes = std::filesystem::file_size(role.source, error);
      if (error) throw std::runtime_error("cannot inspect cache calibration source: " + role.source.string());
      source.input.open(role.source, std::ios::binary);
      if (!source.input) throw std::runtime_error("cannot open cache calibration source: " + role.source.string());
    }
    if (role.offset > source.bytes || role.bytes > source.bytes - role.offset)
      throw std::runtime_error("short cache calibration source: " + role.name);
  }
  std::vector<CacheStoragePair> result(count);
  for (std::size_t index = 0; index < roles.cache_calibration.size(); ++index) {
    const auto &role = roles.cache_calibration[index];
    auto &input = sources.at(role.source).input;
    input.seekg(static_cast<std::streamoff>(role.offset));
    std::array<unsigned char, 2> bytes{};
    input.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!input) throw std::runtime_error("cannot read original cache calibration: " + role.name);
    const auto bits = std::uint32_t(bytes[0]) | (std::uint32_t(bytes[1]) << 8);
    const auto value = std::bit_cast<float>(bits << 16);
    auto storage = c::CacheStorage::float8(value);
    if (!storage) throw std::runtime_error("invalid original cache calibration " + role.name + ": " +
        std::string(storage.error().message()));
    result[index / 2][index % 2] = *storage;
  }
  return result;
}

// Row byte counts come from the real copy carrier. Geometry alone does not
// determine cache precision; scales may differ while physical extents agree.
inline std::array<std::size_t, 2> cache_storage_row_bytes(
    const ModelGeometry &geometry, const CacheStoragePair &storage) {
  namespace c = hyle::compute;
  std::array<std::size_t, 2> result{};
  for (std::size_t role = 0; role < result.size(); ++role) {
    const std::array<std::int64_t, 3> dimensions{
        1, geometry.kv_heads, role ? geometry.value_depth : geometry.key_depth};
    auto description = c::cache_carrier_description(storage[role], dimensions);
    if (!description) throw std::runtime_error("invalid cache row carrier: " + std::string(description.error().message()));
    auto bytes = description->required_bytes();
    if (!bytes) throw std::runtime_error("invalid cache row bytes: " + std::string(bytes.error().message()));
    result[role] = *bytes;
  }
  return result;
}
} // namespace qwen35
