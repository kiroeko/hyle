#pragma once

#include <array>
#include <cstdint>
#include <fstream>
#include <string>

#include <hyle/error/result.h>

namespace gguf_input {
// Consumes a mature-parser checked fixture manifest; does not parse GGUF.
struct Fixture final {
  std::string file;
  std::uint64_t capacity = 0;
  std::array<std::array<std::uint64_t, 4>, 2> offsets{};
};
inline hyle::Result<Fixture> read(const char *path) {
  std::ifstream input(path);
  std::string version;
  std::getline(input, version);
  if (version != "HYLE_F3_GGUF_FIXTURE_V1")
    return std::unexpected(hyle::Status::InvalidArgument("checked GGUF fixture version"));
  Fixture result;
  std::getline(input, result.file);
  input >> result.capacity;
  constexpr std::array<std::size_t, 4> sizes{8, 8, 128, 68};
  for (auto &lane : result.offsets) {
    for (std::size_t index = 0; index < lane.size(); ++index) {
      std::size_t bytes = 0;
      if (!(input >> lane[index] >> bytes) || bytes != sizes[index] ||
          lane[index] > result.capacity || bytes > result.capacity - lane[index])
        return std::unexpected(hyle::Status::OutOfRange("checked GGUF fixture range"));
    }
  }
  return result;
}
} // namespace gguf_input
