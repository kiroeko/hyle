#pragma once

#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "../compute/workload_fixture.h"

namespace d5_qwen35_fixture {
namespace c = hyle::compute;

inline constexpr std::int64_t kHidden = 2048;
inline constexpr std::size_t kActivationBytes = kHidden * 2;
inline constexpr std::size_t kWeightBytes = kHidden * kHidden * 2;
inline constexpr std::size_t kExpectedBytes = kHidden * sizeof(float);

struct Projection final {
  std::vector<std::byte> activation;
  std::vector<std::byte> weight;
  std::vector<float> expected;
};

inline std::optional<std::filesystem::path> root() {
#if defined(_WIN32)
  char *value = nullptr;
  std::size_t size = 0;
  if (_dupenv_s(&value, &size, "HYLE_D5_QWEN35_FIXTURE") != 0 ||
      value == nullptr || *value == '\0') {
    std::free(value);
    return std::nullopt;
  }
  std::filesystem::path result(value);
  std::free(value);
  return result;
#else
  const char *value = std::getenv("HYLE_D5_QWEN35_FIXTURE");
  if (value == nullptr || *value == '\0')
    return std::nullopt;
  return std::filesystem::path(value);
#endif
}

inline std::vector<std::byte> read_bytes(const std::filesystem::path &path,
                                         std::size_t expected) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream || static_cast<std::size_t>(stream.tellg()) != expected)
    throw std::runtime_error("D5 fixture has an unexpected size: " +
                             path.string());
  std::vector<std::byte> result(expected);
  stream.seekg(0);
  if (!stream.read(reinterpret_cast<char *>(result.data()),
                   static_cast<std::streamsize>(result.size())))
    throw std::runtime_error("cannot read D5 fixture: " + path.string());
  return result;
}

inline Projection load_projection(const std::filesystem::path &directory) {
  Projection result;
  result.activation =
      read_bytes(directory / "activation_1x2048.bf16", kActivationBytes);
  result.weight = read_bytes(
      directory / "tensors" /
          "model_language_model_layers_0_linear_attn_out_proj_weight.bin",
      kWeightBytes);
  const auto expected = read_bytes(directory / "expected_out_proj_1x2048.f32",
                                   kExpectedBytes);
  result.expected.resize(kHidden);
  std::memcpy(result.expected.data(), expected.data(), expected.size());
  return result;
}

inline c::WorkloadContext workload(c::WorkloadPhase phase) {
  return {c::ModelArtifactProfile::create(workload_test::kModel).value(), phase,
          {0x48594c454d4f444cULL, 3, 1}};
}

} // namespace d5_qwen35_fixture
