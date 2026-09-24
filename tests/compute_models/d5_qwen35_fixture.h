#pragma once

#include <array>
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
inline constexpr std::int64_t kEmbeddingRows = 7;
inline constexpr std::int64_t kMtpInput = 4096;
inline constexpr std::array<std::int32_t, kEmbeddingRows> kEmbeddingRowIds{
    0, 42, 151643, 151644, 151645, 248318, 248319};

struct Projection final {
  std::vector<std::byte> activation;
  std::vector<std::byte> weight;
  std::vector<float> expected;
};

struct Embedding final {
  std::vector<std::byte> rows;
  std::vector<std::byte> scale;
};

struct MtpProjection final {
  std::vector<std::byte> weight;
  std::array<std::vector<std::byte>, 2> activations;
  std::array<std::vector<float>, 2> expected;
};

struct AttentionNorms final {
  std::vector<std::byte> query;
  std::vector<std::byte> key;
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

inline Embedding load_embedding(const std::filesystem::path &directory) {
  Embedding result;
  result.rows = read_bytes(directory / "embedding_rows_7x2048.bf16",
                           kEmbeddingRows * kHidden * 2);
  result.scale = read_bytes(
      directory / "tensors" /
          "model_language_model_layers_0_input_layernorm_weight.bin",
      kHidden * 2);
  return result;
}

inline MtpProjection load_mtp(const std::filesystem::path &directory) {
  MtpProjection result;
  result.weight = read_bytes(directory / "tensors" / "mtp_fc_weight.bin",
                             kHidden * kMtpInput * 2);
  constexpr std::array seeds{3, 11};
  for (std::size_t i = 0; i < seeds.size(); ++i) {
    result.activations[i] = read_bytes(
        directory / ("mtp_activation_seed" + std::to_string(seeds[i]) +
                     "_1x4096.bf16"),
        kMtpInput * 2);
    const auto expected = read_bytes(
        directory / ("mtp_expected_seed" + std::to_string(seeds[i]) +
                     "_1x2048.f32"),
        kExpectedBytes);
    result.expected[i].resize(kHidden);
    std::memcpy(result.expected[i].data(), expected.data(), expected.size());
  }
  return result;
}

inline AttentionNorms
load_attention_norms(const std::filesystem::path &directory) {
  AttentionNorms result;
  result.query = read_bytes(
      directory / "tensors" /
          "model_language_model_layers_3_self_attn_q_norm_weight.bin",
      256 * 2);
  result.key = read_bytes(
      directory / "tensors" /
          "model_language_model_layers_3_self_attn_k_norm_weight.bin",
      256 * 2);
  return result;
}

inline c::WorkloadContext workload(c::WorkloadPhase phase,
                                   std::uint64_t region = 3) {
  return {c::ModelArtifactProfile::create(workload_test::kModel).value(), phase,
          {0x48594c454d4f444cULL, region, 1}};
}

} // namespace d5_qwen35_fixture
