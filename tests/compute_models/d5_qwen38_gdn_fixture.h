#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "hyle/compute/tensor_desc.h"
#include "hyle/compute/workload.h"

namespace d5_qwen38_gdn_fixture {
namespace c = hyle::compute;

inline constexpr std::int64_t kBatches = 1;
inline constexpr std::int64_t kTokens = 1;
inline constexpr std::int64_t kKeyHeads = 16;
inline constexpr std::int64_t kValueHeads = 48;
inline constexpr std::int64_t kDepth = 128;

// Fixed official artifact identity. The manifest digest is the SHA-256 of
// model.safetensors.index.json at the pinned revision.
inline constexpr c::ModelArtifactProfileDescription kOfficialProfile{
    .model_publisher = "Qwen",
    .model_repository = "Qwen/Qwen3.8-27B",
    .model_family = "qwen3.8",
    .model_name = "Qwen3.8-27B",
    .model_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    .configuration_digest =
        "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab",
    .artifact_repository = "Qwen/Qwen3.8-27B",
    .artifact_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    .variant = "official-bf16",
    .manifest_digest =
        "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df",
    .format_family = "safetensors"};

// Fixed one-shard deployment artifact. The manifest digest is a stable hash
// of the pinned GGUF header identity plus all 866 tensor name/shape/type rows
// retained by the model-weight audit; it does not claim to be the 16 GiB file
// object's content hash.
inline constexpr c::ModelArtifactProfileDescription kPrimaryGgufProfile{
    .model_publisher = "Qwen",
    .model_repository = "Qwen/Qwen3.8-27B",
    .model_family = "qwen3.8",
    .model_name = "Qwen3.8-27B",
    .model_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    .configuration_digest =
        "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab",
    .artifact_repository = "unsloth/Qwen3.8-27B-GGUF",
    .artifact_revision = "4ca720788d1e01f1bff70c033e0d0028fd02e502",
    .variant = "UD-Q4_K_M",
    .manifest_digest =
        "380ddd71a9afe228097df16cfbbc8fd33f2707fe562ea9f695f25eb40b01e774",
    .format_family = "gguf"};

inline c::WorkloadContext
workload(const c::ModelArtifactProfileDescription &profile,
         c::WorkloadPhase phase) {
  return {c::ModelArtifactProfile::create(profile).value(), phase,
          {0x48594c454d4f444cULL, 5, 1}};
}

inline c::WorkloadContext workload(c::WorkloadPhase phase) {
  return workload(kOfficialProfile, phase);
}

struct Fixture final {
  std::array<c::TensorDesc, 6> inputs;
  std::array<c::TensorDesc, 2> outputs;
  std::array<std::vector<float>, 6> data;

  explicit Fixture(int seed = 0) {
    inputs = {
        dense({kBatches, kTokens, kKeyHeads, kDepth}),
        dense({kBatches, kTokens, kKeyHeads, kDepth}),
        dense({kBatches, kTokens, kValueHeads, kDepth}),
        dense({kBatches, kTokens, kValueHeads, 1}),
        dense({kBatches, kTokens, kValueHeads, 1}),
        dense({kBatches, kValueHeads, kDepth, kDepth}),
    };
    outputs = {inputs[2], inputs[5]};
    for (std::size_t port = 0; port < data.size(); ++port) {
      data[port].resize(*inputs[port].element_count());
      for (std::size_t i = 0; i < data[port].size(); ++i) {
        const auto wave = static_cast<float>(
            std::sin(double(i + 31 * port + seed) * 0.013));
        data[port][i] =
            port == 3 ? -0.02F - 0.001F * float(i % 7)
            : port == 4 ? 0.2F + 0.01F * float(i % 5)
            : port == 5 ? wave * 0.0005F
                        : wave * 0.025F;
      }
    }
  }

  std::array<std::vector<double>, 2> oracle() const {
    std::array<std::vector<double>, 2> result{
        std::vector<double>(data[2].size()),
        std::vector<double>(data[5].begin(), data[5].end())};
    auto &state = result[1];
    for (std::int64_t value_head = 0; value_head < kValueHeads;
         ++value_head) {
      const auto key_head = value_head % kKeyHeads;
      const auto qk = static_cast<std::size_t>(key_head * kDepth);
      const auto value = static_cast<std::size_t>(value_head * kDepth);
      const auto matrix =
          static_cast<std::size_t>(value_head * kDepth * kDepth);
      const double decay = std::exp(double(data[3][value_head]));
      const double beta = data[4][value_head];
      for (std::int64_t column = 0; column < kDepth; ++column) {
        const auto row = matrix + static_cast<std::size_t>(column * kDepth);
        double projected = 0;
        for (std::int64_t key = 0; key < kDepth; ++key) {
          state[row + key] *= decay;
          projected += state[row + key] * data[1][qk + key];
        }
        const double delta =
            beta * (double(data[2][value + column]) - projected);
        double score = 0;
        for (std::int64_t key = 0; key < kDepth; ++key) {
          state[row + key] += delta * data[1][qk + key];
          score += state[row + key] * data[0][qk + key];
        }
        result[0][value + column] = score / std::sqrt(double(kDepth));
      }
    }
    return result;
  }

private:
  static c::TensorDesc dense(std::initializer_list<std::int64_t> shape) {
    return c::TensorDesc::dense(c::DataType::kFloat32,
                                {shape.begin(), shape.size()})
        .value();
  }
};

} // namespace d5_qwen38_gdn_fixture
