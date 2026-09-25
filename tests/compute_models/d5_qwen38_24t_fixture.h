#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "d5_qwen38_official_fixture.h"

namespace d5_qwen38_24t_fixture {
namespace c = hyle::compute;

using d5_qwen38_official_fixture::BF16;
using d5_qwen38_official_fixture::as_bf16;
using d5_qwen38_official_fixture::coefficient;
using d5_qwen38_official_fixture::decode;
using d5_qwen38_official_fixture::dense_head;
using d5_qwen38_official_fixture::encode;
using d5_qwen38_official_fixture::rms_norm;
using d5_qwen38_official_fixture::scales;
using d5_qwen38_official_fixture::source_index;
using d5_qwen38_official_fixture::sparse_output_major_weight;
using d5_qwen38_official_fixture::sparse_projection;
using d5_qwen38_official_fixture::values;

inline constexpr std::int64_t kHidden = 8192;
inline constexpr std::int64_t kExperts = 512;
inline constexpr std::int64_t kTopK = 10;
inline constexpr std::int64_t kExpertHidden = 2048;
inline constexpr std::int64_t kSharedExpertHidden = 2048;
inline constexpr std::int64_t kQueryHeads = 64;
inline constexpr std::int64_t kKvHeads = 4;
inline constexpr std::int64_t kHeadDepth = 256;
inline constexpr std::int64_t kRotaryDimensions = 64;
inline constexpr std::int64_t kGdnKeyHeads = 16;
inline constexpr std::int64_t kGdnValueHeads = 128;
inline constexpr std::int64_t kGdnHeadDepth = 128;
inline constexpr std::int64_t kConvKernel = 4;
inline constexpr std::int64_t kMtpInput = 2 * kHidden;
inline constexpr std::int64_t kEmbeddingRows = 7;
inline constexpr std::int64_t kActiveExperts = kTopK;
// The integrated router also keeps the near-tie eleventh expert resident, so
// either side of the top-10 boundary can be dispatched without swapping a
// prepared weight bank between replays.
inline constexpr std::int64_t kRouterResidentExperts = kTopK + 1;
inline constexpr int kExpertSeedStride = 7;
// ggml's BF16 matrix traits may convert F32 dot-product inputs to BF16. Three
// consecutive expert projections therefore use the same tolerance as the
// existing exact-width BF16 FFN acceptance chain.
inline constexpr float kExpertRelativeTolerance = 0.025F;
inline constexpr float kExpertAbsoluteTolerance = 2.0e-5F;

// model.safetensors.index.json and config.json were fetched from this pinned
// revision. Their digests make this synthetic-capacity fixture an exact model
// profile even though it does not materialize the 2.4T checkpoint.
inline constexpr c::ModelArtifactProfileDescription kOfficialProfile{
    .model_publisher = "Qwen",
    .model_repository = "Qwen/Qwen3.8-2.4T-A95B",
    .model_family = "qwen3.8",
    .model_name = "Qwen3.8-2.4T-A95B",
    .model_revision = "207bd685a7e3696cfaff12ded7c6a7ea0f88c996",
    .configuration_digest =
        "4e3819548967e319ab435d044a3a331dbe3b078590ce822e9d74b79430533987",
    .artifact_repository = "Qwen/Qwen3.8-2.4T-A95B",
    .artifact_revision = "207bd685a7e3696cfaff12ded7c6a7ea0f88c996",
    .variant = "official-bf16",
    .manifest_digest =
        "e36c40d4e99b2714fff821218a0433bda2dec46afdb1ebee8ce96ced997928ee",
    .format_family = "safetensors"};

inline c::WorkloadContext workload(c::WorkloadPhase phase,
                                   std::uint64_t region) {
  return {c::ModelArtifactProfile::create(kOfficialProfile).value(), phase,
          {0x48594c454d4f444cULL, region, 1}};
}

// D5 reduces the resident expert capacity while retaining model inner widths
// and all top-k ports. The ten selected global experts are remapped to ten
// distinct local ids after the 512-way router, matching real top-k semantics.
inline std::array<std::int32_t, kTopK> active_expert_ids() {
  std::array<std::int32_t, kTopK> result{};
  for (std::size_t i = 0; i < result.size(); ++i)
    result[i] = static_cast<std::int32_t>(i);
  return result;
}

inline std::vector<BF16> expert_weight(std::size_t outputs,
                                      std::size_t inputs, int seed,
                                      std::int64_t resident_experts =
                                          kActiveExperts) {
  const auto expert_size = outputs * inputs;
  std::vector<BF16> result(resident_experts * expert_size);
  for (std::int64_t expert = 0; expert < resident_experts; ++expert) {
    const auto one_expert = sparse_output_major_weight(
        outputs, inputs, seed + static_cast<int>(expert) * kExpertSeedStride);
    std::ranges::copy(one_expert,
                      result.begin() + expert * expert_size);
  }
  return result;
}

inline std::vector<float> expert_input(int replay) {
  std::vector<float> result(kTopK * kHidden);
  for (std::int64_t slot = 0; slot < kTopK; ++slot)
    for (std::int64_t column = 0; column < kHidden; ++column) {
      const auto residue =
          (column * 17 + slot * 13 + static_cast<std::int64_t>(replay) * 7) %
          61;
      result[slot * kHidden + column] =
          static_cast<float>(residue - 30) / 64.0F;
    }
  return result;
}

inline std::vector<float> project_slots(std::span<const float> input,
                                        std::size_t input_width,
                                        std::size_t outputs, int seed,
                                        std::span<const std::int32_t> ids,
                                        std::int64_t resident_experts =
                                            kActiveExperts) {
  const auto slots = input.size() / input_width;
  if (input.size() % input_width != 0 || ids.size() != slots)
    throw std::invalid_argument("expert oracle slot shape mismatch");
  std::vector<float> result(slots * outputs);
  for (std::size_t slot = 0; slot < slots; ++slot) {
    if (ids[slot] < 0 || ids[slot] >= resident_experts)
      throw std::invalid_argument("expert oracle id outside resident bank");
    const int expert_seed = seed + ids[slot] * kExpertSeedStride;
    for (std::size_t output = 0; output < outputs; ++output) {
      const auto value = input[slot * input_width +
                               source_index(output, input_width, expert_seed)] *
                         static_cast<float>(
                             decode(coefficient(output, expert_seed)));
      result[slot * outputs + output] = value;
    }
  }
  return result;
}

inline std::vector<float> expert_oracle(std::span<const float> input,
                                        int seed,
                                        std::span<const std::int32_t> ids,
                                        std::int64_t resident_experts =
                                            kActiveExperts) {
  auto gate = project_slots(input, kHidden, kExpertHidden, seed, ids,
                            resident_experts);
  const auto up = project_slots(input, kHidden, kExpertHidden, seed + 1, ids,
                                resident_experts);
  for (std::size_t i = 0; i < gate.size(); ++i)
    gate[i] = gate[i] / (1.0F + std::exp(-gate[i])) * up[i];
  return project_slots(gate, kExpertHidden, kHidden, seed + 2, ids,
                       resident_experts);
}

inline std::vector<float> expert_oracle(std::span<const float> input,
                                        int seed) {
  const auto ids = active_expert_ids();
  return expert_oracle(input, seed, ids);
}

} // namespace d5_qwen38_24t_fixture
