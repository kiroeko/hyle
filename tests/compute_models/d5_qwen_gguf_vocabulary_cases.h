#pragma once

#include <array>
#include <cstdint>
#include <string_view>

#include "d5_qwen35_fixture.h"
#include "d5_qwen38_24t_fixture.h"
#include "d5_qwen38_gdn_fixture.h"

namespace d5_qwen_gguf_vocabulary_cases {
namespace c = hyle::compute;

enum class Storage : std::uint8_t { kFloat16, kBFloat16, kQ4K, kQ6K, kQ8_0 };

struct Case final {
  std::string_view name;
  const c::ModelArtifactProfileDescription *profile;
  std::int64_t hidden;
  Storage embedding;
  Storage output;
  bool tied_output;
  std::int64_t vocabulary_rows = 248320;
};

// The two digests hash the audited tensor table sorted by name, using compact
// JSON objects with dtype/name/shape keys. The fixed GGUF revision and exact
// variants are the model identity; these digests bind the table seen by D5.
inline constexpr c::ModelArtifactProfileDescription k24TPrimaryProfile{
    .model_publisher = "Qwen",
    .model_repository = "Qwen/Qwen3.8-2.4T-A95B",
    .model_family = "qwen3.8",
    .model_name = "Qwen3.8-2.4T-A95B",
    .model_revision = "207bd685a7e3696cfaff12ded7c6a7ea0f88c996",
    .configuration_digest =
        "4e3819548967e319ab435d044a3a331dbe3b078590ce822e9d74b79430533987",
    .artifact_repository = "unsloth/Qwen3.8-2.4T-A95B-GGUF",
    .artifact_revision = "567d3e6ac26c5474b18311e619c04350fb9a5556",
    .variant = "UD-IQ4_XS",
    .manifest_digest =
        "7fb36517338fbf54be64b1048f903e7f9805c521e58d06e4cd8c81f35f61ae1e",
    .format_family = "gguf"};

inline constexpr c::ModelArtifactProfileDescription k24TFidelityProfile{
    .model_publisher = "Qwen",
    .model_repository = "Qwen/Qwen3.8-2.4T-A95B",
    .model_family = "qwen3.8",
    .model_name = "Qwen3.8-2.4T-A95B",
    .model_revision = "207bd685a7e3696cfaff12ded7c6a7ea0f88c996",
    .configuration_digest =
        "4e3819548967e319ab435d044a3a331dbe3b078590ce822e9d74b79430533987",
    .artifact_repository = "unsloth/Qwen3.8-2.4T-A95B-GGUF",
    .artifact_revision = "567d3e6ac26c5474b18311e619c04350fb9a5556",
    .variant = "Q8_0",
    .manifest_digest =
        "04a4d0782ebdd13d8cf5a69ad88c503e1ef91377e15028f2917be8c26baf3cbe",
    .format_family = "gguf"};

inline constexpr std::array<Case, 6> kCases{{
    {"qwen35_2b_primary", &d5_qwen35_fixture::kPrimaryGgufProfile, 2048,
     Storage::kQ6K, Storage::kQ6K, true},
    {"qwen35_2b_fidelity", &d5_qwen35_fixture::kFidelityGgufProfile, 2048,
     Storage::kFloat16, Storage::kFloat16, true},
    {"qwen38_27b_primary", &d5_qwen38_gdn_fixture::kPrimaryGgufProfile,
     5120, Storage::kQ4K, Storage::kQ6K, false},
    {"qwen38_27b_fidelity", &d5_qwen38_gdn_fixture::kFidelityGgufProfile,
     5120, Storage::kQ8_0, Storage::kBFloat16, false},
    {"qwen38_24t_primary", &k24TPrimaryProfile, 8192, Storage::kQ4K,
     Storage::kQ4K, false},
    {"qwen38_24t_fidelity", &k24TFidelityProfile, 8192, Storage::kQ8_0,
     Storage::kQ8_0, false},
}};

} // namespace d5_qwen_gguf_vocabulary_cases
