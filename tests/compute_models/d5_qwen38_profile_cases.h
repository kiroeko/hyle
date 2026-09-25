#pragma once

#include <cstdint>

#include "d5_qwen38_24t_fixture.h"

namespace d5_qwen38_profile_cases {
namespace q = d5_qwen38_official_fixture;
namespace q24 = d5_qwen38_24t_fixture;

struct ProfileCase final {
  const hyle::compute::ModelArtifactProfileDescription *profile;
  std::int64_t hidden;
  std::int64_t ffn_hidden;
  std::int64_t query_heads;
  std::int64_t kv_heads;
  std::int64_t gdn_key_heads;
  std::int64_t gdn_value_heads;
};

inline constexpr ProfileCase kOfficial27B{
    &d5_qwen38_gdn_fixture::kOfficialProfile, q::kHidden,
    q::kFfnHidden, q::kQueryHeads, q::kKvHeads,
    d5_qwen38_gdn_fixture::kKeyHeads, d5_qwen38_gdn_fixture::kValueHeads};
inline constexpr ProfileCase kPrimaryGguf27B{
    &d5_qwen38_gdn_fixture::kPrimaryGgufProfile, q::kHidden,
    q::kFfnHidden, q::kQueryHeads, q::kKvHeads,
    d5_qwen38_gdn_fixture::kKeyHeads, d5_qwen38_gdn_fixture::kValueHeads};
inline constexpr ProfileCase kFidelityGguf27B{
    &d5_qwen38_gdn_fixture::kFidelityGgufProfile, q::kHidden,
    q::kFfnHidden, q::kQueryHeads, q::kKvHeads,
    d5_qwen38_gdn_fixture::kKeyHeads, d5_qwen38_gdn_fixture::kValueHeads};
inline constexpr ProfileCase kOfficial24T{
    &q24::kOfficialProfile, q24::kHidden, q24::kSharedExpertHidden,
    q24::kQueryHeads, q24::kKvHeads, q24::kGdnKeyHeads, q24::kGdnValueHeads};

} // namespace d5_qwen38_profile_cases
