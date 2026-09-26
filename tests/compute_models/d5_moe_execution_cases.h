#pragma once

#include <array>
#include <cstdint>
#include "d5_moe_profile_cases.h"

// Exact tensor observations: d5_moe_execution_facts.json.
namespace d5_moe_execution_cases {
namespace c = hyle::compute;
enum class Storage { kQ4K, kQ5K, kQ8_0, kMxfp4, kBFloat16, kQ6K };
struct Case final {
  const c::ModelArtifactProfileDescription *profile;
  std::int64_t hidden, experts, top_k, latent, intermediate, shared_intermediate;
  float scaling;
  bool sqrt_softplus, router_bf16, gguf;
  Storage gate, up, down, shared_gate, shared_up, shared_down, latent_storage;
};
inline constexpr std::array kCases{
    Case{&d5_moe_profile_cases::kimi_k3_official,
         7168, 896, 16, 3584, 3072, 6144,
         1.0F, false, true, false,
         Storage::kBFloat16, Storage::kBFloat16, Storage::kBFloat16, Storage::kBFloat16, Storage::kBFloat16, Storage::kBFloat16, Storage::kBFloat16},
    Case{&d5_moe_profile_cases::kimi_k3_primary_gguf,
         7168, 896, 16, 3584, 3072, 6144,
         1.0F, false, false, true,
         Storage::kMxfp4, Storage::kMxfp4, Storage::kMxfp4, Storage::kQ8_0, Storage::kQ8_0, Storage::kQ8_0, Storage::kQ8_0},
    Case{&d5_moe_profile_cases::kimi_k3_fidelity_gguf,
         7168, 896, 16, 3584, 3072, 6144,
         1.0F, false, false, true,
         Storage::kMxfp4, Storage::kMxfp4, Storage::kMxfp4, Storage::kBFloat16, Storage::kBFloat16, Storage::kBFloat16, Storage::kBFloat16},
    Case{&d5_moe_profile_cases::deepseek_v4_pro_official,
         7168, 384, 6, 7168, 3072, 3072,
         2.5F, true, true, false,
         Storage::kBFloat16, Storage::kBFloat16, Storage::kBFloat16, Storage::kBFloat16, Storage::kBFloat16, Storage::kBFloat16, Storage::kBFloat16},
    Case{&d5_moe_profile_cases::deepseek_v4_flash_official,
         4096, 256, 6, 4096, 2048, 2048,
         1.5F, true, true, false,
         Storage::kBFloat16, Storage::kBFloat16, Storage::kBFloat16, Storage::kBFloat16, Storage::kBFloat16, Storage::kBFloat16, Storage::kBFloat16},
    Case{&d5_moe_profile_cases::deepseek_v4_flash_primary_gguf,
         4096, 256, 6, 4096, 2048, 2048,
         1.5F, true, true, true,
         Storage::kMxfp4, Storage::kMxfp4, Storage::kMxfp4, Storage::kQ8_0, Storage::kQ8_0, Storage::kQ8_0, Storage::kBFloat16},
    Case{&d5_moe_profile_cases::deepseek_v4_flash_fidelity_gguf,
         4096, 256, 6, 4096, 2048, 2048,
         1.5F, true, true, true,
         Storage::kMxfp4, Storage::kMxfp4, Storage::kMxfp4, Storage::kBFloat16, Storage::kBFloat16, Storage::kBFloat16, Storage::kBFloat16},
    Case{&d5_moe_profile_cases::glm53_official,
         6144, 256, 8, 6144, 2048, 2048,
         2.5F, false, true, false,
         Storage::kBFloat16, Storage::kBFloat16, Storage::kBFloat16, Storage::kBFloat16, Storage::kBFloat16, Storage::kBFloat16, Storage::kBFloat16},
    Case{&d5_moe_profile_cases::glm53_primary_gguf,
         6144, 256, 8, 6144, 2048, 2048,
         2.5F, false, false, true,
         Storage::kQ4K, Storage::kQ4K, Storage::kQ5K, Storage::kQ8_0, Storage::kQ8_0, Storage::kQ8_0, Storage::kBFloat16},
    Case{&d5_moe_profile_cases::glm53_fidelity_gguf,
         6144, 256, 8, 6144, 2048, 2048,
         2.5F, false, false, true,
         Storage::kQ8_0, Storage::kQ8_0, Storage::kQ8_0, Storage::kQ8_0, Storage::kQ8_0, Storage::kQ8_0, Storage::kBFloat16},
    Case{&d5_moe_profile_cases::glm53_flash_official,
         4096, 288, 8, 4096, 2048, 2048,
         2.5F, false, true, false,
         Storage::kBFloat16, Storage::kBFloat16, Storage::kBFloat16, Storage::kBFloat16, Storage::kBFloat16, Storage::kBFloat16, Storage::kBFloat16},
    Case{&d5_moe_profile_cases::glm53_flash_primary_gguf,
         4096, 288, 8, 4096, 2048, 2048,
         2.5F, false, false, true,
         Storage::kQ4K, Storage::kQ4K, Storage::kQ5K, Storage::kQ8_0, Storage::kQ8_0, Storage::kQ8_0, Storage::kBFloat16},
    Case{&d5_moe_profile_cases::glm53_flash_fidelity_gguf,
         4096, 288, 8, 4096, 2048, 2048,
         2.5F, false, false, true,
         Storage::kQ8_0, Storage::kQ8_0, Storage::kQ8_0, Storage::kQ8_0, Storage::kQ8_0, Storage::kQ8_0, Storage::kBFloat16},
};
} // namespace d5_moe_execution_cases
