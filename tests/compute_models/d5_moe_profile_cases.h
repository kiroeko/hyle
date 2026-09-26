#pragma once

#include <array>
#include <cstddef>

#include "d5_qwen_gguf_vocabulary_cases.h"

// Fixed official/GGUF tensor observations and digest convention are in
// d5_moe_profile_facts.json. Payloads are synthetic, not checkpoint bytes.
namespace d5_moe_profile_cases {
namespace c = hyle::compute;
namespace vocabulary = d5_qwen_gguf_vocabulary_cases;

inline constexpr c::ModelArtifactProfileDescription kimi_k3_official{
    .model_publisher = "moonshotai",
    .model_repository = "moonshotai/Kimi-K3",
    .model_family = "kimi_linear",
    .model_name = "Kimi-K3",
    .model_revision = "f831ab66814297da540d832a5235f8e904f29d06",
    .configuration_digest = "9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213",
    .artifact_repository = "moonshotai/Kimi-K3",
    .artifact_revision = "f831ab66814297da540d832a5235f8e904f29d06",
    .variant = "official-mxfp4",
    .manifest_digest = "ca7a26533de68addb8d51194afb9841e1ccb9b5b8e6769b3972bb59c996b602f",
    .format_family = "safetensors",
};

inline constexpr c::ModelArtifactProfileDescription kimi_k3_primary_gguf{
    .model_publisher = "moonshotai",
    .model_repository = "moonshotai/Kimi-K3",
    .model_family = "kimi_linear",
    .model_name = "Kimi-K3",
    .model_revision = "f831ab66814297da540d832a5235f8e904f29d06",
    .configuration_digest = "9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213",
    .artifact_repository = "unsloth/Kimi-K3-GGUF",
    .artifact_revision = "a0836360ce58dfec088d966a97f2ddc8a606279b",
    .variant = "UD-Q4_K_XL",
    .manifest_digest = "54e46f96077adb0e323ca09ab0bbcb29788da246fae496c71ea9b9ec86c29c3b",
    .format_family = "gguf",
};

inline constexpr c::ModelArtifactProfileDescription kimi_k3_fidelity_gguf{
    .model_publisher = "moonshotai",
    .model_repository = "moonshotai/Kimi-K3",
    .model_family = "kimi_linear",
    .model_name = "Kimi-K3",
    .model_revision = "f831ab66814297da540d832a5235f8e904f29d06",
    .configuration_digest = "9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213",
    .artifact_repository = "unsloth/Kimi-K3-GGUF",
    .artifact_revision = "a0836360ce58dfec088d966a97f2ddc8a606279b",
    .variant = "UD-Q8_K_XL",
    .manifest_digest = "1fec4eb2b43a64faeecb57998fbf5c59f0f3b1626596e720adb5f36d7bc3e28a",
    .format_family = "gguf",
};

inline constexpr c::ModelArtifactProfileDescription deepseek_v4_pro_official{
    .model_publisher = "deepseek-ai",
    .model_repository = "deepseek-ai/DeepSeek-V4-Pro",
    .model_family = "deepseek_v4",
    .model_name = "DeepSeek-V4-Pro",
    .model_revision = "b5968e9190ef611bbf34a7229255be88a0e937c1",
    .configuration_digest = "5fe4568daee51c208cb8a79538eaeda090ae011ade1dee2c386aa95f569c810e",
    .artifact_repository = "deepseek-ai/DeepSeek-V4-Pro",
    .artifact_revision = "b5968e9190ef611bbf34a7229255be88a0e937c1",
    .variant = "official-fp4-fp8",
    .manifest_digest = "40884c4e913d4dcba82c5539092755fab780b808820eab7101a6c275dc796384",
    .format_family = "safetensors",
};

inline constexpr c::ModelArtifactProfileDescription deepseek_v4_flash_official{
    .model_publisher = "deepseek-ai",
    .model_repository = "deepseek-ai/DeepSeek-V4-Flash",
    .model_family = "deepseek_v4",
    .model_name = "DeepSeek-V4-Flash",
    .model_revision = "60d8d70770c6776ff598c94bb586a859a38244f1",
    .configuration_digest = "b628e63398a645abc711d92207f8737dd8140f7a4ef1e0a5b3616019e0ddd818",
    .artifact_repository = "deepseek-ai/DeepSeek-V4-Flash",
    .artifact_revision = "60d8d70770c6776ff598c94bb586a859a38244f1",
    .variant = "official-fp4-fp8",
    .manifest_digest = "278d715ba88c69afd98d5182045f38d9c92661375424a222322d73e7a3879630",
    .format_family = "safetensors",
};

inline constexpr c::ModelArtifactProfileDescription deepseek_v4_flash_primary_gguf{
    .model_publisher = "deepseek-ai",
    .model_repository = "deepseek-ai/DeepSeek-V4-Flash",
    .model_family = "deepseek_v4",
    .model_name = "DeepSeek-V4-Flash",
    .model_revision = "60d8d70770c6776ff598c94bb586a859a38244f1",
    .configuration_digest = "b628e63398a645abc711d92207f8737dd8140f7a4ef1e0a5b3616019e0ddd818",
    .artifact_repository = "unsloth/DeepSeek-V4-Flash-GGUF",
    .artifact_revision = "e3aa0d6a5fa4f820d9e132ac1fd1d01e1b2b49e0",
    .variant = "UD-Q4_K_XL",
    .manifest_digest = "73d1e17f912decd1ff0e3c668265166fc4f284d9cea64ed3a9c63f6b27941417",
    .format_family = "gguf",
};

inline constexpr c::ModelArtifactProfileDescription deepseek_v4_flash_fidelity_gguf{
    .model_publisher = "deepseek-ai",
    .model_repository = "deepseek-ai/DeepSeek-V4-Flash",
    .model_family = "deepseek_v4",
    .model_name = "DeepSeek-V4-Flash",
    .model_revision = "60d8d70770c6776ff598c94bb586a859a38244f1",
    .configuration_digest = "b628e63398a645abc711d92207f8737dd8140f7a4ef1e0a5b3616019e0ddd818",
    .artifact_repository = "unsloth/DeepSeek-V4-Flash-GGUF",
    .artifact_revision = "e3aa0d6a5fa4f820d9e132ac1fd1d01e1b2b49e0",
    .variant = "UD-Q8_K_XL",
    .manifest_digest = "3db0a9874cb4adad2d9525c63bfac3279ab791ca823f452fff1a588442e934af",
    .format_family = "gguf",
};

inline constexpr c::ModelArtifactProfileDescription glm53_official{
    .model_publisher = "zai-org",
    .model_repository = "zai-org/GLM-5.3",
    .model_family = "glm_moe_dsa",
    .model_name = "GLM-5.3",
    .model_revision = "aca966e4e02791568aa6a4ced368624b3d897f42",
    .configuration_digest = "3ac72612095574542f7fff847ada8e59d9199dd8af44bdf625d7e02615572e69",
    .artifact_repository = "zai-org/GLM-5.3",
    .artifact_revision = "aca966e4e02791568aa6a4ced368624b3d897f42",
    .variant = "official-fp8",
    .manifest_digest = "eed7f19a7a3eafc78bc8f2c1fb029e4af437b6b0c3a9a8ea5aee2a183e8a7cce",
    .format_family = "safetensors",
};

inline constexpr c::ModelArtifactProfileDescription glm53_primary_gguf{
    .model_publisher = "zai-org",
    .model_repository = "zai-org/GLM-5.3",
    .model_family = "glm_moe_dsa",
    .model_name = "GLM-5.3",
    .model_revision = "aca966e4e02791568aa6a4ced368624b3d897f42",
    .configuration_digest = "3ac72612095574542f7fff847ada8e59d9199dd8af44bdf625d7e02615572e69",
    .artifact_repository = "unsloth/GLM-5.3-GGUF",
    .artifact_revision = "346b3591c7f28d1a23716f97a065ecf12ec14771",
    .variant = "UD-Q4_K_XL",
    .manifest_digest = "19c10ac0fccdeacda4ba7321b85fcca494ce038072ff049dacfb9920f0c109a3",
    .format_family = "gguf",
};

inline constexpr c::ModelArtifactProfileDescription glm53_fidelity_gguf{
    .model_publisher = "zai-org",
    .model_repository = "zai-org/GLM-5.3",
    .model_family = "glm_moe_dsa",
    .model_name = "GLM-5.3",
    .model_revision = "aca966e4e02791568aa6a4ced368624b3d897f42",
    .configuration_digest = "3ac72612095574542f7fff847ada8e59d9199dd8af44bdf625d7e02615572e69",
    .artifact_repository = "unsloth/GLM-5.3-GGUF",
    .artifact_revision = "346b3591c7f28d1a23716f97a065ecf12ec14771",
    .variant = "Q8_0",
    .manifest_digest = "0a8a982a9375f3c647c118150561b1773ac023615f39db9853852f7d8b83796b",
    .format_family = "gguf",
};

inline constexpr c::ModelArtifactProfileDescription glm53_flash_official{
    .model_publisher = "zai-org",
    .model_repository = "zai-org/GLM-5.3-Flash",
    .model_family = "glm5_next_text",
    .model_name = "GLM-5.3-Flash",
    .model_revision = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
    .configuration_digest = "bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f",
    .artifact_repository = "zai-org/GLM-5.3-Flash",
    .artifact_revision = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
    .variant = "official-fp8",
    .manifest_digest = "2e4a1effe3b70cf08f456cd80db609d71d23f2cbb344d26dd39a6a292c4fa23b",
    .format_family = "safetensors",
};

inline constexpr c::ModelArtifactProfileDescription glm53_flash_primary_gguf{
    .model_publisher = "zai-org",
    .model_repository = "zai-org/GLM-5.3-Flash",
    .model_family = "glm5_next_text",
    .model_name = "GLM-5.3-Flash",
    .model_revision = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
    .configuration_digest = "bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f",
    .artifact_repository = "unsloth/GLM-5.3-Flash-GGUF",
    .artifact_revision = "621d456e93e926e4b52f85cff5f634358c1828f9",
    .variant = "UD-Q4_K_XL",
    .manifest_digest = "72cf4f09694303e01389e0e60cda591f42266b54b8422521d6ec422132ef03d3",
    .format_family = "gguf",
};

inline constexpr c::ModelArtifactProfileDescription glm53_flash_fidelity_gguf{
    .model_publisher = "zai-org",
    .model_repository = "zai-org/GLM-5.3-Flash",
    .model_family = "glm5_next_text",
    .model_name = "GLM-5.3-Flash",
    .model_revision = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
    .configuration_digest = "bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f",
    .artifact_repository = "unsloth/GLM-5.3-Flash-GGUF",
    .artifact_revision = "621d456e93e926e4b52f85cff5f634358c1828f9",
    .variant = "Q8_0",
    .manifest_digest = "bc4e88f3eb7a36105b7ffff98fe524eaae303519edba2fd844d371a9746783b2",
    .format_family = "gguf",
};

inline constexpr std::array<vocabulary::Case, 13> kVocabulary{{
    {"kimi_k3_official", &kimi_k3_official, 7168,
     vocabulary::Storage::kBFloat16, vocabulary::Storage::kBFloat16, false, 163840},
    {"kimi_k3_primary_gguf", &kimi_k3_primary_gguf, 7168,
     vocabulary::Storage::kQ8_0, vocabulary::Storage::kQ8_0, false, 163840},
    {"kimi_k3_fidelity_gguf", &kimi_k3_fidelity_gguf, 7168,
     vocabulary::Storage::kBFloat16, vocabulary::Storage::kBFloat16, false, 163840},
    {"deepseek_v4_pro_official", &deepseek_v4_pro_official, 7168,
     vocabulary::Storage::kBFloat16, vocabulary::Storage::kBFloat16, false, 129280},
    {"deepseek_v4_flash_official", &deepseek_v4_flash_official, 4096,
     vocabulary::Storage::kBFloat16, vocabulary::Storage::kBFloat16, false, 129280},
    {"deepseek_v4_flash_primary_gguf", &deepseek_v4_flash_primary_gguf, 4096,
     vocabulary::Storage::kQ8_0, vocabulary::Storage::kQ8_0, false, 129280},
    {"deepseek_v4_flash_fidelity_gguf", &deepseek_v4_flash_fidelity_gguf, 4096,
     vocabulary::Storage::kBFloat16, vocabulary::Storage::kBFloat16, false, 129280},
    {"glm53_official", &glm53_official, 6144,
     vocabulary::Storage::kBFloat16, vocabulary::Storage::kBFloat16, false, 154880},
    {"glm53_primary_gguf", &glm53_primary_gguf, 6144,
     vocabulary::Storage::kQ8_0, vocabulary::Storage::kQ8_0, false, 154880},
    {"glm53_fidelity_gguf", &glm53_fidelity_gguf, 6144,
     vocabulary::Storage::kQ8_0, vocabulary::Storage::kQ8_0, false, 154880},
    {"glm53_flash_official", &glm53_flash_official, 4096,
     vocabulary::Storage::kBFloat16, vocabulary::Storage::kBFloat16, false, 154880},
    {"glm53_flash_primary_gguf", &glm53_flash_primary_gguf, 4096,
     vocabulary::Storage::kQ8_0, vocabulary::Storage::kQ8_0, false, 154880},
    {"glm53_flash_fidelity_gguf", &glm53_flash_fidelity_gguf, 4096,
     vocabulary::Storage::kQ8_0, vocabulary::Storage::kQ8_0, false, 154880},
}};

inline constexpr auto kVocabularyPointers = [] {
  std::array<const vocabulary::Case *, kVocabulary.size()> result{};
  for (std::size_t i = 0; i < result.size(); ++i)
    result[i] = &kVocabulary[i];
  return result;
}();

} // namespace d5_moe_profile_cases
