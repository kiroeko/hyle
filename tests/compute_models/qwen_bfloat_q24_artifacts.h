#pragma once

#include "qwen_bfloat_selected_experts.h"

// Fixed source metadata generated from the complete B05 33-role manifest.
// Specs authenticate original BF16 bytes. Expert contracts describe the original
// full banks; selected planes have separate actual acquisition hashes.
// This header adds no loader, executor, converted bank or acquisition policy.
namespace qwen_bfloat_original_artifacts {
namespace q24 {
namespace roles=qwen_bfloat_original_roles;
inline constexpr qwen_original_mtp_roles::Identity kIdentity{
    "Qwen/Qwen3.8-2.4T-A95B","207bd685a7e3696cfaff12ded7c6a7ea0f88c996",
    "official-bf16","model.safetensors.index.json"};
inline constexpr qwen_full_attention_fragment::Geometry kGeometry{
    8192,64,4,256,64,1.e-6F,0.F,hyle::compute::DataType::kBFloat16};
inline constexpr std::int64_t kIntermediate=2048,kVocabulary=248320;
inline constexpr float kRopeTheta=10'000'000.F;
inline constexpr std::uint64_t kOriginalBytes=10301376512ULL;
inline constexpr std::string_view kManifestSha256="abc95f30f8c549e0e7f6bbc47e94aa1c42bc3cf221d98703efbf9ff3ee365f09";

// HF original RMS weights are BF16 deltas: F32 (1 + delta), then BF16 store.
inline constexpr roles::DenseNormConventions kNormConventions{
    roles::NormConvention::kDeltaPlusOne,roles::NormConvention::kDeltaPlusOne,
    roles::NormConvention::kDeltaPlusOne,roles::NormConvention::kDeltaPlusOne,
    roles::NormConvention::kDeltaPlusOne,roles::NormConvention::kDeltaPlusOne,
    roles::NormConvention::kDeltaPlusOne};
inline constexpr roles::VocabularyNames kVocabularyNames{
    "model.embed_tokens.weight","lm_head.weight"};
inline constexpr std::array<std::string_view,14> kTargetNames{
    "model.layers.91.input_layernorm.weight",
    "model.layers.91.self_attn.q_proj.weight",
    "model.layers.91.self_attn.k_proj.weight",
    "model.layers.91.self_attn.v_proj.weight",
    "model.layers.91.self_attn.q_norm.weight",
    "model.layers.91.self_attn.k_norm.weight",
    "model.layers.91.self_attn.o_proj.weight",
    "model.layers.91.post_attention_layernorm.weight",
    "model.layers.91.mlp.gate.weight",
    "model.layers.91.mlp.shared_expert_gate.weight",
    "model.layers.91.mlp.shared_expert.gate_proj.weight",
    "model.layers.91.mlp.shared_expert.up_proj.weight",
    "model.layers.91.mlp.shared_expert.down_proj.weight",
    "model.norm.weight",
};

inline std::array<roles::Spec,33> specs() {
  return {{
    {"model.embed_tokens.weight","model.embed_tokens.weight.bf16",
     "1a3897611305e01423af9c19b113b0e2153094039bdc951eb08a5bee0ae5a474",
     roles::sem::dense({248320,8192},hyle::compute::DataType::kBFloat16)},
    {"model.norm.weight","target.output_norm.weight.bf16",
     "e1811a67762be1480edd65b19c6b0b849a7a7191c500d3d143a87fc3dfa50253",
     roles::sem::dense({8192},hyle::compute::DataType::kBFloat16)},
    {"mtp.fc.weight","mtp.fc.weight.bf16",
     "0cc43555786f07135c86d5053da3c1ee1bf0d8b334482b96302072476fb0b8b5",
     roles::sem::dense({8192,16384},hyle::compute::DataType::kBFloat16)},
    {"mtp.layers.0.input_layernorm.weight","mtp.layers.0.input_layernorm.weight.bf16",
     "449638f6a1cc1c4826af2e6b9c69b51c4195fc3dd8248eec17ed3519a8144671",
     roles::sem::dense({8192},hyle::compute::DataType::kBFloat16)},
    {"mtp.layers.0.mlp.gate.weight","mtp.layers.0.mlp.gate.weight.bf16",
     "3a275df61a3e78985e66bd0fe74d0264faf41ec022f1468c27d9e0e14665234d",
     roles::sem::dense({512,8192},hyle::compute::DataType::kBFloat16)},
    {"mtp.layers.0.mlp.shared_expert.down_proj.weight","mtp.layers.0.mlp.shared_expert.down_proj.weight.bf16",
     "f8a596499f46f28263c27e6e20eb9298845f92d77018c7c35e887469d1d5acff",
     roles::sem::dense({8192,2048},hyle::compute::DataType::kBFloat16)},
    {"mtp.layers.0.mlp.shared_expert.gate_proj.weight","mtp.layers.0.mlp.shared_expert.gate_proj.weight.bf16",
     "4881c414d2cdab0ed06e6b2eb6994d356fe7c166237c6195c61ac94feeebd201",
     roles::sem::dense({2048,8192},hyle::compute::DataType::kBFloat16)},
    {"mtp.layers.0.mlp.shared_expert.up_proj.weight","mtp.layers.0.mlp.shared_expert.up_proj.weight.bf16",
     "f8c44a4d5561fa6bf8ea29603b56b200b3fc228ce834118cee1a7bb5683771b5",
     roles::sem::dense({2048,8192},hyle::compute::DataType::kBFloat16)},
    {"mtp.layers.0.mlp.shared_expert_gate.weight","mtp.layers.0.mlp.shared_expert_gate.weight.bf16",
     "ca01bc14e296bdddc9e73903a770234b9e905accafa6c0f953f122509df5b6b5",
     roles::sem::dense({1,8192},hyle::compute::DataType::kBFloat16)},
    {"mtp.layers.0.post_attention_layernorm.weight","mtp.layers.0.post_attention_layernorm.weight.bf16",
     "4bd83422885c423afe4f263305152a07970f78d51bc9e70ccd8b984cd1d4cb06",
     roles::sem::dense({8192},hyle::compute::DataType::kBFloat16)},
    {"mtp.layers.0.self_attn.k_norm.weight","mtp.layers.0.self_attn.k_norm.weight.bf16",
     "f2c365d2920ac55468603ed9557dd912de2fcf7a7db731c831458487796339fb",
     roles::sem::dense({256},hyle::compute::DataType::kBFloat16)},
    {"mtp.layers.0.self_attn.k_proj.weight","mtp.layers.0.self_attn.k_proj.weight.bf16",
     "46f42b6c44bee18f3d6c7a1fba2cbff198bed638e86ee2657c828141cec853ed",
     roles::sem::dense({1024,8192},hyle::compute::DataType::kBFloat16)},
    {"mtp.layers.0.self_attn.o_proj.weight","mtp.layers.0.self_attn.o_proj.weight.bf16",
     "3a629277e3d926746e53fd1755bf68dc559c9337d863d9811d7235d09bec8955",
     roles::sem::dense({8192,16384},hyle::compute::DataType::kBFloat16)},
    {"mtp.layers.0.self_attn.q_norm.weight","mtp.layers.0.self_attn.q_norm.weight.bf16",
     "d873954f43c44171029d5f356b3f00baa97e585c4fb4aab2cf7bbda460c482ba",
     roles::sem::dense({256},hyle::compute::DataType::kBFloat16)},
    {"mtp.layers.0.self_attn.q_proj.weight","mtp.layers.0.self_attn.q_proj.weight.bf16",
     "45c08c36bf8d1a801d941234306382cb90445c87bd689964043f3b202348b818",
     roles::sem::dense({32768,8192},hyle::compute::DataType::kBFloat16)},
    {"mtp.layers.0.self_attn.v_proj.weight","mtp.layers.0.self_attn.v_proj.weight.bf16",
     "8eb8804b103b7705fd67f3227943de4c26119f85892321c92af9ee9cda25ab14",
     roles::sem::dense({1024,8192},hyle::compute::DataType::kBFloat16)},
    {"mtp.norm.weight","mtp.norm.weight.bf16",
     "1ad381efa773ba24a5daea8565978cdc876b13541419c499425e0d98aa89ec69",
     roles::sem::dense({8192},hyle::compute::DataType::kBFloat16)},
    {"mtp.pre_fc_norm_embedding.weight","mtp.pre_fc_norm_embedding.weight.bf16",
     "7fe595841d2791a7e0ecc70e855360de02282c2e53e094311413007f93a56943",
     roles::sem::dense({8192},hyle::compute::DataType::kBFloat16)},
    {"mtp.pre_fc_norm_hidden.weight","mtp.pre_fc_norm_hidden.weight.bf16",
     "515580a43f4af5dbfab41a038fe66b6f2af9171cea88c6fdc9330aa09970df3f",
     roles::sem::dense({8192},hyle::compute::DataType::kBFloat16)},
    {"model.layers.91.input_layernorm.weight","target91.input_layernorm.weight.bf16",
     "ccd4d4250d0225c3ea19e50880d771defdb8601ea400f144c4472e7b10d88a6d",
     roles::sem::dense({8192},hyle::compute::DataType::kBFloat16)},
    {"model.layers.91.mlp.gate.weight","target91.mlp.gate.weight.bf16",
     "29bcbfb8cb950ec38fea08e1e0e9e4c5d9b5234315013a77842ded2827e13699",
     roles::sem::dense({512,8192},hyle::compute::DataType::kBFloat16)},
    {"model.layers.91.mlp.shared_expert.down_proj.weight","target91.mlp.shared_expert.down_proj.weight.bf16",
     "521da7f6c586ae1bff3cb9a3d575a7ec0c2ed7a62641527487eaea197cb4d8dd",
     roles::sem::dense({8192,2048},hyle::compute::DataType::kBFloat16)},
    {"model.layers.91.mlp.shared_expert.gate_proj.weight","target91.mlp.shared_expert.gate_proj.weight.bf16",
     "c99eb30a2c52d7e74784aa288f1af118ca7ec600464fe207d02b1ae2dd0feb17",
     roles::sem::dense({2048,8192},hyle::compute::DataType::kBFloat16)},
    {"model.layers.91.mlp.shared_expert.up_proj.weight","target91.mlp.shared_expert.up_proj.weight.bf16",
     "de0e9c6fa0f49ca63474ce6b12c896c375c0ee09331e6827341594e5d9967b6c",
     roles::sem::dense({2048,8192},hyle::compute::DataType::kBFloat16)},
    {"model.layers.91.mlp.shared_expert_gate.weight","target91.mlp.shared_expert_gate.weight.bf16",
     "c3a8f397c9119e8a0adeffb92daf84d48ce6a09ee67ad925cb76e3531e1b3143",
     roles::sem::dense({1,8192},hyle::compute::DataType::kBFloat16)},
    {"model.layers.91.post_attention_layernorm.weight","target91.post_attention_layernorm.weight.bf16",
     "a70540d4f89c53667df168341484a984462f0f6dbcb0924454d8a2056b85e86b",
     roles::sem::dense({8192},hyle::compute::DataType::kBFloat16)},
    {"model.layers.91.self_attn.k_norm.weight","target91.self_attn.k_norm.weight.bf16",
     "283bf47e280a32e6bc1e2e3458d221b9664c4f70764d0dd1027b98964b4173be",
     roles::sem::dense({256},hyle::compute::DataType::kBFloat16)},
    {"model.layers.91.self_attn.o_proj.weight","target91.self_attn.o_proj.weight.bf16",
     "cd10eeee125d136427da1ea2348b9294cc44b6e82bbf7640775292f8feae305a",
     roles::sem::dense({8192,16384},hyle::compute::DataType::kBFloat16)},
    {"lm_head.weight","lm_head.weight.bf16",
     "cd5ebd87061432da8ee2305045e94a0165142c80c138ebc01f1664777406c8ef",
     roles::sem::dense({248320,8192},hyle::compute::DataType::kBFloat16)},
    {"model.layers.91.self_attn.k_proj.weight","target91.self_attn.k_proj.weight.bf16",
     "cb359be33264fd24bcfa272f7cf1b975a429d390ede7032c2237d0d47acd2691",
     roles::sem::dense({1024,8192},hyle::compute::DataType::kBFloat16)},
    {"model.layers.91.self_attn.q_norm.weight","target91.self_attn.q_norm.weight.bf16",
     "29ac776a3e6fde1dd0da34c2c4261e4887680f24569c226120786af6174c95a5",
     roles::sem::dense({256},hyle::compute::DataType::kBFloat16)},
    {"model.layers.91.self_attn.q_proj.weight","target91.self_attn.q_proj.weight.bf16",
     "282ea15dc6af12d891a83d522962caa148021f4ef4298e41578eef9caf8ad85e",
     roles::sem::dense({32768,8192},hyle::compute::DataType::kBFloat16)},
    {"model.layers.91.self_attn.v_proj.weight","target91.self_attn.v_proj.weight.bf16",
     "eb359c9c4c2221833d4d1eee3ecd2e19f3ca8fe3e3a235040bfd613a14759079",
     roles::sem::dense({1024,8192},hyle::compute::DataType::kBFloat16)},
  }};
}

inline qwen_bfloat_selected_experts::Contract expert_contract(
    qwen_bfloat_selected_experts::Section section) {
  namespace selected=qwen_bfloat_selected_experts;
  switch(section) {
    case selected::Section::kNextn:return {
      "Qwen/Qwen3.8-2.4T-A95B","207bd685a7e3696cfaff12ded7c6a7ea0f88c996",
      "7fea129068439f515d38f44f93c944ccf8b526ee8a91f88d297954a43b7f8c51",section,
      {"mtp.layers.0.mlp.experts.gate_up_proj","model-00185-of-00213.safetensors",roles::sem::dense({512,4096,8192},hyle::compute::DataType::kBFloat16),152ULL,34359738520ULL},
      {"mtp.layers.0.mlp.experts.down_proj","model-00186-of-00213.safetensors",roles::sem::dense({512,8192,2048},hyle::compute::DataType::kBFloat16),152ULL,17179869336ULL}};
    case selected::Section::kTarget91:return {
      "Qwen/Qwen3.8-2.4T-A95B","207bd685a7e3696cfaff12ded7c6a7ea0f88c996",
      "7fea129068439f515d38f44f93c944ccf8b526ee8a91f88d297954a43b7f8c51",section,
      {"model.layers.91.mlp.experts.gate_up_proj","model-00209-of-00213.safetensors",roles::sem::dense({512,4096,8192},hyle::compute::DataType::kBFloat16),160ULL,34359738528ULL},
      {"model.layers.91.mlp.experts.down_proj","model-00210-of-00213.safetensors",roles::sem::dense({512,8192,2048},hyle::compute::DataType::kBFloat16),152ULL,17179869336ULL}};
    default:throw std::runtime_error("unknown original Q24 BF16 expert section");
  }
}
} // namespace q24
} // namespace qwen_bfloat_original_artifacts
