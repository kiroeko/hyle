#pragma once

#include "qwen_bfloat_original_roles.h"

// Fixed source metadata only. All acquisition hashes below come from the
// complete B02 manifest (SHA256 0c81c7c18e2ac7cf21308a0cd07d3e34d8b24896b2af80028e7b77b24ace1008).
// Geometry is checked again by the shared schema constructors/Data loader;
// this header adds no parser, loader, executor or converted weight bank.
namespace qwen_bfloat_original_artifacts {
namespace roles=qwen_bfloat_original_roles;
namespace q27 {
inline constexpr qwen_original_mtp_roles::Identity kIdentity{
    "Qwen/Qwen3.8-27B","1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "official-bf16","model.safetensors.index.json"};
inline constexpr qwen_full_attention_fragment::Geometry kGeometry{
    5120,24,4,256,64,1.e-6F,0.F,hyle::compute::DataType::kBFloat16};
inline constexpr std::int64_t kIntermediate=17408,kVocabulary=248320;
inline constexpr float kRopeTheta=10'000'000.F;
inline constexpr std::uint64_t kOriginalBytes=6679513088ULL;
inline constexpr std::string_view kManifestSha256=
    "0c81c7c18e2ac7cf21308a0cd07d3e34d8b24896b2af80028e7b77b24ace1008";
inline constexpr std::string_view kTargetLayer="model.language_model.layers.63";
inline constexpr std::string_view kTargetNorm="model.language_model.norm.weight";
inline constexpr roles::VocabularyNames kVocabularyNames{
    "model.language_model.embed_tokens.weight","lm_head.weight"};

// Qwen3_5RMSNorm (HF modeling_qwen3_5.py:808-822) and the fixed
// qwen3_5_mtp.py pre-FC/final norms use F32 (1 + original BF16 delta),
// followed by the actual BF16 norm output store.
inline constexpr roles::DenseNormConventions kNormConventions{
    roles::NormConvention::kDeltaPlusOne,roles::NormConvention::kDeltaPlusOne,
    roles::NormConvention::kDeltaPlusOne,roles::NormConvention::kDeltaPlusOne,
    roles::NormConvention::kDeltaPlusOne,roles::NormConvention::kDeltaPlusOne,
    roles::NormConvention::kDeltaPlusOne};

inline constexpr std::array<roles::PinnedFile,17> kNextnFiles{{
    {"mtp.pre_fc_norm_hidden.weight.bf16",
     "f2e1aa414ce84d89d016f2260923f7d47eb9827be043d1823dc6e783e4205efe"},
    {"mtp.pre_fc_norm_embedding.weight.bf16",
     "9b03e3af7824cd765abdcfc21da960c25f3c7ee1294ce25c226f0ebc5ff39998"},
    {"mtp.fc.weight.bf16",
     "4eee377b67ec2122cf214dbe6946d16261873441f1851d64409d9c7566bb20cc"},
    {"mtp.layers.0.input_layernorm.weight.bf16",
     "bf04147cf2ecff83cd7b4f8b8611df55955041bc7749c95f6c9eb741e2ce7ec0"},
    {"mtp.layers.0.self_attn.q_proj.weight.bf16",
     "8b27ef96b91e0b0564f82f11d9010efee6aa40a0eaa1900336bf597f767e7c98"},
    {"mtp.layers.0.self_attn.k_proj.weight.bf16",
     "0ab4e67930451a2f1140e865d2e3e7e23cd752a3bb25042c36e7e546361073d0"},
    {"mtp.layers.0.self_attn.v_proj.weight.bf16",
     "9606be457a48d7d9a78d72f255246f4e346845d752c1e82e41e14728255c5e13"},
    {"mtp.layers.0.self_attn.o_proj.weight.bf16",
     "9165a161837a5d4489a838279805b15dd5b5827f1d3ac2f3f4067a71131a5fd7"},
    {"mtp.layers.0.self_attn.q_norm.weight.bf16",
     "76aae220c35e2dea96f0985239b719fdcab93e208add6c32ca399b1166ec444e"},
    {"mtp.layers.0.self_attn.k_norm.weight.bf16",
     "73723b12e89816afcc8faab883f88079f0eccf6c3208b2d7d851adc2597b3c63"},
    {"mtp.layers.0.post_attention_layernorm.weight.bf16",
     "248f158f48a4b9b1d140e6728b77b19f132ae19de5cb60dac292742f7a695bbf"},
    {"mtp.layers.0.mlp.gate_proj.weight.bf16",
     "e2fa26989b3103cc595990f53c13a295f95507d0a480276e642a084e6b1f62cc"},
    {"mtp.layers.0.mlp.up_proj.weight.bf16",
     "8c6a44250d5976dc707703ea09e994eaeabef309c3faac8599e2e282eb4933ba"},
    {"mtp.layers.0.mlp.down_proj.weight.bf16",
     "2fdda9751e3b7c1444632878283116550a1824dcad691aa8366793ba21faf935"},
    {"mtp.norm.weight.bf16",
     "414fb74a971b168fb9a073c3486cbc61c46e992757179e724a345bbf56896fb1"},
    {"model.language_model.embed_tokens.weight.bf16",
     "a158f24c6ad72736067768b4255e70a384e2e931f601b12f853e98ce83ec6bbe"},
    {"lm_head.weight.bf16",
     "d922b751f014ee1139488cb94c0e164a1eb3da5c14048070ed1b784e00f92723"},
}};

inline constexpr std::array<std::string_view,12> kTargetNames{
    "model.language_model.layers.63.input_layernorm.weight",
    "model.language_model.layers.63.self_attn.q_proj.weight",
    "model.language_model.layers.63.self_attn.k_proj.weight",
    "model.language_model.layers.63.self_attn.v_proj.weight",
    "model.language_model.layers.63.self_attn.q_norm.weight",
    "model.language_model.layers.63.self_attn.k_norm.weight",
    "model.language_model.layers.63.self_attn.o_proj.weight",
    "model.language_model.layers.63.post_attention_layernorm.weight",
    "model.language_model.layers.63.mlp.gate_proj.weight",
    "model.language_model.layers.63.mlp.up_proj.weight",
    "model.language_model.layers.63.mlp.down_proj.weight",
    "model.language_model.norm.weight",
};

inline constexpr std::array<roles::PinnedFile,12> kTargetFiles{{
    {"target63.input_layernorm.weight.bf16",
     "2f14073bc5a0014672d21dab66710e0ff1830cb06d968dd64fa7ec16a35bf730"},
    {"target63.self_attn.q_proj.weight.bf16",
     "244299e7cbfe6fbec4bb845bc308142fb802c6e4542a0bbc46fdf930487c6206"},
    {"target63.self_attn.k_proj.weight.bf16",
     "80f1eff496440c0f23001bfc0e9548715172bb9c895dbc6e7b3f9f6f13e23011"},
    {"target63.self_attn.v_proj.weight.bf16",
     "0a2b4e4a3dab11844e0f597f02c1fc8720d142e1f53d3e25aeb40db5b1537f5c"},
    {"target63.self_attn.q_norm.weight.bf16",
     "a724704567558321f52e7057362a98c185857f6255bc507e3452275dc31f29f1"},
    {"target63.self_attn.k_norm.weight.bf16",
     "956090c277d9eefa1d7fcaa55ebfa03ae0c0f3eed08a7cd667d28e6e3e5160a1"},
    {"target63.self_attn.o_proj.weight.bf16",
     "cbe0c553f6301804232ce150603467f1f18069c0b57523b0389cdf5f36b36d80"},
    {"target63.post_attention_layernorm.weight.bf16",
     "19008b67d6ad56bdfeb6cab38aaa52fd182757b22912f1eeecc28345fa10dd2b"},
    {"target63.mlp.gate_proj.weight.bf16",
     "a842d895d8f6389dc2502abaac7c427227d6ad2e7b3aac08a6d79c925c43cca1"},
    {"target63.mlp.up_proj.weight.bf16",
     "c197a391cc07c401cfcd696a74b4545dff40ec8167786ee6b7ec51b709805581"},
    {"target63.mlp.down_proj.weight.bf16",
     "ac1eb35a8380925e1f4d6dd91f534a0df58f2158eac4b189407e9f460f52cffd"},
    {"target.output_norm.weight.bf16",
     "d1d9fbdef187db987c117e021c7e4574ef32a5784c29f99d2722ce7654214738"},
}};

inline std::array<roles::Spec,29> specs() {
  const auto nextn=roles::dense_specs(kGeometry,kIntermediate,kVocabulary,kNextnFiles,kVocabularyNames);
  const auto target=roles::dense_target_specs(kGeometry,kIntermediate,kTargetNames,kTargetFiles);
  std::array<roles::Spec,29> result;
  std::copy(nextn.begin(),nextn.end(),result.begin());
  std::copy(target.begin(),target.end(),result.begin()+nextn.size());
  return result;
}
} // namespace q27
} // namespace qwen_bfloat_original_artifacts
