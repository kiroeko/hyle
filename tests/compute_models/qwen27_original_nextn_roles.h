#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "qwen_original_mtp_roles.h"
#include "original_named_role_source.h"
#include "hyle/mem/mapping.h"

// One authenticated mapped-role source for the Q27 GGUF family. Every load
// requires an explicit artifact identity and manifest. B03 primary metadata
// below is only a named manifest constructor, never a loading fallback. The
// same source admits the S01 sidecar and plain BF16 matrix descriptions.
namespace qwen27_original_nextn_roles {
namespace original = qwen_original_mtp_roles;
namespace input = original_role_input;
namespace c = hyle::compute;
namespace m = hyle::mem;
inline constexpr std::int64_t kHidden = 5120, kVocabulary = 248320;
inline constexpr std::int64_t kHeads = 24, kKvHeads = 4, kDepth = 256, kIntermediate = 17408;
inline constexpr float kEpsilon = 1.0e-6F;
inline constexpr std::size_t kPrimaryRemainingBytes = 2066065408;
inline constexpr auto &kPrimaryIdentity = original::q27::kIdentity;
enum class Role : std::size_t {
  kKey, kKeyNorm, kAttentionNorm, kOutput, kQuery, kQueryNorm, kValue,
  kFfnDown, kFfnGate, kFfnUp, kHeadNorm, kPostNorm, kEmbedding, kHead, kCount
};
inline constexpr std::size_t kRoles = static_cast<std::size_t>(Role::kCount);
struct PrimarySpec final {
  std::string_view name, file, sha256;
  std::optional<c::GgufQuantizedType> codec;
  ggml_type native;
  std::int64_t k, n; // k0 is an original plain F32[n] norm.
  std::size_t bytes;
};
// Pins are filled only from root's completed fixed-range acquisition receipt.
// An empty pin rejects loading, never permits an unverified parameter file.
inline constexpr std::array<PrimarySpec, kRoles> kPrimarySpecs{
    PrimarySpec{"blk.64.attn_k.weight", "attn_k.weight.q8_0", "eb6d2cd6907d1026a760cabf9047b7775a2f6b2f58f19d4b2b5d1ab018e5fc56", c::GgufQuantizedType::kQ8_0, GGML_TYPE_Q8_0, 5120, 1024, 5570560},
    PrimarySpec{"blk.64.attn_k_norm.weight", "attn_k_norm.weight.f32", "13e8086a61052d510f01a7984b2ac7c5012337887e80594331ddef3a6a0f05c7", std::nullopt, GGML_TYPE_F32, 0, 256, 1024},
    PrimarySpec{"blk.64.attn_norm.weight", "attn_norm.weight.f32", "afb539011550a9d32f6cfb14db69b9aaed81e63fbd732124e020ae9903e0f275", std::nullopt, GGML_TYPE_F32, 0, 5120, 20480},
    PrimarySpec{"blk.64.attn_output.weight", "attn_output.weight.q6_k", "c3b17401794e74c1962846f897fb673f82f212f61d7f06337815a20bfeac1d42", c::GgufQuantizedType::kQ6K, GGML_TYPE_Q6_K, 6144, 5120, 25804800},
    PrimarySpec{"blk.64.attn_q.weight", "attn_q.weight.q6_k", "ebec153220a43ff360e3fcc4670f4ab6a21ee24cbfa4a54ad9dad7e671be8f10", c::GgufQuantizedType::kQ6K, GGML_TYPE_Q6_K, 5120, 12288, 51609600},
    PrimarySpec{"blk.64.attn_q_norm.weight", "attn_q_norm.weight.f32", "4a7f959094211b3dade3324736bb75e25c61a9197e025102646e7f74bd094a48", std::nullopt, GGML_TYPE_F32, 0, 256, 1024},
    PrimarySpec{"blk.64.attn_v.weight", "attn_v.weight.q8_0", "d55480291068e8ecce5c3253c52725f1733784cee7b0115ebfd4d1dd7bb9c108", c::GgufQuantizedType::kQ8_0, GGML_TYPE_Q8_0, 5120, 1024, 5570560},
    PrimarySpec{"blk.64.ffn_down.weight", "ffn_down.weight.q6_k", "a485f5e9ef13510f6d120425054dd1fceb1fcf9908ea14425b072734897bfcd7", c::GgufQuantizedType::kQ6K, GGML_TYPE_Q6_K, 17408, 5120, 73113600},
    PrimarySpec{"blk.64.ffn_gate.weight", "ffn_gate.weight.q6_k", "5d5eb6f849bb6b737f100cec801e57642e15299978534fe8cbbccdc342092536", c::GgufQuantizedType::kQ6K, GGML_TYPE_Q6_K, 5120, 17408, 73113600},
    PrimarySpec{"blk.64.ffn_up.weight", "ffn_up.weight.q6_k", "65b581fbed26058c4e002e317f8b295344e46ab0ec8519a10533e1f11e7801c7", c::GgufQuantizedType::kQ6K, GGML_TYPE_Q6_K, 5120, 17408, 73113600},
    PrimarySpec{"blk.64.nextn.shared_head_norm.weight", "shared_head_norm.weight.f32", "afdf2c4845dcce4470aeaf40e1726690b606eb642c5a4145a0e3f2598b0734d0", std::nullopt, GGML_TYPE_F32, 0, 5120, 20480},
    PrimarySpec{"blk.64.post_attention_norm.weight", "post_attention_norm.weight.f32", "fcfcdbdb492b9a1c31ca44997a188982a68e0741df67f36080c0562da6b7c6dd", std::nullopt, GGML_TYPE_F32, 0, 5120, 20480},
    PrimarySpec{"token_embd.weight", "embedding.weight.q4_k", "4265c91fb90a57e4f8fb93a433540ea9827668e0a476378bc6c0d16e4f4a7834", c::GgufQuantizedType::kQ4K, GGML_TYPE_Q4_K, 5120, 248320, 715161600},
    PrimarySpec{"output.weight", "head.weight.q6_k", "2155d6e1f7c5549077438c4760e6e150945dd5e406484b29162dff7b03d05da0", c::GgufQuantizedType::kQ6K, GGML_TYPE_Q6_K, 5120, 248320, 1042944000}
};
inline std::size_t index(Role role) {
  const auto result = static_cast<std::size_t>(role);
  if (result >= kRoles) throw std::runtime_error("unknown original Q27 NextN role");
  return result;
}
inline const PrimarySpec &primary_spec(Role role) { return kPrimarySpecs[index(role)]; }
inline c::TensorDesc primary_description(Role role) {
  const auto &s = primary_spec(role);
  if (!s.codec)
    return original::detail::take(c::TensorDesc::dense(c::DataType::kFloat32,
        std::array{s.n}), "original Q27 F32 norm description");
  auto result = original::detail::take(c::gguf_weight_description(*s.codec,
      c::DataType::kFloat32, s.k, s.n), "original Q27 packed matrix description");
  // Gather addresses the same physical original[N,K] row bank. No repacking,
  // duplicate upload, element cast or weight expansion is needed for it.
  if (role == Role::kEmbedding)
    result = original::detail::take(result.transposed(), "original Q27 embedding orientation");
  return result;
}

// Names are schema aliases, not a source of codec/SHA/identity defaults.
inline std::string_view role_name(Role role) {return primary_spec(role).name;}
inline constexpr std::array<std::string_view,3> kInputRoleNames{
    "blk.64.nextn.enorm.weight","blk.64.nextn.hnorm.weight","blk.64.nextn.eh_proj.weight"};
inline constexpr std::string_view kOutputNormName="output_norm.weight";
namespace named=original_named_role_source;
using ManifestSpec=named::ManifestSpec;
using PinnedFile=named::PinnedFile;
using Spec=named::Spec;
using named::pinned;
using named::native_codec;
using named::checked;
// Existing B03 files are reused at their original two locations. No combined
// file, payload copy, upload, JSON parser or implicit artifact selection.
inline std::array<ManifestSpec,17> primary_manifest(
    const std::filesystem::path &remaining_directory,const std::filesystem::path &input_directory) {
  std::array<ManifestSpec,17> result;
  for(std::size_t slot=0;slot<kRoles;++slot) {
    const auto role=static_cast<Role>(slot);const auto &s=primary_spec(role);
    result[slot]={std::string(s.name),remaining_directory/std::string(s.file),std::string(s.sha256),s.native,primary_description(role)};
  }
  const auto norm=original::detail::take(c::TensorDesc::dense(c::DataType::kFloat32,std::array{kHidden}),"original Q27 input norm");
  const auto eh=original::detail::take(c::gguf_weight_description(c::GgufQuantizedType::kQ6K,c::DataType::kFloat32,2*kHidden,kHidden),"original Q27 input EH");
  result[14]={std::string(kInputRoleNames[0]),input_directory/"enorm.weight.f32",original::q27::kEmbeddingNormSha,GGML_TYPE_F32,norm};
  result[15]={std::string(kInputRoleNames[1]),input_directory/"hnorm.weight.f32",original::q27::kHiddenNormSha,GGML_TYPE_F32,norm};
  result[16]={std::string(kInputRoleNames[2]),input_directory/"eh_proj.weight.q6_k",original::q27::kProjectionSha,GGML_TYPE_Q6_K,eh};
  return result;
}
inline constexpr original::Identity kSidecarIdentity{
    "unsloth/Qwen3.8-27B-GGUF","4ca720788d1e01f1bff70c033e0d0028fd02e502",
    "sidecar:MTP/mtp-Qwen3.8-27B-Q4_0.gguf","MTP/mtp-Qwen3.8-27B-Q4_0.gguf"};
// The18 files/pins must come from the completed S01 acquisition. This schema
// reuses role dimensions, not primary bytes or codecs. Empty pins fail load.
// Order: existing Role0..13, enorm, hnorm, EH, independent output_norm.
inline std::array<ManifestSpec,18> sidecar_manifest(const std::array<PinnedFile,18> &files) {
  constexpr std::array<std::optional<c::GgufQuantizedType>,kRoles> codecs{
      c::GgufQuantizedType::kQ6K,std::nullopt,std::nullopt,
      c::GgufQuantizedType::kQ6K,c::GgufQuantizedType::kQ6K,std::nullopt,
      c::GgufQuantizedType::kQ6K,c::GgufQuantizedType::kQ4K,c::GgufQuantizedType::kQ4K,
      c::GgufQuantizedType::kQ4K,std::nullopt,std::nullopt,
      c::GgufQuantizedType::kQ3K,c::GgufQuantizedType::kQ3K};
  std::array<ManifestSpec,18> result;
  for(std::size_t slot=0;slot<kRoles;++slot) {
    const auto role=static_cast<Role>(slot);const auto &s=primary_spec(role);
    auto d=primary_description(role);auto native=GGML_TYPE_F32;
    if(s.k!=0) {
      if(!codecs[slot])throw std::runtime_error("original S01 matrix schema lacks its codec");
      d=original::detail::take(c::gguf_weight_description(*codecs[slot],c::DataType::kFloat32,s.k,s.n),"original S01 packed role");
      if(role==Role::kEmbedding)d=original::detail::take(d.transposed(),"original S01 embedding rows");
      native=native_codec(*codecs[slot]);
    }
    result[slot]={std::string(s.name),files[slot].file,files[slot].sha256,native,d};
  }
  const auto norm=original::detail::take(c::TensorDesc::dense(c::DataType::kFloat32,std::array{kHidden}),"original S01 norm");
  result[14]={std::string(kInputRoleNames[0]),files[14].file,files[14].sha256,GGML_TYPE_F32,norm};
  result[15]={std::string(kInputRoleNames[1]),files[15].file,files[15].sha256,GGML_TYPE_F32,norm};
  result[16]={std::string(kInputRoleNames[2]),files[16].file,files[16].sha256,GGML_TYPE_Q4_K,
      original::detail::take(c::gguf_weight_description(c::GgufQuantizedType::kQ4K,c::DataType::kFloat32,2*kHidden,kHidden),"original S01 EH")};
  result[17]={std::string(kOutputNormName),files[17].file,files[17].sha256,GGML_TYPE_F32,norm};
  return result;
}

using Data=named::Data;
using Roles=named::Roles;
using named::bind;
// Mechanical bridge for the old three-input signature. Copies readonly lease
// permissions from this SAME binding; no file load or second bank upload.
inline original::q27::Roles input_roles(const Roles &source) {
  return {source.at(kInputRoleNames[0]),source.at(kInputRoleNames[1]),source.at(kInputRoleNames[2])};
}
} // namespace qwen27_original_nextn_roles
