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
#include <utility>
#include <vector>

#include "qwen_original_mtp_roles.h"
#include "hyle/mem/mapping.h"

// MAIN Q24 UD-IQ4_XS source, not an IQ1/fidelity-profile surrogate. Three
// complete512-expert banks remain mapped in original Q4_K bytes. Only actual
// selected experts are supplied to the explicit E10 Graph inputs. The other
//16 new roles are prepared once; the prior original K/V/K-norm are reused.
namespace qwen24_original_nextn_roles {
namespace original = qwen_original_mtp_roles;
namespace input = original_role_input;
namespace c = hyle::compute;
namespace m = hyle::mem;
inline constexpr auto &kIdentity = original::q24::kIdentity;
inline constexpr std::int64_t kHidden = 8192, kIntermediate = 2048, kVocabulary = 248320;
inline constexpr std::int32_t kExpertDomain = 512;
inline constexpr std::size_t kResidentCapacity = 10;
inline constexpr std::size_t kOriginalBytes = 17257268224ULL;
inline constexpr std::size_t kStaticBytes = 2761753600ULL;
enum class Leg : std::size_t { kGate, kUp, kDown };
enum class Role : std::size_t {
  kAttentionNorm, kOutput, kQuery, kQueryNorm, kExpertDown, kSharedDown,
  kExpertGate, kRouter, kSharedRouter, kSharedGate, kExpertUp, kSharedUp,
  kProjection, kEmbeddingNorm, kHiddenNorm, kHeadNorm, kPostNorm,
  kEmbedding, kHead, kCount
};
inline constexpr std::size_t kRoles = static_cast<std::size_t>(Role::kCount);
struct Spec final {
  std::string_view name, file, sha256;
  std::optional<c::GgufQuantizedType> codec;
  ggml_type native;
  std::int64_t k, n, e; // k0 is a plain F32[n] norm; e512 is a mapped source bank.
  std::size_t bytes;
};
// Only real completed acquisition hashes may fill these pins. Empty rejects
// loading instead of accepting unverified or generated data.
inline constexpr std::array<Spec, kRoles> kSpecs{
    Spec{"blk.92.attn_norm.weight", "attn_norm.weight.f32", "cf0b140e2bf1e2b879c3b4e501c979b9be42e7f872cbe1b23502638c31e659a7", std::nullopt, GGML_TYPE_F32, 0, 8192, 1, 32768},
    Spec{"blk.92.attn_output.weight", "attn_output.weight.q5_k", "9908c6f3f41328ed44ee37c15779bf2c18d9df793c8fda90a50a7ab213f3fdab", c::GgufQuantizedType::kQ5K, GGML_TYPE_Q5_K, 16384, 8192, 1, 92274688},
    Spec{"blk.92.attn_q.weight", "attn_q.weight.q5_k", "18ebc4065274caebad5335050253120a17391e5c2fa53b2c4c489d0ca1c855ee", c::GgufQuantizedType::kQ5K, GGML_TYPE_Q5_K, 8192, 32768, 1, 184549376},
    Spec{"blk.92.attn_q_norm.weight", "attn_q_norm.weight.f32", "63f5b65322b3810ee1c0517f34128331d30adce9860efab873380558a4a54b8d", std::nullopt, GGML_TYPE_F32, 0, 256, 1, 1024},
    Spec{"blk.92.ffn_down_exps.weight", "ffn_down_exps.weight.q4_k", "7854e87ee0d32bf52ca9d03235a050c9ae83d39faecd169dba9925e22fea73c2", c::GgufQuantizedType::kQ4K, GGML_TYPE_Q4_K, 2048, 8192, 512, 4831838208ULL},
    Spec{"blk.92.ffn_down_shexp.weight", "ffn_down_shexp.weight.q6_k", "810ce54c179dbb6f9179a1e557f2f5716142793587b179c0dc952d615750c78a", c::GgufQuantizedType::kQ6K, GGML_TYPE_Q6_K, 2048, 8192, 1, 13762560},
    Spec{"blk.92.ffn_gate_exps.weight", "ffn_gate_exps.weight.q4_k", "de7781e99a2fc89d111dbf05c4bf1c9dc35314a88e98dc3746c5669c1fdc53e6", c::GgufQuantizedType::kQ4K, GGML_TYPE_Q4_K, 8192, 2048, 512, 4831838208ULL},
    Spec{"blk.92.ffn_gate_inp.weight", "ffn_gate_inp.weight.f32", "02151dcd70510967ebf688277965458e07664da440460f73321b830ddf54aca0", std::nullopt, GGML_TYPE_F32, 8192, 512, 1, 16777216},
    Spec{"blk.92.ffn_gate_inp_shexp.weight", "ffn_gate_inp_shexp.weight.f32", "7d506c35ad678bb0ee1f6bbf4986819d0a88f668b3e2809c4387b0e96ac7a29b", std::nullopt, GGML_TYPE_F32, 8192, 1, 1, 32768},
    Spec{"blk.92.ffn_gate_shexp.weight", "ffn_gate_shexp.weight.q5_k", "168de8956d83cc4161f3a51f122c71da9e75e0a49f64298de920d57524031ecb", c::GgufQuantizedType::kQ5K, GGML_TYPE_Q5_K, 8192, 2048, 1, 11534336},
    Spec{"blk.92.ffn_up_exps.weight", "ffn_up_exps.weight.q4_k", "13cd7d815dff21bcc22896c097f7cdb915b00029f154e6592637a37b491caf7a", c::GgufQuantizedType::kQ4K, GGML_TYPE_Q4_K, 8192, 2048, 512, 4831838208ULL},
    Spec{"blk.92.ffn_up_shexp.weight", "ffn_up_shexp.weight.q5_k", "ed4f3906ffef10e08dbd1ccebff41c376a328de3fa10a5d17ba2d34c2064aeaa", c::GgufQuantizedType::kQ5K, GGML_TYPE_Q5_K, 8192, 2048, 1, 11534336},
    Spec{"blk.92.nextn.eh_proj.weight", "eh_proj.weight.q8_0", "6751ef5e6b912e3870e5f9caa02b9a40786aab3c3cb24e1b55af94ca4bfc8e3a", c::GgufQuantizedType::kQ8_0, GGML_TYPE_Q8_0, 16384, 8192, 1, 142606336},
    Spec{"blk.92.nextn.enorm.weight", "enorm.weight.f32", "6c97b1107715a0a97d85f2f6b663ed2f8d22896d65f67cdb3cebc48cd0ab433e", std::nullopt, GGML_TYPE_F32, 0, 8192, 1, 32768},
    Spec{"blk.92.nextn.hnorm.weight", "hnorm.weight.f32", "f2818f79d7c61980a063f600c9d06d07ff6e4e8f9f7b866a244c8e01bbce2a9d", std::nullopt, GGML_TYPE_F32, 0, 8192, 1, 32768},
    Spec{"blk.92.nextn.shared_head_norm.weight", "shared_head_norm.weight.f32", "84d91000988b8995b93fe268a3ac7fe6a0239203ffd23277b64e2a39b31b287a", std::nullopt, GGML_TYPE_F32, 0, 8192, 1, 32768},
    Spec{"blk.92.post_attention_norm.weight", "post_attention_norm.weight.f32", "4d5a6f2409e1d1f814cc5c541fa964c1a41aa5167ecd3e5206c0ee864aec2312", std::nullopt, GGML_TYPE_F32, 0, 8192, 1, 32768},
    Spec{"token_embd.weight", "embedding.weight.q4_k", "fe0d270c8e9fc9db68dfe3d7dc84aec5f6df1309a3ecf27ba3594ecdbbc868aa", c::GgufQuantizedType::kQ4K, GGML_TYPE_Q4_K, 8192, 248320, 1, 1144258560},
    Spec{"output.weight", "head.weight.q4_k", "ca17b07bf13dcfdb4f83b48c6da6a3db06768a235f8db5654cffc28108f7c733", c::GgufQuantizedType::kQ4K, GGML_TYPE_Q4_K, 8192, 248320, 1, 1144258560}
};
inline constexpr std::array<Role, 16> kStaticRoles{
    Role::kAttentionNorm, Role::kOutput, Role::kQuery, Role::kQueryNorm,
    Role::kSharedDown, Role::kRouter, Role::kSharedRouter, Role::kSharedGate,
    Role::kSharedUp, Role::kProjection, Role::kEmbeddingNorm, Role::kHiddenNorm,
    Role::kHeadNorm, Role::kPostNorm, Role::kEmbedding, Role::kHead};
inline std::size_t index(Role role) {
  const auto result = static_cast<std::size_t>(role);
  if (result >= kRoles) throw std::runtime_error("unknown original Q24 NextN role");
  return result;
}
inline const Spec &spec(Role role) { return kSpecs[index(role)]; }
inline Role expert_role(Leg leg) {
  switch (leg) {
  case Leg::kGate: return Role::kExpertGate;
  case Leg::kUp: return Role::kExpertUp;
  case Leg::kDown: return Role::kExpertDown;
  default: throw std::runtime_error("unknown original Q24 expert leg");
  }
}
inline c::TensorDesc description(Role role) {
  const auto &s = spec(role);
  if (s.e > 1)
    return original::detail::take(c::gguf_grouped_weight_description(*s.codec,
        c::DataType::kFloat32, s.e, s.n, s.k), "original Q24 mapped grouped source description");
  if (s.k == 0)
    return original::detail::take(c::TensorDesc::dense(c::DataType::kFloat32,
        std::array{s.n}), "original Q24 norm description");
  auto result = s.codec ? original::detail::take(c::gguf_weight_description(*s.codec,
      c::DataType::kFloat32, s.k, s.n), "original Q24 packed matrix description") :
      original::detail::take(original::detail::take(c::TensorDesc::dense(c::DataType::kFloat32,
          std::array{s.n, s.k}), "original Q24 F32 row matrix").transposed(), "original Q24 F32 MatMul RHS");
  if (role == Role::kEmbedding)
    result = original::detail::take(result.transposed(), "original Q24 Gather orientation");
  return result;
}
inline c::TensorDesc resident_description(Leg leg) {
  const auto &s = spec(expert_role(leg));
  return original::detail::take(c::gguf_grouped_weight_description(*s.codec,
      c::DataType::kFloat32, std::int64_t(kResidentCapacity), s.n, s.k), "original Q24 resident E10 description");
}
struct Selection final {
  std::vector<std::int32_t> globals;
  std::array<std::int32_t, kExpertDomain> map;
};
inline Selection select(std::span<const std::int32_t> actual_ids) {
  if (actual_ids.empty() || actual_ids.size() > kResidentCapacity)
    throw std::runtime_error("actual Q24 selection extent must be1..10 global IDs");
  Selection result; result.map.fill(-1); result.globals.reserve(actual_ids.size());
  for (const auto global : actual_ids) {
    if (global < 0 || global >= kExpertDomain)
      throw std::runtime_error("actual Q24 global expert outside0..511 domain");
    result.globals.push_back(global);
  }
  std::ranges::sort(result.globals);
  result.globals.erase(std::unique(result.globals.begin(), result.globals.end()), result.globals.end());
  for (std::size_t local = 0; local < result.globals.size(); ++local)
    result.map[std::size_t(result.globals[local])] = std::int32_t(local);
  return result;
}

class Data final {
 public:
  static Data load(const std::filesystem::path &directory) {
    for (const auto &s : kSpecs)
      if (s.sha256.size() != 64) throw std::runtime_error("original Q24 NextN role is not pinned");
    Data result;
    for (std::size_t slot = 0; slot < kRoles; ++slot) {
      const auto &s = kSpecs[slot];
      if (original::detail::take(description(static_cast<Role>(slot)).required_bytes(), "original Q24 typed extent") != s.bytes)
        throw std::runtime_error("original Q24 codec/geometry extent differs");
      const auto path = directory / std::string(s.file);
      auto mapped = original::detail::take(m::FileMapping::open(path.string().c_str()), "map original Q24 role");
      if (mapped.size() != s.bytes) throw std::runtime_error("original Q24 role extent differs: " + path.string());
      const auto raw = original::detail::take(mapped.bytes(), "original Q24 mapped bytes");
      if (input::sha256(raw) != s.sha256) throw std::runtime_error("original Q24 SHA256 differs: " + path.string());
      result.mappings_[slot] = std::move(mapped);
      if (s.k == 0) {
        auto &values = result.norms_[slot]; values.resize(std::size_t(s.n));
        for (std::size_t i = 0; i < values.size(); ++i) {
          std::uint32_t bits = 0;
          for (std::size_t byte = 0; byte < 4; ++byte)
            bits |= std::to_integer<std::uint32_t>(raw[i * 4 + byte]) << (8 * byte);
          values[i] = std::bit_cast<float>(bits);
          if (!std::isfinite(values[i])) throw std::runtime_error("original Q24 norm is nonfinite");
        }
      }
    }
    return result;
  }
  std::span<const std::byte> bytes(Role role) const {
    return original::detail::take(mappings_[index(role)].bytes(), "original Q24 mapped role bytes");
  }
  m::ConstBufferView view(Role role) const {
    return original::detail::take(mappings_[index(role)].view(), "original Q24 readonly mapped role");
  }
  std::span<const float> norm(Role role) const {
    if (spec(role).k != 0) throw std::runtime_error("original Q24 requested matrix as norm");
    return norms_[index(role)];
  }
  c::TensorDesc expert_description(Leg leg) const { return description(expert_role(leg)); }
  std::span<const std::byte> expert_bytes(Leg leg, std::int32_t global) const {
    const auto range = expert_range(leg, global);
    return bytes(expert_role(leg)).subspan(range[0], range[1]);
  }
  m::ConstBufferView expert_view(Leg leg, std::int32_t global) const {
    const auto range = expert_range(leg, global);
    return original::detail::take(mappings_[index(expert_role(leg))].view(
        range[0], range[1]), "original Q24 readonly selected expert");
  }
 private:
  std::array<std::size_t, 2> expert_range(Leg leg, std::int32_t global) const {
    if (global < 0 || global >= kExpertDomain)
      throw std::runtime_error("original Q24 expert outside full source domain");
    const auto desc = expert_description(leg);
    if (desc.rank() != 3 || desc.dimensions()[0] != kExpertDomain || desc.byte_strides()[0] <= 0)
      throw std::runtime_error("original Q24 expert source stride is not canonical");
    const auto encoded_stride = desc.byte_strides()[0];
    const auto stride = static_cast<std::size_t>(encoded_stride);
    if (std::uint64_t(stride) != std::uint64_t(encoded_stride) ||
        std::size_t(global) > std::numeric_limits<std::size_t>::max() / stride)
      throw std::runtime_error("original Q24 expert address exceeds size_t");
    const auto offset = std::size_t(global) * stride;
    const auto source_bytes = mappings_[index(expert_role(leg))].size();
    if (offset > source_bytes || stride > source_bytes - offset)
      throw std::runtime_error("original Q24 selected expert exceeds mapped source");
    return {offset, stride};
  }
  // All borrowed original spans/views keep this Data's strong owner in Roles
  // or their reference closure. No full expert decode/expanded bank is held.
  std::array<m::FileMapping, kRoles> mappings_;
  std::array<std::vector<float>, kRoles> norms_;
};
struct Roles final {
  std::shared_ptr<const Data> source;
  std::array<original::ReadonlyRole, kStaticRoles.size()> roles;
  const original::ReadonlyRole &at(Role role) const {
    const auto found = std::find(kStaticRoles.begin(), kStaticRoles.end(), role);
    if (found == kStaticRoles.end()) throw std::runtime_error("full original Q24 expert banks are source-only; bind the actual E10 generation");
    return roles[std::size_t(found - kStaticRoles.begin())];
  }
};
inline Roles bind(std::shared_ptr<const Data> source, const c::StorageOwner &owner) {
  if (!source || !owner) throw std::runtime_error("original Q24 NextN needs original data and its actual storage owner");
  Roles result; result.source = std::move(source);
  for (std::size_t slot = 0; slot < kStaticRoles.size(); ++slot) {
    const auto role = kStaticRoles[slot];
    result.roles[slot] = original::detail::bind(spec(role).name, description(role),
        owner, result.source->bytes(role), &kIdentity);
  }
  return result;
}
} // namespace qwen24_original_nextn_roles
