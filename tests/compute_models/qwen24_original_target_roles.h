#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "qwen24_original_nextn_roles.h"
#include "hyle/mem/mapping.h"

// Original MAIN Q24 UD-IQ4_XS final target91 and independent global norm.
// All512 expert rows remain mapped in their real IQ4_XS encoding. Only actual
// selected rows enter bounded resident Graph inputs; these trained roles are
// distinct from NextN92 Q4_K experts. Existing vocabulary/NextN are borrowed.
namespace qwen24_original_target_roles {
namespace original = qwen_original_mtp_roles;
namespace input = original_role_input;
namespace nextn = qwen24_original_nextn_roles;
namespace c = hyle::compute;
namespace m = hyle::mem;
inline constexpr auto &kIdentity = original::q24::kIdentity;
inline constexpr std::int64_t kHidden = 8192, kIntermediate = 2048, kVocabulary = 248320;
inline constexpr std::int32_t kExpertDomain = 512;
inline constexpr std::size_t kResidentCapacity = 10, kExpertBytes = 8912896;
inline constexpr std::size_t kOriginalBytes = 14032308224ULL;
inline constexpr std::size_t kStaticBytes = 342099968ULL;
using Leg = nextn::Leg;
inline constexpr std::int64_t kBlock = 91, kHeads = 64, kKvHeads = 4, kDepth = 256;
inline constexpr float kEpsilon = 1.0e-6F, kRopeTheta = 1.0e7F;
enum class Role : std::size_t {
  kKey, kKeyNorm, kAttentionNorm, kOutput, kQuery, kQueryNorm, kValue,
  kExpertDown, kSharedDown, kExpertGate, kRouter, kSharedRouter, kSharedGate,
  kExpertUp, kSharedUp, kPostNorm, kTargetNorm, kCount
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
    Spec{"blk.91.attn_k.weight", "attn_k.weight.q5_k", "de4ee94d516cbae65723d1ff2481b5c1abbe6cf4327d95de0cff5a6b62ba0145", c::GgufQuantizedType::kQ5K, GGML_TYPE_Q5_K, 8192, 1024, 1, 5767168ULL},
    Spec{"blk.91.attn_k_norm.weight", "attn_k_norm.weight.f32", "36489e8fbe3b618d03885fec3e24959a616ffa4c41c7ddb04ea61397f1cfd60c", std::nullopt, GGML_TYPE_F32, 0, 256, 1, 1024ULL},
    Spec{"blk.91.attn_norm.weight", "attn_norm.weight.f32", "4da43d09a561163a684f4e9d5426e5b51d428a91ee827b5140bec05caa469f54", std::nullopt, GGML_TYPE_F32, 0, 8192, 1, 32768ULL},
    Spec{"blk.91.attn_output.weight", "attn_output.weight.q5_k", "7688a1ed9fa55cfe1919c247252d44a32d4fc3d46148d970abed76e0df3b989a", c::GgufQuantizedType::kQ5K, GGML_TYPE_Q5_K, 16384, 8192, 1, 92274688ULL},
    Spec{"blk.91.attn_q.weight", "attn_q.weight.q5_k", "37c7bce5f8a01a7a1b5602d0a31745bee4f05a82ee469a57a4d0793ac7fbd5a4", c::GgufQuantizedType::kQ5K, GGML_TYPE_Q5_K, 8192, 32768, 1, 184549376ULL},
    Spec{"blk.91.attn_q_norm.weight", "attn_q_norm.weight.f32", "0377cdf1c646494a50688c78ed370e9eb2725d0014786793ff8646b0287956af", std::nullopt, GGML_TYPE_F32, 0, 256, 1, 1024ULL},
    Spec{"blk.91.attn_v.weight", "attn_v.weight.q5_k", "1107de0f2ec6484a8f7a7a894537443524323e98a16dcbe6872690e52d11aded", c::GgufQuantizedType::kQ5K, GGML_TYPE_Q5_K, 8192, 1024, 1, 5767168ULL},
    Spec{"blk.91.ffn_down_exps.weight", "ffn_down_exps.weight.iq4_xs", "9313854a1acfb72e9e69ee2225352103284a9b22cb27f82c8ac3ce632c128921", c::GgufQuantizedType::kIq4Xs, GGML_TYPE_IQ4_XS, 2048, 8192, 512, 4563402752ULL},
    Spec{"blk.91.ffn_down_shexp.weight", "ffn_down_shexp.weight.q6_k", "abba1305045bc423be8e8c5433c39c9f52db120f5f334ec3ff1c6ed6eaaa2681", c::GgufQuantizedType::kQ6K, GGML_TYPE_Q6_K, 2048, 8192, 1, 13762560ULL},
    Spec{"blk.91.ffn_gate_exps.weight", "ffn_gate_exps.weight.iq4_xs", "63fdb971d479858e4073b87ee69405607b1b593f12740cfc3fbf9b583171df42", c::GgufQuantizedType::kIq4Xs, GGML_TYPE_IQ4_XS, 8192, 2048, 512, 4563402752ULL},
    Spec{"blk.91.ffn_gate_inp.weight", "ffn_gate_inp.weight.f32", "ef5efde62f9ac68fb6f97d9d3a34661e30b279849b3e8c4a919ab03eb5f3baf3", std::nullopt, GGML_TYPE_F32, 8192, 512, 1, 16777216ULL},
    Spec{"blk.91.ffn_gate_inp_shexp.weight", "ffn_gate_inp_shexp.weight.f32", "fdef3f9399555111dad0fc976ab5437e06cefa32ead03fb7886cb5af55256d56", std::nullopt, GGML_TYPE_F32, 8192, 1, 1, 32768ULL},
    Spec{"blk.91.ffn_gate_shexp.weight", "ffn_gate_shexp.weight.q5_k", "a1ab3ac54eca19da77861424c694cdb2ff38d6b930e188145e6835f70685cc07", c::GgufQuantizedType::kQ5K, GGML_TYPE_Q5_K, 8192, 2048, 1, 11534336ULL},
    Spec{"blk.91.ffn_up_exps.weight", "ffn_up_exps.weight.iq4_xs", "c8eb753c5404316bb8c6d109fdeda798bae22060a4a73318d6b8f71ee4f39b4c", c::GgufQuantizedType::kIq4Xs, GGML_TYPE_IQ4_XS, 8192, 2048, 512, 4563402752ULL},
    Spec{"blk.91.ffn_up_shexp.weight", "ffn_up_shexp.weight.q5_k", "811351e9c511137ed6893a8bdd7cb64b823f727f21e4b5b817ed44233c7da41e", c::GgufQuantizedType::kQ5K, GGML_TYPE_Q5_K, 8192, 2048, 1, 11534336ULL},
    Spec{"blk.91.post_attention_norm.weight", "post_attention_norm.weight.f32", "5fc9df2dfde80ff743ec7c9c3aae9979ada6917aef50579d8e56a4f176cfc4d5", std::nullopt, GGML_TYPE_F32, 0, 8192, 1, 32768ULL},
    Spec{"output_norm.weight", "output_norm.weight.f32", "6cec2111e1c0dabe6dd413a0cc8a102519a510b7ce3272866d938cf518ce7c1c", std::nullopt, GGML_TYPE_F32, 0, 8192, 1, 32768ULL}
};
inline constexpr std::array<Role, 14> kStaticRoles{
    Role::kKey, Role::kKeyNorm, Role::kAttentionNorm, Role::kOutput,
    Role::kQuery, Role::kQueryNorm, Role::kValue, Role::kSharedDown,
    Role::kRouter, Role::kSharedRouter, Role::kSharedGate, Role::kSharedUp,
    Role::kPostNorm, Role::kTargetNorm};
inline std::size_t index(Role role) {
  const auto result = static_cast<std::size_t>(role);
  if (result >= kRoles) throw std::runtime_error("unknown original Q24 target role");
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
  return s.codec ? original::detail::take(c::gguf_weight_description(*s.codec,
      c::DataType::kFloat32, s.k, s.n), "original Q24 packed matrix description") :
      original::detail::take(original::detail::take(c::TensorDesc::dense(c::DataType::kFloat32,
          std::array{s.n, s.k}), "original Q24 F32 row matrix").transposed(), "original Q24 F32 MatMul RHS");
}
inline c::TensorDesc resident_description(Leg leg) {
  const auto &s = spec(qwen24_original_target_roles::expert_role(leg));
  return original::detail::take(c::gguf_grouped_weight_description(*s.codec,
      c::DataType::kFloat32, std::int64_t(kResidentCapacity), s.n, s.k), "original Q24 resident E10 description");
}

class Data final {
 public:
  static Data load(const std::filesystem::path &directory) {
    for (const auto &s : kSpecs)
      if (s.sha256.size() != 64) throw std::runtime_error("original Q24 target role is not pinned");
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
  c::TensorDesc expert_description(Leg leg) const {
    return description(qwen24_original_target_roles::expert_role(leg));
  }
  std::span<const std::byte> expert_bytes(Leg leg, std::int32_t global) const {
    const auto range = expert_range(leg, global);
    return bytes(qwen24_original_target_roles::expert_role(leg)).subspan(range[0], range[1]);
  }
  m::ConstBufferView expert_view(Leg leg, std::int32_t global) const {
    const auto range = expert_range(leg, global);
    return original::detail::take(mappings_[index(qwen24_original_target_roles::expert_role(leg))].view(
        range[0], range[1]), "original Q24 readonly selected expert");
  }
 private:
  std::array<std::size_t, 2> expert_range(Leg leg, std::int32_t global) const {
    if (global < 0 || global >= kExpertDomain)
      throw std::runtime_error("original Q24 expert outside full source domain");
    const auto desc = expert_description(leg);
    if (desc.rank() != 3 || desc.dimensions()[0] != kExpertDomain)
      throw std::runtime_error("original Q24 expert source geometry is not canonical");
    const auto encoded_stride = desc.byte_strides()[0];
    if (encoded_stride <= 0)
      throw std::runtime_error("original Q24 expert source stride is not positive");
    const auto stride = static_cast<std::size_t>(encoded_stride);
    if (std::uint64_t(stride) != std::uint64_t(encoded_stride) ||
        std::size_t(global) > std::numeric_limits<std::size_t>::max() / stride)
      throw std::runtime_error("original Q24 expert address exceeds size_t");
    const auto offset = std::size_t(global) * stride;
    const auto source_bytes = mappings_[index(qwen24_original_target_roles::expert_role(leg))].size();
    if (offset > source_bytes || stride > source_bytes - offset)
      throw std::runtime_error("original Q24 selected expert exceeds mapped source");
    return {offset, stride};
  }

  // All borrowed original spans/views keep this Data's strong owner in Bound
  // or their reference closure. No full expert decode/expanded bank is held.
  std::array<m::FileMapping, kRoles> mappings_;
  std::array<std::vector<float>, kRoles> norms_;
};
struct Bound final {
  std::shared_ptr<const Data> source;
  std::array<original::ReadonlyRole, kStaticRoles.size()> roles;
  const original::ReadonlyRole &at(Role role) const {
    const auto found = std::find(kStaticRoles.begin(), kStaticRoles.end(), role);
    if (found == kStaticRoles.end()) throw std::runtime_error("full original Q24 expert banks are source-only; bind the actual E10 generation");
    return roles[std::size_t(found - kStaticRoles.begin())];
  }
};
inline Bound bind(std::shared_ptr<const Data> source, const c::StorageOwner &owner) {
  if (!source || !owner) throw std::runtime_error("original Q24 target needs original data and its actual storage owner");
  Bound result; result.source = std::move(source);
  for (std::size_t slot = 0; slot < kStaticRoles.size(); ++slot) {
    const auto role = kStaticRoles[slot];
    result.roles[slot] = original::detail::bind(spec(role).name, description(role),
        owner, result.source->bytes(role), &kIdentity);
  }
  return result;
}
} // namespace qwen24_original_target_roles
