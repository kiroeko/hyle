#pragma once

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
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

// Original final main-model target block63 and its independent global
// output_norm, MAIN Q27 UD-Q4_K_M. These are distinct trained roles from
// NextN64: Q/O/FFN-gate are Q5_K and K is Q6_K. Existing full vocabulary and
// NextN banks are borrowed by the caller rather than duplicated here.
namespace qwen27_original_target_roles {
namespace original = qwen_original_mtp_roles;
namespace input = original_role_input;
namespace c = hyle::compute;
namespace m = hyle::mem;
inline constexpr std::int64_t kHidden = 5120, kVocabulary = 248320;
inline constexpr std::int64_t kHeads = 24, kKvHeads = 4, kDepth = 256, kIntermediate = 17408;
inline constexpr float kEpsilon = 1.0e-6F;
inline constexpr float kRopeTheta = 1.0e7F;
inline constexpr std::int64_t kBlock = 63, kRotaryDimensions = 64;
inline constexpr std::size_t kOriginalBytes = 282318848;
inline constexpr auto &kIdentity = original::q27::kIdentity;
enum class Role : std::size_t {
  kKey, kKeyNorm, kAttentionNorm, kOutput, kQuery, kQueryNorm, kValue,
  kFfnDown, kFfnGate, kFfnUp, kPostNorm, kTargetNorm, kCount
};
inline constexpr std::size_t kRoles = static_cast<std::size_t>(Role::kCount);
struct Spec final {
  std::string_view name, file, sha256;
  std::optional<c::GgufQuantizedType> codec;
  ggml_type native;
  std::int64_t k, n; // k0 is an original plain F32[n] norm.
  std::size_t bytes;
};
// Pins are filled only from root's completed fixed-range acquisition receipt.
// An empty pin rejects loading, never permits an unverified parameter file.
inline constexpr std::array<Spec, kRoles> kSpecs{
    Spec{"blk.63.attn_k.weight", "attn_k.weight.q6_k", "54c18f2b51afeed9b7b198cf66fdd89de58a5ced75c33dfd58553664fa947c61", c::GgufQuantizedType::kQ6K, GGML_TYPE_Q6_K, 5120, 1024, 4300800},
    Spec{"blk.63.attn_k_norm.weight", "attn_k_norm.weight.f32", "dd2837b022e7bb5f24223f7937c377658d72258baf44072def2bff47fbf5c039", std::nullopt, GGML_TYPE_F32, 0, 256, 1024},
    Spec{"blk.63.attn_norm.weight", "attn_norm.weight.f32", "fff0b300a720e0626a1b5982729a66769cfcb796f9cfbfe2e02293a7e448b073", std::nullopt, GGML_TYPE_F32, 0, 5120, 20480},
    Spec{"blk.63.attn_output.weight", "attn_output.weight.q5_k", "1ebdefeb5257844af90f8ba689c549e716a84d94b14c7a3f3f32c78b5a34c883", c::GgufQuantizedType::kQ5K, GGML_TYPE_Q5_K, 6144, 5120, 21626880},
    Spec{"blk.63.attn_q.weight", "attn_q.weight.q5_k", "2774ced2653738b83dce2fda1ee370f55acadbca8c40cd40b8688088d2d1f581", c::GgufQuantizedType::kQ5K, GGML_TYPE_Q5_K, 5120, 12288, 43253760},
    Spec{"blk.63.attn_q_norm.weight", "attn_q_norm.weight.f32", "4d82f12e9d3e60003032eee0c0a413b85588b9253ba3bfe43456eed32a91960c", std::nullopt, GGML_TYPE_F32, 0, 256, 1024},
    Spec{"blk.63.attn_v.weight", "attn_v.weight.q8_0", "cfd03beab4da353c3629418b64040dc37545c8b1682bd22e04f8766c9d0563de", c::GgufQuantizedType::kQ8_0, GGML_TYPE_Q8_0, 5120, 1024, 5570560},
    Spec{"blk.63.ffn_down.weight", "ffn_down.weight.q6_k", "6ea4d6dcff1be3773e7f39e5e837e9d1e78d5fc9e226024dfe4726936de0b92a", c::GgufQuantizedType::kQ6K, GGML_TYPE_Q6_K, 17408, 5120, 73113600},
    Spec{"blk.63.ffn_gate.weight", "ffn_gate.weight.q5_k", "9005ad5a532db916ffd7236e589b8398c1fd094bf1013ce6452e46663c785818", c::GgufQuantizedType::kQ5K, GGML_TYPE_Q5_K, 5120, 17408, 61276160},
    Spec{"blk.63.ffn_up.weight", "ffn_up.weight.q6_k", "0878f5aa34bb63cb353284e89f642325093dc3b8f729ac65887d29ec4f13578d", c::GgufQuantizedType::kQ6K, GGML_TYPE_Q6_K, 5120, 17408, 73113600},
    Spec{"blk.63.post_attention_norm.weight", "post_attention_norm.weight.f32", "bd1d77170441438d96258413ed00462a96a4290341dbcc78cf66c9990aa06506", std::nullopt, GGML_TYPE_F32, 0, 5120, 20480},
    Spec{"output_norm.weight", "output_norm.weight.f32", "b3ba4f3231a21b1559f52115d3bffc7c28dcd03e4be22df269ef97a3f5d4bfbf", std::nullopt, GGML_TYPE_F32, 0, 5120, 20480}
};
inline std::size_t index(Role role) {
  const auto result = static_cast<std::size_t>(role);
  if (result >= kRoles) throw std::runtime_error("unknown original Q27 target role");
  return result;
}
inline const Spec &spec(Role role) { return kSpecs[index(role)]; }
inline c::TensorDesc description(Role role) {
  const auto &s = spec(role);
  if (!s.codec)
    return original::detail::take(c::TensorDesc::dense(c::DataType::kFloat32,
        std::array{s.n}), "original Q27 F32 norm description");
  return original::detail::take(c::gguf_weight_description(*s.codec,
      c::DataType::kFloat32, s.k, s.n), "original Q27 target packed matrix description");
}

class Data final {
 public:
  static Data load(const std::filesystem::path &directory) {
    for (const auto &s : kSpecs)
      if (s.sha256.size() != 64) throw std::runtime_error("original Q27 target input is not pinned");
    Data result;
    for (std::size_t slot = 0; slot < kRoles; ++slot) {
      const auto role = static_cast<Role>(slot);
      const auto &s = kSpecs[slot];
      if (original::detail::take(description(role).required_bytes(), "original Q27 typed extent") != s.bytes)
        throw std::runtime_error("original Q27 codec/geometry extent differs");
      const auto path = directory / std::string(s.file);
      auto mapped = original::detail::take(m::FileMapping::open(path.string().c_str()), "map original Q27 target role");
      if (mapped.size() != s.bytes) throw std::runtime_error("original Q27 role extent differs: " + path.string());
      const auto raw = original::detail::take(mapped.bytes(), "original Q27 mapped input");
      // Authenticate original encoded bytes with bounded SHA block/tail IO;
      // no decoded matrix or padded whole-file copy is created.
      if (input::sha256(raw) != s.sha256) throw std::runtime_error("original Q27 SHA256 differs: " + path.string());
      result.mappings_[slot] = std::move(mapped);
      if (!s.codec) {
        auto &values = result.norms_[slot];
        values.resize(std::size_t(s.n));
        for (std::size_t i = 0; i < values.size(); ++i) {
          std::uint32_t bits = 0;
          for (std::size_t byte = 0; byte < 4; ++byte)
            bits |= std::to_integer<std::uint32_t>(raw[i * 4 + byte]) << (8 * byte);
          values[i] = std::bit_cast<float>(bits);
          if (!std::isfinite(values[i])) throw std::runtime_error("original Q27 norm is nonfinite");
        }
      }
    }
    return result;
  }
  std::span<const std::byte> bytes(Role role) const {
    return original::detail::take(mappings_[index(role)].bytes(), "original Q27 mapped role bytes");
  }
  m::ConstBufferView view(Role role) const {
    return original::detail::take(mappings_[index(role)].view(), "original Q27 readonly mapped view");
  }
  std::span<const float> norm(Role role) const {
    if (spec(role).codec) throw std::runtime_error("original Q27 requested matrix as norm");
    return norms_[index(role)];
  }
 private:
  // Returned views/spans borrow Data. Bound and every reference closure keep
  // its shared owner alive through all consumers; mappings are never writable.
  std::array<m::FileMapping, kRoles> mappings_;
  std::array<std::vector<float>, kRoles> norms_;
};
struct Bound final {
  std::shared_ptr<const Data> source;
  std::array<original::ReadonlyRole, kRoles> roles;
  const original::ReadonlyRole &at(Role role) const { return roles[index(role)]; }
};
inline Bound bind(std::shared_ptr<const Data> source, const c::StorageOwner &owner) {
  if (!source || !owner) throw std::runtime_error("original Q27 target needs original data and its actual storage owner");
  Bound result;
  result.source = std::move(source);
  for (std::size_t slot = 0; slot < kRoles; ++slot) {
    const auto role = static_cast<Role>(slot);
    result.roles[slot] = original::detail::bind(kSpecs[slot].name, description(role),
        owner, result.source->bytes(role), &kIdentity);
  }
  return result;
}
} // namespace qwen27_original_target_roles
