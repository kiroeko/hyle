#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "glm_original_moe_roles.h"

// Pinned official Flash target44 parameters and final target model.norm.
// Stored BF16 matrices stay BF16. Fixed-source parameter loaders require only
// the two small HC matrices and three short convolution banks widened to F32.
// This supplier owns no Graph, executor, routing policy or mutable model state.
namespace glm_original_target44_roles {
namespace c = hyle::compute;
namespace m = hyle::mem;
namespace shared = glm_shared_vocabulary;
namespace moe = glm_original_moe_roles;
inline constexpr std::string_view kRepository = shared::kRepository;
inline constexpr std::string_view kRevision = shared::kRevision;
inline constexpr std::int64_t kHidden = 4096, kStreams = 4, kHeads = 64, kDepth = 128;
inline constexpr float kNormEpsilon = 1.0e-5F, kHcEpsilon = 1.0e-6F;
inline constexpr std::size_t kOriginalBytes = 277078744;
// Root fills this only from the completed independent layer44 MoE acquisition.
// An absent pin rejects loading; it cannot silently select layer45 parameters.
inline constexpr std::string_view kMoeIndexSha256 = "1eb3e9547778827f8fc5945d665df43209b3538a8f63a8e90ba3d2d7787ca598";
enum class Role : std::size_t {
  kHcAttnBase, kHcAttnFn, kHcAttnScale, kHcFfnBase, kHcFfnFn, kHcFfnScale,
  kInputNorm, kPostNorm, kALog, kBeta, kDtBias, kForgetA, kForgetB, kGateA,
  kGateB, kKeyConv, kKey, kOutputNorm, kOutput, kQueryConv, kQuery,
  kValueConv, kValue, kModelNorm, kCount
};
inline constexpr std::size_t kRoles = static_cast<std::size_t>(Role::kCount);
struct Spec final {
  std::string_view name, file, sha256;
  c::DataType raw_type;
  std::array<std::int64_t, 3> shape;
  std::size_t rank, bytes;
};
inline constexpr std::array<Spec, kRoles> kSpecs{
    Spec{"model.language_model.layers.44.hc_attn_base", "hc_attn_base.f32", "441bfea20b5575f487d2f9d742d5a5403064b83d570db3326cee092af66350a0", c::DataType::kFloat32, {24, 0, 0}, 1, 96},
    Spec{"model.language_model.layers.44.hc_attn_fn", "hc_attn_fn.bf16", "b5f3e569fc060345c3f4c9d1e7ec35d106587766325848717adea09e236d12ac", c::DataType::kBFloat16, {24, 16384, 0}, 2, 786432},
    Spec{"model.language_model.layers.44.hc_attn_scale", "hc_attn_scale.f32", "892c4503190695517bec43acd35bf1de881fc243af97277ec698ee40be769997", c::DataType::kFloat32, {3, 0, 0}, 1, 12},
    Spec{"model.language_model.layers.44.hc_ffn_base", "hc_ffn_base.f32", "57998d7d83e42c111a0758ae478f34a1b13ccf9912869fd4e1b5d31cdd076936", c::DataType::kFloat32, {24, 0, 0}, 1, 96},
    Spec{"model.language_model.layers.44.hc_ffn_fn", "hc_ffn_fn.bf16", "a4d59b06a0258af0f2c3026722c1a441edb9198967d1641e19e0a3f9d80c4dfd", c::DataType::kBFloat16, {24, 16384, 0}, 2, 786432},
    Spec{"model.language_model.layers.44.hc_ffn_scale", "hc_ffn_scale.f32", "7c66c9379664cb8b2332b2675d33f4b5bca7050e024c1b47bf2cec074379f9b1", c::DataType::kFloat32, {3, 0, 0}, 1, 12},
    Spec{"model.language_model.layers.44.input_layernorm.weight", "input_layernorm.weight.bf16", "6a8b917f4f110355207a77b56ebc68e4b5c0fac8cda9f0c820078fa1e1ddcddd", c::DataType::kBFloat16, {4096, 0, 0}, 1, 8192},
    Spec{"model.language_model.layers.44.post_attention_layernorm.weight", "post_attention_layernorm.weight.bf16", "ca52c0b03f154e1ae561bc47fdd54f662edcf797d7fb1af9ec13bb88a40eaafb", c::DataType::kBFloat16, {4096, 0, 0}, 1, 8192},
    Spec{"model.language_model.layers.44.self_attn.A_log", "self_attn.A_log.f32", "c0227d397bdcefef8553cc1981d9580bdb312443fda1fc2dffbb1ef12b9040b4", c::DataType::kFloat32, {64, 0, 0}, 1, 256},
    Spec{"model.language_model.layers.44.self_attn.b_proj.weight", "self_attn.b_proj.weight.bf16", "22437572dcf63f8fd0215e3e6b621cd0f6718fae53000a47e08dfac73774bfee", c::DataType::kBFloat16, {64, 4096, 0}, 2, 524288},
    Spec{"model.language_model.layers.44.self_attn.dt_bias", "self_attn.dt_bias.f32", "3a0296f7b6eff5f2b714f1ee9b5786668cc20e906e3dad4ada9123783beed7cc", c::DataType::kFloat32, {8192, 0, 0}, 1, 32768},
    Spec{"model.language_model.layers.44.self_attn.f_a_proj.weight", "self_attn.f_a_proj.weight.bf16", "670b3c0619eca43aa022b85dcdf80ad61066107d3cd10a8e6f3db1a55dcbc897", c::DataType::kBFloat16, {128, 4096, 0}, 2, 1048576},
    Spec{"model.language_model.layers.44.self_attn.f_b_proj.weight", "self_attn.f_b_proj.weight.bf16", "ce802d2af7d55a771cd5a7ee9264cb64f6d9e0e9059c8b2c43144bce156218d6", c::DataType::kBFloat16, {8192, 128, 0}, 2, 2097152},
    Spec{"model.language_model.layers.44.self_attn.g_a_proj.weight", "self_attn.g_a_proj.weight.bf16", "214f7316fe05cef05ce379b9207e73c56d624869f3a4232e3460a6efb7447889", c::DataType::kBFloat16, {128, 4096, 0}, 2, 1048576},
    Spec{"model.language_model.layers.44.self_attn.g_b_proj.weight", "self_attn.g_b_proj.weight.bf16", "0394a7c94463a477bd4e0f6c56445a3dae057b4fae0e7a012a27b995b4edae16", c::DataType::kBFloat16, {8192, 128, 0}, 2, 2097152},
    Spec{"model.language_model.layers.44.self_attn.k_conv1d.weight", "self_attn.k_conv1d.weight.bf16", "689e51f88abd9c25b536e2af7e36c78302e5f7ed1c6f328dc631ac802a217cf1", c::DataType::kBFloat16, {8192, 1, 4}, 3, 65536},
    Spec{"model.language_model.layers.44.self_attn.k_proj.weight", "self_attn.k_proj.weight.bf16", "67b8a0eec7be8ab87773c0784bfb0bdff403e5f02c560bf3e644871cc23b5b29", c::DataType::kBFloat16, {8192, 4096, 0}, 2, 67108864},
    Spec{"model.language_model.layers.44.self_attn.o_norm.weight", "self_attn.o_norm.weight.bf16", "4c109a7bdd01133da1cb8bf9500d62db3237f9354db4eee55624994d38de64ea", c::DataType::kBFloat16, {128, 0, 0}, 1, 256},
    Spec{"model.language_model.layers.44.self_attn.o_proj.weight", "self_attn.o_proj.weight.bf16", "fc33867c99b23bcd8a9de21fad878248c968f439fd50f250e963ef891086ea7c", c::DataType::kBFloat16, {4096, 8192, 0}, 2, 67108864},
    Spec{"model.language_model.layers.44.self_attn.q_conv1d.weight", "self_attn.q_conv1d.weight.bf16", "225d2510e8ee5f8e8e053a12c62c3996dc4051e402396a693c9e2411b7820b42", c::DataType::kBFloat16, {8192, 1, 4}, 3, 65536},
    Spec{"model.language_model.layers.44.self_attn.q_proj.weight", "self_attn.q_proj.weight.bf16", "c2570fd60a3add837dfff06db309271cec6fd323f5ed5d6937d3c2bf54b8524a", c::DataType::kBFloat16, {8192, 4096, 0}, 2, 67108864},
    Spec{"model.language_model.layers.44.self_attn.v_conv1d.weight", "self_attn.v_conv1d.weight.bf16", "d4eac8ba7eb744cb02e32d65ee4708759e5d10a68f832d10032fbfb4cb7e66e9", c::DataType::kBFloat16, {8192, 1, 4}, 3, 65536},
    Spec{"model.language_model.layers.44.self_attn.v_proj.weight", "self_attn.v_proj.weight.bf16", "12d10807246c8981d815b9284b933ddd2dcc5ec602fbf1f98772d5bc68701cf0", c::DataType::kBFloat16, {8192, 4096, 0}, 2, 67108864},
    Spec{"model.language_model.norm.weight", "model_norm.weight.bf16", "aeae580add5eb1328bd06dbf47a218b316e01d7ecf857fb4ab8e987973f7ed22", c::DataType::kBFloat16, {4096, 0, 0}, 1, 8192},
};
inline std::size_t index(Role role) {
  const auto slot = static_cast<std::size_t>(role);
  if (slot >= kRoles) throw std::runtime_error("unknown original GLM target44 role");
  return slot;
}
inline const Spec &spec(Role role) { return kSpecs[index(role)]; }
using Pins = std::array<std::string_view,kRoles>;
inline constexpr Pins kDefaultPins=[] {
  Pins result{};
  for(std::size_t i=0;i<kRoles;++i)result[i]=kSpecs[i].sha256;
  return result;
}();
inline bool is_hc(Role role) noexcept {
  return role == Role::kHcAttnFn || role == Role::kHcFfnFn;
}
inline bool is_conv(Role role) noexcept {
  return role == Role::kKeyConv || role == Role::kQueryConv || role == Role::kValueConv;
}
inline bool is_norm(Role role) noexcept {
  return role == Role::kInputNorm || role == Role::kPostNorm ||
      role == Role::kOutputNorm || role == Role::kModelNorm;
}
inline c::TensorDesc raw_description(Role role) {
  const auto &s = spec(role);
  return shared::take(c::TensorDesc::dense(s.raw_type,
      std::span(s.shape.data(), s.rank)), "original target44 raw descriptor");
}
inline c::TensorDesc description(Role role) {
  const auto &s = spec(role);
  if (is_hc(role))
    return shared::take(c::TensorDesc::dense(c::DataType::kFloat32,
        std::array<std::int64_t, 2>{24, kStreams * kHidden}), "original target44 HC runtime descriptor");
  if (is_conv(role))
    return shared::take(c::TensorDesc::dense(c::DataType::kFloat32,
        std::array<std::int64_t, 2>{kHeads * kDepth, 4}), "original target44 conv runtime descriptor");
  return shared::take(c::TensorDesc::dense(s.raw_type,
      std::span(s.shape.data(), s.rank)), "original target44 runtime descriptor");
}
class Data final {
 public:
  static Data load(const std::filesystem::path &directory,const Pins &pins=kDefaultPins) {
    for (const auto pin : pins)
      if (pin.size()!=64||!std::ranges::all_of(pin,[](char ch) {
            return (ch>='0'&&ch<='9')||(ch>='a'&&ch<='f'); }))
        throw std::runtime_error("original target44 role is not pinned");
    Data result;
    for (std::size_t slot = 0; slot < kRoles; ++slot) {
      const auto role = static_cast<Role>(slot);
      const auto &s = spec(role);
      if (shared::take(raw_description(role).required_bytes(), "original target44 raw extent") != s.bytes)
        throw std::runtime_error("original target44 raw codec/geometry differs");
      const auto path = directory / std::string(s.file);
      auto mapped = shared::take(m::FileMapping::open(path.string().c_str()), "map original target44 role");
      if (mapped.size() != s.bytes)
        throw std::runtime_error("original target44 role byte length differs: " + path.string());
      const auto raw = shared::take(mapped.bytes(), "original target44 mapped bytes");
      if (shared::detail::mapped_sha256(raw) != pins[slot])
        throw std::runtime_error("original target44 SHA256 differs: " + path.string());
      result.mappings_[slot] = std::move(mapped);
      if (is_hc(role) || is_conv(role)) {
        auto &values = result.floats_[slot]; values.resize(s.bytes / 2);
        for (std::size_t i = 0; i < values.size(); ++i) {
          values[i] = widen(raw, i);
          if (!std::isfinite(values[i])) throw std::runtime_error("original target44 widened parameter is nonfinite");
        }
      } else if (s.raw_type == c::DataType::kFloat32) {
        auto &values = result.floats_[slot]; values.resize(s.bytes / 4);
        for (std::size_t i = 0; i < values.size(); ++i) {
          std::uint32_t bits = 0;
          for (std::size_t byte = 0; byte < 4; ++byte)
            bits |= std::to_integer<std::uint32_t>(raw[i * 4 + byte]) << (8 * byte);
          values[i] = std::bit_cast<float>(bits);
          if (!std::isfinite(values[i])) throw std::runtime_error("original target44 F32 parameter is nonfinite");
        }
      } else if (is_norm(role)) {
        auto &values = result.norms_[slot]; values.resize(s.bytes / 2);
        for (std::size_t i = 0; i < values.size(); ++i) {
          values[i] = bfloat(raw, i);
          if (!std::isfinite(widen(raw, i))) throw std::runtime_error("original target44 norm is nonfinite");
        }
      }
    }
    return result;
  }
  std::span<const std::byte> bytes(Role role) const {
    return shared::take(mappings_[index(role)].bytes(), "original target44 raw readonly bytes");
  }
  std::span<const std::byte> payload(Role role) const {
    if (is_hc(role) || is_conv(role)) return std::as_bytes(widened(role));
    return bytes(role);
  }
  std::span<const float> widened(Role role) const {
    if (!is_hc(role) && !is_conv(role))
      throw std::runtime_error("original target44 requested unprepared widening");
    return floats_[index(role)];
  }
  std::span<const float> floats(Role role) const {
    if (spec(role).raw_type != c::DataType::kFloat32 && !is_hc(role) && !is_conv(role))
      throw std::runtime_error("original target44 requested large BF16 matrix as F32");
    return floats_[index(role)];
  }
  std::span<const std::uint16_t> norm(Role role) const {
    if (!is_norm(role)) throw std::runtime_error("original target44 requested matrix as gamma");
    return norms_[index(role)];
  }
 private:
  std::array<m::FileMapping, kRoles> mappings_;
  // These contain ONLY2 HC matrices,3 short convolutions and original small
  // F32 metadata. The nine dense BF16 banks never acquire an F32 shadow.
  std::array<std::vector<float>, kRoles> floats_;
  std::array<std::vector<std::uint16_t>, kRoles> norms_;
  static std::uint16_t bfloat(std::span<const std::byte> raw, std::size_t i) {
    return std::uint16_t(std::to_integer<unsigned>(raw[i * 2]) |
        (std::to_integer<unsigned>(raw[i * 2 + 1]) << 8u));
  }
  static float widen(std::span<const std::byte> raw, std::size_t i) {
    return std::bit_cast<float>(std::uint32_t{bfloat(raw, i)} << 16u);
  }
};
struct Bound final {
  std::shared_ptr<const Data> source;
  std::array<shared::ReadonlyRole, kRoles> roles;
  const shared::ReadonlyRole &at(Role role) const { return roles[index(role)]; }
};
inline Bound bind(std::shared_ptr<const Data> source, const c::StorageOwner &owner) {
  if (!source || !owner) throw std::runtime_error("original target44 needs original data and its actual owner");
  Bound result; result.source = std::move(source);
  for (std::size_t slot = 0; slot < kRoles; ++slot) {
    const auto role = static_cast<Role>(slot);
    const auto desc = description(role);
    const auto raw = result.source->payload(role);
    const auto extent = shared::take(desc.required_bytes(), "original target44 runtime extent");
    if (raw.size() != extent) throw std::runtime_error("original target44 prepared extent differs");
    auto bank = shared::take(m::Buffer::create(owner.ref(), extent, 256), "original target44 immutable bank");
    shared::check(bank.view().write(raw), "original target44 once-only upload");
    result.roles[slot] = {spec(role).name, desc, owner,
        shared::take(owner.adopt_buffer(std::move(bank)), "original target44 readonly lease")};
  }
  return result;
}
inline moe::Data load_moe(const std::filesystem::path &directory) {
  return moe::Data::load(directory, kMoeIndexSha256);
}
} // namespace glm_original_target44_roles
