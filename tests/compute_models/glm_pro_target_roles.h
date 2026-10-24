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

#include "glm_shared_vocabulary.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/weight_import.h"

// Distinct original Pro target77 shared-index MLA parameters and model.norm.
// There are no EH/shared_head/indexer roles in target77. The original target
// MoE uses its own authenticated catalogue with the same Pro geometry.
// This source loads only 15 static roles; it never executes a model.
namespace glm_pro_target_roles {
namespace c = hyle::compute;
namespace m = hyle::mem;
namespace shared = glm_shared_vocabulary;
inline constexpr std::string_view kRepository = "zai-org/GLM-5.3";
inline constexpr std::string_view kRevision = "aca966e4e02791568aa6a4ced368624b3d897f42";
inline constexpr std::int64_t kHidden = 6144, kQueryRank = 2048;
inline constexpr std::int64_t kHeads = 64, kNope = 192, kRope = 64;
inline constexpr std::int64_t kLatent = 512, kValue = 256, kVocabulary = 154880;
inline constexpr float kNormEpsilon = 1.0e-5F, kIndexNormEpsilon = 1.0e-6F;
inline constexpr std::size_t kRawRoles = 15, kRoles = 10;
inline constexpr std::uint64_t kOriginalBytes = 165102016ULL;
inline constexpr std::string_view kMoeIndexSha256 = "37cb5928d417784b49ce6c5765c3859dbc7f3a7e56a2a347bb5041d98c03d0e7";
enum class RawRole : std::size_t {kInputNorm,kPostNorm,kKvNorm,kKvACodes,kKvAScale,kKvBCodes,kKvBScale,kOutputCodes,kOutputScale,kQueryNorm,kQueryACodes,kQueryAScale,kQueryBCodes,kQueryBScale,kTargetNorm,kCount};
enum class Role : std::size_t {kInputNorm,kPostNorm,kKvNorm,kKvA,kKvB,kOutput,kQueryNorm,kQueryA,kQueryB,kTargetNorm,kCount};
struct Spec final {
  std::string_view name, file, sha256;
  c::DataType type;
  std::int64_t rows, columns; // columns0 identifies a vector.
};
inline constexpr std::array<Spec,kRawRoles> kSpecs{
    Spec{"model.layers.77.input_layernorm.weight","input_layernorm.weight.bf16","e67eac2bc0c65f5c31f5f7626a9f1d4eaae5b672cc16aec1664796166bde57ff",c::DataType::kBFloat16,6144,0},
    Spec{"model.layers.77.post_attention_layernorm.weight","post_attention_layernorm.weight.bf16","1a7169c605279de527e6c2de8c3b2bb06f0834decaff77da696716ff387d0d90",c::DataType::kBFloat16,6144,0},
    Spec{"model.layers.77.self_attn.kv_a_layernorm.weight","kv_a_layernorm.weight.bf16","4741fd33b499ff91f03a05e93007a51eaade626e4e8936caf716fe1427a17486",c::DataType::kBFloat16,512,0},
    Spec{"model.layers.77.self_attn.kv_a_proj_with_mqa.weight","kv_a.weight.e4m3","d6d6761841e0294e2ef7249e110b37ef84b1fd5ad73eb7c9573a755a2b08636d",c::DataType::kFloat8E4M3FN,576,6144},
    Spec{"model.layers.77.self_attn.kv_a_proj_with_mqa.weight_scale_inv","kv_a.weight_scale.f32","895580333d767b5e9eace2484ef7f46e028c788948e89187d896e19e00f5cf2f",c::DataType::kFloat32,5,48},
    Spec{"model.layers.77.self_attn.kv_b_proj.weight","kv_b.weight.e4m3","e4085d30e465396442418a5283c5a9ae3eb040313335c2aae16f7c88438eb53b",c::DataType::kFloat8E4M3FN,28672,512},
    Spec{"model.layers.77.self_attn.kv_b_proj.weight_scale_inv","kv_b.weight_scale.f32","ecbeadb8d308e31b3b33650ae8fdbd5151b068ff44b17b5616fef85d15a49303",c::DataType::kFloat32,224,4},
    Spec{"model.layers.77.self_attn.o_proj.weight","o.weight.e4m3","56024b8288cc9b00f2c924cf68b2bb39b499f35d2702d9141030ba8ee954ab9d",c::DataType::kFloat8E4M3FN,6144,16384},
    Spec{"model.layers.77.self_attn.o_proj.weight_scale_inv","o.weight_scale.f32","3414e514ae93773e332ccac54cd833dec507187c6c6a736e4a364448a8dcedb5",c::DataType::kFloat32,48,128},
    Spec{"model.layers.77.self_attn.q_a_layernorm.weight","q_a_layernorm.weight.bf16","b1533e7a1a9b197bb06f24dd175b66429c78b44e8ab21de383224edddb259be7",c::DataType::kBFloat16,2048,0},
    Spec{"model.layers.77.self_attn.q_a_proj.weight","q_a.weight.e4m3","a4a79f6d0bc7da4872cbe71702b7ec3de2e4e4b90690468f1e36dad830baf9d2",c::DataType::kFloat8E4M3FN,2048,6144},
    Spec{"model.layers.77.self_attn.q_a_proj.weight_scale_inv","q_a.weight_scale.f32","eb34c809654dcc118602c8ceec46972278b0f98a5b84ab290c2eb999e7620207",c::DataType::kFloat32,16,48},
    Spec{"model.layers.77.self_attn.q_b_proj.weight","q_b.weight.e4m3","05351fddffd70877f391d6a870fa76146b186d067f9694c3f38979b6572cec9b",c::DataType::kFloat8E4M3FN,16384,2048},
    Spec{"model.layers.77.self_attn.q_b_proj.weight_scale_inv","q_b.weight_scale.f32","bb08194ef1a4582a3908802d8c1686ffbda1de73e35b7d19035bc8e02fe2124a",c::DataType::kFloat32,128,16},
    Spec{"model.norm.weight","target_norm.weight.bf16","4ecafb80e014a27ed61983b299f6c6344cb6a67d55c3b12259ddefade6235c6a",c::DataType::kBFloat16,6144,0}
};
inline std::size_t index(RawRole role) {
  const auto slot = static_cast<std::size_t>(role);
  if (slot >= kRawRoles) throw std::runtime_error("unknown original GLM Pro raw role");
  return slot;
}
inline std::size_t index(Role role) {
  const auto slot = static_cast<std::size_t>(role);
  if (slot >= kRoles) throw std::runtime_error("unknown original GLM Pro runtime role");
  return slot;
}
inline const Spec &spec(RawRole role) { return kSpecs[index(role)]; }
inline constexpr std::array<RawRole,kRoles> kRawForRole{RawRole::kInputNorm,RawRole::kPostNorm,RawRole::kKvNorm,RawRole::kKvACodes,RawRole::kKvBCodes,RawRole::kOutputCodes,RawRole::kQueryNorm,RawRole::kQueryACodes,RawRole::kQueryBCodes,RawRole::kTargetNorm};
inline RawRole raw_role(Role role) { return kRawForRole[index(role)]; }
inline bool is_matrix(Role role) {
  return spec(raw_role(role)).type == c::DataType::kFloat8E4M3FN;
}
inline c::TensorDesc raw_description(RawRole role) {
  const auto &s = spec(role);
  if (s.columns == 0)
    return shared::take(c::TensorDesc::dense(s.type, std::array{s.rows}), "original Pro target77 vector");
  return shared::take(c::TensorDesc::dense(s.type, std::array{s.rows, s.columns}), "original Pro target77 matrix");
}
inline c::TensorDesc description(Role role) {
  const auto &s = spec(raw_role(role));
  if (is_matrix(role))
    return shared::take(c::float8_weight_description(s.columns, s.rows,
        c::Float8ScaleMode::kFloat32Block128), "original Pro target77 canonical E4");
  return raw_description(raw_role(role));
}
struct Matrix final {
  std::span<const std::byte> codes;
  std::span<const float> scales;
  std::int64_t outputs, reduction;
};
class Data final {
 public:
  Data()=default;
  Data(Data&&) noexcept=default;
  Data& operator=(Data&&)=delete;
  static Data load(const std::filesystem::path &directory) {
    for (const auto &s : kSpecs)
      if (s.sha256.size() != 64) throw std::runtime_error("original Pro target77 static pins are incomplete");
    Data result;
    for (std::size_t slot = 0; slot < kRawRoles; ++slot) {
      const auto role = static_cast<RawRole>(slot);
      const auto &s = kSpecs[slot]; const auto path = directory / std::string(s.file);
      auto mapped = shared::take(m::FileMapping::open(path.string().c_str()), "map original Pro target77 parameter");
      const auto extent = shared::take(raw_description(role).required_bytes(), "original Pro target77 raw extent");
      if (mapped.size() != extent) throw std::runtime_error("original Pro target77 role size differs: " + path.string());
      const auto raw = shared::take(mapped.bytes(), "original Pro target77 mapped bytes");
      if (shared::detail::mapped_sha256(raw) != s.sha256)
        throw std::runtime_error("original Pro target77 role SHA256 differs: " + path.string());
      result.mappings_[slot] = std::move(mapped);
    }
    for (std::size_t slot = 0; slot < kRoles; ++slot) {
      const auto role = static_cast<Role>(slot); const auto raw_id = raw_role(role);
      const auto &s = spec(raw_id);
      if (is_matrix(role)) {
        result.prepare(role);
      } else if (s.columns == 0) {
        const auto raw = result.bytes(raw_id);
        auto &small = result.norms_[slot]; small.resize(raw.size() / 2);
        for (std::size_t i = 0; i < small.size(); ++i) {
          small[i] = static_cast<std::uint16_t>(read_le(raw, i * 2, 2));
          const auto value = std::bit_cast<float>(std::uint32_t{small[i]} << 16u);
          if (!std::isfinite(value)) throw std::runtime_error("original Pro target77 norm/bias is nonfinite");
        }
      }
    }
    return result;
  }
  std::span<const std::byte> bytes(RawRole role) const {
    return shared::take(mappings_[index(role)].bytes(), "original Pro target77 borrowed raw span");
  }
  std::span<const std::byte> payload(Role role) const {
    if (is_matrix(role)) return canonical(role);
    return bytes(raw_role(role));
  }
  std::span<const std::uint16_t> norm(Role role) const {
    if (spec(raw_role(role)).columns != 0)
      throw std::runtime_error("original Pro target77 requested non-vector BF16 norm");
    return norms_[index(role)];
  }
  Matrix matrix(Role role) const {
    if (!is_matrix(role)) throw std::runtime_error("original Pro target77 requested non-FP8 matrix");
    const auto raw_id = raw_role(role); const auto &s = spec(raw_id);
    return {bytes(raw_id), scales_[index(role)], s.rows, s.columns};
  }
 private:
  // Returned spans borrow these strong owners. No large BF16/F32 shadow banks.
  std::array<m::FileMapping, kRawRoles> mappings_;
  std::array<m::Buffer, kRoles> canonical_;
  // Maps outlive every borrowed canonical span and retire before buffers.
  std::array<m::HostMapping,kRoles> canonical_maps_;
  std::array<std::vector<float>, kRoles> scales_;
  std::array<std::vector<std::uint16_t>, kRoles> norms_;
  static std::uint32_t read_le(std::span<const std::byte> raw, std::size_t offset, std::size_t count) {
    std::uint32_t result = 0;
    for (std::size_t byte = 0; byte < count; ++byte)
      result |= std::to_integer<std::uint32_t>(raw[offset + byte]) << (8u * unsigned(byte));
    return result;
  }
  std::span<const std::byte> canonical(Role role) const {
    const auto &mapped=canonical_maps_[index(role)];
    if(!mapped)throw std::runtime_error("original Pro target77 target canonical mapping is absent");
    return mapped.bytes();
  }
  void prepare(Role role) {
    const auto codes_id = raw_role(role);
    const auto scales_id = static_cast<RawRole>(index(codes_id) + 1);
    const auto raw_codes = bytes(codes_id), raw_scales = bytes(scales_id);
    for (const auto code : raw_codes)
      if ((std::to_integer<unsigned>(code) & 0x7fu) == 0x7fu)
        throw std::runtime_error("original Pro target77 E4 parameter is nonfinite");
    auto &scale_values = scales_[index(role)]; scale_values.resize(raw_scales.size() / 4);
    for (std::size_t i = 0; i < scale_values.size(); ++i) {
      const auto value = std::bit_cast<float>(read_le(raw_scales, i * 4, 4));
      if (!std::isfinite(value) || !(value > 0)) throw std::runtime_error("original Pro target77 scale is not positive finite");
      scale_values[i] = value;
    }
    const auto desc = description(role);
    canonical_[index(role)] = shared::take(m::Buffer::create(m::system_resource(),
        shared::take(desc.required_bytes(), "original Pro target77 canonical extent"), 64), "original Pro target77 canonical bank");
    const auto source = [&](RawRole raw_id) {
      return shared::take(c::ConstTensorView::create(raw_description(raw_id),
          shared::take(mappings_[index(raw_id)].view(), "original Pro target77 import readonly view")), "original Pro target77 import tensor");
    };
    auto target = shared::take(c::TensorView::create(desc, canonical_[index(role)].view()), "original Pro target77 import target");
    shared::check(c::import_float8_block128_weight(source(codes_id), source(scales_id), target), "original Pro target77 canonical import");
    canonical_maps_[index(role)]=shared::take(std::as_const(canonical_[index(role)]).view().map_host(),
        "retain original target canonical read map");
    const auto packed = canonical(role);
    if (packed.size() != raw_codes.size() + raw_scales.size() ||
        !std::equal(raw_codes.begin(), raw_codes.end(), packed.begin()) ||
        !std::equal(raw_scales.begin(), raw_scales.end(), packed.begin() + static_cast<std::ptrdiff_t>(raw_codes.size())))
      throw std::runtime_error("original Pro target77 import changed original E4/SF bytes");
  }
};
struct Bound final {
  std::shared_ptr<const Data> source;
  std::array<shared::ReadonlyRole, kRoles> roles;
  const shared::ReadonlyRole &at(Role role) const { return roles[index(role)]; }
};
inline Bound bind(std::shared_ptr<const Data> source, const c::StorageOwner &owner) {
  if (!source || !owner) throw std::runtime_error("original Pro target77 banks need source and actual Graph owner");
  Bound result; result.source = std::move(source);
  for (std::size_t slot = 0; slot < kRoles; ++slot) {
    const auto role = static_cast<Role>(slot); const auto desc = description(role);
    const auto extent = shared::take(desc.required_bytes(), "original Pro target77 ready extent");
    const auto raw = result.source->payload(role);
    if (raw.size() != extent) throw std::runtime_error("original Pro target77 prepared byte extent differs");
    auto bank = shared::take(m::Buffer::create(owner.ref(), extent, 256), "original Pro target77 ready immutable bank");
    shared::check(bank.view().write(raw), "original Pro target77 once-only upload");
    result.roles[slot] = {spec(raw_role(role)).name, desc, owner,
        shared::take(owner.adopt_buffer(std::move(bank)), "original Pro target77 ready readonly lease")};
  }
  // The original KV_b ready lease may retire after the shared cold K/V split.
  return result;
}
} // namespace glm_pro_target_roles
