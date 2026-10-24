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

// Immutable original Pro NextN78 parameters. The 24 layer-owned files become
// 17 runtime banks; seven E4/SF pairs retain their original canonical bytes.
// Embedding, head and NextN shared-head norm use ONE shared-vocabulary source.
// This supplier owns mappings and banks, with no model execution or cache policy.
namespace glm_pro_original_roles {
namespace c = hyle::compute;
namespace m = hyle::mem;
namespace shared = glm_shared_vocabulary;
inline constexpr std::string_view kRepository = "zai-org/GLM-5.3";
inline constexpr std::string_view kRevision = "aca966e4e02791568aa6a4ced368624b3d897f42";
inline constexpr std::int64_t kHidden = 6144, kQueryRank = 2048;
inline constexpr std::int64_t kHeads = 64, kNope = 192, kRope = 64;
inline constexpr std::int64_t kLatent = 512, kValue = 256, kVocabulary = 154880;
inline constexpr float kNormEpsilon = 1.0e-5F, kIndexNormEpsilon = 1.0e-6F;
inline constexpr std::size_t kOriginalRoles = 27, kOwnOriginalBytes = 325680256;
inline constexpr std::uint64_t kOriginalBytes = 4132023424ULL;
inline constexpr shared::Geometry kVocabularyGeometry{kVocabulary, kHidden, kNormEpsilon};
// Only actual range-acquisition receipts may fill these pins. Incomplete pins
// fail before mapping, including the shared norm (not model.norm.weight).
inline constexpr shared::Pins kVocabularyPins{kRepository, kRevision, {"89ee5fc32885854e933a4518560cf6192e1b601f4219e0ffae9b2519bf1738a4", "864f488a0074d236062a4f24800940df7ba611dba76238d4f9f847d9659e11cd", "85b9f1c8edd878dd20da5223167c455ef941232a3af891e9a48af365ffda6ff0"}};
inline constexpr std::string_view kMoeIndexSha256 = "df719ecb84c4e15730b3443cb441ef7947c79e7f1c25eb9bc9b18852d3c9ce85";

enum class RawRole : std::size_t {
  kEh, kEmbeddingNorm, kHiddenNorm, kInputNorm, kPostNorm,
  kIndexBias, kIndexNorm, kIndexWeights, kIndexKeyCodes, kIndexKeyScale,
  kIndexQueryCodes, kIndexQueryScale, kKvNorm, kKvACodes, kKvAScale,
  kKvBCodes, kKvBScale, kOutputCodes, kOutputScale, kQueryNorm,
  kQueryACodes, kQueryAScale, kQueryBCodes, kQueryBScale, kCount
};
enum class Role : std::size_t {
  kEh, kEmbeddingNorm, kHiddenNorm, kInputNorm, kPostNorm,
  kIndexBias, kIndexNorm, kIndexWeights, kIndexKey, kIndexQuery,
  kKvNorm, kKvA, kKvB, kOutput, kQueryNorm, kQueryA, kQueryB, kCount
};
inline constexpr std::size_t kRawRoles = static_cast<std::size_t>(RawRole::kCount);
inline constexpr std::size_t kRoles = static_cast<std::size_t>(Role::kCount);
struct Spec final {
  std::string_view name, file, sha256;
  c::DataType type;
  std::int64_t rows, columns; // columns0 identifies a vector.
};
inline constexpr std::array<Spec, kRawRoles> kSpecs{
    Spec{"model.layers.78.eh_proj.weight", "eh_proj.weight.bf16", "43d12184a3d54615723db7352e7141943f3935698e5b902e1bb41bbb58fd94b1", c::DataType::kBFloat16, 6144, 12288},
    Spec{"model.layers.78.enorm.weight", "enorm.weight.bf16", "088679357dac75c7ef240e6ca24e3af9d1a167629447e8dc75fb69f53a93b802", c::DataType::kBFloat16, 6144, 0},
    Spec{"model.layers.78.hnorm.weight", "hnorm.weight.bf16", "cadc2c6238a40bd72ac75a08b33397e4627d762d080cd984f7c99607e99d1ebc", c::DataType::kBFloat16, 6144, 0},
    Spec{"model.layers.78.input_layernorm.weight", "input_layernorm.weight.bf16", "f7d30c42379a7d5f281600a0b09ce7b282c469a56f297c801cb211aab189c8bf", c::DataType::kBFloat16, 6144, 0},
    Spec{"model.layers.78.post_attention_layernorm.weight", "post_attention_layernorm.weight.bf16", "39d71ad2f414dfa582619c411711e671478cde4fc389b4f55d93ca8667fe6d77", c::DataType::kBFloat16, 6144, 0},
    Spec{"model.layers.78.self_attn.indexer.k_norm.bias", "indexer.k_norm.bias.bf16", "18df18fb52306a3bbeebabf919e928a5082b86dd28aca72f332b73d910ad3640", c::DataType::kBFloat16, 128, 0},
    Spec{"model.layers.78.self_attn.indexer.k_norm.weight", "indexer.k_norm.weight.bf16", "cd7bb7b38b9c2a58604699cd7b2aa6b48ab13727a08e6dd6cec504413ecbd3e1", c::DataType::kBFloat16, 128, 0},
    Spec{"model.layers.78.self_attn.indexer.weights_proj.weight", "indexer.weights_proj.weight.bf16", "5fc3b62fb4b33eb9bd5ff38336f630a7766e75f51f9b9c0c74a1009243a25d59", c::DataType::kBFloat16, 32, 6144},
    Spec{"model.layers.78.self_attn.indexer.wk.weight", "indexer.wk.weight.e4m3", "70a62e0a7c87caeaa727ccd5e46cf7fa9a1b0e678b825171b6c686abdc9f8f1f", c::DataType::kFloat8E4M3FN, 128, 6144},
    Spec{"model.layers.78.self_attn.indexer.wk.weight_scale_inv", "indexer.wk.weight_scale.f32", "7283fc1ea4495294a09e2406392ad8b337d27c708f9f5ed1b93cae76ed1b7e6d", c::DataType::kFloat32, 1, 48},
    Spec{"model.layers.78.self_attn.indexer.wq_b.weight", "indexer.wq_b.weight.e4m3", "a6619cb7fe8b8c88e0f6ba5fd4a68fefa697d6423dc0f7ecfdb3d9abc4fea76a", c::DataType::kFloat8E4M3FN, 4096, 2048},
    Spec{"model.layers.78.self_attn.indexer.wq_b.weight_scale_inv", "indexer.wq_b.weight_scale.f32", "4337e9f1d3bcc1da11e3b69c4e4a7dbad8056899160b3a92a0c0c77311b8f0a9", c::DataType::kFloat32, 32, 16},
    Spec{"model.layers.78.self_attn.kv_a_layernorm.weight", "kv_a_layernorm.weight.bf16", "5fe4dad946f8157554dd4c46db17ef500a5b0930910e1d831eab84c707e4ab04", c::DataType::kBFloat16, 512, 0},
    Spec{"model.layers.78.self_attn.kv_a_proj_with_mqa.weight", "kv_a.weight.e4m3", "54f460b593c665e517a3e6f97275a720ce9939cdc18e835a21673bf2abd92da9", c::DataType::kFloat8E4M3FN, 576, 6144},
    Spec{"model.layers.78.self_attn.kv_a_proj_with_mqa.weight_scale_inv", "kv_a.weight_scale.f32", "f321ebd055afdb602c5d4a135c90e2d3d805ae807716d842ba8a7f306b458519", c::DataType::kFloat32, 5, 48},
    Spec{"model.layers.78.self_attn.kv_b_proj.weight", "kv_b.weight.e4m3", "b800c66731cf27e9d6f4251423d6851585a8e7a28561217136edf4dbce66bca2", c::DataType::kFloat8E4M3FN, 28672, 512},
    Spec{"model.layers.78.self_attn.kv_b_proj.weight_scale_inv", "kv_b.weight_scale.f32", "25e54d71ffa86f1614919be6a54e65ec8f0cedd5a6ebc00ae938d738a0fb3cbb", c::DataType::kFloat32, 224, 4},
    Spec{"model.layers.78.self_attn.o_proj.weight", "o.weight.e4m3", "07d1af4486c5f2bac32222b9ede7498178880e429b6772f2e7b9a7e983a9624d", c::DataType::kFloat8E4M3FN, 6144, 16384},
    Spec{"model.layers.78.self_attn.o_proj.weight_scale_inv", "o.weight_scale.f32", "dbd015bed9bde749924fead0561b3ab1cbda8ba42fba94bac93ec8758f1aaa44", c::DataType::kFloat32, 48, 128},
    Spec{"model.layers.78.self_attn.q_a_layernorm.weight", "q_a_layernorm.weight.bf16", "e5edd4384e2e8ba04f3cff0e8d2ebd2a78abf8f67fe1403427da30ff57052f82", c::DataType::kBFloat16, 2048, 0},
    Spec{"model.layers.78.self_attn.q_a_proj.weight", "q_a.weight.e4m3", "bc08e64b4aac2ad84d6694b5b3ee9c7670a761d75c3aaebccf73652564b3a78a", c::DataType::kFloat8E4M3FN, 2048, 6144},
    Spec{"model.layers.78.self_attn.q_a_proj.weight_scale_inv", "q_a.weight_scale.f32", "8c78f42a8f02b4cff4efca454b20c2d5f1a0f8a23915be0280380e5985c2e80e", c::DataType::kFloat32, 16, 48},
    Spec{"model.layers.78.self_attn.q_b_proj.weight", "q_b.weight.e4m3", "fe7709ea0f9be2d874f34d507c1278f919971575ab8fbff2fcffc58a302d7077", c::DataType::kFloat8E4M3FN, 16384, 2048},
    Spec{"model.layers.78.self_attn.q_b_proj.weight_scale_inv", "q_b.weight_scale.f32", "1fadf5053299764db8561b4ec1e1ecce4a662413841c0049c38d6f027104db6d", c::DataType::kFloat32, 128, 16}
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
inline constexpr std::array<RawRole, kRoles> kRawForRole{
    RawRole::kEh, RawRole::kEmbeddingNorm, RawRole::kHiddenNorm,
    RawRole::kInputNorm, RawRole::kPostNorm, RawRole::kIndexBias,
    RawRole::kIndexNorm, RawRole::kIndexWeights, RawRole::kIndexKeyCodes,
    RawRole::kIndexQueryCodes, RawRole::kKvNorm, RawRole::kKvACodes,
    RawRole::kKvBCodes, RawRole::kOutputCodes, RawRole::kQueryNorm,
    RawRole::kQueryACodes, RawRole::kQueryBCodes};
inline RawRole raw_role(Role role) { return kRawForRole[index(role)]; }
inline bool is_matrix(Role role) {
  return spec(raw_role(role)).type == c::DataType::kFloat8E4M3FN;
}
inline bool is_layer_norm(Role role) noexcept {
  return role == Role::kIndexBias || role == Role::kIndexNorm;
}
inline c::TensorDesc raw_description(RawRole role) {
  const auto &s = spec(role);
  if (s.columns == 0)
    return shared::take(c::TensorDesc::dense(s.type, std::array{s.rows}), "original Pro vector");
  return shared::take(c::TensorDesc::dense(s.type, std::array{s.rows, s.columns}), "original Pro matrix");
}
inline c::TensorDesc description(Role role) {
  const auto &s = spec(raw_role(role));
  if (is_matrix(role))
    return shared::take(c::float8_weight_description(s.columns, s.rows,
        c::Float8ScaleMode::kFloat32Block128), "original Pro canonical E4");
  // Official CUDA LayerNorm loads these BF16 coefficients into F32 parameters.
  if (is_layer_norm(role))
    return shared::take(c::TensorDesc::dense(c::DataType::kFloat32, std::array{s.rows}), "original Pro index LayerNorm");
  return raw_description(raw_role(role));
}
struct Matrix final {
  std::span<const std::byte> codes;
  std::span<const float> scales;
  std::int64_t outputs, reduction;
};
class Data final {
 public:
  Data() = default;
  Data(Data&&) noexcept = default;
  // Retire a live map before releasing its Buffer; memberwise assignment
  // would replace the allocation first.
  Data& operator=(Data&&) = delete;
  shared::Data vocabulary;
  static Data load(const std::filesystem::path &directory) {
    for (const auto &s : kSpecs)
      if (s.sha256.size() != 64) throw std::runtime_error("original Pro static pins are incomplete");
    for (const auto pin : kVocabularyPins.sha256)
      if (pin.size() != 64) throw std::runtime_error("original Pro vocabulary pins are incomplete");
    Data result;
    for (std::size_t slot = 0; slot < kRawRoles; ++slot) {
      const auto role = static_cast<RawRole>(slot);
      const auto &s = kSpecs[slot]; const auto path = directory / std::string(s.file);
      auto mapped = shared::take(m::FileMapping::open(path.string().c_str()), "map original Pro parameter");
      const auto extent = shared::take(raw_description(role).required_bytes(), "original Pro raw extent");
      if (mapped.size() != extent) throw std::runtime_error("original Pro role size differs: " + path.string());
      const auto raw = shared::take(mapped.bytes(), "original Pro mapped bytes");
      if (shared::detail::mapped_sha256(raw) != s.sha256)
        throw std::runtime_error("original Pro role SHA256 differs: " + path.string());
      result.mappings_[slot] = std::move(mapped);
    }
    result.vocabulary = shared::Data::load(directory, kVocabularyGeometry, kVocabularyPins);
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
          if (!std::isfinite(value)) throw std::runtime_error("original Pro norm/bias is nonfinite");
          if (is_layer_norm(role)) result.widened_[slot].push_back(value);
        }
      }
    }
    return result;
  }
  std::span<const std::byte> bytes(RawRole role) const {
    return shared::take(mappings_[index(role)].bytes(), "original Pro borrowed raw span");
  }
  std::span<const std::byte> payload(Role role) const {
    if (is_matrix(role)) return canonical(role);
    if (is_layer_norm(role)) return std::as_bytes(std::span(widened_[index(role)]));
    return bytes(raw_role(role));
  }
  std::span<const std::uint16_t> norm(Role role) const {
    if (spec(raw_role(role)).columns != 0)
      throw std::runtime_error("original Pro requested non-vector BF16 norm");
    return norms_[index(role)];
  }
  std::span<const float> floats(Role role) const {
    if (!is_layer_norm(role)) throw std::runtime_error("original Pro only index LayerNorm has widened coefficients");
    return widened_[index(role)];
  }
  Matrix matrix(Role role) const {
    if (!is_matrix(role)) throw std::runtime_error("original Pro requested non-FP8 matrix");
    const auto raw_id = raw_role(role); const auto &s = spec(raw_id);
    return {bytes(raw_id), scales_[index(role)], s.rows, s.columns};
  }
 private:
  // Returned spans borrow these strong owners. No large BF16/F32 shadow banks.
  std::array<m::FileMapping, kRawRoles> mappings_;
  std::array<m::Buffer, kRoles> canonical_;
  std::array<m::HostMapping, kRoles> canonical_maps_;
  std::array<std::vector<float>, kRoles> scales_, widened_;
  std::array<std::vector<std::uint16_t>, kRoles> norms_;
  static std::uint32_t read_le(std::span<const std::byte> raw, std::size_t offset, std::size_t count) {
    std::uint32_t result = 0;
    for (std::size_t byte = 0; byte < count; ++byte)
      result |= std::to_integer<std::uint32_t>(raw[offset + byte]) << (8u * unsigned(byte));
    return result;
  }
  std::span<const std::byte> canonical(Role role) const {
    return canonical_maps_[index(role)].bytes();
  }
  void prepare(Role role) {
    const auto codes_id = raw_role(role);
    const auto scales_id = static_cast<RawRole>(index(codes_id) + 1);
    const auto raw_codes = bytes(codes_id), raw_scales = bytes(scales_id);
    for (const auto code : raw_codes)
      if ((std::to_integer<unsigned>(code) & 0x7fu) == 0x7fu)
        throw std::runtime_error("original Pro E4 parameter is nonfinite");
    auto &scale_values = scales_[index(role)]; scale_values.resize(raw_scales.size() / 4);
    for (std::size_t i = 0; i < scale_values.size(); ++i) {
      const auto value = std::bit_cast<float>(read_le(raw_scales, i * 4, 4));
      if (!std::isfinite(value) || !(value > 0)) throw std::runtime_error("original Pro scale is not positive finite");
      scale_values[i] = value;
    }
    const auto desc = description(role);
    canonical_[index(role)] = shared::take(m::Buffer::create(m::system_resource(),
        shared::take(desc.required_bytes(), "original Pro canonical extent"), 64), "original Pro canonical bank");
    const auto source = [&](RawRole raw_id) {
      return shared::take(c::ConstTensorView::create(raw_description(raw_id),
          shared::take(mappings_[index(raw_id)].view(), "original Pro import readonly view")), "original Pro import tensor");
    };
    auto target = shared::take(c::TensorView::create(desc, canonical_[index(role)].view()), "original Pro import target");
    shared::check(c::import_float8_block128_weight(source(codes_id), source(scales_id), target), "original Pro canonical import");
    canonical_maps_[index(role)] = shared::take(
        canonical_[index(role)].view().map_host(), "original Pro canonical span");
    const auto packed = canonical(role);
    if (packed.size() != raw_codes.size() + raw_scales.size() ||
        !std::equal(raw_codes.begin(), raw_codes.end(), packed.begin()) ||
        !std::equal(raw_scales.begin(), raw_scales.end(), packed.begin() + static_cast<std::ptrdiff_t>(raw_codes.size())))
      throw std::runtime_error("original Pro import changed original E4/SF bytes");
  }
};
struct Bound final {
  std::shared_ptr<const Data> source;
  std::array<shared::ReadonlyRole, kRoles> roles;
  const shared::ReadonlyRole &at(Role role) const { return roles[index(role)]; }
};
inline Bound bind(std::shared_ptr<const Data> source, const c::StorageOwner &owner) {
  if (!source || !owner) throw std::runtime_error("original Pro banks need source and actual Graph owner");
  Bound result; result.source = std::move(source);
  for (std::size_t slot = 0; slot < kRoles; ++slot) {
    const auto role = static_cast<Role>(slot); const auto desc = description(role);
    const auto extent = shared::take(desc.required_bytes(), "original Pro ready extent");
    const auto raw = result.source->payload(role);
    if (raw.size() != extent) throw std::runtime_error("original Pro prepared byte extent differs");
    auto bank = shared::take(m::Buffer::create(owner.ref(), extent, 256), "original Pro ready immutable bank");
    shared::check(bank.view().write(raw), "original Pro once-only upload");
    result.roles[slot] = {spec(raw_role(role)).name, desc, owner,
        shared::take(owner.adopt_buffer(std::move(bank)), "original Pro ready readonly lease")};
  }
  // Bind the aliasing shared_ptr(data, &data->vocabulary) separately once.
  // The original KV_b ready lease may retire after the shared cold K/V split.
  return result;
}
} // namespace glm_pro_original_roles
