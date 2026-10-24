#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "glm_shared_vocabulary.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/weight_import.h"

// Exact official Flash MLA attention supplier, separate from Graph math.
// The default pins identify NextN45; another actual layer supplies its own
// complete authenticated pin array and retains the same geometry contract.
// Nine original bodies become six immutable runtime banks: three original
// E4/F32 block128 pairs use the existing canonical importer; kv_b remains ONE
// original BF16 bank. K/V absorption uses views of that bank, never unrelated
// generated matrices or a full F32 weight expansion.
namespace glm_original_attention_roles {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
namespace shared = glm_shared_vocabulary;
inline constexpr std::string_view kRepository = shared::kRepository;
inline constexpr std::string_view kRevision = shared::kRevision;
inline constexpr std::int64_t kHidden = 4096, kQueryRank = 1536;
inline constexpr std::int64_t kHeads = 64, kNope = 256, kLatent = 512, kValue = 256;
inline constexpr float kEpsilon = 1.0e-5F, kAttentionScale = 1.0F / 16.0F;
inline constexpr std::size_t kOriginalBytes = 127958528;

enum class RawRole : std::size_t {
  kQueryCodes, kQueryScale, kKvCodes, kKvScale, kKvNorm, kKvB,
  kOutputCodes, kOutputScale, kPostNorm, kCount
};
inline constexpr std::size_t kRawRoles = static_cast<std::size_t>(RawRole::kCount);
enum class Parameter : std::size_t {
  kQueryProjection, kKvProjection, kKvNorm, kKvB, kOutput, kPostNorm, kCount
};
inline constexpr std::size_t kParameters = static_cast<std::size_t>(Parameter::kCount);
struct RawSpec final {
  std::string_view name, file, sha256;
  c::DataType type;
  std::int64_t rows, columns; // columns0 is an original one-dimensional norm.
};
inline constexpr std::array<RawSpec, kRawRoles> kSpecs{
    RawSpec{"self_attn.q_b_proj.weight", "q_b.weight.e4m3", "c5b33a523ec8aa477e0a1a3f02049ed8f122c83e70dc77b9777741cd076a79d9",
        c::DataType::kFloat8E4M3FN, 16384, 1536},
    RawSpec{"self_attn.q_b_proj.weight_scale_inv", "q_b.weight_scale.f32", "58f7973cd4f6cd70dc2e05ebfc307dd2c0998f41ea12e84267590f1f443463a1",
        c::DataType::kFloat32, 128, 12},
    RawSpec{"self_attn.kv_a_proj_with_mqa.weight", "kv_a.weight.e4m3", "c1b6d576b22544c67a0a5d99c3cfdb0a70a56a38e96d68970d15490074c8ceb4",
        c::DataType::kFloat8E4M3FN, 512, 4096},
    RawSpec{"self_attn.kv_a_proj_with_mqa.weight_scale_inv", "kv_a.weight_scale.f32", "6a262630b028697ea60d85d7da1042de2b8b2e4e3f37b52246b17798bc6ede8d",
        c::DataType::kFloat32, 4, 32},
    RawSpec{"self_attn.kv_a_layernorm.weight", "kv_a_layernorm.weight.bf16", "937e8ba89741a026ddefd5d2db42d3066aab723ab43995b0ed3b53bf4c3811e7",
        c::DataType::kBFloat16, 512, 0},
    RawSpec{"self_attn.kv_b_proj.weight", "kv_b.weight.bf16", "2c9d7ecb2b3863a52d5b27bc113b7241471a5d955f5046557b24b7e3b8289dc9",
        c::DataType::kBFloat16, 32768, 512},
    RawSpec{"self_attn.o_proj.weight", "o.weight.e4m3", "c6402d1f100e7e99776a97bd1ce33618f960e98c63f56c4ddd0ced8cc494c2dd",
        c::DataType::kFloat8E4M3FN, 4096, 16384},
    RawSpec{"self_attn.o_proj.weight_scale_inv", "o.weight_scale.f32", "346ca13b45aca744256624a84a5d39740d10be9f2ba31714bd8daf8378b09237",
        c::DataType::kFloat32, 32, 128},
    RawSpec{"post_attention_layernorm.weight", "post_attention_layernorm.weight.bf16", "85d70bee15e028083a5a38ab63fbaabb6fabafa64d9a66d9a61ab4604c2a52dc",
        c::DataType::kBFloat16, 4096, 0}
};
using Pins = std::array<std::string_view, kRawRoles>;
inline constexpr Pins kDefaultSha256 = [] {
  Pins pins{};
  for (std::size_t slot = 0; slot < kRawRoles; ++slot) pins[slot] = kSpecs[slot].sha256;
  return pins;
}();
inline constexpr std::array<std::string_view, kParameters> kNames{
    "self_attn.q_b_proj", "self_attn.kv_a_proj_with_mqa", "self_attn.kv_a_layernorm.weight",
    "self_attn.kv_b_proj.weight", "self_attn.o_proj", "post_attention_layernorm.weight"};
inline std::size_t index(RawRole role) {
  const auto value = static_cast<std::size_t>(role);
  if (value >= kRawRoles) throw std::runtime_error("unknown original GLM attention raw role");
  return value;
}
inline std::size_t index(Parameter parameter) {
  const auto value = static_cast<std::size_t>(parameter);
  if (value >= kParameters) throw std::runtime_error("unknown original GLM attention parameter");
  return value;
}
inline c::TensorDesc raw_description(RawRole role) {
  const auto &spec = kSpecs[index(role)];
  if (spec.columns == 0)
    return shared::take(c::TensorDesc::dense(spec.type, std::array{spec.rows}), "original attention norm type");
  return shared::take(c::TensorDesc::dense(spec.type, std::array{spec.rows, spec.columns}), "original attention raw type");
}
inline c::TensorDesc description(Parameter parameter) {
  switch (parameter) {
  case Parameter::kQueryProjection:
    return shared::take(c::float8_weight_description(kQueryRank, kHeads * kNope,
        c::Float8ScaleMode::kFloat32Block128), "original q_b packed type");
  case Parameter::kKvProjection:
    return shared::take(c::float8_weight_description(kHidden, kLatent,
        c::Float8ScaleMode::kFloat32Block128), "original kv_a packed type");
  case Parameter::kOutput:
    return shared::take(c::float8_weight_description(kHeads * kValue, kHidden,
        c::Float8ScaleMode::kFloat32Block128), "original o packed type");
  case Parameter::kKvNorm: return raw_description(RawRole::kKvNorm);
  case Parameter::kKvB: return raw_description(RawRole::kKvB);
  case Parameter::kPostNorm: return raw_description(RawRole::kPostNorm);
  default: throw std::runtime_error("unknown original GLM attention description");
  }
}

class Data final {
 public:
  Data()=default;
  Data(Data&&)=default;
  Data &operator=(Data &&other) noexcept {
    if(this!=&other) {
      std::swap(mappings_,other.mappings_);
      std::swap(canonical_,other.canonical_);
      std::swap(canonical_maps_,other.canonical_maps_);
    }
    return *this;
  }
  static Data load(const std::filesystem::path &directory, const Pins &pins = kDefaultSha256) {
    for (const auto pin : pins)
      if (pin.size() != 64) throw std::runtime_error("original GLM attention requires complete actual-layer SHA256 pins");
    Data result;
    for (std::size_t slot = 0; slot < kRawRoles; ++slot) {
      const auto role = static_cast<RawRole>(slot);
      const auto &spec = kSpecs[slot];
      const auto path = directory / std::string(spec.file);
      auto mapped = shared::take(m::FileMapping::open(path.string().c_str()), "map original GLM attention");
      const auto extent = shared::take(raw_description(role).required_bytes(), "original attention byte extent");
      if (mapped.size() != extent)
        throw std::runtime_error("original GLM attention role byte length differs: " + path.string());
      const auto raw = shared::take(mapped.bytes(), "original attention mapped bytes");
      if (shared::detail::mapped_sha256(raw) != pins[slot])
        throw std::runtime_error("original GLM attention SHA256 differs: " + path.string());
      result.mappings_[slot] = std::move(mapped);
    }
    result.prepare(0, Parameter::kQueryProjection, RawRole::kQueryCodes, RawRole::kQueryScale);
    result.prepare(1, Parameter::kKvProjection, RawRole::kKvCodes, RawRole::kKvScale);
    result.prepare(2, Parameter::kOutput, RawRole::kOutputCodes, RawRole::kOutputScale);
    // The two norm vectors are small; validate their original BF16 values.
    (void)result.norm(Parameter::kKvNorm);
    (void)result.norm(Parameter::kPostNorm);
    return result;
  }
  std::span<const std::byte> bytes(RawRole role) const {
    return shared::take(mappings_[index(role)].bytes(), "original attention mapped span");
  }
  m::ConstBufferView view(RawRole role) const {
    return shared::take(mappings_[index(role)].view(), "original attention readonly mapping");
  }
  std::vector<std::uint16_t> norm(Parameter parameter) const {
    const auto role = parameter == Parameter::kKvNorm ? RawRole::kKvNorm : RawRole::kPostNorm;
    if (parameter != Parameter::kKvNorm && parameter != Parameter::kPostNorm)
      throw std::runtime_error("original attention norm role differs");
    const auto raw = bytes(role);
    std::vector<std::uint16_t> result(raw.size() / 2);
    for (std::size_t i = 0; i < result.size(); ++i) {
      result[i] = static_cast<std::uint16_t>(std::to_integer<std::uint32_t>(raw[i * 2]) |
          (std::to_integer<std::uint32_t>(raw[i * 2 + 1]) << 8u));
      if (!std::isfinite(std::bit_cast<float>(std::uint32_t{result[i]} << 16u)))
        throw std::runtime_error("original attention norm is nonfinite");
    }
    return result;
  }
  std::span<const std::byte> payload(Parameter parameter) const {
    switch (parameter) {
    case Parameter::kQueryProjection: return canonical(0);
    case Parameter::kKvProjection: return canonical(1);
    case Parameter::kOutput: return canonical(2);
    case Parameter::kKvNorm: return bytes(RawRole::kKvNorm);
    case Parameter::kKvB: return bytes(RawRole::kKvB);
    case Parameter::kPostNorm: return bytes(RawRole::kPostNorm);
    default: throw std::runtime_error("unknown original attention runtime payload");
    }
  }
 private:
  // FileMapping and system Buffer are strong owners. Returned spans/views
  // borrow Data; the caller retains its shared Data through oracle/native use.
  std::array<m::FileMapping, kRawRoles> mappings_;
  std::array<m::Buffer, 3> canonical_;
  // Maps retire before their Buffers. Swapping assignment keeps each old
  // Buffer alive until the corresponding scoped map has moved with it.
  std::array<m::HostMapping,3> canonical_maps_;

  std::span<const std::byte> canonical(std::size_t slot) const {
    return canonical_maps_.at(slot).bytes();
  }
  void prepare(std::size_t slot, Parameter parameter, RawRole codes, RawRole scales) {
    const auto raw_codes = bytes(codes), raw_scales = bytes(scales);
    for (const auto code : raw_codes)
      if ((std::to_integer<unsigned>(code) & 0x7fu) == 0x7fu)
        throw std::runtime_error("original attention E4 code is nonfinite");
    for (std::size_t offset = 0; offset < raw_scales.size(); offset += 4) {
      std::uint32_t bits = 0;
      for (std::size_t byte = 0; byte < 4; ++byte)
        bits |= std::to_integer<std::uint32_t>(raw_scales[offset + byte]) << (8 * byte);
      const auto scale = std::bit_cast<float>(bits);
      if (!std::isfinite(scale) || !(scale > 0))
        throw std::runtime_error("original attention stored scale is not positive finite");
    }
    const auto desc = description(parameter);
    const auto extent = shared::take(desc.required_bytes(), "original attention canonical extent");
    canonical_.at(slot) = shared::take(m::Buffer::create(m::system_resource(), extent, 64),
                                     "original attention canonical host bank");
    const auto source_codes = shared::take(c::ConstTensorView::create(raw_description(codes), view(codes)),
                                           "original attention source E4");
    const auto source_scales = shared::take(c::ConstTensorView::create(raw_description(scales), view(scales)),
                                            "original attention source scales");
    auto target = shared::take(c::TensorView::create(desc, canonical_[slot].view()),
                               "original attention canonical destination");
    shared::check(c::import_float8_block128_weight(source_codes, source_scales, target),
                  "original attention canonical block128 import");
    canonical_maps_[slot]=shared::take(canonical_[slot].view().map_host(),"map canonical attention bytes");
    const auto packed = canonical(slot);
    if (packed.size() != raw_codes.size() + raw_scales.size() ||
        !std::equal(raw_codes.begin(), raw_codes.end(), packed.begin()) ||
        !std::equal(raw_scales.begin(), raw_scales.end(), packed.begin() + static_cast<std::ptrdiff_t>(raw_codes.size())))
      throw std::runtime_error("attention canonical import changed original E4/F32 bytes or order");
  }
};
inline std::span<const std::byte> payload(const Data &data, Parameter parameter) {
  return data.payload(parameter);
}
// Share the existing immutable role carrier; no alternate binding protocol.
using ReadonlyRole = shared::ReadonlyRole;
using Roles = std::array<ReadonlyRole, kParameters>;
inline Roles bind(const Data &data, const c::StorageOwner &owner) {
  if (!owner) throw std::runtime_error("original attention needs its actual Graph storage owner");
  Roles result;
  for (std::size_t slot = 0; slot < kParameters; ++slot) {
    const auto parameter = static_cast<Parameter>(slot);
    const auto desc = description(parameter);
    const auto extent = shared::take(desc.required_bytes(), "original attention ready bank extent");
    const auto raw = payload(data, parameter);
    if (raw.size() != extent) throw std::runtime_error("original attention ready payload extent differs");
    auto buffer = shared::take(m::Buffer::create(owner.ref(), extent, 256), "original attention immutable bank");
    shared::check(buffer.view().write(raw), "original attention once-only bank upload");
    result[slot] = {kNames[slot], desc, owner,
        shared::take(owner.adopt_buffer(std::move(buffer)), "original attention readonly lease")};
  }
  return result;
}
} // namespace glm_original_attention_roles
