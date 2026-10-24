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

#include "glm_original_attention_roles.h"
#include "glm_original_indexer_roles.h"
#include "glm_original_moe_roles.h"

// Original Flash layer43. Its MLA/indexer data reuse the existing authenticated
// suppliers with explicit43 pins; only HC, input norm and q_a live here. EH,
// embedding/hidden norms and the shared LM are NextN concerns, absent from43.
// The owner retains mapped originals and once-prepared banks, never an executor.
namespace glm_original_target43_roles {
namespace c = hyle::compute;
namespace m = hyle::mem;
namespace shared = glm_shared_vocabulary;
namespace attention = glm_original_attention_roles;
namespace indexer = glm_original_indexer_roles;
namespace moe = glm_original_moe_roles;
inline constexpr std::string_view kRepository = shared::kRepository;
inline constexpr std::string_view kRevision = shared::kRevision;
inline constexpr std::int64_t kHidden = 4096, kStreams = 4, kQueryRank = 1536;
inline constexpr float kNormEpsilon = 1.0e-5F, kHcEpsilon = 1.0e-6F;
inline constexpr std::size_t kOriginalRoles = 26, kOriginalBytes = 150779608;
inline constexpr std::size_t kOwnOriginalBytes = 7877336;
// Exact acquisition pins identify all original43 bodies; incomplete pins fail;
// there is no fallback to another layer's payload or default45 catalogue.
inline constexpr std::string_view kMoeIndexSha256 = "a7221a1d60c5348b124797e528318ff2f2dc2977cc7d58e38e8fe71f971b783a";
inline constexpr attention::Pins kAttentionSha256{
    "f25d9c90fae9246b1faa736968cb2ba15bf12028cb97ba16bcb8d0d3c72e9272",
    "1935561604fe8e93cbebc6fbe75918a7cf362c21f25949b2f12ac7d2c03964ee",
    "06d3d83403b2fd3da16e7e9d991145628886c41a73664211be6c6f5c87cd01cf",
    "b80caa10b610fcd8890c12399d299de17f7382f7341e31fb94ba9eb27418dd64",
    "f11ae45b3220a86879ed24b22db77a958a97501c706fd9da6f800112a1fe7a78",
    "383a3246c5985b8a8c8ee7dac6955ce3292746b19f54687f1a5d564b2f403522",
    "8e153c8adef30a480e44396f7c1ae3c09a694268e7402859bab6fb26272591f4",
    "53981a5b2e0b56b9a3445950b6ef44a65175192330efa1ee91558340951a6d40",
    "d5a0aff5e2533d698e1958f881a55120448e8ac0aa5cc291b83a1eec622f852b"};
inline constexpr indexer::Pins kIndexerSha256{
    "4af70e57e3ece6448a410f7800e046ddfdb32d6fa835925360404060bafbd991",
    "2bda06ad717874478f2413dd9f5d8484097686e967e5cd05ca39aac712acb840",
    "4998508622c40255ec31824d8db73c7efcac203f4d9c7771542ade3caf87bcd6",
    "25b5022f401267c722ed328f5f8157a2888c2e14f30711f0a69fd6ff06683ae1",
    "610c0972a246f0e3fe0066238bf43160dc2b81c6c7f0420634970deaa89a781d",
    "596d10661e1ba868e4d26d59f0eabe3d15eefafcde4f656a2d122da42b8cd24d",
    "12a9693f54ef865d8eb974f9e6d3e9e42cde50652d57017a30a249b6ea202052"};

enum class RawRole : std::size_t {
  kHcAttnBase, kHcAttnFn, kHcAttnScale, kHcFfnBase, kHcFfnFn, kHcFfnScale,
  kInputNorm, kQueryCodes, kQueryScale, kQueryNorm, kCount
};
enum class Role : std::size_t {
  kHcAttnBase, kHcAttnFn, kHcAttnScale, kHcFfnBase, kHcFfnFn, kHcFfnScale,
  kInputNorm, kQueryProjection, kQueryNorm, kCount
};
inline constexpr std::size_t kRawRoles = static_cast<std::size_t>(RawRole::kCount);
inline constexpr std::size_t kRoles = static_cast<std::size_t>(Role::kCount);
struct Spec final {
  std::string_view name, file, sha256;
  c::DataType type;
  std::int64_t rows, columns; // columns0 denotes a vector.
};
inline constexpr std::array<Spec, kRawRoles> kSpecs{
    Spec{"model.language_model.layers.43.hc_attn_base", "hc_attn_base.f32", "b95e45dac4eddaf979e5f90ea42ff599ca91b1edefd57485350f8b1ca2f1226b", c::DataType::kFloat32, 24, 0},
    Spec{"model.language_model.layers.43.hc_attn_fn", "hc_attn_fn.bf16", "4070a1fe33f2e95178aa92fca292c0f0102f591194e393821b59389b6db63d73", c::DataType::kBFloat16, 24, 16384},
    Spec{"model.language_model.layers.43.hc_attn_scale", "hc_attn_scale.f32", "d2a40c7cc9d8189d439d482f54e38d734ae62fa7636d6c860ffa62d6b17f484d", c::DataType::kFloat32, 3, 0},
    Spec{"model.language_model.layers.43.hc_ffn_base", "hc_ffn_base.f32", "19af9c4158561db1106e7fc15ced68755205b1b007d149784425376dedaee022", c::DataType::kFloat32, 24, 0},
    Spec{"model.language_model.layers.43.hc_ffn_fn", "hc_ffn_fn.bf16", "800dbe6e6cb08d60d8559b6453985584b5559eff22eab3c0347eb879f01df8d1", c::DataType::kBFloat16, 24, 16384},
    Spec{"model.language_model.layers.43.hc_ffn_scale", "hc_ffn_scale.f32", "79f53704021df20fd2bd69f7d63a438f281a8fa5a88b321409f6c0d763d8d33c", c::DataType::kFloat32, 3, 0},
    Spec{"model.language_model.layers.43.input_layernorm.weight", "input_layernorm.weight.bf16", "83deadc36c0d4cb49376e9a76f67964c53f2c1b2610e39c96a57243c6d4767b5", c::DataType::kBFloat16, kHidden, 0},
    Spec{"model.language_model.layers.43.self_attn.q_a_proj.weight", "q_a.weight.e4m3", "31feb1d307d0046f847f6576f1ec3d8f3ee9190a4820b253c06178f6552394d1", c::DataType::kFloat8E4M3FN, kQueryRank, kHidden},
    Spec{"model.language_model.layers.43.self_attn.q_a_proj.weight_scale_inv", "q_a.weight_scale.f32", "882586f237a1dfd3a7e21b9ad537d059739cb2af4cf67d615619700e131d9f70", c::DataType::kFloat32, kQueryRank / 128, kHidden / 128},
    Spec{"model.language_model.layers.43.self_attn.q_a_layernorm.weight", "q_a_layernorm.weight.bf16", "3c1820761e31a221f7db8cf02268e982da1eee237eadbfe3daee999888bd6a05", c::DataType::kBFloat16, kQueryRank, 0}
};
inline std::size_t index(RawRole role) {
  const auto slot = static_cast<std::size_t>(role);
  if (slot >= kRawRoles) throw std::runtime_error("unknown original GLM43 raw role");
  return slot;
}
inline std::size_t index(Role role) {
  const auto slot = static_cast<std::size_t>(role);
  if (slot >= kRoles) throw std::runtime_error("unknown original GLM43 parameter");
  return slot;
}
inline const Spec &spec(RawRole role) { return kSpecs[index(role)]; }
struct Pins final {
  std::array<std::string_view,kRawRoles> own;
  attention::Pins attention;
  indexer::Pins indexer;
};
inline constexpr Pins kDefaultPins=[] {
  Pins result{};
  for(std::size_t i=0;i<kRawRoles;++i)result.own[i]=kSpecs[i].sha256;
  result.attention=kAttentionSha256;result.indexer=kIndexerSha256;
  return result;
}();
inline RawRole raw_role(Role role) {
  switch (role) {
  case Role::kHcAttnBase: return RawRole::kHcAttnBase;
  case Role::kHcAttnFn: return RawRole::kHcAttnFn;
  case Role::kHcAttnScale: return RawRole::kHcAttnScale;
  case Role::kHcFfnBase: return RawRole::kHcFfnBase;
  case Role::kHcFfnFn: return RawRole::kHcFfnFn;
  case Role::kHcFfnScale: return RawRole::kHcFfnScale;
  case Role::kInputNorm: return RawRole::kInputNorm;
  case Role::kQueryProjection: return RawRole::kQueryCodes;
  case Role::kQueryNorm: return RawRole::kQueryNorm;
  default: throw std::runtime_error("unknown original GLM43 parameter raw role");
  }
}
inline bool is_hc(Role role) noexcept { return role == Role::kHcAttnFn || role == Role::kHcFfnFn; }
inline c::TensorDesc raw_description(RawRole role) {
  const auto &s = spec(role);
  if (s.columns == 0)
    return shared::take(c::TensorDesc::dense(s.type, std::array{s.rows}), "original GLM43 vector");
  return shared::take(c::TensorDesc::dense(s.type, std::array{s.rows, s.columns}), "original GLM43 matrix");
}
inline c::TensorDesc description(Role role) {
  if (is_hc(role))
    return shared::take(c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 2>{24, 16384}), "original GLM43 runtime HC");
  if (role == Role::kQueryProjection)
    return shared::take(c::float8_weight_description(kHidden, kQueryRank,
        c::Float8ScaleMode::kFloat32Block128), "original GLM43 q_a canonical");
  return raw_description(raw_role(role));
}

class Data final {
 public:
  Data()=default;
  Data(Data&&)=default;
  Data &operator=(Data &&other) noexcept {
    if(this!=&other) {
      std::swap(attention,other.attention);std::swap(indexer,other.indexer);
      std::swap(mappings_,other.mappings_);std::swap(floats_,other.floats_);
      std::swap(norms_,other.norms_);std::swap(canonical_,other.canonical_);
      std::swap(canonical_map_,other.canonical_map_);
    }
    return *this;
  }
  glm_original_attention_roles::Data attention;
  glm_original_indexer_roles::Data indexer;
  static Data load(const std::filesystem::path &directory,const Pins &pins=kDefaultPins) {
    const auto pinned=[](std::span<const std::string_view> values) {
      for(const auto pin:values)if(pin.size()!=64||!std::ranges::all_of(pin,[](char ch) {
            return (ch>='0'&&ch<='9')||(ch>='a'&&ch<='f'); }))
        throw std::runtime_error("original GLM43 requires every actual static/MLA/indexer pin");
    };
    pinned(pins.own);pinned(pins.attention);pinned(pins.indexer);
    Data result;
    for (std::size_t slot = 0; slot < kRawRoles; ++slot) {
      const auto raw_role_id = static_cast<RawRole>(slot);
      const auto &s = kSpecs[slot]; const auto path = directory / std::string(s.file);
      auto mapped = shared::take(m::FileMapping::open(path.string().c_str()), "map original GLM43 parameter");
      if (mapped.size() != shared::take(raw_description(raw_role_id).required_bytes(), "original GLM43 byte extent"))
        throw std::runtime_error("original GLM43 role size differs: " + path.string());
      const auto raw = shared::take(mapped.bytes(), "original GLM43 mapped bytes");
      if (shared::detail::mapped_sha256(raw) != pins.own[slot])
        throw std::runtime_error("original GLM43 SHA256 differs: " + path.string());
      result.mappings_[slot] = std::move(mapped);
    }
    result.attention = glm_original_attention_roles::Data::load(directory, pins.attention);
    result.indexer = glm_original_indexer_roles::Data::load(directory, pins.indexer);
    for (const auto role : {Role::kHcAttnFn, Role::kHcFfnFn}) {
      const auto raw = result.bytes(raw_role(role));
      auto &values = result.floats_[index(raw_role(role))]; values.resize(raw.size() / 2);
      for (std::size_t i = 0; i < values.size(); ++i) values[i] = decode_bfloat(raw, i);
    }
    for (const auto raw_role_id : {RawRole::kHcAttnBase, RawRole::kHcAttnScale,
        RawRole::kHcFfnBase, RawRole::kHcFfnScale, RawRole::kQueryScale}) {
      const auto raw = result.bytes(raw_role_id);
      auto &values = result.floats_[index(raw_role_id)]; values.resize(raw.size() / 4);
      for (std::size_t i = 0; i < values.size(); ++i) values[i] = decode_float(raw, i);
    }
    for (const auto role : {Role::kInputNorm, Role::kQueryNorm}) {
      const auto raw = result.bytes(raw_role(role));
      auto &values = result.norms_[role == Role::kInputNorm ? 0 : 1]; values.resize(raw.size() / 2);
      for (std::size_t i = 0; i < values.size(); ++i) {
        (void)decode_bfloat(raw, i); values[i] = std::uint16_t(read_le(raw, i * 2, 2));
      }
    }
    result.prepare_query(); return result;
  }
  std::span<const std::byte> bytes(RawRole role) const {
    return shared::take(mappings_[index(role)].bytes(), "original GLM43 raw span");
  }
  std::span<const std::byte> payload(Role role) const {
    if (is_hc(role)) return std::as_bytes(std::span(floats_[index(raw_role(role))]));
    if (role == Role::kQueryProjection) return canonical();
    return bytes(raw_role(role));
  }
  std::span<const float> floats(Role role) const {
    if (!is_hc(role) && raw_description(raw_role(role)).data_type() != c::DataType::kFloat32)
      throw std::runtime_error("original GLM43 requested unprepared F32 matrix");
    return floats_[index(raw_role(role))];
  }
  std::span<const float> qa_scales() const { return floats_[index(RawRole::kQueryScale)]; }
  std::span<const std::uint16_t> norm(Role role) const {
    if (role != Role::kInputNorm && role != Role::kQueryNorm)
      throw std::runtime_error("original GLM43 requested non-norm parameter");
    return norms_[role == Role::kInputNorm ? 0 : 1];
  }
 private:
  std::array<m::FileMapping, kRawRoles> mappings_;
  std::array<std::vector<float>, kRawRoles> floats_;
  std::array<std::vector<std::uint16_t>, 2> norms_;
  m::Buffer canonical_;
  m::HostMapping canonical_map_;
  static std::uint32_t read_le(std::span<const std::byte> raw, std::size_t offset, std::size_t count) {
    std::uint32_t bits = 0;
    for (std::size_t byte = 0; byte < count; ++byte)
      bits |= std::to_integer<std::uint32_t>(raw[offset + byte]) << (8u * unsigned(byte));
    return bits;
  }
  static float decode_bfloat(std::span<const std::byte> raw, std::size_t i) {
    const auto value = std::bit_cast<float>(read_le(raw, i * 2, 2) << 16u);
    if (!std::isfinite(value)) throw std::runtime_error("original GLM43 BF16 parameter is nonfinite");
    return value;
  }
  static float decode_float(std::span<const std::byte> raw, std::size_t i) {
    const auto value = std::bit_cast<float>(read_le(raw, i * 4, 4));
    if (!std::isfinite(value)) throw std::runtime_error("original GLM43 F32 parameter is nonfinite");
    return value;
  }
  std::span<const std::byte> canonical() const {
    return canonical_map_.bytes();
  }
  void prepare_query() {
    const auto codes = bytes(RawRole::kQueryCodes), scales = bytes(RawRole::kQueryScale);
    for (const auto code : codes)
      if ((std::to_integer<unsigned>(code) & 0x7fu) == 0x7fu)
        throw std::runtime_error("original GLM43 q_a has nonfinite E4 code");
    for (const auto scale : qa_scales())
      if (!(scale > 0)) throw std::runtime_error("original GLM43 q_a requires positive finite scales");
    const auto desc = description(Role::kQueryProjection);
    canonical_ = shared::take(m::Buffer::create(m::system_resource(),
        shared::take(desc.required_bytes(), "original GLM43 canonical size"), 64), "original GLM43 canonical bank");
    const auto source = [&](RawRole role) {
      return shared::take(c::ConstTensorView::create(raw_description(role),
          shared::take(mappings_[index(role)].view(), "original GLM43 import source")), "original GLM43 source tensor");
    };
    auto target = shared::take(c::TensorView::create(desc, canonical_.view()), "original GLM43 canonical destination");
    shared::check(c::import_float8_block128_weight(source(RawRole::kQueryCodes), source(RawRole::kQueryScale), target),
        "original GLM43 canonical block128 import");
    canonical_map_=shared::take(canonical_.view().map_host(),"original GLM43 canonical host bytes");
    const auto packed = canonical();
    if (packed.size() != codes.size() + scales.size() ||
        !std::equal(codes.begin(), codes.end(), packed.begin()) ||
        !std::equal(scales.begin(), scales.end(), packed.begin() + static_cast<std::ptrdiff_t>(codes.size())))
      throw std::runtime_error("original GLM43 importer changed original q_a codes/scales");
  }
};
struct Bound final {
  std::shared_ptr<const Data> source;
  std::array<shared::ReadonlyRole, kRoles> roles;
  const shared::ReadonlyRole &at(Role role) const { return roles[index(role)]; }
};
inline Bound bind(std::shared_ptr<const Data> source, const c::StorageOwner &owner) {
  if (!source || !owner) throw std::runtime_error("original GLM43 binding requires strong source and actual Graph owner");
  Bound result; result.source = std::move(source);
  for (std::size_t slot = 0; slot < kRoles; ++slot) {
    const auto role = static_cast<Role>(slot); const auto desc = description(role);
    const auto raw = result.source->payload(role);
    const auto extent = shared::take(desc.required_bytes(), "original GLM43 ready bytes");
    if (raw.size() != extent) throw std::runtime_error("original GLM43 prepared payload differs");
    auto bank = shared::take(m::Buffer::create(owner.ref(), extent, 256), "original GLM43 immutable bank");
    shared::check(bank.view().write(raw), "original GLM43 once-only upload");
    result.roles[slot] = {spec(raw_role(role)).name, desc, owner,
        shared::take(owner.adopt_buffer(std::move(bank)), "original GLM43 readonly lease")};
  }
  // MLA and indexer are deliberately bound by their existing factories. No
  // extra raw KV_b device lease is retained here after their cold K/V split.
  return result;
}
inline moe::Data load_moe(const std::filesystem::path &directory) {
  return moe::Data::load(directory, kMoeIndexSha256);
}
} // namespace glm_original_target43_roles
