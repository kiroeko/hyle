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

#include "original_role_input.h"
#include "dspark_original_role_source.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/storage_owner.h"
#include "hyle/compute/weight_import.h"
#include "hyle/graph/executable.h"
#include "hyle/mem/mapping.h"

// Original formal Flash0731 target40->41->42 and distinct main finalHC/norm.
// One authenticated bundle owns all raw roles; experts remain original packed
// FP4 plus UE8M0 group32, not GGUF MXFP4. This file does not execute a Graph.
namespace dspark_original_target_roles {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
namespace carrier = dspark_original_role_source;
inline constexpr std::string_view kRepository = "deepseek-ai/DeepSeek-V4-Flash-0731";
inline constexpr std::string_view kRevision = "7872f01b1d1fe23eabc4c98b48bffcef5a386062";
inline constexpr std::int64_t kHidden = 4096, kIntermediate = 2048;
inline constexpr std::int32_t kDomain = 256;
inline constexpr std::size_t kStages = 3, kResidentCapacity = 6, kOriginalRoles = 4725;
inline constexpr std::size_t kOriginalBytes = 10748569116ULL, kIndexBytes = 64 + kOriginalRoles * 16;
// Filled only by root's completed acquisition; empty pins reject loading.
inline constexpr std::string_view kIndexSha256 = "891133bf24c829c51b7c4819df2104e4ce0616c1b5682dbf8025c42fa447aa8a";

template<class T> inline T take(hyle::Result<T> result, std::string_view where) {
  if (!result) throw std::runtime_error(std::string(where) + ": " + std::string(result.error().message()));
  return std::move(*result);
}
inline void check(hyle::Status status, std::string_view where) {
  if (!status.ok()) throw std::runtime_error(std::string(where) + ": " + std::string(status.message()));
}
inline std::size_t stage_index(std::size_t stage) {
  if (stage >= kStages) throw std::runtime_error("original target stage outside40..42");
  return stage;
}
inline int compression_ratio(std::size_t stage) { return stage_index(stage) == 1 ? 128 : 4; }
enum class RawRole : std::size_t {
    kAttentionSink, kCompressorApe, kIndexerApe, kRouterBias, kAttentionHcBase, kAttentionHcFn, kAttentionHcScale, kFfnHcBase, kFfnHcFn, kFfnHcScale, kCompressorNorm, kCompressorGate, kCompressorKey, kIndexerNorm, kIndexerGate, kIndexerKey, kIndexerWeights, kKvNorm, kQueryNorm, kAttentionNorm, kRouter, kFfnNorm, kIndexerQueryScale, kKvScale, kOutputAScale, kOutputBScale, kQueryAScale, kQueryBScale, kSharedGateScale, kSharedDownScale, kSharedUpScale, kIndexerQueryCodes, kKvCodes, kOutputACodes, kOutputBCodes, kQueryACodes, kQueryBCodes, kSharedGateCodes, kSharedDownCodes, kSharedUpCodes, kCount
};
enum class Role : std::size_t {
    kAttentionSink, kCompressorApe, kIndexerApe, kRouterBias, kAttentionHcBase, kAttentionHcFn, kAttentionHcScale, kFfnHcBase, kFfnHcFn, kFfnHcScale, kCompressorNorm, kCompressorGate, kCompressorKey, kIndexerNorm, kIndexerGate, kIndexerKey, kIndexerWeights, kKvNorm, kQueryNorm, kAttentionNorm, kRouter, kFfnNorm, kIndexerQuery, kKv, kOutputA, kOutputB, kQueryA, kQueryB, kSharedGate, kSharedDown, kSharedUp, kCount
};
enum class FinalRole : std::size_t { kBase, kFn, kScale, kNorm, kCount };
enum class Leg : std::size_t { kGate, kUp, kDown };
inline constexpr std::size_t kRawRoles = static_cast<std::size_t>(RawRole::kCount);
inline constexpr std::size_t kRoles = static_cast<std::size_t>(Role::kCount);
inline constexpr std::size_t kFinalRoles = static_cast<std::size_t>(FinalRole::kCount);
template<class E> inline std::size_t enum_index(E role, std::size_t count) {
  const auto value = static_cast<std::size_t>(role);
  if (value >= count) throw std::runtime_error("unknown formal target role");
  return value;
}
inline std::size_t index(RawRole role) { return enum_index(role, kRawRoles); }
inline std::size_t index(Role role) { return enum_index(role, kRoles); }
inline std::size_t index(FinalRole role) { return enum_index(role, kFinalRoles); }
inline std::size_t index(Leg leg) { return enum_index(leg, 3); }
struct RawSpec final { std::string_view suffix; c::DataType type; std::int64_t rows, columns; };
inline constexpr std::array<RawSpec, kRawRoles> kRawSpecs{
    RawSpec{"attn.attn_sink", c::DataType::kFloat32, 64, 0},
    RawSpec{"attn.compressor.ape", c::DataType::kFloat32, 4, 1024},
    RawSpec{"attn.indexer.compressor.ape", c::DataType::kFloat32, 4, 256},
    RawSpec{"ffn.gate.bias", c::DataType::kFloat32, 256, 0},
    RawSpec{"hc_attn_base", c::DataType::kFloat32, 24, 0},
    RawSpec{"hc_attn_fn", c::DataType::kFloat32, 24, 16384},
    RawSpec{"hc_attn_scale", c::DataType::kFloat32, 3, 0},
    RawSpec{"hc_ffn_base", c::DataType::kFloat32, 24, 0},
    RawSpec{"hc_ffn_fn", c::DataType::kFloat32, 24, 16384},
    RawSpec{"hc_ffn_scale", c::DataType::kFloat32, 3, 0},
    RawSpec{"attn.compressor.norm.weight", c::DataType::kBFloat16, 512, 0},
    RawSpec{"attn.compressor.wgate.weight", c::DataType::kBFloat16, 1024, 4096},
    RawSpec{"attn.compressor.wkv.weight", c::DataType::kBFloat16, 1024, 4096},
    RawSpec{"attn.indexer.compressor.norm.weight", c::DataType::kBFloat16, 128, 0},
    RawSpec{"attn.indexer.compressor.wgate.weight", c::DataType::kBFloat16, 256, 4096},
    RawSpec{"attn.indexer.compressor.wkv.weight", c::DataType::kBFloat16, 256, 4096},
    RawSpec{"attn.indexer.weights_proj.weight", c::DataType::kBFloat16, 64, 4096},
    RawSpec{"attn.kv_norm.weight", c::DataType::kBFloat16, 512, 0},
    RawSpec{"attn.q_norm.weight", c::DataType::kBFloat16, 1024, 0},
    RawSpec{"attn_norm.weight", c::DataType::kBFloat16, 4096, 0},
    RawSpec{"ffn.gate.weight", c::DataType::kBFloat16, 256, 4096},
    RawSpec{"ffn_norm.weight", c::DataType::kBFloat16, 4096, 0},
    RawSpec{"attn.indexer.wq_b.scale", c::DataType::kUInt8, 64, 8},
    RawSpec{"attn.wkv.scale", c::DataType::kUInt8, 4, 32},
    RawSpec{"attn.wo_a.scale", c::DataType::kUInt8, 64, 32},
    RawSpec{"attn.wo_b.scale", c::DataType::kUInt8, 32, 64},
    RawSpec{"attn.wq_a.scale", c::DataType::kUInt8, 8, 32},
    RawSpec{"attn.wq_b.scale", c::DataType::kUInt8, 256, 8},
    RawSpec{"ffn.shared_experts.w1.scale", c::DataType::kUInt8, 16, 32},
    RawSpec{"ffn.shared_experts.w2.scale", c::DataType::kUInt8, 32, 16},
    RawSpec{"ffn.shared_experts.w3.scale", c::DataType::kUInt8, 16, 32},
    RawSpec{"attn.indexer.wq_b.weight", c::DataType::kFloat8E4M3FN, 8192, 1024},
    RawSpec{"attn.wkv.weight", c::DataType::kFloat8E4M3FN, 512, 4096},
    RawSpec{"attn.wo_a.weight", c::DataType::kFloat8E4M3FN, 8192, 4096},
    RawSpec{"attn.wo_b.weight", c::DataType::kFloat8E4M3FN, 4096, 8192},
    RawSpec{"attn.wq_a.weight", c::DataType::kFloat8E4M3FN, 1024, 4096},
    RawSpec{"attn.wq_b.weight", c::DataType::kFloat8E4M3FN, 32768, 1024},
    RawSpec{"ffn.shared_experts.w1.weight", c::DataType::kFloat8E4M3FN, 2048, 4096},
    RawSpec{"ffn.shared_experts.w2.weight", c::DataType::kFloat8E4M3FN, 4096, 2048},
    RawSpec{"ffn.shared_experts.w3.weight", c::DataType::kFloat8E4M3FN, 2048, 4096}
};
struct RoleSource final { RawRole value,scale; };
inline constexpr std::array<RoleSource,kRoles> kRoleSources{
    RoleSource{RawRole::kAttentionSink,RawRole::kCount},
    RoleSource{RawRole::kCompressorApe,RawRole::kCount},
    RoleSource{RawRole::kIndexerApe,RawRole::kCount},
    RoleSource{RawRole::kRouterBias,RawRole::kCount},
    RoleSource{RawRole::kAttentionHcBase,RawRole::kCount},
    RoleSource{RawRole::kAttentionHcFn,RawRole::kCount},
    RoleSource{RawRole::kAttentionHcScale,RawRole::kCount},
    RoleSource{RawRole::kFfnHcBase,RawRole::kCount},
    RoleSource{RawRole::kFfnHcFn,RawRole::kCount},
    RoleSource{RawRole::kFfnHcScale,RawRole::kCount},
    RoleSource{RawRole::kCompressorNorm,RawRole::kCount},
    RoleSource{RawRole::kCompressorGate,RawRole::kCount},
    RoleSource{RawRole::kCompressorKey,RawRole::kCount},
    RoleSource{RawRole::kIndexerNorm,RawRole::kCount},
    RoleSource{RawRole::kIndexerGate,RawRole::kCount},
    RoleSource{RawRole::kIndexerKey,RawRole::kCount},
    RoleSource{RawRole::kIndexerWeights,RawRole::kCount},
    RoleSource{RawRole::kKvNorm,RawRole::kCount},
    RoleSource{RawRole::kQueryNorm,RawRole::kCount},
    RoleSource{RawRole::kAttentionNorm,RawRole::kCount},
    RoleSource{RawRole::kRouter,RawRole::kCount},
    RoleSource{RawRole::kFfnNorm,RawRole::kCount},
    RoleSource{RawRole::kIndexerQueryCodes,RawRole::kIndexerQueryScale},
    RoleSource{RawRole::kKvCodes,RawRole::kKvScale},
    RoleSource{RawRole::kOutputACodes,RawRole::kOutputAScale},
    RoleSource{RawRole::kOutputBCodes,RawRole::kOutputBScale},
    RoleSource{RawRole::kQueryACodes,RawRole::kQueryAScale},
    RoleSource{RawRole::kQueryBCodes,RawRole::kQueryBScale},
    RoleSource{RawRole::kSharedGateCodes,RawRole::kSharedGateScale},
    RoleSource{RawRole::kSharedDownCodes,RawRole::kSharedDownScale},
    RoleSource{RawRole::kSharedUpCodes,RawRole::kSharedUpScale}
};
inline RawRole raw_role(Role role) { return kRoleSources[index(role)].value; }
inline bool present(std::size_t stage, RawRole role) {
  (void)stage_index(stage); (void)index(role);
  if (stage != 1) return true;
  return role != RawRole::kIndexerApe && role != RawRole::kIndexerNorm &&
      role != RawRole::kIndexerGate && role != RawRole::kIndexerKey &&
      role != RawRole::kIndexerWeights && role != RawRole::kIndexerQueryCodes &&
      role != RawRole::kIndexerQueryScale;
}
inline bool present(std::size_t stage, Role role) { return present(stage, raw_role(role)); }
inline RawSpec raw_spec(std::size_t stage, RawRole role) {
  if (!present(stage, role)) throw std::runtime_error("ratio128 layer has no indexer parameters");
  auto result = kRawSpecs[index(role)];
  if (stage == 1) {
    if (role == RawRole::kCompressorApe) { result.rows = 128; result.columns = 512; }
    if (role == RawRole::kCompressorGate || role == RawRole::kCompressorKey) result.rows = 512;
  }
  return result;
}
inline c::TensorDesc raw_description(std::size_t stage, RawRole role) {
  const auto spec = raw_spec(stage, role);
  if (spec.columns == 0) return take(c::TensorDesc::dense(spec.type, std::array{spec.rows}), "formal target raw vector");
  return take(c::TensorDesc::dense(spec.type, std::array{spec.rows,spec.columns}), "formal target raw matrix");
}
inline bool is_matrix(Role role) { return kRawSpecs[index(raw_role(role))].type == c::DataType::kFloat8E4M3FN; }
inline bool cold_widen(Role role) {
  return role == Role::kCompressorGate || role == Role::kCompressorKey ||
      role == Role::kIndexerGate || role == Role::kIndexerKey;
}
inline RawRole scale_role(Role role) {
  if (!is_matrix(role)) throw std::runtime_error("formal target scale requested for non-FP8 role");
  return kRoleSources[index(role)].scale;
}
inline c::TensorDesc description(std::size_t stage, Role role) {
  const auto spec = raw_spec(stage,raw_role(role));
  if (is_matrix(role)) return take(c::float8_weight_description(spec.columns,spec.rows,
      c::Float8ScaleMode::kFloat32Block128), "formal target canonical FP8");
  if (cold_widen(role)) return take(c::TensorDesc::dense(c::DataType::kFloat32,
      std::array{spec.rows,spec.columns}), "formal target source-required F32 compressor");
  return raw_description(stage,raw_role(role));
}
inline RawSpec final_spec(FinalRole role) {
  switch (role) {
    case FinalRole::kBase: return {"hc_head_base",c::DataType::kFloat32,4,0};
    case FinalRole::kFn: return {"hc_head_fn",c::DataType::kFloat32,4,16384};
    case FinalRole::kScale: return {"hc_head_scale",c::DataType::kFloat32,1,0};
    case FinalRole::kNorm: return {"norm.weight",c::DataType::kBFloat16,4096,0};
    default: throw std::runtime_error("unknown formal target final role");
  }
}
inline c::TensorDesc description(FinalRole role) {
  const auto spec = final_spec(role);
  if (!spec.columns) return take(c::TensorDesc::dense(spec.type,std::array{spec.rows}),"formal target final vector");
  return take(c::TensorDesc::dense(spec.type,std::array{spec.rows,spec.columns}),"formal target final matrix");
}
inline std::pair<std::int64_t,std::int64_t> geometry(Leg leg) {
  (void)index(leg);
  return leg == Leg::kDown ? std::pair{kHidden,kIntermediate} : std::pair{kIntermediate,kHidden};
}
inline c::TensorDesc resident_description(Leg leg,std::size_t capacity=kResidentCapacity) {
  if (!capacity || capacity > std::size_t(kDomain)) throw std::runtime_error("invalid formal target resident expert capacity");
  const auto [outputs,reduction] = geometry(leg);
  return take(c::float4_grouped_weight_description(std::int64_t(capacity),outputs,reduction),"formal target FP4 resident bank");
}
struct Selection final { std::vector<std::int32_t> globals; std::array<std::int32_t,kDomain> map{}; };
inline Selection select(std::span<const std::int32_t> actual_ids) {
  if (actual_ids.empty() || actual_ids.size() > kResidentCapacity) throw std::runtime_error("formal target selection requires actual Top6 IDs");
  Selection result; result.map.fill(-1); result.globals.reserve(actual_ids.size());
  for (const auto global : actual_ids) {
    if (global < 0 || global >= kDomain) throw std::runtime_error("formal target expert outside original256 domain");
    result.globals.push_back(global);
  }
  std::ranges::sort(result.globals);
  result.globals.erase(std::unique(result.globals.begin(),result.globals.end()),result.globals.end());
  for (std::size_t local=0;local<result.globals.size();++local) result.map[std::size_t(result.globals[local])]=std::int32_t(local);
  return result;
}
struct Matrix final { std::span<const std::byte> codes; std::span<const float> scales; std::int64_t outputs,reduction; };

class Data final {
 public:
  Data()=default;
  Data(const Data&)=delete;
  Data& operator=(const Data&)=delete;
  Data(Data&&) noexcept=default;
  // A move constructor transfers the live mapping and its allocation together.
  // Assignment cannot replace a Buffer before retiring its current map.
  Data& operator=(Data&&)=delete;
  c::TensorDesc expert_description(std::size_t stage, Leg leg) const {
    (void)stage_index(stage);
    return resident_description(leg, kDomain);
  }
  static Data load(const std::filesystem::path &directory) {
    if (kIndexSha256.size()!=64) throw std::runtime_error("formal target catalogue has no acquisition pin");
    auto catalog=take(m::FileMapping::open((directory/"catalog.bin").string().c_str()),"map formal target catalogue");
    if (catalog.size()!=kIndexBytes) throw std::runtime_error("formal target catalogue extent differs");
    const auto raw=take(catalog.bytes(),"formal target catalogue bytes");
    if (original_role_input::sha256(raw)!=kIndexSha256) throw std::runtime_error("formal target catalogue SHA256 differs");
    constexpr std::string_view magic="HYDSPT01";
    for (std::size_t i=0;i<magic.size();++i)
      if (raw[i]!=std::byte(static_cast<unsigned char>(magic[i]))) throw std::runtime_error("formal target catalogue magic differs");
    if (u32(raw,8)!=1 || u32(raw,12)!=kOriginalRoles || u64(raw,16)!=kOriginalBytes || u64(raw,24)!=0)
      throw std::runtime_error("formal target catalogue version/count/extent differs");
    Data result;
    std::size_t ordinal=0;
    for (std::size_t stage=0;stage<kStages;++stage) {
      result.static_ordinals_[stage].fill(kAbsent);
      for (std::size_t slot=0;slot<kRawRoles;++slot) {
        const auto role=static_cast<RawRole>(slot);
        if (!present(stage,role)) continue;
        result.static_ordinals_[stage][slot]=ordinal;
        result.read_range(raw,ordinal++,take(raw_description(stage,role).required_bytes(),"formal target raw extent"));
      }
      result.expert_first_[stage]=ordinal;
      for (std::int32_t global=0;global<kDomain;++global) {
        for (std::size_t slot=0;slot<3;++slot) {
          const auto [outputs,reduction]=geometry(static_cast<Leg>(slot));
          const auto elements=std::size_t(outputs)*std::size_t(reduction);
          result.read_range(raw,ordinal++,elements/2);
          result.read_range(raw,ordinal++,elements/32);
        }
      }
    }
    result.final_first_=ordinal;
    for (std::size_t slot=0;slot<kFinalRoles;++slot)
      result.read_range(raw,ordinal++,take(description(static_cast<FinalRole>(slot)).required_bytes(),"formal target final raw extent"));
    if (ordinal!=kOriginalRoles) throw std::runtime_error("formal target catalogue role order differs");
    auto ordered=result.ranges_;
    std::ranges::sort(ordered,{},&Range::offset);
    std::size_t cursor=0;
    for (const auto &range:ordered) {
      if (range.offset!=cursor) throw std::runtime_error("formal target catalogue has overlap or holes");
      cursor+=range.bytes;
    }
    if (cursor!=kOriginalBytes) throw std::runtime_error("formal target catalogue does not cover original bundle");
    result.mapping_=take(m::FileMapping::open((directory/"roles.bin").string().c_str()),"map formal target original bundle");
    if (result.mapping_.size()!=kOriginalBytes) throw std::runtime_error("formal target original bundle extent differs");
    constexpr char hex[]="0123456789abcdef";
    std::string expected(64,'0');
    for (std::size_t i=0;i<32;++i) {
      const auto byte=std::to_integer<unsigned>(raw[32+i]);
      expected[2*i]=hex[byte>>4u]; expected[2*i+1]=hex[byte&15u];
    }
    // Exactly one bounded whole-bundle SHA pass; no copy and no per-role rehash.
    if (original_role_input::sha256(take(result.mapping_.bytes(),"formal target whole original bytes"))!=expected)
      throw std::runtime_error("formal target original bundle SHA256 differs");
    for (std::size_t stage=0;stage<kStages;++stage) {
      for (std::size_t slot=0;slot<kRoles;++slot) {
        const auto role=static_cast<Role>(slot);
        if (!present(stage,role)) continue;
        if (is_matrix(role)) result.prepare_matrix(stage,role);
        else result.decode_small(stage,role);
      }
      // A UE8M0 code255 denotes NaN. FP4 codes themselves are all finite.
      for (std::int32_t global=0;global<kDomain;++global)
        for (std::size_t slot=0;slot<3;++slot)
          for (const auto code:result.expert_scales(stage,static_cast<Leg>(slot),global))
            if (code==std::byte{255}) throw std::runtime_error("formal target expert UE8M0 scale is nonfinite");
    }
    for (std::size_t slot=0;slot<kFinalRoles;++slot) {
      const auto role=static_cast<FinalRole>(slot);
      const auto source=result.bytes(role);
      if (role==FinalRole::kNorm) result.final_norm_=decode_bfloat(source);
      else result.final_floats_[slot]=decode_f32(source);
    }
    return result;
  }
  std::span<const std::byte> raw(std::size_t stage,RawRole role) const {
    return bytes_at(raw_ordinal(stage,role));
  }
  m::ConstBufferView view(std::size_t stage,RawRole role) const {
    return view_at(raw_ordinal(stage,role));
  }
  std::span<const std::byte> raw_codes(std::size_t stage,Role role) const {
    if (!is_matrix(role)) throw std::runtime_error("formal target codes requested for non-FP8 role");
    return raw(stage,raw_role(role));
  }
  std::span<const std::byte> raw_scales(std::size_t stage,Role role) const { return raw(stage,scale_role(role)); }
  std::span<const std::byte> payload(std::size_t stage,Role role) const {
    const auto slot=index(role); (void)raw_spec(stage,raw_role(role));
    if (is_matrix(role)) return canonical(stage,role);
    if (cold_widen(role)) return std::as_bytes(std::span(floats_[stage][slot]));
    return raw(stage,raw_role(role));
  }
  std::span<const std::uint16_t> norm(std::size_t stage,Role role) const {
    const auto spec=raw_spec(stage,raw_role(role));
    if (spec.type!=c::DataType::kBFloat16 || spec.columns!=0) throw std::runtime_error("formal target norm requested for non-vector BF16 role");
    return norms_[stage][index(role)];
  }
  std::span<const float> floats(std::size_t stage,Role role) const {
    const auto spec=raw_spec(stage,raw_role(role));
    if (spec.type!=c::DataType::kFloat32 && !cold_widen(role)) throw std::runtime_error("formal target F32 coefficients requested for another dtype");
    return floats_[stage][index(role)];
  }
  Matrix matrix(std::size_t stage,Role role) const {
    const auto spec=raw_spec(stage,raw_role(role));
    if (!is_matrix(role)) throw std::runtime_error("formal target FP8 matrix requested for another role");
    return {raw_codes(stage,role),scales_[stage][index(role)],spec.rows,spec.columns};
  }
  std::span<const std::byte> bytes(FinalRole role) const { return bytes_at(final_first_+index(role)); }
  m::ConstBufferView view(FinalRole role) const { return view_at(final_first_+index(role)); }
  std::span<const float> floats(FinalRole role) const {
    if (role==FinalRole::kNorm) throw std::runtime_error("formal target final norm remains BF16");
    return final_floats_[index(role)];
  }
  std::span<const std::uint16_t> norm() const { return final_norm_; }
  std::span<const std::byte> expert_codes(std::size_t stage,Leg leg,std::int32_t global) const {
    return bytes_at(expert_ordinal(stage,leg,global,false));
  }
  std::span<const std::byte> expert_scales(std::size_t stage,Leg leg,std::int32_t global) const {
    return bytes_at(expert_ordinal(stage,leg,global,true));
  }
  m::ConstBufferView expert_view(std::size_t stage,Leg leg,std::int32_t global,bool scales) const {
    return view_at(expert_ordinal(stage,leg,global,scales));
  }
 private:
  struct Range final { std::size_t offset=0,bytes=0; };
  static constexpr std::size_t kAbsent=std::numeric_limits<std::size_t>::max();
  m::FileMapping mapping_;
  std::array<Range,kOriginalRoles> ranges_{};
  std::array<std::array<std::size_t,kRawRoles>,kStages> static_ordinals_{};
  std::array<std::size_t,kStages> expert_first_{};
  std::size_t final_first_=0;
  std::array<std::array<m::Buffer,kRoles>,kStages> canonical_;
  // Reverse member destruction closes every read mapping before its Buffer.
  std::array<std::array<m::HostMapping,kRoles>,kStages> canonical_maps_;
  std::array<std::array<std::vector<float>,kRoles>,kStages> scales_,floats_;
  std::array<std::array<std::vector<std::uint16_t>,kRoles>,kStages> norms_;
  std::array<std::vector<float>,kFinalRoles> final_floats_;
  std::vector<std::uint16_t> final_norm_;
  static std::uint32_t u32(std::span<const std::byte> raw,std::size_t offset) {
    std::uint32_t result=0;
    for (std::size_t byte=0;byte<4;++byte) result|=std::to_integer<std::uint32_t>(raw[offset+byte])<<(8*byte);
    return result;
  }
  static std::uint64_t u64(std::span<const std::byte> raw,std::size_t offset) {
    std::uint64_t result=0;
    for (std::size_t byte=0;byte<8;++byte) result|=std::to_integer<std::uint64_t>(raw[offset+byte])<<(8*byte);
    return result;
  }
  void read_range(std::span<const std::byte> catalog,std::size_t role,std::size_t expected) {
    if (role>=kOriginalRoles) throw std::runtime_error("formal target catalogue role overflow");
    const auto offset=u64(catalog,64+role*16),extent=u64(catalog,72+role*16);
    if (offset>kOriginalBytes || extent>kOriginalBytes-offset || extent!=expected ||
        offset>std::numeric_limits<std::size_t>::max() || extent>std::numeric_limits<std::size_t>::max())
      throw std::runtime_error("formal target catalogue role range/shape differs");
    ranges_[role]={std::size_t(offset),std::size_t(extent)};
  }
  std::size_t raw_ordinal(std::size_t stage,RawRole role) const {
    const auto ordinal=static_ordinals_[stage_index(stage)][index(role)];
    if (ordinal==kAbsent) throw std::runtime_error("ratio128 layer has no original indexer role");
    return ordinal;
  }
  std::size_t expert_ordinal(std::size_t stage,Leg leg,std::int32_t global,bool scales) const {
    if (global<0 || global>=kDomain) throw std::runtime_error("formal target expert outside original256 domain");
    return expert_first_[stage_index(stage)]+std::size_t(global)*6+index(leg)*2+std::size_t(scales);
  }
  std::span<const std::byte> bytes_at(std::size_t ordinal) const {
    const auto &range=ranges_.at(ordinal);
    return take(mapping_.bytes(range.offset,range.bytes),"formal target original role span");
  }
  m::ConstBufferView view_at(std::size_t ordinal) const {
    const auto &range=ranges_.at(ordinal);
    return take(mapping_.view(range.offset,range.bytes),"formal target original readonly role view");
  }
  static std::vector<std::uint16_t> decode_bfloat(std::span<const std::byte> raw) {
    return carrier::decode_bfloat(raw);
  }
  static std::vector<float> decode_f32(std::span<const std::byte> raw) {
    return carrier::decode_f32(raw);
  }
  void decode_small(std::size_t stage,Role role) {
    const auto spec=raw_spec(stage,raw_role(role));
    const auto source=raw(stage,raw_role(role));
    if (cold_widen(role)) {
      floats_[stage][index(role)]=dspark_original_role_source::widen_bfloat(source);
    } else if (spec.type==c::DataType::kFloat32) floats_[stage][index(role)]=decode_f32(source);
    else if (spec.type==c::DataType::kBFloat16 && spec.columns==0) norms_[stage][index(role)]=decode_bfloat(source);
    // Router and indexer weights stay original BF16, not an F32 shadow.
  }
  std::span<const std::byte> canonical(std::size_t stage,Role role) const {
    const auto &mapped=canonical_maps_[stage_index(stage)][index(role)];
    if (!mapped) throw std::runtime_error("formal target canonical read mapping is not ready");
    return mapped.bytes(); // Borrowed from the live map strongly held by Data.
  }
  void prepare_matrix(std::size_t stage,Role role) {
    const auto codes_id=raw_role(role),scales_id=scale_role(role);
    auto prepared=carrier::prepare_float8(raw_description(stage,codes_id),view(stage,codes_id),raw(stage,codes_id),
        raw_description(stage,scales_id),view(stage,scales_id),raw(stage,scales_id));
    if(prepared.description!=description(stage,role))
      throw std::runtime_error("formal target canonical role description differs");
    scales_[stage][index(role)]=std::move(prepared.scales);
    canonical_[stage][index(role)]=std::move(prepared.buffer);
    canonical_maps_[stage][index(role)]=std::move(prepared.mapping);
  }
};
using ReadonlyRole=carrier::ReadonlyRole;

template<class SourceData> struct TargetBindings final {
  // Raw views and reference spans borrow this strong Data/mapping lifetime.
  std::shared_ptr<const SourceData> source;
  std::array<std::array<std::optional<ReadonlyRole>,kRoles>,kStages> roles;
  std::array<ReadonlyRole,kFinalRoles> final;
  const ReadonlyRole &at(std::size_t stage,Role role) const {
    const auto &value=roles[stage_index(stage)][index(role)];
    if (!value) throw std::runtime_error("formal target unavailable runtime role");
    return *value;
  }
  const ReadonlyRole &at(FinalRole role) const { return final[index(role)]; }
};
using Bound=TargetBindings<Data>;
inline Bound bind(std::shared_ptr<const Data> source,const c::StorageOwner &owner) {
  if (!source || !owner) throw std::runtime_error("formal target bind needs original data and actual Graph owner");
  Bound result; result.source=std::move(source);
  const auto ready=[&](std::string_view name,const c::TensorDesc &desc,std::span<const std::byte> payload) {
    return carrier::bind(name,desc,payload,owner);
  };
  for (std::size_t stage=0;stage<kStages;++stage)
    for (std::size_t slot=0;slot<kRoles;++slot) {
      const auto role=static_cast<Role>(slot);
      if (present(stage,role)) result.roles[stage][slot]=ready(raw_spec(stage,raw_role(role)).suffix,
          description(stage,role),result.source->payload(stage,role));
    }
  for (std::size_t slot=0;slot<kFinalRoles;++slot) {
    const auto role=static_cast<FinalRole>(slot);
    result.final[slot]=ready(final_spec(role).suffix,description(role),result.source->bytes(role));
  }
  return result;
}
} // namespace dspark_original_target_roles
