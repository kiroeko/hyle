#pragma once

#include <algorithm>
#include <bit>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "model_geometry.h"

// The prepared input describes original physical storage, including disjoint
// cached source ranges. It does not prescribe an executor or a Provider.
namespace qwen35::prepared {
inline constexpr std::string_view role_signature = "HYLE_QWEN_ROLE_FILES_V2";
inline constexpr std::string_view gguf_role_signature = "HYLE_QWEN_GGUF_ROLE_FILES_V2";
inline constexpr std::size_t maximum_manifest_line_bytes = 64u << 10;
inline constexpr std::size_t maximum_manifest_path_bytes = 32u << 10;
// A logical parameter has at most the four original GPTQ/NVFP4 components.
inline bool read_manifest_line(std::istream &input, std::string &line,
                               std::size_t limit = maximum_manifest_line_bytes) {
  line.clear();
  char ch;
  while (input.get(ch)) {
    if (ch == '\n') return true;
    if (line.size() == limit) throw std::runtime_error("prepared manifest line length limit exceeded");
    line += ch;
  }
  if (input.bad()) throw std::runtime_error("failed reading prepared manifest");
  return !line.empty();
}
enum class SourceRepresentation { kHfOriginal, kGgufConverted };
enum class ParameterMeaning {
  kHfRaw, kLinearWeight, kEmbeddingRows, kRmsEffectiveGamma,
  kGdnEffectiveGamma, kGdnNegativeExpDecay, kGdnDtBias, kGdnConvolution
};
// Derived from a complete physical role group, never a user/provider switch.
// GGUF conversion retains its independent storage descriptor and parameter
// meaning; this discriminator describes original HF projection groups.
enum class ProjectionCodec { kUnspecified, kDense, kGptqV1Int4, kNvfp4, kCtSymmetric4, kFloat8Block128,
                             kCtFloat8ChannelToken, kCtLocalNvfp4 };
// This is an explicit selected-reader arithmetic contract, not a claim that
// every reader of the HF config chooses the same activation quantization.
inline constexpr std::string_view fp8_reader = "sglang@06992c92ad4494fe9f9636e75be3e35040512447";
inline constexpr std::string_view fp8_policy_line =
    "fp8_block128\tdivide\t2edbe6ff\tBF16\tF32\tBF16\tbf16_to_f32_exact\tsglang@06992c92ad4494fe9f9636e75be3e35040512447";
struct Float8BlockPolicy final {
  bool enabled = false;
  static constexpr float floor = std::bit_cast<float>(std::uint32_t{0x2edbe6ff});
};
inline constexpr std::string_view ct_mixed_policy_line =
    "ct_mixed\tchannel_token_bf16_zero_only\tlocal_nvfp4_f32\tcompressed-tensors@0.17.2a20260716";
struct CtMixedPolicy final { bool enabled = false; };
inline constexpr std::string_view bf16_cache_policy_line = "kv_cache\tBF16\tomit_original_calibration";
struct Bf16CachePolicy final { bool omit_calibration = false; };
struct OriginalSource final {
  std::string repository, revision, file, lfs_sha256, xet_hash, header_sha256;
  std::uint64_t bytes = 0, header_bytes = 0;
};
struct Role final {
  std::string name, dtype;
  std::vector<std::int64_t> shape;
  std::filesystem::path source;
  std::uint64_t offset = 0;
  std::size_t bytes = 0;
  std::string physical_name{};
  ParameterMeaning meaning = ParameterMeaning::kHfRaw;
  std::uint64_t original_offset = 0;
  ProjectionCodec projection_codec = ProjectionCodec::kUnspecified;
  std::uint32_t projection_group_size = 0;
};
struct Parameter final {
  std::vector<std::int64_t> shape;
  bool linear = false, allow_f32 = false;
};
inline std::map<std::string, Parameter> parameter_inventory(bool mtp, const ModelGeometry &model) {
  model.validate();
  if (mtp && model.mtp_layers != 1) throw std::runtime_error("source geometry has no supported one-block NextN");
  const auto h = model.hidden, ff = model.intermediate;
  std::map<std::string, Parameter> result;
  const auto add = [&](std::string name, std::initializer_list<std::int64_t> shape,
                       bool linear = false, bool allow_f32 = false) {
    result.emplace(std::move(name), Parameter{{shape.begin(), shape.end()}, linear, allow_f32});
  };
  add("model.language_model.embed_tokens.weight", {model.vocabulary, h});
  if (!model.tied_head) add("lm_head.weight", {model.vocabulary, h}, true);
  add("model.language_model.norm.weight", {h}, false, true);
  const auto full_attention = [&](const std::string &p) {
    add(p + "self_attn.q_proj.weight", {model.query_gate_width(), h}, true);
    add(p + "self_attn.k_proj.weight", {ModelGeometry::multiply(model.kv_heads, model.key_depth), h}, true);
    add(p + "self_attn.v_proj.weight", {ModelGeometry::multiply(model.kv_heads, model.value_depth), h}, true);
    add(p + "self_attn.o_proj.weight", {h, model.attention_width()}, true);
    add(p + "self_attn.q_norm.weight", {model.key_depth}, false, true);
    add(p + "self_attn.k_norm.weight", {model.key_depth}, false, true);
  };
  const auto common = [&](const std::string &p) {
    add(p + "input_layernorm.weight", {h}, false, true);
    add(p + "post_attention_layernorm.weight", {h}, false, true);
    add(p + "mlp.gate_proj.weight", {ff, h}, true);
    add(p + "mlp.up_proj.weight", {ff, h}, true);
    add(p + "mlp.down_proj.weight", {h, ff}, true);
  };
  for (std::size_t i = 0; i < model.layers.size(); ++i) {
    const auto p = "model.language_model.layers." + std::to_string(i) + ".";
    common(p);
    if (!model.is_gdn(i)) full_attention(p);
    else {
      add(p + "linear_attn.A_log", {model.gdn_value_heads}, false, true);
      add(p + "linear_attn.dt_bias", {model.gdn_value_heads}, false, true);
      add(p + "linear_attn.conv1d.weight", {model.gdn_qkv_width(), 1, model.conv_kernel}, false, true);
      add(p + "linear_attn.norm.weight", {model.gdn_value_depth}, false, true);
      add(p + "linear_attn.in_proj_qkv.weight", {model.gdn_qkv_width(), h}, true);
      add(p + "linear_attn.in_proj_z.weight", {model.gdn_value_width(), h}, true);
      add(p + "linear_attn.in_proj_a.weight", {model.gdn_value_heads, h}, true);
      add(p + "linear_attn.in_proj_b.weight", {model.gdn_value_heads, h}, true);
      add(p + "linear_attn.out_proj.weight", {h, model.gdn_value_width()}, true);
    }
  }
  if (mtp) {
    add("mtp.fc.weight", {h, ModelGeometry::multiply(2, h)}, true);
    for (const auto *name : {"mtp.norm.weight", "mtp.pre_fc_norm_embedding.weight",
                             "mtp.pre_fc_norm_hidden.weight"}) add(name, {h}, false, true);
    common("mtp.layers.0."); full_attention("mtp.layers.0.");
  }
  return result;
}
struct Roles final {
  std::string identity;
  std::vector<Role> selected;
  std::size_t bytes = 0;
  SourceRepresentation representation = SourceRepresentation::kHfOriginal;
  OriginalSource original{};
  ModelGeometry geometry{};
  Float8BlockPolicy float8_policy{};
  CtMixedPolicy ct_mixed_policy{};
  std::vector<Role> omitted_cache_calibration{};
};
inline Roles read_gguf_roles(const std::filesystem::path &path, bool include_mtp);
inline bool is_role_signature(std::string_view signature) noexcept {
  if (signature.ends_with('\r')) signature.remove_suffix(1);
  return signature == role_signature || signature == gguf_role_signature;
}
inline std::uint64_t unsigned_field(std::istream &input) {
  std::string word;
  if (!(input >> word)) throw std::runtime_error("missing prepared role integer");
  std::uint64_t result = 0;
  const auto parsed = std::from_chars(word.data(), word.data() + word.size(), result);
  if (parsed.ec != std::errc{} || parsed.ptr != word.data() + word.size())
    throw std::runtime_error("invalid or overflowing prepared role integer");
  return result;
}
inline std::size_t element_bytes(std::string_view dtype) {
  if (dtype == "BF16" || dtype == "F16") return 2;
  if (dtype == "F32" || dtype == "I32") return 4;
  if (dtype == "I64") return 8;
  if (dtype == "U8" || dtype == "F8_E4M3") return 1;
  throw std::runtime_error("unsupported prepared role dtype: " + std::string(dtype));
}
inline void validate_identity(std::string_view identity) {
  if (identity.size() != 64 ||
      !std::ranges::all_of(identity, [](char ch) {
        return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
      })) throw std::runtime_error("invalid prepared role source identity");
}
inline std::size_t physical_bytes(std::string_view dtype, const std::vector<std::int64_t> &shape) {
  if (shape.size() > 3) throw std::runtime_error("prepared Qwen role rank exceeds model geometry");
  std::size_t bytes = element_bytes(dtype);
  for (const auto extent : shape) {
    if (extent <= 0 || static_cast<std::uint64_t>(extent) > (std::numeric_limits<std::size_t>::max)() / bytes)
      throw std::runtime_error("invalid or overflowing prepared role extent");
    bytes *= static_cast<std::size_t>(extent);
  }
  return bytes;
}
// Loading remains outside metadata selection. The loader calls this on the
// original16 host bytes before admitting a CT projection into its weight bank.
inline void validate_ct_shape_values(const Role &codes, std::span<const std::byte> original) {
  if (codes.projection_codec != ProjectionCodec::kCtSymmetric4 || codes.dtype != "I32" ||
      codes.shape.size() != 2 || codes.shape[0] <= 0 || codes.shape[1] <= 0 ||
      codes.shape[1] > INT64_MAX / 8 || original.size() != 16)
    throw std::runtime_error("invalid original CT shape value input");
  const auto field = [&](std::size_t first) {
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < 8; ++i)
      value |= static_cast<std::uint64_t>(std::to_integer<unsigned char>(original[first + i])) << (8 * i);
    return value;
  };
  if (field(0) != static_cast<std::uint64_t>(codes.shape[0]) ||
      field(8) != static_cast<std::uint64_t>(codes.shape[1] * 8))
    throw std::runtime_error("original CT weight_shape values differ: " + codes.name);
}
// Both prepared file layouts enter through the same physical/logical contract.
// Validation covers the entire inventory even when MTP is not selected for IO.
inline Roles select_roles(const std::map<std::string, Role> &all, std::string identity, bool include_mtp, const ModelGeometry &geometry,
                          Float8BlockPolicy float8_policy = {}, CtMixedPolicy ct_mixed_policy = {},
                          Bf16CachePolicy bf16_cache_policy = {}) {
  geometry.validate();
  if (geometry.gdn_layout != GdnHeadLayout::kGrouped) throw std::runtime_error("HF roles require their original grouped GDN layout");
  const auto maximum_hf_physical_roles = parameter_inventory(geometry.mtp_layers != 0, geometry).size() * 4;
  validate_identity(identity);
  if (all.size() > maximum_hf_physical_roles)
    throw std::runtime_error("prepared physical role count exceeds model limit");
  Roles result{std::move(identity), {}, 0}; result.geometry = geometry;
  result.float8_policy = float8_policy;
  result.ct_mixed_policy = ct_mixed_policy;
  bool has_float8_block = false;
  bool has_ct_mixed = false;
  for (const auto &[name, role] : all) {
    if (name != role.name) throw std::runtime_error("prepared role name/key differs");
    if (!role.physical_name.empty() || role.meaning != ParameterMeaning::kHfRaw || role.original_offset)
      throw std::runtime_error("HF prepared roles cannot describe converted GGUF parameters");
    if (role.source.empty() || role.source.native().find(std::filesystem::path::value_type{}) !=
        std::filesystem::path::string_type::npos)
      throw std::runtime_error("prepared role requires a source file");
    if (role.bytes != physical_bytes(role.dtype, role.shape) || role.bytes > UINT64_MAX - role.offset)
      throw std::runtime_error("prepared role byte extent differs: " + name);
  }
  const bool has_mtp = std::ranges::any_of(all, [](const auto &entry) { return entry.first.starts_with("mtp."); });
  if (include_mtp && !has_mtp) throw std::runtime_error("MTP requires prepared MTP roles");
  const auto inventory = parameter_inventory(has_mtp, geometry);
  std::set<std::string> consumed;
  for (const auto &[name, parameter] : inventory) {
    const bool selected = include_mtp || !name.starts_with("mtp.");
    const auto take = [&](const std::string &physical, std::string_view dtype,
                          std::vector<std::int64_t> shape,
                          ProjectionCodec codec = ProjectionCodec::kUnspecified,
                          std::uint32_t group_size = 0) {
      const auto found = all.find(physical);
      if (found == all.end()) throw std::runtime_error("missing prepared role: " + physical);
      const auto &role = found->second;
      if (role.dtype != dtype || role.shape != shape)
        throw std::runtime_error("prepared role type/shape differs: " + physical);
      if (!consumed.insert(physical).second) throw std::runtime_error("reused prepared physical role");
      if (selected) {
        if (role.bytes > (std::numeric_limits<std::size_t>::max)() - result.bytes)
          throw std::runtime_error("prepared model byte inventory overflows");
        result.bytes += role.bytes;
        auto derived = role;
        derived.projection_codec = codec;
        derived.projection_group_size = group_size;
        result.selected.push_back(std::move(derived));
      }
    };
    const auto base = name.ends_with(".weight") ? name.substr(0, name.size() - 7) : name;
    if (parameter.linear && all.contains(base + ".weight_packed") &&
        (all.at(base + ".weight_packed").dtype == "U8" ||
         all.contains(base + ".weight_global_scale") || all.contains(base + ".input_global_scale"))) {
      if (!ct_mixed_policy.enabled) throw std::runtime_error("prepared CT local NVFP4 requires its fixed reader policy");
      const auto n = parameter.shape[0], k = parameter.shape[1];
      if (k % 16 != 0) throw std::runtime_error("prepared CT local NVFP4 needs complete K groups: " + name);
      has_ct_mixed = true;
      take(base + ".weight_packed", "U8", {n, k / 2}, ProjectionCodec::kCtLocalNvfp4, 16);
      take(base + ".weight_scale", "F8_E4M3", {n, k / 16});
      take(base + ".weight_global_scale", "F32", {1});
      take(base + ".input_global_scale", "F32", {1});
    } else if (parameter.linear && (all.contains(base + ".weight_packed") || all.contains(base + ".weight_shape"))) {
      const auto n = parameter.shape[0], k = parameter.shape[1];
      const auto scales = all.find(base + ".weight_scale");
      if (scales == all.end() || scales->second.shape.size() != 2 ||
          scales->second.shape[0] != n || scales->second.shape[1] <= 0 ||
          k % scales->second.shape[1] != 0 || k % 8 != 0 ||
          k / scales->second.shape[1] > UINT32_MAX)
        throw std::runtime_error("invalid prepared CT symmetric group geometry: " + name);
      const auto &scale_type = scales->second.dtype;
      if (scale_type != "BF16" && scale_type != "F16" && scale_type != "F32")
        throw std::runtime_error("prepared CT symmetric scale type differs: " + name);
      const auto groups = scales->second.shape[1];
      take(base + ".weight_packed", "I32", {n, k / 8}, ProjectionCodec::kCtSymmetric4,
          static_cast<std::uint32_t>(k / groups));
      take(base + ".weight_scale", scale_type, {n, groups});
      // The loader must check the two original little-endian I64 values against
      // [N,K]. Metadata selection does not read payloads or pretend to know them.
      take(base + ".weight_shape", "I64", {2});
    } else if (parameter.linear && all.contains(base + ".qweight")) {
      const auto n = parameter.shape[0], k = parameter.shape[1];
      const auto scales = all.find(base + ".scales");
      if (k % 8 != 0 || n % 8 != 0 || scales == all.end() || scales->second.shape.size() != 2 ||
          scales->second.shape[0] <= 0 || k % scales->second.shape[0] != 0 ||
          k / scales->second.shape[0] > UINT32_MAX)
        throw std::runtime_error("invalid prepared GPTQ group geometry: " + name);
      const auto groups = scales->second.shape[0];
      take(base + ".qweight", "I32", {k / 8, n}, ProjectionCodec::kGptqV1Int4,
          static_cast<std::uint32_t>(k / groups));
      take(base + ".qzeros", "I32", {groups, n / 8});
      take(base + ".scales", "F16", {groups, n});
      take(base + ".g_idx", "I32", {k});
    } else if (parameter.linear && all.contains(base + ".weight_scale") &&
               (all.contains(base + ".weight_scale_2") || all.contains(base + ".input_scale"))) {
      const auto n = parameter.shape[0], k = parameter.shape[1];
      if (k % 16 != 0)
        throw std::runtime_error("prepared NVFP4 weight needs complete K groups: " + name);
      take(name, "U8", {n, k / 2}, ProjectionCodec::kNvfp4, 16);
      take(base + ".weight_scale", "F8_E4M3", {n, k / 16});
      for (const auto *suffix : {".weight_scale_2", ".input_scale"}) {
        const auto scalar = all.find(base + suffix);
        if (scalar == all.end() || (!scalar->second.shape.empty() && scalar->second.shape != std::vector<std::int64_t>{1}))
          throw std::runtime_error("invalid prepared NVFP4 scalar: " + base + suffix);
        take(base + suffix, "F32", scalar->second.shape);
      }
    } else if (parameter.linear && all.contains(name) && all.at(name).dtype == "F8_E4M3" &&
               all.contains(base + ".weight_scale")) {
      if (!ct_mixed_policy.enabled) throw std::runtime_error("prepared CT channel/token FP8 requires its fixed reader policy");
      const auto n = parameter.shape[0], k = parameter.shape[1];
      has_ct_mixed = true;
      take(name, "F8_E4M3", {n, k}, ProjectionCodec::kCtFloat8ChannelToken);
      take(base + ".weight_scale", "BF16", {n, 1});
    } else if (parameter.linear && (all.contains(base + ".weight_scale_inv") ||
               (all.contains(name) && all.at(name).dtype == "F8_E4M3"))) {
      const auto n = parameter.shape[0], k = parameter.shape[1];
      if (k % 128 != 0) throw std::runtime_error("prepared FP8 block128 needs complete K blocks: " + name);
      if (!float8_policy.enabled) throw std::runtime_error("prepared FP8 block128 requires an explicit reader policy");
      has_float8_block = true;
      take(name, "F8_E4M3", {n, k}, ProjectionCodec::kFloat8Block128, 128);
      take(base + ".weight_scale_inv", "BF16", {(n + 127) / 128, k / 128});
    } else {
      const auto found = all.find(name);
      const auto dtype = found != all.end() && parameter.allow_f32 && found->second.dtype == "F32" ? "F32" : "BF16";
      take(name, dtype, parameter.shape,
          parameter.linear ? ProjectionCodec::kDense : ProjectionCodec::kUnspecified);
    }
  }
  // The manifest explicitly selects BF16 KV storage. Only the original
  // full-attention calibration pair is omitted; arbitrary suffixes stay errors.
  for (std::size_t i = 0; i < geometry.layers.size(); ++i) if (!geometry.is_gdn(i)) {
    const auto base = "model.language_model.layers." + std::to_string(i) + ".self_attn.";
    if (!all.contains(base + "k_scale") && !all.contains(base + "v_scale")) continue;
    if (!ct_mixed_policy.enabled || !bf16_cache_policy.omit_calibration)
      throw std::runtime_error("cache calibration omission requires explicit CT BF16 KV policy");
    for (const auto *suffix : {"k_scale", "v_scale"}) {
      const auto found = all.find(base + suffix);
      if (found == all.end() || found->second.dtype != "BF16" || found->second.shape != std::vector<std::int64_t>{1})
        throw std::runtime_error("invalid original BF16 cache calibration pair: " + base + suffix);
      consumed.insert(found->first);
      result.omitted_cache_calibration.push_back(found->second);
    }
  }
  if (float8_policy.enabled && !has_float8_block) throw std::runtime_error("prepared FP8 policy has no block128 weights");
  if (ct_mixed_policy.enabled && !has_ct_mixed) throw std::runtime_error("prepared CT reader policy has no matching weights");
  if (bf16_cache_policy.omit_calibration && result.omitted_cache_calibration.empty())
    throw std::runtime_error("prepared BF16 cache omission has no calibration roles");
  if (consumed.size() != all.size()) throw std::runtime_error("unclaimed prepared physical role");
  return result;
}
inline Roles read_roles(const std::filesystem::path &path, bool include_mtp) {
  std::ifstream input(path);
  std::string signature, identity;
  read_manifest_line(input, signature, 128); read_manifest_line(input, identity, 65);
  // A prepared manifest may move between Windows and Unix without rewriting it.
  if (signature.ends_with('\r')) signature.pop_back();
  if (identity.ends_with('\r')) identity.pop_back();
  if (signature == gguf_role_signature) return read_gguf_roles(path, include_mtp);
  if (signature != role_signature) throw std::runtime_error("invalid prepared role signature");
  validate_identity(identity);
  std::string geometry_line; read_manifest_line(input, geometry_line);
  const auto geometry = read_model_geometry(geometry_line);
  const auto maximum_hf_physical_roles = parameter_inventory(geometry.mtp_layers != 0, geometry).size() * 4;
  std::map<std::string, Role> all;
  Float8BlockPolicy float8_policy;
  CtMixedPolicy ct_mixed_policy;
  Bf16CachePolicy bf16_cache_policy;
  std::string line;
  while (read_manifest_line(input, line)) {
    if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
    if (line.starts_with("kv_cache")) {
      if (line.ends_with('\r')) line.pop_back();
      if (bf16_cache_policy.omit_calibration || !ct_mixed_policy.enabled || !all.empty() || line != bf16_cache_policy_line)
        throw std::runtime_error("invalid or misplaced prepared BF16 cache omission policy");
      bf16_cache_policy.omit_calibration = true; continue;
    }
    if (line.starts_with("ct_mixed")) {
      if (line.ends_with('\r')) line.pop_back();
      if (ct_mixed_policy.enabled || float8_policy.enabled || !all.empty() || line != ct_mixed_policy_line)
        throw std::runtime_error("invalid or misplaced prepared CT reader/cache policy");
      ct_mixed_policy.enabled = true; continue;
    }
    if (line.starts_with("fp8_block128")) {
      if (line.ends_with('\r')) line.pop_back();
      if (float8_policy.enabled || ct_mixed_policy.enabled || !all.empty() || line != fp8_policy_line)
        throw std::runtime_error("invalid or misplaced prepared FP8 reader policy");
      float8_policy.enabled = true; continue;
    }
    if (all.size() == maximum_hf_physical_roles)
      throw std::runtime_error("prepared physical role count exceeds model limit");
    std::istringstream fields(line);
    Role role;
    if (!(fields >> role.name >> role.dtype)) throw std::runtime_error("invalid prepared role");
    const auto rank = unsigned_field(fields);
    if (rank > 3) throw std::runtime_error("prepared Qwen role rank exceeds model geometry");
    for (std::size_t i = 0; i < rank; ++i) {
      const auto extent = unsigned_field(fields);
      if (extent > static_cast<std::uint64_t>((std::numeric_limits<std::int64_t>::max)()))
        throw std::runtime_error("invalid or overflowing prepared role extent");
      role.shape.push_back(static_cast<std::int64_t>(extent));
    }
    const auto bytes = physical_bytes(role.dtype, role.shape);
    std::string source, extra;
    fields >> std::ws;
    if (fields.peek() != '"' || !(fields >> std::quoted(source)) || source.empty() || source.find('\0') != std::string::npos)
      throw std::runtime_error("prepared role requires a quoted source file");
    if (source.size() > maximum_manifest_path_bytes)
      throw std::runtime_error("prepared source path length limit exceeded");
    role.source = path.parent_path() / std::filesystem::path(source);
    role.offset = unsigned_field(fields);
    const auto declared = unsigned_field(fields);
    if (declared != bytes || declared > UINT64_MAX - role.offset || (fields >> extra))
      throw std::runtime_error("prepared role byte extent differs: " + role.name);
    role.bytes = bytes;
    if (!all.emplace(role.name, std::move(role)).second)
      throw std::runtime_error("duplicate prepared role");
  }
  if (!input.eof()) throw std::runtime_error("failed reading prepared role manifest");
  return select_roles(all, std::move(identity), include_mtp, geometry, float8_policy, ct_mixed_policy, bf16_cache_policy);
}
} // namespace qwen35::prepared

#include "prepared_gguf_roles.h"
