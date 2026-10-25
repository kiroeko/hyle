#pragma once

#include "original_named_role_source.h"
#include "d5_kda_full_fragments.h"
#include "d5_kimi_remaining_fragments.h"

// Cold parameter binding. ReadonlyRole objects are the existing
// authenticated native operand owners, not another weight/state owner table.
// Their source reader and the bit-preserving layout preparation are separate.
namespace kimi_original_attention_values {
namespace source = original_named_role_source;
namespace kda = d5_kda_full_fragments;
namespace mla = d5_kimi_remaining_fragments;
namespace sem = d5_qwen38_24t_semantic_fragments;
namespace c = hyle::compute;

inline constexpr std::string_view repository = "moonshotai/Kimi-K3";
inline constexpr std::string_view revision = "f831ab66814297da540d832a5235f8e904f29d06";
inline constexpr std::string_view kda_file = "model-00001-of-000096.safetensors";
inline constexpr std::string_view mla_file = "model-00004-of-000096.safetensors";

inline void require(const source::ReadonlyRole &role, std::string_view name,
    std::string_view file, std::initializer_list<std::int64_t> shape,
    c::DataType type) {
  const auto *identity = role.identity;
  if (role.name != name || !identity || identity->repository != repository ||
      identity->revision != revision || identity->file != file ||
      role.description != sem::dense(shape, type) || !role.owner || !role.lease ||
      role.lease.size() != source::detail::take(role.description.required_bytes(), "original Kimi operand extent"))
    throw std::runtime_error("original Kimi supplied role identity/description/lease differs");
}

// Borrow actual native operands. Packing the three original convolution banks
// produces one operand; reshaping A/dt preserves their original F32 bytes.
struct Kda final {
  const source::ReadonlyRole &q, &k, &v, &f_a, &f_b, &beta, &gate, &output;
  const source::ReadonlyRole &convolution, &a_log, &dt_bias, &output_norm;
};

// Constant is the consumer's cold Graph-constant declaration/binding function.
// It must retain the supplied original ReadLease in the existing Run literals
// and bind it through GraphConstantBinding; it never generates a weight bank.
template<class Constant>
kda::KdaParameters declare_kda(const Kda &p, Constant &&constant) {
  constexpr auto bf16 = c::DataType::kBFloat16;
  constexpr auto f32 = c::DataType::kFloat32;
  require(p.q, "language_model.model.layers.0.self_attn.q_proj.weight", kda_file, {12288,7168}, bf16);
  require(p.k, "language_model.model.layers.0.self_attn.k_proj.weight", kda_file, {12288,7168}, bf16);
  require(p.v, "language_model.model.layers.0.self_attn.v_proj.weight", kda_file, {12288,7168}, bf16);
  require(p.f_a, "language_model.model.layers.0.self_attn.f_a_proj.weight", kda_file, {128,7168}, bf16);
  require(p.f_b, "language_model.model.layers.0.self_attn.f_b_proj.weight", kda_file, {12288,128}, bf16);
  require(p.beta, "language_model.model.layers.0.self_attn.b_proj.weight", kda_file, {96,7168}, bf16);
  require(p.gate, "language_model.model.layers.0.self_attn.g_proj.weight", kda_file, {12288,7168}, bf16);
  require(p.output, "language_model.model.layers.0.self_attn.o_proj.weight", kda_file, {7168,12288}, bf16);
  require(p.convolution, "original layer0 QKV convolution", kda_file, {36864,4}, f32);
  require(p.a_log, "language_model.model.layers.0.self_attn.A_log", kda_file, {128,1}, f32);
  require(p.dt_bias, "language_model.model.layers.0.self_attn.dt_bias", kda_file, {1,1,96,128}, f32);
  require(p.output_norm, "language_model.model.layers.0.self_attn.o_norm.weight", kda_file, {128}, f32);
  // Every original operand is checked before the first declaration mutates f.
  return {constant(p.q), constant(p.k), constant(p.v), constant(p.f_a),
      constant(p.f_b), constant(p.beta), constant(p.gate), constant(p.output),
      constant(p.convolution), constant(p.a_log), constant(p.dt_bias),
      constant(p.output_norm), std::nullopt, kda::DecayEncoding::kLog};
}

struct Mla final {
  const source::ReadonlyRole &input_norm, &q_a, &q_norm, &q_b, &kv_a, &kv_norm;
  const source::ReadonlyRole &absorbed_key, &absorbed_value, &gate, &output;
};

template<class Constant>
mla::MlaParameters declare_mla(const Mla &p, Constant &&constant) {
  constexpr auto bf16 = c::DataType::kBFloat16;
  require(p.input_norm, "language_model.model.layers.3.input_layernorm.weight", mla_file, {7168}, bf16);
  require(p.q_a, "language_model.model.layers.3.self_attn.q_a_proj.weight", mla_file, {1536,7168}, bf16);
  require(p.q_norm, "language_model.model.layers.3.self_attn.q_a_layernorm.weight", mla_file, {1536}, bf16);
  require(p.q_b, "language_model.model.layers.3.self_attn.q_b_proj.weight", mla_file, {18432,1536}, bf16);
  require(p.kv_a, "language_model.model.layers.3.self_attn.kv_a_proj_with_mqa.weight", mla_file, {576,7168}, bf16);
  require(p.kv_norm, "language_model.model.layers.3.self_attn.kv_a_layernorm.weight", mla_file, {512}, bf16);
  // Both derived operands retain the true KV-b role as their source identity.
  require(p.absorbed_key, "language_model.model.layers.3.self_attn.kv_b_proj.weight", mla_file, {96,512,128}, bf16);
  require(p.absorbed_value, "language_model.model.layers.3.self_attn.kv_b_proj.weight", mla_file, {96,128,512}, bf16);
  require(p.gate, "language_model.model.layers.3.self_attn.g_proj.weight", mla_file, {12288,7168}, bf16);
  require(p.output, "language_model.model.layers.3.self_attn.o_proj.weight", mla_file, {7168,12288}, bf16);
  return {constant(p.input_norm), constant(p.q_a), constant(p.q_norm),
      constant(p.q_b), constant(p.kv_a), constant(p.kv_norm),
      constant(p.absorbed_key), constant(p.absorbed_value), constant(p.gate), constant(p.output)};
}
} // namespace kimi_original_attention_values
