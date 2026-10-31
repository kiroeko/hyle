#pragma once

#include "../../engine/models/kimi_k3/roles.h"
#include "d5_kda_full_fragments.h"
#include "d5_kimi_remaining_fragments.h"

// Test Fragment adapter only. Fixed source validation belongs to the model.
namespace kimi_original_attention_values {
namespace source=hyle::inference::kimi_k3::roles;
namespace kda=d5_kda_full_fragments;
namespace mla=d5_kimi_remaining_fragments;
inline constexpr auto repository=source::repository,revision=source::revision;
inline constexpr auto kda_file=source::kda_file,mla_file=source::mla_file;
using Kda=source::KdaView;
using Mla=source::MlaView;
template<class Constant> kda::KdaParameters declare_kda(const Kda &p,Constant &&constant) {
  source::validate(p);
  return {constant(p.q), constant(p.k), constant(p.v), constant(p.f_a),
      constant(p.f_b), constant(p.beta), constant(p.gate), constant(p.output),
      constant(p.convolution), constant(p.a_log), constant(p.dt_bias),
      constant(p.output_norm), std::nullopt, kda::DecayEncoding::kLog};
}
template<class Constant> mla::MlaParameters declare_mla(const Mla &p,Constant &&constant) {
  source::validate(p);
  return {constant(p.input_norm), constant(p.q_a), constant(p.q_norm),
      constant(p.q_b), constant(p.kv_a), constant(p.kv_norm),
      constant(p.absorbed_key), constant(p.absorbed_value), constant(p.gate), constant(p.output)};
}
} // namespace kimi_original_attention_values
