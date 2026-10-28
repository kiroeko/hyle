#pragma once

#include <initializer_list>
#include <span>
#include <string>

#include "hyle/compute/operation.h"
#include "hyle/graph/graph.h"

// Qwen block equations in the caller's original cold Recipe. These helpers
// create no compiled program, storage, invocation or state authority. The
// caller keeps placement scopes, state Store effects and output publication.
namespace qwen35::block_graph {
using Value = hyle::graph::ValueId;
using Dimension = hyle::graph::Dimension;
using Type = hyle::compute::DataType;
namespace compute = hyle::compute;

// absolute_prefix identifies either the target layer or the trained NextN
// layer. Optional target probes use the original nine-value layer0 inventory.
template<class R>
Value emit_ffn(R &r, Value residual, const std::string &absolute_prefix,
               std::span<Value> probes = {}) {
  const auto normalized = r.norm_absolute(residual, absolute_prefix + "post_attention_layernorm.weight");
  const auto projected_gate = r.linear_absolute(normalized, absolute_prefix + "mlp.gate_proj.weight");
  const auto gate = r.unary(compute::UnaryKind::kSilu, projected_gate);
  const auto up = r.linear_absolute(normalized, absolute_prefix + "mlp.up_proj.weight");
  const auto product = r.binary(compute::BinaryKind::kMultiply, gate, up);
  if (!probes.empty()) {
    probes[3] = product; probes[5] = residual; probes[6] = normalized;
    probes[7] = projected_gate; probes[8] = up;
  }
  const auto result = r.linear_absolute(product, absolute_prefix + "mlp.down_proj.weight");
  return r.binary(compute::BinaryKind::kAdd, residual, result);
}

// Called only after the caller emitted the original recurrent Store. Keep the
// packed [R,Hv,Dv] or exact [B,T,Hv,Dv] dimensions supplied by that caller;
// do not rebuild symbolic dimensions or normalize the physical row layout.
// Optional probes retain the original recurrent/gated/projected activation
// values, including the value before the metadata-only out-projection reshape.
template<class R>
Value emit_gdn_epilogue(R &r, Value recurrent, Value normalized,
    const std::string &absolute_prefix, std::initializer_list<Dimension> row_shape,
    Dimension packed_rows, std::span<Value> probes = {}) {
  auto y = r.cast(recurrent, r.activation_type());
  if (!probes.empty()) probes[0] = y;
  const auto y_f32 = r.cast(y, Type::kFloat32);
  y = r.cast(r.same(compute::RmsNorm{.epsilon = r.geometry.epsilon}, {y_f32}, y_f32), r.activation_type());
  // Official F32 and converted BF16 norm scales keep their actual product
  // rounding. Do not fold this scale into the RMS operation.
  const auto norm_scale = r.weight_absolute(absolute_prefix + "linear_attn.norm.weight");
  y = r.binary(compute::BinaryKind::kMultiply, r.cast(y, r.spec(norm_scale).data_type()), norm_scale);
  const auto z = r.reshape(r.linear_absolute(normalized, absolute_prefix + "linear_attn.in_proj_z.weight"), row_shape);
  y = r.cast(r.binary(compute::BinaryKind::kMultiply, r.cast(y, Type::kFloat32),
      r.unary(compute::UnaryKind::kSilu, r.cast(z, Type::kFloat32))), r.activation_type());
  const auto projection_input = r.reshape(y,
      {packed_rows, Dimension::fixed(r.geometry.gdn_value_width())});
  if (!probes.empty()) probes[1] = y;
  const auto result = r.linear_absolute(projection_input, absolute_prefix + "linear_attn.out_proj.weight");
  if (!probes.empty()) probes[2] = result;
  return result;
}
} // namespace qwen35::block_graph
