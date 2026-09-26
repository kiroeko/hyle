#pragma once

#include <cstddef>
#include <cstdint>

#include "ggml.h"

namespace hyle::compute::ggml_runtime {

// Private cold graph construction for dense [B,S,H,D] explicit-frequency RoPE.
// ggml_rope_ext accepts inverse factors: rounding 1/f and then p/(1/f) loses
// phase precision at long positions. Keep p*f in F32, as the public input and
// MLX lowering require. All arithmetic and storage remain native ggml nodes.
inline ggml_tensor *explicit_rope(ggml_context *context, ggml_tensor *data,
                                 ggml_tensor *positions,
                                 ggml_tensor *frequencies,
                                 std::int64_t dimensions, bool interleaved) {
  const auto output_type = data->type;
  if (data->type != GGML_TYPE_F32)
    data = ggml_cast(context, data, GGML_TYPE_F32);
  const auto pairs = dimensions / 2;
  const auto heads = data->ne[1];
  const auto sequence = data->ne[2];
  const auto batch = data->ne[3];
  auto *position = ggml_reshape_4d(
      context, ggml_cast(context, positions, GGML_TYPE_F32), 1, 1, sequence, 1);
  auto *frequency = ggml_repeat_4d(context, frequencies, pairs, 1, sequence, 1);
  auto *phase = ggml_mul(context, frequency, position);
  auto *cosine = ggml_cos(context, phase);
  auto *sine = ggml_sin(context, phase);

  const auto component = [&](bool second) {
    if (interleaved) {
      auto *view = ggml_view_4d(context, data, 1, pairs, heads, sequence * batch,
                                2 * data->nb[0], data->nb[1], data->nb[2],
                                second ? data->nb[0] : 0);
      return ggml_reshape_4d(context, ggml_cont(context, view), pairs, heads,
                             sequence, batch);
    }
    return ggml_cont(context, ggml_view_4d(
        context, data, pairs, heads, sequence, batch, data->nb[1], data->nb[2],
        data->nb[3], second ? static_cast<std::size_t>(pairs) * data->nb[0] : 0));
  };
  auto *first = component(false);
  auto *second = component(true);
  auto *rotated_first = ggml_sub(context, ggml_mul(context, first, cosine),
                                 ggml_mul(context, second, sine));
  auto *rotated_second = ggml_add(context, ggml_mul(context, first, sine),
                                  ggml_mul(context, second, cosine));
  if (interleaved) {
    rotated_first = ggml_reshape_4d(context, rotated_first, 1, pairs, heads,
                                     sequence * batch);
    rotated_second = ggml_reshape_4d(context, rotated_second, 1, pairs, heads,
                                      sequence * batch);
  }
  auto *result = ggml_concat(context, rotated_first, rotated_second, 0);
  result = ggml_reshape_4d(context, result, dimensions, heads, sequence, batch);
  if (dimensions < data->ne[0]) {
    auto *tail = ggml_view_4d(
        context, data, data->ne[0] - dimensions, heads, sequence, batch,
        data->nb[1], data->nb[2], data->nb[3],
        static_cast<std::size_t>(dimensions) * data->nb[0]);
    result = ggml_concat(context, result, tail, 0);
  }
  return output_type == GGML_TYPE_F32 ? result
                                       : ggml_cast(context, result, output_type);
}

} // namespace hyle::compute::ggml_runtime
