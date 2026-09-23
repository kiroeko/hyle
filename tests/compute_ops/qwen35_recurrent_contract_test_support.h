#pragma once

#include <array>
#include <cstdint>
#include <span>

#include "hyle/compute/operation.h"

namespace qwen35_recurrent_contract_test {
namespace c = hyle::compute;

// Qwen3.5-2B's first linear-attention boundary.  These are descriptor-only
// fixtures: preparation proves whether a route accepts the real model shape
// and storage types without allocating the model payload.
inline std::array<c::TensorDesc, 6>
inputs(c::DataType activation_type, std::int64_t tokens) {
  constexpr std::int64_t batch = 1;
  constexpr std::int64_t heads = 16;
  constexpr std::int64_t depth = 128;
  const auto activation =
      c::TensorDesc::dense(activation_type,
                           std::array{batch, tokens, heads, depth})
          .value();
  const auto gate =
      c::TensorDesc::dense(c::DataType::kFloat32,
                           std::array{batch, tokens, heads, std::int64_t{1}})
          .value();
  const auto state =
      c::TensorDesc::dense(c::DataType::kFloat32,
                           std::array{batch, heads, depth, depth})
          .value();
  return {activation, activation, activation, gate, gate, state};
}

inline std::array<c::TensorDesc, 2>
outputs(std::span<const c::TensorDesc> inputs) {
  std::array<c::TensorDesc, 2> result;
  auto status = c::infer_output_descriptions(c::Operation(c::GatedDeltaNet{}),
                                              inputs, result);
  if (!status.ok())
    return {};
  return result;
}

} // namespace qwen35_recurrent_contract_test
