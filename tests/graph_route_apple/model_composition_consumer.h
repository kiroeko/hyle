#pragma once

#include <array>
#include <cmath>
#include <initializer_list>
#include <span>

#include "hyle/compute/operation.h"

namespace apple_model_consumer {
namespace c = hyle::compute;

// Model-side composition, shared by ordinary Direct and Graph consumers.
// L2 follows llama.cpp 4c9233c0 src/models/models.h:14-18. The two gates follow
// qwen35.cpp:243-251,320-325 with the C0 storage boundaries made explicit.
template <class Value, class Emit>
hyle::Result<std::array<Value, 3>> compose(
    Value input, Value projection, Value weight, c::DataType storage, int width,
    float epsilon, Emit &&emit) {
  const auto call = [&](auto operation,
                        std::initializer_list<Value> arguments)
      -> hyle::Result<Value> {
    return emit(c::Operation(operation),
                std::span<const Value>(arguments.begin(), arguments.size()));
  };
  HYLE_ASSIGN_OR_RETURN(auto fp32, call(c::Cast{c::DataType::kFloat32}, {input}));
  HYLE_ASSIGN_OR_RETURN(
      auto l2_rms, call(c::RmsNorm{.epsilon = epsilon / static_cast<float>(width)},
                       {fp32}));
  const c::Fill scale{c::TensorShape::scalar(), c::DataType::kFloat32,
                      c::ScalarValue(1.0F / std::sqrt(static_cast<float>(width)))};
  HYLE_ASSIGN_OR_RETURN(auto factor, call(scale, {}));
  HYLE_ASSIGN_OR_RETURN(
      auto scaled, call(c::Binary{c::BinaryKind::kMultiply}, {l2_rms, factor}));
  HYLE_ASSIGN_OR_RETURN(auto l2, call(c::Cast{storage}, {scaled}));

  HYLE_ASSIGN_OR_RETURN(auto gate_view, c::View::slice(-1, width, width));
  HYLE_ASSIGN_OR_RETURN(auto gate, call(gate_view, {projection}));
  HYLE_ASSIGN_OR_RETURN(
      auto normalized, call(c::RmsNorm{.epsilon = epsilon}, {input, weight}));
  // GatedActivation consumes the learned gate first. Its other input is the
  // already-stored normalization output, preserving norm-before-gate rounding.
  HYLE_ASSIGN_OR_RETURN(
      auto swish, call(c::GatedActivation{.inputs = c::GatedInputKind::kSeparate},
                       {gate, normalized}));
  HYLE_ASSIGN_OR_RETURN(
      auto sigmoid, call(c::Unary{c::UnaryKind::kSigmoid}, {gate}));
  HYLE_ASSIGN_OR_RETURN(
      auto attention, call(c::Binary{c::BinaryKind::kMultiply}, {input, sigmoid}));
  return std::array{l2, swish, attention};
}
} // namespace apple_model_consumer
