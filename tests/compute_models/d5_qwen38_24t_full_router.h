#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "d5_qwen38_24t_fixture.h"
#include "hyle/compute/model_ops.h"
#include "hyle/graph/graph.h"

namespace d5_qwen38_24t_full_router {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace q = d5_qwen38_24t_fixture;

using BF16 = q::BF16;

inline constexpr int kRoutedSeed = 71;
inline constexpr int kSharedSeed = 113;

template <std::size_t N>
c::TensorDesc dense(c::DataType type,
                    const std::array<std::int64_t, N> &dimensions) {
  return c::TensorDesc::dense(type, dimensions).value();
}

// Constant order is the Graph's declared constant order. All four route
// consumers use this one model-shaped fixture rather than rebuilding a
// different synthetic router for each backend.
struct Weights final {
  std::vector<BF16> router;
  std::array<std::int32_t, q::kTopK> replication{};
  std::vector<BF16> routed_gate;
  std::vector<BF16> routed_up;
  std::vector<BF16> routed_down;
  std::vector<BF16> shared_gate;
  std::vector<BF16> shared_up;
  std::vector<BF16> shared_down;
  std::vector<BF16> shared_gate_input;
  std::array<std::int32_t, 1> shared_ids{0};
};

inline Weights make_weights() {
  Weights weights;
  weights.router.resize(q::kExperts * q::kHidden);
  for (std::int64_t expert = 0; expert < q::kTopK; ++expert)
    weights.router[expert * q::kHidden + expert] = q::encode(1.0);
  weights.router[q::kTopK * q::kHidden + 20] = q::encode(1.0);
  weights.router[q::kTopK * q::kHidden + 21] = q::encode(1.0);
  weights.routed_gate =
      q::expert_weight(q::kExpertHidden, q::kHidden, kRoutedSeed,
                       q::kRouterResidentExperts);
  weights.routed_up =
      q::expert_weight(q::kExpertHidden, q::kHidden, kRoutedSeed + 1,
                       q::kRouterResidentExperts);
  weights.routed_down =
      q::expert_weight(q::kHidden, q::kExpertHidden, kRoutedSeed + 2,
                       q::kRouterResidentExperts);
  weights.shared_gate = q::sparse_output_major_weight(
      q::kSharedExpertHidden, q::kHidden, kSharedSeed);
  weights.shared_up = q::sparse_output_major_weight(
      q::kSharedExpertHidden, q::kHidden, kSharedSeed + 1);
  weights.shared_down = q::sparse_output_major_weight(
      q::kHidden, q::kSharedExpertHidden, kSharedSeed + 2);
  weights.shared_gate_input =
      q::sparse_output_major_weight(1, q::kHidden, kSharedSeed + 3);
  return weights;
}

struct Expected final {
  std::vector<BF16> activation;
  std::array<std::int32_t, q::kTopK> ids{};
  std::vector<float> routed;
  std::vector<float> combined;
  std::vector<float> shared;
  std::vector<float> full;
};

inline Expected expected(int replay) {
  Expected result;
  const auto base = q::expert_input(replay);
  result.activation.resize(q::kHidden);
  for (std::int64_t column = 0; column < q::kHidden; ++column)
    result.activation[column] = q::encode(base[column]);
  if (replay == 29) {
    for (std::int64_t expert = 0; expert < q::kTopK - 1; ++expert)
      result.activation[expert] = q::encode(2.0 + expert / 8.0);
    result.activation[q::kTopK - 1] = q::encode(1.0);
    result.activation[20] = q::encode(1.0);
    result.activation[21] = q::encode(0.001953125);
    for (std::int64_t slot = 0; slot < q::kTopK - 1; ++slot)
      result.ids[slot] = static_cast<std::int32_t>(q::kTopK - 2 - slot);
    result.ids[q::kTopK - 1] = static_cast<std::int32_t>(q::kTopK);
  } else {
    for (std::int64_t expert = 0; expert < q::kTopK; ++expert) {
      const double logit =
          replay == 3 ? 1.0 + expert / 8.0
                      : 1.0 + (q::kTopK - 1 - expert) / 8.0;
      result.activation[expert] = q::encode(logit);
      result.ids[expert] = static_cast<std::int32_t>(
          replay == 3 ? q::kTopK - 1 - expert : expert);
    }
  }

  std::vector<float> slots(q::kTopK * q::kHidden);
  for (std::int64_t slot = 0; slot < q::kTopK; ++slot)
    for (std::int64_t column = 0; column < q::kHidden; ++column)
      slots[slot * q::kHidden + column] =
          static_cast<float>(q::decode(result.activation[column]));
  result.routed = q::expert_oracle(slots, kRoutedSeed, result.ids,
                                   q::kRouterResidentExperts);
  result.combined.resize(q::kHidden);
  double selected_sum = 0.0;
  const auto score = [&](std::int32_t expert) {
    return expert == q::kTopK
               ? q::decode(result.activation[20]) +
                     q::decode(result.activation[21])
               : q::decode(result.activation[expert]);
  };
  for (const auto id : result.ids)
    selected_sum += std::exp(score(id));
  for (std::int64_t column = 0; column < q::kHidden; ++column) {
    double combined = 0.0;
    for (std::int64_t slot = 0; slot < q::kTopK; ++slot)
      combined += std::exp(score(result.ids[slot])) / selected_sum *
                  result.routed[slot * q::kHidden + column];
    result.combined[column] = static_cast<float>(combined);
  }

  const auto token = std::span<const float>(slots.data(), q::kHidden);
  const std::array<std::int32_t, 1> shared_ids{0};
  auto shared_gate = q::project_slots(
      token, q::kHidden, q::kSharedExpertHidden, kSharedSeed, shared_ids);
  const auto shared_up = q::project_slots(
      token, q::kHidden, q::kSharedExpertHidden, kSharedSeed + 1, shared_ids);
  for (std::size_t i = 0; i < shared_gate.size(); ++i)
    shared_gate[i] = shared_gate[i] / (1.0F + std::exp(-shared_gate[i])) *
                     shared_up[i];
  const auto gate_score =
      q::project_slots(token, q::kHidden, 1, kSharedSeed + 3, shared_ids);
  const float gate_probability =
      static_cast<float>(1.0 / (1.0 + std::exp(-gate_score[0])));
  result.shared = q::project_slots(shared_gate, q::kSharedExpertHidden,
                                    q::kHidden, kSharedSeed + 2, shared_ids);
  result.full.resize(q::kHidden);
  for (std::int64_t column = 0; column < q::kHidden; ++column) {
    result.shared[column] *= gate_probability;
    result.full[column] = result.combined[column] + result.shared[column];
  }
  return result;
}

struct GraphCase final {
  g::Graph graph;
  c::TensorDesc activation_desc;
  std::array<c::TensorDesc, 10> constant_descs;
  std::array<c::TensorDesc, 5> output_descs;
  g::ValueId activation;
  std::array<g::ValueId, 10> constants;
  // IDs, per-slot routed result, combined routed result, gated shared result,
  // and the final sum, in publication order.
  std::array<g::ValueId, 5> outputs;
};

inline GraphCase make_graph() {
  const auto activation_desc =
      dense(c::DataType::kBFloat16, std::array<std::int64_t, 2>{1, q::kHidden});
  const auto router_weight_desc = dense(
      c::DataType::kBFloat16,
      std::array<std::int64_t, 2>{q::kExperts, q::kHidden});
  const auto replication_desc =
      dense(c::DataType::kInt32, std::array<std::int64_t, 1>{q::kTopK});
  const auto routed_weight_desc = dense(
      c::DataType::kBFloat16,
      std::array<std::int64_t, 3>{q::kRouterResidentExperts,
                                  q::kExpertHidden, q::kHidden});
  const auto routed_down_desc = dense(
      c::DataType::kBFloat16,
      std::array<std::int64_t, 3>{q::kRouterResidentExperts, q::kHidden,
                                  q::kExpertHidden});
  const auto shared_weight_desc = dense(
      c::DataType::kBFloat16,
      std::array<std::int64_t, 3>{1, q::kSharedExpertHidden, q::kHidden});
  const auto shared_down_desc = dense(
      c::DataType::kBFloat16,
      std::array<std::int64_t, 3>{1, q::kHidden, q::kSharedExpertHidden});
  const auto shared_gate_input_desc =
      dense(c::DataType::kBFloat16, std::array<std::int64_t, 2>{1, q::kHidden});
  const auto shared_ids_desc =
      dense(c::DataType::kInt32, std::array<std::int64_t, 2>{1, 1});
  const std::array constant_descs{
      router_weight_desc, replication_desc, routed_weight_desc,
      routed_weight_desc, routed_down_desc, shared_weight_desc,
      shared_weight_desc, shared_down_desc, shared_gate_input_desc,
      shared_ids_desc};
  const auto ids_desc = dense(
      c::DataType::kInt32, std::array<std::int64_t, 2>{1, q::kTopK});
  const auto routed_desc = dense(
      c::DataType::kFloat32,
      std::array<std::int64_t, 3>{1, q::kTopK, q::kHidden});
  const auto combined_desc = dense(
      c::DataType::kFloat32, std::array<std::int64_t, 3>{1, 1, q::kHidden});
  const std::array output_descs{ids_desc, routed_desc, combined_desc,
                                 combined_desc, combined_desc};

  auto builder = g::GraphBuilder::create().value();
  const auto activation = builder.input(activation_desc).value();
  std::array<g::ValueId, 10> constants{};
  for (std::size_t i = 0; i < constants.size(); ++i)
    constants[i] = builder.constant(constant_descs[i]).value();
  const auto score =
      builder
          .result(builder
                      .add_node(c::model_ops::bfloat_linear_f32(),
                                std::array{activation, constants[0]})
                      .value())
          .value();
  const auto probability = builder.add(c::Softmax{}, score).value();
  const auto topk =
      builder
          .add_node(c::Operation(c::TopK{.count = q::kTopK}),
                    std::array{probability})
          .value();
  const auto selected = builder.result(topk, 0).value();
  const auto ids = builder.result(topk, 1).value();
  const auto selected_sum =
      builder
          .add(c::Reduce{.kind = c::ReductionKind::kSum,
                         .axes = c::Axes::from(std::array<std::int32_t, 1>{-1})
                                     .value(),
                         .keep_dimensions = true,
                         .accumulation_type = c::DataType::kFloat32},
               selected)
          .value();
  const auto normalized =
      builder.add(c::Binary{c::BinaryKind::kDivide}, selected,
                  selected_sum)
          .value();
  const auto token_f32 =
      builder.add(c::Cast{c::DataType::kFloat32}, activation).value();
  const auto replicated =
      builder.add(c::Gather{}, token_f32, constants[1]).value();
  const auto routed_activation =
      builder
          .add(c::View::reshape(
                   c::TensorShape::create(std::array<std::int64_t, 3>{
                                              1, q::kTopK, q::kHidden})
                       .value())
                   .value(),
               replicated)
          .value();
  const auto routed_gate =
      builder
          .result(builder
                      .add_node(c::model_ops::grouped_matmul(),
                                std::array{constants[2], routed_activation,
                                           ids})
                      .value())
          .value();
  const auto routed_up =
      builder
          .result(builder
                      .add_node(c::model_ops::grouped_matmul(),
                                std::array{constants[3], routed_activation,
                                           ids})
                      .value())
          .value();
  const auto routed_gated =
      builder
          .add(c::GatedActivation{.inputs = c::GatedInputKind::kSeparate},
               routed_gate, routed_up)
          .value();
  const auto routed =
      builder
          .result(builder
                      .add_node(c::model_ops::grouped_matmul(),
                                std::array{constants[4], routed_gated, ids})
                      .value())
          .value();
  const auto selected_weights =
      builder
          .add(c::View::reshape(
                   c::TensorShape::create(
                       std::array<std::int64_t, 3>{1, q::kTopK, 1})
                       .value())
                   .value(),
               normalized)
          .value();
  const auto weighted =
      builder.add(c::Binary{c::BinaryKind::kMultiply}, routed,
                  selected_weights)
          .value();
  const auto combined =
      builder
          .add(c::Reduce{.kind = c::ReductionKind::kSum,
                         .axes = c::Axes::from(std::array<std::int32_t, 1>{1})
                                     .value(),
                         .keep_dimensions = true,
                         .accumulation_type = c::DataType::kFloat32},
               weighted)
          .value();
  const auto shared_activation =
      builder
          .add(c::View::reshape(
                   c::TensorShape::create(
                       std::array<std::int64_t, 3>{1, 1, q::kHidden})
                       .value())
                   .value(),
               token_f32)
          .value();
  const auto shared_gate =
      builder
          .result(builder
                      .add_node(c::model_ops::grouped_matmul(),
                                std::array{constants[5], shared_activation,
                                           constants[9]})
                      .value())
          .value();
  const auto shared_up =
      builder
          .result(builder
                      .add_node(c::model_ops::grouped_matmul(),
                                std::array{constants[6], shared_activation,
                                           constants[9]})
                      .value())
          .value();
  const auto shared_gated =
      builder
          .add(c::GatedActivation{.inputs = c::GatedInputKind::kSeparate},
               shared_gate, shared_up)
          .value();
  const auto shared_down =
      builder
          .result(builder
                      .add_node(c::model_ops::grouped_matmul(),
                                std::array{constants[7], shared_gated,
                                           constants[9]})
                      .value())
          .value();
  const auto shared_gate_score =
      builder
          .result(builder
                      .add_node(c::model_ops::bfloat_linear_f32(),
                                std::array{activation, constants[8]})
                      .value())
          .value();
  const auto shared_gate_probability =
      builder.add(c::Unary{c::UnaryKind::kSigmoid}, shared_gate_score).value();
  const auto shared_gate_broadcast =
      builder
          .add(c::View::reshape(
                   c::TensorShape::create(std::array<std::int64_t, 3>{1, 1, 1})
                       .value())
                   .value(),
               shared_gate_probability)
          .value();
  const auto shared =
      builder.add(c::Binary{c::BinaryKind::kMultiply}, shared_down,
                  shared_gate_broadcast)
          .value();
  const auto full =
      builder.add(c::Binary{c::BinaryKind::kAdd}, combined, shared).value();
  const std::array outputs{ids, routed, combined, shared, full};
  for (const auto output : outputs)
    if (!builder.publish(output).ok())
      throw std::runtime_error("full router output publication failed");
  return {std::move(builder).finish().value(), activation_desc,
          constant_descs, output_descs, activation, constants, outputs};
}

} // namespace d5_qwen38_24t_full_router
