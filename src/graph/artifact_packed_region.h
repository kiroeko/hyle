#pragma once

#include "hyle/graph/region_provider.h"
#include "hyle/container/small_vector.h"
#include "../compute/provider_internal.h"
#include "../compute_packed_shared/artifact_packed_linear.h"
#include "closed_region.h"

#include <algorithm>
#include <array>
#include <limits>
#include <optional>

namespace hyle::graph::detail {
namespace c = compute;
namespace native = c::packed::internal;

// The exact public recipe and reader/control proof are shared by provider
// regions and native graph lowerers; neither needs a second discovery policy.
struct ArtifactPackedPattern : ClosedRegionPattern {
inline static constexpr c::RoutePolicyId kPolicy = kArtifactPackedProjectionPolicy;
inline static constexpr std::uint64_t kRoute = 1;
inline static constexpr std::size_t kMaxNodes = 4;
inline static constexpr std::size_t kMaxInputs = 7;

struct Pattern final {
  std::array<NodeId, kMaxNodes> nodes{};
  std::size_t size = 0;
  std::array<ValueId, kMaxInputs> inputs{}; // Original A and complete typed decoder operands (plus grouped IDs).
  std::size_t input_count = 0;
  ValueId output{};
  native::ArtifactPackedLinear descriptor{};
  std::span<const NodeId> span() const noexcept { return {nodes.data(), size}; }
  bool contains(NodeId node) const noexcept {
    return std::ranges::find(span(), node) != span().end();
  }
  void append(NodeId node) noexcept { nodes[size++] = node; }
  std::size_t boundary_count() const noexcept {
    std::size_t count = 0;
    for (std::size_t n = 0; n < input_count; ++n) {
      const auto prefix = std::span(inputs).first(n);
      count += std::ranges::find(prefix, inputs[n]) == prefix.end();
    }
    return count;
  }
  c::Operation composite() const noexcept { return c::Operation::custom(descriptor); }
};

static bool pattern(const Graph &graph, NodeId matrix, Pattern &p) noexcept {
  const auto matrix_operation = graph.operation(matrix);
  if (matrix_operation && matrix_operation->custom_as<c::model_ops::GroupedMatMul>())
    return grouped_pattern(graph, matrix, p);
  const auto mm = operation<c::MatMul>(graph, matrix);
  if (!mm || !arity(graph, matrix, 2) || mm->accumulation_type != c::DataType::kFloat32)
    return false;
  const auto activation = graph.node_input(matrix, 0).value();
  const auto weight = graph.node_input(matrix, 1).value();
  const auto operand = graph.value_description(weight);
  if (!operand || operand->rank() != 2) return false;

  auto value = weight;
  NodeId after_cast;
  auto node = producer(graph, value);
  if (operation<c::Cast>(graph, node)) {
    if (!arity(graph, node, 1)) return false;
    after_cast = node;
    value = graph.node_input(node, 0).value();
  }
  const auto transpose = producer(graph, value);
  const auto view = operation<c::View>(graph, transpose);
  if (!view || !arity(graph, transpose, 1) || view->kind() != c::ViewKind::kPermute)
    return false;
  const auto axes = view->permutation();
  if (axes.size() != 2 || (axes[0] != 1 && axes[0] != -1) ||
      (axes[1] != 0 && axes[1] != -2)) return false;
  value = graph.node_input(transpose, 0).value();
  const auto before_view = graph.value_description(value);
  const auto after_view = graph.value_description(graph.result(transpose).value());
  if (!before_view || !after_view || before_view->rank() != 2) return false;
  const auto expected_view = before_view->transposed();
  if (!expected_view || *expected_view != *after_view) return false;

  NodeId before_cast;
  node = producer(graph, value);
  if (operation<c::Cast>(graph, node)) {
    if (after_cast.valid() || !arity(graph, node, 1)) return false;
    before_cast = node;
    value = graph.node_input(node, 0).value();
  }
  const auto decode = producer(graph, value);
  if (!decode.valid()) return false;
  const auto decode_operation = graph.operation(decode);
  if (!decode_operation) return false;
  const auto artifact = decode_operation->custom_as<c::model_ops::ArtifactWeightDecode>();
  if (!artifact ||
      (artifact->format != c::model_ops::WeightArtifactFormat::kCompressedTensorsSymmetric4 &&
       artifact->format != c::model_ops::WeightArtifactFormat::kCompressedTensorsAsymmetric4 &&
       artifact->format != c::model_ops::WeightArtifactFormat::kGptqV1Int4 &&
       artifact->format != c::model_ops::WeightArtifactFormat::kNf4DoubleQuant))
    return false;
  const auto count = artifact->operation_traits().inputs.minimum;
  if (!arity(graph, decode, count)) return false;
  const auto decoded = graph.value_description(value);
  if (!decoded || decoded->rank() != 2) return false;
  p.descriptor = {artifact->format, artifact->group_size, decoded->data_type(), operand->data_type()};
  p.inputs[0] = activation;
  p.input_count = std::size_t(count) + 1;
  for (std::size_t n = 0; n < count; ++n)
    p.inputs[n + 1] = graph.node_input(decode, n).value();
  p.output = graph.result(matrix).value();
  p.append(decode);
  if (before_cast.valid()) p.append(before_cast);
  p.append(transpose);
  if (after_cast.valid()) p.append(after_cast);
  p.append(matrix);
  std::ranges::sort(std::span(p.nodes).first(p.size), {}, &NodeId::ordinal);
  return true;
}
static bool grouped_pattern(const Graph &graph, NodeId matrix, Pattern &p) noexcept {
  if (!arity(graph, matrix, 3)) return false;
  const auto weight = graph.node_input(matrix, 0).value();
  const auto operand = graph.value_description(weight);
  if (!operand || operand->rank() != 3 || !operand->is_dense()) return false;
  auto value = weight;
  NodeId after_cast;
  auto node = producer(graph, value);
  if (operation<c::Cast>(graph, node)) {
    if (!arity(graph, node, 1)) return false;
    after_cast = node;
    value = graph.node_input(node, 0).value();
  }
  const auto reshape = producer(graph, value);
  const auto view = operation<c::View>(graph, reshape);
  if (!view || !arity(graph, reshape, 1) || view->kind() != c::ViewKind::kReshape)
    return false;
  const auto reshaped = graph.value_description(value);
  value = graph.node_input(reshape, 0).value();
  const auto flat = graph.value_description(value);
  if (!flat || !reshaped || flat->rank() != 2 || !flat->is_dense() ||
      reshaped->rank() != 3 || !reshaped->is_dense() ||
      flat->data_type() != reshaped->data_type() ||
      !std::ranges::equal(reshaped->dimensions(), operand->dimensions()) ||
      flat->dimensions()[1] != operand->dimensions()[2] ||
      flat->element_count() != reshaped->element_count()) return false;
  NodeId before_cast;
  node = producer(graph, value);
  if (operation<c::Cast>(graph, node)) {
    if (after_cast.valid() || !arity(graph, node, 1)) return false;
    before_cast = node;
    value = graph.node_input(node, 0).value();
  }
  const auto decode = producer(graph, value);
  if (!decode.valid()) return false;
  const auto decode_operation = graph.operation(decode);
  if (!decode_operation) return false;
  const auto artifact = decode_operation->custom_as<c::model_ops::ArtifactWeightDecode>();
  if (!artifact ||
      (artifact->format != c::model_ops::WeightArtifactFormat::kCompressedTensorsSymmetric4 &&
       artifact->format != c::model_ops::WeightArtifactFormat::kCompressedTensorsAsymmetric4))
    return false;
  const auto count = artifact->operation_traits().inputs.minimum;
  const auto decoded = graph.value_description(value);
  if (!arity(graph, decode, count) || !decoded || decoded->rank() != 2) return false;
  p.descriptor = {artifact->format, artifact->group_size, decoded->data_type(),
      operand->data_type(), static_cast<std::uint64_t>(operand->dimensions()[0])};
  p.inputs[0] = graph.node_input(matrix, 1).value();
  for (std::size_t n = 0; n < count; ++n)
    p.inputs[n + 1] = graph.node_input(decode, n).value();
  p.input_count = std::size_t(count) + 2;
  p.inputs[p.input_count - 1] = graph.node_input(matrix, 2).value();
  p.output = graph.result(matrix).value();
  p.append(decode);
  if (before_cast.valid()) p.append(before_cast);
  p.append(reshape);
  if (after_cast.valid()) p.append(after_cast);
  p.append(matrix);
  std::ranges::sort(std::span(p.nodes).first(p.size), {}, &NodeId::ordinal);
  return true;
}
// Only the chosen bounded node set is reconstructed here. Graph's exact
// boundary/contractibility check preserves escapes; discovery checked readers.
static Status qualify(const RegionCompileRequest &request, Pattern &result) noexcept {
  if (!request.graph || request.route != 1 || request.outputs.size() != 1 ||
      request.nodes.size() < 3 || request.nodes.size() > kMaxNodes ||
      request.has_unknown_effects || !request.effect_accesses.empty())
    return Status::Unimplemented("packed artifact region boundary differs");
  for (const auto node : request.nodes) {
    Pattern p;
    if (!pattern(*request.graph, node, p) || p.size != request.nodes.size() ||
        p.output != request.outputs[0] || p.boundary_count() != request.inputs.size()) continue;
    const auto logical_inputs = std::span(p.inputs).first(p.input_count);
    if (!std::ranges::all_of(p.span(), [&](NodeId id) {
          return std::ranges::count(request.nodes, id) == 1;
        }) || !std::ranges::all_of(logical_inputs, [&](ValueId id) {
          return std::ranges::count(request.inputs, id) == 1;
        })) continue;
    result = p;
    return Status::Ok();
  }
  return Status::Unimplemented("packed artifact region does not preserve the exact public recipe");
}
};

// Bind discovery to an existing provider without replacing its context,
// resources, prepared operation or submit path.
} // namespace hyle::graph::detail
