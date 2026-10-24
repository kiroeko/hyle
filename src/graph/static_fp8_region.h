#pragma once

#include "closed_region.h"
#include "../compute_packed_shared/static_fp8_linear.h"

#include <array>

namespace hyle::graph::detail {

struct StaticFloat8Pattern : ClosedRegionPattern {
  inline static constexpr compute::RoutePolicyId kPolicy{0x48594c4553465038ULL, 1, 2};
  inline static constexpr std::size_t kMaxNodes = 12, kMaxInputs = 4;
  inline static constexpr std::uint64_t kRoute = 2;
  struct Pattern final {
    std::array<NodeId, kMaxNodes> nodes{};
    std::size_t size = 0;
    std::array<ValueId, kMaxInputs> inputs{};
    std::size_t input_count = 4;
    ValueId output{};
    compute::packed::internal::StaticFloat8Linear descriptor{};
    std::span<const NodeId> span() const noexcept { return {nodes.data(), size}; }
    bool contains(NodeId n) const noexcept { return std::ranges::find(span(), n) != span().end(); }
    void append(NodeId n) noexcept { nodes[size++] = n; }
    std::size_t boundary_count() const noexcept {
      std::size_t count = 0;
      for (std::size_t n = 0; n < inputs.size(); ++n)
        count += std::ranges::find(std::span(inputs).first(n), inputs[n]) == std::span(inputs).first(n).end();
      return count;
    }
    compute::Operation composite() const noexcept { return compute::Operation::custom(descriptor); }
  };
  static bool cast(const Graph &g, NodeId n, compute::DataType type) noexcept {
    const auto op = operation<compute::Cast>(g, n);
    return op && arity(g, n, 1) && op->output_type == type;
  }
  static bool binary(const Graph &g, NodeId n, compute::BinaryKind kind) noexcept {
    const auto op = operation<compute::Binary>(g, n);
    return op && arity(g, n, 2) && op->kind == kind;
  }
  static bool pattern(const Graph &g, NodeId terminal, Pattern &p) noexcept {
    using T = compute::DataType;
    using B = compute::BinaryKind;
    p = {};
    const auto result_cast = operation<compute::Cast>(g, terminal);
    if (!result_cast || !arity(g, terminal, 1)) return false;
    const auto second_mul = producer(g, g.node_input(terminal, 0).value());
    if (!binary(g, second_mul, B::kMultiply)) return false;
    const auto first_mul = producer(g, g.node_input(second_mul, 0).value());
    if (!binary(g, first_mul, B::kMultiply)) return false;
    const auto matrix = producer(g, g.node_input(first_mul, 0).value());
    const auto mm = operation<compute::MatMul>(g, matrix);
    if (!mm || !arity(g, matrix, 2) || mm->accumulation_type != T::kFloat32) return false;
    const auto transpose = producer(g, g.node_input(matrix, 1).value());
    const auto view = operation<compute::View>(g, transpose);
    if (!view || !arity(g, transpose, 1) || view->kind() != compute::ViewKind::kPermute)
      return false;
    const auto axes = view->permutation();
    if (axes.size() != 2 || (axes[0] != 1 && axes[0] != -1) ||
        (axes[1] != 0 && axes[1] != -2)) return false;
    const auto w_cast = producer(g, g.node_input(transpose, 0).value());
    if (!cast(g, w_cast, T::kFloat32)) return false;
    const auto before = g.value_description(g.result(w_cast).value());
    const auto after = g.value_description(g.result(transpose).value());
    if (!before || !after || before->rank() != 2) return false;
    const auto expected = before->transposed();
    if (!expected || *expected != *after) return false;
    const auto widened = producer(g, g.node_input(matrix, 0).value());
    if (!cast(g, widened, T::kFloat32)) return false;
    const auto encoded = producer(g, g.node_input(widened, 0).value());
    if (!cast(g, encoded, T::kFloat8E4M3FN)) return false;
    const auto normalized = producer(g, g.node_input(encoded, 0).value());
    if (!binary(g, normalized, B::kMultiply)) return false;
    const auto a_cast = producer(g, g.node_input(normalized, 0).value());
    if (!cast(g, a_cast, T::kFloat32)) return false;
    const auto reciprocal = producer(g, g.node_input(normalized, 1).value());
    if (!binary(g, reciprocal, B::kDivide)) return false;
    const auto one_node = producer(g, g.node_input(reciprocal, 0).value());
    const auto one = operation<compute::Fill>(g, one_node);
    if (!one || !arity(g, one_node, 0) || one->output_type != T::kFloat32 ||
        one->value.data_type() != T::kFloat32 || one->value.bits() != 0x3f800000u)
      return false;
    const auto sa = g.node_input(reciprocal, 1).value();
    // One and SA must have the same explicit scalar/[M,1] description. The
    // private descriptor below separately proves SW's scalar/[N]/[1,N] axis;
    // no arbitrary same-element-count broadcasting is contracted.
    if (g.value_description(g.result(one_node).value()).value() != g.value_description(sa).value())
      return false;
    const auto first_scale = g.node_input(first_mul, 1).value();
    const auto second_scale = g.node_input(second_mul, 1).value();
    using Order = compute::packed::internal::StaticFloat8ScaleOrder;
    Order order;
    ValueId sw;
    if (first_scale == sa) {
      order = Order::kActivationThenWeight;
      sw = second_scale;
    } else if (second_scale == sa) {
      order = Order::kWeightThenActivation;
      sw = first_scale;
    } else return false;
    p.inputs = {g.node_input(a_cast, 0).value(), g.node_input(w_cast, 0).value(),
                sa, sw};
    p.output = g.result(terminal).value();
    p.descriptor = {result_cast->output_type, order};
    const std::array descriptions{g.value_description(p.inputs[0]).value(),
        g.value_description(p.inputs[1]).value(), g.value_description(p.inputs[2]).value(),
        g.value_description(p.inputs[3]).value()};
    std::array<compute::TensorDesc, 1> inferred;
    if (!p.descriptor.infer_output_descriptions(descriptions, inferred).ok() ||
        inferred[0] != g.value_description(p.output).value()) return false;
    p.nodes = {a_cast, one_node, reciprocal, normalized, encoded, widened,
               w_cast, transpose, matrix, first_mul, second_mul, terminal};
    p.size = p.nodes.size();
    std::ranges::sort(p.nodes, {}, &NodeId::ordinal);
    return std::adjacent_find(p.nodes.begin(), p.nodes.end()) == p.nodes.end();
  }
  static Status qualify(const RegionCompileRequest &r, Pattern &out) noexcept {
    if (!r.graph || r.route != kRoute || r.outputs.size() != 1 || r.nodes.size() != kMaxNodes ||
        r.has_unknown_effects || !r.effect_accesses.empty())
      return Status::Unimplemented("static FP8 region boundary differs");
    for (const auto n : r.nodes) {
      Pattern p;
      if (!pattern(*r.graph, n, p) || p.output != r.outputs[0] ||
          p.boundary_count() != r.inputs.size()) continue;
      if (!std::ranges::all_of(p.span(), [&](NodeId id) { return std::ranges::count(r.nodes, id) == 1; }) ||
          !std::ranges::all_of(p.inputs, [&](ValueId id) { return std::ranges::count(r.inputs, id) == 1; }))
        continue;
      out = p;
      return Status::Ok();
    }
    return Status::Unimplemented("static FP8 region changes the stored-scale public graph");
  }
};
} // namespace hyle::graph::detail
