#include "view_domain.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <optional>

#include "hyle/math/scalar.h"

namespace hyle::graph::detail {
namespace {

using Operand = compute::detail::ShapeOperand;
using Kind = compute::detail::ShapeDimensionKind;
using Guard = compute::detail::ShapeGuardKind;
using Range = compute::detail::ProviderShapeRange;
using ViewKind = compute::ViewKind;
constexpr auto kRank = compute::kMaxTensorRank;

Operand operand(Dimension value) noexcept {
  return value.is_fixed() ? Operand::fixed(value.fixed_value())
                          : Operand::value(value.symbol().ordinal());
}

class Relations final {
public:
  explicit Relations(const GraphState &graph) noexcept : graph_(graph) {}

  Operand canonical(Operand value) const noexcept {
    // Strictly decreasing ordinal (or a fixed terminal) bounds this walk.
    for (;;) {
      if (value.is_fixed())
        return value;
      const auto bounds = graph_.dimensions[value.ordinal].bounds;
      if (bounds.exact())
        return Operand::fixed(bounds.first());
      Operand next = value;
      for (const auto &guard : graph_.shape_guards) {
        if (guard.kind != Guard::kEqual)
          continue;
        const auto other = guard.left == value ? guard.right : guard.left;
        if ((guard.left == value || guard.right == value) &&
            (other.is_fixed() || other.ordinal < next.ordinal)) {
          next = other;
          if (next.is_fixed())
            return next;
        }
      }
      if (next == value)
        return value;
      value = next;
    }
  }

  bool equal(Operand left, Operand right) const noexcept {
    return canonical(left) == canonical(right);
  }

  Range range(Operand value) const noexcept {
    value = canonical(value);
    return value.is_fixed() ? Range{value.constant, value.constant, 1}
                            : graph_.dimensions[value.ordinal].bounds;
  }

  bool less_equal(Operand left, Operand right) const noexcept {
    if (equal(left, right) || range(left).last() <= range(right).first())
      return true;
    for (const auto &guard : graph_.shape_guards)
      if (guard.kind == Guard::kLessEqual && equal(left, guard.left) &&
          equal(right, guard.right))
        return true;
    return false;
  }

  std::optional<Operand> expression(Kind kind, Operand left,
                                    Operand right) const noexcept {
    if (kind == Kind::kAdd) {
      if (equal(left, Operand::fixed(0)))
        return right;
      if (equal(right, Operand::fixed(0)))
        return left;
    }
    if (kind == Kind::kMultiply) {
      if (equal(left, Operand::fixed(1)))
        return right;
      if (equal(right, Operand::fixed(1)))
        return left;
    }
    for (std::size_t index = 0; index < graph_.dimensions.size(); ++index) {
      const auto &entry = graph_.dimensions[index];
      if (entry.kind == kind &&
          ((equal(entry.left, left) && equal(entry.right, right)) ||
           (equal(entry.left, right) && equal(entry.right, left))))
        return Operand::value(static_cast<std::uint32_t>(index));
    }
    return std::nullopt;
  }

  bool broadcast(Operand source, Operand target) const noexcept {
    if (equal(source, target) || equal(source, Operand::fixed(1)))
      return true;
    target = canonical(target);
    if (target.is_fixed())
      return false;
    const auto &entry = graph_.dimensions[target.ordinal];
    return entry.kind == Kind::kBroadcast &&
           (equal(source, entry.left) || equal(source, entry.right));
  }

  bool slice(Operand start, Operand count, std::int64_t step,
             Operand extent) const noexcept {
    const auto starts = range(start), counts = range(count);
    if (counts.last() == 0)
      return less_equal(start, extent);
    // This exact integer bound also handles step > 1 without rounded division.
    auto span = math::try_multiply(counts.last() - 1, step);
    auto end = span ? math::try_add(*span, starts.last()) : std::nullopt;
    if (end && *end < range(extent).first())
      return true;
    // start + count*step <= extent is sufficient, including empty slices.
    // A caller's guard on an existing program expression can prove this when
    // independent box extrema cannot occur together.
    const auto scaled =
        expression(Kind::kMultiply, count, Operand::fixed(step));
    const auto sum =
        scaled ? expression(Kind::kAdd, start, *scaled) : std::nullopt;
    return sum && less_equal(*sum, extent);
  }

  Result<bool> same_product(std::span<const Dimension> left,
                            std::span<const Dimension> right,
                            mem::AllocatorRef allocator) const noexcept {
    const auto zero = [&](auto dimensions) noexcept {
      return std::ranges::any_of(dimensions, [&](Dimension value) noexcept {
        return equal(operand(value), Operand::fixed(0));
      });
    };
    if (zero(left) || zero(right))
      return zero(left) && zero(right);
    container::SmallVector<std::int64_t, 8, mem::AllocatorRef> powers(
        allocator);
    if (!powers.try_resize(graph_.dimensions.size()))
      return std::unexpected(Status::ResourceExhausted(
          "graph View relation scratch allocation failed"));
    std::fill(powers.begin(), powers.end(), 0);
    std::uint64_t numerator = 1, denominator = 1;
    const auto append = [&](Operand value, std::int64_t power) noexcept {
      value = canonical(value);
      if (!value.is_fixed()) {
        const auto sum = math::try_add(powers[value.ordinal], power);
        if (!sum)
          return false;
        powers[value.ordinal] = *sum;
        return true;
      }
      if (value.constant <= 0 || power == INT64_MIN)
        return false;
      auto base = static_cast<std::uint64_t>(value.constant);
      auto exponent = static_cast<std::uint64_t>(power < 0 ? -power : power);
      auto &own = power < 0 ? denominator : numerator;
      auto &opposite = power < 0 ? numerator : denominator;
      while (exponent != 0) {
        if ((exponent & 1) != 0) {
          const auto common = std::gcd(base, opposite);
          opposite /= common;
          const auto product = math::try_multiply(own, base / common);
          if (!product)
            return false;
          own = *product;
        }
        exponent >>= 1;
        if (exponent != 0) {
          const auto square = math::try_multiply(base, base);
          if (!square)
            return false;
          base = *square;
        }
      }
      return true;
    };
    for (const auto value : left)
      if (!append(operand(value), 1))
        return false;
    for (const auto value : right)
      if (!append(operand(value), -1))
        return false;
    // A DAG pass, not recursive expansion: a repeated expression is visited
    // once. Division is exact and nonzero at every admitted program binding.
    for (std::size_t index = powers.size(); index != 0; --index) {
      const auto ordinal = index - 1;
      const auto power = powers[ordinal];
      if (power == 0)
        continue;
      const auto &entry = graph_.dimensions[ordinal];
      if (entry.kind != Kind::kMultiply && entry.kind != Kind::kExactDivide)
        continue;
      powers[ordinal] = 0;
      if (power == INT64_MIN || !append(entry.left, power) ||
          !append(entry.right, entry.kind == Kind::kMultiply ? power : -power))
        return false;
    }
    return numerator == denominator &&
           std::ranges::all_of(powers, [](auto power) { return power == 0; });
  }

private:
  const GraphState &graph_;
};

Result<std::size_t> axis_index(std::int32_t axis, std::size_t rank) noexcept {
  const auto normalized = static_cast<std::int64_t>(axis) +
                          (axis < 0 ? static_cast<std::int64_t>(rank) : 0);
  if (normalized < 0 || static_cast<std::size_t>(normalized) >= rank)
    return std::unexpected(
        Status::InvalidArgument("graph View axis is invalid"));
  return static_cast<std::size_t>(normalized);
}

Status infer_domain_view(const GraphState &graph, const NodeRecord &node,
                         const compute::TensorDesc &input, bool input_dense,
                         compute::TensorDesc &output, bool &output_dense,
                         mem::AllocatorRef allocator) noexcept {
  const Relations relation(graph);
  const auto &source = graph.values[graph.edges[node.input_begin]].spec;
  const auto &target = graph.values[node.output_begin].spec;
  const auto *symbolic = view_spec(graph, node);
  const auto *fixed =
      symbolic ? nullptr : node.operation().get_if<compute::View>();
  const auto kind = symbolic ? symbolic->kind() : fixed->kind();
  if (!source.is_symbolic() && (!symbolic || !symbolic->is_symbolic()) &&
      target.kind() == TensorSpecKind::kConcrete) {
    HYLE_ASSIGN_OR_RETURN(const auto operation,
                          bind_operation(graph, node, {}));
    HYLE_ASSIGN_OR_RETURN(
        const auto inferred,
        compute::infer_single_output(operation, std::span(&input, 1)));
    HYLE_ASSIGN_OR_RETURN(const auto required, target.concrete_description());
    if (inferred != required)
      return Status::InvalidArgument(
          "fixed View output differs from its constraint");
    output = inferred;
    output_dense = output.is_dense();
    return Status::Ok();
  }
  if (!input.is_plain() || target.storage_format() != input.storage_format())
    return Status::Unimplemented("dynamic View requires plain storage");
  if (target.data_type() != input.data_type())
    return Status::InvalidArgument("View output data type differs from input");
  std::array<Dimension, kRank> expected{};
  std::array<std::int64_t, kRank> strides{}, extents{};
  std::size_t rank = source.rank();
  std::copy(source.dimensions().begin(), source.dimensions().end(),
            expected.begin());
  std::copy(input.byte_strides().begin(), input.byte_strides().end(),
            strides.begin());
  output_dense = false;
  if (kind == ViewKind::kReshape || kind == ViewKind::kBroadcast) {
    rank = symbolic ? symbolic->dimensions().size() : fixed->shape().rank();
    for (std::size_t axis = 0; axis < rank; ++axis)
      expected[axis] =
          symbolic ? symbolic->dimensions()[axis]
                   : Dimension::fixed(fixed->shape().dimensions()[axis]);
    if (kind == ViewKind::kReshape) {
      if (!input_dense)
        return Status::Unimplemented(
            "whole-domain reshape needs canonical dense input");
      HYLE_ASSIGN_OR_RETURN(
          const auto equal,
          relation.same_product(source.dimensions(),
                                std::span(expected).first(rank), allocator));
      if (!equal)
        return Status::Unimplemented(
            "reshape element-count relation is not proved over the domain");
      output_dense = true;
    } else {
      if (rank < source.rank())
        return Status::InvalidArgument(
            "broadcast output rank is smaller than input");
      const auto prefix = rank - source.rank();
      output_dense = input_dense && prefix == 0;
      for (std::size_t axis = 0; axis < rank; ++axis) {
        if (axis < prefix) {
          strides[axis] = 0;
          continue;
        }
        const auto from = operand(source.dimensions()[axis - prefix]);
        const auto to = operand(expected[axis]);
        if (!relation.broadcast(from, to))
          return Status::Unimplemented(
              "broadcast relation is not proved over the domain");
        // Keep the upper source stride even when most bindings use zero;
        // source==target==1 legally preserves it. This is a span envelope.
        strides[axis] = input.byte_strides()[axis - prefix];
        output_dense = output_dense && relation.equal(from, to);
      }
    }
  } else if (kind == ViewKind::kSlice) {
    HYLE_ASSIGN_OR_RETURN(
        const auto axis,
        axis_index(symbolic ? symbolic->axis() : fixed->axis(), source.rank()));
    const auto start =
        symbolic ? operand(symbolic->start()) : Operand::fixed(fixed->start());
    const auto count =
        symbolic ? operand(symbolic->count()) : Operand::fixed(fixed->count());
    const auto step = symbolic ? symbolic->step() : fixed->step();
    if (!relation.slice(start, count, step, operand(source.dimensions()[axis])))
      return Status::Unimplemented(
          "slice bounds are not proved over the domain");
    expected[axis] =
        symbolic ? symbolic->count() : Dimension::fixed(fixed->count());
    const auto stepped = math::try_multiply(strides[axis], step);
    if (!stepped)
      return Status::OutOfRange("dynamic slice stride envelope overflows");
    strides[axis] = *stepped;
    output_dense = input_dense && axis == 0 && step == 1 &&
                   relation.range(count).first() > 0;
    for (auto dim : source.dimensions())
      output_dense = output_dense && relation.range(operand(dim)).first() > 0;
  } else {
    const auto permutation_size =
        symbolic ? symbolic->permutation().size() : fixed->permutation().size();
    if (permutation_size != source.rank())
      return Status::InvalidArgument(
          "View permutation rank differs from input");
    std::array<bool, kRank> seen{};
    output_dense = input_dense;
    for (std::size_t axis = 0; axis < rank; ++axis) {
      HYLE_ASSIGN_OR_RETURN(const auto from,
                            axis_index(symbolic ? symbolic->permutation()[axis]
                                                : fixed->permutation()[axis],
                                       rank));
      if (seen[from])
        return Status::InvalidArgument("View permutation has duplicate axes");
      seen[from] = true;
      expected[axis] = source.dimensions()[from];
      strides[axis] = input.byte_strides()[from];
      output_dense = output_dense && from == axis;
    }
  }
  if (rank != target.rank())
    return Status::InvalidArgument("View output constraint rank differs");
  for (std::size_t axis = 0; axis < rank; ++axis) {
    if (!relation.equal(operand(expected[axis]),
                        operand(target.dimensions()[axis])))
      return Status::Unimplemented(
          "View output shape relation is not proved over the domain");
    const auto dim = target.dimensions()[axis];
    // Use the original range, not guard-narrowed representatives: this must
    // match the Provider axis map's envelope, while actual binds check guards.
    extents[axis] =
        dim.is_fixed() ? dim.fixed_value()
                       : graph.dimensions[dim.symbol().ordinal()].bounds.last();
  }
  const auto shape = std::span(extents).first(rank);
  HYLE_ASSIGN_OR_RETURN(
      output,
      kind == ViewKind::kReshape
          ? compute::TensorDesc::dense(input.data_type(), shape)
          : compute::TensorDesc::strided_bytes(input.data_type(), shape,
                                               std::span(strides).first(rank)));
  if (target.kind() != TensorSpecKind::kInferred) {
    if (!output_dense || target.kind() != TensorSpecKind::kDense)
      return Status::Unimplemented("explicit View layout lacks a whole-domain "
                                   "proof; use inferred output");
  }
  return Status::Ok();
}

} // namespace

bool has_strided_view_layout(const GraphState &graph,
                             std::size_t value) noexcept {
  const auto &record = graph.values[value];
  if (record.spec.kind() != TensorSpecKind::kInferred ||
      record.producer == kNoNode)
    return false;
  const auto &node = graph.nodes[record.producer];
  if (node.kind() != compute::OperationKind::kView)
    return false;
  const auto *view = view_spec(graph, node);
  return (view ? view->kind()
               : node.operation().get_if<compute::View>()->kind()) !=
         ViewKind::kReshape;
}

Status prove_view_domains(const GraphState &graph,
                          std::span<compute::TensorDesc> descriptions,
                          mem::AllocatorRef allocator) noexcept {
  container::SmallVector<std::uint8_t, 16, mem::AllocatorRef> dense(allocator);
  if (!dense.try_resize(graph.values.size()))
    return Status::ResourceExhausted(
        "graph View layout scratch allocation failed");
  for (std::size_t index = 0; index < dense.size(); ++index)
    dense[index] =
        graph.values[index].spec.kind() == TensorSpecKind::kDense ||
        (graph.values[index].spec.kind() == TensorSpecKind::kConcrete &&
         descriptions[index].is_dense());
  for (const auto &node : graph.nodes) {
    if (node.kind() != compute::OperationKind::kView) {
      for (std::size_t port = 0; port < node.output_count; ++port)
        if (graph.values[node.output_begin + port].spec.kind() ==
            TensorSpecKind::kInferred)
          return Status::Unimplemented(
              "non-View inferred layout has no declared-domain proof");
      continue;
    }
    if (node.input_count != 1 || node.output_count != 1)
      return Status::InvalidArgument("View arity differs from its schema");
    const auto input = graph.edges[node.input_begin];
    bool output_dense = false;
    HYLE_RETURN_IF_ERROR(infer_domain_view(
        graph, node, descriptions[input], dense[input] != 0,
        descriptions[node.output_begin], output_dense, allocator));
    dense[node.output_begin] = output_dense;
  }
  return Status::Ok();
}

} // namespace hyle::graph::detail
