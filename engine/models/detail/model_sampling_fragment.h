#pragma once

#include <array>
#include <bit>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <vector>

#include "component_definition.h"
#include "hyle/graph/graph.h"
#include "../../inference/sampling_graph.h"

// Only fixed-shape recipe declaration adaption. Local fragment identities stay
// local; no Graph generation is forged and no sampling mathematics is copied.
namespace model_sampling_fragment {
namespace sem = hyle::inference::model_definition;
namespace c = hyle::compute;
namespace g = hyle::graph;
using Value = sem::ValueId;

class Builder final {
  sem::Fragment &fragment_;
public:
  explicit Builder(sem::Fragment &fragment) noexcept : fragment_(fragment) {}
  Value value(Value original) const noexcept { return original; }
  Value local(Value original) const noexcept { return original; }
  g::TensorSpec spec(Value value) const {
    auto result = g::TensorSpec::concrete(fragment_.description(value));
    if (!result) throw std::runtime_error(std::string(result.error().message()));
    return std::move(*result);
  }
  static g::TensorSpec dense_spec(c::DataType type, std::span<const g::Dimension> dimensions) {
    auto result = g::TensorSpec::dense(type, dimensions);
    if (!result) throw std::runtime_error(std::string(result.error().message()));
    return std::move(*result);
  }
  static g::TensorSpec dense_spec(c::DataType type, std::initializer_list<g::Dimension> dimensions) {
    return dense_spec(type, std::span(dimensions.begin(), dimensions.size()));
  }
  Value input(c::TensorDesc description) { return fragment_.input(std::move(description)); }
  Value constant(c::TensorDesc description, std::vector<std::byte> bytes) {
    return fragment_.constant(std::move(description), std::move(bytes));
  }
  Value scalar(float value) {
    return constant(sem::dense({1}), sem::bytes(std::vector{value}));
  }
  std::vector<Value> op(c::Operation operation, std::initializer_list<Value> ports) {
    const auto count = c::operation_traits(operation).outputs;
    return fragment_.append(std::move(operation), ports, count);
  }
  template<class Operation>
  Value one(Operation operation, std::initializer_list<Value> ports) {
    return fragment_.one(c::Operation(std::move(operation)), ports);
  }
  template<class Operation>
  Value one(Operation operation, std::initializer_list<Value> ports, g::TensorSpec output) {
    auto expected = output.concrete_description();
    if (!expected) throw std::runtime_error(std::string(expected.error().message()));
    const auto actual = one(std::move(operation), ports);
    // This adapter owns fixed physical descriptions. Preserve the original
    // schema inference and check the declared output; never replace it with
    // a guessed descriptor or drop the common history branch.
    if (fragment_.description(actual) != *expected)
      throw std::runtime_error("sampling fragment output differs from its declared fixed spec");
    return actual;
  }
  Value copy(Value value) { return one(c::Copy{}, {value}); }
  Value cast(Value value, c::DataType type) { return one(c::Cast{.output_type=type}, {value}); }
  Value unary(c::UnaryKind kind, Value value) { return one(c::Unary{kind}, {value}); }
  Value binary(c::BinaryKind kind, Value a, Value b) { return one(c::Binary{kind}, {a,b}); }
  Value reshape(Value value, std::initializer_list<g::Dimension> shape) {
    std::vector<std::int64_t> extents;
    for (const auto dimension : shape) {
      if (!dimension.is_fixed()) throw std::runtime_error("sampling fragment shape must be fixed");
      extents.push_back(dimension.fixed_value());
    }
    auto actual = c::TensorShape::create(extents);
    if (!actual) throw std::runtime_error(std::string(actual.error().message()));
    auto view = c::View::reshape(*actual);
    if (!view) throw std::runtime_error(std::string(view.error().message()));
    return one(*view,{value});
  }
  Value slice(Value value, std::uint32_t axis, g::Dimension offset, g::Dimension count) {
    if (!offset.is_fixed() || !count.is_fixed())
      throw std::runtime_error("sampling fragment slice must be fixed");
    auto view = c::View::slice(axis,offset.fixed_value(),count.fixed_value());
    if (!view) throw std::runtime_error(std::string(view.error().message()));
    return one(*view,{value});
  }
  std::size_t publish(Value value) {
    fragment_.outputs.push_back(value);
    return fragment_.outputs.size()-1;
  }
};
struct RandomProposalPorts final { Value log_probabilities, proposal, valid; };
RandomProposalPorts append_proposal(sem::Fragment &fragment,
    Value actual_nextn_logits, Value proposal_identity,
    hyle::inference::sampling::Policy policy = {});
struct RandomDecisionPorts final { Value accepted, pending, valid, target_log, bonus_log; };
RandomDecisionPorts append_decision(sem::Fragment &fragment,
    Value actual_target_logits, Value actual_log_q, Value actual_proposal,
    Value accept_identity, Value final_identity,
    hyle::inference::sampling::Policy policy = {});
} // namespace model_sampling_fragment
