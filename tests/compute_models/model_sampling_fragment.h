#pragma once

#include <array>
#include <bit>
#include <initializer_list>
#include <stdexcept>
#include <vector>

#include "d5_qwen38_24t_semantic_fragments.h"
#include "../../engine/inference/sampling_graph.h"

// Only fixed-shape recipe declaration adaption. Local fragment identities stay
// local; no Graph generation is forged and no sampling mathematics is copied.
namespace model_sampling_fragment {
namespace sem = d5_qwen38_24t_semantic_fragments;
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
inline RandomProposalPorts append_proposal(sem::Fragment &fragment,
    Value actual_nextn_logits, Value proposal_identity,
    hyle::inference::sampling::Policy policy = {}) {
  Builder r(fragment);
  const auto q = hyle::inference::sampling::distribution(r, actual_nextn_logits, policy);
  return {r.copy(q.log_probabilities),
      r.copy(r.one(c::sampling_ops::categorical(), {q.log_probabilities, proposal_identity})),
      r.copy(q.valid)};
}
struct RandomDecisionPorts final { Value accepted, pending, valid, target_log, bonus_log; };
inline RandomDecisionPorts append_decision(sem::Fragment &fragment,
    Value actual_target_logits, Value actual_log_q, Value actual_proposal,
    Value accept_identity, Value final_identity,
    hyle::inference::sampling::Policy policy = {}) {
  Builder r(fragment);
  const auto target = hyle::inference::sampling::distribution(r,
      r.copy(r.slice(actual_target_logits, 0, hyle::inference::sampling::fixed(0),
          hyle::inference::sampling::fixed(1))), policy);
  const auto bonus = hyle::inference::sampling::distribution(r,
      r.copy(r.slice(actual_target_logits, 0, hyle::inference::sampling::fixed(1),
          hyle::inference::sampling::fixed(1))), policy);
  const auto selected = hyle::inference::sampling::decision(r,
      target.log_probabilities, actual_log_q, bonus.log_probabilities,
      actual_proposal, accept_identity, final_identity);
  const auto no = r.cast(r.scalar(0), c::DataType::kBool);
  const auto valid = r.one(c::Select{}, {target.valid,
      r.one(c::Select{}, {bonus.valid, selected.valid, no}), no});
  return {r.copy(selected.accepted), r.copy(selected.final_id), r.copy(valid),
      r.copy(target.log_probabilities), r.copy(bonus.log_probabilities)};
}
} // namespace model_sampling_fragment
