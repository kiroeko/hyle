#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#include "hyle/compute/sampling_ops.h"
#include "hyle/graph/graph.h"

// Cold speculative-sampling mathematics. This appends ordinary public
// Compute nodes to the caller's existing Recipe/Graph; it owns no invocation,
// callback, random generator, token policy, session cursor or output storage.
namespace hyle::inference::sampling {
namespace c = hyle::compute;
namespace g = hyle::graph;
using Type = c::DataType;
constexpr auto fixed(std::int64_t n) noexcept { return g::Dimension::fixed(n); }

template<class Value> struct DecisionFor final {
  Value accepted; // Bool[1]. Accepts the caller's proposal as the first output.
  Value final_id; // I32[1]. Accepted: target bonus; rejected: residual token.
  Value valid;    // Bool[1]. All outputs are unusable until this is checked.
  // Optional scalar diagnostics, not implicitly published Graph outputs.
  Value p_valid, q_valid, bonus_valid, proposal_valid, residual_positive;
};

struct Policy final {
  float temperature = 1;
  std::size_t top_k = 0; // Zero disables top-k; a positive value must fit V.
  float top_p = 1;
};

// Immutable cold port contract. Values are ordinary Graph constants or ready
// per-episode inputs; neither their contents nor their lifetime belong to KV.
struct LogitsPolicy final {
  bool bias = false, allowed = false;
  bool empty() const noexcept { return !bias && !allowed; }
  friend bool operator==(const LogitsPolicy &, const LogitsPolicy &) = default;
};
template<class Value> struct LogitsInputsFor final {
  std::optional<Value> bias;    // F32[1,V]; every value must be finite.
  std::optional<Value> allowed; // Bool[1,V]; false removes that token.
};
template<class Value> struct SelectionFor final {
  Value id;
  std::optional<Value> valid; // Present for categorical or any modifier.
};
template<class Value> struct DistributionFor final {
  Value log_probabilities; // F32[1,V], in original vocabulary order.
  Value valid;             // Bool[1]; caller must check before using the row.
};

// Concrete Graph identities remain the ordinary engine-facing types. Cold
// fragment builders can reuse the same mathematics with their own local IDs.
using Decision = DecisionFor<g::ValueId>;
using Distribution = DistributionFor<g::ValueId>;

// Fixed stochastic policy: temperature -> top-k -> top-p -> normalization.
// This is not an arbitrary sampler chain. Temperature is finite and positive;
// top_p is in (0,1]. Greedy temperature==0 belongs to the caller's greedy path.
// Stable TopK breaks equal-logit ties by original token ID. The nucleus keeps
// the smallest descending prefix whose mass reaches/exceeds top_p: entries
// whose PRIOR cumulative mass equals top_p are excluded. The first token is
// always retained. top_p<1 with top_k==0 requires a complete vocabulary sort.
// NaN/+Inf/all--Inf rows are invalid before filtering can hide bad columns.
// Invalid rows can have safe placeholder probabilities for eager execution;
// valid MUST join the decision/session publication check, never be ignored.
template<class R, class Value>
DistributionFor<Value> distribution(R &r, Value logits, Policy policy = {}) {
  const auto spec = r.spec(logits);
  if (spec.data_type() != Type::kFloat32 || spec.rank() != 2 || spec.dimensions()[0] != fixed(1) ||
      !spec.dimensions()[1].is_fixed() || spec.dimensions()[1].fixed_value() <= 0 ||
      spec.dimensions()[1].fixed_value() > (std::numeric_limits<std::int32_t>::max)())
    throw std::runtime_error("sampling policy requires one F32[1,V] row with positive I32 vocabulary");
  const auto vocabulary_size = static_cast<std::size_t>(spec.dimensions()[1].fixed_value());
  if (!std::isfinite(policy.temperature) || policy.temperature <= 0 || !std::isfinite(policy.top_p) ||
      policy.top_p <= 0 || policy.top_p > 1 || policy.top_k > vocabulary_size)
    throw std::runtime_error("sampling policy requires positive finite temperature, top_p in (0,1], top_k <= V");
  const auto zero = r.scalar(0), one = r.scalar(1);
  const auto no = r.cast(zero, Type::kBool), yes = r.cast(one, Type::kBool);
  const auto select = [&](Value condition, Value a, Value b) { return r.one(c::Select{}, {condition, a, b}); };
  const auto reduce = [&](c::ReductionKind kind, Value value) {
    return r.copy(r.reshape(r.one(c::Reduce{.kind = kind, .accumulation_type = Type::kFloat32}, {value}), {fixed(1)}));
  };
  const auto valid_mass = [&](Value log_weights) {
    const auto mass = reduce(c::ReductionKind::kSum, r.unary(c::UnaryKind::kExp, log_weights));
    // SumExp is nonnegative. A finite nonzero mass is valid; 0, NaN and
    // infinity are not. Native F32 division may approximate s/s, so comparing
    // that quotient with exact one would reject ordinary finite rows.
    const auto nonfinite = r.cast(r.binary(c::BinaryKind::kSubtract, mass, mass), Type::kBool);
    return select(r.cast(mass, Type::kBool), select(nonfinite, no, yes), no);
  };
  // Center first for small temperatures. For larger temperatures, halve
  // both logits and temperature before centering: opposite finite F32 ends
  // then have a representable difference. Unlike dividing by temperature
  // first, exact binary scaling also preserves nearby logits with a large
  // common offset. The default temperature keeps its original graph.
  const bool halve = policy.temperature > 1.F;
  const auto bounded = halve ? r.binary(c::BinaryKind::kMultiply, logits, r.scalar(.5F)) : logits;
  const auto centered = r.binary(c::BinaryKind::kSubtract, bounded, reduce(c::ReductionKind::kMaximum, bounded));
  // Check the full row before TopK/Select can mask any nonfinite input.
  const auto input_valid = valid_mass(centered);
  auto selected = r.binary(c::BinaryKind::kDivide, select(input_valid, centered, zero),
      r.scalar(halve ? policy.temperature * .5F : policy.temperature));
  if ((policy.top_k && policy.top_k < vocabulary_size) || policy.top_p < 1) {
    const auto count = policy.top_k ? policy.top_k : vocabulary_size;
    // Positive temperature preserves the original order. Half scaling can
    // round tiny distinct logits to a tie, so retain raw ranking and gather
    // its scaled weights only on this cold-selected wider-range path.
    const auto ranked = r.op(c::Operation(c::TopK{.count = count}),
        {halve ? select(input_valid, logits, zero) : selected});
    auto values = ranked.at(0);
    const auto indices = r.copy(r.reshape(ranked.at(1), {fixed(static_cast<std::int64_t>(count))}));
    if (halve) values = r.one(c::Gather{.axis = 1}, {selected, indices});
    if (policy.top_p < 1) {
      const auto probabilities = r.one(c::Softmax{.axis = 1}, {values});
      const auto prior = r.one(c::Scan{.axis = 1, .exclusive = true}, {probabilities});
      const auto under_cutoff = r.cast(r.unary(c::UnaryKind::kRelu,
          r.binary(c::BinaryKind::kSubtract, r.scalar(policy.top_p), prior)), Type::kBool);
      // Exclusive CDF begins at exact zero. Keep that first entry explicitly,
      // rather than relying on a possibly subnormal top_p arithmetic result.
      const auto first = select(r.cast(prior, Type::kBool), no, yes);
      const auto keep = select(first, yes, under_cutoff);
      values = select(keep, values, r.scalar(-(std::numeric_limits<float>::infinity)()));
    }
    const auto shape = c::TensorShape::create(std::array<std::int64_t, 2>{1, static_cast<std::int64_t>(vocabulary_size)});
    if (!shape) throw std::runtime_error("sampling vocabulary fill shape differs");
    const auto empty = r.one(c::Fill{.output_shape = *shape, .output_type = Type::kFloat32,
        .value = c::ScalarValue(-(std::numeric_limits<float>::infinity)())}, {});
    selected = r.one(c::Scatter{.axis = 1}, {empty, indices, values});
  }
  const auto normalized = r.one(c::Softmax{.axis = 1, .kind = c::SoftmaxKind::kLogSoftmax}, {selected});
  return {normalized, select(input_valid, valid_mass(normalized), no)};
}

// Ordinary precommit selection. The empty modifier branch emits the original
// greedy/temperature-top-k-top-p graph, without new ports or validity work.
// Raw NaN/+Inf must not disappear behind a mask. Dynamic bias itself must be
// finite even in disallowed columns. All-masked greedy has valid=false too.
// Failed rows only produce unusable placeholders: owners MUST check valid
// before publishing state, IDs or the original RNG cursor.
template<class R, class Value>
SelectionFor<Value> ordinary_selection(R &r, Value raw, std::optional<Value> identity,
    Policy policy = {}, LogitsInputsFor<Value> inputs = {}) {
  if (!inputs.bias && !inputs.allowed) {
    if (!identity) return {r.op(c::Operation(c::TopK{.count = 1}), {raw}).at(1), {}};
    const auto p = distribution(r, raw, policy);
    return {r.one(c::sampling_ops::categorical(), {p.log_probabilities, *identity}), p.valid};
  }
  const auto description = r.spec(raw);
  if (description.data_type() != Type::kFloat32 || description.rank() != 2 ||
      description.dimensions()[0] != fixed(1) || !description.dimensions()[1].is_fixed() ||
      description.dimensions()[1].fixed_value() <= 0 ||
      description.dimensions()[1].fixed_value() > (std::numeric_limits<std::int32_t>::max)())
    throw std::runtime_error("ordinary selection requires one F32[1,V] row");
  const auto require_input = [&](Value value, Type type) {
    const auto spec = r.spec(value);
    if (spec.data_type() != type || spec.rank() != 2 ||
        !std::ranges::equal(spec.dimensions(), description.dimensions()))
      throw std::runtime_error("logits modifier differs from its cold typed [1,V] port");
  };
  if (inputs.bias) require_input(*inputs.bias, Type::kFloat32);
  if (inputs.allowed) require_input(*inputs.allowed, Type::kBool);
  const auto zero = r.scalar(0), one = r.scalar(1);
  const auto no = r.cast(zero, Type::kBool), yes = r.cast(one, Type::kBool);
  const auto select = [&](Value condition, Value a, Value b) {
    return r.one(c::Select{}, {condition, a, b});
  };
  const auto reduce = [&](c::ReductionKind kind, Value value) {
    return r.copy(r.reshape(r.one(c::Reduce{.kind = kind,
        .accumulation_type = Type::kFloat32}, {value}), {fixed(1)}));
  };
  const auto row_valid = [&](Value row) {
    const auto centered = r.binary(c::BinaryKind::kSubtract, row,
        reduce(c::ReductionKind::kMaximum, row));
    const auto mass = reduce(c::ReductionKind::kSum, r.unary(c::UnaryKind::kExp, centered));
    const auto nonfinite = r.cast(r.binary(c::BinaryKind::kSubtract, mass, mass), Type::kBool);
    return select(r.cast(mass, Type::kBool), select(nonfinite, no, yes), no);
  };
  auto valid = row_valid(raw), processed = raw;
  if (inputs.bias) {
    // Every finite x has x-x==0, without overflowing a sum of large biases.
    // NaN and either infinity produce NaN, including masked columns.
    const auto differences = r.binary(c::BinaryKind::kSubtract, *inputs.bias, *inputs.bias);
    const auto bad = r.cast(reduce(c::ReductionKind::kSum,
        r.unary(c::UnaryKind::kAbs, differences)), Type::kBool);
    valid = select(bad, no, valid);
    processed = r.binary(c::BinaryKind::kAdd, raw, *inputs.bias);
  }
  if (inputs.allowed) processed = select(*inputs.allowed, processed,
      r.scalar(-(std::numeric_limits<float>::infinity)()));
  if (identity) {
    const auto p = distribution(r, processed, policy);
    return {r.one(c::sampling_ops::categorical(), {p.log_probabilities, *identity}),
        select(valid, p.valid, no)};
  }
  const auto combined = select(valid, row_valid(processed), no);
  const auto safe = select(combined, processed, zero);
  return {r.op(c::Operation(c::TopK{.count = 1}), {safe}).at(1), combined};
}

// log_p/log_q/log_bonus are F32[1,V], already normalized AFTER the caller's
// temperature/top-k/top-p policy. log_p verifies proposal; log_bonus is the
// target's next conditional distribution after that proposal. -Inf means zero
// mass. Normalization is a caller precondition, not silently recomputed here.
// Entire rows are checked for NaN/+Inf/all-zero probability, including columns
// that were not proposed. A proposed q(y)==0 is invalid, even when p(y)==0.
// proposal is I32[1]; out-of-range IDs fail the existing checked Gather.
// The two U32[1,4] identities are independent key/draw addresses, also distinct
// from the draw that produced proposal. The caller commits/restores their draw
// cursor with its original session transaction.
// Eager evaluation of an unselected zero residual is safe. A residual with no
// positive support on an actual rejection is invalid; no fallback is published.
template<class R, class Value>
DecisionFor<Value> decision(R &r, Value log_p, Value log_q, Value log_bonus, Value proposal,
                  Value accept_identity, Value final_identity) {
  const auto p = r.spec(log_p);
  if (p.data_type() != Type::kFloat32 || p.rank() != 2 || p.dimensions()[0] != fixed(1) ||
      !p.dimensions()[1].is_fixed() || p.dimensions()[1].fixed_value() <= 0 ||
      p.dimensions()[1].fixed_value() > (std::numeric_limits<std::int32_t>::max)())
    throw std::runtime_error("sampling requires one normalized F32[1,V] row with positive I32 vocabulary");
  for (const auto value : {log_q, log_bonus}) {
    const auto spec = r.spec(value);
    if (spec.data_type() != p.data_type() || !std::ranges::equal(spec.dimensions(), p.dimensions()))
      throw std::runtime_error("sampling p/q/bonus row shapes or types differ");
  }
  const auto require_port = [&](Value value, Type type, std::initializer_list<g::Dimension> shape) {
    const auto spec = r.spec(value);
    if (spec.data_type() != type || spec.rank() != shape.size() ||
        !std::equal(spec.dimensions().begin(), spec.dimensions().end(), shape.begin()))
      throw std::runtime_error("sampling proposal or key/draw port differs");
  };
  require_port(proposal, Type::kInt32, {fixed(1)});
  require_port(accept_identity, Type::kUInt32, {fixed(1), fixed(4)});
  require_port(final_identity, Type::kUInt32, {fixed(1), fixed(4)});

  const auto zero = r.scalar(0), one = r.scalar(1);
  const auto no = r.cast(zero, Type::kBool), yes = r.cast(one, Type::kBool);
  const auto select = [&](Value condition, Value a, Value b) {
    return r.one(c::Select{}, {condition, a, b});
  };
  const auto negate_bool = [&](Value value) { return select(value, no, yes); };
  const auto both = [&](Value a, Value b) { return select(a, b, no); };
  const auto either = [&](Value a, Value b) { return select(a, yes, b); };
  const auto sum = [&](Value value) {
    // There is exactly one request. Reducing all axes avoids a second row
    // convention while keeping the public output a one-element scalar port.
    return r.copy(r.reshape(r.one(c::Reduce{.kind = c::ReductionKind::kSum,
        .accumulation_type = Type::kFloat32}, {value}), {fixed(1)}));
  };
  const auto row_valid = [&](Value log_weights) {
    const auto mass = sum(r.unary(c::UnaryKind::kExp, log_weights));
    // Runtime subtraction preserves IEEE nonfinite detection without assuming
    // that a native approximate divide returns bit-exact s/s == 1. SumExp's
    // nonnegative domain also requires a separate nonzero-mass check. Neither
    // an unselected vocabulary column nor an eager branch may hide a NaN.
    const auto nonfinite = r.cast(r.binary(c::BinaryKind::kSubtract, mass, mass), Type::kBool);
    return both(r.cast(mass, Type::kBool), negate_bool(nonfinite));
  };
  const auto p_valid = row_valid(log_p), q_valid = row_valid(log_q), bonus_valid = row_valid(log_bonus);
  const auto safe_p = select(p_valid, log_p, zero), safe_q = select(q_valid, log_q, zero);
  const auto safe_bonus = select(bonus_valid, log_bonus, zero);
  const auto selected = [&](Value logits) {
    return r.copy(r.reshape(r.one(c::Gather{.axis = 1}, {logits, proposal}), {fixed(1)}));
  };
  const auto raw_qy = selected(log_q);
  const auto proposal_valid = negate_bool(r.cast(
      r.binary(c::BinaryKind::kSubtract, raw_qy, raw_qy), Type::kBool));
  const auto qy = select(proposal_valid, raw_qy, zero);
  const auto log_accept = r.binary(c::BinaryKind::kMinimum,
      r.binary(c::BinaryKind::kSubtract, selected(safe_p), qy), zero);
  // log(1-exp(a)) = softplus(a) + log(tanh(-a/2)), a<=0.
  // It preserves both exact endpoints (a=0 and a=-Inf) without cancellation.
  const auto minus_half = r.scalar(-0.5F);
  const auto log_one_minus = [&](Value a) {
    return r.binary(c::BinaryKind::kAdd, r.unary(c::UnaryKind::kSoftplus, a),
        r.unary(c::UnaryKind::kLog, r.unary(c::UnaryKind::kTanh,
            r.binary(c::BinaryKind::kMultiply, a, minus_half))));
  };
  const auto log_reject = log_one_minus(log_accept);
  const auto accept_weights = r.one(c::Concat{.axis = 1}, {
      r.copy(r.reshape(log_accept, {fixed(1), fixed(1)})),
      r.copy(r.reshape(log_reject, {fixed(1), fixed(1)}))});
  const auto choice = r.one(c::sampling_ops::categorical(), {accept_weights, accept_identity});
  const auto accepted = negate_bool(r.cast(r.cast(choice, Type::kFloat32), Type::kBool));

  // Construct log(max(p-q,0)) directly. Subtracting rounded exp(lp)-exp(lq)
  // would erase real residual mass for adjacent F32 log values. Per-column
  // support masks also avoid eager (-Inf)-(-Inf) and permit q(i)==0 away from y.
  const auto finite = [&](Value value) {
    return negate_bool(r.cast(r.binary(c::BinaryKind::kSubtract, value, value), Type::kBool));
  };
  const auto p_support = finite(safe_p), q_support = finite(safe_q);
  const auto finite_p = select(p_support, safe_p, zero), finite_q = select(q_support, safe_q, zero);
  const auto positive_difference = r.cast(r.unary(c::UnaryKind::kRelu,
      r.binary(c::BinaryKind::kSubtract, finite_p, finite_q)), Type::kBool);
  const auto residual_support = both(p_support, either(negate_bool(q_support), positive_difference));
  const auto negative_infinity = r.scalar(-(std::numeric_limits<float>::infinity)());
  const auto gap = select(q_support, r.binary(c::BinaryKind::kMinimum,
      r.binary(c::BinaryKind::kSubtract, finite_q, finite_p), zero), negative_infinity);
  const auto log_residual = select(residual_support,
      r.binary(c::BinaryKind::kAdd, finite_p, log_one_minus(gap)), negative_infinity);
  const auto residual_positive = r.cast(sum(r.cast(residual_support, Type::kFloat32)), Type::kBool);
  const auto safe_residual = select(residual_positive, log_residual, safe_p);
  // Only the selected branch enters Categorical, so p==q never submits an
  // all--Inf residual row. The final draw is independent of the accept draw.
  const auto final_weights = select(accepted, safe_bonus, safe_residual);
  const auto final_id = r.one(c::sampling_ops::categorical(), {final_weights, final_identity});
  const auto valid = both(both(p_valid, q_valid), both(bonus_valid,
      both(proposal_valid, either(accepted, residual_positive))));
  return {accepted, final_id, valid, p_valid, q_valid, bonus_valid, proposal_valid, residual_positive};
}

template<class Value> struct PrefixDecisionsFor final {
  std::vector<Value> q, accepted, final_ids, valid;
};

// The actual finite verifier supplies logits[m+1,V], proposals[m] and
// identities[m+1,4]. Identity row i<m is accept_i; row m is one independent
// final draw shared by the mutually exclusive residual/bonus outcomes.
// Create each already-policy-normalized q_i input at its original cold row.
// m==0 ignores proposals and uses target row0 with the sole final address.
// All validity ports must pass before publishing any selected prefix/state.
// Acceptance preserves these supplied verifier conditionals, not a promise of
// bit-identical distributions from a different target execution row width.
using PrefixDecisions = PrefixDecisionsFor<g::ValueId>;

template<class R, class Value>
PrefixDecisionsFor<Value> speculative_prefix(R &r, Value logits, Value identities,
                                  Value proposals, Policy policy = {}) {
  const auto shape = r.spec(logits), keys = r.spec(identities);
  if (shape.data_type() != Type::kFloat32 || shape.rank() != 2 ||
      !shape.dimensions()[0].is_fixed() || shape.dimensions()[0].fixed_value() <= 0 ||
      !shape.dimensions()[1].is_fixed() || shape.dimensions()[1].fixed_value() <= 0 ||
      keys.data_type() != Type::kUInt32 || keys.rank() != 2 ||
      keys.dimensions()[0] != shape.dimensions()[0] || keys.dimensions()[1] != fixed(4))
    throw std::runtime_error("speculative sampling requires matching finite F32 target rows and U32 identities");
  const auto count = static_cast<std::size_t>(shape.dimensions()[0].fixed_value()-1);
  if (count) {
    const auto ids = r.spec(proposals);
    if (ids.data_type() != Type::kInt32 || ids.rank() != 1 ||
        ids.dimensions()[0] != fixed(static_cast<std::int64_t>(count)))
      throw std::runtime_error("speculative sampling needs one original proposal per q row");
  }
  PrefixDecisionsFor<Value> result;
  std::vector<DistributionFor<Value>> target;
  for (std::size_t row = 0; row <= count; ++row)
    target.push_back(distribution(r,r.copy(r.slice(logits,0,
        fixed(static_cast<std::int64_t>(row)),fixed(1))),policy));
  const auto final_identity = r.copy(r.slice(identities,0,fixed(static_cast<std::int64_t>(count)),fixed(1)));
  if (!count) {
    result.final_ids.push_back(r.one(c::sampling_ops::categorical(),{target[0].log_probabilities,final_identity}));
    result.valid.push_back(target[0].valid);
  }
  const auto q_description = c::TensorDesc::dense(Type::kFloat32,
      std::array<std::int64_t,2>{1,shape.dimensions()[1].fixed_value()});
  if (!q_description) throw std::runtime_error("speculative q row geometry is invalid");
  for (std::size_t row = 0; row < count; ++row) {
    result.q.push_back(r.input(*q_description));
    const auto id = r.copy(r.slice(proposals,0,fixed(static_cast<std::int64_t>(row)),fixed(1)));
    const auto accept_identity = r.copy(r.slice(identities,0,fixed(static_cast<std::int64_t>(row)),fixed(1)));
    const auto selected = decision(r,target[row].log_probabilities,result.q.back(),
        target[row+1].log_probabilities,id,accept_identity,final_identity);
    result.accepted.push_back(selected.accepted); result.final_ids.push_back(selected.final_id);
    const auto no = r.cast(r.scalar(0),Type::kBool);
    const auto valid_target = r.one(c::Select{},{target[row].valid,target[row+1].valid,no});
    result.valid.push_back(r.one(c::Select{},{valid_target,selected.valid,no}));
  }
  return result;
}
} // namespace hyle::inference::sampling
