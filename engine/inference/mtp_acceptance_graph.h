#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "hyle/compute/operation.h"
#include "hyle/graph/graph.h"

// Cold fragments in the caller's existing Graph. They own no invocation,
// proposal storage, sampling cursor, stop policy or publication authority.
namespace hyle::inference::mtp {
namespace acceptance_graph_detail {
namespace c = hyle::compute;
namespace g = hyle::graph;
using Type = c::DataType;
constexpr auto fixed(std::int64_t n) noexcept { return g::Dimension::fixed(n); }

template<class R, class Value>
void require_port(R &r, Value value, Type type, std::size_t count) {
  const auto spec = r.spec(value);
  if (count > static_cast<std::size_t>((std::numeric_limits<std::int64_t>::max)()) ||
      spec.data_type() != type || spec.rank() != 1 ||
      spec.dimensions()[0] != fixed(static_cast<std::int64_t>(count)))
    throw std::runtime_error("MTP packet port has a different dtype or finite row extent");
}

template<class R, class Value>
void require_ids(R &r, std::span<const Value> ids, std::span<const Value> valid) {
  if (ids.empty() || ids.size() != valid.size() ||
      ids.size() >= static_cast<std::size_t>((std::numeric_limits<std::int64_t>::max)()))
    throw std::runtime_error("MTP packet needs matching nonempty final-ID and validity rows");
  for (std::size_t i = 0; i < ids.size(); ++i) {
    require_port(r, ids[i], Type::kInt32, 1);
    require_port(r, valid[i], Type::kBool, 1);
  }
}

template<class R>
auto integer(R &r, std::int32_t value) {
  const auto shape = c::TensorShape::create(std::array<std::int64_t, 1>{1});
  if (!shape) throw std::runtime_error("MTP packet scalar shape differs");
  return r.one(c::Fill{.output_shape = *shape, .output_type = Type::kInt32,
      .value = c::ScalarValue(value)}, {}, r.dense_spec(Type::kInt32, {fixed(1)}));
}

template<class R>
auto false_value(R &r) {
  const auto shape = c::TensorShape::create(std::array<std::int64_t, 1>{1});
  if (!shape) throw std::runtime_error("MTP packet predicate shape differs");
  return r.one(c::Fill{.output_shape = *shape, .output_type = Type::kBool,
      .value = c::ScalarValue(false)}, {}, r.dense_spec(Type::kBool, {fixed(1)}));
}

template<class R, class Value>
Value all_valid(R &r, std::span<const Value> valid, Value no) {
  auto result = valid.front();
  for (std::size_t i = 1; i < valid.size(); ++i)
    result = r.one(c::Select{}, {result, valid[i], no}, r.dense_spec(Type::kBool, {fixed(1)}));
  return result;
}
} // namespace acceptance_graph_detail

// proposals: I32[m], continuation: F32[m], accepted: m Bool[1] ports.
// final_ids/valid have max(m,1) I32[1]/Bool[1] ports. For m==0 the
// proposals and continuation values are ignored and may be empty handles.
// The caller supplies exact 0/1 continuation gates: proposal i is not a stop
// token AND i+1 is below the output quota. Acceptance stays device-derived.
// Output I32[m+5]: all_valid, consumed, next_id, first_agrees, emitted[0..m].
// Only emitted[0..consumed) may be published, and only when all_valid is 1.
template<class R, class Value>
Value sampled_prefix_packet(R &r, Value proposals, Value continuation,
    std::span<const Value> accepted, std::span<const Value> final_ids,
    std::span<const Value> valid) {
  namespace d = acceptance_graph_detail;
  const auto count = accepted.size();
  if (count >= static_cast<std::size_t>((std::numeric_limits<std::int32_t>::max)()) ||
      final_ids.size() != (count ? count : 1))
    throw std::runtime_error("MTP packet decision counts or consumed extent differ");
  d::require_ids(r, final_ids, valid);
  for (const auto flag : accepted) d::require_port(r, flag, d::Type::kBool, 1);
  if (count) {
    d::require_port(r, proposals, d::Type::kInt32, count);
    d::require_port(r, continuation, d::Type::kFloat32, count);
  }
  const auto zero = d::integer(r, 0), one = d::integer(r, 1), no = d::false_value(r);
  const auto choose = [&](Value condition, Value a, Value b) {
    return r.one(d::c::Select{}, {condition, a, b}, r.dense_spec(d::Type::kInt32, {d::fixed(1)}));
  };
  std::vector<Value> packet(4);
  packet.reserve(count + 5);
  for (std::size_t i = 0; i < count; ++i) {
    const auto proposal = r.copy(r.slice(proposals, 0, d::fixed(static_cast<std::int64_t>(i)), d::fixed(1)));
    packet.push_back(choose(accepted[i], proposal, final_ids[i]));
  }
  packet.push_back(final_ids.back());
  auto consumed = d::integer(r, static_cast<std::int32_t>(count + 1));
  auto next = final_ids.back();
  for (std::size_t i = count; i-- > 0;) {
    const auto gate = r.cast(r.copy(r.slice(continuation, 0,
        d::fixed(static_cast<std::int64_t>(i)), d::fixed(1))), d::Type::kBool);
    const auto proceed = r.one(d::c::Select{}, {gate, accepted[i], no},
        r.dense_spec(d::Type::kBool, {d::fixed(1)}));
    consumed = choose(proceed, consumed, d::integer(r, static_cast<std::int32_t>(i + 1)));
    next = choose(proceed, next, packet[4 + i]);
  }
  packet[0] = choose(d::all_valid(r, valid, no), one, zero);
  packet[1] = consumed; packet[2] = next;
  packet[3] = count ? choose(accepted.front(), one, zero) : zero;
  return r.concat(std::span<const Value>(packet), 0,
      r.dense_spec(d::Type::kInt32, {d::fixed(static_cast<std::int64_t>(packet.size()))}));
}

// Greedy modifier results share the same one-packet observation boundary.
// All rows remain checked, including rows beyond the ultimately used prefix.
template<class R, class Value>
Value validated_ids_packet(R &r, std::span<const Value> final_ids,
    std::span<const Value> valid) {
  namespace d = acceptance_graph_detail;
  d::require_ids(r, final_ids, valid);
  const auto zero = d::integer(r, 0), one = d::integer(r, 1), no = d::false_value(r);
  const auto predicate = d::all_valid(r, valid, no);
  std::vector<Value> packet;
  packet.reserve(final_ids.size() + 1);
  packet.push_back(r.one(d::c::Select{}, {predicate, one, zero}, r.dense_spec(d::Type::kInt32, {d::fixed(1)})));
  packet.insert(packet.end(), final_ids.begin(), final_ids.end());
  return r.concat(std::span<const Value>(packet), 0,
      r.dense_spec(d::Type::kInt32, {d::fixed(static_cast<std::int64_t>(packet.size()))}));
}
} // namespace hyle::inference::mtp
