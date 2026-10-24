#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/batch_ops.h"
#include "hyle/graph/batch.h"
#include "hyle/graph/result.h"
#if defined(HYLE_NATIVE_RAGGED_APPLE)
#include "hyle/compute/apple_memory.h"
#include "hyle/graph/apple.h"
#else
#include "hyle/compute/vulkan_memory.h"
#include "hyle/graph/vulkan.h"
#endif

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace gb = g::batch;
#if defined(HYLE_NATIVE_RAGGED_APPLE)
namespace native = c::apple;
namespace ng = g::apple;
#else
namespace native = c::vulkan;
namespace ng = g::vulkan;
#endif
using F16 = hyle::simd::float16_t;
using BF16 = hyle::simd::bfloat16_t;
constexpr unsigned heads = 8, kv_heads = 2, depth = 64;

void check(hyle::Status status, const char *what) {
  if (!status.ok()) throw std::runtime_error(std::string(what) + ": " + std::string(status.message()));
}
template<class T> T take(hyle::Result<T> result, const char *what) {
  if (!result) throw std::runtime_error(std::string(what) + ": " + std::string(result.error().message()));
  return std::move(*result);
}
struct Route final {
  native::Context context = native::Context::create().value();
#if defined(HYLE_NATIVE_RAGGED_APPLE)
  native::MemoryResource device = native::MemoryResource::create(context).value();
  native::MemoryResource staging = native::MemoryResource::create(context).value();
#else
  native::MemoryResource device = native::MemoryResource::create(context, native::MemoryKind::kDevice).value();
  native::MemoryResource staging = native::MemoryResource::create(context, native::MemoryKind::kStaging).value();
#endif
  c::LeasePool host = c::LeasePool::create(staging.ref(), 1u << 20).value();
  c::LeasePool pool = c::LeasePool::create(device.ref(), 2u << 20).value();
  native::CopyBatchProgram copy_program = native::CopyBatchProgram::prepare(context,
      std::array{c::CopyResourceSpec{staging.ref(), 1u << 20},
                 c::CopyResourceSpec{device.ref(), 1u << 20}}).value();
  native::CopyBatchInvocation copy = native::CopyBatchInvocation::create(copy_program).value();
  template<class T> c::ReadLease load(const std::vector<T> &values) {
    const auto bytes = std::as_bytes(std::span(values));
    auto host_writer = host.allocate(bytes.size()).value();
    check(host_writer.view()->write(bytes), "write original copy source");
    auto payload = host_writer.publish().value();
    check(copy.reset(), "reset original F2 upload");
    check(copy.bind_source({0}, payload), "bind original F2 source");
    check(copy.bind_mutable({1}, pool.allocate(bytes.size()).value()), "bind original F2 destination");
    const std::array<c::CopyPhase, 1> phases{{{}}};
    const std::array<c::CopyRange, 1> ranges{{{{0}, {1}, {0}, 0, 0, bytes.size()}}};
    check(copy.prepare_replay(phases, ranges), "prepare original F2 upload");
    { auto done = take(copy.submit(), "submit original F2 upload"); check(done.wait(), "wait original F2 upload"); }
    auto writer = take(copy.unbind_mutable({1}), "return original uploaded writer");
    check(copy.reset(), "release original F2 source");
    return writer.publish().value();
  }
};
std::vector<std::uint16_t> read(Route &route, const g::GraphOutput &output) {
  const auto description = output.description().value();
  std::vector<std::uint16_t> result(description.element_count().value());
#if defined(HYLE_NATIVE_RAGGED_APPLE)
  (void)route;
  check(output.storage_lease().value().view()->read(std::as_writable_bytes(std::span(result))), "read original Apple Graph output");
#else
  const auto tensor = route.device.binding(output.storage_lease().value(), description).value();
  check(route.context.download(tensor, std::as_writable_bytes(std::span(result))), "download original Vulkan Graph output");
#endif
  return result;
}
template<class T> c::BatchCapacity capacity(unsigned requests, unsigned queries, unsigned keys) {
  const auto storage = c::CacheStorage::plain(half_test::data_type<T>()).value();
  return {.storage = c::BatchStorage::kRagged, .query_type = half_test::data_type<T>(),
      .output_type = half_test::data_type<T>(), .key_storage = storage, .value_storage = storage,
      .max_requests = requests, .max_query_rows = queries, .max_kv_rows = keys,
      .max_pages = 0, .max_page_refs = 0, .query_heads = heads, .kv_heads = kv_heads,
      .qk_head_dim = depth, .value_head_dim = depth, .page_size = 1};
}
template<class T> struct Inputs final {
  std::array<c::TensorDesc, 3> descriptions;
  std::array<std::vector<T>, 3> values;
  std::array<c::ReadLease, 3> ready;
  Inputs(Route &route, const c::BatchCapacity &cap)
      : Inputs(route, cap, cap.max_query_rows, true) {}
  Inputs(Route &route, const c::BatchCapacity &cap, unsigned query_rows,
      bool poison_query_tail = false)
      : descriptions{c::batch_query_description(cap, query_rows).value(),
          c::batch_cache_description(cap, c::BatchCacheRole::kKey).value(),
          c::batch_cache_description(cap, c::BatchCacheRole::kValue).value()} {
    for (unsigned port = 0; port < 3; ++port) {
      const auto rows = port ? cap.max_kv_rows : query_rows;
      const auto count = port ? kv_heads : heads;
      values[port].resize(std::size_t(rows) * count * depth);
      for (unsigned row = 0; row < rows; ++row)
        for (unsigned head = 0; head < count; ++head)
          for (unsigned d = 0; d < depth; ++d) {
            const double value = row + 1 == rows && (port || poison_query_tail) ?
                std::numeric_limits<double>::quiet_NaN() :
                port == 0 ? double(int((row * 7 + head * 3 + d) % 13) - 6) / 32. :
                port == 1 ? double(int((row * 5 + head * 2 + d * 3) % 17) - 8) / 64. :
                            double(int((row * 11 + head * 7 + d * 3) % 29) - 14) / 32.;
            values[port][(std::size_t(row) * count + head) * depth + d] = half_test::encode<T>(value);
          }
      ready[port] = route.load(values[port]);
    }
  }
};
struct Recipe final {
  g::Graph graph;
  std::array<g::ValueId, 3> inputs;
  std::array<g::NodeId, 3> producers;
  g::ValueId metadata;
  std::array<g::NodeId, 2> nodes;
  std::array<g::ValueId, 2> outputs, negatives;
  unsigned count = 1;
};
Recipe recipe(c::BatchCapacity cap, bool generated, bool shared_metadata, unsigned query_rows) {
  auto builder = g::GraphBuilder::create().value();
  Recipe result;
  std::array<g::ValueId, 4> ports;
  for (unsigned port = 0; port < 3; ++port) {
    const auto description = port == 0 ? c::batch_query_description(cap, query_rows).value() :
        c::batch_cache_description(cap, port == 1 ? c::BatchCacheRole::kKey : c::BatchCacheRole::kValue).value();
    result.inputs[port] = builder.input(description).value();
    ports[port] = result.inputs[port];
    if (generated) {
      result.producers[port] = builder.add_node(c::Operation(c::Copy{}), std::array{ports[port]}).value();
      ports[port] = builder.result(result.producers[port]).value();
    }
  }
  result.metadata = builder.input(c::batch_metadata_description(cap).value()).value();
  ports[3] = result.metadata;
  result.count = shared_metadata ? 2 : 1;
  for (unsigned branch = 0; branch < result.count; ++branch) {
    const c::RaggedAttention operation{.capacity = cap, .scale = .125F,
        .causal = true, .window_left = branch ? 2 : -1};
    result.nodes[branch] = builder.add_node(c::Operation::custom(operation), ports).value();
    result.outputs[branch] = builder.result(result.nodes[branch]).value();
    result.negatives[branch] = builder.add(c::Unary{c::UnaryKind::kNeg}, result.outputs[branch]).value();
    check(builder.publish(result.outputs[branch]), "publish ordinary ragged result");
    check(builder.publish(result.negatives[branch]), "publish ordinary downstream Neg");
  }
  result.graph = std::move(builder).finish().value();
  return result;
}
Recipe recipe(c::BatchCapacity cap, bool generated = false, bool shared_metadata = false) {
  return recipe(cap, generated, shared_metadata, cap.max_query_rows);
}
g::CompiledGraph compile(Route &route, const Recipe &model) {
  const std::array domains{g::StorageDomain{route.device.ref()}, g::StorageDomain{route.staging.ref()}};
  const std::array targets{g::ExecutionTarget{ng::execution_source(route.context).value(), g::StorageDomainId{0}}};
  return take(gb::compile(model.graph, {.storage_domains = domains, .targets = targets}), "ordinary native RaggedAttention compile");
}
std::size_t step(const g::CompileReport &report, g::NodeId node) {
  for (std::size_t i = 0; i < report.step_count(); ++i)
    for (std::size_t j = 0; j < report.step(i)->node_count; ++j)
      if (report.step_node(i, j).value() == node) return i;
  throw std::runtime_error("ordinary native node absent from actual compile report");
}
std::vector<gb::NodeLayout> layouts(const Recipe &model, const c::BatchLayoutVersion &layout) {
  std::vector<gb::NodeLayout> result;
  for (unsigned branch = 0; branch < model.count; ++branch) result.push_back({model.nodes[branch], layout});
  return result;
}
template<class T> std::vector<double> expected(const c::BatchCapacity &cap,
    const Inputs<T> &data, const c::BatchLayoutVersion &layout, int window) {
  (void)cap;
  std::vector<double> result(data.descriptions[0].element_count().value(), 0.);
  const auto host = layout.host();
  for (unsigned request = 0; request < layout.facts().requests; ++request) {
    const auto nq = unsigned(host.query_indptr[request + 1] - host.query_indptr[request]);
    const auto nk = unsigned(host.kv_indptr[request + 1] - host.kv_indptr[request]);
    for (unsigned row = 0; row < nq; ++row) {
      const auto end = nk - nq + row + 1;
      const auto begin = window < 0 ? 0u : end > unsigned(window + 1) ? end - unsigned(window + 1) : 0u;
      for (unsigned head = 0; head < heads; ++head) {
        std::vector<double> scores(end - begin);
        double maximum = -std::numeric_limits<double>::infinity(), denominator = 0;
        for (unsigned column = begin; column < end; ++column) {
          double dot = 0;
          for (unsigned d = 0; d < depth; ++d) {
            const auto qi = ((std::size_t(host.query_indptr[request]) + row) * heads + head) * depth + d;
            const auto ki = ((std::size_t(host.kv_indptr[request]) + column) * kv_heads + head / 4) * depth + d;
            dot += half_test::decode(data.values[0][qi]) * half_test::decode(data.values[1][ki]);
          }
          scores[column - begin] = dot * .125;
          maximum = std::max(maximum, scores[column - begin]);
        }
        for (auto &score : scores) { score = std::exp(score - maximum); denominator += score; }
        for (unsigned d = 0; d < depth; ++d)
          for (unsigned column = begin; column < end; ++column) {
            const auto vi = ((std::size_t(host.kv_indptr[request]) + column) * kv_heads + head / 4) * depth + d;
            const auto oi = ((std::size_t(host.query_indptr[request]) + row) * heads + head) * depth + d;
            result[oi] += scores[column - begin] / denominator * half_test::decode(data.values[2][vi]);
          }
      }
    }
  }
  return result;
}
template<class T> void verify(Route &route, g::Invocation &slot, const Recipe &model,
    const c::BatchCapacity &cap, const Inputs<T> &data, const c::BatchLayoutVersion &layout) {
  for (unsigned branch = 0; branch < model.count; ++branch) {
    const auto actual = read(route, slot.result(model.outputs[branch]).value());
    const auto negative = read(route, slot.result(model.negatives[branch]).value());
    const auto wanted = expected(cap, data, layout, branch ? 2 : -1);
    ASSERT_EQ(actual.size(), wanted.size()); ASSERT_EQ(negative.size(), wanted.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
      const double value = half_test::decode_bits<T>(actual[i]);
      ASSERT_TRUE(std::isfinite(value)) << i;
      // Unchanged independent FP64 budget from native_ragged_layout_test.h.
      ASSERT_NEAR(value, wanted[i], (std::is_same_v<T, BF16> ? 8e-3 : 2e-3)) << i;
      if (i >= std::size_t(layout.facts().query_rows) * heads * depth) ASSERT_EQ(actual[i], 0u) << i;
      ASSERT_EQ(negative[i], std::uint16_t(actual[i] ^ 0x8000u)) << i;
    }
  }
}
template<class T> void run(Route &route, g::Invocation &slot, const Recipe &model,
    const c::BatchCapacity &cap, const Inputs<T> &data, const c::BatchLayoutVersion &layout,
    const c::ReadLease &replacement_metadata, bool bind_payloads = true) {
  const auto selected = layouts(model, layout);
  check(gb::prepare(slot, {}, selected, {.host_workspace = route.staging.ref()}), "prepare exact owned ragged layout");
  // This is a valid native ready allocation with the right descriptor/domain.
  // Reject its ordinary rebind without losing the canonical cold configuration.
  const auto refused = slot.bind(model.metadata, c::batch_metadata_description(cap).value(), replacement_metadata);
  ASSERT_TRUE(refused.IsFailedPrecondition()) << refused.message();
  if (bind_payloads)
    for (unsigned port = 0; port < 3; ++port)
      check(slot.bind(model.inputs[port], data.descriptions[port], data.ready[port]), "bind only ordinary Q/K/V");
  check(slot.prepare_replay({}, {.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kAllow}), "prepare ordinary Graph commands");
  { auto done = take(slot.submit(), "submit ordinary RaggedAttention Graph"); check(done.wait(), "wait original Graph Completion"); }
  verify(route, slot, model, cap, data, layout);
}
template<class T> void unequal_requests() {
  SCOPED_TRACE((std::is_same_v<T, BF16> ? "BF16" : "F16"));
  Route route;
#if !defined(HYLE_NATIVE_RAGGED_APPLE)
  ASSERT_FALSE(route.context.environment().software_device);
#endif
  const auto cap = capacity<T>(3, 6, 12);
  const auto schema = c::BatchSchema::create(cap).value();
  const std::array<std::int32_t, 4> q{0, 1, 4, 4}, kv{0, 3, 9, 9}, next_kv{0, 4, 11, 11}, next_q{0, 2, 5, 5};
  const auto first = schema.version({q, kv, {}, {}}).value();
  const auto same_q = schema.version({q, next_kv, {}, {}}).value();
  const auto changed_q = schema.version({next_q, next_kv, {}, {}}).value();
  const Inputs<T> data(route, cap);
  const auto replacement = route.load(std::vector<std::uint8_t>(c::batch_metadata_description(cap).value().required_bytes().value()));
  const auto model = recipe(cap);
  auto compiled = compile(route, model);
  EXPECT_EQ(compiled.report.step(step(compiled.report, model.nodes[0]))->node_count, 1u);
  std::array slots{compiled.executable.create_invocation({.resources = g::ResourceOptions{}}).value(),
                   compiled.executable.create_invocation({.resources = g::ResourceOptions{}}).value()};
  run(route, slots[0], model, cap, data, first, replacement);
  auto old = slots[0].result(model.outputs[0]).value();
  const auto old_bits = read(route, old);
  run(route, slots[1], model, cap, data, first, replacement);
  // Reuse the original ordinary lease bindings through both kinds of native
  // layout update; only the configuration episode changes.
  run(route, slots[1], model, cap, data, same_q, replacement, false);
  EXPECT_NE(expected(cap, data, first, -1), expected(cap, data, same_q, -1));
  EXPECT_EQ(read(route, old), old_bits);
  EXPECT_TRUE(gb::prepare(slots[0], {}, layouts(model, changed_q),
      {.host_workspace = route.staging.ref()}).IsUnavailable());
  EXPECT_EQ(read(route, old), old_bits);
  old.reset();
  run(route, slots[0], model, cap, data, changed_q, replacement, false);
}
TEST(NativeRaggedAttentionGraph, UnequalRequestsUpdateSlotsKeepOldOutputAndZeroCapacityTail) {
  unequal_requests<F16>();
  unequal_requests<BF16>();
}
template<class T> void generated_payloads() {
  SCOPED_TRACE((std::is_same_v<T, BF16> ? "BF16" : "F16"));
  Route route;
#if !defined(HYLE_NATIVE_RAGGED_APPLE)
  ASSERT_FALSE(route.context.environment().software_device);
#endif
  const auto cap = capacity<T>(8, 12, 28);
  const auto schema = c::BatchSchema::create(cap).value();
  const std::array<std::int32_t, 9> q{0, 1, 3, 3, 4, 6, 7, 8, 9}, kv{0, 2, 6, 6, 9, 14, 16, 20, 23};
  const auto layout = schema.version({q, kv, {}, {}}).value();
  const Inputs<T> data(route, cap);
  const auto replacement = route.load(std::vector<std::uint8_t>(c::batch_metadata_description(cap).value().required_bytes().value()));
  // These are ordinary device-produced Graph values, not externally bound
  // cache frames. Both real attention nodes share one owned metadata input.
  const auto model = recipe(cap, true, true);
  auto compiled = compile(route, model);
  const auto attention_step = step(compiled.report, model.nodes[0]);
  ASSERT_NE(attention_step, step(compiled.report, model.nodes[1]));
  for (const auto producer : model.producers) EXPECT_NE(step(compiled.report, producer), attention_step);
  auto slot = compiled.executable.create_invocation({.resources = g::ResourceOptions{}}).value();
  run(route, slot, model, cap, data, layout, replacement);
}
TEST(NativeRaggedAttentionGraph, EightRequestsUseGraphProducedPayloadsAndSharedOwnedMetadata) {
  generated_payloads<F16>();
  generated_payloads<BF16>();
}
template<class T> void exact_query_extent() {
  SCOPED_TRACE((std::is_same_v<T, BF16> ? "BF16" : "F16"));
  Route route;
#if !defined(HYLE_NATIVE_RAGGED_APPLE)
  ASSERT_FALSE(route.context.environment().software_device);
#endif
  // The immutable schema admits C=128. This concrete Graph and its real Q
  // backing have R=1; the owned layout runs A=1 with no 127-row output tail.
  const auto cap = capacity<T>(1, 128, 4);
  const auto schema = c::BatchSchema::create(cap).value();
  const std::array<std::int32_t, 2> query{0, 1}, too_many_queries{0, 2}, keys{0, 3};
  const auto layout = schema.version({query, keys, {}, {}}).value();
  const auto overflow = schema.version({too_many_queries, keys, {}, {}}).value();
  const Inputs<T> data(route, cap, 1);
  ASSERT_EQ(data.descriptions[0].dimensions()[0], 1);
  ASSERT_EQ(data.ready[0].view()->size(), std::size_t(heads) * depth * sizeof(T));
  const auto replacement = route.load(std::vector<std::uint8_t>(
      c::batch_metadata_description(cap).value().required_bytes().value()));
  const auto model = recipe(cap, false, true, 1);
  ASSERT_EQ(model.graph.value_description(model.outputs[0])->dimensions()[0], 1);
  auto compiled = compile(route, model);
  std::array slots{compiled.executable.create_invocation({.resources = g::ResourceOptions{}}).value(),
                   compiled.executable.create_invocation({.resources = g::ResourceOptions{}}).value()};
  run(route, slots[0], model, cap, data, layout, replacement);
  auto old = slots[0].result(model.outputs[0]).value();
  ASSERT_EQ(old.description()->dimensions()[0], 1);
  const auto old_bits = read(route, old);
  run(route, slots[1], model, cap, data, layout, replacement);
  // A=2 is schema-valid but exceeds this concrete port extent. Refuse it
  // during cold preparation, before changing native layout or metadata.
  const auto refused = gb::prepare(slots[1], {}, layouts(model, overflow),
      {.host_workspace = route.staging.ref()});
  ASSERT_TRUE(refused.IsInvalidArgument()) << refused.message();
  EXPECT_EQ(read(route, old), old_bits);
  run(route, slots[1], model, cap, data, layout, replacement);
  EXPECT_EQ(read(route, old), old_bits);
}
TEST(NativeRaggedAttentionGraph, ConcreteOneRowWithinLargeSchemaRejectsMoreActiveQueries) {
  exact_query_extent<F16>();
  exact_query_extent<BF16>();
}
} // namespace
