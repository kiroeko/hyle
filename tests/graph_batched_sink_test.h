#pragma once

#include "compute_models/batched_sink_fixture.h"
#include "hyle/graph/executable.h"

namespace batched_sink_graph_test {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace mem = hyle::mem;

// Per-query indices select distinct KV sets from one shared history. A single
// batched SinkAttention replaces the former per-query nodes and final Concat.
inline void exercise(c::Context &context, mem::MemoryResourceRef resource) {
  batched_sink_test::Fixture fixture(64, true, 1);
  constexpr std::int64_t history_rows = 127;
  const auto cache_desc = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array{history_rows, fixture.depth}).value();
  const auto ids_desc = c::TensorDesc::dense(c::DataType::kInt32,
      std::array{fixture.batch, fixture.keys}).value();
  const std::array descriptions{cache_desc, ids_desc, fixture.descriptions[0],
      fixture.descriptions[3], fixture.descriptions[4]};
  auto builder = g::GraphBuilder::create().value();
  std::array<g::ValueId, 5> ports;
  for (std::size_t i = 0; i < ports.size(); ++i)
    ports[i] = builder.input(descriptions[i]).value();
  auto gather_node = builder.add_node(c::Operation(c::Gather{.axis = 0}),
      std::array{ports[0], ports[1]}).value();
  auto gathered = builder.result(gather_node).value();
  auto shape = c::TensorShape::create(
      std::array{fixture.batch, fixture.keys, std::int64_t{1}, fixture.depth}).value();
  auto kv = builder.add(c::View::reshape(shape).value(), gathered).value();
  auto sink_node = builder.add_node(fixture.operation,
      std::array{ports[2], kv, kv, ports[3], ports[4]});
  ASSERT_TRUE(sink_node) << sink_node.error().message();
  auto attended = builder.result(*sink_node).value();
  auto negative = builder.add(c::Unary{c::UnaryKind::kNeg}, attended).value();
  const std::array published{attended, negative};
  for (const auto value : published) ASSERT_TRUE(builder.publish(value).ok());
  auto graph = std::move(builder).finish().value();
  const std::array domains{g::StorageDomain{.resource = resource}};
  const std::array targets{g::ExecutionTarget{.source = hyle::graph::ExecutionSource::acquire(context).value(),
      .default_storage = g::StorageDomainId{0}}};
  const std::array lanes{g::ExecutionLane{.target = g::ExecutionTargetId{0}}};
  auto compiled = g::compile(graph,
      {.storage_domains = domains, .targets = targets, .lanes = lanes});
  ASSERT_TRUE(compiled) << compiled.error().message();
  auto invocation = compiled->executable.create_invocation().value();
  std::array<mem::Buffer, 5> inputs;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    inputs[i] = mem::Buffer::create(resource,
        descriptions[i].required_bytes().value(), 256).value();
    ASSERT_TRUE(invocation.bind(ports[i], c::ConstTensorView::create(
        descriptions[i], inputs[i].view()).value()).ok());
  }
  std::array<mem::Buffer, 2> outputs;
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    outputs[i] = mem::Buffer::create(resource,
        fixture.descriptions[0].required_bytes().value(), 256).value();
    ASSERT_TRUE(invocation.bind(published[i], c::TensorView::create(
        fixture.descriptions[0], outputs[i].view()).value()).ok());
  }
  auto queue = context.create_queue().value();
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  for (int phase = 0; phase < 2; ++phase) {
    fixture.update(phase);
    std::vector<float> cache(history_rows * fixture.depth);
    for (std::size_t i = 0; i < cache.size(); ++i)
      cache[i] = float(std::int64_t((i * 11 + phase * 13) % 71) - 35) / 32;
    std::vector<std::int32_t> ids(fixture.batch * fixture.keys);
    for (std::size_t i = 0; i < ids.size(); ++i) {
      ids[i] = std::int32_t((i * 19 + phase * 7) % history_rows);
      if (i % 7 == 0) ids[i] = 126; // Repeated high rows across selected sets.
      for (std::int64_t d = 0; d < fixture.depth; ++d) {
        const auto value = cache[std::size_t(ids[i]) * fixture.depth + d];
        fixture.payloads[1][i * fixture.depth + d] = value;
        fixture.payloads[2][i * fixture.depth + d] = value;
      }
    }
    ASSERT_TRUE(inputs[0].view().write(std::as_bytes(std::span(cache))).ok());
    ASSERT_TRUE(inputs[1].view().write(std::as_bytes(std::span(ids))).ok());
    for (std::size_t port = 2; port < inputs.size(); ++port) {
      const auto source = port == 2 ? 0U : port;
      ASSERT_TRUE(inputs[port].view().write(
          std::as_bytes(std::span(fixture.payloads[source]))).ok());
    }
    auto done = invocation.submit(queue);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    auto expected = fixture.expected();
    for (std::size_t port = 0; port < outputs.size(); ++port) {
      std::vector<float> actual(expected.size());
      ASSERT_TRUE(outputs[port].view().read(std::as_writable_bytes(std::span(actual))).ok());
      if (port == 1) for (auto &value : expected) value = -value;
      batched_sink_test::check(actual, expected);
    }
  }
}
} // namespace batched_sink_graph_test
