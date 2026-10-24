#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "hyle/graph/executable.h"
#include "hyle/mem/buffer.h"

namespace ggml_discovery_test {
namespace c = hyle::compute;
namespace g = hyle::graph;

// TopK is an actual provider boundary in both default rosters. Exercise data
// and control re-entry, while retaining the useful independent interleaving.
inline void exercise(c::Context &context, hyle::mem::MemoryResourceRef resource) {
  const auto input_desc = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{2, 9}).value();
  for (int boundary = 0; boundary < 3; ++boundary) {
    SCOPED_TRACE(boundary);
    auto builder = g::GraphBuilder::create().value();
    const auto input = builder.input(input_desc).value();
    const auto first = builder.add_node(c::Operation(c::Unary{c::UnaryKind::kNeg}),
        std::array{input}).value();
    const auto first_result = builder.result(first).value();
    g::NodeId second, topk;
    if (boundary == 2) {
      topk = builder.add_node(c::Operation(c::TopK{.count = 3}), std::array{input}).value();
      second = builder.add_node(c::Operation(c::Unary{c::UnaryKind::kAbs}),
          std::array{first_result}).value();
    } else {
      second = builder.add_node(c::Operation(c::Unary{c::UnaryKind::kAbs}),
          std::array{first_result}).value();
      topk = builder.add_node(c::Operation(c::TopK{.count = 3}),
          std::array{boundary == 0 ? builder.result(second).value() : input}).value();
    }
    const auto prefix = builder.result(second).value();
    const auto values = builder.result(topk, 0).value();
    const auto ids = builder.result(topk, 1).value();
    const auto third = builder.add_node(c::Operation(c::Unary{c::UnaryKind::kNeg}),
        std::array{boundary == 0 ? values : prefix}).value();
    if (boundary == 1) {
      ASSERT_TRUE(builder.depends_on(topk, second).ok());
      ASSERT_TRUE(builder.depends_on(third, topk).ok());
    }
    const auto output = builder.add(c::Unary{c::UnaryKind::kAbs},
        builder.result(third).value()).value();
    const std::array published{prefix, values, ids, output};
    for (auto value : published)
      ASSERT_TRUE(builder.publish(value).ok());
    auto graph = std::move(builder).finish().value();
    const std::array domains{g::StorageDomain{.resource = resource}};
    const std::array targets{g::ExecutionTarget{
        .source = hyle::graph::ExecutionSource::acquire(context).value(), .default_storage = g::StorageDomainId{0}}};
    const std::array lanes{g::ExecutionLane{.target = g::ExecutionTargetId{0}}};
    auto compiled = g::compile(graph,
        {.storage_domains = domains, .targets = targets, .lanes = lanes});
    ASSERT_TRUE(compiled) << compiled.error().message();
    EXPECT_EQ(compiled->report.region_count(), boundary == 2 ? 1U : 2U);
    EXPECT_EQ(compiled->report.step_count(), boundary == 2 ? 2U : 3U);
    auto input_buffer = hyle::mem::Buffer::create(resource,
        input_desc.required_bytes().value(), 256).value();
    std::array<hyle::mem::Buffer, 4> buffers;
    auto invocation = compiled->executable.create_invocation().value();
    ASSERT_TRUE(invocation.bind(input, c::ConstTensorView::create(
        input_desc, input_buffer.view()).value()).ok());
    for (std::size_t i = 0; i < published.size(); ++i) {
      const auto desc = graph.value_description(published[i]).value();
      buffers[i] = hyle::mem::Buffer::create(resource,
          desc.required_bytes().value(), 256).value();
      ASSERT_TRUE(invocation.bind(published[i], c::TensorView::create(
          desc, buffers[i].view()).value()).ok());
    }
    auto queue = context.create_queue().value();
    const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
    ASSERT_TRUE(invocation.prepare_replay(queues).ok());
    for (int phase = 0; phase < 2; ++phase) {
      std::vector<float> data(18), absolute(18), selected(6);
      std::vector<std::int32_t> indices(6);
      for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = (float(int((i * 7 + phase * 11) % 23) - 11) + 0.125F) / 16.F;
        absolute[i] = std::abs(data[i]);
      }
      for (int row = 0; row < 2; ++row) {
        std::array<int, 9> order{0, 1, 2, 3, 4, 5, 6, 7, 8};
        const auto &source = boundary == 0 ? absolute : data;
        std::sort(order.begin(), order.end(), [&](int a, int b) {
          return source[row * 9 + a] > source[row * 9 + b];
        });
        for (int k = 0; k < 3; ++k) {
          indices[row * 3 + k] = order[k];
          selected[row * 3 + k] = source[row * 9 + order[k]];
        }
      }
      ASSERT_TRUE(input_buffer.view().write(std::as_bytes(std::span(data))).ok());
      auto done = invocation.submit(queue);
      ASSERT_TRUE(done) << done.error().message();
      ASSERT_TRUE(done->wait().ok());
      for (std::size_t port = 0; port < published.size(); ++port) {
        if (port == 2) {
          std::vector<std::int32_t> actual(6);
          ASSERT_TRUE(buffers[port].view().read(std::as_writable_bytes(std::span(actual))).ok());
          EXPECT_EQ(actual, indices);
        } else {
          const auto &expected = port == 0 || (port == 3 && boundary != 0)
                                     ? absolute : selected;
          std::vector<float> actual(expected.size());
          ASSERT_TRUE(buffers[port].view().read(std::as_writable_bytes(std::span(actual))).ok());
          EXPECT_EQ(actual, expected);
        }
      }
    }
  }
}

} // namespace ggml_discovery_test
