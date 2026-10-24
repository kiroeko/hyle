#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "hyle/compute/device.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/mem/buffer.h"

namespace batched_sink_test {
namespace c = hyle::compute;
namespace mem = hyle::mem;

struct Fixture final {
  static constexpr std::int64_t depth = 512;
  std::int64_t heads, queries, batch, keys;
  bool batched_mask;
  float scale = 1.F / std::sqrt(float(depth));
  c::Operation operation = c::model_ops::sink_attention(scale);
  std::array<c::TensorDesc, 5> descriptions;
  std::array<std::vector<float>, 5> payloads;

  explicit Fixture(std::int64_t h, bool independent_mask, std::int64_t q = 3,
                   std::int64_t b = 3, std::int64_t s = 17)
      : heads(h), queries(q), batch(b), keys(s), batched_mask(independent_mask) {
    const auto dense = [](std::initializer_list<std::int64_t> shape) {
      return c::TensorDesc::dense(c::DataType::kFloat32,
          std::span(shape.begin(), shape.size())).value();
    };
    descriptions = {dense({batch, queries, heads, depth}),
        dense({batch, keys, 1, depth}), dense({batch, keys, 1, depth}),
        batched_mask ? dense({batch, queries, keys}) : dense({queries, keys}),
        dense({heads})};
    for (std::size_t i = 0; i < payloads.size(); ++i)
      payloads[i].resize(descriptions[i].element_count().value());
    update(0);
  }

  void update(int phase) {
    for (std::size_t i = 0; i < payloads[0].size(); ++i)
      payloads[0][i] = float(std::int64_t((i * 13 + phase * 7) % 43) - 21) / 32;
    for (std::size_t i = 0; i < payloads[1].size(); ++i) {
      payloads[1][i] = float(std::int64_t((i * 7 + phase * 11) % 37) - 18) / 32;
      payloads[2][i] = 0.25F + float(std::int64_t((i * 11 + phase * 3) % 29) - 14) / 32;
    }
    const auto batches = batched_mask ? batch : 1;
    for (std::int64_t b = 0; b < batches; ++b)
      for (std::int64_t q = 0; q < queries; ++q)
        for (std::int64_t s = 0; s < keys; ++s)
          payloads[3][(b * queries + q) * keys + s] =
              s == 0 || (b * 5 + q * 3 + s + phase) % 7 >= 3
                  ? -float(s % 3) / 8 : -std::numeric_limits<float>::infinity();
    for (std::int64_t h = 0; h < heads; ++h)
      payloads[4][h] = float((h * 7 + phase * 3) % 17 - 8) / 8;
  }

  std::vector<float> expected() const {
    std::vector<float> result(payloads[0].size());
    std::vector<double> scores(keys), weights(keys);
    for (std::int64_t b = 0; b < batch; ++b)
      for (std::int64_t q = 0; q < queries; ++q)
        for (std::int64_t h = 0; h < heads; ++h) {
          const auto base = ((b * queries + q) * heads + h) * depth;
          double maximum = payloads[4][h];
          for (std::int64_t s = 0; s < keys; ++s) {
            double dot = 0;
            for (std::int64_t d = 0; d < depth; ++d)
              dot += double(payloads[0][base + d]) * payloads[1][(b * keys + s) * depth + d];
            scores[s] = dot * scale + payloads[3][((batched_mask ? b : 0) * queries + q) * keys + s];
            maximum = (std::max)(maximum, scores[s]);
          }
          double denominator = std::exp(double(payloads[4][h]) - maximum);
          for (std::int64_t s = 0; s < keys; ++s) {
            weights[s] = std::exp(scores[s] - maximum);
            denominator += weights[s];
          }
          for (std::int64_t d = 0; d < depth; ++d) {
            double value = 0;
            for (std::int64_t s = 0; s < keys; ++s)
              value += weights[s] * payloads[2][(b * keys + s) * depth + d];
            result[base + d] = float(value / denominator);
          }
        }
    return result;
  }
};

inline void check(std::span<const float> actual, std::span<const float> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  double worst = 0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    ASSERT_TRUE(std::isfinite(actual[i])) << i;
    worst = (std::max)(worst, std::abs(double(actual[i]) - expected[i]) /
        (2e-5 + 2e-5 * std::abs(double(expected[i]))));
  }
  EXPECT_LE(worst, 1.0);
}

inline void direct_case(c::Context &context, c::Queue &queue,
                        mem::MemoryResourceRef resource, Fixture &fixture) {
  auto prepared = c::prepare(context, fixture.operation, fixture.descriptions,
      std::span(&fixture.descriptions[0], 1), resource);
  ASSERT_TRUE(prepared) << prepared.error().message();
  const auto requirement = prepared->workspace_requirement();
  auto workspace = requirement.size == 0
      ? mem::Buffer{}
      : mem::Buffer::create(resource, requirement.size, requirement.alignment)
            .value();
  auto invocation = prepared->create_invocation(workspace.view()).value();
  std::array<mem::Buffer, 5> buffers;
  std::array<c::ConstTensorView, 5> inputs;
  for (std::size_t port = 0; port < buffers.size(); ++port) {
    buffers[port] = mem::Buffer::create(resource,
        fixture.descriptions[port].required_bytes().value(), 256).value();
    inputs[port] = c::ConstTensorView::create(fixture.descriptions[port], buffers[port].view()).value();
  }
  auto output = mem::Buffer::create(resource,
      fixture.descriptions[0].required_bytes().value(), 256).value();
  auto destination = c::TensorView::create(fixture.descriptions[0], output.view()).value();
  for (int phase = 0; phase < 2; ++phase) {
    fixture.update(phase);
    for (std::size_t port = 0; port < buffers.size(); ++port)
      ASSERT_TRUE(buffers[port].view().write(std::as_bytes(std::span(fixture.payloads[port]))).ok());
    auto done = invocation.submit(queue, inputs, std::span(&destination, 1));
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    std::vector<float> actual(output.size() / sizeof(float));
    ASSERT_TRUE(output.view().read(std::as_writable_bytes(std::span(actual))).ok());
    check(actual, fixture.expected());
  }
}

inline void direct(c::Context &context, mem::MemoryResourceRef resource) {
  auto queue = context.create_queue().value();
  for (const auto heads : {64, 128})
    for (const bool independent_mask : {false, true})
      for (const auto queries : {1, 3}) {
        SCOPED_TRACE(::testing::Message() << "H=" << heads << " mask3=" << independent_mask << " Q=" << queries);
        Fixture fixture(heads, independent_mask, queries);
        direct_case(context, queue, resource, fixture);
      }
}
} // namespace batched_sink_test
