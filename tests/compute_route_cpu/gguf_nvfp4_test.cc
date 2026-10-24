#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "hyle/compute/cpu.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/ggml_cpu.h"
#include "../compute_models/gguf_nvfp4_fixture.h"
#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
#include "hyle/graph/cpu.h"
#include "hyle/graph/executable.h"
#endif

namespace {
namespace c = hyle::compute;
namespace mem = hyle::mem;
using Weights = nvfp4_test::Fixture;

mem::Buffer upload(const c::TensorDesc &desc, std::span<const std::byte> data) {
  auto buffer = mem::Buffer::create(mem::system_resource(),
      desc.required_bytes().value(), 64).value();
  EXPECT_TRUE(buffer.view().write(data).ok());
  return buffer;
}

std::vector<float> read(const mem::Buffer &buffer) {
  std::vector<float> result(buffer.view().size() / sizeof(float));
  EXPECT_TRUE(buffer.view().read(std::as_writable_bytes(std::span(result))).ok());
  return result;
}

std::vector<float> projection(const Weights &weights, std::span<const float> x,
                              std::span<const std::int32_t> ids = {}) {
  const auto count = x.size() / static_cast<std::size_t>(weights.columns);
  std::vector<float> result(count * static_cast<std::size_t>(weights.rows));
  for (std::size_t t = 0; t < count; ++t)
    for (std::int64_t n = 0; n < weights.rows; ++n) {
      double sum = 0;
      const auto expert = ids.empty() ? 0 : ids[t];
      for (std::int64_t k = 0; k < weights.columns; ++k)
        sum += double(x[t * weights.columns + k]) *
            weights.decoded[(expert * weights.rows + n) * weights.columns + k];
      result[t * weights.rows + n] = static_cast<float>(sum);
    }
  return result;
}

// The sparse +/-127 activation is exactly representable by the original Q8
// conversion. These dyadic products/sums are exact; this is not a general
// promise about association of native floating dot products.
template<class Update, class Reference>
void exercise(c::Operation op, std::span<const c::TensorDesc> descriptions,
              std::span<mem::Buffer> buffers, const c::TensorDesc &output,
              Update update, Reference reference, bool checked = false) {
  auto result = mem::Buffer::create(mem::system_resource(),
      output.required_bytes().value(), 64).value();
  std::vector<c::ConstTensorView> inputs;
  for (std::size_t i = 0; i < buffers.size(); ++i)
    inputs.push_back(c::ConstTensorView::create(descriptions[i], buffers[i].view()).value());
  auto destination = c::TensorView::create(output, result.view()).value();
#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
  namespace g = hyle::graph;
  auto context = g::cpu::create_context().value();
  auto builder = g::GraphBuilder::create().value();
  std::vector<g::ValueId> values;
  for (const auto &description : descriptions)
    values.push_back(builder.input(description).value());
  const auto node = builder.add_node(op, values).value();
  const auto value = builder.result(node).value();
  ASSERT_TRUE(builder.publish(value).ok());
  auto compiled = g::compile(std::move(builder).finish().value(), context);
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(compiled->report.step(0)->provider.id, c::ggml_cpu::kProviderId);
  auto invocation = compiled->executable.create_invocation().value();
  auto queue = context.create_queue().value();
  const std::array queues{g::QueueBinding{.lane = g::ExecutionLaneId{0}, .queue = &queue}};
  const auto bind = [&]() {
    for (std::size_t i = 0; i < values.size(); ++i)
      EXPECT_TRUE(invocation.bind(values[i], inputs[i]).ok());
    EXPECT_TRUE(invocation.bind(value, destination).ok());
    EXPECT_TRUE(invocation.prepare_replay(queues).ok());
  };
  bind();
  const auto run = [&]() -> hyle::Status {
    auto completion = invocation.submit(queue);
    if (!completion) return completion.error();
    return completion->wait();
  };
#else
  auto context = c::cpu::create_context().value();
  auto prepared = c::prepare(context, op, inputs, std::span(&destination, 1));
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, c::ggml_cpu::kProviderId);
  auto invocation = prepared->create_invocation().value();
  auto queue = context.create_queue().value();
  const auto run = [&]() -> hyle::Status {
    auto completion = invocation.submit(queue, inputs, std::span(&destination, 1));
    if (!completion) return completion.error();
    return completion->wait();
  };
#endif
  for (int pass = 0; pass < 2; ++pass) {
    update(pass);
    ASSERT_TRUE(run().ok());
    const auto actual = read(result), expected = reference();
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
      EXPECT_EQ(actual[i], expected[i]) << i;
  }
  if (!checked) return;
  const auto before = read(result);
  std::vector<std::int32_t> ids(buffers.back().view().size() / sizeof(std::int32_t));
  ASSERT_TRUE(buffers.back().view().read(std::as_writable_bytes(std::span(ids))).ok());
  ids[0] = -1;
  ASSERT_TRUE(buffers.back().view().write(std::as_bytes(std::span(ids))).ok());
  EXPECT_TRUE(run().IsOutOfRange());
  EXPECT_EQ(read(result), before);
  update(0);
  // Accepted checked-index failure is terminal for this invocation.
#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
  invocation = compiled->executable.create_invocation().value();
  bind();
#else
  invocation = prepared->create_invocation().value();
#endif
  ASSERT_TRUE(run().ok());
  EXPECT_EQ(read(result), reference());
}

#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
#define NVFP4_CPU_SUITE CpuNvfp4Graph
#else
#define NVFP4_CPU_SUITE CpuNvfp4Direct
#endif

TEST(NVFP4_CPU_SUITE, MatMulKeeps64BlockSubgroupsAndOriginalDot) {
  for (const auto k : {64, 128, 320})
    for (const auto tokens : {1, 9}) {
      SCOPED_TRACE(::testing::Message() << k << '/' << tokens);
      const Weights weights(5, k);
      auto x = weights.activations(tokens);
      const auto wd = c::gguf_weight_description(c::GgufQuantizedType::kNvfp4,
          c::DataType::kFloat32, k, weights.rows).value();
      const auto ad = c::TensorDesc::dense(c::DataType::kFloat32,
          std::array<std::int64_t, 2>{tokens, k}).value();
      const auto yd = c::TensorDesc::dense(c::DataType::kFloat32,
          std::array<std::int64_t, 2>{tokens, weights.rows}).value();
      std::array buffers{upload(ad, std::as_bytes(std::span(x))),
          upload(wd, std::as_bytes(std::span(weights.bytes)))};
      exercise(c::Operation(c::MatMul{}), std::array{ad, wd}, buffers, yd,
          [&](int pass) { x = weights.activations(tokens); if (pass) for (auto &v : x) v = -v;
            EXPECT_TRUE(buffers[0].view().write(std::as_bytes(std::span(x))).ok()); },
          [&] { return projection(weights, x); });
    }
}

TEST(NVFP4_CPU_SUITE, GroupedExpertsKeepRepeatedIdsAndCheckedFailure) {
  const Weights weights(5, 128, 3);
  for (const auto tokens : {1, 9}) {
    constexpr std::int64_t slots = 2;
    auto x = weights.activations(tokens * slots);
    std::vector<std::int32_t> ids(static_cast<std::size_t>(tokens * slots));
    const auto wd = c::gguf_grouped_weight_description(c::GgufQuantizedType::kNvfp4,
        c::DataType::kFloat32, 3, weights.rows, weights.columns).value();
    const auto ad = c::TensorDesc::dense(c::DataType::kFloat32,
        std::array<std::int64_t, 3>{tokens, slots, weights.columns}).value();
    const auto id = c::TensorDesc::dense(c::DataType::kInt32,
        std::array<std::int64_t, 2>{tokens, slots}).value();
    const auto yd = c::TensorDesc::dense(c::DataType::kFloat32,
        std::array<std::int64_t, 3>{tokens, slots, weights.rows}).value();
    std::array buffers{upload(wd, std::as_bytes(std::span(weights.bytes))),
        upload(ad, std::as_bytes(std::span(x))), upload(id, std::as_bytes(std::span(ids)))};
    exercise(c::model_ops::grouped_matmul(), std::array{wd, ad, id}, buffers, yd,
        [&](int pass) { for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = std::int32_t((i / 2 + pass) % 3);
          EXPECT_TRUE(buffers[2].view().write(std::as_bytes(std::span(ids))).ok()); },
        [&] { return projection(weights, x, ids); }, true);
  }
}

TEST(NVFP4_CPU_SUITE, GetRowsKeeps64BlockTailAndSelectedDecode) {
  for (const auto k : {64, 128, 320}) {
    const Weights weights(5, k);
    const auto wd = c::gguf_weight_description(c::GgufQuantizedType::kNvfp4,
        c::DataType::kFloat32, k, weights.rows).value().transposed().value();
    const auto id = c::TensorDesc::dense(c::DataType::kInt32,
        std::array<std::int64_t, 1>{3}).value();
    const auto yd = c::TensorDesc::dense(c::DataType::kFloat32,
        std::array<std::int64_t, 2>{3, k}).value();
    std::array<std::int32_t, 3> ids{4, 1, 1};
    std::array buffers{upload(wd, std::as_bytes(std::span(weights.bytes))),
        upload(id, std::as_bytes(std::span(ids)))};
    exercise(c::Operation(c::Gather{}), std::array{wd, id}, buffers, yd,
        [&](int pass) { ids = pass ? std::array<std::int32_t, 3>{0, 3, 0} : std::array<std::int32_t, 3>{4, 1, 1};
          EXPECT_TRUE(buffers[1].view().write(std::as_bytes(std::span(ids))).ok()); },
        [&] { std::vector<float> result; for (auto row : ids)
            result.insert(result.end(), weights.decoded.begin() + row * k, weights.decoded.begin() + (row + 1) * k);
          return result; }, true);
  }
}
#undef NVFP4_CPU_SUITE
} // namespace
