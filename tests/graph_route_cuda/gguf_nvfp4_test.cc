#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include <gtest/gtest.h>
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cuda.h>

#include "hyle/compute/cuda.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/providers/ggml_cuda.h"
#include "hyle/graph/cuda.h"
#include "../compute_models/gguf_nvfp4_fixture.h"
#include "../compute_provider_native/half_test_support.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace mem = hyle::mem;
using Weights = nvfp4_test::Fixture;

c::TensorDesc dense(c::DataType type, std::initializer_list<std::int64_t> shape) {
  return c::TensorDesc::dense(type, {shape.begin(), shape.size()}).value();
}

mem::Buffer upload(mem::MemoryResourceRef memory, std::span<const std::byte> bytes) {
  auto result = mem::Buffer::create(memory, bytes.size(), 256).value();
  EXPECT_TRUE(result.view().write(bytes).ok());
  return result;
}

std::vector<float> project(const Weights &weights, std::span<const float> input,
                           std::span<const std::int32_t> ids) {
  const auto count = input.size() / static_cast<std::size_t>(weights.columns);
  std::vector<float> output(count * static_cast<std::size_t>(weights.rows));
  for (std::size_t row = 0; row < count; ++row)
    for (std::int64_t n = 0; n < weights.rows; ++n) {
      double sum = 0;
      const auto expert = ids.empty() ? 0 : ids[row];
      for (std::int64_t k = 0; k < weights.columns; ++k)
        sum += double(input[row * weights.columns + k]) *
            weights.decoded[(expert * weights.rows + n) * weights.columns + k];
      output[row * weights.rows + n] = static_cast<float>(sum);
    }
  return output;
}

// Independent raw native coverage of MMQ's inverse-map/scatter consumer.
void broadcast_native(std::int64_t slots) {
  constexpr std::int64_t tokens = 9;
  const Weights weights(5, 128, 3);
  std::unique_ptr<ggml_context, decltype(&ggml_free)> context_owner(
      ggml_init({1U << 20, nullptr, true}), ggml_free);
  auto *context = context_owner.get();
  ASSERT_NE(context, nullptr);
  auto *bank = ggml_new_tensor_3d(context, GGML_TYPE_NVFP4,
      weights.columns, weights.rows, weights.experts);
  auto *x = ggml_new_tensor_3d(context, GGML_TYPE_F32, weights.columns, 1, tokens);
  auto *indices = ggml_new_tensor_2d(context, GGML_TYPE_I32, slots, tokens);
  auto *y = ggml_mul_mat_id(context, bank, x, indices);
  ggml_set_input(bank); ggml_set_input(x); ggml_set_input(indices); ggml_set_output(y);
  auto *graph = ggml_new_graph(context);
  ggml_build_forward_expand(graph, y);
  auto backend = ggml_backend_cuda_init(0);
  ASSERT_NE(backend, nullptr);
  std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend_owner(backend, ggml_backend_free);
  auto allocation = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
  std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocation_owner(allocation, ggml_gallocr_free);
  ASSERT_TRUE(ggml_gallocr_alloc_graph(allocation, graph));
  ggml_backend_tensor_set(bank, weights.bytes.data(), 0, weights.bytes.size());
  const auto activation = weights.activations(tokens);
  ggml_backend_tensor_set(x, activation.data(), 0, activation.size() * sizeof(float));
  std::vector<float> expanded(static_cast<std::size_t>(tokens * slots * weights.columns));
  for (std::int64_t t = 0; t < tokens; ++t)
    for (std::int64_t slot = 0; slot < slots; ++slot)
      std::copy_n(activation.begin() + t * weights.columns, weights.columns,
          expanded.begin() + (t * slots + slot) * weights.columns);
  for (const bool distinct : {false, true}) {
    std::vector<std::int32_t> ids(static_cast<std::size_t>(tokens * slots));
    for (std::int64_t t = 0; t < tokens; ++t)
      for (std::int64_t slot = 0; slot < slots; ++slot)
        ids[static_cast<std::size_t>(t * slots + slot)] =
            static_cast<std::int32_t>((t + (distinct ? slot : 0)) % weights.experts);
    ggml_backend_tensor_set(indices, ids.data(), 0, ids.size() * sizeof(std::int32_t));
    EXPECT_EQ(ggml_backend_graph_compute(backend, graph), GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    std::vector<float> actual(static_cast<std::size_t>(tokens * slots * weights.rows));
    ggml_backend_tensor_get(y, actual.data(), 0, actual.size() * sizeof(float));
    EXPECT_EQ(actual, project(weights, expanded, ids)) << "A=1 S=" << slots << " distinct=" << distinct;
  }
}

// Original public operations, ordinary default provider roster, real checked
// indices and caller-owned buffers. The independently written packed fixture
// uses sparse exact-Q8 activations; only this dyadic domain is compared exactly.
void exercise(const Weights &weights, std::int64_t tokens, bool grouped, bool gather,
              std::int64_t slots = 2) {
  SCOPED_TRACE(::testing::Message() << "K=" << weights.columns << " T=" << tokens
      << " S=" << slots << " grouped=" << grouped << " gather=" << gather);
  const auto wd = grouped ? c::gguf_grouped_weight_description(c::GgufQuantizedType::kNvfp4,
      c::DataType::kFloat32, weights.experts, weights.rows, weights.columns).value() :
      c::gguf_weight_description(c::GgufQuantizedType::kNvfp4,
          c::DataType::kFloat32, weights.columns, weights.rows).value();
  const auto table = gather ? wd.transposed().value() : wd;
  const auto ad = grouped ? dense(c::DataType::kFloat32, {tokens, slots, weights.columns}) :
      dense(c::DataType::kFloat32, {tokens, weights.columns});
  const auto id = grouped ? dense(c::DataType::kInt32, {tokens, slots}) : dense(c::DataType::kInt32, {3});
  const auto yd = grouped ? dense(c::DataType::kFloat32, {tokens, slots, weights.rows}) :
      gather ? dense(c::DataType::kFloat32, {3, weights.columns}) : dense(c::DataType::kFloat32, {tokens, weights.rows});
  auto context = g::cuda::create_context().value();
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  auto packed = upload(memory.ref(), std::as_bytes(std::span(weights.bytes)));
  auto builder = g::GraphBuilder::create().value();
  const auto weight = builder.constant(table).value();
  const auto argument = builder.input(gather ? id : ad).value();
  g::ValueId indices;
  g::ValueId output;
  if (grouped) {
    indices = builder.input(id).value();
    output = builder.result(builder.add_node(c::model_ops::grouped_matmul(),
        std::array{weight, argument, indices}).value()).value();
  } else if (gather) {
    output = builder.result(builder.add_node(c::Operation(c::Gather{}), std::array{weight, argument}).value()).value();
  } else {
    output = builder.add(c::MatMul{}, argument, weight).value();
  }
  ASSERT_TRUE(builder.publish(output).ok());
  const std::array constants{g::ConstantBinding{weight,
      c::ConstTensorView::create(table, packed.view()).value()}};
  auto compiled = g::cuda::compile(std::move(builder).finish().value(), context,
      memory.ref(), {.constants = constants});
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->report.step_count(), 1U);
  EXPECT_EQ(compiled->report.step(0)->provider.id, c::ggml_cuda::kProviderId);
  std::vector<mem::Buffer> retained_inputs, retained_outputs;
  std::vector<std::vector<float>> retained_values;
  std::array queues{context.create_queue().value(), context.create_queue().value()};
  std::array calls{compiled->executable.create_invocation().value(),
      compiled->executable.create_invocation().value()};
  for (int round = 0; round < 2; ++round) {
    std::array<mem::Buffer, 2> activation, selected, outputs;
    std::array<std::vector<std::int32_t>, 2> ids;
    std::array<std::vector<float>, 2> expected;
    const auto bind = [&](std::size_t lane) {
      EXPECT_TRUE(calls[lane].bind(argument, c::ConstTensorView::create(
          gather ? id : ad, (gather ? selected[lane] : activation[lane]).view()).value()).ok());
      if (grouped) EXPECT_TRUE(calls[lane].bind(indices,
          c::ConstTensorView::create(id, selected[lane].view()).value()).ok());
      EXPECT_TRUE(calls[lane].bind(output,
          c::TensorView::create(yd, outputs[lane].view()).value()).ok());
      const auto status = g::cuda::prepare_replay(calls[lane], queues[lane],
          {.mode = g::ReplayMode::kRequireCapture, .host_wait = g::HostWaitPolicy::kReject});
      EXPECT_TRUE(status.ok()) << status.message();
    };
    for (std::size_t lane = 0; lane < calls.size(); ++lane) {
      SCOPED_TRACE(::testing::Message() << weights.columns << '/' << tokens << '/' << round << '/' << lane);
      if (grouped || gather) {
        ids[lane].resize(static_cast<std::size_t>(grouped ? tokens * slots : 3));
        for (std::size_t i = 0; i < ids[lane].size(); ++i)
          // Lane zero keeps repeated experts; lane one is the distinct-ID
          // neighbour in the same MMQ domain. Both use real per-slot inputs.
          ids[lane][i] = static_cast<std::int32_t>((i / static_cast<std::size_t>(slots) + lane +
              static_cast<std::size_t>(round) + (grouped && lane == 1 ? i % static_cast<std::size_t>(slots) : 0)) %
              static_cast<std::size_t>(grouped ? weights.experts : weights.rows));
        selected[lane] = upload(memory.ref(), std::as_bytes(std::span(ids[lane])));
      }
      if (gather) {
        for (const auto row : ids[lane]) expected[lane].insert(expected[lane].end(),
            weights.decoded.begin() + row * weights.columns,
            weights.decoded.begin() + (row + 1) * weights.columns);
      } else {
        auto values = weights.activations(grouped ? tokens * slots : tokens);
        if ((lane + static_cast<std::size_t>(round)) % 2) for (auto &value : values) value = -value;
        expected[lane] = project(weights, values, grouped ? std::span(ids[lane]) : std::span<const std::int32_t>{});
        activation[lane] = upload(memory.ref(), std::as_bytes(std::span(values)));
      }
      outputs[lane] = mem::Buffer::create(memory.ref(), yd.required_bytes().value(), 256).value();
      bind(lane);
    }
    std::array<c::Completion, 2> work;
    for (std::size_t lane = 0; lane < calls.size(); ++lane) {
      auto completion = calls[lane].submit(queues[lane]);
      ASSERT_TRUE(completion) << completion.error().message();
      work[lane] = std::move(*completion);
    }
    ASSERT_TRUE(work[1].wait().ok()); ASSERT_TRUE(work[0].wait().ok()); work = {};
    for (std::size_t lane = 0; lane < calls.size(); ++lane) {
      EXPECT_EQ(half_test::read<float>(outputs[lane]), expected[lane]);
      if (grouped || gather) {
        EXPECT_EQ(half_test::read<std::int32_t>(selected[lane]), ids[lane]);
        for (const auto bad : {std::int32_t{-1}, static_cast<std::int32_t>(grouped ? weights.experts : weights.rows)}) {
          auto invalid = ids[lane]; invalid.back() = bad;
          ASSERT_TRUE(selected[lane].view().write(std::as_bytes(std::span(invalid))).ok());
          auto failure = calls[lane].submit(queues[lane]); ASSERT_TRUE(failure);
          EXPECT_TRUE(failure->wait().IsOutOfRange()); *failure = {};
          EXPECT_EQ(half_test::read<float>(outputs[lane]), expected[lane]);
          EXPECT_EQ(half_test::read<std::int32_t>(selected[lane]), invalid);
          ASSERT_TRUE(selected[lane].view().write(std::as_bytes(std::span(ids[lane]))).ok());
          EXPECT_FALSE(calls[lane].reset().ok());
          calls[lane] = compiled->executable.create_invocation().value();
          bind(lane);
          auto recovered = calls[lane].submit(queues[lane]); ASSERT_TRUE(recovered);
          ASSERT_TRUE(recovered->wait().ok()); *recovered = {};
          EXPECT_EQ(half_test::read<float>(outputs[lane]), expected[lane]);
        }
      }
      retained_values.push_back(expected[lane]); retained_outputs.push_back(std::move(outputs[lane]));
      if (activation[lane].valid()) retained_inputs.push_back(std::move(activation[lane]));
      if (selected[lane].valid()) retained_inputs.push_back(std::move(selected[lane]));
    }
    for (std::size_t i = 0; i < retained_outputs.size(); ++i)
      EXPECT_EQ(half_test::read<float>(retained_outputs[i]), retained_values[i]);
  }
}

TEST(CudaNvfp4Graph, MatMulKeeps64BlockSubgroupsWithoutDenseWeightBank) {
  for (const auto k : {64, 128, 320})
    for (const auto tokens : {1, 9}) exercise(Weights(5, k), tokens, false, false);
}
TEST(CudaNvfp4Graph, GroupedExpertsKeepRepeatedReboundAndCheckedIds) {
  for (const auto tokens : {1, 9}) exercise(Weights(5, 128, 3), tokens, true, false);
  // S=3 dispatches the native generic ID sorter rather than its S=2 template.
  exercise(Weights(5, 128, 3), 9, true, false, 3);
  broadcast_native(2);
  broadcast_native(3);
}
TEST(CudaNvfp4Graph, SelectedRowsDecode64BlockWidthsAndRejectInvalidPublication) {
  for (const auto k : {64, 128, 320}) exercise(Weights(5, k), 1, false, true);
}
} // namespace
