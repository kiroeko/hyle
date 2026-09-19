#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <span>
#include <vector>

#include "../compute_provider_native/half_test_support.h"
#include "../compute/workload_fixture.h"
#include "hyle/graph/cpu.h"
#include "hyle/graph/providers/onednn.h"
#include "hyle/mem/stats.h"

namespace {
namespace compute = hyle::compute;
namespace graph = hyle::graph;

#if defined(HYLE_GRAPH_ONEDNN)
void check_norm_post_op_graph(std::int64_t rows, bool rms, bool scalar,
                             bool multiply_residual) {
  constexpr std::int64_t width = 31;
  const auto description =
      compute::TensorDesc::dense(compute::DataType::kFloat32,
                                 std::array{rows, width})
          .value();
  const auto row_description =
      compute::TensorDesc::dense(compute::DataType::kFloat32,
                                 std::array<std::int64_t, 1>{scalar ? 1 : width})
          .value();
  std::vector<float> values(rows * width), scales(scalar ? 1 : width);
  for (std::size_t index = 0; index < values.size(); ++index)
    values[index] =
        static_cast<float>(static_cast<int>(index % 29) - 14) / 16.F;
  for (std::size_t index = 0; index < scales.size(); ++index)
    scales[index] = 0.5F + static_cast<float>(index % 3) / 8.F;
  for (const bool published : {false, true}) {
    auto stats =
        hyle::mem::StatsAllocator::create(hyle::mem::system_allocator())
            .value();
    {
      auto context =
          graph::cpu::create_context({.metadata_allocator = stats.get()})
              .value();
      auto builder = graph::GraphBuilder::create().value();
      const auto input = builder.input(description).value();
      const auto scale = builder.input(row_description).value();
      const auto normalized =
          builder.add(rms ? compute::Operation(compute::RmsNorm{})
                          : compute::Operation(compute::LayerNorm{}),
                      input)
              .value();
      const auto weighted =
          builder
              .add(compute::Binary{compute::BinaryKind::kMultiply}, scale,
                   normalized)
              .value();
      // A full multirow operand belongs after the optimized region. Single
      // rows can retain the whole chain in the original vendor kernel.
      const auto residual =
          builder
              .add(compute::Binary{multiply_residual
                                      ? compute::BinaryKind::kMultiply
                                      : compute::BinaryKind::kAdd},
                   weighted, input)
              .value();
      if (published) {
        ASSERT_TRUE(builder.publish(normalized).ok());
      }
      ASSERT_TRUE(builder.publish(residual).ok());
      auto logical = std::move(builder).finish().value();
      auto workload = workload_test::known(rows == 1
                                              ? compute::WorkloadPhase::kDecode
                                              : compute::WorkloadPhase::kPrefill);
      auto compiled = graph::compile(logical, context, {.workload = workload});
      ASSERT_TRUE(compiled) << compiled.error().message();
      EXPECT_EQ(compiled->executable.workload(), workload);
      workload = {};
      EXPECT_TRUE(
          compiled->executable.workload().model.matches(workload_test::kModel));
      EXPECT_EQ(compiled->report.region_count(), published ? 0U : 1U);
      if (!published) {
        ASSERT_EQ(compiled->report.step_count(), rows == 1 ? 1U : 2U);
        auto prefix = compiled->report.step(0);
        ASSERT_TRUE(prefix);
        EXPECT_EQ(prefix->provider.id, graph::onednn::kProviderId);
        EXPECT_EQ(prefix->node_count, rows == 1 ? 3U : 2U);
        EXPECT_EQ(compiled->report.mixed_providers(), rows != 1);
      }
      auto source = half_test::buffer<float>(values);
      auto weights = half_test::buffer<float>(scales);
      const auto source_view =
          compute::ConstTensorView::create(description, source.view()).value();
      const auto scale_view =
          compute::ConstTensorView::create(row_description, weights.view())
              .value();
      std::array<hyle::mem::Buffer, 2> outputs{
          hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                    values.size() * sizeof(float), 64)
              .value(),
          hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                    values.size() * sizeof(float), 64)
              .value()};
      auto intermediate =
          hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                    values.size() * sizeof(float), 64)
              .value();
      auto queue = context.create_queue().value();
      std::array invocations{compiled->executable.create_invocation().value(),
                             compiled->executable.create_invocation().value()};
      std::vector<float> first;
      for (std::size_t run = 0; run < 2; ++run) {
        auto &invocation = invocations[run];
        ASSERT_TRUE(invocation.bind(input, source_view).ok());
        ASSERT_TRUE(invocation.bind(scale, scale_view).ok());
        ASSERT_TRUE(invocation
                        .bind(residual, compute::TensorView::create(
                                            description, outputs[run].view())
                                            .value())
                        .ok());
        if (published) {
          ASSERT_TRUE(invocation
                          .bind(normalized,
                                compute::TensorView::create(description,
                                                            intermediate.view())
                                    .value())
                          .ok());
        }
        std::vector<float> current = values;
        if (run != 0)
          for (auto &value : current)
            value += 0.25F;
        ASSERT_TRUE(
            source.view().write(std::as_bytes(std::span(current))).ok());
        auto submitted = invocation.submit(queue);
        ASSERT_TRUE(submitted) << submitted.error().message();
        ASSERT_TRUE(submitted->wait().ok());
        std::vector<float> actual(values.size());
        ASSERT_TRUE(outputs[run]
                        .view()
                        .read(std::as_writable_bytes(std::span(actual)))
                        .ok());
        for (std::int64_t row = 0; row < rows; ++row) {
          double mean = 0;
          if (!rms) {
            for (std::int64_t col = 0; col < width; ++col)
              mean += current[static_cast<std::size_t>(row * width + col)];
            mean /= width;
          }
          double squares = 0;
          for (std::int64_t col = 0; col < width; ++col) {
            const double value =
                current[static_cast<std::size_t>(row * width + col)] - mean;
            squares += value * value;
          }
          for (std::int64_t col = 0; col < width; ++col) {
            const auto index = static_cast<std::size_t>(row * width + col);
            const double weighted_value =
                (current[index] - mean) / std::sqrt(squares / width + 1.e-5) *
                scales[scalar ? 0 : static_cast<std::size_t>(col)];
            const double expected = multiply_residual
                                        ? weighted_value * current[index]
                                        : weighted_value + current[index];
            EXPECT_NEAR(actual[index], expected, 4.e-5) << index;
          }
        }
        if (run == 0)
          first = actual;
        else {
          ASSERT_TRUE(outputs[0]
                          .view()
                          .read(std::as_writable_bytes(std::span(actual)))
                          .ok());
          EXPECT_EQ(first, actual);
        }
      }
      ASSERT_TRUE(context.close().ok());
      EXPECT_FALSE(invocations[0].submit(queue));
    }
    EXPECT_TRUE(stats.balanced());
  }
}

TEST(CpuDefaultGraph, MatureNormPostOpsFuseAndPreserveObservableBoundaries) {
  for (const std::int64_t rows : {1, 17})
    for (const bool rms : {false, true})
      for (const bool scalar : {false, true})
        for (const bool multiply_residual : {false, true})
          check_norm_post_op_graph(rows, rms, scalar, multiply_residual);
}
#endif

TEST(CpuDefaultGraph, NonModelGraphReplaysAndReclaimsItsOwners) {
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  {
    auto context =
        graph::cpu::create_context({.metadata_allocator = stats.get()}).value();
    auto queue = context.create_queue().value();
    const auto description =
        compute::TensorDesc::dense(compute::DataType::kFloat32,
                                   std::array<std::int64_t, 1>{5})
            .value();
    auto builder = graph::GraphBuilder::create().value();
    const auto input = builder.input(description).value();
    const auto negative =
        builder.add(compute::Unary{compute::UnaryKind::kNeg}, input).value();
    const auto absolute =
        builder.add(compute::Unary{compute::UnaryKind::kAbs}, negative).value();
    ASSERT_TRUE(builder.publish(absolute).ok());
    auto logical = std::move(builder).finish().value();
    auto compiled = graph::compile(logical, context).value();
    EXPECT_EQ(compiled.report.node_count(), 2U);
    EXPECT_EQ(compiled.report.region_count(), 0U);
    auto source =
        hyle::mem::Buffer::create(hyle::mem::system_resource(), 20, 64).value();
    auto destination =
        hyle::mem::Buffer::create(hyle::mem::system_resource(), 20, 64).value();
    auto invocation = compiled.executable.create_invocation().value();
    ASSERT_TRUE(invocation
                    .bind(input, compute::ConstTensorView::create(description,
                                                                  source.view())
                                     .value())
                    .ok());
    ASSERT_TRUE(invocation
                    .bind(absolute, compute::TensorView::create(
                                        description, destination.view())
                                        .value())
                    .ok());
    const std::array queues{graph::QueueBinding{
        .lane = graph::ExecutionLaneId{0}, .queue = &queue}};
    ASSERT_TRUE(invocation.prepare_replay(queues).ok());
    const auto before = stats.snapshot();
    for (float seed : {1.0F, 3.0F, 7.0F}) {
      const std::array values{-seed, 0.0F, seed, -2 * seed, 3 * seed};
      ASSERT_TRUE(source.view().write(std::as_bytes(std::span(values))).ok());
      auto completion = invocation.submit(queue);
      ASSERT_TRUE(completion.has_value()) << completion.error().message();
      ASSERT_TRUE(completion->wait().ok());
      std::array<float, 5> actual{};
      ASSERT_TRUE(destination.view()
                      .read(std::as_writable_bytes(std::span(actual)))
                      .ok());
      EXPECT_EQ(actual, (std::array{seed, 0.0F, seed, 2 * seed, 3 * seed}));
    }
    EXPECT_EQ(stats.snapshot().total_allocs, before.total_allocs);
    EXPECT_EQ(stats.snapshot().reallocs, before.reallocs);
    ASSERT_TRUE(context.close().ok());
    auto closed = invocation.submit(queue);
    ASSERT_FALSE(closed.has_value());
    EXPECT_TRUE(closed.error().IsFailedPrecondition());
  }
  EXPECT_TRUE(stats.balanced());
}

template <class Value> void expect_half_state_graph() {
  constexpr std::size_t hidden = 2048, vocabulary = 7, tokens = 2, slots = 4;
  const auto type = half_test::data_type<Value>();
  const auto table_desc =
      compute::TensorDesc::dense(
          type, std::array<std::int64_t, 2>{vocabulary, hidden})
          .value();
  const auto state_desc = compute::TensorDesc::dense(
                              type, std::array<std::int64_t, 2>{slots, hidden})
                              .value();
  const auto token_desc = compute::TensorDesc::dense(
                              type, std::array<std::int64_t, 2>{tokens, hidden})
                              .value();
  const auto index_desc =
      compute::TensorDesc::dense(compute::DataType::kInt32,
                                 std::array<std::int64_t, 1>{tokens})
          .value();
  const auto scale_desc =
      compute::TensorDesc::dense(compute::DataType::kFloat32,
                                 std::array<std::int64_t, 1>{hidden})
          .value();
  std::vector<Value> table(vocabulary * hidden);
  std::vector<float> scale(hidden);
  for (std::size_t i = 0; i < table.size(); ++i)
    table[i] = half_test::encode<Value>((static_cast<int>(i % 37) - 18) / 32.0);
  for (std::size_t i = 0; i < hidden; ++i)
    scale[i] = 1.0F + static_cast<float>(static_cast<int>(i % 13) - 6) / 32.0F;
  auto table_buffer = half_test::buffer<Value>(table);
  auto scale_buffer = half_test::buffer<float>(scale);
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  {
    auto context =
        graph::cpu::create_context({.metadata_allocator = stats.get()}).value();
    auto queue = context.create_queue().value();
    auto builder = graph::GraphBuilder::create().value();
    const auto table_id = builder.constant(table_desc).value();
    const auto scale_id = builder.constant(scale_desc).value();
    const auto token_ids = builder.input(index_desc).value();
    const auto positions = builder.input(index_desc).value();
    const auto old_state = builder.input(state_desc).value();
    const auto embedded =
        builder.add(compute::Gather{}, table_id, token_ids).value();
    const auto normalized =
        builder.add(compute::RmsNorm{.epsilon = 1e-6F}, embedded, scale_id)
            .value();
    const auto gated =
        builder
            .add(
                compute::GatedActivation{
                    .inputs = compute::GatedInputKind::kSeparate},
                normalized, embedded)
            .value();
    const auto residual =
        builder.add(compute::Binary{compute::BinaryKind::kAdd}, embedded, gated)
            .value();
    const auto state =
        builder
            .add(compute::Scatter{.kind = compute::ScatterKind::kAdd},
                 old_state, positions, residual)
            .value();
    const auto probabilities =
        builder.add(compute::Softmax{}, residual).value();
    ASSERT_TRUE(builder.publish(state).ok());
    ASSERT_TRUE(builder.publish(probabilities).ok());
    auto logical = std::move(builder).finish().value();
    const std::array constants{
        graph::ConstantBinding{table_id, compute::ConstTensorView::create(
                                             table_desc, table_buffer.view())
                                             .value()},
        graph::ConstantBinding{scale_id, compute::ConstTensorView::create(
                                             scale_desc, scale_buffer.view())
                                             .value()}};
    auto result = graph::compile(logical, context, {.constants = constants});
    ASSERT_TRUE(result.has_value()) << result.error().message();
    auto compiled = std::move(*result);
    const auto allocate = [](const compute::TensorDesc &desc) {
      return hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                       desc.required_bytes().value(), 64)
          .value();
    };
    std::array invocations{compiled.executable.create_invocation().value(),
                           compiled.executable.create_invocation().value()};
    std::array token_buffers{allocate(index_desc), allocate(index_desc)};
    std::array position_buffers{allocate(index_desc), allocate(index_desc)};
    std::array states{allocate(state_desc), allocate(state_desc)};
    std::array outputs{allocate(state_desc), allocate(state_desc)};
    std::array probability_buffers{allocate(token_desc), allocate(token_desc)};
    const std::array queues{graph::QueueBinding{
        .lane = graph::ExecutionLaneId{0}, .queue = &queue}};
    for (std::size_t run = 0; run < invocations.size(); ++run) {
      auto &invocation = invocations[run];
      ASSERT_TRUE(
          invocation
              .bind(token_ids, compute::ConstTensorView::create(
                                   index_desc, token_buffers[run].view())
                                   .value())
              .ok());
      ASSERT_TRUE(
          invocation
              .bind(positions, compute::ConstTensorView::create(
                                   index_desc, position_buffers[run].view())
                                   .value())
              .ok());
      ASSERT_TRUE(invocation
                      .bind(old_state, compute::ConstTensorView::create(
                                           state_desc, states[run].view())
                                           .value())
                      .ok());
      ASSERT_TRUE(invocation
                      .bind(state, compute::TensorView::create(
                                       state_desc, outputs[run].view())
                                       .value())
                      .ok());
      ASSERT_TRUE(invocation
                      .bind(probabilities,
                            compute::TensorView::create(
                                token_desc, probability_buffers[run].view())
                                .value())
                      .ok());
      ASSERT_TRUE(invocation.prepare_replay(queues).ok());
    }
    const auto before = stats.snapshot();
    std::array<std::vector<Value>, 2> retained;
    for (std::size_t step = 0; step < 3; ++step) {
      for (std::size_t run = 0; run < invocations.size(); ++run) {
        SCOPED_TRACE(step);
        SCOPED_TRACE(run);
        const std::array indices{
            static_cast<std::int32_t>((step + run) % vocabulary),
            static_cast<std::int32_t>((step + run + 3) % vocabulary)};
        const std::array destinations{
            static_cast<std::int32_t>((step + run) % slots),
            static_cast<std::int32_t>((step + run + 2) % slots)};
        std::vector<Value> initial(slots * hidden,
                                   half_test::encode<Value>((run + 1) / 8.0));
        ASSERT_TRUE(
            states[run].view().write(std::as_bytes(std::span(initial))).ok());
        ASSERT_TRUE(token_buffers[run]
                        .view()
                        .write(std::as_bytes(std::span(indices)))
                        .ok());
        ASSERT_TRUE(position_buffers[run]
                        .view()
                        .write(std::as_bytes(std::span(destinations)))
                        .ok());
        auto done = invocations[run].submit(queue);
        ASSERT_TRUE(done.has_value()) << done.error().message();
        ASSERT_TRUE(done->wait().ok());
        const auto actual = half_test::read<Value>(outputs[run]);
        const auto probability =
            half_test::read<Value>(probability_buffers[run]);
        auto expected = initial;
        for (std::size_t token = 0; token < tokens; ++token) {
          const auto base = static_cast<std::size_t>(indices[token]) * hidden;
          double sum = 0;
          for (std::size_t col = 0; col < hidden; ++col) {
            const double value = half_test::decode(table[base + col]);
            sum += value * value;
          }
          const double inverse = 1.0 / std::sqrt(sum / hidden + 1e-6);
          std::vector<double> row(hidden);
          double denominator = 0;
          for (std::size_t col = 0; col < hidden; ++col) {
            const double value = half_test::decode(table[base + col]);
            const double norm = half_test::decode(
                half_test::encode<Value>(value * inverse * scale[col]));
            const double gate = half_test::decode(half_test::encode<Value>(
                norm / (1.0 + std::exp(-norm)) * value));
            row[col] =
                half_test::decode(half_test::encode<Value>(value + gate));
            const auto destination =
                static_cast<std::size_t>(destinations[token]) * hidden + col;
            expected[destination] = half_test::encode<Value>(
                half_test::decode(initial[destination]) + row[col]);
            denominator += std::exp(row[col]);
          }
          for (std::size_t col = 0; col < hidden; ++col)
            half_test::expect_near(probability[token * hidden + col],
                                   std::exp(row[col]) / denominator);
        }
        for (std::size_t i = 0; i < actual.size(); ++i)
          half_test::expect_near(actual[i], half_test::decode(expected[i]));
        const auto unchanged = half_test::read<Value>(states[run]);
        for (std::size_t i = 0; i < unchanged.size(); ++i)
          EXPECT_EQ(unchanged[i].bits, initial[i].bits);
        retained[run] = actual;
        const auto other = 1 - run;
        if (!retained[other].empty()) {
          const auto previous = half_test::read<Value>(outputs[other]);
          for (std::size_t i = 0; i < previous.size(); ++i)
            EXPECT_EQ(previous[i].bits, retained[other][i].bits);
        }
      }
    }
    EXPECT_EQ(stats.snapshot().total_allocs, before.total_allocs);
    EXPECT_EQ(stats.snapshot().reallocs, before.reallocs);
    ASSERT_TRUE(context.close().ok());
    for (auto &invocation : invocations) {
      auto closed = invocation.submit(queue);
      ASSERT_FALSE(closed.has_value());
      EXPECT_TRUE(closed.error().IsFailedPrecondition());
    }
  }
  EXPECT_TRUE(stats.balanced());
}

TEST(CpuDefaultGraph, HalfTensorFragmentHasIndependentReplayState) {
  expect_half_state_graph<hyle::simd::float16_t>();
  expect_half_state_graph<hyle::simd::bfloat16_t>();
}

TEST(CpuDefaultGraph, ExpertCountsBecomeGatherOffsetsWithoutRecompilation) {
  auto context = graph::cpu::create_context().value();
  auto queue = context.create_queue().value();
  const auto counts_desc =
      compute::TensorDesc::dense(compute::DataType::kInt32,
                                 std::array<std::int64_t, 1>{5})
          .value();
  const auto data_desc =
      compute::TensorDesc::dense(compute::DataType::kFloat32,
                                 std::array<std::int64_t, 1>{6})
          .value();
  const auto output_desc =
      compute::TensorDesc::dense(compute::DataType::kFloat32,
                                 std::array<std::int64_t, 1>{5})
          .value();
  const std::array<float, 6> data{10, 20, 30, 40, 50, 60};
  const std::array<std::array<std::int32_t, 5>, 2> counts{
      {{0, 3, 0, 2, 1}, {1, 0, 1, 0, 4}}};
  const std::array<std::array<float, 5>, 2> expected{
      {{10, 10, 40, 40, 60}, {10, 20, 20, 30, 30}}};
  auto data_buffer = half_test::buffer<float>(data);
  auto counts_buffer = half_test::buffer<std::int32_t>(counts[0]);
  auto output_buffer = half_test::buffer<float>(expected[0]);
  auto builder = graph::GraphBuilder::create().value();
  const auto input = builder.input(counts_desc).value();
  const auto table = builder.constant(data_desc).value();
  const auto offsets =
      builder.add(compute::Scan{.exclusive = true}, input).value();
  const auto gathered = builder.add(compute::Gather{}, table, offsets).value();
  ASSERT_TRUE(builder.publish(gathered).ok());
  auto logical = std::move(builder).finish().value();
  const std::array constants{graph::ConstantBinding{
      table,
      compute::ConstTensorView::create(data_desc, data_buffer.view()).value()}};
  auto compiled = graph::compile(logical, context, {.constants = constants});
  ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
  auto invocation = compiled->executable.create_invocation().value();
  ASSERT_TRUE(invocation
                  .bind(input, compute::ConstTensorView::create(
                                   counts_desc, counts_buffer.view())
                                   .value())
                  .ok());
  ASSERT_TRUE(invocation
                  .bind(gathered, compute::TensorView::create(
                                      output_desc, output_buffer.view())
                                      .value())
                  .ok());
  const std::array queues{
      graph::QueueBinding{.lane = graph::ExecutionLaneId{0}, .queue = &queue}};
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  for (std::size_t run = 0; run < counts.size(); ++run) {
    ASSERT_TRUE(
        counts_buffer.view().write(std::as_bytes(std::span(counts[run]))).ok());
    auto done = invocation.submit(queue);
    ASSERT_TRUE(done.has_value()) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    EXPECT_EQ(half_test::read<float>(output_buffer),
              std::vector<float>(expected[run].begin(), expected[run].end()));
  }
}

} // namespace
