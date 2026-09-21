#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

#include "../compute/workload_fixture.h"
#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/providers/ggml_cpu.h"
#include "hyle/compute/providers/onednn.h"
#include "hyle/graph/cpu.h"
#include "hyle/graph/providers/onednn.h"
#include "hyle/mem/stats.h"

namespace {
namespace compute = hyle::compute;
namespace graph = hyle::graph;

#if defined(HYLE_GRAPH_ONEDNN) || defined(HYLE_COMPUTE_GGML_CPU)
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
      // ggml can retain a full multirow residual and published intermediate.
      // oneDNN post-ops retain their narrower leaf/region eligibility.
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
#if defined(HYLE_COMPUTE_GGML_CPU)
      EXPECT_EQ(compiled->report.region_count(), 1U);
#else
      EXPECT_EQ(compiled->report.region_count(), published ? 0U : 1U);
#endif
      if (!published) {
#if defined(HYLE_COMPUTE_GGML_CPU)
        ASSERT_EQ(compiled->report.step_count(), 1U);
        auto prefix = compiled->report.step(0);
        ASSERT_TRUE(prefix);
        EXPECT_EQ(prefix->provider.id, compute::ggml_cpu::kProviderId);
        EXPECT_EQ(prefix->node_count, 3U);
        EXPECT_FALSE(compiled->report.mixed_providers());
#else
        ASSERT_EQ(compiled->report.step_count(), rows == 1 ? 1U : 2U);
        auto prefix = compiled->report.step(0);
        ASSERT_TRUE(prefix);
        EXPECT_EQ(prefix->provider.id, graph::onednn::kProviderId);
        EXPECT_EQ(prefix->node_count, rows == 1 ? 3U : 2U);
        EXPECT_EQ(compiled->report.mixed_providers(), rows != 1);
#endif
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
        std::vector<float> published_values;
        if (published) {
          published_values.resize(values.size());
          ASSERT_TRUE(intermediate.view()
                          .read(std::as_writable_bytes(std::span(published_values)))
                          .ok());
        }
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
            if (published)
              EXPECT_NEAR(published_values[index],
                          (current[index] - mean) / std::sqrt(squares / width + 1.e-5),
                          4.e-5) << index;
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
#if defined(HYLE_COMPUTE_GGML_CPU)
    EXPECT_EQ(compiled.report.region_count(), 1U);
    EXPECT_EQ(compiled.report.step(0)->provider.id, compute::ggml_cpu::kProviderId);
#else
    EXPECT_EQ(compiled.report.region_count(), 0U);
#endif
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
#if defined(HYLE_COMPUTE_GGML_CPU)
  ASSERT_EQ(compiled->report.step_count(), 2U);
  for (std::size_t step = 0; step < 2; ++step)
    EXPECT_EQ(compiled->report.step(step)->provider.id, compute::ggml_cpu::kProviderId);
#endif
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

#if defined(HYLE_COMPUTE_GGML_CPU)
TEST(CpuDefaultGraph, SelectedGroupMeansBroadcastWithinOneMatureRegion) {
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  {
    auto context =
        graph::cpu::create_context({.metadata_allocator = stats.get()}).value();
    auto queue = context.create_queue().value();
    const auto description = compute::TensorDesc::dense(
                                 compute::DataType::kFloat32,
                                 std::array<std::int64_t, 6>{2, 3, 2, 2, 2, 5})
                                 .value();
    auto builder = graph::GraphBuilder::create().value();
    const auto input = builder.input(description).value();
    const compute::Reduce reduce{
        .kind = compute::ReductionKind::kMean,
        .axes =
            compute::Axes::from(std::array<std::int32_t, 3>{4, -6, 2}).value(),
        .keep_dimensions = true,
        .accumulation_type = compute::DataType::kFloat32};
    const auto mean = builder.add(reduce, input).value();
    const auto centered =
        builder
            .add(compute::Binary{compute::BinaryKind::kSubtract}, input, mean)
            .value();
    const auto squared =
        builder
            .add(compute::Binary{compute::BinaryKind::kMultiply}, centered,
                 centered)
            .value();
    const auto variance = builder.add(reduce, squared).value();
    ASSERT_TRUE(builder.publish(mean).ok());
    ASSERT_TRUE(builder.publish(variance).ok());
    auto logical = std::move(builder).finish().value();
    const auto output_desc = logical.value_description(mean).value();
    auto compiled = graph::compile(logical, context);
    ASSERT_TRUE(compiled) << compiled.error().message();
    ASSERT_EQ(compiled->report.region_count(), 1U);
    ASSERT_EQ(compiled->report.step_count(), 1U);
    EXPECT_EQ(compiled->report.step(0)->node_count, 4U);
    EXPECT_EQ(compiled->report.step(0)->provider.id,
              compute::ggml_cpu::kProviderId);
    std::vector<float> values(240);
    auto source = half_test::buffer<float>(values);
    std::array means{half_test::buffer<float>(std::vector<float>(30)),
                     half_test::buffer<float>(std::vector<float>(30))};
    std::array variances{half_test::buffer<float>(std::vector<float>(30)),
                         half_test::buffer<float>(std::vector<float>(30))};
    std::array invocations{compiled->executable.create_invocation().value(),
                           compiled->executable.create_invocation().value()};
    const std::array queues{graph::QueueBinding{
        .lane = graph::ExecutionLaneId{0}, .queue = &queue}};
    for (std::size_t slot = 0; slot < invocations.size(); ++slot) {
      ASSERT_TRUE(invocations[slot]
                      .bind(input, compute::ConstTensorView::create(
                                       description, source.view())
                                       .value())
                      .ok());
      ASSERT_TRUE(invocations[slot]
                      .bind(mean, compute::TensorView::create(
                                      output_desc, means[slot].view())
                                      .value())
                      .ok());
      ASSERT_TRUE(invocations[slot]
                      .bind(variance, compute::TensorView::create(
                                          output_desc, variances[slot].view())
                                          .value())
                      .ok());
      ASSERT_TRUE(invocations[slot].prepare_replay(queues).ok());
    }
    const auto allocations = stats.snapshot().total_allocs;
    std::array<std::vector<float>, 2> previous;
    const auto group = [](std::size_t index) {
      return ((index / 40) % 3) * 10 + ((index / 10) % 2) * 5 + index % 5;
    };
    for (std::size_t run = 0; run < 4; ++run) {
      const auto slot = run % 2;
      std::array<double, 30> expected_mean{}, expected_variance{};
      for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] =
            static_cast<float>(static_cast<int>((i + run * 5) % 31) - 15) / 16;
        expected_mean[group(i)] += values[i] / 8.0;
      }
      for (std::size_t i = 0; i < values.size(); ++i) {
        const auto difference =
            static_cast<double>(values[i]) - expected_mean[group(i)];
        expected_variance[group(i)] += difference * difference / 8.0;
      }
      ASSERT_TRUE(source.view().write(std::as_bytes(std::span(values))).ok());
      auto done = invocations[slot].submit(queue);
      ASSERT_TRUE(done) << done.error().message();
      ASSERT_TRUE(done->wait().ok());
      const auto actual_mean = half_test::read<float>(means[slot]);
      const auto actual_variance = half_test::read<float>(variances[slot]);
      for (std::size_t i = 0; i < expected_mean.size(); ++i) {
        EXPECT_NEAR(actual_mean[i], expected_mean[i], 2.e-6);
        EXPECT_NEAR(actual_variance[i], expected_variance[i], 2.e-6);
      }
      previous[slot] = actual_variance;
      if (run != 0)
        EXPECT_EQ(half_test::read<float>(variances[1 - slot]),
                  previous[1 - slot]);
    }
    EXPECT_EQ(stats.snapshot().total_allocs, allocations);
  }
  EXPECT_TRUE(stats.balanced());
}

template <class Value> void check_l2_region() {
  for (const std::int64_t width : {1, 7, 31}) {
    auto stats =
        hyle::mem::StatsAllocator::create(hyle::mem::system_allocator())
            .value();
    {
      auto context =
          graph::cpu::create_context({.metadata_allocator = stats.get()})
              .value();
      auto queue = context.create_queue().value();
      const auto description =
          compute::TensorDesc::dense(
              half_test::data_type<Value>(),
              std::array<std::int64_t, 6>{2, 1, 3, 1, 2, width})
              .value();
      const auto scalar =
          compute::TensorDesc::dense(compute::DataType::kFloat32,
                                     std::span<const std::int64_t>{})
              .value();
      constexpr float epsilon = 0.0001F;
      auto epsilon_buffer = half_test::buffer<float>(std::array{epsilon});
      auto builder = graph::GraphBuilder::create().value();
      const auto input = builder.input(description).value();
      const auto eps = builder.constant(scalar).value();
      auto fp32 = input;
      if constexpr (!std::is_same_v<Value, float>)
        fp32 = builder.add(compute::Cast{compute::DataType::kFloat32}, input)
                   .value();
      const auto squares =
          builder
              .add(compute::Binary{compute::BinaryKind::kMultiply}, fp32, fp32)
              .value();
      const auto sum =
          builder
              .add(compute::Reduce{.kind = compute::ReductionKind::kSum,
                                   .axes = compute::Axes::from(
                                               std::array<std::int32_t, 1>{-1})
                                               .value(),
                                   .keep_dimensions = true,
                                   .accumulation_type =
                                       compute::DataType::kFloat32},
                   squares)
              .value();
      const auto adjusted =
          builder.add(compute::Binary{compute::BinaryKind::kAdd}, sum, eps)
              .value();
      const auto length =
          builder.add(compute::Unary{compute::UnaryKind::kSqrt}, adjusted)
              .value();
      auto normalized =
          builder
              .add(compute::Binary{compute::BinaryKind::kDivide}, fp32, length)
              .value();
      if constexpr (!std::is_same_v<Value, float>)
        normalized =
            builder
                .add(compute::Cast{half_test::data_type<Value>()}, normalized)
                .value();
      ASSERT_TRUE(builder.publish(sum).ok());
      ASSERT_TRUE(builder.publish(normalized).ok());
      auto logical = std::move(builder).finish().value();
      const std::array constants{graph::ConstantBinding{
          eps, compute::ConstTensorView::create(scalar, epsilon_buffer.view())
                   .value()}};
      auto compiled =
          graph::compile(logical, context, {.constants = constants});
      ASSERT_TRUE(compiled) << compiled.error().message();
      ASSERT_EQ(compiled->report.region_count(), 1U);
      ASSERT_EQ(compiled->report.step_count(), 1U);
      EXPECT_EQ(compiled->report.step(0)->provider.id,
                compute::ggml_cpu::kProviderId);
      EXPECT_EQ(compiled->report.step(0)->node_count,
                (std::is_same_v<Value, float> ? 5U : 7U));
      const auto size = description.element_count().value();
      std::vector<Value> values(size);
      auto source = half_test::buffer<Value>(values);
      std::array outputs{half_test::buffer<Value>(values),
                         half_test::buffer<Value>(values)};
      std::array sums{half_test::buffer<float>(std::vector<float>(12)),
                      half_test::buffer<float>(std::vector<float>(12))};
      std::array invocations{compiled->executable.create_invocation().value(),
                             compiled->executable.create_invocation().value()};
      const std::array queues{graph::QueueBinding{
          .lane = graph::ExecutionLaneId{0}, .queue = &queue}};
      for (std::size_t slot = 0; slot < invocations.size(); ++slot) {
        ASSERT_TRUE(invocations[slot]
                        .bind(input, compute::ConstTensorView::create(
                                         description, source.view())
                                         .value())
                        .ok());
        ASSERT_TRUE(invocations[slot]
                        .bind(sum, compute::TensorView::create(
                                       logical.value_description(sum).value(),
                                       sums[slot].view())
                                       .value())
                        .ok());
        ASSERT_TRUE(invocations[slot]
                        .bind(normalized, compute::TensorView::create(
                                              description, outputs[slot].view())
                                              .value())
                        .ok());
        ASSERT_TRUE(invocations[slot].prepare_replay(queues).ok());
      }
      const auto allocations = stats.snapshot().total_allocs;
      std::array<std::vector<Value>, 2> previous;
      for (std::size_t run = 0; run < 4; ++run) {
        const auto slot = run % 2;
        for (std::size_t i = 0; i < size; ++i) {
          const auto row = i / static_cast<std::size_t>(width);
          const double value =
              row == 0   ? 0
              : row == 1 ? 0.00001 * static_cast<double>(run + 1)
                         : static_cast<double>(static_cast<int>(i % 17) - 8) *
                               static_cast<double>(run + 1) / 16;
          values[i] = half_test::encode<Value>(value);
        }
        ASSERT_TRUE(source.view().write(std::as_bytes(std::span(values))).ok());
        auto done = invocations[slot].submit(queue);
        ASSERT_TRUE(done) << done.error().message();
        ASSERT_TRUE(done->wait().ok());
        const auto actual = half_test::read<Value>(outputs[slot]);
        const auto actual_sums = half_test::read<float>(sums[slot]);
        for (std::size_t row = 0; row < 12; ++row) {
          double squared = 0;
          for (std::size_t col = 0; col < static_cast<std::size_t>(width);
               ++col) {
            const float value = static_cast<float>(half_test::decode(
                values[row * static_cast<std::size_t>(width) + col]));
            squared += static_cast<double>(value * value);
          }
          EXPECT_NEAR(actual_sums[row], squared,
                      1.e-6 + std::abs(squared) * 1.e-6);
          const float divisor =
              std::sqrt(static_cast<float>(squared) + epsilon);
          for (std::size_t col = 0; col < static_cast<std::size_t>(width);
               ++col) {
            const auto offset = row * static_cast<std::size_t>(width) + col;
            const float expected =
                static_cast<float>(half_test::decode(values[offset])) / divisor;
            if constexpr (std::is_same_v<Value, float>)
              EXPECT_NEAR(actual[offset], expected, 2.e-6F);
            else
              half_test::expect_near(actual[offset], expected);
          }
        }
        previous[slot] = actual;
        if (run != 0) {
          const auto old = half_test::read<Value>(outputs[1 - slot]);
          ASSERT_EQ(old.size(), previous[1 - slot].size());
          EXPECT_EQ(std::memcmp(old.data(), previous[1 - slot].data(),
                                old.size() * sizeof(Value)),
                    0);
        }
      }
      EXPECT_EQ(stats.snapshot().total_allocs, allocations);
    }
    EXPECT_TRUE(stats.balanced());
  }
}

TEST(CpuDefaultGraph,
     MatureL2RegionPreservesEpsilonRoundingAndIndependentReplay) {
  check_l2_region<float>();
  check_l2_region<hyle::simd::float16_t>();
  check_l2_region<hyle::simd::bfloat16_t>();
}

TEST(CpuDefaultGraph, StandardTopKPublishesBothOutputsAcrossReplay) {
  auto context = graph::cpu::create_context().value();
  auto queue = context.create_queue().value();
  const auto description = compute::TensorDesc::dense(compute::DataType::kFloat32,
      std::array<std::int64_t, 1>{7}).value();
  auto builder = graph::GraphBuilder::create().value();
  const auto input = builder.input(description).value();
  const std::array arguments{input};
  const auto node = builder.add_node(compute::Operation(compute::TopK{.count = 3}), arguments).value();
  const auto values = builder.result(node, 0).value();
  const auto indices = builder.result(node, 1).value();
  ASSERT_TRUE(builder.publish(values).ok());
  ASSERT_TRUE(builder.publish(indices).ok());
  auto logical = std::move(builder).finish().value();
  auto compiled = graph::compile(logical, context).value();
  ASSERT_EQ(compiled.report.step_count(), 1U);
  EXPECT_EQ(compiled.report.step(0)->provider.id, compute::ggml_cpu::kProviderId);
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const std::array<std::array<float, 7>, 2> inputs{{
      {7, 3, 7, nan, -1, 9, 9}, {7, 9, 7, nan, -1, 3, 9}}};
  const std::array<std::array<std::int32_t, 3>, 2> expected{{{5, 6, 0}, {1, 6, 0}}};
  auto source = half_test::buffer<float>(inputs[0]);
  auto value_buffer = half_test::buffer<float>(std::array<float, 3>{});
  auto index_buffer = half_test::buffer<std::int32_t>(std::array<std::int32_t, 3>{});
  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(input, compute::ConstTensorView::create(description, source.view()).value()).ok());
  ASSERT_TRUE(invocation.bind(values, compute::TensorView::create(logical.value_description(values).value(), value_buffer.view()).value()).ok());
  ASSERT_TRUE(invocation.bind(indices, compute::TensorView::create(logical.value_description(indices).value(), index_buffer.view()).value()).ok());
  const std::array queues{graph::QueueBinding{.lane = graph::ExecutionLaneId{0}, .queue = &queue}};
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  for (std::size_t run = 0; run < inputs.size(); ++run) {
    ASSERT_TRUE(source.view().write(std::as_bytes(std::span(inputs[run]))).ok());
    auto done = invocation.submit(queue);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    EXPECT_EQ(half_test::read<std::int32_t>(index_buffer),
              std::vector<std::int32_t>(expected[run].begin(), expected[run].end()));
    EXPECT_EQ(half_test::read<float>(value_buffer), (std::vector<float>{9, 9, 7}));
  }
}
#endif
} // namespace

#if defined(HYLE_COMPUTE_ONEDNN)
namespace {
TEST(CpuGraph, StableSoftplusAndSelectKeepTwoPublishedResultsAcrossReplay) {
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  {
    auto context =
        graph::cpu::create_context({.metadata_allocator = stats.get()}).value();
    auto queue = context.create_queue().value();
    const auto shape = std::array<std::int64_t, 2>{2, 4};
    const auto data =
        compute::TensorDesc::dense(compute::DataType::kFloat32, shape).value();
    const auto condition =
        compute::TensorDesc::dense(compute::DataType::kBool, shape).value();
    auto builder = graph::GraphBuilder::create().value();
    const auto input = builder.input(data).value();
    const auto mask = builder.input(condition).value();
    const auto stable =
        builder.add(compute::Unary{compute::UnaryKind::kSoftplus}, input)
            .value();
    const auto selected =
        builder.add(compute::Select{}, mask, stable, input).value();
    ASSERT_TRUE(builder.publish(stable).ok());
    ASSERT_TRUE(builder.publish(selected).ok());
    auto logical = std::move(builder).finish().value();
    auto compiled = graph::compile(logical, context);
    ASSERT_TRUE(compiled) << compiled.error().message();
    ASSERT_EQ(compiled->report.step_count(), 2U);
    for (std::size_t i = 0; i < 2; ++i)
      EXPECT_EQ(compiled->report.step(i)->provider.id,
                compute::onednn::kProviderId);
    std::array<float, 8> values{-90, -40, -20, -1, -0.F, 1, 20, 100};
    std::array<std::uint8_t, 8> masks{0, 1, 0, 1, 0, 1, 0, 1};
    auto source = half_test::buffer<float>(values);
    auto condition_buffer = half_test::buffer<std::uint8_t>(masks);
    std::array stable_outputs{half_test::buffer<float>(values),
                              half_test::buffer<float>(values)};
    std::array selected_outputs{half_test::buffer<float>(values),
                                half_test::buffer<float>(values)};
    std::array invocations{compiled->executable.create_invocation().value(),
                           compiled->executable.create_invocation().value()};
    const std::array queues{graph::QueueBinding{
        .lane = graph::ExecutionLaneId{0}, .queue = &queue}};
    for (std::size_t slot = 0; slot < 2; ++slot) {
      auto &invocation = invocations[slot];
      ASSERT_TRUE(
          invocation
              .bind(
                  input,
                  compute::ConstTensorView::create(data, source.view()).value())
              .ok());
      ASSERT_TRUE(invocation
                      .bind(mask, compute::ConstTensorView::create(
                                      condition, condition_buffer.view())
                                      .value())
                      .ok());
      ASSERT_TRUE(invocation
                      .bind(stable, compute::TensorView::create(
                                        data, stable_outputs[slot].view())
                                        .value())
                      .ok());
      ASSERT_TRUE(invocation
                      .bind(selected, compute::TensorView::create(
                                          data, selected_outputs[slot].view())
                                          .value())
                      .ok());
      ASSERT_TRUE(invocation.prepare_replay(queues).ok());
    }
    const auto before = stats.snapshot();
    std::array<std::vector<float>, 2> previous_stable, previous_selected;
    for (std::size_t run = 0; run < 4; ++run) {
      const auto slot = run % 2;
      for (std::size_t i = 0; i < masks.size(); ++i)
        masks[i] = static_cast<std::uint8_t>((i + run) % 2);
      ASSERT_TRUE(source.view().write(std::as_bytes(std::span(values))).ok());
      ASSERT_TRUE(
          condition_buffer.view().write(std::as_bytes(std::span(masks))).ok());
      auto done = invocations[slot].submit(queue);
      ASSERT_TRUE(done) << done.error().message();
      ASSERT_TRUE(done->wait().ok());
      const auto actual = half_test::read<float>(stable_outputs[slot]);
      const auto chosen = half_test::read<float>(selected_outputs[slot]);
      for (std::size_t i = 0; i < values.size(); ++i) {
        const double x = values[i];
        const double expected =
            x > 0 ? x + std::log1p(std::exp(-x)) : std::log1p(std::exp(x));
        EXPECT_NEAR(actual[i], expected, 5e-6 * std::abs(expected) + 1e-43);
        EXPECT_EQ(chosen[i], masks[i] ? actual[i] : values[i]);
        values[i] += 0.25F;
      }
      previous_stable[slot] = actual;
      previous_selected[slot] = chosen;
      if (run != 0) {
        EXPECT_EQ(half_test::read<float>(stable_outputs[1 - slot]),
                  previous_stable[1 - slot]);
        EXPECT_EQ(half_test::read<float>(selected_outputs[1 - slot]),
                  previous_selected[1 - slot]);
      }
    }
    EXPECT_EQ(stats.snapshot().total_allocs, before.total_allocs);
    EXPECT_EQ(stats.snapshot().reallocs, before.reallocs);
    ASSERT_TRUE(context.close().ok());
    EXPECT_FALSE(invocations[0].submit(queue));
  }
  EXPECT_TRUE(stats.balanced());
}

TEST(CpuGraph, SelectRetainsBitsAndResultsAcrossTwoInvocations) {
  auto context = graph::cpu::create_context().value();
  auto queue = context.create_queue().value();
  const auto condition =
      compute::TensorDesc::dense(compute::DataType::kBool,
                                 std::array<std::int64_t, 1>{4})
          .value();
  const auto data = compute::TensorDesc::dense(compute::DataType::kFloat32,
                                               std::array<std::int64_t, 1>{4})
                        .value();
  auto builder = graph::GraphBuilder::create().value();
  const auto mask = builder.input(condition).value();
  const auto a = builder.input(data).value(), b = builder.input(data).value();
  const auto out = builder.add(compute::Select{}, mask, a, b).value();
  ASSERT_TRUE(builder.publish(out).ok());
  auto logical = std::move(builder).finish().value();
  auto compiled = graph::compile(logical, context);
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->report.step_count(), 1U);
  EXPECT_EQ(compiled->report.step(0)->provider.id,
            compute::onednn::kProviderId);
  std::array<std::uint8_t, 4> masks{0, 1, 0, 1};
  const std::array<std::uint32_t, 4> left{0x7fa12345U, 0x80000000U, 0x00000001U,
                                          0xff800000U};
  const std::array<std::uint32_t, 4> right{0x7fc54321U, 0, 0x807fffffU,
                                           0x7f800000U};
  auto mask_buffer = half_test::buffer<std::uint8_t>(masks);
  auto a_buffer = half_test::buffer<std::uint32_t>(left);
  auto b_buffer = half_test::buffer<std::uint32_t>(right);
  std::array outputs{half_test::buffer<std::uint32_t>(left),
                     half_test::buffer<std::uint32_t>(left)};
  std::array invocations{compiled->executable.create_invocation().value(),
                         compiled->executable.create_invocation().value()};
  const std::array queues{
      graph::QueueBinding{.lane = graph::ExecutionLaneId{0}, .queue = &queue}};
  for (std::size_t slot = 0; slot < 2; ++slot) {
    ASSERT_TRUE(invocations[slot]
                    .bind(mask, compute::ConstTensorView::create(
                                    condition, mask_buffer.view())
                                    .value())
                    .ok());
    ASSERT_TRUE(
        invocations[slot]
            .bind(
                a,
                compute::ConstTensorView::create(data, a_buffer.view()).value())
            .ok());
    ASSERT_TRUE(
        invocations[slot]
            .bind(
                b,
                compute::ConstTensorView::create(data, b_buffer.view()).value())
            .ok());
    ASSERT_TRUE(
        invocations[slot]
            .bind(
                out,
                compute::TensorView::create(data, outputs[slot].view()).value())
            .ok());
    ASSERT_TRUE(invocations[slot].prepare_replay(queues).ok());
  }
  std::array<std::vector<std::uint32_t>, 2> previous;
  for (std::size_t run = 0; run < 4; ++run) {
    for (std::size_t i = 0; i < masks.size(); ++i)
      masks[i] = static_cast<std::uint8_t>((i + run) % 2);
    ASSERT_TRUE(mask_buffer.view().write(std::as_bytes(std::span(masks))).ok());
    auto done = invocations[run % 2].submit(queue);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto actual = half_test::read<std::uint32_t>(outputs[run % 2]);
    for (std::size_t i = 0; i < masks.size(); ++i)
      EXPECT_EQ(actual[i], masks[i] ? left[i] : right[i]);
    previous[run % 2] = actual;
    if (run != 0)
      EXPECT_EQ(half_test::read<std::uint32_t>(outputs[(run + 1) % 2]),
                previous[(run + 1) % 2]);
  }
}
} // namespace
#endif
