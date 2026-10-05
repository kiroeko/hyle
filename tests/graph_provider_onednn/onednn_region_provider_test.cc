#include "hyle/compute/backends/cpu.h"
#include "hyle/graph/executable.h"
#include "hyle/graph/providers/onednn.h"
#include "hyle/mem/stats.h"
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <thread>
#include <utility>

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;

c::TensorDesc matrix(std::int64_t rows, std::int64_t columns) {
  return c::TensorDesc::dense(c::DataType::kFloat32, std::array{rows, columns})
      .value();
}
struct NormGraph final {
  g::Graph graph;
  g::ValueId input, scale, output;
};
NormGraph make_norm() {
  auto builder = g::GraphBuilder::create().value();
  auto input = builder.input(matrix(1, 8)).value();
  auto scale = builder.input(matrix(1, 8)).value();
  auto normalized = builder.add(c::RmsNorm{}, input).value();
  auto weighted =
      builder.add(c::Binary{c::BinaryKind::kMultiply}, normalized, scale)
          .value();
  auto output =
      builder.add(c::Binary{c::BinaryKind::kAdd}, weighted, input).value();
  EXPECT_TRUE(builder.publish(output).ok());
  return {std::move(builder).finish().value(), input, scale, output};
}
auto bindings() {
  return std::array{c::ProviderBinding{g::onednn::provider(), 100}};
}
m::Buffer buffer(const c::TensorDesc &desc) {
  return m::Buffer::create(m::system_resource(), desc.required_bytes().value(),
                           64)
      .value();
}

struct AllocatorProbe final {
  hyle::mem::AllocatorRef backing = hyle::mem::system_allocator();
  std::atomic<std::size_t> attempts{0};
  std::atomic<std::size_t> fail_on_attempt{
      (std::numeric_limits<std::size_t>::max)()};
};

void *probe_allocate(void *opaque, std::size_t size,
                     std::size_t alignment) noexcept {
  auto &probe = *static_cast<AllocatorProbe *>(opaque);
  const std::size_t attempt = probe.attempts.fetch_add(1);
  if (attempt == probe.fail_on_attempt.load())
    return nullptr;
  return probe.backing.allocate(size, alignment);
}

void probe_free(void *opaque, void *pointer, std::size_t size,
                std::size_t alignment) noexcept {
  auto &probe = *static_cast<AllocatorProbe *>(opaque);
  probe.backing.deallocate(pointer, size, alignment);
}

void *probe_reallocate(void *opaque, void *pointer, std::size_t old_size,
                       std::size_t new_size, std::size_t alignment) noexcept {
  auto &probe = *static_cast<AllocatorProbe *>(opaque);
  const std::size_t attempt = probe.attempts.fetch_add(1);
  if (attempt == probe.fail_on_attempt.load())
    return nullptr;
  return probe.backing.reallocate(pointer, old_size, new_size, alignment);
}

hyle::mem::AllocatorRef probe_allocator(AllocatorProbe &probe) noexcept {
  return {&probe_allocate, &probe_free, &probe_reallocate, &probe};
}

TEST(OneDnnRegionProvider, HybridGenericFfnIsNotAQualifiedRegion) {
  auto builder = g::GraphBuilder::create().value();
  auto input = builder.input(matrix(1, 8)).value();
  std::array<g::ValueId, 3> weights;
  std::array<m::Buffer, 3> buffers;
  std::array<g::ConstantBinding, 3> constants;
  for (std::size_t i = 0; i < weights.size(); ++i) {
    weights[i] = builder.constant(matrix(8, 8)).value();
    buffers[i] = buffer(matrix(8, 8));
    constants[i] = {
        weights[i],
        c::ConstTensorView::create(matrix(8, 8), buffers[i].view()).value()};
  }
  const c::MatMul mm{.accumulation_type = c::DataType::kFloat32};
  auto gate = builder.add(mm, input, weights[0]).value();
  auto up = builder.add(mm, input, weights[1]).value();
  auto activated =
      builder
          .add(c::GatedActivation{.inputs = c::GatedInputKind::kSeparate}, gate,
               up)
          .value();
  auto output = builder.add(mm, activated, weights[2]).value();
  ASSERT_TRUE(builder.publish(output).ok());
  auto graph = std::move(builder).finish().value();
  const auto providers = bindings();
  auto context =
      c::Context::create(c::cpu::device(), {.providers = providers}).value();
  auto result = g::compile(graph, context, {.constants = constants});
  ASSERT_FALSE(result);
  EXPECT_TRUE(result.error().IsUnimplemented()) << result.error().message();
}

TEST(OneDnnRegionProvider, NativeNormRegionKeepsIndependentReplayAndClose) {
  auto stats = m::StatsAllocator::create(m::system_allocator()).value();
  {
    const auto providers = bindings();
    auto context =
        c::Context::create(c::cpu::device(), {.metadata_allocator = stats.get(),
                                              .providers = providers})
            .value();
    auto graph = make_norm();
    auto compiled = g::compile(graph.graph, context).value();
    ASSERT_EQ(compiled.report.step_count(), 1U);
    EXPECT_EQ(compiled.report.step(0)->provider.id, g::onednn::kProviderId);
    EXPECT_EQ(compiled.report.step(0)->node_count, 3U);
    EXPECT_EQ(compiled.report.step(0)->implementation,
              c::ProviderImplementationId{2});
    auto queue0 = context.create_queue().value(),
         queue1 = context.create_queue().value();
    std::array invocations{compiled.executable.create_invocation().value(),
                           compiled.executable.create_invocation().value()};
    std::array sources{buffer(matrix(1, 8)), buffer(matrix(1, 8))};
    std::array scales{buffer(matrix(1, 8)), buffer(matrix(1, 8))};
    std::array outputs{buffer(matrix(1, 8)), buffer(matrix(1, 8))};
    for (std::size_t slot = 0; slot < 2; ++slot) {
      ASSERT_TRUE(invocations[slot]
                      .bind(graph.input, c::ConstTensorView::create(
                                             matrix(1, 8), sources[slot].view())
                                             .value())
                      .ok());
      ASSERT_TRUE(invocations[slot]
                      .bind(graph.scale, c::ConstTensorView::create(
                                             matrix(1, 8), scales[slot].view())
                                             .value())
                      .ok());
      ASSERT_TRUE(
          invocations[slot]
              .bind(graph.output,
                    c::TensorView::create(matrix(1, 8), outputs[slot].view())
                        .value())
              .ok());
      auto &queue = slot == 0 ? queue0 : queue1;
      const std::array lanes{
          g::QueueBinding{.lane = g::ExecutionLaneId{0}, .queue = &queue}};
      ASSERT_TRUE(invocations[slot].prepare_replay(lanes).ok());
    }
    const auto before = stats.snapshot();
    for (int replay = 0; replay < 3; ++replay) {
      std::array<std::array<float, 8>, 2> values;
      std::array<float, 8> scale;
      scale.fill(1.25F);
      for (std::size_t slot = 0; slot < 2; ++slot) {
        for (std::size_t i = 0; i < 8; ++i)
          values[slot][i] = float(int(i) - 5 + replay + int(slot) * 2) / 8.F;
        ASSERT_TRUE(sources[slot]
                        .view()
                        .write(std::as_bytes(std::span(values[slot])))
                        .ok());
        ASSERT_TRUE(
            scales[slot].view().write(std::as_bytes(std::span(scale))).ok());
      }
      std::array<hyle::Status, 2> status;
      std::thread first([&] {
        auto done = invocations[0].submit(queue0);
        status[0] = done ? done->wait() : done.error();
      });
      std::thread second([&] {
        auto done = invocations[1].submit(queue1);
        status[1] = done ? done->wait() : done.error();
      });
      first.join();
      second.join();
      for (std::size_t slot = 0; slot < 2; ++slot) {
        ASSERT_TRUE(status[slot].ok()) << status[slot].message();
        std::array<float, 8> actual;
        ASSERT_TRUE(outputs[slot]
                        .view()
                        .read(std::as_writable_bytes(std::span(actual)))
                        .ok());
        for (std::size_t row = 0; row < 1; ++row) {
          float squared = 0;
          for (std::size_t col = 0; col < 8; ++col)
            squared +=
                values[slot][row * 8 + col] * values[slot][row * 8 + col];
          const float denominator =
              std::sqrt(squared / 8.F + c::RmsNorm{}.epsilon);
          for (std::size_t col = 0; col < 8; ++col) {
            const auto i = row * 8 + col;
            const float expected =
                values[slot][i] / denominator * scale[col] + values[slot][i];
            EXPECT_NEAR(actual[i], expected,
                        8e-6F + std::abs(expected) * 8e-5F);
          }
        }
      }
    }
    EXPECT_EQ(stats.snapshot().total_allocs, before.total_allocs);
    ASSERT_TRUE(context.close().ok());
    EXPECT_FALSE(invocations[0].submit(queue0));
  }
  EXPECT_TRUE(stats.balanced());
}

TEST(OneDnnRegionProvider, NormCompileAndInvocationColdOomRecover) {
  AllocatorProbe probe;
  const auto providers = bindings();
  auto context =
      c::Context::create(c::cpu::device(),
                         {.metadata_allocator = probe_allocator(probe),
                          .providers = providers})
          .value();
  auto graph = make_norm();
  probe.attempts = 0;
  {
    auto compiled = g::compile(graph.graph, context);
    ASSERT_TRUE(compiled) << compiled.error().message();
  }
  const auto compile_count = probe.attempts.load();
  ASSERT_GT(compile_count, 4U);
  for (std::size_t failure = 0; failure < compile_count; ++failure) {
    probe.attempts = 0;
    probe.fail_on_attempt = failure;
    auto failed = g::compile(graph.graph, context);
    ASSERT_FALSE(failed) << failure;
    EXPECT_TRUE(failed.error().IsResourceExhausted())
        << failed.error().message();
    probe.fail_on_attempt = (std::numeric_limits<std::size_t>::max)();
    auto retried = g::compile(graph.graph, context);
    ASSERT_TRUE(retried) << failure;
  }
  auto compiled = g::compile(graph.graph, context).value();
  probe.attempts = 0;
  {
    auto invocation = compiled.executable.create_invocation();
    ASSERT_TRUE(invocation);
  }
  const auto invocation_count = probe.attempts.load();
  ASSERT_GT(invocation_count, 1U);
  for (std::size_t failure = 0; failure < invocation_count; ++failure) {
    probe.attempts = 0;
    probe.fail_on_attempt = failure;
    auto failed = compiled.executable.create_invocation();
    ASSERT_FALSE(failed) << failure;
    EXPECT_TRUE(failed.error().IsResourceExhausted())
        << failed.error().message();
    probe.fail_on_attempt = (std::numeric_limits<std::size_t>::max)();
    auto retried = compiled.executable.create_invocation();
    ASSERT_TRUE(retried) << failure;
  }
}
} // namespace
