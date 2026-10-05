#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <span>
#include <utility>

#include "hyle/compute/cpu.h"
#include "hyle/compute/providers/ggml_cpu.h"
#include "hyle/compute/providers/native.h"
#include "hyle/compute/providers/onednn.h"
#include "hyle/graph/cpu.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;

c::TensorDesc matrix(std::int64_t rows, std::int64_t columns) {
  return c::TensorDesc::dense(c::DataType::kFloat32, std::array{rows, columns})
      .value();
}
m::Buffer buffer(const c::TensorDesc &desc) {
  return m::Buffer::create(m::system_resource(), desc.required_bytes().value(),
                           64)
      .value();
}

TEST(CpuDefaultGraph, FullConstantFfnUsesQualifiedGateAcrossTwoReplays) {
#if !defined(HYLE_TEST_TORCH_CPU)
  GTEST_SKIP() << "This qualified managed carrier requires the Torch CPU SDK";
#else
  for (float x : {-80.F, -90.F, -100.F}) {
    for (float value : {1e38F, (std::numeric_limits<float>::infinity)()}) {
      SCOPED_TRACE(x);
      SCOPED_TRACE(value);
      const auto input_desc = matrix(1, 8), weight_desc = matrix(8, 8);
      auto gate_weights = buffer(weight_desc), up_weights = buffer(weight_desc);
      auto down_weights = buffer(weight_desc);
      std::array<float, 64> gate{}, up{}, down{};
      for (std::size_t i = 0; i < 8; ++i) {
        gate[i] = x;
        up[i] = value;
      }
      down.fill(0.125F);
      ASSERT_TRUE(
          gate_weights.view().write(std::as_bytes(std::span(gate))).ok());
      ASSERT_TRUE(up_weights.view().write(std::as_bytes(std::span(up))).ok());
      ASSERT_TRUE(
          down_weights.view().write(std::as_bytes(std::span(down))).ok());
      auto builder = g::GraphBuilder::create().value();
      auto input = builder.input(input_desc).value();
      auto wg = builder.constant(weight_desc).value();
      auto wu = builder.constant(weight_desc).value();
      auto wd = builder.constant(weight_desc).value();
      const c::MatMul mm{.accumulation_type = c::DataType::kFloat32};
      auto score = builder.add(mm, input, wg).value();
      auto values = builder.add(mm, input, wu).value();
      auto gated =
          builder
              .add(c::GatedActivation{.inputs = c::GatedInputKind::kSeparate},
                   score, values)
              .value();
      auto output = builder.add(mm, gated, wd).value();
      ASSERT_TRUE(builder.publish(output).ok());
      auto graph = std::move(builder).finish().value();
      auto context = g::cpu::create_context().value();
      const std::array constants{
          g::ConstantBinding{
              wg, c::ConstTensorView::create(weight_desc, gate_weights.view())
                      .value()},
          g::ConstantBinding{
              wu, c::ConstTensorView::create(weight_desc, up_weights.view())
                      .value()},
          g::ConstantBinding{
              wd, c::ConstTensorView::create(weight_desc, down_weights.view())
                      .value()}};
      auto result = g::compile(graph, context, {.constants = constants});
      ASSERT_TRUE(result) << result.error().message();
      auto compiled = std::move(*result);
      bool qualified_gate = false;
      for (std::size_t i = 0; i < compiled.report.step_count(); ++i) {
        const auto step = compiled.report.step(i).value();
        const auto id = step.provider.id;
        EXPECT_TRUE(id == c::aten_cpu::kProviderId ||
                    id == c::onednn::kProviderId ||
                    id == c::ggml_cpu::kProviderId);
        qualified_gate |= id == c::aten_cpu::kProviderId;
      }
      ASSERT_TRUE(qualified_gate);
      auto queue = context.create_queue().value();
      std::array invocations{compiled.executable.create_invocation().value(),
                             compiled.executable.create_invocation().value()};
      std::array inputs{buffer(input_desc), buffer(input_desc)};
      std::array outputs{buffer(input_desc), buffer(input_desc)};
      const std::array lanes{
          g::QueueBinding{.lane = g::ExecutionLaneId{0}, .queue = &queue}};
      for (std::size_t slot = 0; slot < 2; ++slot) {
        ASSERT_TRUE(invocations[slot]
                        .bind(input, c::ConstTensorView::create(
                                         input_desc, inputs[slot].view())
                                         .value())
                        .ok());
        ASSERT_TRUE(invocations[slot]
                        .bind(output, c::TensorView::create(
                                          input_desc, outputs[slot].view())
                                          .value())
                        .ok());
        ASSERT_TRUE(invocations[slot].prepare_replay(lanes).ok());
      }
      std::array<std::array<float, 8>, 2> retained{};
      for (int replay = 0; replay < 4; ++replay) {
        const auto slot = std::size_t(replay % 2);
        const float scale = replay < 2 ? 1.F : 0.5F;
        std::array<float, 8> activation{};
        activation[0] = scale;
        ASSERT_TRUE(inputs[slot]
                        .view()
                        .write(std::as_bytes(std::span(activation)))
                        .ok());
        auto done = invocations[slot].submit(queue);
        ASSERT_TRUE(done);
        ASSERT_TRUE(done->wait().ok());
        std::array<float, 8> actual;
        ASSERT_TRUE(outputs[slot]
                        .view()
                        .read(std::as_writable_bytes(std::span(actual)))
                        .ok());
        const auto gate_value = double(x * scale);
        const double e = std::exp(gate_value);
        const float expected =
            float(gate_value * e / (1 + e)) * (value * scale);
        for (float item : actual) {
          if (std::isinf(expected)) {
            EXPECT_EQ(item, expected);
          } else {
            EXPECT_NEAR(item, expected, 1e-8F + 1e-5F * std::abs(expected));
          }
        }
        retained[slot] = actual;
        if (replay != 0) {
          std::array<float, 8> other;
          ASSERT_TRUE(outputs[1 - slot]
                          .view()
                          .read(std::as_writable_bytes(std::span(other)))
                          .ok());
          EXPECT_EQ(other, retained[1 - slot]);
        }
      }
    }
  }
#endif
}
} // namespace
