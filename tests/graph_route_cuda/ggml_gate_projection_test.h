#pragma once

#include "../compute_models/d5_gguf_test_support.h"
#include "hyle/graph/cuda.h"
#include "hyle/compute/providers/ggml_cuda.h"
#include "hyle/compute/backends/cuda.h"

namespace ggml_gate_projection_test {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;

// Reduced GLM-5.3 GGUF dense-FFN SwiGLU topology, with smaller K/N. Dyadic
// inputs are exact in the Q8 activation codec; the oracle uses decoded weights.
inline void exercise(c::Context &context, m::MemoryResourceRef resource) {
  constexpr std::int64_t reduction = 128, columns = 64;
  const auto input_desc = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{1, reduction}).value();
  const auto output_desc = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{1, columns}).value();
  const auto weight_desc = c::gguf_weight_description(c::GgufQuantizedType::kQ8_0,
      c::DataType::kFloat32, reduction, columns).value();
  const auto weight = [&](int seed) {
    std::vector<float> values(reduction * columns);
    for (std::int64_t row = 0; row < columns; ++row)
      for (std::int64_t k = 0; k < reduction; ++k)
        values[row * reduction + k] =
            float(k % 32 == 0 ? 127 : (row * 19 + k * 7 + seed * 11) % 127 - 63) / 1024.F;
    return d5_gguf_test::pack(c::GgufQuantizedType::kQ8_0, values, columns,
        reduction, weight_desc.required_bytes().value());
  };
  const auto gate_weights = weight(3), up_weights = weight(7);
  auto gate_buffer = m::Buffer::create(resource, gate_weights.bytes.size(), 256).value();
  auto up_buffer = m::Buffer::create(resource, up_weights.bytes.size(), 256).value();
  ASSERT_TRUE(gate_buffer.view().write(gate_weights.bytes).ok());
  ASSERT_TRUE(up_buffer.view().write(up_weights.bytes).ok());
  auto input = m::Buffer::create(resource, reduction * sizeof(float), 256).value();
  for (const bool publish_intermediates : {false, true}) {
    SCOPED_TRACE(publish_intermediates);
    auto builder = g::GraphBuilder::create().value();
    const auto x = builder.input(input_desc).value();
    const auto wg = builder.constant(weight_desc).value(), wu = builder.constant(weight_desc).value();
    const auto gate = builder.add(c::MatMul{}, x, wg).value();
    const auto up = builder.add(c::MatMul{}, x, wu).value();
    const auto output = builder.add(c::GatedActivation{.inputs = c::GatedInputKind::kSeparate},
        gate, up).value();
    if (publish_intermediates) {
      ASSERT_TRUE(builder.publish(gate).ok());
      ASSERT_TRUE(builder.publish(up).ok());
    }
    ASSERT_TRUE(builder.publish(output).ok());
    const auto logical = std::move(builder).finish().value();
    const std::array constants{
        g::ConstantBinding{wg, c::ConstTensorView::create(weight_desc, gate_buffer.view()).value()},
        g::ConstantBinding{wu, c::ConstTensorView::create(weight_desc, up_buffer.view()).value()}};
    auto compiled = g::cuda::compile(logical, context, resource, {.constants = constants});
    ASSERT_TRUE(compiled) << compiled.error().message();
    ASSERT_EQ(compiled->report.region_count(), 1U);
    ASSERT_EQ(compiled->report.step_count(), 1U);
    EXPECT_EQ(compiled->report.step(0)->provider.id, c::ggml_cuda::kProviderId);
    auto invocation = compiled->executable.create_invocation().value();
    auto queue = context.create_queue().value();
    std::array<m::Buffer, 3> buffers;
    const std::array ids{gate, up, output};
    const std::size_t first = publish_intermediates ? 0 : 2;
    for (std::size_t port = first; port < ids.size(); ++port) {
      buffers[port] = m::Buffer::create(resource, columns * sizeof(float), 256).value();
      ASSERT_TRUE(invocation.bind(ids[port],
          c::TensorView::create(output_desc, buffers[port].view()).value()).ok());
    }
    ASSERT_TRUE(invocation.bind(x, c::ConstTensorView::create(input_desc, input.view()).value()).ok());
    ASSERT_TRUE(g::cuda::prepare_replay(invocation, queue).ok());
    for (int phase = 0; phase < 2; ++phase) {
      SCOPED_TRACE(phase);
      std::array<float, reduction> values;
      for (std::int64_t k = 0; k < reduction; ++k)
        values[k] = float(k % 32 == 0 ? 127 : (k * 11 + phase * 17) % 127 - 63) / 128.F;
      ASSERT_TRUE(input.view().write(std::as_bytes(std::span(values))).ok());
      auto completed = invocation.submit(queue);
      ASSERT_TRUE(completed) << completed.error().message();
      ASSERT_TRUE(completed->wait().ok());
      std::array<std::array<float, columns>, 3> actual{};
      for (std::size_t port = first; port < ids.size(); ++port)
        ASSERT_TRUE(buffers[port].view().read(std::as_writable_bytes(std::span(actual[port]))).ok());
      for (std::int64_t column = 0; column < columns; ++column) {
        double gate_value = 0, up_value = 0;
        for (std::int64_t k = 0; k < reduction; ++k) {
          gate_value += double(values[k]) * gate_weights.decoded[column * reduction + k];
          up_value += double(values[k]) * up_weights.decoded[column * reduction + k];
        }
        const std::array expected{gate_value, up_value,
            gate_value / (1.0 + std::exp(-gate_value)) * up_value};
        for (std::size_t port = first; port < ids.size(); ++port)
          EXPECT_NEAR(actual[port][column], expected[port], 3.0e-6);
      }
    }
  }
}
} // namespace ggml_gate_projection_test
