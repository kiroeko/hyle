#pragma once

#include "../compute_provider_ggml_cuda/test_support.h"
#include "hyle/compute/providers/cublas.h"
#include "hyle/graph/executable.h"

namespace ggml_cuda_graph_test {
using namespace ggml_cuda_test;
namespace graph = hyle::graph;

template <class Compile>
void exercise(compute::Context &context, bool projection, Compile compile) {
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  Tensor input(memory.ref()), weights(memory.ref(), 128, 16);
  Tensor first_output(memory.ref(), 4, projection ? 16 : 128);
  Tensor second_output(memory.ref(), 4, projection ? 16 : 128);
  const auto weight_values = weights.write(1);
  auto builder = graph::GraphBuilder::create().value();
  const auto x = builder.input(input.description).value();
  const auto norm = builder.add(compute::RmsNorm{}, x).value();
  auto y = builder.add(compute::Binary{}, norm, x).value();
  graph::ValueId w;
  if (projection) {
    w = builder.input(weights.description).value();
    y = builder.add(compute::MatMul{}, y, w).value();
  }
  ASSERT_TRUE(builder.publish(y).ok());
  auto logical = std::move(builder).finish().value();
  auto compiled = compile(logical, memory.ref());
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->report.step_count(), projection ? 2u : 1u);
  EXPECT_EQ(compiled->report.step(0)->provider.id,
            compute::ggml_cuda::kProviderId);
  if (projection) {
    EXPECT_EQ(compiled->report.step(1)->provider.id,
              compute::cublas::kProviderId);
    EXPECT_TRUE(compiled->report.mixed_providers());
  }
  auto first_queue = context.create_queue().value();
  auto second_queue = context.create_queue().value();
  auto first = compiled->executable.create_invocation();
  ASSERT_TRUE(first) << first.error().message();
  auto second = compiled->executable.create_invocation();
  ASSERT_TRUE(second) << second.error().message();
  ASSERT_TRUE(first->bind(x, input.view.as_const()).ok());
  ASSERT_TRUE(second->bind(x, input.view.as_const()).ok());
  ASSERT_TRUE(first->bind(y, first_output.view).ok());
  ASSERT_TRUE(second->bind(y, second_output.view).ok());
  if (projection) {
    ASSERT_TRUE(first->bind(w, weights.view.as_const()).ok());
    ASSERT_TRUE(second->bind(w, weights.view.as_const()).ok());
  }
  const std::array first_bindings{graph::QueueBinding{
      .lane = graph::ExecutionLaneId{0}, .queue = &first_queue}};
  const std::array second_bindings{graph::QueueBinding{
      .lane = graph::ExecutionLaneId{0}, .queue = &second_queue}};
  ASSERT_TRUE(first->prepare_replay(first_bindings).ok());
  ASSERT_TRUE(second->prepare_replay(second_bindings).ok());
  EXPECT_EQ(first->replay_route(), graph::ReplayRoute::kCommands);
  const auto check = [&](const std::vector<float> &values, Tensor &output) {
    if (!projection) {
      check_norm(values, output.read(), true);
      return;
    }
    const auto actual = output.read();
    for (std::size_t row = 0; row < 4; ++row) {
      double sum = 0;
      for (std::size_t k = 0; k < 128; ++k) {
        const double value = values[row * 128 + k];
        sum += value * value;
      }
      const double scale = 1 + 1 / std::sqrt(sum / 128 + 1e-5);
      for (std::size_t col = 0; col < 16; ++col) {
        double expected = 0;
        for (std::size_t k = 0; k < 128; ++k)
          expected +=
              values[row * 128 + k] * scale * weight_values[k * 16 + col];
        EXPECT_NEAR(actual[row * 16 + col], expected, 0.002);
      }
    }
  };
  std::vector<float> old;
  for (const int seed : {0, 3, 7}) {
    old = input.write(seed);
    auto a = first->submit(first_queue);
    ASSERT_TRUE(a) << a.error().message();
    auto b = second->submit(second_queue);
    ASSERT_TRUE(b) << b.error().message();
    ASSERT_TRUE(b->wait().ok());
    ASSERT_TRUE(a->wait().ok());
    check(old, first_output);
    check(old, second_output);
  }
  const auto changed = input.write(12);
  ASSERT_TRUE(second->submit(second_queue)->wait().ok());
  check(old, first_output);
  check(changed, second_output);
}
} // namespace ggml_cuda_graph_test
