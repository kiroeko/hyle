#pragma once

#include "../compute_provider_ggml_cuda/softmax_test_support.h"
#include "region_test_support.h"

namespace ggml_cuda_graph_test {

template <class T, class Compile>
void exercise_softmax_graph(compute::Context &context, Compile compile) {
  using namespace compute;
  namespace support = ggml_cuda_tensor_test;
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const auto type = half_test::data_type<T>();
  const auto description = support::dense(type, {2, 3, 17});
  const auto mask_desc = support::dense(type, {1, 3, 1});
  auto input = support::output_buffer(memory.ref(), description);
  const std::array mask_values{
      half_test::encode<T>(0.25),
      half_test::encode<T>(-std::numeric_limits<double>::infinity()),
      half_test::encode<T>(-0.5)};
  auto mask = support::upload<T>(memory.ref(), mask_values);
  auto first_probabilities = support::output_buffer(memory.ref(), description);
  auto second_probabilities = support::output_buffer(memory.ref(), description);
  auto first_result = support::output_buffer(memory.ref(), description);
  auto second_result = support::output_buffer(memory.ref(), description);
  auto builder = graph::GraphBuilder::create().value();
  const auto x = builder.input(description).value();
  const auto m = builder.input(mask_desc).value();
  const auto p =
      builder
          .add(Softmax{.axis = 1, .scale = 0.5F, .mask = MaskKind::kAdditive},
               x, m)
          .value();
  const auto y = builder.add(Binary{}, p, x).value();
  ASSERT_TRUE(builder.publish(p).ok());
  ASSERT_TRUE(builder.publish(y).ok());
  auto logical = std::move(builder).finish().value();
  auto compiled = compile(logical, memory.ref());
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->report.step_count(), 1u);
  EXPECT_EQ(compiled->report.step(0)->provider.id,
            compute::ggml_cuda::kProviderId);
  auto first = compiled->executable.create_invocation().value();
  auto second = compiled->executable.create_invocation().value();
  auto q1 = context.create_queue().value();
  auto q2 = context.create_queue().value();
  const auto bind = [&](graph::Invocation &slot, mem::Buffer &probabilities,
                        mem::Buffer &result, Queue &queue) {
    ASSERT_TRUE(
        slot.bind(x, ConstTensorView::create(description, input.view()).value())
            .ok());
    ASSERT_TRUE(
        slot.bind(m, ConstTensorView::create(mask_desc, mask.view()).value())
            .ok());
    ASSERT_TRUE(
        slot.bind(p,
                  TensorView::create(description, probabilities.view()).value())
            .ok());
    ASSERT_TRUE(
        slot.bind(y, TensorView::create(description, result.view()).value())
            .ok());
    const std::array queues{
        graph::QueueBinding{graph::ExecutionLaneId{0}, &queue}};
    ASSERT_TRUE(slot.prepare_replay(queues).ok());
  };
  bind(first, first_probabilities, first_result, q1);
  bind(second, second_probabilities, second_result, q2);
  std::vector<T> values(description.element_count().value());
  const auto write = [&](int seed) {
    for (std::size_t i = 0; i < values.size(); ++i)
      values[i] =
          half_test::encode<T>((static_cast<int>(i % 19) - 9 + seed) / 4.0);
    ASSERT_TRUE(input.view().write(std::as_bytes(std::span(values))).ok());
  };
  const auto check = [&](mem::Buffer &probabilities, mem::Buffer &result) {
    const auto expected =
        support::softmax_reference<T>(values, description.dimensions(), 1, 0.5F,
                                      mask_values, mask_desc.dimensions());
    const auto actual_p = half_test::read<T>(probabilities);
    const auto actual_y = half_test::read<T>(result);
    for (std::size_t i = 0; i < values.size(); ++i) {
      half_test::expect_near(actual_p[i], expected[i]);
      const auto rounded = half_test::decode(half_test::encode<T>(expected[i]));
      half_test::expect_near(actual_y[i],
                             rounded + half_test::decode(values[i]));
    }
  };
  write(0);
  auto a = first.submit(q1);
  auto b = second.submit(q2);
  ASSERT_TRUE(a) << a.error().message();
  ASSERT_TRUE(b) << b.error().message();
  ASSERT_TRUE(b->wait().ok());
  ASSERT_TRUE(a->wait().ok());
  check(first_probabilities, first_result);
  check(second_probabilities, second_result);
  const auto retained_p = half_test::read<T>(first_probabilities);
  const auto retained_y = half_test::read<T>(first_result);
  for (const int seed : {3, 6}) {
    write(seed);
    auto done = second.submit(q2);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    check(second_probabilities, second_result);
    const auto old_p = half_test::read<T>(first_probabilities);
    const auto old_y = half_test::read<T>(first_result);
    EXPECT_EQ(
        std::memcmp(old_p.data(), retained_p.data(), old_p.size() * sizeof(T)),
        0);
    EXPECT_EQ(
        std::memcmp(old_y.data(), retained_y.data(), old_y.size() * sizeof(T)),
        0);
  }
}

} // namespace ggml_cuda_graph_test
