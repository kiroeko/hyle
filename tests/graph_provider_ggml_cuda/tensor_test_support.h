#pragma once

#include "../compute_provider_ggml_cuda/tensor_test_support.h"
#include "region_test_support.h"

namespace ggml_cuda_graph_test {

template <class T, class Compile>
void exercise_tensor_graph(compute::Context &context, Compile compile) {
  using namespace compute;
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const auto type = half_test::data_type<T>();
  const auto description =
      ggml_cuda_tensor_test::dense(type, {2, 1, 3, 1, 2, 7});
  const auto scalar = ggml_cuda_tensor_test::dense(type, {});
  const auto reduced = ggml_cuda_tensor_test::dense(type, {1, 1, 1, 1, 1, 7});
  auto input = ggml_cuda_tensor_test::output_buffer(memory.ref(), description);
  const std::array epsilon_value{half_test::encode<T>(0.125)};
  auto epsilon = ggml_cuda_tensor_test::upload<T>(memory.ref(), epsilon_value);
  auto first_sum = ggml_cuda_tensor_test::output_buffer(memory.ref(), reduced);
  auto second_sum = ggml_cuda_tensor_test::output_buffer(memory.ref(), reduced);
  auto first_result =
      ggml_cuda_tensor_test::output_buffer(memory.ref(), description);
  auto second_result =
      ggml_cuda_tensor_test::output_buffer(memory.ref(), description);
  auto builder = graph::GraphBuilder::create().value();
  const auto x = builder.input(description).value();
  const auto e = builder.input(scalar).value();
  const auto squared = builder.add(Binary{BinaryKind::kMultiply}, x, x).value();
  const auto sum =
      builder
          .add(Reduce{.axes = Axes::from(std::array<std::int32_t, 3>{0, 2, 4})
                                  .value(),
                      .keep_dimensions = true,
                      .accumulation_type = DataType::kFloat32},
               squared)
          .value();
  const auto stabilized = builder.add(Binary{}, sum, e).value();
  const auto denominator =
      builder.add(Unary{UnaryKind::kSqrt}, stabilized).value();
  const auto y =
      builder.add(Binary{BinaryKind::kDivide}, x, denominator).value();
  ASSERT_TRUE(builder.publish(sum).ok());
  ASSERT_TRUE(builder.publish(y).ok());
  auto logical = std::move(builder).finish().value();
  auto compiled = compile(logical, memory.ref());
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->report.step_count(), 1u);
  EXPECT_EQ(compiled->report.step(0)->provider.id,
            compute::ggml_cuda::kProviderId);
  auto first_queue = context.create_queue().value();
  auto second_queue = context.create_queue().value();
  auto first = compiled->executable.create_invocation().value();
  auto second = compiled->executable.create_invocation().value();
  const auto bind = [&](graph::Invocation &slot, mem::Buffer &s,
                        mem::Buffer &result) {
    ASSERT_TRUE(
        slot.bind(x, ConstTensorView::create(description, input.view()).value())
            .ok());
    ASSERT_TRUE(
        slot.bind(e, ConstTensorView::create(scalar, epsilon.view()).value())
            .ok());
    ASSERT_TRUE(
        slot.bind(sum, TensorView::create(reduced, s.view()).value()).ok());
    ASSERT_TRUE(
        slot.bind(y, TensorView::create(description, result.view()).value())
            .ok());
  };
  bind(first, first_sum, first_result);
  bind(second, second_sum, second_result);
  const std::array q1{
      graph::QueueBinding{graph::ExecutionLaneId{0}, &first_queue}};
  const std::array q2{
      graph::QueueBinding{graph::ExecutionLaneId{0}, &second_queue}};
  EXPECT_FALSE(
      first
          .prepare_replay(q1, {.mode = graph::ReplayMode::kRequireCapture,
                               .host_wait = graph::HostWaitPolicy::kReject})
          .ok());
  ASSERT_TRUE(first.prepare_replay(q1).ok());
  ASSERT_TRUE(second.prepare_replay(q2).ok());
  std::vector<T> values(description.element_count().value());
  const auto write = [&](int seed) {
    for (std::size_t i = 0; i < values.size(); ++i)
      values[i] =
          half_test::encode<T>((static_cast<int>(i % 13) - 6 + seed) / 8.0);
    ASSERT_TRUE(input.view().write(std::as_bytes(std::span(values))).ok());
  };
  const auto rounded = [](double value) {
    return half_test::decode(half_test::encode<T>(value));
  };
  const auto check = [&](mem::Buffer &s, mem::Buffer &result) {
    const auto sums = half_test::read<T>(s);
    const auto actual = half_test::read<T>(result);
    // Reducing nonadjacent leading axes publishes a reshape view of the
    // native sum producer. Its root must stay external and fusion-protected.
    for (std::size_t col = 0; col < 7; ++col) {
      double expected_sum = 0;
      for (std::size_t row = 0; row < values.size() / 7; ++row) {
        const auto v = half_test::decode(values[row * 7 + col]);
        expected_sum += rounded(v * v);
      }
      expected_sum = rounded(expected_sum);
      half_test::expect_near(sums[col], expected_sum);
      const auto divisor = rounded(std::sqrt(rounded(expected_sum + 0.125)));
      for (std::size_t row = 0; row < values.size() / 7; ++row)
        half_test::expect_near(actual[row * 7 + col],
                               half_test::decode(values[row * 7 + col]) /
                                   divisor);
    }
  };
  write(0);
  auto a = first.submit(first_queue);
  auto b = second.submit(second_queue);
  ASSERT_TRUE(a) << a.error().message();
  ASSERT_TRUE(b) << b.error().message();
  ASSERT_TRUE(b->wait().ok());
  ASSERT_TRUE(a->wait().ok());
  check(first_sum, first_result);
  check(second_sum, second_result);
  const auto retained = half_test::read<T>(first_result);
  for (const int seed : {2, 5, 9}) {
    write(seed);
    auto done = second.submit(second_queue);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    check(second_sum, second_result);
    const auto old = half_test::read<T>(first_result);
    EXPECT_EQ(std::memcmp(old.data(), retained.data(), old.size() * sizeof(T)),
              0);
  }
}
} // namespace ggml_cuda_graph_test
