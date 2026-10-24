#include "hyle/compute/providers/ggml_cuda.h"
#include "tensor_test_support.h"

#include <bit>

namespace {
using namespace ggml_cuda_tensor_test;

template <class T>
void exercise_hadamard(DataType type, std::int64_t width, bool inverse) {
  SCOPED_TRACE(static_cast<int>(type));
  SCOPED_TRACE(width);
  SCOPED_TRACE(inverse);
  auto context =
      Context::create(cuda::device(0).value(), ggml_cuda::provider()).value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  // Multiple leading axes must be flattened into independent transform rows.
  const auto description = dense(type, {2, 2, width});
  const auto make_payload = [&](int seed) {
    std::vector<T> values(static_cast<std::size_t>(4 * width));
    for (std::size_t i = 0; i < values.size(); ++i)
      values[i] = half_test::encode<T>(
          double((i * 13 + std::size_t(seed)) % 29) / 32.0 - 14.0 / 32.0);
    // Keep the signed reference sums away from exact zero, so the established
    // relative narrow-port budget tests rounding rather than a new abs floor.
    for (std::size_t i = 0; i < values.size(); i += std::size_t(width))
      values[i] =
          half_test::encode<T>(half_test::decode(values[i]) + 1.0 / 64.0);
    return values;
  };
  const auto a = make_payload(0), b = make_payload(7);
  auto first_input = upload<T>(memory.ref(), a),
       second_input = upload<T>(memory.ref(), b);
  auto first_output = output_buffer(memory.ref(), description);
  auto second_output = output_buffer(memory.ref(), description);
  const std::array inputs{
      ConstTensorView::create(description, first_input.view()).value()};
  const std::array other_inputs{
      ConstTensorView::create(description, second_input.view()).value()};
  const std::array outputs{
      TensorView::create(description, first_output.view()).value()};
  const std::array other_outputs{
      TensorView::create(description, second_output.view()).value()};
  auto prepared = prepare(context, Operation(Hadamard{.inverse = inverse}),
                          inputs, outputs, memory.ref());
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, ggml_cuda::kProviderId);
  auto invocation_workspace =
      ggml_cuda_test::workspace_buffer(*prepared, memory.ref());
  auto invocation = prepared->create_invocation(invocation_workspace.view());
  ASSERT_TRUE(invocation) << invocation.error().message();
  auto queue = context.create_queue().value();
  const auto check = [&](const mem::Buffer &output,
                         const std::vector<T> &input) {
    const auto actual = half_test::read<T>(output);
    double maximum_error = 0;
    // Independent normalized signed-matrix reference; no butterfly/factor code.
    for (std::size_t begin = 0; begin < input.size();
         begin += std::size_t(width)) {
      for (std::int64_t row = 0; row < width; ++row) {
        double sum = 0;
        for (std::int64_t column = 0; column < width; ++column) {
          const auto sign =
              std::popcount(std::uint64_t(row & column)) % 2 ? -1 : 1;
          sum += sign * half_test::decode(input[begin + std::size_t(column)]);
        }
        const double expected = sum / std::sqrt(double(width));
        const auto value = actual[begin + std::size_t(row)];
        ASSERT_TRUE(std::isfinite(half_test::decode(value)));
        maximum_error = std::max(maximum_error,
                                 std::abs(half_test::decode(value) - expected));
        if constexpr (!std::is_same_v<T, float>)
          half_test::expect_near(value, expected);
      }
    }
    if constexpr (std::is_same_v<T, float>)
      EXPECT_LE(maximum_error, 2.e-6); // Existing F32 Hadamard absolute budget.
  };
  auto done = invocation->submit(queue, inputs, outputs);
  ASSERT_TRUE(done) << done.error().message();
  ASSERT_TRUE(done->wait().ok());
  check(first_output, a);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  auto rebound = invocation->submit(queue, other_inputs, other_outputs);
  ASSERT_TRUE(rebound) << rebound.error().message();
  ASSERT_TRUE(rebound->wait().ok());
  check(second_output, b);
  check(first_output, a);
}

TEST(GgmlCudaHadamard, LargeFactorsKeepSylvesterOrderAcrossRowsAndRebinding) {
  for (auto width : {1024, 2048, 4096})
    for (bool inverse : {false, true})
      exercise_hadamard<float>(DataType::kFloat32, width, inverse);
}

TEST(GgmlCudaHadamard, LargeFactorsKeepEstablishedNarrowPortRounding) {
  exercise_hadamard<hyle::simd::float16_t>(DataType::kFloat16, 2048, false);
  exercise_hadamard<hyle::simd::bfloat16_t>(DataType::kBFloat16, 2048, false);
}
} // namespace
