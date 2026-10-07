#include "tensor_test_support.h"
#include "../compute_models/cuda_iq1_fixture.h"

#include "hyle/compute/model_ops.h"

namespace {
using namespace ggml_cuda_tensor_test;
TEST(GgmlCudaIq1, OriginalDecodeMmvqAndMmqExecuteAndReplay) {
  auto ctx = Context::create(cuda::device(0).value(), ggml_cuda::provider()).value();
  auto memory = cuda::MemoryResource::create(ctx, cuda::MemoryKind::kDevice).value();
  constexpr std::int64_t k = 256, n = 4;
  for (auto format : cuda_iq1_fixture::formats) {
    SCOPED_TRACE(static_cast<int>(format.type));
    const auto desc = gguf_weight_description(format.type, DataType::kFloat32, k, n).value();
    const auto packed = cuda_iq1_fixture::weights(format, n, k);
    const auto expected = cuda_iq1_fixture::sums(format, packed, n, k);
    auto weights = upload<std::byte>(memory.ref(), packed);
    for (std::int64_t m : {1, 65}) {
      SCOPED_TRACE(m);
      const auto a = dense(DataType::kFloat32, {m, k});
      const auto y = dense(DataType::kFloat32, {m, n});
      std::vector<float> values(m * k, 0.25F);
      auto activation = upload<float>(memory.ref(), values);
      auto output = output_buffer(memory.ref(), y);
      const std::array inputs{ConstTensorView::create(a, activation.view()).value(),
                              ConstTensorView::create(desc, weights.view()).value()};
      const auto view = TensorView::create(y, output.view()).value();
      auto prepared = prepare(ctx, Operation(MatMul{}), inputs, std::span(&view, 1));
      ASSERT_TRUE(prepared) << prepared.error().message();
      EXPECT_EQ(prepared->provider().id, ggml_cuda::kProviderId);
      auto invocation = prepared->create_invocation().value();
      auto queue = ctx.create_queue().value();
      for (int replay = 0; replay < 2; ++replay) {
        auto done = invocation.submit(queue, inputs, std::span(&view, 1));
        ASSERT_TRUE(done) << done.error().message();
        ASSERT_TRUE(done->wait().ok());
        const auto observed = half_test::read<float>(output);
        for (std::int64_t row = 0; row < m; ++row)
          for (std::int64_t column = 0; column < n; ++column)
            EXPECT_NEAR(observed[row * n + column], expected[column] * (replay ? 2 : 1),
                         std::abs(expected[column]) * 0.002F + 1.0e-5F);
        std::fill(values.begin(), values.end(), 0.5F);
        ASSERT_TRUE(activation.view().write(std::as_bytes(std::span(values))).ok());
      }
    }
  }
}
} // namespace
