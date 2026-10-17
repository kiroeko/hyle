#include "../compute_provider_native/half_test_support.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <type_traits>

#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/ggml_cpu.h"

namespace {
using namespace hyle::compute;

template <class Value> void exercise_hadamard() {
  auto context =
      Context::create(cpu::device(), ggml_cpu::provider({.native_threads = 2}))
          .value();
  auto queue = context.create_queue().value();
  for (const int width : {1, 4, 128, 4096}) {
    SCOPED_TRACE(width);
    const std::array<std::int64_t, 3> shape{2, 3, width};
    const auto description =
        TensorDesc::dense(half_test::data_type<Value>(), shape).value();
    auto output = hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                            description.required_bytes().value(),
                                            64)
                      .value();
    const auto destination = TensorView::create(description, output.view()).value();
    for (const bool inverse : {false, true}) {
      SCOPED_TRACE(inverse);
      auto prepared = prepare(context, Operation(Hadamard{.inverse = inverse}),
                              std::span(&description, 1),
                              std::span(&description, 1),
                              hyle::mem::system_resource());
      ASSERT_TRUE(prepared) << prepared.error().message();
      EXPECT_EQ(prepared->provider().id, ggml_cpu::kProviderId);
      auto first = prepared->create_invocation();
      auto second = prepared->create_invocation();
      ASSERT_TRUE(first);
      ASSERT_TRUE(second);
      for (int replay = 0; replay < 3; ++replay) {
        // Sparse dyadic inputs give an independent O(rows*W) Sylvester oracle
        // even for W4096; input quantization is accounted for explicitly.
        std::vector<Value> values(6 * width, half_test::encode<Value>(0));
        for (int row = 0; row < 6; ++row)
          for (int term = 0; term < std::min(width, 3); ++term)
            values[row * width + (term * (width / 3) + row) % width] =
                half_test::encode<Value>(double(row + term + 1) /
                                         (replay % 2 == 0 ? 8 : -16));
        auto source = half_test::buffer<Value>(values);
        const auto input =
            ConstTensorView::create(description, source.view()).value();
        auto &slot = replay % 2 == 0 ? *first : *second;
        auto done = slot.submit(queue, std::span(&input, 1),
                                std::span(&destination, 1));
        ASSERT_TRUE(done) << done.error().message();
        ASSERT_TRUE(done->wait().ok());
        const auto observed = half_test::read<Value>(output);
        for (int row = 0; row < 6; ++row)
          for (int column = 0; column < width; ++column) {
            double sum = 0;
            for (int term = 0; term < std::min(width, 3); ++term) {
              const auto index = (term * (width / 3) + row) % width;
              sum += (std::popcount(unsigned(index & column)) % 2 ? -1 : 1) *
                     half_test::decode(values[row * width + index]);
            }
            const double expected = sum / std::sqrt(double(width));
            if constexpr (std::is_same_v<Value, float>)
              EXPECT_NEAR(observed[row * width + column], expected,
                          2e-6 * std::max(1.0, std::abs(expected)));
            else
              half_test::expect_near(observed[row * width + column], expected);
          }
      }
    }
  }
}

TEST(GgmlCpuHadamard, FloatForwardInverseAndIndependentInvocations) {
  exercise_hadamard<float>();
}
TEST(GgmlCpuHadamard, HalfForwardInverseAndIndependentInvocations) {
  exercise_hadamard<hyle::simd::float16_t>();
}
TEST(GgmlCpuHadamard, BFloatForwardInverseAndIndependentInvocations) {
  exercise_hadamard<hyle::simd::bfloat16_t>();
}
} // namespace
