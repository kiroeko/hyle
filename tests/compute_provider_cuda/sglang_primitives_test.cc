#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <span>
#include <vector>

#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/cuda.h"
#include "hyle/mem/stats.h"

namespace {
namespace compute = hyle::compute;
namespace cuda = compute::cuda;
namespace mem = hyle::mem;

struct Tensor {
  mem::Buffer buffer;
  compute::TensorView view;
  Tensor(compute::TensorDesc description, mem::MemoryResourceRef resource)
      : buffer(mem::Buffer::create(resource,
                                   description.required_bytes().value(), 256)
                   .value()),
        view(compute::TensorView::create(description, buffer.view()).value()) {}
  template <class T> void write(const std::vector<T> &values) {
    ASSERT_TRUE(buffer.view().write(std::as_bytes(std::span(values))).ok());
  }
};

template <class T> void rms_shapes() {
  auto stats = mem::StatsAllocator::create(mem::system_allocator()).value();
  {
    const std::array providers{compute::cuda_native::sglang_provider()};
    auto owner = compute::Context::create(cuda::device(0).value(),
                                          {.metadata_allocator = stats.get(),
                                           .providers = providers})
                     .value();
    auto resource =
        cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
    auto queue = owner.create_queue().value();
    std::vector<std::int64_t> widths{64, 128, 256};
    for (std::int64_t width = 512; width <= 8192; width += 256)
      widths.push_back(width);
    for (std::int64_t width = 8704; width <= 16384; width += 512)
      widths.push_back(width);
    for (const auto width : widths) {
      for (const bool inplace : {false, true}) {
        const std::int64_t rows = width == 768 ? 4097 : 17;
        SCOPED_TRACE(::testing::Message() << width << '/' << inplace);
        const auto desc = compute::TensorDesc::dense(half_test::data_type<T>(),
                                                     std::array{rows, width})
                              .value();
        const auto weights = compute::TensorDesc::dense(
                                 half_test::data_type<T>(), std::array{width})
                                 .value();
        Tensor input(desc, resource.ref()), weight(weights, resource.ref()),
            output(desc, resource.ref());
        const auto target = inplace ? input.view : output.view;
        const std::array inputs{input.view.as_const(), weight.view.as_const()};
        auto prepared =
            compute::prepare(owner, compute::Operation(compute::RmsNorm{}),
                             inputs, std::array{target});
        ASSERT_TRUE(prepared.has_value()) << prepared.error().message();
        EXPECT_EQ(prepared->provider().id,
                  compute::cuda_native::kSglangProviderId);
        EXPECT_EQ(prepared->workspace_requirement().size, 0u);
        EXPECT_EQ(prepared->invocation_private_bytes(), 0u);
        auto slot = prepared->create_invocation().value();
        std::vector<T> values(static_cast<std::size_t>(rows * width));
        std::vector<T> scales(static_cast<std::size_t>(width));
        for (std::size_t i = 0; i < scales.size(); ++i)
          scales[i] =
              half_test::encode<T>((static_cast<int>(i % 23) - 11) * 0.0625);
        weight.write(scales);
        for (int repeat = 0; repeat < 2; ++repeat) {
          for (std::size_t i = 0; i < values.size(); ++i)
            values[i] = half_test::encode<T>(
                (static_cast<int>(i % 41) - 20 + repeat) * 0.125);
          input.write(values);
          auto done = slot.submit(queue, inputs, std::array{target});
          ASSERT_TRUE(done.has_value()) << done.error().message();
          ASSERT_TRUE(done->wait().ok());
          const auto actual =
              half_test::read<T>(inplace ? input.buffer : output.buffer);
          for (std::size_t row = 0; row < static_cast<std::size_t>(rows);
               ++row) {
            double sum = 0;
            for (std::size_t col = 0; col < scales.size(); ++col) {
              const double value =
                  half_test::decode(values[row * scales.size() + col]);
              sum += value * value;
            }
            const double inverse = 1 / std::sqrt(sum / width + 1.0e-5F);
            for (std::size_t col = 0; col < scales.size(); ++col) {
              const auto index = row * scales.size() + col;
              half_test::expect_near(
                  actual[index], half_test::decode(values[index]) * inverse *
                                     half_test::decode(scales[col]));
            }
          }
        }
        const auto after = half_test::read<T>(weight.buffer);
        for (std::size_t i = 0; i < after.size(); ++i)
          EXPECT_EQ(after[i].bits, scales[i].bits);
      }
    }
    ASSERT_TRUE(owner.close().ok());
  }
  EXPECT_TRUE(stats.balanced());
}

TEST(SglangPrimitives, AllHalfRmsWidthsPreserveWeightsAliasAndReuse) {
  rms_shapes<hyle::simd::float16_t>();
  rms_shapes<hyle::simd::bfloat16_t>();
}

template <class T> void unary_cases() {
  auto owner = compute::Context::create(cuda::device(0).value(),
                                        compute::cuda_native::sglang_provider())
                   .value();
  auto resource =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  auto queue = owner.create_queue().value();
  const auto desc =
      compute::TensorDesc::dense(half_test::data_type<T>(),
                                 std::array<std::int64_t, 2>{3, 64})
          .value();
  Tensor input(desc, resource.ref()), output(desc, resource.ref());
  const std::array edges{-std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN(),
                         -100.0,
                         -80.0,
                         -20.0,
                         -15.0,
                         -10.0,
                         -8.0,
                         -6.0,
                         -5.0,
                         -3.0,
                         -1.0,
                         -0.0,
                         0.0,
                         1.0,
                         4.0,
                         80.0};
  std::vector<T> values(192);
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = half_test::encode<T>(edges[i % edges.size()]);
  for (const auto kind : {compute::UnaryKind::kSilu, compute::UnaryKind::kGelu,
                          compute::UnaryKind::kGeluTanh}) {
    for (const bool inplace : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << static_cast<int>(kind) << '/' << inplace);
      input.write(values);
      const auto target = inplace ? input.view : output.view;
      auto prepared = compute::prepare(
                          owner, compute::Operation(compute::Unary{kind}),
                          std::array{input.view.as_const()}, std::array{target})
                          .value();
      EXPECT_EQ(prepared.provider().id,
                compute::cuda_native::kSglangProviderId);
      auto slot = prepared.create_invocation().value();
      auto done = slot.submit(queue, std::array{input.view.as_const()},
                              std::array{target});
      ASSERT_TRUE(done.has_value()) << done.error().message();
      ASSERT_TRUE(done->wait().ok());
      const auto actual =
          half_test::read<T>(inplace ? input.buffer : output.buffer);
      for (std::size_t i = 0; i < values.size(); ++i) {
        const double x = half_test::decode(values[i]);
        double expected;
        if (std::isinf(x))
          expected = x < 0 ? 0 : x;
        else if (kind == compute::UnaryKind::kGelu)
          expected = 0.5 * x * std::erfc(-x / std::sqrt(2.0));
        else {
          const double z = kind == compute::UnaryKind::kSilu
                               ? x
                               : 2 * std::sqrt(2 / 3.14159265358979323846) *
                                     (x + 0.044715 * x * x * x);
          expected = x / (1 + std::exp(-z));
        }
        if constexpr (std::is_same_v<T, float>) {
          if (std::isnan(expected)) {
            EXPECT_TRUE(std::isnan(actual[i]));
          } else if (std::isinf(expected)) {
            EXPECT_EQ(actual[i], expected);
          } else {
            EXPECT_NEAR(actual[i], expected,
                        std::abs(expected) * 6e-6 +
                            128.0 * std::numeric_limits<float>::denorm_min());
          }
        } else {
          half_test::expect_near(actual[i], expected);
        }
        if (x == 0) {
          EXPECT_EQ(std::signbit(half_test::decode(actual[i])),
                    std::signbit(x));
        }
      }
    }
  }
  ASSERT_TRUE(owner.close().ok());
}

TEST(SglangPrimitives, UnaryStableTailsSpecialValuesAndInplace) {
  unary_cases<float>();
  unary_cases<hyle::simd::float16_t>();
  unary_cases<hyle::simd::bfloat16_t>();
}

TEST(SglangPrimitives, RejectsUnmatchedNormAndUnaryDomains) {
  auto owner = compute::Context::create(cuda::device(0).value(),
                                        compute::cuda_native::sglang_provider())
                   .value();
  auto resource =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  for (const auto width : {63, 129, 257, 8448, 16896}) {
    const auto desc =
        compute::TensorDesc::dense(compute::DataType::kBFloat16,
                                   std::array<std::int64_t, 2>{1, width})
            .value();
    const auto w =
        compute::TensorDesc::dense(compute::DataType::kBFloat16,
                                   std::array<std::int64_t, 1>{width})
            .value();
    Tensor input(desc, resource.ref()), weight(w, resource.ref()),
        output(desc, resource.ref());
    EXPECT_FALSE(compute::prepare(
                     owner, compute::Operation(compute::RmsNorm{}),
                     std::array{input.view.as_const(), weight.view.as_const()},
                     std::array{output.view})
                     .has_value());
  }
  const auto desc = compute::TensorDesc::dense(compute::DataType::kBFloat16,
                                               std::array<std::int64_t, 1>{128})
                        .value();
  const auto f32 = compute::TensorDesc::dense(compute::DataType::kFloat32,
                                              std::array<std::int64_t, 1>{128})
                       .value();
  Tensor input(desc, resource.ref()), weight(f32, resource.ref()),
      output(desc, resource.ref());
  EXPECT_FALSE(compute::prepare(
                   owner, compute::Operation(compute::RmsNorm{}),
                   std::array{input.view.as_const(), weight.view.as_const()},
                   std::array{output.view})
                   .has_value());
  EXPECT_FALSE(compute::prepare(owner, compute::Operation(compute::RmsNorm{}),
                                std::array{input.view.as_const()},
                                std::array{output.view})
                   .has_value());
  EXPECT_FALSE(
      compute::prepare(owner, compute::Operation(compute::LayerNorm{}),
                       std::array{input.view.as_const(), input.view.as_const()},
                       std::array{output.view})
          .has_value());
  EXPECT_FALSE(compute::prepare(
                   owner,
                   compute::Operation(compute::Unary{compute::UnaryKind::kAbs}),
                   std::array{input.view.as_const()}, std::array{output.view})
                   .has_value());
}
} // namespace
