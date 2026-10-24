#include <array>
#include <climits>
#include <cstdint>
#include <initializer_list>

#include <gtest/gtest.h>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/low_precision.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/ggml_cuda.h"
#include "native_bridge.h"
#include "packed_matmul_native.h"

namespace {
namespace c = hyle::compute;

c::TensorDesc dense(c::DataType type,
                    std::initializer_list<std::int64_t> shape) {
  return c::TensorDesc::dense(type, {shape.begin(), shape.size()}).value();
}

auto prepare(c::Context &context, hyle::mem::MemoryResourceRef resource,
             const c::TensorDesc &left,
             const c::TensorDesc &right, const c::TensorDesc &output) {
  return c::prepare(context, c::Operation(c::MatMul{}), std::array{left, right},
                    std::array{output}, resource);
}

TEST(GgmlCudaPackedBounds, FullVocabularySmallBatchPreparesWithoutWeightStorage) {
  auto context = c::Context::create(c::cuda::device(0).value(),
                                    c::ggml_cuda::provider()).value();
  auto memory = c::cuda::MemoryResource::create(
      context, c::cuda::MemoryKind::kDevice).value();
  constexpr auto f32 = c::DataType::kFloat32;
  constexpr std::int64_t vocabulary = 248320;
  const auto environment = c::cuda::environment(context).value();
  const int native_device = c::ggml_cuda::bridge::device_index(environment.device_ordinal);
  struct Head {
    c::GgufQuantizedType type;
    std::int64_t width;
    std::size_t bytes;
    ggml_type native;
  };
  for (const auto &head : std::array{
           Head{c::GgufQuantizedType::kQ6K, 5120, 1042944000, GGML_TYPE_Q6_K},
           Head{c::GgufQuantizedType::kQ4K, 8192, 1144258560, GGML_TYPE_Q4_K},
           Head{c::GgufQuantizedType::kQ8_0, 8192, 2161377280ULL, GGML_TYPE_Q8_0}}) {
    SCOPED_TRACE(head.width);
    const auto weight = c::gguf_weight_description(
        head.type, f32, head.width, vocabulary).value();
    ASSERT_EQ(weight.required_bytes().value(), head.bytes);
    ASSERT_GT(weight.element_count().value(),
              std::size_t{(INT32_MAX - 1024) / sizeof(float)});
    for(const auto batch : std::array<std::int64_t,3>{1,2,4}) {
      SCOPED_TRACE(batch);
      const auto result = prepare(context, memory.ref(), dense(f32, {batch, head.width}), weight,
                                  dense(f32, {batch, vocabulary}));
      // Expectations come from the nominated native device and its actual
      // selector/tile configuration, not a copied model/device codec table.
      const auto route = c::ggml_cuda::internal::packed_matmul_route(
          native_device, head.native, head.width, vocabulary, batch);
      if(route == c::ggml_cuda::internal::PackedMatmulRoute::kDeclined) {
        ASSERT_FALSE(result);
        EXPECT_TRUE(result.error().IsUnimplemented()) << result.error().message();
      } else {
        ASSERT_TRUE(result) << result.error().message();
        EXPECT_EQ(result->provider().id, c::ggml_cuda::kProviderId);
      }
    }
    // Only descriptor preparation: no weight/output Buffer, Invocation,
    // Queue or tensor payload is created here. Actual execution is checked
    // by the full-vocabulary Q27/Q24 state consumers.
  }
}

TEST(GgmlCudaPackedBounds, LargeHeadsDeclineFallbackAndPaddedLayouts) {
  auto context = c::Context::create(c::cuda::device(0).value(),
                                    c::ggml_cuda::provider()).value();
  auto memory = c::cuda::MemoryResource::create(
      context, c::cuda::MemoryKind::kDevice).value();
  constexpr auto f32 = c::DataType::kFloat32;
  constexpr auto bf16 = c::DataType::kBFloat16;
  constexpr std::int64_t width = 8192, vocabulary = 248320;
  const auto environment = c::cuda::environment(context).value();
  const int native_device = c::ggml_cuda::bridge::device_index(environment.device_ordinal);
  const auto weight = c::gguf_weight_description(
      c::GgufQuantizedType::kQ4K, f32, width, vocabulary).value();
  auto declined = [&](const c::TensorDesc &a, const c::TensorDesc &w,
                      const c::TensorDesc &y) {
    const auto result = prepare(context, memory.ref(), a, w, y);
    ASSERT_FALSE(result);
    EXPECT_TRUE(result.error().IsUnimplemented()) << result.error().message();
  };
  // The first row outside the bounded packed small-batch domain cannot
  // borrow the packed exception and silently take full-weight cuBLAS decode.
  const std::int64_t beyond = 5;
  declined(dense(f32, {beyond, width}), weight, dense(f32, {beyond, vocabulary}));
  // Native row guards handle an odd final output row. The packed exception
  // still requires the nominated device's actual MMVQ or legal MMQ route.
  const auto odd = c::gguf_weight_description(
      c::GgufQuantizedType::kQ4K, f32, width, vocabulary - 1).value();
  if (c::ggml_cuda::internal::packed_matmul_route(native_device, GGML_TYPE_Q4_K,
      width, vocabulary - 1, 2) == c::ggml_cuda::internal::PackedMatmulRoute::kDeclined) {
    declined(dense(f32, {2, width}), odd, dense(f32, {2, vocabulary - 1}));
  } else {
    const auto odd_result = prepare(context, memory.ref(), dense(f32, {2, width}), odd,
                                    dense(f32, {2, vocabulary - 1}));
    ASSERT_TRUE(odd_result) << odd_result.error().message();
  }
  declined(dense(f32, {1, 2, width}), weight, dense(f32, {1, 2, vocabulary}));
  const auto bfloat_weight = c::gguf_weight_description(
      c::GgufQuantizedType::kQ4K, bf16, width, vocabulary).value();
  declined(dense(bf16, {1, width}), bfloat_weight,
           dense(bf16, {1, vocabulary}));
  // Native padding must never require a writable/expanded external weight.
  const auto needs_padding = c::gguf_weight_description(
      c::GgufQuantizedType::kQ6K, f32, 256, 3000000).value();
  declined(dense(f32, {1, 256}), needs_padding, dense(f32, {1, 3000000}));
  const auto row_bytes = weight.byte_strides()[1];
  const auto padded = c::TensorDesc::blocked_strided(f32, weight.dimensions(),
      weight.storage_format(), 0,
      std::array{weight.byte_strides()[0], row_bytes + 144}).value();
  declined(dense(f32, {1, width}), padded, dense(f32, {1, vocabulary}));
  // An ordinary small multirow projection retains its existing admission.
  const auto small = c::gguf_weight_description(
      c::GgufQuantizedType::kQ4K, f32, 512, 8).value();
  const auto result = prepare(context, memory.ref(), dense(f32, {2, 512}), small,
                              dense(f32, {2, 8}));
  ASSERT_TRUE(result) << result.error().message();
}

TEST(GgmlCudaPackedBounds, EncodedAndLogicalLimitsKeepOtherPortsBounded) {
  auto context = c::Context::create(c::cuda::device(0).value(),
                                    c::ggml_cuda::provider()).value();
  auto memory = c::cuda::MemoryResource::create(
      context, c::cuda::MemoryKind::kDevice).value();
  constexpr auto f32 = c::DataType::kFloat32;
  struct Shape { c::GgufQuantizedType type; std::int64_t k, n; };
  for (const auto &shape : std::array{
           // Encoded bytes fit I32, but the native logical product does not.
           Shape{c::GgufQuantizedType::kQ4K, 512, 4194304},
           // The RHS cannot grant a large ordinary activation an exception.
           Shape{c::GgufQuantizedType::kQ4K, 536871424, 1}}) {
    SCOPED_TRACE(shape.k);
    SCOPED_TRACE(shape.n);
    const auto weight = c::gguf_weight_description(
        shape.type, f32, shape.k, shape.n).value();
    const auto result = prepare(context, memory.ref(), dense(f32, {1, shape.k}), weight,
                                dense(f32, {1, shape.n}));
    ASSERT_FALSE(result);
    EXPECT_TRUE(result.error().IsUnimplemented()) << result.error().message();
  }
  // An odd N above I32 encoded bytes is legal with native tail-row guards;
  // logical/block indices, each stride and ordinary A/Y remain bounded.
  constexpr std::int64_t odd_width = 512, odd_columns = 4000001;
  const auto odd_weight = c::gguf_weight_description(
      c::GgufQuantizedType::kQ8_0, f32, odd_width, odd_columns).value();
  ASSERT_GT(odd_weight.required_bytes().value(), std::size_t{INT32_MAX});
  for (const auto batch : std::array<std::int64_t, 2>{1, 2}) {
    SCOPED_TRACE(batch);
    const auto odd_result = prepare(context, memory.ref(), dense(f32, {batch, odd_width}),
        odd_weight, dense(f32, {batch, odd_columns}));
    ASSERT_TRUE(odd_result) << odd_result.error().message();
  }
  const auto result = prepare(context, memory.ref(), dense(f32, {1, 8192}),
      dense(f32, {8192, 248320}), dense(f32, {1, 248320}));
  ASSERT_FALSE(result);
  EXPECT_TRUE(result.error().IsUnimplemented()) << result.error().message();
}
} // namespace
