#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <ggml.h>

#include "../../src/compute_provider_ggml_metal/native_program.h"
#include "hyle/compute/low_precision.h"
#include "hyle/compute/model_ops.h"
#include "hyle/mem/quota_allocator.h"

namespace {
namespace c = hyle::compute;
namespace metal = c::ggml_metal;

TEST(GgmlMetalProgram, CpuFormatPortDoesNotQualifyGpuMatrices) {
  const auto activation = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{2, 256}).value();
  const auto grouped_activation = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 3>{1, 2, 256}).value();
  const auto ids = c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t, 2>{1, 2}).value();
  for (const auto type : {c::GgufQuantizedType::kQ4_0, c::GgufQuantizedType::kQ4_1,
       c::GgufQuantizedType::kQ2K, c::GgufQuantizedType::kIq2Xxs,
       c::GgufQuantizedType::kIq2Xs, c::GgufQuantizedType::kIq2S,
       c::GgufQuantizedType::kIq1S, c::GgufQuantizedType::kIq1M,
       c::GgufQuantizedType::kIq1Xs, c::GgufQuantizedType::kIq1Xxs,
       c::GgufQuantizedType::kIq1Xxxs}) {
    SCOPED_TRACE(static_cast<int>(type));
    const auto weight = c::gguf_weight_description(type, c::DataType::kFloat32, 256, 3).value();
    const std::array inputs{activation, weight};
    EXPECT_TRUE(metal::NativeProgram::validate(c::Operation(c::MatMul{}), inputs).IsUnimplemented());
    const auto grouped = c::gguf_grouped_weight_description(type, c::DataType::kFloat32, 2, 3, 256).value();
    const std::array group_inputs{grouped, grouped_activation, ids};
    EXPECT_TRUE(metal::NativeProgram::validate(c::model_ops::grouped_matmul(), group_inputs).IsUnimplemented());
  }
}

struct Buffer final {
  id<MTLBuffer> value = nil;
  Buffer() = default;
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  ~Buffer() { [value release]; }
};

std::uint16_t bfloat16(float value) {
  auto bits = std::bit_cast<std::uint32_t>(value);
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

std::array<c::TensorDesc, 6> schema(int tokens, int heads, int depth) {
  auto tensor = [](std::array<std::int64_t, 4> shape) {
    return c::TensorDesc::dense(c::DataType::kFloat32, shape).value();
  };
  const auto q = tensor({2, tokens, heads, depth});
  const auto gate = tensor({2, tokens, heads, 1});
  return {q, q, q, gate, gate, tensor({2, heads, depth, depth})};
}

ggml_type native_type(c::GgufQuantizedType type) {
  switch (type) {
  case c::GgufQuantizedType::kQ3K:
    return GGML_TYPE_Q3_K;
  case c::GgufQuantizedType::kQ4K:
    return GGML_TYPE_Q4_K;
  case c::GgufQuantizedType::kQ5K:
    return GGML_TYPE_Q5_K;
  case c::GgufQuantizedType::kQ6K:
    return GGML_TYPE_Q6_K;
  case c::GgufQuantizedType::kQ8_0:
    return GGML_TYPE_Q8_0;
  case c::GgufQuantizedType::kIq3Xxs:
    return GGML_TYPE_IQ3_XXS;
  case c::GgufQuantizedType::kIq3S:
    return GGML_TYPE_IQ3_S;
  case c::GgufQuantizedType::kIq4Nl:
    return GGML_TYPE_IQ4_NL;
  case c::GgufQuantizedType::kIq4Xs:
    return GGML_TYPE_IQ4_XS;
  case c::GgufQuantizedType::kMxfp4:
    return GGML_TYPE_MXFP4;
  }
  return GGML_TYPE_COUNT;
}

std::vector<std::byte> quantize(c::GgufQuantizedType type,
                                std::span<const float> values, int rows,
                                int width, std::size_t bytes) {
  std::vector<std::byte> result(bytes);
  const auto native = native_type(type);
  std::vector<float> importance(width, 1.0F);
  const auto written = ggml_quantize_chunk(
      native, values.data(), result.data(), 0, rows, width,
      ggml_quantize_requires_imatrix(native) ? importance.data() : nullptr);
  EXPECT_EQ(written, bytes);
  return result;
}

std::vector<float> dequantize(c::GgufQuantizedType type,
                              std::span<const std::byte> values, int rows,
                              int width) {
  const auto native = native_type(type);
  const auto *traits = ggml_get_type_traits(native);
  EXPECT_NE(traits, nullptr);
  EXPECT_NE(traits->to_float, nullptr);
  const auto row_bytes = ggml_row_size(native, width);
  std::vector<float> result(static_cast<std::size_t>(rows * width));
  for (int row = 0; row < rows; ++row)
    traits->to_float(values.data() + row * row_bytes,
                     result.data() + row * width, width);
  return result;
}

// Independent double recurrence in logical [value,key] orientation. This is
// a test oracle, not a product provider or a transcription of Metal reductions.
std::vector<double> reference(const std::array<std::vector<float>, 6> &x,
                              int tokens, int heads, int depth) {
  const std::size_t score_size = x[2].size();
  std::vector<double> result(score_size + x[5].size());
  std::vector<double> state(x[5].begin(), x[5].end());
  for (int batch = 0; batch < 2; ++batch) {
    for (int head = 0; head < heads; ++head) {
      for (int time = 0; time < tokens; ++time) {
        const auto token = std::size_t((batch * tokens + time) * heads + head);
        const auto matrix = std::size_t(batch * heads + head) * depth * depth;
        const double decay = std::exp(double(x[3][token]));
        for (int value = 0; value < depth; ++value) {
          const auto row = matrix + std::size_t(value * depth);
          double projected = 0;
          for (int key = 0; key < depth; ++key) {
            state[row + key] *= decay;
            projected += state[row + key] * x[1][token * depth + key];
          }
          const double delta = double(x[4][token]) *
                               (double(x[2][token * depth + value]) - projected);
          double output = 0;
          for (int key = 0; key < depth; ++key) {
            state[row + key] += delta * x[1][token * depth + key];
            output += state[row + key] * x[0][token * depth + key];
          }
          result[token * depth + value] = output / std::sqrt(double(depth));
        }
      }
    }
  }
  std::copy(state.begin(), state.end(), result.begin() + score_size);
  return result;
}

void run_case(int tokens, int heads, int depth) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    ASSERT_NE(device, nil);
    auto descriptions = schema(tokens, heads, depth);
    auto program = metal::NativeProgram::create(descriptions);
    ASSERT_TRUE(program) << program.error().message();
    std::array<Buffer, 7> owners;
    std::array<metal::BufferView, 7> ports;
    std::array<std::vector<float>, 6> data;
    for (std::size_t i = 0; i < ports.size(); ++i) {
      const auto bytes = i == 6 ? (*program)->output_bytes()
                                : descriptions[i].required_bytes().value();
      owners[i].value = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
      ASSERT_NE(owners[i].value, nil);
      ports[i] = {(__bridge void *)owners[i].value, 0};
      if (i == 6)
        continue;
      data[i].resize(bytes / sizeof(float));
      for (std::size_t j = 0; j < data[i].size(); ++j) {
        data[i][j] = i == 3 ? -0.03F * float(1 + j % 7)
                   : i == 4 ? 0.1F + 0.02F * float(j % 5)
                            : float(std::sin(double(j + 13 * i) * 0.17) * 0.025);
      }
    }
    const auto expected = reference(data, tokens, heads, depth);
    for (std::size_t i = 0; i < data.size(); ++i)
      std::memcpy(owners[i].value.contents, data[i].data(), data[i].size() * sizeof(float));
    for (int repeat = 0; repeat < 3; ++repeat) {
      const auto status = (*program)->run(ports);
      ASSERT_TRUE(status.ok()) << status.message();
      const auto *actual = static_cast<const float *>(owners[6].value.contents);
      double max_error = 0;
      for (std::size_t i = 0; i < expected.size(); ++i)
        max_error = std::max(max_error, std::abs(double(actual[i]) - expected[i]));
      EXPECT_LT(max_error, 3e-6);
      EXPECT_EQ(std::memcmp(owners[5].value.contents, data[5].data(),
                            data[5].size() * sizeof(float)), 0);
    }
    // Failed rebinding must roll back all newly imported wrappers and leave
    // the original program usable; no command has been submitted yet.
    auto invalid = ports;
    invalid[6].offset = (*program)->output_bytes();
    EXPECT_TRUE((*program)->run(invalid).IsOutOfRange());
    EXPECT_TRUE((*program)->run(ports).ok());
    program->reset();
    EXPECT_TRUE(std::isfinite(static_cast<float *>(owners[6].value.contents)[0]));
    [device release];
  }
}

TEST(GgmlMetalProgram, OriginalRecurrentRealModelShapeAndUnlabelledDomain) {
  run_case(17, 16, 128);
  run_case(1, 16, 128);
  run_case(5, 2, 32);
}

TEST(GgmlMetalProgram, EveryCatalogFormatRunsPackedMatMulAndReplays) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    ASSERT_NE(device, nil);
    constexpr int k = 256;
    constexpr int n = 8;
    constexpr int m = 2;
    constexpr std::array types{
        c::GgufQuantizedType::kQ3K,   c::GgufQuantizedType::kQ4K,
        c::GgufQuantizedType::kQ5K,   c::GgufQuantizedType::kQ6K,
        c::GgufQuantizedType::kQ8_0,  c::GgufQuantizedType::kIq3Xxs,
        c::GgufQuantizedType::kIq3S,  c::GgufQuantizedType::kIq4Nl,
        c::GgufQuantizedType::kIq4Xs, c::GgufQuantizedType::kMxfp4};
    std::vector<float> weights(n * k);
    for (std::size_t i = 0; i < weights.size(); ++i)
      weights[i] = 0.25F * std::sin(0.031F * static_cast<float>(i + 1));
    for (const auto type : types) {
      SCOPED_TRACE(static_cast<int>(type));
      const auto weight =
          c::gguf_weight_description(type, c::DataType::kFloat32, k, n)
              .value();
      const std::array descriptions{
          c::TensorDesc::dense(c::DataType::kFloat32,
                               std::array<std::int64_t, 2>{m, k})
              .value(),
          weight};
      auto program =
          metal::NativeProgram::create(c::Operation(c::MatMul{}), descriptions);
      ASSERT_TRUE(program) << program.error().message();
      auto packed = quantize(type, weights, n, k,
                             weight.required_bytes().value());
      const auto decoded = dequantize(type, packed, n, k);
      std::vector<float> activation(m * k);
      std::array<Buffer, 3> owners;
      std::array<metal::BufferView, 3> ports;
      const std::array bytes{activation.size() * sizeof(float), packed.size(),
                             (*program)->output_bytes()};
      for (std::size_t i = 0; i < owners.size(); ++i) {
        owners[i].value = [device newBufferWithLength:bytes[i]
                                             options:MTLResourceStorageModeShared];
        ASSERT_NE(owners[i].value, nil);
        ports[i] = {(__bridge void *)owners[i].value, 0};
      }
      std::memcpy(owners[1].value.contents, packed.data(), packed.size());
      for (int replay = 1; replay <= 2; ++replay) {
        for (std::size_t i = 0; i < activation.size(); ++i)
          activation[i] = replay *
                          (0.2F * std::cos(0.017F * static_cast<float>(i + 3)));
        std::memcpy(owners[0].value.contents, activation.data(), bytes[0]);
        ASSERT_TRUE((*program)->run(ports).ok());
        const auto *actual = static_cast<const float *>(owners[2].value.contents);
        for (int row = 0; row < m; ++row)
          for (int column = 0; column < n; ++column) {
            double expected = 0;
            for (int inner = 0; inner < k; ++inner)
              expected += double(activation[row * k + inner]) *
                          decoded[column * k + inner];
            EXPECT_NEAR(actual[row * n + column], expected, 2.0e-4)
                << row << ' ' << column;
          }
      }
      auto invalid = ports;
      invalid[1].offset = 1;
      EXPECT_TRUE((*program)->run(invalid).IsInvalidArgument());
      EXPECT_TRUE((*program)->run(ports).ok());
    }
    ggml_quantize_free();
    [device release];
  }
}

TEST(GgmlMetalProgram,
     TargetEmbeddingFormatsUseNativeGetRowsAndCheckBoundsBeforeSubmit) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    ASSERT_NE(device, nil);
    constexpr int rows = 3;
    constexpr int width = 256;
    constexpr std::array types{c::GgufQuantizedType::kQ4K,
                               c::GgufQuantizedType::kQ6K,
                               c::GgufQuantizedType::kQ8_0};
    const auto ids_desc =
        c::TensorDesc::dense(c::DataType::kInt32,
                             std::array<std::int64_t, 1>{2})
            .value();
    std::vector<float> weights(rows * width);
    for (std::size_t i = 0; i < weights.size(); ++i)
      weights[i] = 0.125F * std::sin(0.023F * static_cast<float>(i + 11));

    for (const auto type : types) {
      SCOPED_TRACE(static_cast<int>(type));
      const auto stored =
          c::gguf_weight_description(type, c::DataType::kFloat32, width, rows)
              .value();
      const auto table_desc = stored.transposed().value();
      const std::array descriptions{table_desc, ids_desc};
      auto program =
          metal::NativeProgram::create(c::Operation(c::Gather{}), descriptions);
      ASSERT_TRUE(program) << program.error().message();
      auto packed = quantize(type, weights, rows, width,
                             table_desc.required_bytes().value());
      const auto decoded = dequantize(type, packed, rows, width);
      std::array<std::int32_t, 2> ids{2, 0};
      std::array<Buffer, 3> owners;
      std::array<metal::BufferView, 3> ports;
      const std::array bytes{packed.size(), ids.size() * sizeof(std::int32_t),
                             (*program)->output_bytes()};
      for (std::size_t i = 0; i < owners.size(); ++i) {
        owners[i].value = [device newBufferWithLength:bytes[i]
                                             options:MTLResourceStorageModeShared];
        ASSERT_NE(owners[i].value, nil);
        ports[i] = {(__bridge void *)owners[i].value, 0};
      }
      std::memcpy(owners[0].value.contents, packed.data(), packed.size());
      for (int replay = 0; replay < 2; ++replay) {
        std::memcpy(owners[1].value.contents, ids.data(), bytes[1]);
        ASSERT_TRUE((*program)->run(ports).ok());
        const auto *actual = static_cast<const float *>(owners[2].value.contents);
        for (std::size_t token = 0; token < ids.size(); ++token)
          for (int column = 0; column < width; ++column) {
            const auto expected =
                decoded[static_cast<std::size_t>(ids[token]) * width + column];
            const auto tolerance =
                type == c::GgufQuantizedType::kQ4K
                    // The pinned Metal Q4_K shader reconstructs scales at its
                    // native precision.  An independent decode measured a
                    // 0.4% relative bound plus a 2e-4 near-zero error floor;
                    // Q6_K and Q8_0 retain their stricter comparison below.
                    ? std::abs(expected) * 0.004F + 2.0e-4F
                    : 1.0e-6F;
            EXPECT_NEAR(actual[token * width + column], expected, tolerance)
                << token << ' ' << column;
          }
        ids = {1, 2};
      }

      std::vector<std::byte> before(bytes[2]);
      std::memcpy(before.data(), owners[2].value.contents, before.size());
      ids = {0, rows};
      std::memcpy(owners[1].value.contents, ids.data(), bytes[1]);
      EXPECT_TRUE((*program)->run(ports).IsOutOfRange());
      EXPECT_EQ(std::memcmp(before.data(), owners[2].value.contents, before.size()),
                0);
      ids = {2, 0};
      std::memcpy(owners[1].value.contents, ids.data(), bytes[1]);
      EXPECT_TRUE((*program)->run(ports).ok());
    }
    ggml_quantize_free();
    [device release];
  }
}

TEST(GgmlMetalProgram, PackedIq4XsGroupedExpertsUseNativeMulMatId) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    ASSERT_NE(device, nil);
    constexpr int experts = 4;
    constexpr int n = 8;
    constexpr int k = 256;
    constexpr int tokens = 2;
    constexpr int slots = 2;
    const auto weight = c::gguf_grouped_weight_description(
                            c::GgufQuantizedType::kIq4Xs,
                            c::DataType::kFloat32, experts, n, k)
                            .value();
    const std::array descriptions{
        weight,
        c::TensorDesc::dense(
            c::DataType::kFloat32,
            std::array<std::int64_t, 3>{tokens, slots, k})
            .value(),
        c::TensorDesc::dense(c::DataType::kInt32,
                             std::array<std::int64_t, 2>{tokens, slots})
            .value()};
    auto program = metal::NativeProgram::create(c::model_ops::grouped_matmul(),
                                                 descriptions);
    ASSERT_TRUE(program) << program.error().message();
    std::vector<float> weights(experts * n * k);
    for (std::size_t i = 0; i < weights.size(); ++i)
      weights[i] = 0.2F * std::sin(0.019F * static_cast<float>(i + 7));
    auto packed = quantize(c::GgufQuantizedType::kIq4Xs, weights, experts * n,
                           k, weight.required_bytes().value());
    const auto decoded = dequantize(c::GgufQuantizedType::kIq4Xs, packed,
                                    experts * n, k);
    std::vector<float> activation(tokens * slots * k);
    for (std::size_t i = 0; i < activation.size(); ++i)
      activation[i] = 0.15F * std::cos(0.013F * static_cast<float>(i + 5));
    const std::array<std::int32_t, tokens * slots> ids{0, 3, 1, 2};
    std::array<Buffer, 4> owners;
    std::array<metal::BufferView, 4> ports;
    const std::array bytes{packed.size(), activation.size() * sizeof(float),
                           ids.size() * sizeof(std::int32_t),
                           (*program)->output_bytes()};
    for (std::size_t i = 0; i < owners.size(); ++i) {
      owners[i].value = [device newBufferWithLength:bytes[i]
                                           options:MTLResourceStorageModeShared];
      ASSERT_NE(owners[i].value, nil);
      ports[i] = {(__bridge void *)owners[i].value, 0};
    }
    std::memcpy(owners[0].value.contents, packed.data(), packed.size());
    std::memcpy(owners[1].value.contents, activation.data(), bytes[1]);
    std::memcpy(owners[2].value.contents, ids.data(), bytes[2]);
    for (int replay = 0; replay < 2; ++replay) {
      ASSERT_TRUE((*program)->run(ports).ok());
      const auto *actual = static_cast<const float *>(owners[3].value.contents);
      for (int token = 0; token < tokens * slots; ++token)
        for (int column = 0; column < n; ++column) {
          double expected = 0;
          const auto row = (ids[token] * n + column) * k;
          for (int inner = 0; inner < k; ++inner)
            expected += double(activation[token * k + inner]) *
                        decoded[row + inner];
          EXPECT_NEAR(actual[token * n + column], expected, 2.0e-4)
              << token << ' ' << column;
        }
    }
    ggml_quantize_free();
    [device release];
  }
}

TEST(GgmlMetalProgram, DenseBfloatGroupedExpertsUseNativeMulMatIdAndReplay) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    ASSERT_NE(device, nil);
    constexpr int experts = 3;
    constexpr int n = 8;
    constexpr int k = 64;
    constexpr int tokens = 1;
    constexpr int slots = 2;
    const std::array descriptions{
        c::TensorDesc::dense(c::DataType::kBFloat16,
                             std::array<std::int64_t, 3>{experts, n, k})
            .value(),
        c::TensorDesc::dense(c::DataType::kFloat32,
                             std::array<std::int64_t, 3>{tokens, slots, k})
            .value(),
        c::TensorDesc::dense(c::DataType::kInt32,
                             std::array<std::int64_t, 2>{tokens, slots})
            .value()};
    auto program = metal::NativeProgram::create(c::model_ops::grouped_matmul(),
                                                 descriptions);
    ASSERT_TRUE(program) << program.error().message();

    constexpr std::array<float, experts * n> row_values{
        1.0F,  2.0F,  3.0F,  4.0F,  5.0F,  6.0F,  7.0F, 8.0F,
        0.5F,  1.0F,  1.5F,  2.0F,  2.5F,  3.0F,  3.5F, 4.0F,
        -1.0F, -2.0F, -3.0F, -4.0F, -5.0F, -6.0F, -7.0F, -8.0F};
    std::vector<std::uint16_t> weights(experts * n * k);
    for (std::size_t row = 0; row < row_values.size(); ++row)
      std::fill_n(weights.begin() + row * k, k, bfloat16(row_values[row]));
    std::vector<float> activations(tokens * slots * k, 0.25F);
    constexpr std::array<std::int32_t, tokens * slots> ids{0, 2};
    std::array<Buffer, 4> owners;
    std::array<metal::BufferView, 4> ports;
    const std::array bytes{weights.size() * sizeof(std::uint16_t),
                           activations.size() * sizeof(float),
                           ids.size() * sizeof(std::int32_t),
                           (*program)->output_bytes()};
    for (std::size_t i = 0; i < owners.size(); ++i) {
      owners[i].value = [device newBufferWithLength:bytes[i]
                                           options:MTLResourceStorageModeShared];
      ASSERT_NE(owners[i].value, nil);
      ports[i] = {(__bridge void *)owners[i].value, 0};
    }
    std::memcpy(owners[0].value.contents, weights.data(), bytes[0]);
    std::memcpy(owners[2].value.contents, ids.data(), bytes[2]);
    for (int replay = 1; replay <= 2; ++replay) {
      std::fill(activations.begin(), activations.end(), 0.25F * replay);
      std::memcpy(owners[1].value.contents, activations.data(), bytes[1]);
      const auto status = (*program)->run(ports);
      ASSERT_TRUE(status.ok()) << status.message();
      const auto *actual = static_cast<const float *>(owners[3].value.contents);
      for (int column = 0; column < n; ++column) {
        EXPECT_NEAR(actual[column], row_values[column] * k * 0.25F * replay,
                    1.0e-4F)
            << column;
        EXPECT_NEAR(actual[n + column], row_values[2 * n + column] * k *
                                                0.25F * replay,
                    1.0e-4F)
            << column;
      }
    }
    [device release];
  }
}

TEST(GgmlMetalProgram, BfloatRouterLinearRetainsNativeF32Score) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    ASSERT_NE(device, nil);
    const std::array descriptions{
        c::TensorDesc::dense(c::DataType::kBFloat16,
                             std::array<std::int64_t, 2>{1, 2})
            .value(),
        c::TensorDesc::dense(c::DataType::kBFloat16,
                             std::array<std::int64_t, 2>{2, 2})
            .value()};
    auto program = metal::NativeProgram::create(
        c::model_ops::bfloat_linear_f32(), descriptions);
    ASSERT_TRUE(program) << program.error().message();
    const std::array<std::uint16_t, 2> activation{
        bfloat16(1.0F), bfloat16(0.001953125F)};
    const std::array<std::uint16_t, 4> weights{
        bfloat16(1.0F), bfloat16(1.0F),
        bfloat16(1.0F), bfloat16(0.0F)};
    std::array<Buffer, 3> owners;
    std::array<metal::BufferView, 3> ports;
    const std::array bytes{sizeof(activation), sizeof(weights),
                           (*program)->output_bytes()};
    for (std::size_t i = 0; i < owners.size(); ++i) {
      owners[i].value = [device newBufferWithLength:bytes[i]
                                           options:MTLResourceStorageModeShared];
      ASSERT_NE(owners[i].value, nil);
      ports[i] = {(__bridge void *)owners[i].value, 0};
    }
    std::memcpy(owners[0].value.contents, activation.data(), bytes[0]);
    std::memcpy(owners[1].value.contents, weights.data(), bytes[1]);
    const auto status = (*program)->run(ports);
    ASSERT_TRUE(status.ok()) << status.message();
    const auto *scores = static_cast<const float *>(owners[2].value.contents);
    EXPECT_NEAR(scores[0], 1.001953125F, 1.0e-6F);
    EXPECT_NEAR(scores[1], 1.0F, 1.0e-6F);
    [device release];
  }
}

TEST(GgmlMetalProgram, DeclinesUnqualifiedShapesAndMetadataOomIsRecoverable) {
  auto inputs = schema(1, 2, 96);
  EXPECT_TRUE(metal::NativeProgram::validate(inputs).IsUnimplemented());
  inputs = schema(1, 2, 32);
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(), 0);
  auto failed = metal::NativeProgram::create(inputs, quota.get());
  ASSERT_FALSE(failed);
  EXPECT_TRUE(failed.error().IsResourceExhausted());
  EXPECT_EQ(quota.used(), 0u);
}
} // namespace
