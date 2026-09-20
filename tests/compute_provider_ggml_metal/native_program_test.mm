#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../../src/compute_provider_ggml_metal/native_program.h"
#include "hyle/mem/quota_allocator.h"

namespace {
namespace c = hyle::compute;
namespace metal = c::ggml_metal;

struct Buffer final {
  id<MTLBuffer> value = nil;
  Buffer() = default;
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  ~Buffer() { [value release]; }
};

std::array<c::TensorDesc, 6> schema(int tokens, int heads, int depth) {
  auto tensor = [](std::array<std::int64_t, 4> shape) {
    return c::TensorDesc::dense(c::DataType::kFloat32, shape).value();
  };
  const auto q = tensor({2, tokens, heads, depth});
  const auto gate = tensor({2, tokens, heads, 1});
  return {q, q, q, gate, gate, tensor({2, heads, depth, depth})};
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
