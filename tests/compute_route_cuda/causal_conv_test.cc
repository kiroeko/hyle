#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/cuda.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/cuda.h"
#include "hyle/compute/providers/ggml_cuda.h"

namespace {
namespace c = hyle::compute;
namespace mem = hyle::mem;

struct Shape final {
  std::int64_t batch, window, channels, taps;
};
std::array<c::TensorDesc, 3> descriptions(Shape shape,
    c::DataType type = c::DataType::kFloat32) {
  return {
      c::TensorDesc::dense(type, std::array{shape.batch, shape.window, shape.channels}).value(),
      c::TensorDesc::dense(type, std::array{shape.channels, shape.taps}).value(),
      c::TensorDesc::dense(type, std::array{shape.batch, shape.window - shape.taps + 1, shape.channels}).value()};
}

void exercise(c::Context &context, mem::MemoryResourceRef resource, Shape shape) {
  const auto desc = descriptions(shape);
  auto prepared = c::prepare(context, c::model_ops::causal_conv1d(),
      std::span(desc).first<2>(), std::span(desc).last<1>(), resource);
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, c::cuda_native::kProviderId);
  EXPECT_EQ(prepared->workspace_requirement().size, 0U);
  EXPECT_EQ(prepared->invocation_private_bytes(), 0U);
  EXPECT_TRUE(prepared->capture_compatible());
  const auto runtime = prepared->runtime_properties();
  EXPECT_EQ(runtime.mode, c::ProviderRuntimeMode::kSharedQueue);
  EXPECT_EQ(runtime.replay_class, c::ProviderReplayClass::kPrepared);
  EXPECT_EQ(runtime.workspace_owner, c::ProviderWorkspaceOwner::kCaller);
  EXPECT_TRUE(runtime.supports_concurrent_invocations);
  EXPECT_FALSE(runtime.blocks_host);
  EXPECT_FALSE(runtime.uses_private_threads);
  for (std::size_t input = 0; input < 2; ++input) {
    auto alias = prepared->alias_support(0, input);
    ASSERT_TRUE(alias) << alias.error().message();
    EXPECT_FALSE(alias->supported);
  }
  auto invocation = prepared->create_invocation();
  ASSERT_TRUE(invocation) << invocation.error().message();
  auto queue = context.create_queue();
  ASSERT_TRUE(queue) << queue.error().message();

  // Keep both sets live so replay genuinely rebinds all three allocations.
  std::array<std::array<mem::Buffer, 3>, 2> buffers;
  for (auto &set : buffers) for (std::size_t index = 0; index < set.size(); ++index)
    set[index] = mem::Buffer::create(resource, desc[index].required_bytes().value(), 256).value();
  const auto batch = static_cast<std::size_t>(shape.batch);
  const auto window = static_cast<std::size_t>(shape.window);
  const auto channels = static_cast<std::size_t>(shape.channels);
  const auto taps = static_cast<std::size_t>(shape.taps);
  const auto tokens = window - taps + 1;
  std::array<std::vector<float>, 2> observed;
  for (std::size_t repeat = 0; repeat < buffers.size(); ++repeat) {
    SCOPED_TRACE(::testing::Message() << "rebind " << repeat);
    auto &set = buffers[repeat];
    std::vector<float> samples(batch * window * channels), weights(channels * taps);
    for (std::size_t index = 0; index < samples.size(); ++index)
      samples[index] = static_cast<float>(static_cast<int>((index * 17 + repeat * 13) % 97) - 48) * .03721F;
    for (std::size_t index = 0; index < weights.size(); ++index)
      weights[index] = static_cast<float>(static_cast<int>((index * 11 + repeat * 7) % 41) - 20) * .06173F;
    auto status = set[0].view().write(std::as_bytes(std::span(samples)));
    ASSERT_TRUE(status.ok()) << status.message();
    status = set[1].view().write(std::as_bytes(std::span(weights)));
    ASSERT_TRUE(status.ok()) << status.message();
    const std::array inputs{
        c::ConstTensorView::create(desc[0], set[0].view()).value(),
        c::ConstTensorView::create(desc[1], set[1].view()).value()};
    const auto output = c::TensorView::create(desc[2], set[2].view()).value();
    auto done = invocation->submit(*queue, inputs, output);
    ASSERT_TRUE(done) << done.error().message();
    status = done->wait();
    ASSERT_TRUE(status.ok()) << status.message();
    EXPECT_FALSE(invocation->in_flight());
    auto &actual = observed[repeat]; actual.resize(batch * tokens * channels);
    status = set[2].view().read(std::as_writable_bytes(std::span(actual)));
    ASSERT_TRUE(status.ok()) << status.message();
    for (std::size_t b = 0; b < batch; ++b)
      for (std::size_t token = 0; token < tokens; ++token)
        for (std::size_t channel = 0; channel < channels; ++channel) {
          float expected = 0.F;
          for (std::size_t tap = 0; tap < taps; ++tap)
            expected = std::fma(samples[(b * window + token + tap) * channels + channel],
                weights[channel * taps + tap], expected);
          const auto index = (b * tokens + token) * channels + channel;
          ASSERT_EQ(std::bit_cast<std::uint32_t>(actual[index]), std::bit_cast<std::uint32_t>(expected))
              << "b=" << b << " t=" << token << " c=" << channel;
        }
  }
  EXPECT_NE(observed[0], observed[1]);
}

TEST(CudaCausalConv, NativeArbitraryChannelsAndTapsRebindOnePreparedInvocation) {
  auto context = c::Context::create(c::cuda::device(0).value(), c::cuda_native::provider());
  ASSERT_TRUE(context) << context.error().message();
  auto memory = c::cuda::MemoryResource::create(*context, c::cuda::MemoryKind::kDevice);
  ASSERT_TRUE(memory) << memory.error().message();
  constexpr std::array shapes{Shape{2, 6, 192, 4}, Shape{2, 7, 7, 2}, Shape{2, 1, 1, 1}};
  for (const auto shape : shapes) {
    SCOPED_TRACE(::testing::Message() << shape.batch << '/' << shape.window << '/' << shape.channels << '/' << shape.taps);
    exercise(*context, memory->ref(), shape);
  }
}

TEST(CudaCausalConv, DefaultRoutingRetainsGgmlAndUsesNativeOutsideItsActualDomain) {
  auto context = c::cuda::create_context();
  ASSERT_TRUE(context) << context.error().message();
  auto memory = c::cuda::MemoryResource::create(*context, c::cuda::MemoryKind::kDevice);
  ASSERT_TRUE(memory) << memory.error().message();
  constexpr std::array shapes{Shape{2, 6, 128, 4}, Shape{2, 6, 192, 4}, Shape{2, 7, 7, 2}, Shape{2, 7, 128, 2}};
  for (std::size_t index = 0; index < shapes.size(); ++index) {
    SCOPED_TRACE(index);
    const auto desc = descriptions(shapes[index]);
    auto prepared = c::prepare(*context, c::model_ops::causal_conv1d(),
        std::span(desc).first<2>(), std::span(desc).last<1>(), memory->ref());
    ASSERT_TRUE(prepared) << prepared.error().message();
    EXPECT_EQ(prepared->provider().id, index == 0 ? c::ggml_cuda::kProviderId : c::cuda_native::kProviderId);
  }
}

TEST(CudaCausalConv, NativeDoesNotClaimHalfOrBfloatStorage) {
  auto context = c::Context::create(c::cuda::device(0).value(), c::cuda_native::provider());
  ASSERT_TRUE(context) << context.error().message();
  auto memory = c::cuda::MemoryResource::create(*context, c::cuda::MemoryKind::kDevice);
  ASSERT_TRUE(memory) << memory.error().message();
  for (const auto type : {c::DataType::kFloat16, c::DataType::kBFloat16}) {
    SCOPED_TRACE(static_cast<int>(type));
    const auto desc = descriptions({2, 6, 192, 4}, type);
    auto prepared = c::prepare(*context, c::model_ops::causal_conv1d(),
        std::span(desc).first<2>(), std::span(desc).last<1>(), memory->ref());
    ASSERT_FALSE(prepared);
    EXPECT_TRUE(prepared.error().IsUnimplemented()) << prepared.error().message();
  }
}
} // namespace
