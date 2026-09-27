#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

#include "../compute_models/d5_quantization_roundtrip.h"
#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/cpu.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/native.h"
#include "hyle/mem/quota_allocator.h"
#include "hyle/mem/stats.h"
#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
#include "hyle/graph/cpu.h"
#endif

namespace {
namespace c = hyle::compute;
namespace mem = hyle::mem;
using Format = c::model_ops::BlockQuantizationFormat;

void run_roundtrip(bool fp4, std::vector<std::uint16_t> values) {
  const auto description = c::TensorDesc::dense(
      c::DataType::kBFloat16,
      std::array<std::int64_t, 2>{1, static_cast<std::int64_t>(values.size())}).value();
  const auto operation = c::model_ops::block_quantization_roundtrip(
      fp4 ? Format::kFloat4E2M1Block32 : Format::kFloat8E4M3Block64);
  auto input_storage = half_test::buffer<std::uint16_t>(values);
  auto output_storage = half_test::buffer<std::uint16_t>(values);
  const auto input = c::ConstTensorView::create(description, input_storage.view()).value();
  const auto output = c::TensorView::create(description, output_storage.view()).value();
  auto stats = mem::StatsAllocator::create(mem::system_allocator()).value();
  {
#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
    namespace g = hyle::graph;
    auto context = g::cpu::create_context({.metadata_allocator = stats.get()}).value();
    auto builder = g::GraphBuilder::create().value();
    auto x = builder.input(description).value();
    auto y = builder.add(operation, x).value();
    ASSERT_TRUE(builder.publish(y).ok());
    auto compiled = g::compile(std::move(builder).finish().value(), context).value();
    ASSERT_EQ(compiled.report.step_count(), 1U);
    EXPECT_EQ(compiled.report.step(0)->provider.id, c::native::kProviderId);
    auto invocation = compiled.executable.create_invocation().value();
    ASSERT_TRUE(invocation.bind(x, input).ok());
    ASSERT_TRUE(invocation.bind(y, output).ok());
    auto queue = context.create_queue().value();
    const std::array queues{g::QueueBinding{.lane = g::ExecutionLaneId{0}, .queue = &queue}};
    ASSERT_TRUE(invocation.prepare_replay(queues).ok());
    auto submit = [&] { return invocation.submit(queue); };
#else
    auto context = c::cpu::create_context({.metadata_allocator = stats.get()}).value();
    const std::array descriptions{description};
    auto prepared = c::prepare(context, operation, descriptions, descriptions,
                               mem::system_resource()).value();
    EXPECT_EQ(prepared.provider().id, c::native::kProviderId);
    EXPECT_EQ(prepared.workspace_requirement().size, 0U);
    auto invocation = prepared.create_invocation().value();
    auto queue = context.create_queue().value();
    auto submit = [&] { return invocation.submit(queue, input, output); };
    auto alias = c::TensorView::create(description, input_storage.view()).value();
    auto rejected = invocation.submit(queue, input, alias);
    ASSERT_FALSE(rejected);
    EXPECT_TRUE(rejected.error().IsInvalidArgument());
#endif
    for (int replay = 0; replay < 2; ++replay) {
      std::vector<float> decoded;
      for (auto value : values)
        decoded.push_back(std::bit_cast<float>(std::uint32_t(value) << 16));
      const auto expected = d5_quantization_roundtrip::reference(decoded, fp4);
      ASSERT_TRUE(input_storage.view().write(std::as_bytes(std::span(values))).ok());
      auto done = submit();
      ASSERT_TRUE(done) << done.error().message();
      ASSERT_TRUE(done->wait().ok());
      const auto actual = half_test::read<std::uint16_t>(output_storage);
      ASSERT_EQ(actual.size(), expected.size());
      for (std::size_t i = 0; i < actual.size(); ++i)
        EXPECT_EQ(std::bit_cast<float>(std::uint32_t(actual[i]) << 16), expected[i]) << i;
      std::ranges::reverse(values);
    }
    // An accepted terminal failure retires the invocation, not its program.
    values[0] = 0x7fc0;
    ASSERT_TRUE(input_storage.view().write(std::as_bytes(std::span(values))).ok());
    {
      auto done = submit();
      ASSERT_TRUE(done);
      EXPECT_TRUE(done->wait().IsInvalidArgument());
    }
    values[0] = 0;
    ASSERT_TRUE(input_storage.view().write(std::as_bytes(std::span(values))).ok());
    auto retired = submit();
    ASSERT_FALSE(retired);
    EXPECT_TRUE(retired.error().IsFailedPrecondition());
#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
    invocation = compiled.executable.create_invocation().value();
    ASSERT_TRUE(invocation.bind(x, input).ok());
    ASSERT_TRUE(invocation.bind(y, output).ok());
    ASSERT_TRUE(invocation.prepare_replay(queues).ok());
#else
    invocation = prepared.create_invocation().value();
#endif
    auto recovered = submit();
    ASSERT_TRUE(recovered) << recovered.error().message();
    EXPECT_TRUE(recovered->wait().ok());
  }
  EXPECT_TRUE(stats.balanced());
}

TEST(CpuDs4Roundtrip, OrdinaryEntryExhaustsFiniteBfloatAndReplaysBothFormats) {
  std::vector<std::uint16_t> values;
  for (unsigned bits = 0; bits < 65536; ++bits)
    if ((bits & 0x7f80) != 0x7f80)
      values.push_back(static_cast<std::uint16_t>(bits));
  for (bool fp4 : {false, true}) {
    SCOPED_TRACE(fp4);
    run_roundtrip(fp4, values);
  }
}

TEST(CpuDs4Roundtrip, MatureMidpointsAndLowScalesKeepExactNumericalBoundary) {
  std::vector<std::uint16_t> values;
  for (float cap : {6.0F, 448.0F})
    for (int exponent : {-126, -125, -10, 0, 10})
      for (float value : {-.75F, -.25F, .25F, .75F, 1.25F, 1.75F, 2.5F, 3.5F, 5.F})
        for (unsigned i = 0; i < 64; ++i) {
          const float x = std::ldexp(i == 0 ? cap : value, exponent);
          values.push_back(half_test::encode<hyle::simd::bfloat16_t>(x).bits);
        }
  for (bool fp4 : {false, true})
    run_roundtrip(fp4, values);
}

TEST(CpuDs4Roundtrip, RejectsUnavailableGroupAndRecoversAfterColdOom) {
  const auto description = c::TensorDesc::dense(
      c::DataType::kBFloat16, std::array<std::int64_t, 2>{1, 128}).value();
  mem::QuotaAllocator quota(mem::system_allocator(), 4 * 1024 * 1024);
  auto context = c::cpu::create_context({.metadata_allocator = quota.get()}).value();
  const std::array descriptions{description};
  auto unsupported = c::prepare(context, c::model_ops::block_quantization_roundtrip(
      Format::kFloat8E4M3Block128), descriptions, descriptions, mem::system_resource());
  ASSERT_FALSE(unsupported);
  EXPECT_TRUE(unsupported.error().IsUnimplemented());
  const auto operation = c::model_ops::block_quantization_roundtrip(Format::kFloat4E2M1Block32);
  {
    auto reserved = quota.try_reserve_bytes(quota.remaining());
    ASSERT_TRUE(reserved);
    auto denied = c::prepare(context, operation, descriptions, descriptions,
                             mem::system_resource());
    ASSERT_FALSE(denied);
    EXPECT_TRUE(denied.error().IsResourceExhausted());
  }
  EXPECT_TRUE(c::prepare(context, operation, descriptions, descriptions,
                         mem::system_resource()));
}
} // namespace
