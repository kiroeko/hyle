#include <gtest/gtest.h>

#include "gate_norm_test_support.h"
#include "vulkan_program_internal.h"

namespace {
using namespace vulkan_gate_norm_test;
namespace d = v::detail;

TEST(VulkanManaged, DirectNormRebindsIndependentSlotsAndRetainsOldResults) {
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(),
                                  64 * 1024 * 1024);
  auto context = vulkan_test::context(&quota).value();
  constexpr int width = 2048;
  const auto scale_desc = dense(c::DataType::kFloat32, {width});
  std::vector<double> scale(width);
  for (int i = 0; i < width; ++i)
    scale[i] = 1 + (i % 7 - 3) * 0.0625;
  const auto weight = make(context, scale_desc, scale);
  for (int rows : {1, 17, 128}) {
    SCOPED_TRACE(rows);
    const auto desc = dense(c::DataType::kFloat32, {rows, width});
    auto prepared = v::prepare(context, c::RmsNorm{}, desc, weight);
    ASSERT_TRUE(prepared) << prepared.error().message();
    const auto &plan = d::Access::program(*prepared);
    ASSERT_EQ(plan->dispatches.size(), 2U); // Origin adapter + original region.
    ASSERT_TRUE(plan->dispatches.back().region);
    auto first = prepared->create_invocation().value();
    auto second = prepared->create_invocation().value();
    EXPECT_FALSE(first.submit());
    std::array<v::Tensor, 3> inputs;
    std::array<std::vector<double>, 3> expected;
    for (std::size_t phase = 0; phase < inputs.size(); ++phase) {
      std::vector<double> data(static_cast<std::size_t>(rows) * width);
      for (std::size_t i = 0; i < data.size(); ++i)
        data[i] = (static_cast<int>((i + phase * 17) % 61) - 30) * 0.125;
      inputs[phase] = make(context, desc, data);
      expected[phase] = norm(data, width, false, 1e-5F, scale);
    }
    const auto idle_bytes = quota.used();
    v::Tensor retained;
    for (std::size_t phase = 0; phase < inputs.size(); ++phase) {
      auto &slot = phase == 0 ? first : second;
      ASSERT_TRUE(slot.bind(inputs[phase]).ok());
      // A rejected shape must not replace the previous successful binding.
      EXPECT_FALSE(slot.bind(weight).ok());
      auto work = slot.submit();
      ASSERT_TRUE(work) << work.error().message();
      auto result = work->wait();
      ASSERT_TRUE(result) << result.error().message();
      expect(context, result->get()->tensors()[0], expected[phase]);
      if (phase == 0)
        retained = result->get()->tensors()[0];
      else
        expect(context, retained, expected[0]);
      EXPECT_EQ(quota.used(), idle_bytes); // HYLE boundary storage is reused.
    }
    EXPECT_FALSE(first.submit());
    retained = {};
    auto replay = first.submit();
    ASSERT_TRUE(replay) << replay.error().message();
    ASSERT_TRUE(replay->wait());
  }
  EXPECT_TRUE(context.close().ok());
}

TEST(VulkanManaged, ByteUnalignedDenseViewsAndFixedRecorderUseFrozenAdapters) {
  auto context = vulkan_test::context().value();
  constexpr std::int64_t width = 129;
  const auto desc = dense(c::DataType::kFloat32, {width});
  const auto parent_desc =
      c::TensorDesc::strided_bytes(c::DataType::kFloat32,
                                   std::array<std::int64_t, 1>{4 * width + 4},
                                   std::array<std::int64_t, 1>{1})
          .value();
  auto program = v::prepare(context, c::RmsNorm{}, desc).value();
  auto slot = program.create_invocation().value();
  for (std::int64_t origin : {1, 3, 4}) {
    SCOPED_TRACE(origin);
    std::vector<std::byte> payload(*parent_desc.required_bytes(),
                                   std::byte{0xcd});
    std::vector<double> data(width);
    for (std::size_t i = 0; i < data.size(); ++i) {
      const float value =
          static_cast<float>(static_cast<int>(i % 29) - 14) * 0.25F;
      data[i] = value;
      std::memcpy(payload.data() + origin + i * sizeof(float), &value,
                  sizeof(value));
    }
    const auto parent = context.upload(parent_desc, payload).value();
    const auto input =
        parent.view(c::View::slice(0, origin, width, 4).value()).value();
    ASSERT_EQ(input.description(), desc);
    ASSERT_EQ(input.native_offset(), origin);
    const auto expected = norm(data, width, false, 1e-5F);
    {
      ASSERT_TRUE(slot.bind(input).ok());
      auto work = slot.submit().value();
      auto result = work.wait().value();
      expect(context, result.get()->tensors()[0], expected);
    }
    auto captured = v::prepare(context, c::RmsNorm{}, input).value();
    auto captured_slot = captured.create_invocation().value();
    auto work = captured_slot.submit().value();
    auto result = work.wait().value();
    expect(context, result.get()->tensors()[0], expected);
    const auto output = context.allocate(desc).value();
    auto recorder = v::Recorder::create(context).value();
    ASSERT_TRUE(
        recorder.append(program, std::array{input}, std::array{output}).ok());
    auto recorded = recorder.finish().value();
    for (int repeat = 0; repeat < 2; ++repeat) {
      auto completion = recorded.submit().value();
      ASSERT_TRUE(completion.wait().ok());
      expect(context, output, expected);
    }
    EXPECT_EQ(bytes(context, parent), payload);
  }
}
} // namespace
