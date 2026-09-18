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
  auto weight = make(context, scale_desc, scale);
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
    std::size_t warmed_bytes = 0;
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
      // Original ggml lazily allocates its per-work auxiliary buffers. Only
      // the third run reuses an already warmed slot; all allocations are
      // charged.
      if (phase == 1)
        warmed_bytes = quota.used();
      else if (phase > 1)
        EXPECT_EQ(quota.used(), warmed_bytes);
    }
    EXPECT_FALSE(first.submit());
    retained = {};
    auto replay = first.submit();
    ASSERT_TRUE(replay) << replay.error().message();
    ASSERT_TRUE(replay->wait());
  }
  EXPECT_TRUE(context.close().ok());
  weight = {};
  context = {};
  EXPECT_EQ(quota.used(), 0U);
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

TEST(VulkanManaged, PointwiseFamilyUsesRealProviderWithoutNormAnchor) {
  auto context = vulkan_test::context().value();
  const std::array kinds{c::UnaryKind::kAbs, c::UnaryKind::kNeg,
                         c::UnaryKind::kExp, c::UnaryKind::kTanh,
                         c::UnaryKind::kSigmoid, c::UnaryKind::kSilu};
  const std::array descriptions{
      dense(c::DataType::kFloat32, {}),
      dense(c::DataType::kFloat32, {129}),
      dense(c::DataType::kFloat32, {17, 257}),
      dense(c::DataType::kFloat32, {2, 1, 3, 43})};
  const std::array<double, 12> edges{-0.0, NAN, -INFINITY, INFINITY,
                                     0.0, -90, 90, -1e-7, 1e-7, -1, 1, 3};
  for (auto kind : kinds) {
    for (const auto &desc : descriptions) {
      SCOPED_TRACE(::testing::Message()
                   << static_cast<int>(kind) << '/' << desc.rank());
      auto program = v::prepare(context, c::Unary{kind}, desc).value();
      const auto &plan = d::Access::program(program);
      ASSERT_EQ(plan->dispatches.size(), 2U);
      ASSERT_TRUE(plan->dispatches.back().region);
      auto first = program.create_invocation().value();
      auto second = program.create_invocation().value();
      v::Tensor retained;
      std::vector<double> saved;
      for (std::size_t phase = 0; phase < 4; ++phase) {
        std::vector<double> data(*desc.element_count());
        for (std::size_t i = 0; i < data.size(); ++i)
          data[i] = i < edges.size()
                        ? edges[(i + phase) % edges.size()]
                        : (static_cast<int>((i + phase * 7) % 41) - 20) * 0.125;
        auto input = make(context, desc, data);
        const auto original = bits(context, input);
        std::vector<double> expected;
        for (double value : data)
          expected.push_back(vulkan_tensor_test::unary(kind, value));
        auto &slot = phase == 0 ? first : second;
        ASSERT_TRUE(slot.bind(input).ok());
        auto work = slot.submit().value();
        auto result = work.wait().value();
        const bool exact = kind == c::UnaryKind::kAbs || kind == c::UnaryKind::kNeg;
        expect(context, result.get()->tensors()[0], expected, exact);
        EXPECT_EQ(bits(context, input), original);
        if (phase == 0) {
          retained = result.get()->tensors()[0];
          saved = expected;
        } else {
          expect(context, retained, saved, exact);
        }
      }
    }
  }
}

TEST(VulkanManaged, SubtractDividePreserveBothBroadcastOrdersAndScalar) {
  auto context = vulkan_test::context().value();
  constexpr std::size_t rows = 17, width = 129;
  const auto full = dense(c::DataType::kFloat32, {rows, width});
  const std::array parameters{dense(c::DataType::kFloat32, {}),
                              dense(c::DataType::kFloat32, {width}),
                              dense(c::DataType::kFloat32, {rows, 1})};
  for (auto kind : {c::BinaryKind::kSubtract, c::BinaryKind::kDivide}) {
    for (const auto &parameter : parameters) {
      for (bool small_first : {false, true}) {
        SCOPED_TRACE(::testing::Message()
                     << static_cast<int>(kind) << '/' << parameter.rank() << '/'
                     << small_first);
        std::vector<double> data(rows * width), small(*parameter.element_count());
        for (std::size_t i = 0; i < data.size(); ++i)
          data[i] = 0.25 + (i % 19) * 0.125;
        for (std::size_t i = 0; i < small.size(); ++i)
          small[i] = 2.0 + (i % 7) * 0.25;
        if (small.size() > 4) {
          small[0] = 0.0;
          small[1] = -0.0;
          small[2] = INFINITY;
          small[3] = NAN;
        }
        const auto input = make(context, full, data);
        const auto broadcast = make(context, parameter, small);
        const std::array arguments = small_first ? std::array{broadcast, input}
                                                : std::array{input, broadcast};
        auto program = v::prepare(context, c::Binary{kind},
                                  arguments[0].description(),
                                  arguments[1].description()).value();
        ASSERT_TRUE(d::Access::program(program)->dispatches.back().region);
        auto slot = program.create_invocation().value();
        ASSERT_TRUE(slot.bind(arguments).ok());
        std::vector<double> expected(data.size());
        for (std::size_t i = 0; i < data.size(); ++i) {
          const auto index = parameter.rank() == 0 ? 0
                             : parameter.rank() == 1 ? i % width : i / width;
          const double left = small_first ? small[index] : data[i];
          const double right = small_first ? data[i] : small[index];
          expected[i] = kind == c::BinaryKind::kSubtract ? left - right
                                                       : left / right;
        }
        {
          auto work = slot.submit().value();
          auto result = work.wait().value();
          expect(context, result.get()->tensors()[0], expected);
        }
        const auto output = context.allocate(full).value();
        auto recorder = v::Recorder::create(context).value();
        ASSERT_TRUE(recorder.append(program, arguments, std::array{output}).ok());
        auto recorded = recorder.finish().value();
        for (int repeat = 0; repeat < 2; ++repeat) {
          auto completion = recorded.submit().value();
          ASSERT_TRUE(completion.wait().ok());
          expect(context, output, expected);
        }
        expect(context, input, data, true);
        expect(context, broadcast, small, true);
      }
    }
  }
}
} // namespace
