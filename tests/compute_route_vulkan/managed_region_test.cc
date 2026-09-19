#include <gtest/gtest.h>

#include <algorithm>

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
  const std::array kinds{c::UnaryKind::kAbs,     c::UnaryKind::kNeg,
                         c::UnaryKind::kExp,     c::UnaryKind::kTanh,
                         c::UnaryKind::kSigmoid, c::UnaryKind::kSilu,
                         c::UnaryKind::kLog,     c::UnaryKind::kSqrt,
                         c::UnaryKind::kRsqrt,   c::UnaryKind::kSoftplus};
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
TEST(VulkanManaged, StableTransformsPreserveSmallValuesAndHighRankHalfStorage) {
  auto context = vulkan_test::context().value();
  const std::array<double, 27> edges{
      -80,  -40,       -20,      -15, -1, -1e-4, -1e-8, -0.0, 0.0,
      1e-8, 1e-4,      0.5,      1,   2,  15,    20,    40,   80,
      NAN,  -INFINITY, INFINITY, -2,  3,  0.25,  -0.25, 8,    16};
  for (auto type :
       {c::DataType::kFloat32, c::DataType::kFloat16, c::DataType::kBFloat16}) {
    const auto size = static_cast<std::int64_t>(c::data_type_size(type));
    const auto desc =
        c::TensorDesc::strided_bytes(
            type, std::array<std::int64_t, 8>{1, 1, 1, 1, 1, 1, 3, 9},
            std::array<std::int64_t, 8>{0, 0, 0, 0, 0, 0, 9 * (size + 1) + 3,
                                        size + 1})
            .value();
    const auto input = make(context, desc, edges);
    const auto logical = values(context, input);
    const auto original = bytes(context, input);
    for (auto kind :
         {c::UnaryKind::kLog, c::UnaryKind::kSqrt, c::UnaryKind::kRsqrt,
          c::UnaryKind::kSoftplus, c::UnaryKind::kTanh, c::UnaryKind::kSigmoid,
          c::UnaryKind::kSilu}) {
      SCOPED_TRACE(::testing::Message()
                   << static_cast<int>(type) << '/' << static_cast<int>(kind));
      auto program = v::prepare(context, c::Unary{kind}, desc).value();
      const auto &dispatches = d::Access::program(program)->dispatches;
      ASSERT_EQ(std::ranges::count_if(dispatches,
                                      [](const auto &dispatch) {
                                        return dispatch.region != nullptr;
                                      }),
                1);
      auto slot = program.create_invocation().value();
      ASSERT_TRUE(slot.bind(input).ok());
      auto work = slot.submit().value();
      auto result = work.wait();
      ASSERT_TRUE(result) << result.error().message();
      const auto observed = values(context, result->get()->tensors()[0]);
      for (std::size_t i = 0; i < logical.size(); ++i) {
        const auto reference = rounded(unary(kind, logical[i]), type);
        if (std::isnan(reference)) {
          EXPECT_TRUE(std::isnan(observed[i])) << i;
        } else if (reference == 0 || std::isinf(reference)) {
          EXPECT_EQ(observed[i], reference) << i;
          EXPECT_EQ(std::signbit(observed[i]), std::signbit(reference)) << i;
        } else {
          const auto relative = type == c::DataType::kFloat32   ? 6e-6
                                : type == c::DataType::kFloat16 ? 0.001
                                                                : 0.008;
          const auto quantum = type == c::DataType::kFloat16    ? 0x1p-24
                               : type == c::DataType::kBFloat16 ? 0x1p-133
                                                                : 1e-43;
          EXPECT_NEAR(observed[i], reference,
                      relative * std::abs(reference) + quantum)
              << i;
        }
      }
      EXPECT_EQ(bytes(context, input), original);
    }
  }
}

TEST(VulkanManaged, CrossBroadcastAndDegenerateRankEightUseMatureArithmetic) {
  auto context = vulkan_test::context().value();
  for (auto type :
       {c::DataType::kFloat32, c::DataType::kFloat16, c::DataType::kBFloat16}) {
    for (bool high : {false, true}) {
      const auto left_desc =
          high ? dense(type, {1, 1, 1, 1, 1, 3, 1, 1}) : dense(type, {3, 1});
      const auto right_desc = high ? dense(type, {1, 1, 1, 1, 1, 1, 1, 129})
                                   : dense(type, {1, 129});
      const std::array<double, 3> rows{0.5, 2, 4};
      std::vector<double> columns(129);
      for (std::size_t i = 0; i < columns.size(); ++i)
        columns[i] = 1 + (i % 13) * 0.125;
      const auto left = make(context, left_desc, rows);
      const auto right = make(context, right_desc, columns);
      for (auto kind : {c::BinaryKind::kAdd, c::BinaryKind::kSubtract,
                        c::BinaryKind::kMultiply, c::BinaryKind::kDivide}) {
        for (bool reverse : {false, true}) {
          const auto args =
              reverse ? std::array{right, left} : std::array{left, right};
          auto program =
              v::prepare(context, c::Binary{kind}, args[0].description(),
                         args[1].description())
                  .value();
          const auto &dispatches = d::Access::program(program)->dispatches;
          ASSERT_EQ(std::ranges::count_if(dispatches,
                                          [](const auto &dispatch) {
                                            return dispatch.region != nullptr;
                                          }),
                    1);
          auto slot = program.create_invocation().value();
          ASSERT_TRUE(slot.bind(args).ok());
          auto work = slot.submit().value();
          auto result = work.wait().value();
          std::vector<double> expected;
          for (double row : rows)
            for (double column : columns) {
              const auto a = reverse ? column : row, b = reverse ? row : column;
              expected.push_back(kind == c::BinaryKind::kAdd        ? a + b
                                 : kind == c::BinaryKind::kSubtract ? a - b
                                 : kind == c::BinaryKind::kMultiply ? a * b
                                                                    : a / b);
            }
          expect(context, result.get()->tensors()[0], expected);
        }
      }
    }
    const auto singleton =
        make(context, dense(type, {1, 1, 1, 1, 1, 1, 1, 1}), std::array{4.0});
    auto result = execute(context, c::Operation(c::Unary{c::UnaryKind::kSqrt}),
                          std::array{singleton})
                      .value();
    expect(context, result, std::array{2.0}, true);
  }
}

TEST(VulkanManaged, ElementwiseDeviceLimitsRetainQualifiedFallbacks) {
  auto context = vulkan_test::context().value();
  auto &limits = d::Access::runtime(context)->device->properties.limits;
  const auto original = limits;
  limits.maxComputeWorkGroupInvocations = 256;
  auto program = v::prepare(context, c::Unary{c::UnaryKind::kSoftplus},
                            dense(c::DataType::kFloat32, {129}));
  limits = original;
  ASSERT_TRUE(program) << program.error().message();
  EXPECT_TRUE(std::ranges::all_of(
      d::Access::program(*program)->dispatches,
      [](const auto &dispatch) { return !dispatch.region; }));
}

TEST(VulkanManaged, PackedAndSeparateSwiGluRetainHighRankRowHalves) {
  auto context = vulkan_test::context().value();
  constexpr std::int64_t width = 129, rows = 3;
  for (auto type :
       {c::DataType::kFloat32, c::DataType::kFloat16, c::DataType::kBFloat16}) {
    for (auto form :
         {c::GatedInputKind::kPackedHalves, c::GatedInputKind::kSeparate}) {
      const auto columns =
          form == c::GatedInputKind::kPackedHalves ? 2 * width : width;
      const auto size = static_cast<std::int64_t>(c::data_type_size(type));
      const auto desc =
          c::TensorDesc::strided_bytes(
              type,
              std::array<std::int64_t, 8>{1, 1, 1, 1, 1, rows, 1, columns},
              std::array<std::int64_t, 8>{
                  0, 0, 0, 0, 0, columns * (size + 1) + 3, 0, size + 1})
              .value();
      std::vector<double> data(rows * columns), factors(rows * width),
          expected(rows * width);
      for (std::size_t i = 0; i < data.size(); ++i)
        data[i] = rounded((static_cast<int>(i % 31) - 15) * 0.5, type);
      for (std::size_t i = 0; i < factors.size(); ++i)
        factors[i] = rounded((static_cast<int>(i % 17) - 8) * 0.125, type);
      std::vector<v::Tensor> inputs{make(context, desc, data)};
      if (form == c::GatedInputKind::kSeparate)
        inputs.push_back(make(context, desc, factors));
      std::vector<v::Argument> arguments;
      for (const auto &input : inputs)
        arguments.emplace_back(input.description());
      auto program = v::prepare(context,
                                c::Operation(c::GatedActivation{
                                    c::GatedActivationKind::kSilu, form}),
                                std::span<const v::Argument>(arguments))
                         .value();
      ASSERT_EQ(std::ranges::count_if(
                    d::Access::program(program)->dispatches,
                    [](const auto &dispatch) { return !!dispatch.region; }),
                1);
      auto slot = program.create_invocation().value();
      ASSERT_TRUE(slot.bind(inputs).ok());
      auto work = slot.submit().value();
      auto result = work.wait().value();
      for (std::int64_t row = 0; row < rows; ++row)
        for (std::int64_t column = 0; column < width; ++column) {
          const auto index = row * width + column;
          const auto factor = form == c::GatedInputKind::kSeparate
                                  ? factors[index]
                                  : data[row * columns + width + column];
          expected[index] = activate(c::GatedActivationKind::kSilu,
                                     data[row * columns + column]) *
                            factor;
        }
      expect(context, result.get()->tensors()[0], expected);
    }
  }
}
} // namespace
