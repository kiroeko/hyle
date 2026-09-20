#include <gtest/gtest.h>

#include <optional>

#include "scan_test_support.h"
#include "vulkan_program_internal.h"

namespace {
using namespace vulkan_scan_test;

TEST(VulkanScan, NativeWidthsAndRaggedRowsMatchIndependentPrefix) {
  auto context = vulkan_test::context().value();
  for (const std::int64_t width : {1, 17, 127, 128, 129, 255, 256, 257, 1023,
                                   1024, 1025, 4096, 4097, 8193}) {
    for (const std::int64_t rows : {1, 17, 128}) {
      const auto desc = dense(c::DataType::kFloat32, {rows, width});
      const auto data = pattern(*desc.element_count());
      const auto input = make(context, desc, data);
      for (const bool exclusive : {false, true}) {
        SCOPED_TRACE(::testing::Message()
                     << rows << '/' << width << '/' << exclusive);
        const c::Scan operation{.exclusive = exclusive};
        auto program = v::prepare(context, operation, desc);
        ASSERT_TRUE(program) << program.error().message();
        if (!exclusive || width > 1) {
          const auto &state = v::detail::Access::program(*program);
          EXPECT_TRUE(std::ranges::any_of(state->dispatches, [](const auto &d) {
            return static_cast<bool>(d.region);
          }));
        }
        auto slot = program->create_invocation();
        ASSERT_TRUE(slot) << slot.error().message();
        ASSERT_TRUE(slot->bind(input).ok());
        auto work = slot->submit().value();
        auto result = work.wait();
        ASSERT_TRUE(result) << result.error().message();
        expect_prefix(context, result->get()->tensors()[0], desc, data,
                      operation);
      }
    }
  }
  // A partial last 512-row tile is unsafe in the original single-pass shader.
  // The adapter keeps both provider pieces exact, including a wide tail that
  // can independently select the original multi-pass program.
  for (const std::int64_t rows : {513, 1025})
    for (const std::int64_t width : {17, 4097}) {
      const auto desc = dense(c::DataType::kFloat32, {rows, width});
      const auto data = pattern(*desc.element_count());
      const auto input = make(context, desc, data);
      for (const bool exclusive : {false, true}) {
        SCOPED_TRACE(::testing::Message()
                     << rows << '/' << width << '/' << exclusive);
        const c::Scan operation{.exclusive = exclusive};
        auto result =
            execute(context, c::Operation(operation), std::array{input});
        ASSERT_TRUE(result) << result.error().message();
        expect_prefix(context, *result, desc, data, operation);
      }
    }
}

TEST(VulkanScan, ArbitraryAxesStridesRankEightAndEmpty) {
  auto context = vulkan_test::context().value();
  const std::array<std::int64_t, 3> shape{2, 129, 3};
  const std::array layouts{
      c::TensorDesc::dense(c::DataType::kFloat32, shape).value(),
      c::TensorDesc::strided_bytes(c::DataType::kFloat32, shape,
                                   std::array<std::int64_t, 3>{5, 31, 10})
          .value(),
      c::TensorDesc::strided_bytes(c::DataType::kFloat32, shape,
                                   std::array<std::int64_t, 3>{0, 12, 4})
          .value(),
      dense(c::DataType::kFloat32, {1, 2, 1, 3, 1, 1, 129, 1}),
      dense(c::DataType::kFloat32, {3, 0, 7}),
      dense(c::DataType::kFloat32, {0, 3})};
  for (const auto &desc : layouts) {
    auto data = pattern(*desc.element_count());
    std::vector<std::uint64_t> raw;
    for (const auto value : data)
      raw.push_back(encode(value, c::DataType::kFloat32));
    const auto host = payload(desc, raw);
    for (std::size_t i = 0; i < data.size(); ++i) {
      std::uint32_t word;
      std::memcpy(&word, host.data() + offset(desc, i), sizeof(word));
      data[i] = std::bit_cast<float>(word);
    }
    const auto input = context.upload(desc, host).value();
    for (std::int32_t axis = 0; axis < static_cast<std::int32_t>(desc.rank());
         ++axis)
      for (const bool exclusive : {false, true}) {
        SCOPED_TRACE(::testing::Message()
                     << desc.rank() << '/' << axis << '/' << exclusive);
        const c::Scan operation{axis - static_cast<std::int32_t>(desc.rank()),
                                exclusive};
        auto result =
            execute(context, c::Operation(operation), std::array{input});
        ASSERT_TRUE(result) << result.error().message();
        expect_prefix(context, *result, desc, data, operation);
      }
    EXPECT_EQ(bytes(context, input), host);
  }
}

TEST(VulkanScan, RowDispatchHandlesTheSecondNativeGridBoundary) {
  auto context = vulkan_test::context().value();
  for (const std::int64_t rows : {262144, 262145, 262657}) {
    const auto desc = dense(c::DataType::kFloat32, {rows, 3});
    const auto data = pattern(*desc.element_count());
    const auto input = make(context, desc, data);
    auto result = execute(context, c::Operation(c::Scan{}), std::array{input});
    ASSERT_TRUE(result) << result.error().message();
    expect_prefix(context, *result, desc, data, c::Scan{});
  }
}

TEST(VulkanScan, RebindsOriginsPreservesSnapshotsAndHandlesNonfinitePrefixes) {
  auto context = vulkan_test::context().value();
  const auto desc = dense(c::DataType::kFloat32, {1, 129});
  for (const bool captured : {false, true})
    for (const bool exclusive : {false, true}) {
      const c::Scan operation{.exclusive = exclusive};
      std::optional<v::Program> program;
      std::optional<v::Invocation> first, second;
      v::Tensor retained;
      std::vector<std::uint64_t> saved;
      for (int phase = 0; phase < 5; ++phase) {
        const std::int64_t origin = phase;
        const auto storage_desc =
            c::TensorDesc::strided_bytes(
                c::DataType::kFloat32,
                std::array<std::int64_t, 2>{1, 129 * 4 + origin},
                std::array<std::int64_t, 2>{129 * 4, 1})
                .value();
        auto data = pattern(129, phase);
        data[31] = std::numeric_limits<float>::infinity();
        data[65] = -std::numeric_limits<float>::infinity();
        data[127] = std::numeric_limits<float>::quiet_NaN();
        std::vector<std::uint64_t> raw;
        for (const auto value : data)
          raw.push_back(encode(value, c::DataType::kFloat32));
        std::vector<std::byte> host(*storage_desc.required_bytes(),
                                    std::byte{0xcd});
        const auto packed = payload(desc, raw);
        std::ranges::copy(packed, host.begin() + origin);
        const auto storage = context.upload(storage_desc, host).value();
        const auto input =
            storage.view(c::View::slice(1, origin, 129, 4).value()).value();
        ASSERT_EQ(input.description(), desc);
        if (!program || captured) {
          program =
              v::prepare(context, operation,
                         captured ? v::Argument(input) : v::Argument(desc))
                  .value();
          first = program->create_invocation().value();
          second = program->create_invocation().value();
        }
        auto &slot = phase == 0 ? *first : *second;
        if (!captured)
          ASSERT_TRUE(slot.bind(input).ok());
        auto work = slot.submit().value();
        auto result = work.wait();
        ASSERT_TRUE(result) << result.error().message();
        expect_prefix(context, result->get()->tensors()[0], desc, data,
                      operation);
        if (phase == 0) {
          retained = result->get()->tensors()[0];
          saved = bits(context, retained);
        } else {
          EXPECT_EQ(bits(context, retained), saved);
        }
        EXPECT_EQ(bytes(context, storage), host);
      }
    }
  EXPECT_TRUE(context.close().ok());
}

TEST(VulkanScan, RejectsFloatNativeLimitsBeforeSubmission) {
  auto context = vulkan_test::context().value();
  auto &limits = v::detail::Access::runtime(context)->device->properties.limits;
  const auto saved = limits;
  const auto input = dense(c::DataType::kFloat32, {2, 8193});
  for (int mode = 0; mode < 3; ++mode) {
    limits = saved;
    if (mode == 0)
      limits.maxComputeWorkGroupCount[0] = 1;
    else if (mode == 1)
      limits.maxComputeWorkGroupInvocations = 64;
    else
      limits.maxStorageBufferRange = 4;
    for (const bool exclusive : {false, true}) {
      auto rejected =
          v::prepare(context, c::Scan{.exclusive = exclusive}, input);
      EXPECT_FALSE(rejected);
      if (!rejected)
        EXPECT_TRUE(rejected.error().IsUnimplemented());
    }
  }
  limits = saved;
}
TEST(VulkanIntegerScan, HierarchicalWidthsMatchIndependentWidePrefix) {
  auto context = vulkan_test::context().value();
  for (const std::int64_t width :
       {0, 1, 127, 128, 129, 255, 256, 257, 8193, 65535, 65536, 65537}) {
    const auto desc = dense(c::DataType::kInt32, {3, width});
    const auto data = integer_pattern(*desc.element_count());
    const auto input = make_integer(context, desc, data);
    for (const bool exclusive : {false, true}) {
      SCOPED_TRACE(::testing::Message() << width << '/' << exclusive);
      const c::Scan operation{.exclusive = exclusive};
      auto output =
          execute(context, c::Operation(operation), std::array{input});
      ASSERT_TRUE(output) << output.error().message();
      expect_integer_prefix(context, *output, desc, data, operation);
    }
  }
}

TEST(VulkanIntegerScan, LocalBlockOverflowDoesNotRejectLegalGlobalPrefixes) {
  auto context = vulkan_test::context().value();
  const auto desc = dense(c::DataType::kInt32, {513});
  for (const bool negative : {false, true}) {
    std::vector<std::int32_t> data(513, 0);
    data[0] = negative ? INT32_MAX : INT32_MIN;
    data[256] = negative ? INT32_MIN : INT32_MAX;
    data[257] = negative ? -INT32_MAX : INT32_MAX;
    const auto input = make_integer(context, desc, data);
    for (const bool exclusive : {false, true}) {
      const c::Scan operation{.exclusive = exclusive};
      auto output =
          execute(context, c::Operation(operation), std::array{input});
      ASSERT_TRUE(output) << output.error().message();
      expect_integer_prefix(context, *output, desc, data, operation);
    }
  }
}

TEST(VulkanIntegerScan, AxesStridesRankEightAndEmptyPreserveInputs) {
  auto context = vulkan_test::context().value();
  const std::array<std::int64_t, 3> shape{2, 129, 3};
  const std::array layouts{
      c::TensorDesc::dense(c::DataType::kInt32, shape).value(),
      c::TensorDesc::strided_bytes(c::DataType::kInt32, shape,
                                   std::array<std::int64_t, 3>{2200, 17, 5})
          .value(),
      c::TensorDesc::strided_bytes(c::DataType::kInt32, shape,
                                   std::array<std::int64_t, 3>{0, 12, 4})
          .value(),
      dense(c::DataType::kInt32, {1, 2, 1, 3, 1, 1, 129, 1}),
      dense(c::DataType::kInt32, {3, 0, 7}),
      dense(c::DataType::kInt32, {0, 3})};
  for (const auto &desc : layouts) {
    auto data = integer_pattern(*desc.element_count());
    const auto host = integer_payload(desc, data);
    // Broadcast layouts share storage; reference the actual logical input.
    for (std::size_t i = 0; i < data.size(); ++i)
      std::memcpy(&data[i], host.data() + offset(desc, i), sizeof(data[i]));
    const auto input = context.upload(desc, host).value();
    for (std::int32_t axis = 0; axis < static_cast<std::int32_t>(desc.rank());
         ++axis)
      for (const bool exclusive : {false, true}) {
        SCOPED_TRACE(::testing::Message()
                     << desc.rank() << '/' << axis << '/' << exclusive);
        const c::Scan operation{axis - static_cast<std::int32_t>(desc.rank()),
                                exclusive};
        auto output =
            execute(context, c::Operation(operation), std::array{input});
        ASSERT_TRUE(output) << output.error().message();
        expect_integer_prefix(context, *output, desc, data, operation);
      }
    EXPECT_EQ(bytes(context, input), host);
  }
}

TEST(VulkanIntegerScan, EveryPrefixAndExclusiveFinalTotalAreChecked) {
  auto context = vulkan_test::context().value();
  for (const std::vector<std::int32_t> &data :
       {std::vector<std::int32_t>{INT32_MAX, 1},
        {INT32_MIN, -1},
        {INT32_MAX, 1, -1},
        {INT32_MIN, -1, 1}}) {
    const auto desc =
        dense(c::DataType::kInt32, {static_cast<std::int64_t>(data.size())});
    const auto input = make_integer(context, desc, data);
    for (const bool exclusive : {false, true}) {
      auto program =
          v::prepare(context, c::Scan{.exclusive = exclusive}, desc).value();
      auto failed = program.create_invocation().value();
      ASSERT_TRUE(failed.bind(input).ok());
      auto work = failed.submit().value();
      auto output = work.wait();
      ASSERT_FALSE(output);
      EXPECT_TRUE(output.error().IsOutOfRange());
      EXPECT_TRUE(failed.bind(input).IsOutOfRange());
      const auto good_data = integer_pattern(data.size());
      const auto good = make_integer(context, desc, good_data);
      auto fresh = program.create_invocation().value();
      ASSERT_TRUE(fresh.bind(good).ok());
      auto recovered = fresh.submit().value();
      auto result = recovered.wait();
      ASSERT_TRUE(result) << result.error().message();
      expect_integer_prefix(context, result->get()->tensors()[0], desc,
                            good_data, c::Scan{.exclusive = exclusive});
    }
  }
}

TEST(VulkanIntegerScan, LateRowFailurePreservesEntireRecordedOutput) {
  auto context = vulkan_test::context().value();
  for (const bool transposed : {false, true}) {
    const auto desc = transposed ? dense(c::DataType::kInt32, {257, 2})
                                 : dense(c::DataType::kInt32, {2, 257});
    std::vector<std::int32_t> data(514, 0);
    data[transposed ? 511 : 512] = INT32_MAX;
    data[513] = 1;
    const auto input = make_integer(context, desc, data);
    const std::vector<std::int32_t> sentinel(514, 0x55aa55aa);
    for (const bool exclusive : {false, true}) {
      const c::Scan operation{transposed ? 0 : 1, exclusive};
      auto program = v::prepare(context, operation, desc).value();
      const auto output = make_integer(context, desc, sentinel);
      auto recorder = v::Recorder::create(context).value();
      ASSERT_TRUE(
          recorder.append(program, std::array{input}, std::array{output}).ok());
      auto recorded = recorder.finish().value();
      auto completion = recorded.submit().value();
      EXPECT_TRUE(completion.wait().IsOutOfRange());
      EXPECT_EQ(integer_values(context, output), sentinel);
      EXPECT_FALSE(recorded.submit());
      EXPECT_EQ(integer_values(context, input), data);
    }
    auto program = v::prepare(context, c::Scan{}, desc).value();
    auto recorder = v::Recorder::create(context).value();
    EXPECT_TRUE(recorder.append(program, std::array{input}, std::array{input})
                    .IsInvalidArgument());
  }
}

TEST(VulkanIntegerScan, OriginsIndependentSlotsAndRetainedResults) {
  auto context = vulkan_test::context().value();
  const auto desc = dense(c::DataType::kInt32, {1, 257});
  for (const bool captured : {false, true})
    for (const bool exclusive : {false, true}) {
      const c::Scan operation{.exclusive = exclusive};
      std::optional<v::Program> program;
      std::optional<v::Invocation> first, second;
      v::Tensor retained;
      std::vector<std::uint64_t> saved;
      for (int phase = 0; phase < 5; ++phase) {
        const std::int64_t origin = phase;
        const auto storage_desc =
            c::TensorDesc::strided_bytes(
                c::DataType::kInt32,
                std::array<std::int64_t, 2>{1, 257 * 4 + origin},
                std::array<std::int64_t, 2>{257 * 4, 1})
                .value();
        const auto data = integer_pattern(257, phase);
        std::vector<std::byte> host(*storage_desc.required_bytes(),
                                    std::byte{0xcd});
        const auto packed = integer_payload(desc, data);
        std::ranges::copy(packed, host.begin() + origin);
        const auto storage = context.upload(storage_desc, host).value();
        const auto input =
            storage.view(c::View::slice(1, origin, 257, 4).value()).value();
        ASSERT_EQ(input.description(), desc);
        if (!program || captured) {
          program =
              v::prepare(context, operation,
                         captured ? v::Argument(input) : v::Argument(desc))
                  .value();
          first = program->create_invocation().value();
          second = program->create_invocation().value();
        }
        auto &slot = phase == 0 ? *first : *second;
        if (!captured)
          ASSERT_TRUE(slot.bind(input).ok());
        auto work = slot.submit().value();
        auto result = work.wait();
        ASSERT_TRUE(result) << result.error().message();
        expect_integer_prefix(context, result->get()->tensors()[0], desc, data,
                              operation);
        if (phase == 0) {
          retained = result->get()->tensors()[0];
          saved = bits(context, retained);
        } else {
          EXPECT_EQ(bits(context, retained), saved);
        }
        EXPECT_EQ(bytes(context, storage), host);
      }
    }
  EXPECT_TRUE(context.close().ok());
}

TEST(VulkanIntegerScan, FrozenNonLastAxisLimitsRejectBeforeSubmission) {
  auto context = vulkan_test::context().value();
  auto &limits = v::detail::Access::runtime(context)->device->properties.limits;
  const auto saved = limits;
  // Keep exercising the unchanged non-last-axis implementation. Last-axis
  // dense inputs now use the original IREE program's different launch bounds.
  const auto desc = dense(c::DataType::kInt32, {8193, 2});
  for (int mode = 0; mode < 5; ++mode) {
    limits = saved;
    if (mode == 0)
      limits.maxComputeWorkGroupInvocations = 128;
    else if (mode == 1)
      limits.maxComputeWorkGroupSize[0] = 128;
    else if (mode == 2)
      limits.maxComputeSharedMemorySize = 2047;
    else if (mode == 3)
      limits.maxStorageBufferRange =
          2 * 8193 * 4; // I32 fits; I64 scratch does not.
    else {
      limits.maxComputeWorkGroupCount[0] = 1;
      limits.maxComputeWorkGroupCount[1] = 1;
    }
    auto rejected = v::prepare(context, c::Scan{.axis = 0}, desc);
    EXPECT_FALSE(rejected);
    if (!rejected)
      EXPECT_TRUE(rejected.error().IsUnimplemented() ||
                  rejected.error().IsOutOfRange());
  }
  limits = saved;
}

TEST(VulkanIntegerScan, FrozenNonLastAxisQuotaAndRetainedResult) {
  constexpr std::size_t limit = 64 * 1024 * 1024;
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(), limit);
  {
    auto context = vulkan_test::context(&quota).value();
    const auto desc = dense(c::DataType::kInt32, {8193, 17});
    const c::Scan operation{.axis = 0};
    auto program = v::prepare(context, operation, desc).value();
    const auto before = quota.used();
    auto reservation = quota.try_reserve_bytes(limit - before);
    ASSERT_TRUE(reservation);
    auto failed = program.create_invocation();
    ASSERT_FALSE(failed);
    EXPECT_TRUE(failed.error().IsResourceExhausted());
    EXPECT_EQ(quota.used(), limit);
    reservation.reset();
    EXPECT_EQ(quota.used(), before);
    const auto data = integer_pattern(*desc.element_count());
    const auto input = make_integer(context, desc, data);
    auto slot = program.create_invocation().value();
    program = {};
    ASSERT_TRUE(slot.bind(input).ok());
    const auto charged = quota.used();
    v::Tensor retained;
    for (int phase = 0; phase < 5; ++phase) {
      auto work = slot.submit().value();
      auto output = work.wait();
      ASSERT_TRUE(output) << output.error().message();
      EXPECT_EQ(quota.used(), charged);
      expect_integer_prefix(context, output->get()->tensors()[0], desc, data,
                            operation);
      if (phase == 4)
        retained = output->get()->tensors()[0];
    }
    EXPECT_TRUE(context.close().ok());
    expect_integer_prefix(context, retained, desc, data, operation);
  }
  EXPECT_EQ(quota.used(), 0U);
}
} // namespace
