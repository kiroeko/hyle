#include <gtest/gtest.h>

#include "topk_test_support.h"
#include "vulkan_program_internal.h"

namespace {
using namespace vulkan_topk_test;

TEST(VulkanTopK, FrozenExpertWidthsTailsAndStableRawOutputs) {
  auto context = vulkan_test::context().value();
  for (const std::int64_t width :
       {1, 17, 127, 129, 256, 288, 384, 512, 896, 897}) {
    const auto desc = dense(c::DataType::kFloat32, {17, width});
    const auto raw = pattern(*desc.element_count());
    const auto host = payload(desc, raw);
    const auto input = context.upload(desc, host).value();
    std::vector<std::size_t> counts{1, static_cast<std::size_t>(width)};
    for (std::size_t count : {6U, 8U, 10U, 16U})
      if (count < static_cast<std::size_t>(width))
        counts.push_back(count);
    std::ranges::sort(counts);
    const auto duplicate = std::ranges::unique(counts);
    counts.erase(duplicate.begin(), duplicate.end());
    for (bool largest : {false, true}) {
      for (auto count : counts) {
        SCOPED_TRACE(::testing::Message()
                     << width << '/' << count << '/' << largest);
        const c::TopK operation{.count = count, .largest = largest};
        auto prepared = v::prepare(context, operation, desc);
        ASSERT_TRUE(prepared) << prepared.error().message();
        ASSERT_EQ(prepared->output_descriptions().size(), 2U);
        const auto &plan = v::detail::Access::program(*prepared);
        EXPECT_TRUE(
            std::ranges::any_of(plan->dispatches, [](const auto &dispatch) {
              return static_cast<bool>(dispatch.region);
            }));
        auto outputs = execute_all(*prepared, input);
        ASSERT_TRUE(outputs) << outputs.error().message();
        expect_pair(context, *outputs, reference(desc, raw, operation));
        EXPECT_EQ(bytes(context, input), host);
      }
    }
  }
}

TEST(VulkanTopK, AxisLayoutAdaptersPreserveInputHolesAndZeroStride) {
  auto context = vulkan_test::context().value();
  const std::array<std::int64_t, 3> shape{2, 129, 3};
  const std::array layouts{
      c::TensorDesc::dense(c::DataType::kFloat32, shape).value(),
      c::TensorDesc::strided_bytes(
          c::DataType::kFloat32, shape,
          std::array<std::int64_t, 3>{5, 2 * 3 * 5 + 1, 10})
          .value(),
      c::TensorDesc::strided_bytes(c::DataType::kFloat32, shape,
                                   std::array<std::int64_t, 3>{0, 12, 4})
          .value()};
  for (const auto &desc : layouts) {
    auto raw = pattern(*desc.element_count());
    const auto host = payload(desc, raw);
    // Broadcast addresses intentionally alias; derive logical values from the
    // declared host storage, without consulting the device output.
    for (std::size_t i = 0; i < raw.size(); ++i) {
      raw[i] = 0;
      std::memcpy(&raw[i], host.data() + offset(desc, i), sizeof(float));
    }
    const auto input = context.upload(desc, host).value();
    for (int axis : {0, -2, -1}) {
      const auto extent = shape[axis < 0 ? axis + 3 : axis];
      const c::TopK operation{.count = static_cast<std::size_t>(
                                  (std::min)(extent, std::int64_t{8})),
                              .axis = axis,
                              .largest = false};
      auto program = v::prepare(context, operation, desc);
      ASSERT_TRUE(program) << program.error().message();
      auto outputs = execute_all(*program, input);
      ASSERT_TRUE(outputs) << outputs.error().message();
      expect_pair(context, *outputs, reference(desc, raw, operation));
      EXPECT_EQ(bytes(context, input), host);
    }
  }
}

TEST(VulkanTopK, OriginalLargeSortKeepsTheSameStableAndNanPolicy) {
  auto context = vulkan_test::context().value();
  for (std::int64_t width : {1025, 4097}) {
    const auto desc = dense(c::DataType::kFloat32, {2, width});
    const auto raw = pattern(*desc.element_count());
    const auto input = context.upload(desc, payload(desc, raw)).value();
    for (bool largest : {false, true}) {
      for (std::size_t count :
           {std::size_t{1}, std::size_t{8}, static_cast<std::size_t>(width)}) {
        SCOPED_TRACE(::testing::Message()
                     << width << '/' << count << '/' << largest);
        const c::TopK operation{.count = count, .largest = largest};
        auto prepared = v::prepare(context, operation, desc);
        ASSERT_TRUE(prepared) << prepared.error().message();
        auto outputs = execute_all(*prepared, input);
        ASSERT_TRUE(outputs) << outputs.error().message();
        expect_pair(context, *outputs, reference(desc, raw, operation));
      }
    }
  }
}

TEST(VulkanTopK, RebindingByteOriginsRefreshesFrozenProviderViews) {
  auto context = vulkan_test::context().value();
  const auto desc =
      c::TensorDesc::strided_bytes(c::DataType::kFloat32,
                                   std::array<std::int64_t, 3>{1, 2, 129},
                                   std::array<std::int64_t, 3>{1, 129 * 4, 4})
          .value();
  const auto storage_desc =
      c::TensorDesc::strided_bytes(c::DataType::kFloat32,
                                   std::array<std::int64_t, 3>{9, 2, 129},
                                   std::array<std::int64_t, 3>{1, 129 * 4, 4})
          .value();
  const c::TopK operation{.count = 8};
  auto program = v::prepare(context, operation, desc).value();
  auto slot = program.create_invocation().value();
  for (std::size_t origin : {0U, 1U, 4U, 7U}) {
    SCOPED_TRACE(origin);
    const auto raw = pattern(*desc.element_count(), static_cast<int>(origin));
    auto storage = context.allocate(storage_desc).value();
    std::vector<std::byte> host(*storage_desc.required_bytes(),
                                std::byte{0x5a});
    const auto values = payload(desc, raw);
    std::ranges::copy(values, host.begin() + origin);
    ASSERT_TRUE(context.upload(storage, host).ok());
    // borrow_ready requires an aligned native descriptor range. Tensor::view
    // is the ordinary API for a legal byte-origin slice of an existing owner.
    auto slice = storage.view(
        c::View::slice(0, static_cast<std::int64_t>(origin), 1).value());
    ASSERT_TRUE(slice) << slice.error().message();
    auto input = *slice;
    ASSERT_TRUE(slot.bind(input).ok());
    auto work = slot.submit().value();
    auto result = work.wait();
    ASSERT_TRUE(result) << result.error().message();
    expect_pair(context, result->get()->tensors(),
                reference(desc, raw, operation));
    EXPECT_EQ(bytes(context, storage), host);
    // Captured slices use the same ordinary layout adapter as dynamic inputs.
    auto captured = v::prepare(context, operation, input).value();
    auto fixed = captured.create_invocation().value();
    ASSERT_TRUE(fixed.bind(std::span<const v::Tensor>{}).ok());
    auto fixed_work = fixed.submit().value();
    auto fixed_result = fixed_work.wait();
    ASSERT_TRUE(fixed_result) << fixed_result.error().message();
    expect_pair(context, fixed_result->get()->tensors(),
                reference(desc, raw, operation));
  }

  // A dense, aligned captured input enters the provider without a layout Copy.
  // Its internal reshape/view chain must inherit the nonzero native offset.
  const auto direct_desc = dense(c::DataType::kFloat32, {2, 129});
  const auto alignment =
      (std::max)(VkDeviceSize{4}, v::detail::Access::runtime(context)
                                      ->device->properties.limits
                                      .minStorageBufferOffsetAlignment);
  const auto capacity = (*direct_desc.required_bytes() + 3) & ~std::size_t{3};
  std::vector<std::byte> guarded(alignment + capacity + 8, std::byte{0x5a});
  const auto raw = pattern(*direct_desc.element_count(), 19);
  const auto values = payload(direct_desc, raw);
  std::ranges::copy(values, guarded.begin() + alignment);
  const auto owner_desc =
      dense(c::DataType::kBool, {static_cast<std::int64_t>(guarded.size())});
  auto owner = context.upload(owner_desc, guarded);
  ASSERT_TRUE(owner) << owner.error().message();
  auto borrowed =
      context.borrow_ready(direct_desc, owner->native_buffer(), alignment,
                           capacity, owner->native_owner());
  ASSERT_TRUE(borrowed) << borrowed.error().message();
  auto captured = v::prepare(context, operation, *borrowed);
  ASSERT_TRUE(captured) << captured.error().message();
  auto fixed = captured->create_invocation();
  ASSERT_TRUE(fixed) << fixed.error().message();
  ASSERT_TRUE(fixed->bind(std::span<const v::Tensor>{}).ok());
  auto fixed_work = fixed->submit();
  ASSERT_TRUE(fixed_work) << fixed_work.error().message();
  auto fixed_result = fixed_work->wait();
  ASSERT_TRUE(fixed_result) << fixed_result.error().message();
  expect_pair(context, fixed_result->get()->tensors(),
              reference(direct_desc, raw, operation));
  EXPECT_EQ(bytes(context, *owner), guarded);
}

TEST(VulkanTopK, EqualRowsNanPayloadsAndSignedZerosSurviveTailPadding) {
  auto context = vulkan_test::context().value();
  const auto desc = dense(c::DataType::kFloat32, {3, 129});
  std::vector<std::uint64_t> raw(*desc.element_count());
  for (std::size_t i = 0; i < 129; ++i) {
    raw[i] = 0x7fc00001U + i;
    if (i % 2)
      raw[i] |= 0x80000000U;
    raw[129 + i] = i % 2 ? 0x80000000U : 0U;
    raw[258 + i] = 0x3fa00000U;
  }
  const auto input = context.upload(desc, payload(desc, raw)).value();
  for (std::size_t count : {1U, 8U, 129U}) {
    for (bool largest : {false, true}) {
      const c::TopK operation{.count = count, .largest = largest};
      auto program = v::prepare(context, operation, desc);
      ASSERT_TRUE(program) << program.error().message();
      auto outputs = execute_all(*program, input);
      ASSERT_TRUE(outputs) << outputs.error().message();
      expect_pair(context, *outputs, reference(desc, raw, operation));
    }
  }
}

TEST(VulkanTopK, RankEightAndRelaxedOrderUseTheSameOrdinaryEntry) {
  auto context = vulkan_test::context().value();
  const auto desc = dense(c::DataType::kFloat32, {2, 1, 2, 1, 3, 129, 1, 2});
  const auto raw = pattern(*desc.element_count());
  const auto input = context.upload(desc, payload(desc, raw)).value();
  const c::TopK operation{.count = 8, .axis = 5};
  auto program = v::prepare(context, operation, desc);
  ASSERT_TRUE(program) << program.error().message();
  auto outputs = execute_all(*program, input);
  ASSERT_TRUE(outputs) << outputs.error().message();
  expect_pair(context, *outputs, reference(desc, raw, operation));

  // No ordering is required when sorted=false. Distinct values avoid imposing
  // a stronger tie guarantee than stable=false promises.
  const auto relaxed_desc = dense(c::DataType::kFloat32, {17});
  std::vector<std::uint64_t> unique(17);
  for (std::size_t i = 0; i < unique.size(); ++i)
    unique[i] = encode(static_cast<double>(i) - 8, c::DataType::kFloat32);
  const auto relaxed_input =
      context.upload(relaxed_desc, payload(relaxed_desc, unique)).value();
  const c::TopK relaxed{.count = 8, .sorted = false, .stable = false};
  auto unordered = v::prepare(context, relaxed, relaxed_desc);
  ASSERT_TRUE(unordered) << unordered.error().message();
  auto chosen = execute_all(*unordered, relaxed_input);
  ASSERT_TRUE(chosen) << chosen.error().message();
  auto indices = bits(context, (*chosen)[1]);
  const auto values = bits(context, (*chosen)[0]);
  for (std::size_t i = 0; i < indices.size(); ++i) {
    ASSERT_LT(indices[i], unique.size());
    EXPECT_EQ(values[i], unique[indices[i]]);
  }
  auto expected = reference(relaxed_desc, unique, relaxed).indices;
  std::ranges::sort(indices);
  std::ranges::sort(expected);
  EXPECT_EQ(indices, expected);
}

TEST(VulkanTopK, RecorderValidatesTwoOutputBindingsAndReplaysChangedInput) {
  auto context = vulkan_test::context().value();
  const auto desc = dense(c::DataType::kFloat32, {2, 129});
  const c::TopK operation{.count = 129};
  auto program = v::prepare(context, operation, desc);
  ASSERT_TRUE(program) << program.error().message();
  ASSERT_EQ(program->output_descriptions().size(), 2U);
  const auto input = context.allocate(desc).value();
  const auto output =
      context.allocate(program->output_descriptions()[0]).value();
  const auto index_desc = program->output_descriptions()[1];
  const auto indices = context.allocate(index_desc).value();
  const auto aliases_values =
      context
          .borrow_ready(index_desc, output.native_buffer(),
                        output.native_offset(), *index_desc.required_bytes(),
                        output.native_owner())
          .value();
  const auto aliases_input =
      context
          .borrow_ready(index_desc, input.native_buffer(),
                        input.native_offset(), *index_desc.required_bytes(),
                        input.native_owner())
          .value();
  auto recorder = v::Recorder::create(context).value();
  EXPECT_FALSE(
      recorder.append(*program, std::array{input}, std::array{output}).ok());
  EXPECT_FALSE(recorder
                   .append(*program, std::array{input},
                           std::array{output, aliases_values})
                   .ok());
  EXPECT_FALSE(
      recorder.append(*program, std::array{input}, std::array{input, indices})
          .ok());
  EXPECT_FALSE(recorder
                   .append(*program, std::array{input},
                           std::array{output, aliases_input})
                   .ok());
  ASSERT_TRUE(
      recorder.append(*program, std::array{input}, std::array{output, indices})
          .ok());
  auto recording = recorder.finish().value();
  *program = {};
  for (int phase = 0; phase < 3; ++phase) {
    const auto raw = pattern(*desc.element_count(), phase);
    ASSERT_TRUE(context.upload(input, payload(desc, raw)).ok());
    auto work = recording.submit();
    ASSERT_TRUE(work) << work.error().message();
    ASSERT_TRUE(work->wait().ok());
    expect_pair(context, std::array{output, indices},
                reference(desc, raw, operation));
    EXPECT_EQ(bits(context, input), raw);
  }
}

TEST(VulkanTopK, EmptyDomainAndInvalidPreparationHaveNoPayloadFallback) {
  auto context = vulkan_test::context().value();
  const auto desc = dense(c::DataType::kFloat32, {0, 129});
  const c::TopK operation{.count = 8};
  auto program = v::prepare(context, operation, desc);
  ASSERT_TRUE(program) << program.error().message();
  const auto input = context.upload(desc, {}).value();
  auto outputs = execute_all(*program, input);
  ASSERT_TRUE(outputs) << outputs.error().message();
  expect_pair(context, *outputs, {});
  const auto ordinary = dense(c::DataType::kFloat32, {2, 17});
  EXPECT_FALSE(v::prepare(context, c::TopK{.count = 0}, ordinary));
  EXPECT_FALSE(v::prepare(context, c::TopK{.count = 18}, ordinary));
  EXPECT_FALSE(v::prepare(context, c::TopK{.axis = 2}, ordinary));
  EXPECT_FALSE(
      v::prepare(context, c::TopK{}, dense(c::DataType::kFloat32, {2, 0})));
  for (const auto type :
       {c::DataType::kFloat16, c::DataType::kBFloat16, c::DataType::kInt32})
    EXPECT_FALSE(v::prepare(context, c::TopK{}, dense(type, {2, 17})));
}

TEST(VulkanTopK, QuotaFailureRollsBackAndFreshInvocationStillWorks) {
  constexpr std::size_t limit = 64 * 1024 * 1024;
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(), limit);
  auto context = vulkan_test::context(&quota).value();
  const auto desc = dense(c::DataType::kFloat32, {17, 896});
  const c::TopK operation{.count = 8};
  auto program = v::prepare(context, operation, desc);
  ASSERT_TRUE(program) << program.error().message();
  const auto before = quota.used();
  auto reservation = quota.try_reserve_bytes(limit - before);
  ASSERT_TRUE(reservation);
  auto failed = program->create_invocation();
  ASSERT_FALSE(failed);
  EXPECT_TRUE(failed.error().IsResourceExhausted());
  EXPECT_EQ(quota.used(), limit);
  reservation.reset();
  EXPECT_EQ(quota.used(), before);
  const auto raw = pattern(*desc.element_count());
  const auto input = context.upload(desc, payload(desc, raw)).value();
  auto output = execute_all(*program, input);
  ASSERT_TRUE(output) << output.error().message();
  EXPECT_TRUE(context.close().ok());
  expect_pair(context, *output, reference(desc, raw, operation));
}

TEST(VulkanTopK, ColdMultiOutputSsaRejectsMissingDuplicateAndSelfProducedIds) {
  auto context = vulkan_test::context().value();
  v::detail::Plan valid;
  valid.values = {{dense(c::DataType::kFloat32, {2, 17}), {}},
                  {dense(c::DataType::kFloat32, {2, 8}), {}},
                  {dense(c::DataType::kInt32, {2, 8}), {}}};
  valid.inputs = {0};
  valid.outputs = {1, 2};
  valid.steps.push_back(
      {c::Operation(c::TopK{.count = 8}), {0}, {1, 2}, {}, {1, 1, 1}, {}});
  for (int failure = 0; failure < 5; ++failure) {
    SCOPED_TRACE(failure);
    auto malformed = valid;
    auto &step = malformed.steps[0];
    if (failure == 0)
      step.outputs.pop_back();
    else if (failure == 1)
      step.outputs[1] = step.outputs[0];
    else if (failure == 2)
      step.outputs[1] = 3;
    else if (failure == 3)
      step.inputs[0] = step.outputs[0];
    else
      malformed.values[2].description = malformed.values[1].description;
    EXPECT_FALSE(v::detail::prepare_plan(context, malformed));
  }
  EXPECT_TRUE(v::detail::prepare_plan(context, valid));
}
} // namespace
