#include <chrono>
#include <thread>

#include "iree_test_support.h"
#include "artifact_test_support.h"
#include "mlx_test_support.h"
#include "scan_test_support.h"
#include "vulkan_program_internal.h"

namespace {
using namespace vulkan_iree_test;
namespace t = vulkan_tensor_test;
namespace s = vulkan_scan_test;

hyle::Result<v::Program> prepare_artifact(const v::Context &context,
                                        c::Operation operation,
                                        std::span<const c::TensorDesc> descriptions) {
  std::vector<v::Argument> arguments;
  for (const auto &description : descriptions) arguments.emplace_back(description);
  return v::prepare(context, operation, std::span<const v::Argument>(arguments));
}

TEST(VulkanIree, OriginalMlxBatchPreservesGroupsSourceTypesAndCheckedSlots) {
  vulkan_mlx_test::batch(prepare_artifact);
}

TEST(VulkanIree, OriginalMlxConstantsKeepBanksAcrossRebindingAndSlots) {
  vulkan_mlx_test::constants([](const v::Context &context, c::Operation operation,
                               std::span<const v::Tensor> resident) {
    std::vector<v::Argument> arguments;
    for (std::size_t i = 0; i < resident.size(); ++i)
      if (i == 0 || i + 1 == resident.size()) arguments.emplace_back(resident[i].description());
      else arguments.emplace_back(resident[i]);
    return v::prepare(context, operation, std::span<const v::Argument>(arguments));
  });
}

TEST(VulkanIree, OriginalMlxEmptyEmbeddingDomainOomCancelAndClose) {
  using namespace vulkan_mlx_test;
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(), 384 * 1024 * 1024);
  {
    auto context = vulkan_test::context(&quota).value();
    Inputs data(Format::kAffine4, Kind::kEmbedding, c::DataType::kBFloat16, 64);
    auto descriptions = data.descriptions;
    descriptions.back() = t::dense(c::DataType::kInt32, {0});
    auto empty = prepare_artifact(context, data.operation, descriptions).value();
    auto inputs = data.upload(context);
    inputs.back() = context.upload(descriptions.back(), {}).value();
    { auto slot = empty.create_invocation().value();ASSERT_TRUE(slot.bind(inputs).ok());
      auto done = slot.submit().value();auto result = done.wait();ASSERT_TRUE(result);
      EXPECT_EQ(*result->get()->tensors()[0].description().element_count(), 0U); }
    auto &limits = v::detail::Access::runtime(context)->device->properties.limits;
    const auto saved = limits;
    limits.maxComputeWorkGroupInvocations = 63;
    EXPECT_FALSE(prepare_artifact(context, data.operation, data.descriptions));
    limits = saved;
    auto program = prepare_artifact(context, data.operation, data.descriptions).value();
    { auto reservation = quota.try_reserve_bytes(quota.limit() - quota.used());ASSERT_TRUE(reservation);
      auto denied = program.create_invocation();ASSERT_FALSE(denied);
      EXPECT_TRUE(denied.error().IsResourceExhausted()); }
    auto slot = program.create_invocation().value();ASSERT_TRUE(slot.bind(data.upload(context)).ok());
    { auto done = slot.submit().value();ASSERT_TRUE(done.request_cancel().ok());auto result = done.wait();
      if (result) data.check(context, result->get()->tensors()[0]);else EXPECT_TRUE(result.error().IsCancelled()); }
    { auto done = slot.submit().value();auto result = done.wait();ASSERT_TRUE(result) << result.error().message();
      data.check(context, result->get()->tensors()[0]); }
    EXPECT_TRUE(context.close().ok());EXPECT_FALSE(slot.submit());
  }
  EXPECT_EQ(quota.used(), 0U);
}

TEST(VulkanIree, OriginalMlxMx8SignsZerosAndSourceScaleClassification) {
  using namespace vulkan_mlx_test;
  auto context = vulkan_test::context().value();
  for (auto format : {Format::kMxfp4, Format::kMxfp8}) for (auto type : types)
    for (auto exponent : {0, 102, 103, 127, 142, 143, 255}) {
      SCOPED_TRACE(::testing::Message() << static_cast<int>(format) << '/' << static_cast<int>(type) << '/' << exponent);
      Inputs data(format, Kind::kLinear, type, 32, 0, c::DataType::kUnknown, 32);
      std::fill(data.codes.begin(), data.codes.end(), format == Format::kMxfp4 ? 2 : 0x38);
      std::vector<std::uint64_t> words(*data.descriptions[1].element_count(), format == Format::kMxfp4 ? 0x22222222 : 0x38383838);
      data.payloads[1] = t::payload(data.descriptions[1], words);
      // Small dot at overflow proves the source F16 scale itself overflows;
      // final F16 rounding alone would produce the finite value2048.
      std::fill(data.activation.begin(), data.activation.end(), exponent == 143 ? 1. / 1024 : 1.);
      std::vector<std::uint64_t> a(data.activation.size(), t::encode(data.activation[0], type));
      data.payloads[0] = t::payload(data.descriptions[0], a);
      std::fill(data.exponents.begin(), data.exponents.end(), static_cast<std::uint8_t>(exponent));
      std::vector<std::uint64_t> s(data.exponents.size(), static_cast<std::uint64_t>(exponent));
      data.payloads[2] = t::payload(data.descriptions[2], s);data.calculate();
      auto program = prepare_artifact(context, data.operation, data.descriptions).value();
      auto slot = program.create_invocation().value();ASSERT_TRUE(slot.bind(data.upload(context)).ok());
      auto done = slot.submit().value();auto result = done.wait();ASSERT_TRUE(result) << result.error().message();
      data.check(context, result->get()->tensors()[0]);
    }
}

TEST(VulkanIree, OriginalMlxMx8All256EncodingsUseTheSourceFormat) {
  using namespace vulkan_mlx_test;
  auto context = vulkan_test::context().value();
  for (auto type : types) {
    Inputs data(Format::kMxfp8, Kind::kLinear, type, 32, 0, c::DataType::kUnknown, 32);
    data.n = 256;data.codes.resize(data.n * data.k);
    data.descriptions[1] = t::dense(c::DataType::kUInt32, {256, 8});
    data.descriptions[2] = t::dense(c::DataType::kUInt8, {256, 1});
    data.output = t::dense(type, {2, 256});
    std::vector<std::uint64_t> words(256 * 8), scales(256, 127);
    for (int code = 0; code < 256; ++code)
      for (int j = 0; j < 32; ++j) {
        data.codes[code * 32 + j] = static_cast<std::uint8_t>(code);
        words[code * 8 + j / 4] = static_cast<std::uint64_t>(code) * 0x01010101;
      }
    data.payloads[1] = t::payload(data.descriptions[1], words);
    data.payloads[2] = t::payload(data.descriptions[2], scales);
    data.exponents.assign(256, 127);
    std::fill(data.activation.begin(), data.activation.end(), 1.);
    std::vector<std::uint64_t> a(data.activation.size(), t::encode(1., type));
    data.payloads[0] = t::payload(data.descriptions[0], a);data.calculate();
    auto program = prepare_artifact(context, data.operation, data.descriptions).value();
    auto slot = program.create_invocation().value();ASSERT_TRUE(slot.bind(data.upload(context)).ok());
    auto done = slot.submit().value();auto result = done.wait();ASSERT_TRUE(result) << result.error().message();
    data.check(context, result->get()->tensors()[0]);
  }
}

TEST(VulkanIree, OriginalMlxFloatControlSourceCapture) {
  auto context = vulkan_test::context().value();
  const auto &device = *v::detail::Access::runtime(context)->device;
  VkPhysicalDeviceFloatControlsProperties controls{};
  controls.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES;
  VkPhysicalDeviceProperties2 properties{};
  properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  properties.pNext = &controls;
  device.instance_fn.GetPhysicalDeviceProperties2(device.native.physical_device, &properties);
  std::fprintf(stdout, "MLX_FLOAT_CONTROL denorm16=%u denorm32=%u denorm64=%u signed32=%u independence=%u\n",
      controls.shaderDenormPreserveFloat16, controls.shaderDenormPreserveFloat32,
      controls.shaderDenormPreserveFloat64, controls.shaderSignedZeroInfNanPreserveFloat32,
      static_cast<unsigned>(controls.denormBehaviorIndependence));
}

TEST(VulkanIree, OriginalArtifactBatchPreservesSourceRoundingAndSixPorts) {
  vulkan_artifact_test::batch(prepare_artifact);
}

TEST(VulkanIree, OriginalArtifactColdResultBindsMatureConstantProjection) {
  vulkan_artifact_test::cold_projection(prepare_artifact,
      [](const v::Context &context, const c::TensorDesc &activation, const v::Tensor &weight) {
        return v::prepare(context, c::MatMul{.accumulation_type = c::DataType::kFloat32},
                          v::Argument(activation), v::Argument(weight));
      });
}

TEST(VulkanIree, OriginalArtifactChecksFeaturesDomainFailureAndCancellation) {
  using namespace vulkan_artifact_test;
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(), 256 * 1024 * 1024);
  {
    auto context = vulkan_test::context(&quota).value();
    Inputs data(Format::kNf4DoubleQuant, 3, 192, 64,
                c::DataType::kFloat32, c::DataType::kFloat32);
    auto &device = *v::detail::Access::runtime(context)->device;
    const auto limits = device.properties.limits;
    device.properties.limits.maxComputeWorkGroupInvocations = 63;
    auto rejected = prepare_artifact(context, data.operation, data.descriptions);
    ASSERT_FALSE(rejected);
    EXPECT_TRUE(rejected.error().IsUnimplemented());
    device.properties.limits = limits;
    auto wrong = data.descriptions;
    wrong[3] = t::dense(c::DataType::kFloat32, {255});
    EXPECT_FALSE(prepare_artifact(context, data.operation, wrong));
    auto program = prepare_artifact(context, data.operation, data.descriptions).value();
    {
      auto all_remaining = quota.try_reserve_bytes(quota.limit() - quota.used());
      ASSERT_TRUE(all_remaining);
      auto denied = program.create_invocation();
      ASSERT_FALSE(denied);
      EXPECT_TRUE(denied.error().IsResourceExhausted());
    }
    auto created = program.create_invocation();
    ASSERT_TRUE(created) << created.error().message();
    auto slot = std::move(*created);
    const auto inputs = data.upload(context);
    ASSERT_TRUE(slot.bind(inputs).ok());
    auto cancelled = slot.submit().value();
    ASSERT_TRUE(cancelled.request_cancel().ok());
    auto output = cancelled.wait();
    // A completed operation may win the race. Either outcome must retire all
    // six inputs; a cancelled result is never published.
    if (output) data.check(context, output->get()->tensors()[0]);
    else EXPECT_TRUE(output.error().IsCancelled());
    output = c::ResultLease<v::Results>{};
    cancelled = c::OwnedResult<v::Results>{};
    auto next = slot.submit().value();
    auto result = next.wait();
    ASSERT_TRUE(result) << result.error().message();
    data.check(context, result->get()->tensors()[0]);
    EXPECT_TRUE(context.close().ok());
    EXPECT_FALSE(slot.submit());
  }
  EXPECT_EQ(quota.used(), 0U);
}

TEST(VulkanIree, OriginalArtifactBoundedCauseCapture) {
  using namespace vulkan_artifact_test;
  auto context = vulkan_test::context().value();
  for (auto format : {Format::kNf4, Format::kNf4DoubleQuant, Format::kAwqGemm4,
                      Format::kCompressedTensorsSymmetric4}) {
    const int n = format == Format::kAwqGemm4 ? 16 : 3;
    Inputs data(format, n, 128, 64, c::DataType::kFloat32, c::DataType::kFloat32);
    auto program = prepare_artifact(context, data.operation, data.descriptions).value();
    const auto state = v::detail::Access::program(program);
    const auto spec = *state->dispatches.back().iree;
    std::fprintf(stdout, "ARTIFACT_CAUSE format=%u kind=%u n=%lld k=%lld group=%u ports=%zu\n",
                 static_cast<unsigned>(format), static_cast<unsigned>(spec.kind),
                 static_cast<long long>(spec.rows), static_cast<long long>(spec.columns),
                 spec.group_size, data.descriptions.size());
    auto created = program.create_invocation();
    ASSERT_TRUE(created) << created.error().message();
    auto slot = std::move(*created);
    const auto inputs = data.upload(context);
    ASSERT_TRUE(slot.bind(inputs).ok());
    auto work = slot.submit().value();
    auto result = work.wait();
    ASSERT_TRUE(result) << result.error().message();
    const auto actual = t::values(context, result->get()->tensors()[0]);
    for (int index : {0, 1, 63, 64, 127, 128, 255, 383})
      std::fprintf(stdout, "ARTIFACT_VALUE index=%d actual=%.9g expected=%.9g\n",
                   index, actual[index], data.expected[index]);
    for (std::size_t i = 0; i < inputs.size(); ++i)
      EXPECT_EQ(t::bytes(context, inputs[i]), data.payloads[i]) << i;
  }
}

TEST(VulkanIree, OriginalFloat8DirectShapesRepeatedSlotsAndQuotaRetire) {
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(),
                                  256 * 1024 * 1024);
  {
    auto context = vulkan_test::context(&quota).value();
    for (auto mode : {c::Float8ScaleMode::kFloat32Block128,
                      c::Float8ScaleMode::kUnsignedExponent8Group32})
      for (auto shape : {std::array{1, 128, 128}, std::array{3, 256, 128},
                         std::array{1, 32, 5}}) {
        const auto [m, k, n] = shape;
        if (k == 32 && mode == c::Float8ScaleMode::kFloat32Block128)
          continue;
        SCOPED_TRACE(::testing::Message() << m << '/' << k << '/' << n << '/'
                                          << static_cast<int>(mode));
        Float8Inputs data(m, k, n, mode), changed(m, k, n, mode, 1);
        auto program =
            v::prepare(context, c::MatMul{}, data.activation, data.weight);
        ASSERT_TRUE(program) << program.error().message();
        ASSERT_TRUE(
            v::detail::Access::program(*program)->dispatches.back().iree);
        auto first = program->create_invocation().value();
        auto second = program->create_invocation().value();
        auto a = context.upload(data.activation, data.a).value();
        auto w = context.upload(data.weight, data.w).value();
        auto b = context.upload(changed.activation, changed.a).value();
        auto z = context.upload(changed.weight, changed.w).value();
        ASSERT_TRUE(first.bind(std::array{a, w}).ok());
        ASSERT_TRUE(second.bind(std::array{b, z}).ok());
        v::Tensor retained;
        {
          auto first_work = first.submit().value();
          auto second_work = second.submit().value();
          auto second_result = second_work.wait();
          auto first_result = first_work.wait();
          ASSERT_TRUE(first_result) << first_result.error().message();
          ASSERT_TRUE(second_result) << second_result.error().message();
          t::expect(context, first_result->get()->tensors()[0], data.multiply(),
                    true);
          t::expect(context, second_result->get()->tensors()[0],
                    changed.multiply(), true);
        }
        for (int repeat = 0; repeat < 3; ++repeat) {
          auto work = (repeat == 0 ? first : second).submit().value();
          auto result = work.wait();
          ASSERT_TRUE(result) << result.error().message();
          t::expect(context, result->get()->tensors()[0],
                    repeat == 0 ? data.multiply() : changed.multiply(), true);
          if (repeat == 0)
            retained = result->get()->tensors()[0];
          else
            t::expect(context, retained, data.multiply(), true);
        }
        EXPECT_FALSE(first.submit());
        EXPECT_EQ(t::bytes(context, a), data.a);
        EXPECT_EQ(t::bytes(context, w), data.w);
      }
    EXPECT_TRUE(context.close().ok());
  }
  EXPECT_EQ(quota.used(), 0U);
}

TEST(VulkanIree, OriginalFloat8DirectComposesBFloat16WithoutExpandingInputs) {
  auto context = vulkan_test::context().value();
  for (auto mode : {c::Float8ScaleMode::kFloat32Block128,
                    c::Float8ScaleMode::kUnsignedExponent8Group32}) {
    Float8Inputs data(3, 128, 128, mode, 2);
    auto multiply =
        v::prepare(context, c::MatMul{}, data.activation, data.weight).value();
    auto convert =
        v::prepare(context, c::Cast{c::DataType::kBFloat16},
                   multiply.output_descriptions()[0])
            .value();
    const auto activation = context.upload(data.activation, data.a).value();
    const auto weight = context.upload(data.weight, data.w).value();
    const auto middle =
        context.allocate(multiply.output_descriptions()[0]).value();
    const auto output = context.allocate(convert.output_descriptions()[0]).value();
    auto recorder = v::Recorder::create(context).value();
    ASSERT_TRUE(recorder
                    .append(multiply, std::array{activation, weight},
                            std::array{middle})
                    .ok());
    ASSERT_TRUE(
        recorder.append(convert, std::array{middle}, std::array{output}).ok());
    auto recorded = recorder.finish().value();
    for (int repeat = 0; repeat < 2; ++repeat) {
      auto completion = recorded.submit().value();
      ASSERT_TRUE(completion.wait().ok());
      t::expect(context, output, data.multiply(), true);
    }
    EXPECT_EQ(t::bytes(context, activation), data.a);
    EXPECT_EQ(t::bytes(context, weight), data.w);
  }
}

TEST(VulkanIree,
     CheckedScanAutomaticallyAdvancesAndStopsBeforePublishingFailure) {
  auto context = vulkan_test::context().value();
  const auto desc = t::dense(c::DataType::kInt32, {2, 257});
  auto scan = v::prepare(context, c::Scan{}, desc).value();
  ASSERT_TRUE(v::detail::Access::program(scan)->dispatches.back().iree);
  auto copy = v::prepare(context, c::Copy{}, desc).value();
  const std::vector<std::int32_t> sentinel(514, 0x55aa55aa);
  for (bool fail : {false, true}) {
    auto values = s::integer_pattern(514);
    if (fail) {
      std::ranges::fill(values, 0);
      values[512] = INT32_MAX;
      values[513] = 1;
    }
    auto input = s::make_integer(context, desc, values);
    auto middle = s::make_integer(context, desc, sentinel);
    auto output = s::make_integer(context, desc, sentinel);
    auto recorder = v::Recorder::create(context).value();
    ASSERT_TRUE(
        recorder.append(scan, std::array{input}, std::array{middle}).ok());
    ASSERT_TRUE(
        recorder.append(copy, std::array{middle}, std::array{output}).ok());
    auto recorded = recorder.finish().value();
    auto work = recorded.submit().value();
    const auto &runtime = v::detail::Access::runtime(context);
    bool finished = false;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    // Inspect route state only: no Completion poll/wait/collect or GPU API is
    // allowed to drive the flag readback or downstream native submission.
    while (!finished && std::chrono::steady_clock::now() < deadline) {
      {
        std::lock_guard lock(runtime->mutex);
        ASSERT_NE(runtime->pending, nullptr);
        ASSERT_TRUE(runtime->pending->async);
        finished = runtime->pending->async->finished;
      }
      if (!finished)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_TRUE(finished);
    auto status = work.wait();
    if (fail) {
      EXPECT_TRUE(status.IsOutOfRange());
      EXPECT_EQ(s::integer_values(context, middle), sentinel);
      EXPECT_EQ(s::integer_values(context, output), sentinel);
    } else {
      ASSERT_TRUE(status.ok()) << status.message();
      s::expect_integer_prefix(context, output, desc, values, c::Scan{});
    }
    EXPECT_EQ(s::integer_values(context, input), values);
  }
}

TEST(VulkanIree, Float8RejectsMismatchedScalesAndUnavailableLaunchDomain) {
  auto context = vulkan_test::context().value();
  Float8Inputs block(1, 128, 128, c::Float8ScaleMode::kFloat32Block128);
  Float8Inputs group(1, 128, 128,
                     c::Float8ScaleMode::kUnsignedExponent8Group32);
  EXPECT_FALSE(
      v::prepare(context, c::MatMul{}, block.activation, group.weight));
  EXPECT_FALSE(
      v::prepare(context, c::Unary{c::UnaryKind::kAbs}, block.activation));
  auto &limits = v::detail::Access::runtime(context)->device->properties.limits;
  const auto saved = limits;
  limits.maxComputeWorkGroupInvocations = 63;
  auto rejected =
      v::prepare(context, c::MatMul{}, block.activation, block.weight);
  limits = saved;
  ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().IsUnimplemented());
}

TEST(VulkanIree, Float8BoundsUseNativeGridAndCombinedWorkspace) {
  auto context = vulkan_test::context().value();
  const auto mode = c::Float8ScaleMode::kFloat32Block128;
  const auto a = c::float8_activation_description(1, 128, mode).value();
  const auto w = c::float8_weight_description(128, 128, mode).value();
  auto &limits = v::detail::Access::runtime(context)->device->properties.limits;
  const auto saved = limits;
  // W conversion dispatches 128*ceil(128/64)=256 workgroups, not 16384.
  limits.maxComputeWorkGroupCount[0] = 255;
  auto grid_rejected = v::prepare(context, c::MatMul{}, a, w);
  limits = saved;
  ASSERT_FALSE(grid_rejected);
  EXPECT_TRUE(grid_rejected.error().IsUnimplemented());
  limits.maxComputeWorkGroupCount[0] = 256;
  auto grid_boundary = v::prepare(context, c::MatMul{}, a, w);
  limits = saved;
  ASSERT_TRUE(grid_boundary) << grid_boundary.error().message();

  // Each F32 matrix fits this range separately; their shared allocation does not.
  limits.maxStorageBufferRange = 4 * 128 * 128;
  auto workspace_rejected = v::prepare(context, c::MatMul{}, a, w);
  limits = saved;
  ASSERT_FALSE(workspace_rejected);
  EXPECT_TRUE(workspace_rejected.error().IsUnimplemented());
  limits.maxStorageBufferRange = 4 * (128 * 128 + 128);
  auto workspace_boundary = v::prepare(context, c::MatMul{}, a, w);
  limits = saved;
  ASSERT_TRUE(workspace_boundary) << workspace_boundary.error().message();

  const auto large_a = c::float8_activation_description(1, 65536, mode).value();
  const auto large_w = c::float8_weight_description(65536, 65536, mode).value();
  auto oversized = v::prepare(context, c::MatMul{}, large_a, large_w);
  ASSERT_FALSE(oversized);
  EXPECT_TRUE(oversized.error().IsUnimplemented() || oversized.error().IsOutOfRange());
}
} // namespace
