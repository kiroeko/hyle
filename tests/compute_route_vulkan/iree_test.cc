#include <chrono>
#include <thread>

#include "iree_test_support.h"
#include "artifact_test_support.h"
#include "mlx_test_support.h"
#include "official_test_support.h"
#include "iq1_test_support.h"
#include "scan_test_support.h"
#include "vulkan_program_internal.h"
#include "vulkan_iree.h"
#include "../compute_models/ct_asymmetric_int4_fixture.h"

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

TEST(VulkanIree, OriginalOfficialDynamicLinearsAndWeightTailDecode) {
  vulkan_official_test::linears(prepare_artifact);vulkan_official_test::decodes(prepare_artifact);
}
TEST(VulkanIree, OriginalIq1ExpertProgramsAndChangedInputs) {
  vulkan_iq1_test::batch(prepare_artifact);
}
TEST(VulkanIree, ImmutableMlxBanksShareNativeStorageAcrossInvocationSlots) {
  namespace mx = vulkan_mlx_test;
  namespace iv = c::iree_vulkan;
  auto context = vulkan_test::context().value();
  std::size_t aliased_ports = 0, copied_ports = 0, programs = 0;
  const auto exercise = [&](mx::Format format, mx::Kind kind, unsigned fallback) {
    SCOPED_TRACE(::testing::Message() << "format=" << int(format)
        << " kind=" << int(kind) << " fallback=" << fallback);
    const bool embedding = kind == mx::Kind::kEmbedding;
    const bool affine = format == mx::Format::kAffine;
    const auto type = affine && !embedding ? c::DataType::kFloat32 : c::DataType::kBFloat16;
    const auto scale_type = affine ? c::DataType::kBFloat16 : c::DataType::kUnknown;
    const unsigned bits = kind == mx::Kind::kLinear ? 5u : kind == mx::Kind::kGrouped ? 6u : 4u;
    mx::Inputs first(format, kind, type, 32, 0, scale_type, 64, bits);
    mx::Inputs changed(format, kind, type, 32, 1, scale_type, 64, bits);
    // Change only the actual dynamic A/IDs, keeping the original bank equation.
    changed.codes = first.codes; changed.scales = first.scales;
    changed.biases = first.biases; changed.exponents = first.exponents;
    changed.calculate();
    const std::size_t bank_begin = embedding ? 0u : 1u;
    const std::size_t bank_count = affine ? 3u : 2u;
    const std::size_t fallback_port = bank_begin + 1;
    auto resident = first.upload(context);
    if (fallback == 1) {
      auto reshape = c::View::reshape(c::TensorShape::create(
          first.descriptions[fallback_port].dimensions()).value()).value();
      resident[fallback_port] = resident[fallback_port].view(reshape).value();
    } else if (fallback == 2) {
      const auto &original = resident[fallback_port];
      resident[fallback_port] = context.borrow_ready(first.descriptions[fallback_port],
          original.native_buffer(), original.native_offset(),
          first.descriptions[fallback_port].required_bytes().value(), original.native_owner()).value();
    }
    std::vector<v::Argument> arguments;
    std::vector<v::Tensor> initial_inputs, changed_inputs;
    std::array<VkBuffer, 3> handles{};
    std::array<VkDeviceSize, 3> origins{};
    std::array<std::weak_ptr<const void>, 3> lifetimes;
    for (std::size_t port = 0; port < resident.size(); ++port) {
      if (port >= bank_begin && port < bank_begin + bank_count) {
        const auto bank = port - bank_begin;
        handles[bank] = resident[port].native_buffer();
        origins[bank] = resident[port].native_offset();
        lifetimes[bank] = resident[port].native_owner();
        arguments.emplace_back(resident[port]);
      } else {
        arguments.emplace_back(first.descriptions[port]);
        initial_inputs.push_back(resident[port]);
        changed_inputs.push_back(context.upload(changed.descriptions[port], changed.payloads[port]).value());
      }
    }
    auto program = v::prepare(context, first.operation, std::span<const v::Argument>(arguments));
    ASSERT_TRUE(program) << program.error().message();
    EXPECT_EQ(program->input_descriptions().size(), initial_inputs.size());
    EXPECT_EQ(std::ranges::count_if(v::detail::Access::program(*program)->dispatches,
        [](const auto &dispatch) { return dispatch.iree && iv::mlx_program(dispatch.iree->kind); }), 1);
    std::array slots{program->create_invocation().value(), program->create_invocation().value()};
    for (const auto &slot : slots) {
      const auto &run = v::detail::Access::invocation(slot)->run;
      const v::detail::IreeStage *selected = nullptr;
      for (const auto &stage : run->stages) {
        if (stage.iree) {
          ASSERT_EQ(selected, nullptr);
          selected = stage.iree.get();
          ASSERT_EQ(selected->input_count, first.descriptions.size());
        }
      }
      ASSERT_NE(selected, nullptr);
      EXPECT_EQ(selected->constant_copy != nullptr, fallback != 0);
      for (std::size_t bank = 0; bank < bank_count; ++bank) {
        const auto port = bank_begin + bank;
        const bool borrowed = fallback == 0 || port != fallback_port;
        const auto input = selected->program->input(port);
        ASSERT_TRUE(input) << input.error().message();
        EXPECT_TRUE(selected->inputs[port].constant);
        EXPECT_EQ(selected->inputs[port].borrowed, borrowed);
        EXPECT_EQ(input->bytes, first.descriptions[port].required_bytes().value());
        if (borrowed) {
          EXPECT_EQ(input->buffer, handles[bank]);
          EXPECT_EQ(input->offset, origins[bank]);
          aliased_ports += selected->inputs[port].borrowed;
        } else {
          EXPECT_NE(input->buffer, handles[bank]);
          ++copied_ports;
        }
      }
      for (std::size_t port = 0; port < selected->input_count; ++port)
        if (port < bank_begin || port >= bank_begin + bank_count) {
          EXPECT_FALSE(selected->inputs[port].constant);
          EXPECT_FALSE(selected->inputs[port].borrowed);
        }
    }
    arguments.clear(); resident.clear(); *program = {};
    for (std::size_t bank = 0; bank < bank_count; ++bank) EXPECT_FALSE(lifetimes[bank].expired());
    v::Tensor retained;
    for (unsigned pass = 0; pass < 3; ++pass) {
      auto &slot = slots[pass == 0 ? 0 : 1];
      ASSERT_TRUE(slot.bind(pass == 1 ? changed_inputs : initial_inputs).ok());
      auto work = slot.submit().value();
      auto result = work.wait();
      ASSERT_TRUE(result) << result.error().message();
      (pass == 1 ? changed : first).check(context, result->get()->tensors()[0]);
      if (pass == 0) retained = result->get()->tensors()[0];
      else first.check(context, retained);
    }
    EXPECT_FALSE(slots[0].submit()); // Original output still protects its slot.
    slots = {};
    first.check(context, retained);
    retained = {};
    for (std::size_t bank = 0; bank < bank_count; ++bank) EXPECT_TRUE(lifetimes[bank].expired());
    ++programs;
  };
  for (const auto format : {mx::Format::kAffine, mx::Format::kMxfp4, mx::Format::kMxfp8})
    for (const auto kind : {mx::Kind::kLinear, mx::Kind::kGrouped, mx::Kind::kEmbedding}) {
      if (format != mx::Format::kAffine && kind == mx::Kind::kEmbedding) continue;
      exercise(format, kind, 0);
    }
  exercise(mx::Format::kAffine, mx::Kind::kLinear, 1);
  exercise(mx::Format::kAffine, mx::Kind::kLinear, 2);
  RecordProperty("mlx_bank_programs", static_cast<int>(programs));
  RecordProperty("mlx_native_bank_aliases", static_cast<int>(aliased_ports));
  RecordProperty("mlx_native_bank_copy_fallbacks", static_cast<int>(copied_ports));
  EXPECT_EQ(programs, 9u);
  EXPECT_EQ(aliased_ports, 42u);
  EXPECT_EQ(copied_ports, 4u);
}
TEST(VulkanIree, MixedMlxGroupedKeepsOriginalTypedBanksAcrossExpertCounts) {
  auto context = vulkan_test::context().value();
  for (const std::int64_t experts : {3, 17}) {
    SCOPED_TRACE(experts);
    mlx_grouped_shared_test::Case data(c::DataType::kFloat32, false);
    data.descriptions[1] = c::TensorDesc::dense(c::DataType::kUInt32,
        std::array<std::int64_t, 3>{experts, 2, 10}).value();
    data.payloads[1].resize(static_cast<std::size_t>(experts * 2 * 10 * 4));
    for (const std::size_t port : {2u, 3u}) {
      data.descriptions[port] = c::TensorDesc::dense(c::DataType::kBFloat16,
          std::array<std::int64_t, 3>{experts, 2, 2}).value();
      data.payloads[port] = mlx_grouped_shared_test::floating(c::DataType::kBFloat16,
          std::vector<float>(static_cast<std::size_t>(experts * 4), port == 2 ? .5F : -.25F));
    }
    auto program = prepare_artifact(context, data.operation, data.descriptions);
    ASSERT_TRUE(program) << program.error().message();
    const auto state = v::detail::Access::program(*program);
    const auto selected = std::find_if(state->dispatches.begin(), state->dispatches.end(),
        [](const auto &dispatch) {
          return dispatch.iree && dispatch.iree->kind ==
              c::iree_vulkan::ProgramKind::MlxAffineGrouped;
        });
    ASSERT_NE(selected, state->dispatches.end());
    // No expanded F32 scale/bias bank is needed to widen the selected values.
    for (const std::size_t port : {2u, 3u}) {
      EXPECT_EQ(selected->values[port], state->inputs[port]);
      EXPECT_EQ(state->values[selected->values[port]].description.data_type(),
                c::DataType::kBFloat16);
    }
    std::size_t expanded_bytes = 0;
    for (const auto &value : state->values) {
      const auto &description = value.description;
      if (description.data_type() == c::DataType::kFloat32 && description.rank() == 3 &&
          std::ranges::equal(description.dimensions(),
              std::array<std::int64_t, 3>{experts, 2, 2}))
        expanded_bytes += description.required_bytes().value();
    }
    RecordProperty("expanded_scale_bias_bytes_E" + std::to_string(experts),
                   static_cast<int>(expanded_bytes));
    EXPECT_EQ(expanded_bytes, 0u);
    auto slot = program->create_invocation().value();
    std::vector<v::Tensor> inputs;
    for (std::size_t port = 0; port < data.descriptions.size(); ++port) {
      auto uploaded = context.upload(data.descriptions[port], data.payloads[port]);
      ASSERT_TRUE(uploaded) << uploaded.error().message();
      inputs.push_back(std::move(*uploaded));
    }
    ASSERT_TRUE(slot.bind(inputs).ok());
    auto work = slot.submit().value();
    auto result = work.wait();
    ASSERT_TRUE(result) << result.error().message();
    std::vector<float> actual(data.expected.size());
    ASSERT_TRUE(context.download(result->get()->tensors()[0],
        std::as_writable_bytes(std::span(actual))).ok());
    EXPECT_EQ(actual, data.expected);
  }
}
TEST(VulkanIree, OriginalIq1ActualPackedGeometryAndFeatureBoundaries) {
  vulkan_iq1_test::domains(prepare_artifact);
}
TEST(VulkanIree, OriginalIq1ActualFullBanksReplayAndEdgeIndices) {
  vulkan_iq1_test::actual_banks(prepare_artifact);
}
TEST(VulkanIree, OriginalIq1IndependentLeaseCancelCloseAndOom) {
  vulkan_iq1_test::lifecycle(prepare_artifact);
}
TEST(VulkanIree, OriginalOfficialGroupedFp4Fp8SharedDistinctAndChecked) {
  vulkan_official_test::grouped(prepare_artifact);
}
TEST(VulkanIree, OriginalOfficialGroupedTailRowsKeepBlockScaleOrder) {
  vulkan_official_test::grouped(prepare_artifact, 129);
}
TEST(VulkanIree, OriginalOfficialQatFiniteBfloatExhaustiveAndFailure) {
  vulkan_official_test::roundtrips(prepare_artifact);
}
TEST(VulkanIree, OriginalOfficialIndependentLeaseFiniteCancelCloseAndOom) {
  vulkan_official_test::lifecycle(prepare_artifact);
}
TEST(VulkanIree, OriginalOfficialEncodedBlock128TailAndReplay) {
  auto context = vulkan_test::context().value();
  for (int n : {5, 129}) {
    Float8Inputs a(3, 256, n, c::Float8ScaleMode::kFloat32Block128), b(3, 256, n, c::Float8ScaleMode::kFloat32Block128, 1);
    const std::array descriptions{a.activation, a.weight};
    auto program = prepare_artifact(context, c::Operation(c::MatMul{.accumulation_type=c::DataType::kFloat32}), descriptions);
    ASSERT_TRUE(program) << program.error().message();auto slot = program->create_invocation().value();
    for (const auto *data : {&a, &b}) {
      const std::array inputs{context.upload(data->activation, data->a).value(), context.upload(data->weight, data->w).value()};
      ASSERT_TRUE(slot.bind(inputs).ok());auto work = slot.submit().value();auto result = work.wait();
      ASSERT_TRUE(result) << result.error().message();t::expect(context, result->get()->tensors()[0], data->multiply(), true);
    }
  }
}
TEST(VulkanIree, OriginalOfficialBfloatFloatProgramCause) {
  namespace o = vulkan_official_test;
  auto context = vulkan_test::context().value();
  o::Weight weight(1, 5, 384);
  auto values = o::activation(3, 384, c::DataType::kBFloat16, 0);
  std::vector<double> stored(values.begin(), values.end());
  for (bool power : {false, true}) {
    const auto floor = power ? 1e-4f : 1e-10f;
    const auto operation = c::model_ops::float8_block_linear(power ? c::model_ops::Float8ScalePolicy::kPowerOfTwo : c::model_ops::Float8ScalePolicy::kDivide, floor);
    const auto expected = o::reference(weight, o::quantize(values, 384, floor, power), 3, 1, 1, {}, c::DataType::kFloat32);
    for (auto type : {c::DataType::kFloat32, c::DataType::kBFloat16}) {
      const auto ad = t::dense(type, {3, 384});const std::array desc{ad, weight.description};
      auto program = prepare_artifact(context, operation, desc).value();auto slot = program.create_invocation().value();
      ASSERT_TRUE(slot.bind(std::array{t::make(context, ad, stored), context.upload(weight.description, weight.bytes).value()}).ok());
      auto work = slot.submit().value();auto result = work.wait();ASSERT_TRUE(result);
      const auto actual = t::values(context, result->get()->tensors()[0]);
      double worst = 0;for (std::size_t i = 0; i < actual.size(); ++i) worst = std::max(worst, std::abs(actual[i] - t::rounded(expected[i], type)));
      std::fprintf(stdout, "OFFICIAL_DTYPE_CAUSE type=%u power=%u max_error=%.9g first=%.9g expected=%.9g\n", static_cast<unsigned>(type), power, worst, actual[0], t::rounded(expected[0], type));
      if (type == c::DataType::kFloat32) t::expect(context, result->get()->tensors()[0], expected);
    }
  }
}

TEST(VulkanIree, MlxGroupedSharesOriginalActivationAcrossSelectedExperts) {
  vulkan_mlx_test::grouped_shared_activation(prepare_artifact);
}

TEST(VulkanIree, OriginalMlxBatchPreservesGroupsSourceTypesAndCheckedSlots) {
  vulkan_mlx_test::batch(prepare_artifact);
}

TEST(VulkanIree, OriginalMlxAffineWidthsUseContinuousBitstreamsAndCheckedRebinding) {
  vulkan_mlx_test::affine_widths(prepare_artifact);
}

TEST(VulkanIree, OriginalMlxAffineWidthsKeepDecodedWeightsInF32BeforeDot) {
  using namespace vulkan_mlx_test;
  auto context = vulkan_test::context().value();
  for (unsigned bits : {4U, 5U, 6U, 8U})
    for (auto type : {c::DataType::kFloat16, c::DataType::kBFloat16}) {
      SCOPED_TRACE(::testing::Message() << "bits=" << bits << " type=" << int(type));
      Inputs data(Format::kAffine, Kind::kLinear, type, 64, 0,
                  c::DataType::kUnknown, 128, bits);
      std::fill(data.codes.begin(), data.codes.end(), 0);
      for (int col = 0; col < data.n; ++col) {
        data.codes[col * data.k] = 1;data.codes[col * data.k + 1] = 2;
      }
      data.repack();
      const double bias = type == c::DataType::kBFloat16 ? 256.0 : 2048.0;
      std::fill(data.scales.begin(), data.scales.end(), 1);
      std::fill(data.biases.begin(), data.biases.end(), bias);
      std::vector<std::uint64_t> scales(data.scales.size(), t::encode(1, type));
      std::vector<std::uint64_t> biases(data.biases.size(), t::encode(bias, type));
      data.payloads[2] = t::payload(data.descriptions[2], scales);
      data.payloads[3] = t::payload(data.descriptions[3], biases);
      std::fill(data.activation.begin(), data.activation.end(), 0);
      for (int row = 0; row < data.rows; ++row) {
        data.activation[row * data.k] = 1;data.activation[row * data.k + 1] = -1;
      }
      std::vector<std::uint64_t> raw(data.activation.size());
      for (std::size_t i = 0; i < raw.size(); ++i) raw[i] = t::encode(data.activation[i], type);
      data.payloads[0] = t::payload(data.descriptions[0], raw);data.calculate();
      ASSERT_EQ(data.expected, std::vector<double>(data.expected.size(), -1.0));
      auto program = prepare_artifact(context, data.operation, data.descriptions).value();
      auto slot = program.create_invocation().value();ASSERT_TRUE(slot.bind(data.upload(context)).ok());
      auto done = slot.submit().value();auto result = done.wait();
      ASSERT_TRUE(result) << result.error().message();
      // Premature F16/BF16 weight rounding would give -2; this witness is exact.
      t::expect(context, result->get()->tensors()[0], data.expected, true);
    }
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
    Inputs data(Format::kAffine, Kind::kEmbedding, c::DataType::kBFloat16, 64);
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

TEST(VulkanIree, OriginalAsymmetricArtifactKeepsPaddedZerosAndSourceRoundsAcrossSlots) {
  using F = ct_asymmetric_test::Fixture;
  auto context = vulkan_test::context().value();
  const F original, changed(1);
  for (auto source : vulkan_artifact_test::types)
    for (auto output : vulkan_artifact_test::types) {
      SCOPED_TRACE(::testing::Message() << "source=" << int(source)
                                      << " output=" << int(output));
      const std::array descriptions{
          t::dense(c::DataType::kInt32, {F::rows, F::columns / 8}),
          t::dense(c::DataType::kInt32, {(F::rows + 7) / 8, F::groups}),
          t::dense(source, {F::rows, F::groups})};
      const auto upload = [&](const F &data) {
        const auto packed = [&](const auto &words, std::size_t port) {
          std::vector<std::uint64_t> raw;
          for (auto word : words) raw.push_back(std::bit_cast<std::uint32_t>(word));
          return context.upload(descriptions[port], t::payload(descriptions[port], raw)).value();
        };
        return std::array{packed(data.codes, 0), packed(data.zero_points, 1),
                          t::make(context, descriptions[2], data.scales)};
      };
      const auto check = [&](const v::Tensor &actual, const F &data) {
        EXPECT_EQ(actual.description(), t::dense(output, {F::rows, F::columns}));
        const auto expected = data.expected([&](double x) { return t::rounded(x, source); });
        std::vector<std::uint64_t> raw;
        for (auto value : expected) raw.push_back(t::encode(value, output));
        EXPECT_EQ(t::bits(context, actual), raw);
      };
      const auto operation = c::model_ops::artifact_weight_decode(
          c::model_ops::WeightArtifactFormat::kCompressedTensorsAsymmetric4,
          F::group, output);
      auto created = prepare_artifact(context, operation, descriptions);
      ASSERT_TRUE(created) << created.error().message();
      auto first = created->create_invocation().value();
      auto second = created->create_invocation().value();
      ASSERT_TRUE(first.bind(upload(original)).ok());
      ASSERT_TRUE(second.bind(upload(changed)).ok());
      v::Tensor retained;
      {
        auto a = first.submit().value(), b = second.submit().value();
        auto ar = a.wait(), br = b.wait();
        ASSERT_TRUE(ar) << ar.error().message();
        ASSERT_TRUE(br) << br.error().message();
        retained = ar->get()->tensors()[0];
        check(retained, original);
        check(br->get()->tensors()[0], changed);
      }
      ASSERT_TRUE(second.bind(upload(original)).ok());
      auto replay = second.submit().value();
      auto replayed = replay.wait();
      ASSERT_TRUE(replayed) << replayed.error().message();
      check(replayed->get()->tensors()[0], original);
      check(retained, original);
    }
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
