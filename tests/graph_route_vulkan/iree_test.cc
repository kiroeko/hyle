#include "../compute_route_vulkan/iree_test_support.h"
#include "../compute_route_vulkan/artifact_test_support.h"
#include "../compute_route_vulkan/mlx_test_support.h"
#include "../compute_route_vulkan/official_test_support.h"
#include "../compute_route_vulkan/iq1_test_support.h"
#include "../compute_route_vulkan/scan_test_support.h"
#include "hyle/graph/vulkan.h"
#include "../../src/compute_route_vulkan/vulkan_iree.h"
#include "../../src/compute_route_vulkan/vulkan_memory_internal.h"
#include "../compute_models/mlx_moe_fixture.h"

namespace {
using namespace vulkan_iree_test;
namespace g = hyle::graph;
namespace gv = g::vulkan;
namespace t = vulkan_tensor_test;
namespace s = vulkan_scan_test;

hyle::Result<v::Program> compile_artifact(const v::Context &context,
                                        c::Operation operation,
                                        std::span<const c::TensorDesc> descriptions) {
  auto builder = g::GraphBuilder::create().value();
  std::vector<g::ValueId> inputs;
  for (const auto &description : descriptions)
    inputs.push_back(builder.input(description).value());
  const auto node = builder.add_node(operation, inputs).value();
  const auto result = builder.result(node).value();
  EXPECT_TRUE(builder.publish(result).ok());
  return gv::compile(context, std::move(builder).finish().value());
}

TEST(VulkanGraph, OriginalOfficialDynamicLinearsAndWeightTailDecode) {
  vulkan_official_test::linears(compile_artifact);vulkan_official_test::decodes(compile_artifact);
}
TEST(VulkanGraph, OrdinaryMixedMlxMoeUsesSharedInputsAndOnlyPublishesFinalResults) {
  auto context = vulkan_test::context().value();
  ASSERT_FALSE(context.environment().software_device);
  RecordProperty("moe_device", context.environment().device_name);
  auto device = v::MemoryResource::create(context, v::MemoryKind::kDevice).value();
  auto staging = v::MemoryResource::create(context, v::MemoryKind::kStaging).value();
  // Existing native F2 slots handle setup and observed results. The tested
  // banks, inputs, intermediate values and output remain true device storage.
  constexpr std::size_t transfer_bytes = 17u * 64u * 12u * sizeof(std::uint32_t);
  auto transfer = v::TransferSlot::create(context, staging, transfer_bytes).value();
  auto host = hyle::mem::Buffer::create(hyle::mem::system_resource(), transfer_bytes, 64).value();
  const auto byte_description = [](std::size_t bytes) {
    return c::TensorDesc::dense(c::DataType::kUInt8,
        std::array<std::int64_t, 1>{static_cast<std::int64_t>(bytes)});
  };
  const auto write = [&](hyle::mem::BufferView destination,
                         std::span<const std::byte> payload) -> hyle::Status {
    HYLE_ASSIGN_OR_RETURN(auto description, byte_description(payload.size()));
    HYLE_ASSIGN_OR_RETURN(auto host_view, host.view().subview(0, payload.size()));
    HYLE_RETURN_IF_ERROR(host_view.write(payload));
    HYLE_ASSIGN_OR_RETURN(auto target, c::ConstTensorView::create(description, destination));
    HYLE_ASSIGN_OR_RETURN(auto native, v::detail::MemoryAccess::tensor_view(
        v::detail::Access::runtime(context), target, hyle::mem::system_allocator()));
    HYLE_ASSIGN_OR_RETURN(auto source, c::ConstTensorView::create(description, host_view));
    HYLE_ASSIGN_OR_RETURN(auto done, transfer.upload(native, source));
    return done.wait();
  };
  const auto read = [&](hyle::mem::ConstBufferView source,
                        std::span<std::byte> payload) -> hyle::Status {
    HYLE_ASSIGN_OR_RETURN(auto description, byte_description(payload.size()));
    HYLE_ASSIGN_OR_RETURN(auto view, c::ConstTensorView::create(description, source));
    HYLE_ASSIGN_OR_RETURN(auto native, v::detail::MemoryAccess::tensor_view(
        v::detail::Access::runtime(context), view, hyle::mem::system_allocator()));
    HYLE_ASSIGN_OR_RETURN(auto done, transfer.download(native));
    HYLE_RETURN_IF_ERROR(done.wait());
    HYLE_ASSIGN_OR_RETURN(auto host_view, host.view().subview(0, payload.size()));
    HYLE_ASSIGN_OR_RETURN(auto target, c::TensorView::create(description, host_view));
    HYLE_RETURN_IF_ERROR(transfer.read(target));
    return host_view.read(payload);
  };
  const std::array domains{g::StorageDomain{device.ref()}};
  const std::array targets{g::ExecutionTarget{
      gv::execution_source(context).value(), g::StorageDomainId{0}}};
  mlx_moe_test::run(device.ref(), {},
      [&](const g::Graph &graph, g::CompileOptions options) {
        return g::compile(graph, {.storage_domains = domains, .targets = targets}, std::move(options));
      }, write, read);
}
TEST(VulkanGraph, DenseOfficialBlocksShareOneWeightAcrossTokensAndSlots) {
  namespace o = vulkan_official_test;
  auto context = vulkan_test::context().value();
  constexpr int rows = 3, columns = 129, depth = 256;
  constexpr std::array<float, 8> values{1, -1, 2, -2, .5f, -.5f, 3, -3};
  for (auto type : {c::DataType::kFloat32, c::DataType::kBFloat16})
    for (const bool power : {false, true}) {
      SCOPED_TRACE(::testing::Message() << int(type) << '/' << power);
      o::Weight first(1, columns, depth), second(1, columns, depth, false, 1);
      const auto activation = t::dense(type, {rows, depth});
      const auto operation = c::model_ops::float8_block_linear(power
          ? c::model_ops::Float8ScalePolicy::kPowerOfTwo
          : c::model_ops::Float8ScalePolicy::kDivide);
      auto compiled = compile_artifact(context, operation, std::array{activation, first.description});
      ASSERT_TRUE(compiled) << compiled.error().message();
      auto program = std::move(*compiled);
      std::array slots{program.create_invocation().value(), program.create_invocation().value()};
      for (int phase = 0; phase < 2; ++phase) {
        std::vector<double> input(rows * depth), expected(rows * columns);
        for (std::size_t i = 0; i < input.size(); ++i)
          input[i] = i % 128 == 0 ? 448. : values[(i + phase * 3) % values.size()];
        const auto &weight = phase ? second : first;
        for (int row = 0; row < rows; ++row) for (int column = 0; column < columns; ++column) {
          double total = 0;
          for (int k = 0; k < depth; ++k)
            total += input[row * depth + k] * double(weight.values[column * depth + k]) * weight.scale(0, column, k / 128);
          expected[row * columns + column] = t::rounded(total, type);
        }
        // Every activation group has max448: its original quantization scale
        // is exactly1. The independent FP64 products/scales are all dyadic.
        const std::array inputs{t::make(context, activation, input), context.upload(weight.description, weight.bytes).value()};
        for (auto &slot : slots) {
          ASSERT_TRUE(slot.bind(inputs).ok());
          auto work = slot.submit().value(); auto result = work.wait();
          ASSERT_TRUE(result) << result.error().message();
          t::expect(context, result->get()->tensors()[0], expected, true);
        }
      }
    }
}
TEST(VulkanGraph, OriginalIq1ExpertProgramsAndChangedInputs) {
  vulkan_iq1_test::batch(compile_artifact);
}
TEST(VulkanGraph, OriginalIq1ActualPackedGeometryAndFeatureBoundaries) {
  vulkan_iq1_test::domains(compile_artifact);
}
TEST(VulkanGraph, OriginalIq1ActualFullBanksReplayAndEdgeIndices) {
  vulkan_iq1_test::actual_banks(compile_artifact);
}
TEST(VulkanGraph, OriginalIq1IndependentLeaseCancelCloseAndOom) {
  vulkan_iq1_test::lifecycle(compile_artifact);
}
TEST(VulkanGraph, OriginalOfficialGroupedFp4Fp8SharedDistinctAndChecked) {
  vulkan_official_test::grouped(compile_artifact);
}

TEST(VulkanGraph, OfficialConstantBanksBorrowOwnedStorageAcrossSlots) {
  namespace o = vulkan_official_test;
  using v::detail::Access;
  for (const auto shape : {std::array{0, 256}, std::array{0, 128},
                           std::array{1, 128}}) {
    const bool four = shape[0] != 0;
    const int k = shape[1];
    SCOPED_TRACE(::testing::Message() << "fp4=" << four << "/k=" << k);
    auto context = vulkan_test::context().value();
    o::Weight weights(3, 7, k, four);
    const auto type = four ? c::DataType::kBFloat16 : c::DataType::kFloat32;
    const auto ad = t::dense(type, {2, 1, k});
    const auto id = t::dense(c::DataType::kInt32, {2, 2});
    auto resident = context.upload(weights.description, weights.bytes).value();
    std::weak_ptr<const void> lifetime = resident.native_owner();
    const auto handle = resident.native_buffer();
    auto builder = g::GraphBuilder::create().value();
    const auto w = builder.constant(weights.description).value();
    const auto a = builder.input(ad).value();
    const auto ids = builder.input(id).value();
    const auto operation = four ? c::model_ops::grouped_float4_linear()
                                : c::model_ops::grouped_float8_linear();
    const auto node = builder.add_node(operation, std::array{w, a, ids});
    ASSERT_TRUE(node) << node.error().message();
    const auto result = builder.result(*node).value();
    ASSERT_TRUE(builder.publish(result).ok());
    auto graph = std::move(builder).finish().value();
    auto program = gv::compile(context, graph, std::array{resident});
    ASSERT_TRUE(program) << program.error().message();
    auto &limits = Access::runtime(context)->device->properties.limits;
    const auto native_alignment = limits.minStorageBufferOffsetAlignment;
    // This GPU may naturally align every K128 bank. A stricter cached
    // descriptor bound exercises the scale-only copy fallback on the same data.
    if (four)
      limits.minStorageBufferOffsetAlignment =
          (std::max)(VkDeviceSize{256}, native_alignment);
    auto first = program->create_invocation().value();
    auto second = program->create_invocation().value();
    std::array<v::detail::IreeStage *, 2> stages{};
    int slot_index = 0;
    for (const auto *slot : {&first, &second}) {
      const auto &run = Access::invocation(*slot)->run;
      for (const auto &stage : run->stages)
        if (stage.iree && stage.iree->program->input_count() == 4) {
          ASSERT_EQ(stages[slot_index], nullptr);
          stages[slot_index] = stage.iree.get();
        }
      ASSERT_NE(stages[slot_index], nullptr);
      auto &stage = *stages[slot_index++];
      const auto codes = stage.program->input(1).value();
      EXPECT_TRUE(stage.inputs[1].borrowed);
      EXPECT_EQ(codes.buffer, handle);
      EXPECT_EQ(codes.offset, 0U);
      EXPECT_EQ(codes.bytes, std::size_t(3 * 7 * k / (four ? 2 : 1)));
      const auto scale = stage.program->input(2).value();
      const auto alignment = (std::max)(VkDeviceSize{64},
          run->runtime->device->properties.limits.minStorageBufferOffsetAlignment);
      const bool aligned_scale = codes.bytes % alignment == 0;
      EXPECT_EQ(stage.inputs[2].borrowed, aligned_scale);
      if (aligned_scale) {
        EXPECT_EQ(scale.buffer, handle);
        EXPECT_EQ(scale.offset, codes.bytes);
        EXPECT_EQ(stage.constant_copy, nullptr);
      } else {
        EXPECT_NE(scale.buffer, handle);
        EXPECT_NE(stage.constant_copy, nullptr);
      }
    }
    EXPECT_EQ(stages[0]->program->input(1)->buffer,
              stages[1]->program->input(1)->buffer);
    std::printf("OFFICIAL_CONSTANT fp4=%d k=%d slots=2 codesBytes=%zu "
                "codeAliases=2 scaleAliases=%d alignment=%llu nativeAlignment=%llu\n", int(four), k,
                stages[0]->program->input(1)->bytes,
                stages[0]->inputs[2].borrowed ? 2 : 0,
                static_cast<unsigned long long>(limits.minStorageBufferOffsetAlignment),
                static_cast<unsigned long long>(native_alignment));
    limits.minStorageBufferOffsetAlignment = native_alignment;
    if (!four && k == 256) {
      namespace iv = c::iree_vulkan;
      const auto &runtime = Access::runtime(context);
      const auto &dispatches = Access::program(*program)->dispatches;
      const auto dispatch = std::ranges::find_if(dispatches,
          [](const auto &item) { return item.iree && iv::official_grouped(item.iree->kind); });
      ASSERT_NE(dispatch, dispatches.end());
      const iv::BorrowedInput source{1, runtime->device->native.device,
          weights.bytes.size(), {handle, 0, weights.bytes.size()}, resident.native_owner()};
      const std::array duplicate{source, source};
      auto rejected = iv::NativeProgram::create(runtime->device->iree, *dispatch->iree, duplicate);
      ASSERT_FALSE(rejected);
      EXPECT_EQ(rejected.error().code(), hyle::StatusCode::InvalidArgument);
      auto invalid = source;
      invalid.index = 4;
      rejected = iv::NativeProgram::create(runtime->device->iree, *dispatch->iree, std::span(&invalid, 1));
      ASSERT_FALSE(rejected);
      EXPECT_EQ(rejected.error().code(), hyle::StatusCode::InvalidArgument);
      invalid = source;
      invalid.storage.bytes = 1;
      rejected = iv::NativeProgram::create(runtime->device->iree, *dispatch->iree, std::span(&invalid, 1));
      ASSERT_FALSE(rejected);
      EXPECT_EQ(rejected.error().code(), hyle::StatusCode::InvalidArgument);
    }
    graph = {};
    *program = {};
    resident = {};
    ASSERT_FALSE(lifetime.expired());
    v::Tensor retained;
    std::vector<double> retained_reference;
    for (int pass = 0; pass < 3; ++pass) {
      auto &slot = pass == 0 ? first : second;
      const auto x = o::activation(2, k, type, pass);
      const std::array<std::int32_t, 4> selected =
          pass == 1 ? std::array<std::int32_t, 4>{2, 0, 1, 2}
                    : std::array<std::int32_t, 4>{0, 1, 2, 2};
      const std::vector<double> values(x.begin(), x.end());
      const std::array inputs{t::make(context, ad, values),
          context.upload(id, std::as_bytes(std::span(selected))).value()};
      ASSERT_TRUE(slot.bind(inputs).ok());
      auto work = slot.submit().value();
      auto output = work.wait();
      ASSERT_TRUE(output) << output.error().message();
      const auto expected = o::reference(weights,
          o::quantize(x, k, four ? 1e-4f : 1e-10f, four),
          2, 1, 2, selected, type);
      t::expect(context, output->get()->tensors()[0], expected);
      if (pass == 0) {
        retained = output->get()->tensors()[0];
        retained_reference = expected;
      } else {
        t::expect(context, retained, retained_reference);
      }
    }
    first = {};
    second = {};
    retained = {};
    EXPECT_TRUE(lifetime.expired());
    ASSERT_TRUE(context.close().ok());
  }
}
TEST(VulkanGraph, OfficialFullGlmFp8BankKeepsWordAndScaleAddresses) {
  namespace o = vulkan_official_test;
  using v::detail::Access;
  constexpr std::int64_t experts = 256, columns = 2048, depth = 6144;
  constexpr std::size_t expert_codes = columns * depth;
  constexpr std::size_t codes_bytes = experts * expert_codes;
  constexpr std::size_t expert_scales = (columns / 128) * (depth / 128) * sizeof(float);
  constexpr std::size_t bank_bytes = codes_bytes + experts * expert_scales;
  constexpr std::int32_t highest = experts - 1;
  auto context = vulkan_test::context().value();
  const auto &limits = Access::runtime(context)->device->properties.limits;
  if (limits.maxStorageBufferRange < bank_bytes)
    GTEST_SKIP() << "The real GLM bank exceeds this device's descriptor range";
  const auto description =
      c::float8_grouped_weight_description(experts, columns, depth).value();
  ASSERT_EQ(description.required_bytes().value(), bank_bytes);
  // The narrow carrier admission must not admit ordinary large U8 tensors,
  // nor a second byte-indexed consumer of an otherwise qualified bank.
  for (const auto &copy_description : std::array{
           description, t::dense(c::DataType::kUInt8, {experts, columns, depth})}) {
    auto rejected_builder = g::GraphBuilder::create().value();
    const auto source = rejected_builder.input(copy_description).value();
    const auto copied = rejected_builder.add_node(c::Operation(c::Copy{}),
                                                 std::span(&source, 1));
    if (!copied) {
      EXPECT_TRUE(copied.error().IsInvalidArgument() || copied.error().IsUnimplemented());
      continue; // Public Copy already rejects the opaque compressed encoding.
    }
    ASSERT_TRUE(rejected_builder.publish(rejected_builder.result(*copied).value()).ok());
    auto rejected_graph = std::move(rejected_builder).finish().value();
    const auto rejected = gv::compile(context, rejected_graph);
    ASSERT_FALSE(rejected);
    EXPECT_TRUE(rejected.error().IsUnimplemented());
  }
  // Only the selected expert needs a float oracle. Never expand the full
  // 3 GiB compressed bank into a 12 GiB float host or device representation.
  o::Weight selected(1, int(columns), int(depth), false);
  std::vector<std::byte> bank(bank_bytes);
  std::memcpy(bank.data() + std::size_t(highest) * expert_codes,
              selected.bytes.data(), expert_codes);
  std::memcpy(bank.data() + codes_bytes + std::size_t(highest) * expert_scales,
              selected.bytes.data() + expert_codes, expert_scales);
  auto uploaded = context.upload(description, bank);
  ASSERT_TRUE(uploaded) << uploaded.error().message();
  auto resident = std::move(*uploaded);
  std::vector<std::byte>().swap(bank);
  std::weak_ptr<const void> lifetime = resident.native_owner();
  const auto handle = resident.native_buffer();
  const auto ad = t::dense(c::DataType::kBFloat16, {1, 1, depth});
  const auto id = t::dense(c::DataType::kInt32, {1, 1});
  auto builder = g::GraphBuilder::create().value();
  const auto w = builder.constant(description).value();
  const auto a = builder.input(ad).value();
  const auto ids = builder.input(id).value();
  const auto node = builder.add_node(c::model_ops::grouped_float8_linear(),
                                    std::array{w, a, ids});
  ASSERT_TRUE(node) << node.error().message();
  const auto result = builder.result(*node).value();
  ASSERT_TRUE(builder.publish(result).ok());
  auto graph = std::move(builder).finish().value();
  auto program = gv::compile(context, graph, std::array{resident});
  ASSERT_TRUE(program) << program.error().message();
  auto first = program->create_invocation().value();
  auto second = program->create_invocation().value();
  for (const auto *slot : {&first, &second}) {
    std::size_t found = 0;
    for (const auto &stage : Access::invocation(*slot)->run->stages) {
      if (!stage.iree || stage.iree->program->input_count() != 4)
        continue;
      ++found;
      const auto codes = stage.iree->program->input(1).value();
      const auto scales = stage.iree->program->input(2).value();
      ASSERT_TRUE(stage.iree->inputs[1].borrowed);
      ASSERT_TRUE(stage.iree->inputs[2].borrowed);
      EXPECT_EQ(codes.buffer, handle);
      EXPECT_EQ(codes.offset, 0U);
      EXPECT_EQ(codes.bytes, codes_bytes);
      EXPECT_EQ(scales.buffer, handle);
      EXPECT_EQ(scales.offset, codes_bytes);
      EXPECT_EQ(scales.bytes, experts * expert_scales);
      EXPECT_GT(scales.offset, std::size_t{INT32_MAX});
    }
    ASSERT_EQ(found, 1U);
  }
  graph = {};
  *program = {};
  resident = {};
  ASSERT_FALSE(lifetime.expired());
  v::Tensor retained;
  std::vector<double> retained_reference;
  for (int pass = 0; pass < 2; ++pass) {
    auto &slot = pass == 0 ? first : second;
    const auto x = o::activation(1, int(depth), c::DataType::kBFloat16, pass);
    const std::vector<double> values(x.begin(), x.end());
    const std::array selected_ids{highest};
    const std::array inputs{t::make(context, ad, values),
        context.upload(id, std::as_bytes(std::span(selected_ids))).value()};
    ASSERT_TRUE(slot.bind(inputs).ok());
    auto work = slot.submit().value();
    auto output = work.wait();
    ASSERT_TRUE(output) << output.error().message();
    const std::array<std::int32_t, 1> local_ids{0};
    const auto expected = o::reference(selected,
        o::quantize(x, int(depth), 1.e-10F, false),
        1, 1, 1, local_ids, c::DataType::kBFloat16);
    t::expect(context, output->get()->tensors()[0], expected);
    if (pass == 0) {
      retained = output->get()->tensors()[0];
      retained_reference = expected;
    }
  }
  const auto x = o::activation(1, int(depth), c::DataType::kBFloat16, 0);
  const std::vector<double> values(x.begin(), x.end());
  const std::array<std::int32_t, 1> invalid_ids{experts};
  ASSERT_TRUE(second.bind(std::array{t::make(context, ad, values),
      context.upload(id, std::as_bytes(std::span(invalid_ids))).value()}).ok());
  auto rejected = second.submit().value().wait();
  ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().IsOutOfRange());
  t::expect(context, retained, retained_reference);
  std::printf("GLM_FP8_FULLBANK bytes=%zu codeWords=%zu expert255Byte=%zu "
              "scaleByte=%zu slots=2 highest255+bad256+retained-output PASS\n",
              bank_bytes, codes_bytes / 4, std::size_t(highest) * expert_codes,
              codes_bytes);
  first = {};
  second = {};
  retained = {};
  EXPECT_TRUE(lifetime.expired());
  ASSERT_TRUE(context.close().ok());
}

TEST(VulkanGraph, OriginalOfficialQatFiniteBfloatExhaustiveAndFailure) {
  vulkan_official_test::roundtrips(compile_artifact);
}
TEST(VulkanGraph, OriginalOfficialIndependentLeaseFiniteCancelCloseAndOom) {
  vulkan_official_test::lifecycle(compile_artifact);
}
TEST(VulkanGraph, OriginalOfficialEncodedBlock128TailAndReplay) {
  auto context = vulkan_test::context().value();
  for (int n : {5, 129}) {
    Float8Inputs a(3, 256, n, c::Float8ScaleMode::kFloat32Block128), b(3, 256, n, c::Float8ScaleMode::kFloat32Block128, 1);
    const std::array descriptions{a.activation, a.weight};
    auto program = compile_artifact(context, c::Operation(c::MatMul{.accumulation_type=c::DataType::kFloat32}), descriptions);
    ASSERT_TRUE(program) << program.error().message();auto slot = program->create_invocation().value();
    for (const auto *data : {&a, &b}) {
      const std::array inputs{context.upload(data->activation, data->a).value(), context.upload(data->weight, data->w).value()};
      ASSERT_TRUE(slot.bind(inputs).ok());auto work = slot.submit().value();auto result = work.wait();
      ASSERT_TRUE(result) << result.error().message();t::expect(context, result->get()->tensors()[0], data->multiply(), true);
    }
  }
}

TEST(VulkanGraph, MlxGroupedSharesOriginalActivationAcrossSelectedExperts) {
  vulkan_mlx_test::grouped_shared_activation(compile_artifact);
}

TEST(VulkanGraph, OriginalMlxBatchUsesOrdinaryTypedNodesAndCheckedSlots) {
  vulkan_mlx_test::batch(compile_artifact);
}

TEST(VulkanGraph, OriginalMlxAffineWidthsUseContinuousBitstreamsAndCheckedRebinding) {
  vulkan_mlx_test::affine_widths(compile_artifact);
}

TEST(VulkanGraph, OriginalMlxConstantsKeepBanksAcrossRebindingAndSlots) {
  vulkan_mlx_test::constants([](const v::Context &context, c::Operation operation,
                               std::span<const v::Tensor> resident) {
    auto builder = g::GraphBuilder::create().value();std::vector<g::ValueId> inputs;
    std::vector<v::Tensor> constants;
    for (std::size_t i = 0; i < resident.size(); ++i)
      if (i == 0 || i + 1 == resident.size()) inputs.push_back(builder.input(resident[i].description()).value());
      else { inputs.push_back(builder.constant(resident[i].description()).value());constants.push_back(resident[i]); }
    const auto result = builder.result(builder.add_node(operation, inputs).value()).value();
    EXPECT_TRUE(builder.publish(result).ok());
    return gv::compile(context, std::move(builder).finish().value(), constants);
  });
}

TEST(VulkanGraph, OriginalArtifactBatchUsesOrdinaryTypedEntry) {
  vulkan_artifact_test::batch(compile_artifact);
}

TEST(VulkanGraph, OriginalArtifactColdResultBindsMatureConstantProjection) {
  vulkan_artifact_test::cold_projection(compile_artifact,
      [](const v::Context &context, const c::TensorDesc &activation, const v::Tensor &weight) {
    auto builder = g::GraphBuilder::create().value();
    const auto a = builder.input(activation).value();
    const auto w = builder.constant(weight.description()).value();
    const auto result = builder.add(
        c::MatMul{.accumulation_type = c::DataType::kFloat32}, a, w).value();
    EXPECT_TRUE(builder.publish(result).ok());
    return gv::compile(context, std::move(builder).finish().value(), std::array{weight});
  });
}

TEST(VulkanGraph, OriginalFloat8RegionContinuesIntoGgmlAndNativeCopy) {
  auto context = vulkan_test::context().value();
  for (auto mode : {c::Float8ScaleMode::kFloat32Block128,
                    c::Float8ScaleMode::kUnsignedExponent8Group32}) {
    Float8Inputs data(3, 128, 128, mode, 1);
    auto builder = g::GraphBuilder::create().value();
    const auto input = builder.input(data.activation).value();
    const auto weight = builder.constant(data.weight).value();
    const auto middle = builder.add(c::MatMul{}, input, weight).value();
    const auto absolute =
        builder.add(c::Unary{c::UnaryKind::kAbs}, middle).value();
    const auto output = builder.add(c::Copy{}, absolute).value();
    const auto low =
        builder.add(c::Cast{c::DataType::kBFloat16}, middle).value();
    ASSERT_TRUE(builder.publish(output).ok());
    ASSERT_TRUE(builder.publish(middle).ok());
    ASSERT_TRUE(builder.publish(low).ok());
    auto graph = std::move(builder).finish().value();
    gv::CompileReport report;
    const auto resident = context.upload(data.weight, data.w).value();
    auto program =
        gv::compile(context, graph, std::array{resident}, {.report = &report});
    ASSERT_TRUE(program) << program.error().message();
    EXPECT_GE(report.managed_region_count, 2U);
    EXPECT_NE(report.numerical_policy.find("FP8"), std::string::npos);
    auto slot = program->create_invocation().value();
    ASSERT_TRUE(
        slot.bind(context.upload(data.activation, data.a).value()).ok());
    const auto expected = data.multiply();
    auto positive = expected;
    for (auto &value : positive)
      value = std::abs(value);
    graph = {};
    *program = {};
    for (int repeat = 0; repeat < 3; ++repeat) {
      auto work = slot.submit().value();
      auto result = work.wait();
      ASSERT_TRUE(result) << result.error().message();
      t::expect(context, result->get()->tensors()[0], positive, true);
      t::expect(context, result->get()->tensors()[1], expected, true);
      t::expect(context, result->get()->tensors()[2], expected, true);
    }
    EXPECT_EQ(t::bytes(context, resident), data.w);
  }
}

TEST(VulkanGraph, CheckedScanChainsTwoOriginalProgramsWithIndependentResults) {
  auto context = vulkan_test::context().value();
  const auto desc = t::dense(c::DataType::kInt32, {2, 257});
  auto builder = g::GraphBuilder::create().value();
  const auto input = builder.input(desc).value();
  const auto middle = builder.add(c::Scan{}, input).value();
  const auto output = builder.add(c::Scan{.exclusive = true}, middle).value();
  ASSERT_TRUE(builder.publish(output).ok());
  ASSERT_TRUE(builder.publish(middle).ok());
  const auto graph = std::move(builder).finish().value();
  gv::CompileReport report;
  auto program = gv::compile(context, graph, {}, {.report = &report}).value();
  EXPECT_EQ(report.managed_region_count, 2U);
  auto first = program.create_invocation().value();
  auto second = program.create_invocation().value();
  v::Tensor retained;
  std::vector<std::int32_t> saved;
  for (int phase = 0; phase < 3; ++phase) {
    const auto data = s::integer_pattern(514, phase);
    auto prefix = data;
    for (std::size_t row = 0; row < 2; ++row) {
      std::int64_t sum = 0;
      for (std::size_t col = 0; col < 257; ++col) {
        sum += data[row * 257 + col];
        prefix[row * 257 + col] = static_cast<std::int32_t>(sum);
      }
    }
    auto &slot = phase == 0 ? first : second;
    ASSERT_TRUE(slot.bind(s::make_integer(context, desc, data)).ok());
    auto work = slot.submit().value();
    auto result = work.wait();
    ASSERT_TRUE(result) << result.error().message();
    s::expect_integer_prefix(context, result->get()->tensors()[0], desc, prefix,
                             c::Scan{.exclusive = true});
    EXPECT_EQ(s::integer_values(context, result->get()->tensors()[1]), prefix);
    if (phase == 0) {
      retained = result->get()->tensors()[0];
      saved = s::integer_values(context, retained);
    } else {
      EXPECT_EQ(s::integer_values(context, retained), saved);
    }
  }
  EXPECT_TRUE(context.close().ok());
  EXPECT_EQ(s::integer_values(context, retained), saved);
}
} // namespace
