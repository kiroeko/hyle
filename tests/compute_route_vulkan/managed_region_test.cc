#include <gtest/gtest.h>

#include <algorithm>

#include "../compute_models/d5_qwen35_fixture.h"
#include "../compute/workload_fixture.h"
#include "gate_norm_test_support.h"
#include "vulkan_program_internal.h"
#include "vulkan_policy.h"

namespace {
using namespace vulkan_gate_norm_test;
namespace d = v::detail;

TEST(VulkanManaged, ColdRulesMatchFullConditionAxesAndUnknownDefault) {
  auto context=vulkan_test::context().value();
  auto &environment=d::Access::runtime(context)->device->environment;
  EXPECT_EQ(environment.ggml_revision,"456172ec733a135778adcd32d00e576a58232e45");
  EXPECT_EQ(environment.iree_revision,"6ef606b6fa3e12bd2c12d36a07463069eb71185f");
  EXPECT_FALSE(environment.sdk_version.empty());
  d::ProgramState program;program.runtime=d::Access::runtime(context);
  program.workload=workload_test::known(c::WorkloadPhase::kDecode);
  const auto description=dense(c::DataType::kFloat32,{2,8});
  program.values={{description,{}},{description,{}}};
  const d::PlanStep step{.operation=c::Operation(c::Unary{c::UnaryKind::kSilu}),.inputs={0},.outputs={1}};
  const std::array facts{
      c::RouteFactPredicate::number(c::route_fact::kDeviceVendorId,environment.vendor_id),
      c::RouteFactPredicate::number(c::route_fact::kDeviceId,environment.device_id),
      c::RouteFactPredicate::number(c::route_fact::kDriverVersion,environment.driver_version),
      c::RouteFactPredicate::string(c::route_fact::kSdkVersion,environment.sdk_version),
      c::RouteFactPredicate::string(d::kShaderCompilerAttribute,environment.shader_compiler)};
  const std::array dimensions{c::RouteDimensionPredicate::exact(0,2),c::RouteDimensionPredicate::exact(1,8)};
  std::array ports{c::RouteTensorPredicate{.direction=c::RoutePortDirection::kInput,
      .index=0,.data_type=c::DataType::kFloat32,.storage_kind=c::RouteStorageKind::kPlain,
      .minimum_rank=2,.maximum_rank=2,.dimensions=dimensions}};
  auto rule=d::kGgmlDefaultSelector;rule.platform=c::kBuildRoutePlatform;
  rule.provider_version=environment.ggml_revision;rule.environment=facts;
  rule.profile=c::exact_profile_selector(program.workload.profile.description());
  rule.phase=program.workload.phase;rule.region=program.workload.region;
  rule.operation=c::OperationKind::kUnary;rule.ports=ports;
  EXPECT_TRUE(d::matches_cold_rule(program,step,d::DefaultCarrier::kGgml,rule));
  auto mismatch=rule;mismatch.profile.model_name="other-model";
  EXPECT_FALSE(d::matches_cold_rule(program,step,d::DefaultCarrier::kGgml,mismatch));
  mismatch=rule;mismatch.phase=c::WorkloadPhase::kPrefill;
  EXPECT_FALSE(d::matches_cold_rule(program,step,d::DefaultCarrier::kGgml,mismatch));
  mismatch=rule;mismatch.provider_version="unqualified-revision";
  EXPECT_FALSE(d::matches_cold_rule(program,step,d::DefaultCarrier::kGgml,mismatch));
  ports[0].storage_kind=c::RouteStorageKind::kBlocked;
  EXPECT_FALSE(d::matches_cold_rule(program,step,d::DefaultCarrier::kGgml,rule));
  ports[0].storage_kind=c::RouteStorageKind::kPlain;
  const auto saved=environment;
  environment.device_name.clear();environment.driver_name.clear();environment.sdk_version.clear();
  const auto unknown=d::route_facts(environment);
  EXPECT_TRUE(std::ranges::all_of(unknown.values(),[](const auto &fact){return fact.valid();}));
  EXPECT_FALSE(d::matches_cold_rule(program,step,d::DefaultCarrier::kGgml,rule));
  EXPECT_TRUE(d::matches_mature_default(program,step,d::DefaultCarrier::kGgml));
  EXPECT_TRUE(d::matches_mature_default(program,step,d::DefaultCarrier::kIree));
  auto prepared=v::prepare(context,program.workload,c::Unary{c::UnaryKind::kSilu},description);
  environment=saved;
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->workload(),program.workload);
  EXPECT_TRUE(d::Access::program(*prepared)->dispatches.back().region);
  auto invocation=prepared->create_invocation().value();
  for (double changed : {1.0,-1.0}) {
    const std::vector<double> input(16,changed),expected(16,unary(c::UnaryKind::kSilu,changed));
    ASSERT_TRUE(invocation.bind(make(context,description,input)).ok());
    auto result=invocation.submit().value().wait();ASSERT_TRUE(result);
    expect(context,result->get()->tensors()[0],expected);
  }
}

TEST(VulkanManaged, D5RealQwenProjectionUsesExactProfileAndReplays) {
  const auto fixture_root = d5_qwen35_fixture::root();
  if (!fixture_root)
    GTEST_SKIP() << "set HYLE_D5_QWEN35_FIXTURE to the prepared local fixture";
  d5_qwen35_fixture::Projection fixture;
  ASSERT_NO_THROW(fixture = d5_qwen35_fixture::load_projection(*fixture_root));

  auto context = vulkan_test::context().value();
  const auto activation_desc =
      dense(c::DataType::kBFloat16, {1, d5_qwen35_fixture::kHidden});
  const auto stored_weight_desc =
      dense(c::DataType::kBFloat16,
            {d5_qwen35_fixture::kHidden, d5_qwen35_fixture::kHidden});
  const auto weight_desc = stored_weight_desc.transposed().value();
  auto weight = context.upload(weight_desc, fixture.weight).value();
  const auto workload =
      d5_qwen35_fixture::workload(c::WorkloadPhase::kDecode);
  auto program =
      v::prepare(context, workload, c::MatMul{}, activation_desc, weight);
  ASSERT_TRUE(program) << program.error().message();
  EXPECT_EQ(program->workload(), workload);
  EXPECT_EQ(std::ranges::count_if(
                d::Access::program(*program)->dispatches,
                [](const auto &dispatch) { return !!dispatch.region; }),
            1);

  auto invocation = program->create_invocation().value();
  for (int run = 0; run < 2; ++run) {
    auto payload = fixture.activation;
    if (run != 0) {
      for (std::size_t index = 0; index < d5_qwen35_fixture::kHidden; ++index)
        payload[index * 2 + 1] ^= std::byte{0x80};
    }
    auto activation = context.upload(activation_desc, payload).value();
    ASSERT_TRUE(invocation.bind(activation).ok());
    auto work = invocation.submit();
    ASSERT_TRUE(work) << work.error().message();
    auto result = work->wait();
    ASSERT_TRUE(result) << result.error().message();
    const auto actual = values(context, result->get()->tensors()[0]);
    ASSERT_EQ(actual.size(), fixture.expected.size());
    for (std::size_t index = 0; index < actual.size(); ++index)
      half_test::expect_near(
          half_test::encode<B16>(actual[index]),
          run == 0 ? fixture.expected[index] : -fixture.expected[index]);
  }
}

TEST(VulkanManaged, BfloatRank2SmallMatvecWidensActivationBoundary) {
  auto context = vulkan_test::context().value();
  constexpr std::int64_t depth = 64;
  constexpr std::int64_t columns = 8;
  const auto stored_weight_desc =
      dense(c::DataType::kBFloat16, {columns, depth});
  const auto weight_desc = stored_weight_desc.transposed().value();
  std::vector<double> weight_values(depth * columns);
  for (std::size_t i = 0; i < weight_values.size(); ++i)
    weight_values[i] = (static_cast<int>(i % 31) - 15) / 32.0;
  auto weight = make(context, weight_desc, weight_values);

  for (std::int64_t rows = 1; rows <= 4; ++rows) {
    SCOPED_TRACE(rows);
    const auto activation_desc =
        dense(c::DataType::kBFloat16, {rows, depth});
    auto program = v::prepare(context, c::MatMul{}, activation_desc, weight);
    ASSERT_TRUE(program) << program.error().message();
    auto invocation = program->create_invocation().value();
    std::vector<double> activation_values(rows * depth);
    for (std::size_t i = 0; i < activation_values.size(); ++i)
      activation_values[i] = (static_cast<int>(i % 23) - 11) / 16.0;
    auto activation = make(context, activation_desc, activation_values);
    ASSERT_TRUE(invocation.bind(activation).ok());
    auto result = invocation.submit().value().wait();
    ASSERT_TRUE(result) << result.error().message();
    std::vector<double> expected(rows * columns, 0.0);
    for (std::int64_t row = 0; row < rows; ++row)
      for (std::int64_t column = 0; column < columns; ++column)
        for (std::int64_t inner = 0; inner < depth; ++inner)
          expected[row * columns + column] +=
              rounded(activation_values[row * depth + inner],
                      c::DataType::kBFloat16) *
              rounded(weight_values[inner * columns + column],
                      c::DataType::kBFloat16);
    expect(context, result->get()->tensors()[0], expected);
  }
}

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
    auto workload = workload_test::known(
        rows == 1 ? c::WorkloadPhase::kDecode : c::WorkloadPhase::kPrefill);
    auto prepared = v::prepare(context, workload, c::RmsNorm{}, desc, weight);
    ASSERT_TRUE(prepared) << prepared.error().message();
    EXPECT_EQ(prepared->workload(), workload);
    workload.region.version = 0;
    EXPECT_FALSE(v::prepare(context, workload, c::RmsNorm{}, desc, weight));
    workload = {};
    EXPECT_TRUE(prepared->workload().profile.matches(workload_test::kModel));
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
  const std::array descriptions{dense(c::DataType::kFloat32, {}),
                                dense(c::DataType::kFloat32, {129}),
                                dense(c::DataType::kFloat32, {17, 257}),
                                dense(c::DataType::kFloat32, {2, 1, 3, 43})};
  const std::array<double, 12> edges{-0.0, NAN,   -INFINITY, INFINITY, 0.0, -90,
                                     90,   -1e-7, 1e-7,      -1,       1,   3};
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
        const bool exact =
            kind == c::UnaryKind::kAbs || kind == c::UnaryKind::kNeg;
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
        std::vector<double> data(rows * width),
            small(*parameter.element_count());
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
        auto program =
            v::prepare(context, c::Binary{kind}, arguments[0].description(),
                       arguments[1].description())
                .value();
        ASSERT_TRUE(d::Access::program(program)->dispatches.back().region);
        auto slot = program.create_invocation().value();
        ASSERT_TRUE(slot.bind(arguments).ok());
        std::vector<double> expected(data.size());
        for (std::size_t i = 0; i < data.size(); ++i) {
          const auto index = parameter.rank() == 0   ? 0
                             : parameter.rank() == 1 ? i % width
                                                     : i / width;
          const double left = small_first ? small[index] : data[i];
          const double right = small_first ? data[i] : small[index];
          expected[i] =
              kind == c::BinaryKind::kSubtract ? left - right : left / right;
        }
        {
          auto work = slot.submit().value();
          auto result = work.wait().value();
          expect(context, result.get()->tensors()[0], expected);
        }
        const auto output = context.allocate(full).value();
        auto recorder = v::Recorder::create(context).value();
        ASSERT_TRUE(
            recorder.append(program, arguments, std::array{output}).ok());
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
  const std::array<double, 33> edges{
      -80,     -40,       -20,      -15,   -1, -1e-4, -1e-8, -0.0, 0.0,
      1e-8,    1e-4,      0.5,      1,     2,  15,    20,    40,   80,
      NAN,     -INFINITY, INFINITY, -2,    3,  0.25,  -0.25, 8,    16,
      -4.1588, -4.159,    4.1588,   4.159, -5, 5};
  for (auto type :
       {c::DataType::kFloat32, c::DataType::kFloat16, c::DataType::kBFloat16}) {
    const auto size = static_cast<std::int64_t>(c::data_type_size(type));
    const auto desc =
        c::TensorDesc::strided_bytes(
            type, std::array<std::int64_t, 8>{1, 1, 1, 1, 1, 1, 3, 11},
            std::array<std::int64_t, 8>{0, 0, 0, 0, 0, 0, 11 * (size + 1) + 3,
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
      for (bool mature : {true, false}) {
        if (!mature && kind != c::UnaryKind::kSoftplus)
          continue;
        SCOPED_TRACE(mature);
        auto &limits = d::Access::runtime(context)->device->properties.limits;
        const auto original_limits = limits;
        if (!mature)
          limits.maxComputeWorkGroupInvocations = 256;
        auto prepared = v::prepare(context, c::Unary{kind}, desc);
        limits = original_limits;
        ASSERT_TRUE(prepared) << prepared.error().message();
        auto program = std::move(*prepared);
        const auto &dispatches = d::Access::program(program)->dispatches;
        ASSERT_EQ(std::ranges::count_if(dispatches,
                                        [](const auto &dispatch) {
                                          return dispatch.region != nullptr;
                                        }),
                  mature ? 1 : 0);
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

TEST(VulkanManaged, B2MatureLinearPositionAndTransformReplay) {
  auto context = vulkan_test::context().value();
  for (auto type : {c::DataType::kFloat16, c::DataType::kBFloat16}) {
    const auto left_desc = dense(type, {2, 2, 3});
    const auto right_desc = dense(type, {1, 3, 4});
    const std::array<double, 12> right_values{1, 2, 3, 4,  5,  6,
                                              7, 8, 9, 10, 11, 12};
    const auto right = make(context, right_desc, right_values);
    auto program = v::prepare(context, c::MatMul{}, left_desc, right);
    ASSERT_TRUE(program) << program.error().message();
    EXPECT_EQ(std::ranges::count_if(
                  d::Access::program(*program)->dispatches,
                  [](const auto &dispatch) { return !!dispatch.region; }),
              1);
    auto slot = program->create_invocation().value();
    for (int replay = 0; replay < 2; ++replay) {
      std::array<double, 12> left_values{};
      for (std::size_t i = 0; i < left_values.size(); ++i)
        left_values[i] = static_cast<double>(i + 1 + replay);
      const auto left = make(context, left_desc, left_values);
      ASSERT_TRUE(slot.bind(left).ok());
      auto work = slot.submit().value();
      auto result = work.wait().value();
      std::array<double, 16> expected{};
      for (std::size_t batch = 0; batch < 2; ++batch)
        for (std::size_t row = 0; row < 2; ++row)
          for (std::size_t column = 0; column < 4; ++column)
            for (std::size_t inner = 0; inner < 3; ++inner)
              expected[(batch * 2 + row) * 4 + column] +=
                  rounded(left_values[(batch * 2 + row) * 3 + inner], type) *
                  rounded(right_values[inner * 4 + column], type);
      expect(context, result.get()->tensors()[0], expected);
    }
  }

  const auto rope_desc = dense(c::DataType::kFloat32, {1, 2, 1, 4});
  const auto position_desc = dense(c::DataType::kInt32, {2});
  const auto frequency_desc = dense(c::DataType::kFloat32, {2});
  const std::array<double, 8> rope_values{1, 2, 3, 4, 5, 6, 7, 8};
  const std::array<std::int32_t, 2> positions{0, 2};
  const auto rope_data = make(context, rope_desc, rope_values);
  const auto position =
      context.upload(position_desc, std::as_bytes(std::span(positions)))
          .value();
  const auto frequency = make(context, frequency_desc, std::array{0.5, 0.25});
  auto rope = v::prepare(context,
                         c::RoPE{.rotary_dimensions = 4,
                                 .sequence_axis = 1,
                                 .layout = c::RopeLayout::kInterleaved},
                         rope_desc, position_desc, frequency_desc);
  ASSERT_TRUE(rope) << rope.error().message();
  auto rope_slot = rope->create_invocation().value();
  ASSERT_TRUE(rope_slot.bind(rope_data, position, frequency).ok());
  auto rope_work = rope_slot.submit().value();
  auto rope_result = rope_work.wait().value();
  const std::array<double, 8> rotated{1,
                                      2,
                                      3,
                                      4,
                                      5 * std::cos(1.0) - 6 * std::sin(1.0),
                                      5 * std::sin(1.0) + 6 * std::cos(1.0),
                                      7 * std::cos(0.5) - 8 * std::sin(0.5),
                                      7 * std::sin(0.5) + 8 * std::cos(0.5)};
  expect(context, rope_result.get()->tensors()[0], rotated);

  const auto transform_desc = dense(c::DataType::kFloat32, {1, 64});
  std::array<double, 64> ones{};
  ones.fill(1.0);
  const auto transform_input = make(context, transform_desc, ones);
  auto transform = execute(context, c::Operation(c::Hadamard{}),
                           std::array{transform_input});
  ASSERT_TRUE(transform) << transform.error().message();
  std::array<double, 64> transformed{};
  transformed[0] = 8.0;
  expect(context, *transform, transformed);
}

TEST(VulkanManaged, B2MatureCausalGqaRetainsReplayResults) {
  auto context = vulkan_test::context().value();
  const auto query_desc = dense(c::DataType::kFloat32, {1, 1, 2, 1});
  const auto key_value_desc = dense(c::DataType::kFloat32, {1, 3, 1, 1});
  auto program =
      v::prepare(context,
                 c::ScaledDotProductAttention{
                     .layout = c::AttentionLayout::kBatchSequenceHeadDepth,
                     .scale = 1.0F,
                     .causal = true,
                     .accumulation_type = c::DataType::kFloat32},
                 query_desc, key_value_desc, key_value_desc);
  ASSERT_TRUE(program) << program.error().message();
  auto first = program->create_invocation();
  auto second = program->create_invocation();
  ASSERT_TRUE(first) << first.error().message();
  ASSERT_TRUE(second) << second.error().message();
  const auto query = make(context, query_desc, std::array{0.0, 0.0});
  const auto key = make(context, key_value_desc, std::array{0.0, 0.0, 0.0});
  v::Tensor retained;
  for (int replay = 0; replay < 2; ++replay) {
    const std::array<double, 3> values{1.0 + replay, 2.0 + replay,
                                       6.0 + replay};
    const auto value = make(context, key_value_desc, values);
    auto &slot = replay == 0 ? *first : *second;
    ASSERT_TRUE(slot.bind(query, key, value).ok());
    auto work = slot.submit();
    ASSERT_TRUE(work) << work.error().message();
    auto result = work->wait();
    ASSERT_TRUE(result) << result.error().message();
    const std::array expected{3.0 + replay, 3.0 + replay};
    expect(context, result->get()->tensors()[0], expected);
    if (replay == 0)
      retained = result->get()->tensors()[0];
    else
      expect(context, retained, std::array{3.0, 3.0});
  }
}
} // namespace
