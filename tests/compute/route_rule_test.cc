#include "hyle/compute/route_rule.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>

namespace {
using namespace hyle::compute;

inline constexpr ModelDescription kModel{"qwen3_5", "qwen3.5-2b", "snapshot-a",
                                         "config-a"};
inline constexpr ProviderIdentity kProvider{
    {0x524f555445544553ULL, 1}, "route-test", "7.2"};
inline constexpr StorageFormatId kWeightFormat{0x524f555445544553ULL, 2};
inline constexpr StorageFormatId kScaleFormat{0x524f555445544553ULL, 3};

struct QualifiedRequest final {
  std::array<RouteAttribute, 7> facts{
      RouteAttribute::number(route_attribute::kDeviceVendorId, 0x10de),
      RouteAttribute::number(route_attribute::kDeviceId, 0x2786),
      RouteAttribute::string(route_attribute::kArchitecture, "ada"),
      RouteAttribute::number(route_attribute::kArchitectureMajor, 8),
      RouteAttribute::number(route_attribute::kArchitectureMinor, 9),
      RouteAttribute::number(route_attribute::kDriverVersion, 13020),
      RouteAttribute::number(route_attribute::kToolkitVersion, 13030)};
  std::array<RouteAttributePredicate, 7> predicates{
      RouteAttributePredicate::number(route_attribute::kDeviceVendorId, 0x10de),
      RouteAttributePredicate::number(route_attribute::kDeviceId, 0x2786),
      RouteAttributePredicate::string(route_attribute::kArchitecture, "ada"),
      RouteAttributePredicate::number(route_attribute::kArchitectureMajor, 8),
      RouteAttributePredicate::number(route_attribute::kArchitectureMinor, 9),
      RouteAttributePredicate::range(route_attribute::kDriverVersion, 13000,
                                     13999),
      RouteAttributePredicate::number(route_attribute::kToolkitVersion, 13030)};
  std::array<TensorDesc, 3> inputs;
  std::array<TensorDesc, 1> outputs;
  WorkloadContext workload;

  QualifiedRequest()
      : inputs{TensorDesc::dense(DataType::kFloat16,
                                 std::array<std::int64_t, 2>{1, 128})
                   .value(),
               TensorDesc::opaque(
                   DataType::kFloat16, std::array<std::int64_t, 2>{64, 128},
                   StorageFormat::opaque(kWeightFormat, 16).value(), 4096)
                   .value(),
               TensorDesc::opaque(
                   DataType::kFloat16, std::array<std::int64_t, 2>{64, 1},
                   StorageFormat::opaque(kScaleFormat, 16).value(), 128)
                   .value()},
        outputs{TensorDesc::dense(DataType::kFloat16,
                                  std::array<std::int64_t, 2>{1, 64})
                    .value()},
        workload{ModelIdentity::create(kModel).value(),
                 WorkloadPhase::kDecode,
                 {7, 8, 2}} {}

  RouteRule rule() const noexcept {
    return {.platform = RoutePlatform::kWindows,
            .route = NativeRoute::kCuda,
            .provider = kProvider.id,
            .provider_version = kProvider.version,
            .attributes = predicates,
            .model = kModel,
            .phase = WorkloadPhase::kDecode,
            .region = {7, 8, 2},
            .operation = OperationKind::kQuantizedMatMul,
            .input0_type = DataType::kFloat16,
            .input1_type = DataType::kFloat16,
            .output0_type = DataType::kFloat16,
            .input1_format = kWeightFormat,
            .quantization = {.presence =
                                 RouteQuantizationPresence::kQuantizedMatMul,
                             .mode = QuantizedWeightMode::kSignedSymmetric4,
                             .group_size = 128,
                             .weight_format = kWeightFormat,
                             .scale_format = kScaleFormat}};
  }

  RouteMatchInput input() const noexcept {
    return {.platform = RoutePlatform::kWindows,
            .route = NativeRoute::kCuda,
            .provider = kProvider,
            .attributes = facts,
            .model = workload.model.description(),
            .phase = workload.phase,
            .region = workload.region,
            .operation = Operation(
                QuantizedMatMul{.mode = QuantizedWeightMode::kSignedSymmetric4,
                                .group_size = 128,
                                .accumulation_type = DataType::kFloat32}),
            .inputs = inputs,
            .outputs = outputs};
  }
};

TEST(RouteRule, FullSparseConjunctionMatchesRealRequestFacts) {
  const QualifiedRequest request;
  const auto result = match_route_rule(request.rule(), request.input());
  EXPECT_TRUE(result.matched());
  EXPECT_EQ(result.mismatch, RouteMismatch::kNone);
  EXPECT_FALSE(result.attribute.valid());
}

TEST(RouteRule, EachNamedAxisRejectsUnknownOrDifferentFacts) {
  const QualifiedRequest request;
  const auto rule = request.rule();
  auto input = request.input();

  input.platform = RoutePlatform::kLinux;
  EXPECT_EQ(match_route_rule(rule, input).mismatch, RouteMismatch::kPlatform);
  input = request.input();
  input.route = NativeRoute::kVulkan;
  EXPECT_EQ(match_route_rule(rule, input).mismatch, RouteMismatch::kRoute);
  input = request.input();
  input.provider = {{0x524f555445544553ULL, 99}, "other", "7.2"};
  EXPECT_EQ(match_route_rule(rule, input).mismatch, RouteMismatch::kProvider);
  input = request.input();
  input.provider = {kProvider.id, kProvider.name, "7.3"};
  EXPECT_EQ(match_route_rule(rule, input).mismatch,
            RouteMismatch::kProviderVersion);

  input = request.input();
  input.attributes = std::span(request.facts).first(request.facts.size() - 1);
  auto result = match_route_rule(rule, input);
  EXPECT_EQ(result.mismatch, RouteMismatch::kAttributeMissing);
  EXPECT_EQ(result.attribute, route_attribute::kToolkitVersion);
  auto changed_facts = request.facts;
  changed_facts[2] = RouteAttribute::number(route_attribute::kArchitecture, 89);
  input = request.input();
  input.attributes = changed_facts;
  result = match_route_rule(rule, input);
  EXPECT_EQ(result.mismatch, RouteMismatch::kAttributeType);
  EXPECT_EQ(result.attribute, route_attribute::kArchitecture);
  changed_facts = request.facts;
  changed_facts[5] =
      RouteAttribute::number(route_attribute::kDriverVersion, 14000);
  input.attributes = changed_facts;
  result = match_route_rule(rule, input);
  EXPECT_EQ(result.mismatch, RouteMismatch::kAttributeValue);
  EXPECT_EQ(result.attribute, route_attribute::kDriverVersion);

  input = request.input();
  input.model = {kModel.family, "other", kModel.revision, kModel.configuration};
  EXPECT_EQ(match_route_rule(rule, input).mismatch, RouteMismatch::kModel);
  input = request.input();
  input.phase = WorkloadPhase::kPrefill;
  EXPECT_EQ(match_route_rule(rule, input).mismatch, RouteMismatch::kPhase);
  input = request.input();
  ++input.region.version;
  EXPECT_EQ(match_route_rule(rule, input).mismatch, RouteMismatch::kRegion);

  input = request.input();
  input.operation = Operation(MatMul{});
  EXPECT_EQ(match_route_rule(rule, input).mismatch, RouteMismatch::kOperation);
  input = request.input();
  auto inputs = request.inputs;
  inputs[0] = TensorDesc::dense(DataType::kBFloat16,
                                std::array<std::int64_t, 2>{1, 128})
                  .value();
  input.inputs = inputs;
  EXPECT_EQ(match_route_rule(rule, input).mismatch, RouteMismatch::kDataType);

  input = request.input();
  input.operation =
      Operation(QuantizedMatMul{.mode = QuantizedWeightMode::kSignedSymmetric8,
                                .group_size = 128,
                                .accumulation_type = DataType::kFloat32});
  EXPECT_EQ(match_route_rule(rule, input).mismatch,
            RouteMismatch::kQuantizationMode);
  input = request.input();
  input.operation =
      Operation(QuantizedMatMul{.mode = QuantizedWeightMode::kSignedSymmetric4,
                                .group_size = 64,
                                .accumulation_type = DataType::kFloat32});
  EXPECT_EQ(match_route_rule(rule, input).mismatch,
            RouteMismatch::kQuantizationGroup);
}

TEST(RouteRule, WildcardsAreCheapButMalformedOrDuplicateInputsAreRejected) {
  EXPECT_TRUE(match_route_rule({}, {}).matched());

  RouteRule named{.platform = RoutePlatform::kWindows,
                  .route = NativeRoute::kCpu};
  EXPECT_EQ(match_route_rule(named, {}).mismatch, RouteMismatch::kPlatform);

  std::array duplicate_facts{
      RouteAttribute::number(route_attribute::kDeviceId, 1),
      RouteAttribute::number(route_attribute::kDeviceId, 1)};
  RouteMatchInput facts{.attributes = duplicate_facts};
  EXPECT_EQ(match_route_rule({}, facts).mismatch, RouteMismatch::kInvalidFacts);

  std::array duplicate_predicates{
      RouteAttributePredicate::number(route_attribute::kDeviceId, 1),
      RouteAttributePredicate::range(route_attribute::kDeviceId, 1, 2)};
  RouteRule duplicates{.attributes = duplicate_predicates};
  EXPECT_EQ(match_route_rule(duplicates, {}).mismatch,
            RouteMismatch::kInvalidRule);

  std::array invalid_range{
      RouteAttributePredicate::range(route_attribute::kDriverVersion, 9, 3)};
  RouteRule invalid{.attributes = invalid_range};
  EXPECT_EQ(match_route_rule(invalid, {}).mismatch,
            RouteMismatch::kInvalidRule);

  RouteRule orphan_model{.model = {{}, "orphan", {}, {}}};
  EXPECT_EQ(match_route_rule(orphan_model, {}).mismatch,
            RouteMismatch::kInvalidRule);

  RouteRule ambiguous_quant{.quantization = {.group_size = 128}};
  EXPECT_EQ(match_route_rule(ambiguous_quant, {}).mismatch,
            RouteMismatch::kInvalidRule);
}

TEST(RouteRule, FourNativeRoutesAndBuildPlatformRemainIndependentFacts) {
  for (const auto route : {NativeRoute::kCpu, NativeRoute::kCuda,
                           NativeRoute::kApple, NativeRoute::kVulkan}) {
    RouteRule rule{.route = route};
    RouteMatchInput input{.route = route};
    EXPECT_TRUE(match_route_rule(rule, input).matched());
    input.route = NativeRoute::kUnknown;
    EXPECT_EQ(match_route_rule(rule, input).mismatch, RouteMismatch::kRoute);
  }
#if defined(_WIN32)
  EXPECT_EQ(kBuildRoutePlatform, RoutePlatform::kWindows);
#elif defined(__APPLE__)
  EXPECT_EQ(kBuildRoutePlatform, RoutePlatform::kMacOS);
#elif defined(__linux__)
  EXPECT_EQ(kBuildRoutePlatform, RoutePlatform::kLinux);
#endif
}

TEST(RouteRule, NonQuantizedAndPhysicalFp8RulesStayDistinctFromGroupedQ4Q8) {
  constexpr StorageFormatId fp8{0x524f555445544553ULL, 11};
  const std::array inputs{
      TensorDesc::opaque(DataType::kFloat8E4M3FN,
                         std::array<std::int64_t, 2>{1, 128},
                         StorageFormat::opaque(fp8).value(), 128)
          .value(),
      TensorDesc::opaque(DataType::kFloat8E4M3FN,
                         std::array<std::int64_t, 2>{64, 128},
                         StorageFormat::opaque(fp8).value(), 8192)
          .value()};
  const std::array outputs{
      TensorDesc::dense(DataType::kFloat16, std::array<std::int64_t, 2>{1, 64})
          .value()};
  const RouteRule rule{
      .operation = OperationKind::kMatMul,
      .input0_type = DataType::kFloat8E4M3FN,
      .input1_type = DataType::kFloat8E4M3FN,
      .output0_type = DataType::kFloat16,
      .input0_format = fp8,
      .input1_format = fp8,
      .quantization = {.presence = RouteQuantizationPresence::kNone}};
  const RouteMatchInput input{
      .operation = Operation(MatMul{}), .inputs = inputs, .outputs = outputs};
  EXPECT_TRUE(match_route_rule(rule, input).matched());
}

} // namespace
