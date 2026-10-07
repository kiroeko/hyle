#include "hyle/compute/route_policy.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>

#include "workload_fixture.h"

namespace {
using namespace hyle::compute;

static_assert(HYLE_OS_WINDOWS + HYLE_OS_LINUX + HYLE_OS_MACOS + HYLE_OS_UNKNOWN == 1);
static_assert(kBuildRoutePlatform ==
              (HYLE_OS_WINDOWS ? RoutePlatform::kWindows :
               HYLE_OS_LINUX ? RoutePlatform::kLinux :
               HYLE_OS_MACOS ? RoutePlatform::kMacOS : RoutePlatform::kUnknown));

inline constexpr auto kProfile = workload_test::kModel;
inline constexpr ProviderIdentity kProvider{
    {0x524f555445544553ULL, 1}, "route-test", "7.2"};
inline constexpr StorageFormatId kWeightFormat{0x524f555445544553ULL, 2};
inline constexpr StorageFormatId kScaleFormat{0x524f555445544553ULL, 3};
inline constexpr std::uint64_t kAuditedFormatNamespace = 0x48594c4541554449ULL;
inline constexpr StorageFormatId kGgufQ4K{kAuditedFormatNamespace, 1};
inline constexpr StorageFormatId kGgufQ5K{kAuditedFormatNamespace, 2};
inline constexpr StorageFormatId kGgufIq4Xs{kAuditedFormatNamespace, 3};
inline constexpr StorageFormatId kGgufQ8{kAuditedFormatNamespace, 4};
inline constexpr StorageFormatId kGgufMxfp4{kAuditedFormatNamespace, 5};

// These identities pin the model and artifact revisions recorded by the
// target-model audit. The digests bind this test fixture's reduced manifest;
// they do not claim to be Hugging Face object hashes.
inline constexpr ModelArtifactProfileDescription kQwen35Q4{
    .model_publisher = "Qwen",
    .model_repository = "Qwen/Qwen3.5-2B",
    .model_family = "qwen3_5",
    .model_name = "qwen3.5-2b",
    .model_revision = "15852e8c16360a2fea060d615a32b45270f8a8fc",
    .configuration_digest = kProfile.configuration_digest,
    .artifact_repository = "unsloth/Qwen3.5-2B-GGUF",
    .artifact_revision = "f6d5376be1edb4d416d56da11e5397a961aca8ae",
    .variant = "UD-Q4_K_XL",
    .manifest_digest =
        "c1dab0d3b4bec7704a4dbabff36aca0638ec10ce8e0dc2e8362441d25d29e3fc",
    .format_family = "gguf"};

inline constexpr ModelArtifactProfileDescription kQwen38MoEIq4{
    .model_publisher = "Qwen",
    .model_repository = "Qwen/Qwen3.8-2.4T-A95B",
    .model_family = "qwen3_8_moe",
    .model_name = "qwen3.8-2.4t-a95b",
    .model_revision = "207bd685a7e3696cfaff12ded7c6a7ea0f88c996",
    .configuration_digest =
        "146cfd2e88ea2a1e9953bf8b4edac3de2f63c1b8f51a37d8432f35b1f99932eb",
    .artifact_repository = "unsloth/Qwen3.8-2.4T-A95B-GGUF",
    .artifact_revision = "567d3e6ac26c5474b18311e619c04350fb9a5556",
    .variant = "UD-IQ4_XS",
    .manifest_digest =
        "b328acc71cb817e7e3636235c6caad1d4e33b20f1327c70816ae8ba202280e5f",
    .format_family = "gguf"};

inline constexpr ModelArtifactProfileDescription kKimiK3Q4{
    .model_publisher = "moonshotai",
    .model_repository = "moonshotai/Kimi-K3",
    .model_family = "kimi_k3",
    .model_name = "kimi-k3",
    .model_revision = "f831ab66814297da540d832a5235f8e904f29d06",
    .configuration_digest =
        "b3b7295f8d975bd91b609000eec0b97ee08ea69294bf2b59c8860424905df447",
    .artifact_repository = "unsloth/Kimi-K3-GGUF",
    .artifact_revision = "a0836360ce58dfec088d966a97f2ddc8a606279b",
    .variant = "UD-Q4_K_XL",
    .manifest_digest =
        "831309b6c6be2566b05aad7547ef09f14a9931e62d928a5fddc09bf9858863ba",
    .format_family = "gguf"};

TensorDesc audited_weight(DataType logical_type, StorageFormatId format,
                          std::array<std::int64_t, 2> shape) {
  return TensorDesc::opaque(logical_type, shape,
                            StorageFormat::opaque(format, 16).value(), 64)
      .value();
}

TensorDesc dense_matrix(DataType type, std::array<std::int64_t, 2> shape) {
  return TensorDesc::dense(type, shape).value();
}

struct QualifiedRequest final {
  std::array<RouteFact, 7> environment{
      RouteFact::number(route_fact::kDeviceVendorId, 0x10de),
      RouteFact::number(route_fact::kDeviceId, 0x2786),
      RouteFact::string(route_fact::kArchitecture, "ada"),
      RouteFact::number(route_fact::kArchitectureMajor, 8),
      RouteFact::number(route_fact::kArchitectureMinor, 9),
      RouteFact::number(route_fact::kDriverVersion, 13020),
      RouteFact::number(route_fact::kToolkitVersion, 13030)};
  std::array<RouteFactPredicate, 7> environment_predicates{
      RouteFactPredicate::number(route_fact::kDeviceVendorId, 0x10de),
      RouteFactPredicate::number(route_fact::kDeviceId, 0x2786),
      RouteFactPredicate::string(route_fact::kArchitecture, "ada"),
      RouteFactPredicate::number(route_fact::kArchitectureMajor, 8),
      RouteFactPredicate::number(route_fact::kArchitectureMinor, 9),
      RouteFactPredicate::range(route_fact::kDriverVersion, 13000, 13999),
      RouteFactPredicate::number(route_fact::kToolkitVersion, 13030)};
  std::array<RouteFactPredicate, 3> operation_predicates{
      RouteFactPredicate::number(
          operation_route_fact::kQuantizedWeightMode,
          static_cast<std::uint64_t>(QuantizedWeightMode::kSignedSymmetric4)),
      RouteFactPredicate::number(operation_route_fact::kQuantizationGroupSize,
                                 128),
      RouteFactPredicate::number(
          operation_route_fact::kAccumulationDataType,
          static_cast<std::uint64_t>(DataType::kFloat32))};
  std::array<RouteDimensionPredicate, 1> activation_dimensions{
      RouteDimensionPredicate::exact(1, 128)};
  std::array<RouteDimensionPredicate, 2> weight_dimensions{
      RouteDimensionPredicate::range(0, 64, 4096, 64),
      RouteDimensionPredicate::exact(1, 128)};
  std::array<RouteTensorPredicate, 4> ports{
      RouteTensorPredicate{.direction = RoutePortDirection::kInput,
                           .index = 0,
                           .data_type = DataType::kFloat16,
                           .storage_kind = RouteStorageKind::kPlain,
                           .minimum_rank = 2,
                           .maximum_rank = 2,
                           .dimensions = activation_dimensions},
      RouteTensorPredicate{.direction = RoutePortDirection::kInput,
                           .index = 1,
                           .data_type = DataType::kFloat16,
                           .storage_kind = RouteStorageKind::kOpaque,
                           .storage_format = kWeightFormat,
                           .minimum_rank = 2,
                           .maximum_rank = 2,
                           .dimensions = weight_dimensions},
      RouteTensorPredicate{.direction = RoutePortDirection::kInput,
                           .index = 2,
                           .data_type = DataType::kFloat16,
                           .storage_kind = RouteStorageKind::kOpaque,
                           .storage_format = kScaleFormat,
                           .minimum_rank = 2,
                           .maximum_rank = 2},
      RouteTensorPredicate{.direction = RoutePortDirection::kOutput,
                           .index = 0,
                           .data_type = DataType::kFloat16,
                           .storage_kind = RouteStorageKind::kPlain,
                           .minimum_rank = 2,
                           .maximum_rank = 2}};
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
        workload{ModelArtifactProfile::create(kProfile).value(),
                 WorkloadPhase::kDecode,
                 {7, 8, 2}} {}

  RoutePolicyEntry policy() const noexcept {
    return {.id = {0x524f555445544553ULL, 11, 1},
            .preference = 7,
            .provider = kProvider.id,
            .implementation = ProviderImplementationId{3},
            .selector = {
                .platform = RoutePlatform::kWindows,
                .route = NativeRoute::kCuda,
                .provider = kProvider.id,
                .provider_version = kProvider.version,
                .environment = environment_predicates,
                .profile = {.model_publisher = {},
                            .model_repository = kProfile.model_repository,
                            .model_family = kProfile.model_family,
                            .model_name = kProfile.model_name,
                            .model_revision = kProfile.model_revision,
                            .configuration_digest = {},
                            .artifact_repository = kProfile.artifact_repository,
                            .artifact_revision = kProfile.artifact_revision,
                            .variant = kProfile.variant,
                            .manifest_digest = kProfile.manifest_digest,
                            .format_family = kProfile.format_family},
                .phase = WorkloadPhase::kDecode,
                .region = {7, 8, 2},
                .operation = OperationKind::kQuantizedMatMul,
                .operation_facts = operation_predicates,
                .ports = ports}};
  }

  RouteMatchInput input() const noexcept {
    return {.platform = RoutePlatform::kWindows,
            .route = NativeRoute::kCuda,
            .provider = kProvider,
            .environment = environment,
            .profile = workload.profile.description(),
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

TEST(RoutePolicy, FullEnvironmentProfileOperationAndPortsMatch) {
  const QualifiedRequest request;
  const auto result = match_route_policy(request.policy(), request.input());
  EXPECT_TRUE(result.matched());
  EXPECT_EQ(result.mismatch, RouteMismatch::kNone);
}

TEST(RoutePolicy, UnknownAndDifferentEnvironmentFactsAreDistinct) {
  const QualifiedRequest request;
  auto input = request.input();
  input.environment =
      std::span(request.environment).first(request.environment.size() - 1);
  auto result = match_route_policy(request.policy(), input);
  EXPECT_EQ(result.mismatch, RouteMismatch::kEnvironmentMissing);
  EXPECT_EQ(result.fact, route_fact::kToolkitVersion);

  auto changed = request.environment;
  changed[2] = RouteFact::number(route_fact::kArchitecture, 89);
  input = request.input();
  input.environment = changed;
  result = match_route_policy(request.policy(), input);
  EXPECT_EQ(result.mismatch, RouteMismatch::kEnvironmentType);
  EXPECT_EQ(result.fact, route_fact::kArchitecture);

  changed = request.environment;
  changed[5] = RouteFact::number(route_fact::kDriverVersion, 14000);
  input.environment = changed;
  result = match_route_policy(request.policy(), input);
  EXPECT_EQ(result.mismatch, RouteMismatch::kEnvironmentValue);
}

TEST(RoutePolicy, ProfileOperationAndArbitraryPortMismatchesAreLocated) {
  const QualifiedRequest request;
  auto input = request.input();
  input.profile = {};
  EXPECT_EQ(match_route_policy(request.policy(), input).mismatch,
            RouteMismatch::kProfileMissing);

  input = request.input();
  auto other = input.profile;
  other.variant = "UD-Q4_K_XL";
  input.profile = other;
  EXPECT_EQ(match_route_policy(request.policy(), input).mismatch,
            RouteMismatch::kProfileValue);

  input = request.input();
  input.operation =
      Operation(QuantizedMatMul{.mode = QuantizedWeightMode::kSignedSymmetric8,
                                .group_size = 128,
                                .accumulation_type = DataType::kFloat32});
  auto result = match_route_policy(request.policy(), input);
  EXPECT_EQ(result.mismatch, RouteMismatch::kOperationFactValue);
  EXPECT_EQ(result.fact, operation_route_fact::kQuantizedWeightMode);

  input = request.input();
  input.inputs = std::span(request.inputs).first(2);
  result = match_route_policy(request.policy(), input);
  EXPECT_EQ(result.mismatch, RouteMismatch::kPortMissing);
  EXPECT_EQ(result.direction, RoutePortDirection::kInput);
  EXPECT_EQ(result.port, 2u);

  auto changed_inputs = request.inputs;
  changed_inputs[0] =
      TensorDesc::dense(DataType::kFloat16, std::array<std::int64_t, 2>{1, 127})
          .value();
  input = request.input();
  input.inputs = changed_inputs;
  result = match_route_policy(request.policy(), input);
  EXPECT_EQ(result.mismatch, RouteMismatch::kDimension);
  EXPECT_EQ(result.axis, 1u);
}

TEST(RoutePolicy, MalformedAndDuplicateSelectorsAreRejected) {
  std::array duplicate_facts{RouteFact::number(route_fact::kDeviceId, 1),
                             RouteFact::number(route_fact::kDeviceId, 1)};
  RouteMatchInput facts{.environment = duplicate_facts};
  EXPECT_EQ(match_route_selector({}, facts).mismatch,
            RouteMismatch::kInvalidFacts);

  std::array duplicate_predicates{
      RouteFactPredicate::number(route_fact::kDeviceId, 1),
      RouteFactPredicate::range(route_fact::kDeviceId, 1, 2)};
  RouteSelector duplicates{.environment = duplicate_predicates};
  EXPECT_EQ(match_route_selector(duplicates, {}).mismatch,
            RouteMismatch::kInvalidSelector);

  std::array invalid_range{
      RouteFactPredicate::range(route_fact::kDriverVersion, 9, 3)};
  RouteSelector invalid{.environment = invalid_range};
  EXPECT_EQ(match_route_selector(invalid, {}).mismatch,
            RouteMismatch::kInvalidSelector);

  std::array duplicate_ports{
      RouteTensorPredicate{.direction = RoutePortDirection::kInput, .index = 3},
      RouteTensorPredicate{.direction = RoutePortDirection::kInput,
                           .index = 3}};
  RouteSelector repeated{.ports = duplicate_ports};
  EXPECT_EQ(match_route_selector(repeated, {}).mismatch,
            RouteMismatch::kInvalidSelector);

  RoutePolicyEntry missing_identity{};
  EXPECT_EQ(match_route_policy(missing_identity, {}).mismatch,
            RouteMismatch::kInvalidPolicy);
}

TEST(RoutePolicy, FourNativeRoutesRemainIndependentColdFacts) {
  for (const auto route : {NativeRoute::kCpu, NativeRoute::kCuda,
                           NativeRoute::kApple, NativeRoute::kVulkan}) {
    RouteSelector selector{.route = route};
    RouteMatchInput input{.route = route};
    EXPECT_TRUE(match_route_selector(selector, input).matched());
    input.route = NativeRoute::kUnknown;
    EXPECT_EQ(match_route_selector(selector, input).mismatch,
              RouteMismatch::kRoute);
  }
}

TEST(RoutePolicy, HighIndexOutputAndMultipleOfShapeAreFirstClass) {
  const std::array dimensions{RouteDimensionPredicate::range(0, 32, 512, 32)};
  const std::array ports{
      RouteTensorPredicate{.direction = RoutePortDirection::kOutput,
                           .index = 3,
                           .data_type = DataType::kFloat32,
                           .storage_kind = RouteStorageKind::kPlain,
                           .minimum_rank = 1,
                           .maximum_rank = 1,
                           .dimensions = dimensions}};
  const RouteSelector selector{.ports = ports};
  const auto tensor =
      TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 1>{256});
  ASSERT_TRUE(tensor.has_value());
  const std::array outputs{*tensor, *tensor, *tensor, *tensor};
  EXPECT_TRUE(match_route_selector(selector, {.outputs = outputs}).matched());
}

TEST(RoutePolicy, ExactArtifactAndActualWeightFormatSeparateQwen35Variants) {
  const std::array official_ports{
      RouteTensorPredicate{.direction = RoutePortDirection::kInput,
                           .index = 1,
                           .data_type = DataType::kBFloat16,
                           .storage_kind = RouteStorageKind::kPlain}};
  const std::array quantized_ports{
      RouteTensorPredicate{.direction = RoutePortDirection::kInput,
                           .index = 1,
                           .data_type = DataType::kFloat16,
                           .storage_kind = RouteStorageKind::kOpaque,
                           .storage_format = kGgufQ4K}};
  const RouteSelector official{.profile = exact_profile_selector(kProfile),
                               .operation = OperationKind::kMatMul,
                               .ports = official_ports};
  const RouteSelector quantized{.profile = exact_profile_selector(kQwen35Q4),
                                .operation = OperationKind::kMatMul,
                                .ports = quantized_ports};

  const std::array official_inputs{
      dense_matrix(DataType::kBFloat16, {1, 2048}),
      dense_matrix(DataType::kBFloat16, {2048, 6144})};
  const std::array quantized_inputs{
      dense_matrix(DataType::kFloat16, {1, 2048}),
      audited_weight(DataType::kFloat16, kGgufQ4K, {2048, 6144})};
  const std::array official_outputs{
      dense_matrix(DataType::kBFloat16, {1, 6144})};
  const std::array quantized_outputs{
      dense_matrix(DataType::kFloat16, {1, 6144})};

  EXPECT_TRUE(match_route_selector(official, {.profile = kProfile,
                                              .operation = Operation(MatMul{}),
                                              .inputs = official_inputs,
                                              .outputs = official_outputs})
                  .matched());
  EXPECT_TRUE(match_route_selector(quantized, {.profile = kQwen35Q4,
                                               .operation = Operation(MatMul{}),
                                               .inputs = quantized_inputs,
                                               .outputs = quantized_outputs})
                  .matched());
  EXPECT_EQ(match_route_selector(official, {.profile = kQwen35Q4,
                                            .operation = Operation(MatMul{}),
                                            .inputs = quantized_inputs,
                                            .outputs = quantized_outputs})
                .mismatch,
            RouteMismatch::kProfileValue);
}

TEST(RoutePolicy, OneArtifactCanRouteDifferentTensorRolesIndependently) {
  const auto matches_weight = [](ModelArtifactProfileDescription profile,
                                 StorageFormatId format,
                                 DataType logical_type) {
    const std::array ports{
        RouteTensorPredicate{.direction = RoutePortDirection::kInput,
                             .index = 1,
                             .data_type = logical_type,
                             .storage_kind = RouteStorageKind::kOpaque,
                             .storage_format = format}};
    const RouteSelector selector{.profile = exact_profile_selector(profile),
                                 .operation = OperationKind::kMatMul,
                                 .ports = ports};
    const std::array inputs{dense_matrix(DataType::kFloat16, {1, 4096}),
                            audited_weight(logical_type, format, {4096, 4096})};
    const std::array outputs{dense_matrix(DataType::kFloat16, {1, 4096})};
    return match_route_selector(selector, {.profile = profile,
                                           .operation = Operation(MatMul{}),
                                           .inputs = inputs,
                                           .outputs = outputs})
        .matched();
  };
  const auto matches_f32_role = [](ModelArtifactProfileDescription profile) {
    const std::array ports{
        RouteTensorPredicate{.direction = RoutePortDirection::kInput,
                             .index = 1,
                             .data_type = DataType::kFloat32,
                             .storage_kind = RouteStorageKind::kPlain}};
    const RouteSelector selector{.profile = exact_profile_selector(profile),
                                 .operation = OperationKind::kMatMul,
                                 .ports = ports};
    const std::array inputs{dense_matrix(DataType::kFloat32, {1, 4096}),
                            dense_matrix(DataType::kFloat32, {4096, 512})};
    const std::array outputs{dense_matrix(DataType::kFloat32, {1, 512})};
    return match_route_selector(selector, {.profile = profile,
                                           .operation = Operation(MatMul{}),
                                           .inputs = inputs,
                                           .outputs = outputs})
        .matched();
  };

  // Qwen3.8's one UD-IQ4_XS artifact contains IQ4 routed experts,
  // Q5 shared experts and F32 router/state roles.
  EXPECT_TRUE(matches_weight(kQwen38MoEIq4, kGgufIq4Xs, DataType::kFloat16));
  EXPECT_TRUE(matches_weight(kQwen38MoEIq4, kGgufQ5K, DataType::kFloat16));
  EXPECT_TRUE(matches_f32_role(kQwen38MoEIq4));

  // Kimi K3's Q4-named artifact retains MXFP4 routed experts beside Q8
  // projections and F32 KDA/router/state tensors.
  EXPECT_TRUE(matches_weight(kKimiK3Q4, kGgufMxfp4, DataType::kFloat16));
  EXPECT_TRUE(matches_weight(kKimiK3Q4, kGgufQ8, DataType::kFloat16));
  EXPECT_TRUE(matches_f32_role(kKimiK3Q4));
}

} // namespace
