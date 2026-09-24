#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>

#include "hyle/compute/operation.h"
#include "hyle/compute/provider.h"
#include "hyle/compute/visibility.h"
#include "hyle/compute/workload.h"

namespace hyle::compute {

enum class RoutePlatform : std::uint8_t {
  kUnknown = 0,
  kWindows,
  kLinux,
  kMacOS,
};

enum class NativeRoute : std::uint8_t {
  kUnknown = 0,
  kCpu,
  kCuda,
  kApple,
  kVulkan,
};

#if defined(_WIN32)
inline constexpr RoutePlatform kBuildRoutePlatform = RoutePlatform::kWindows;
#elif defined(__APPLE__)
inline constexpr RoutePlatform kBuildRoutePlatform = RoutePlatform::kMacOS;
#elif defined(__linux__)
inline constexpr RoutePlatform kBuildRoutePlatform = RoutePlatform::kLinux;
#else
inline constexpr RoutePlatform kBuildRoutePlatform = RoutePlatform::kUnknown;
#endif

// Namespaced identities keep route-native encodings from being reinterpreted
// by another route. The version belongs to the fact schema, not the value.
struct RouteFactId final {
  std::uint64_t namespace_id = 0;
  std::uint64_t fact_id = 0;
  std::uint32_t version = 0;

  constexpr bool valid() const noexcept {
    return (namespace_id != 0 || fact_id != 0) && version != 0;
  }
  friend constexpr bool operator==(RouteFactId, RouteFactId) noexcept = default;
};

namespace route_fact {
inline constexpr std::uint64_t kNamespace = 0x48594c45524f5554ULL;
inline constexpr RouteFactId kDeviceVendorId{kNamespace, 1, 1};
inline constexpr RouteFactId kDeviceId{kNamespace, 2, 1};
inline constexpr RouteFactId kDeviceName{kNamespace, 3, 1};
inline constexpr RouteFactId kArchitecture{kNamespace, 4, 1};
inline constexpr RouteFactId kArchitectureMajor{kNamespace, 5, 1};
inline constexpr RouteFactId kArchitectureMinor{kNamespace, 6, 1};
inline constexpr RouteFactId kDriverName{kNamespace, 7, 1};
inline constexpr RouteFactId kDriverVersion{kNamespace, 8, 1};
inline constexpr RouteFactId kRuntimeVersion{kNamespace, 9, 1};
inline constexpr RouteFactId kToolkitVersion{kNamespace, 10, 1};
inline constexpr RouteFactId kSdkVersion{kNamespace, 11, 1};
inline constexpr RouteFactId kApiVersion{kNamespace, 12, 1};
inline constexpr RouteFactId kSubgroupSize{kNamespace, 13, 1};
inline constexpr RouteFactId kSharedMemoryBytes{kNamespace, 14, 1};
} // namespace route_fact

namespace operation_route_fact {
inline constexpr std::uint64_t kNamespace = 0x48594c454f504641ULL;
inline constexpr RouteFactId kQuantizedWeightMode{kNamespace, 1, 1};
inline constexpr RouteFactId kQuantizationGroupSize{kNamespace, 2, 1};
inline constexpr RouteFactId kAccumulationDataType{kNamespace, 3, 1};
} // namespace operation_route_fact

enum class RouteFactKind : std::uint8_t {
  kUnknown = 0,
  kUnsigned,
  kText,
};

struct RouteFact final {
  RouteFactId id{};
  RouteFactKind kind = RouteFactKind::kUnknown;
  std::uint64_t unsigned_value = 0;
  std::string_view text{};

  static constexpr RouteFact number(RouteFactId id,
                                    std::uint64_t value) noexcept {
    return {
        .id = id, .kind = RouteFactKind::kUnsigned, .unsigned_value = value};
  }
  static constexpr RouteFact string(RouteFactId id,
                                    std::string_view value) noexcept {
    return {.id = id, .kind = RouteFactKind::kText, .text = value};
  }
  constexpr bool valid() const noexcept {
    return id.valid() && ((kind == RouteFactKind::kUnsigned && text.empty()) ||
                          (kind == RouteFactKind::kText && !text.empty() &&
                           unsigned_value == 0));
  }
};

enum class RouteFactPredicateKind : std::uint8_t {
  kUnknown = 0,
  kUnsignedExact,
  kUnsignedRange,
  kTextExact,
};

struct RouteFactPredicate final {
  RouteFactId id{};
  RouteFactPredicateKind kind = RouteFactPredicateKind::kUnknown;
  std::uint64_t minimum = 0;
  std::uint64_t maximum = 0;
  std::string_view text{};

  static constexpr RouteFactPredicate number(RouteFactId id,
                                             std::uint64_t value) noexcept {
    return {.id = id,
            .kind = RouteFactPredicateKind::kUnsignedExact,
            .minimum = value,
            .maximum = value};
  }
  static constexpr RouteFactPredicate
  range(RouteFactId id, std::uint64_t minimum, std::uint64_t maximum) noexcept {
    return {.id = id,
            .kind = RouteFactPredicateKind::kUnsignedRange,
            .minimum = minimum,
            .maximum = maximum};
  }
  static constexpr RouteFactPredicate string(RouteFactId id,
                                             std::string_view value) noexcept {
    return {
        .id = id, .kind = RouteFactPredicateKind::kTextExact, .text = value};
  }
  constexpr bool valid() const noexcept {
    return id.valid() && ((kind == RouteFactPredicateKind::kUnsignedExact &&
                           text.empty() && minimum == maximum) ||
                          (kind == RouteFactPredicateKind::kUnsignedRange &&
                           text.empty() && minimum <= maximum) ||
                          (kind == RouteFactPredicateKind::kTextExact &&
                           !text.empty() && minimum == 0 && maximum == 0));
  }
};

// Sparse exact predicates over an otherwise complete workload profile.
struct ModelArtifactProfileSelector final {
  std::string_view model_publisher;
  std::string_view model_repository;
  std::string_view model_family;
  std::string_view model_name;
  std::string_view model_revision;
  std::string_view configuration_digest;
  std::string_view artifact_repository;
  std::string_view artifact_revision;
  std::string_view variant;
  std::string_view manifest_digest;
  std::string_view format_family;

  constexpr bool empty() const noexcept {
    return model_publisher.empty() && model_repository.empty() &&
           model_family.empty() && model_name.empty() &&
           model_revision.empty() && configuration_digest.empty() &&
           artifact_repository.empty() && artifact_revision.empty() &&
           variant.empty() && manifest_digest.empty() && format_family.empty();
  }
};

constexpr ModelArtifactProfileSelector
exact_profile_selector(ModelArtifactProfileDescription profile) noexcept {
  return {profile.model_publisher,
          profile.model_repository,
          profile.model_family,
          profile.model_name,
          profile.model_revision,
          profile.configuration_digest,
          profile.artifact_repository,
          profile.artifact_revision,
          profile.variant,
          profile.manifest_digest,
          profile.format_family};
}

enum class RoutePortDirection : std::uint8_t {
  kUnknown = 0,
  kInput,
  kOutput,
};

enum class RouteStorageKind : std::uint8_t {
  kAny = 0,
  kPlain,
  kBlocked,
  kOpaque,
};

struct RouteDimensionPredicate final {
  std::uint8_t axis = 0;
  std::int64_t minimum = 1;
  std::int64_t maximum = (std::numeric_limits<std::int64_t>::max)();
  std::int64_t multiple = 1;

  static constexpr RouteDimensionPredicate exact(std::uint8_t axis,
                                                 std::int64_t value) noexcept {
    return {.axis = axis, .minimum = value, .maximum = value};
  }
  static constexpr RouteDimensionPredicate
  range(std::uint8_t axis, std::int64_t minimum, std::int64_t maximum,
        std::int64_t multiple = 1) noexcept {
    return {.axis = axis,
            .minimum = minimum,
            .maximum = maximum,
            .multiple = multiple};
  }
  constexpr bool valid() const noexcept {
    return axis < kMaxTensorRank && minimum > 0 && minimum <= maximum &&
           multiple > 0;
  }
};

struct RouteTensorPredicate final {
  RoutePortDirection direction = RoutePortDirection::kUnknown;
  std::uint32_t index = 0;
  DataType data_type = DataType::kUnknown;
  RouteStorageKind storage_kind = RouteStorageKind::kAny;
  StorageFormatId storage_format{};
  std::uint8_t minimum_rank = 0;
  std::uint8_t maximum_rank = static_cast<std::uint8_t>(kMaxTensorRank);
  std::span<const RouteDimensionPredicate> dimensions{};
};

struct RouteSelector final {
  RoutePlatform platform = RoutePlatform::kUnknown;
  NativeRoute route = NativeRoute::kUnknown;
  ProviderId provider{};
  std::string_view provider_version{};
  std::span<const RouteFactPredicate> environment{};

  ModelArtifactProfileSelector profile{};
  WorkloadPhase phase = WorkloadPhase::kUnspecified;
  RegionSemanticId region{};

  OperationKind operation = OperationKind::kUnknown;
  std::span<const RouteFactPredicate> operation_facts{};
  std::span<const RouteTensorPredicate> ports{};
};

struct RoutePolicyEntry final {
  RoutePolicyId id{};
  std::uint32_t preference = 0;
  ProviderId provider{};
  ProviderImplementationId implementation{};
  RouteSelector selector{};
};

struct RouteMatchInput final {
  RoutePlatform platform = RoutePlatform::kUnknown;
  NativeRoute route = NativeRoute::kUnknown;
  ProviderIdentity provider{};
  std::span<const RouteFact> environment{};
  ModelArtifactProfileDescription profile{};
  WorkloadPhase phase = WorkloadPhase::kUnspecified;
  RegionSemanticId region{};
  Operation operation{};
  std::span<const TensorDesc> inputs{};
  std::span<const TensorDesc> outputs{};
};

enum class RouteMismatch : std::uint8_t {
  kNone = 0,
  kInvalidSelector,
  kInvalidPolicy,
  kInvalidFacts,
  kPlatform,
  kRoute,
  kProvider,
  kProviderVersion,
  kEnvironmentMissing,
  kEnvironmentType,
  kEnvironmentValue,
  kProfileMissing,
  kProfileValue,
  kPhase,
  kRegion,
  kOperation,
  kOperationFactMissing,
  kOperationFactValue,
  kPortMissing,
  kDataType,
  kStorageKind,
  kStorageFormat,
  kRank,
  kDimension,
};

struct RouteMatchResult final {
  RouteMismatch mismatch = RouteMismatch::kNone;
  RouteFactId fact{};
  RoutePortDirection direction = RoutePortDirection::kUnknown;
  std::uint32_t port = 0;
  std::uint8_t axis = 0;

  constexpr bool matched() const noexcept {
    return mismatch == RouteMismatch::kNone;
  }
  constexpr explicit operator bool() const noexcept { return matched(); }
};

// Allocation-free cold matchers. They report the first mismatch in stable
// field order and retain no borrowed span.
HYLE_COMPUTE_API RouteMatchResult match_route_selector(
    const RouteSelector &selector, const RouteMatchInput &input) noexcept;
HYLE_COMPUTE_API RouteMatchResult match_route_policy(
    const RoutePolicyEntry &policy, const RouteMatchInput &input) noexcept;

} // namespace hyle::compute
