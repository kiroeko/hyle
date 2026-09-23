#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

#include "hyle/compute/operation.h"
#include "hyle/compute/provider.h"
#include "hyle/compute/visibility.h"
#include "hyle/compute/workload.h"

namespace hyle::compute {

// Stable target identities for cold qualification. These values describe the
// selected native route; they are not build switches and never select or load
// a Provider by themselves.
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

// Extensible route-native fact identity. Builtin IDs below cover facts shared
// by multiple routes. A Provider may define its own namespaced IDs for facts
// whose encoding has meaning only to that implementation.
struct RouteAttributeId final {
  std::uint64_t namespace_id = 0;
  std::uint64_t attribute_id = 0;

  constexpr bool valid() const noexcept {
    return namespace_id != 0 || attribute_id != 0;
  }
  friend constexpr bool operator==(RouteAttributeId,
                                   RouteAttributeId) noexcept = default;
};

namespace route_attribute {
inline constexpr std::uint64_t kNamespace = 0x48594c45524f5554ULL;
inline constexpr RouteAttributeId kDeviceVendorId{kNamespace, 1};
inline constexpr RouteAttributeId kDeviceId{kNamespace, 2};
inline constexpr RouteAttributeId kDeviceName{kNamespace, 3};
inline constexpr RouteAttributeId kArchitecture{kNamespace, 4};
inline constexpr RouteAttributeId kArchitectureMajor{kNamespace, 5};
inline constexpr RouteAttributeId kArchitectureMinor{kNamespace, 6};
inline constexpr RouteAttributeId kDriverName{kNamespace, 7};
inline constexpr RouteAttributeId kDriverVersion{kNamespace, 8};
inline constexpr RouteAttributeId kRuntimeVersion{kNamespace, 9};
inline constexpr RouteAttributeId kToolkitVersion{kNamespace, 10};
inline constexpr RouteAttributeId kSdkVersion{kNamespace, 11};
inline constexpr RouteAttributeId kApiVersion{kNamespace, 12};
inline constexpr RouteAttributeId kSubgroupSize{kNamespace, 13};
inline constexpr RouteAttributeId kSharedMemoryBytes{kNamespace, 14};
} // namespace route_attribute

enum class RouteAttributeKind : std::uint8_t {
  kUnknown = 0,
  kUnsigned,
  kText,
};

// One present fact. Absence from the facts span means unknown; an explicit
// numeric zero remains a known value. Empty text is invalid rather than an
// alternate spelling of unknown.
struct RouteAttribute final {
  RouteAttributeId id{};
  RouteAttributeKind kind = RouteAttributeKind::kUnknown;
  std::uint64_t unsigned_value = 0;
  std::string_view text{};

  static constexpr RouteAttribute number(RouteAttributeId id,
                                         std::uint64_t value) noexcept {
    return {.id = id,
            .kind = RouteAttributeKind::kUnsigned,
            .unsigned_value = value};
  }
  static constexpr RouteAttribute string(RouteAttributeId id,
                                         std::string_view value) noexcept {
    return {.id = id, .kind = RouteAttributeKind::kText, .text = value};
  }
  constexpr bool valid() const noexcept {
    return id.valid() &&
           ((kind == RouteAttributeKind::kUnsigned && text.empty()) ||
            (kind == RouteAttributeKind::kText && !text.empty() &&
             unsigned_value == 0));
  }
};

enum class RoutePredicateKind : std::uint8_t {
  kUnknown = 0,
  kUnsignedExact,
  kUnsignedRange,
  kTextExact,
};

// A numeric range is inclusive. Attribute encodings remain route-native: a
// rule that compares a CUDA integer version must also constrain the route to
// CUDA and must not reinterpret that value as an Apple or Vulkan version.
struct RouteAttributePredicate final {
  RouteAttributeId id{};
  RoutePredicateKind kind = RoutePredicateKind::kUnknown;
  std::uint64_t minimum = 0;
  std::uint64_t maximum = 0;
  std::string_view text{};

  static constexpr RouteAttributePredicate
  number(RouteAttributeId id, std::uint64_t value) noexcept {
    return {.id = id,
            .kind = RoutePredicateKind::kUnsignedExact,
            .minimum = value,
            .maximum = value};
  }
  static constexpr RouteAttributePredicate
  range(RouteAttributeId id, std::uint64_t minimum,
        std::uint64_t maximum) noexcept {
    return {.id = id,
            .kind = RoutePredicateKind::kUnsignedRange,
            .minimum = minimum,
            .maximum = maximum};
  }
  static constexpr RouteAttributePredicate
  string(RouteAttributeId id, std::string_view value) noexcept {
    return {.id = id, .kind = RoutePredicateKind::kTextExact, .text = value};
  }
  constexpr bool valid() const noexcept {
    return id.valid() && ((kind == RoutePredicateKind::kUnsignedExact &&
                           text.empty() && minimum == maximum) ||
                          (kind == RoutePredicateKind::kUnsignedRange &&
                           text.empty() && minimum <= maximum) ||
                          (kind == RoutePredicateKind::kTextExact &&
                           !text.empty() && minimum == 0 && maximum == 0));
  }
};

// Whether the operation must use the grouped Q4/Q8 semantic operation. This
// is deliberately separate from FP8/BF16/F16 dtype and physical-format rules.
enum class RouteQuantizationPresence : std::uint8_t {
  kAny = 0,
  kNone,
  kQuantizedMatMul,
};

struct RouteQuantizationRule final {
  RouteQuantizationPresence presence = RouteQuantizationPresence::kAny;
  std::optional<QuantizedWeightMode> mode{};
  std::uint32_t group_size = 0;
  StorageFormatId weight_format{};
  StorageFormatId scale_format{};
};

// A sparse conjunction. Unknown/empty fields are wildcards in the rule. Every
// named field must be present and equal in the facts. Provider code owns the
// order of multiple rules and the mature fallback; this value never ranks or
// discovers Providers.
struct RouteRule final {
  RoutePlatform platform = RoutePlatform::kUnknown;
  NativeRoute route = NativeRoute::kUnknown;
  ProviderId provider{};
  std::string_view provider_version{};
  std::span<const RouteAttributePredicate> attributes{};

  ModelDescription model{};
  WorkloadPhase phase = WorkloadPhase::kUnspecified;
  RegionSemanticId region{};

  OperationKind operation = OperationKind::kUnknown;
  DataType input0_type = DataType::kUnknown;
  DataType input1_type = DataType::kUnknown;
  DataType output0_type = DataType::kUnknown;
  StorageFormatId input0_format{};
  StorageFormatId input1_format{};
  RouteQuantizationRule quantization{};
};

// Borrowed only for match_route_rule(). Route-native facts should come from
// the Context/Provider snapshot; workload and numeric facts must be the real
// request passed to propose/compile.
struct RouteMatchInput final {
  RoutePlatform platform = RoutePlatform::kUnknown;
  NativeRoute route = NativeRoute::kUnknown;
  ProviderIdentity provider{};
  std::span<const RouteAttribute> attributes{};
  ModelDescription model{};
  WorkloadPhase phase = WorkloadPhase::kUnspecified;
  RegionSemanticId region{};
  Operation operation{};
  std::span<const TensorDesc> inputs{};
  std::span<const TensorDesc> outputs{};
};

enum class RouteMismatch : std::uint8_t {
  kNone = 0,
  kInvalidRule,
  kInvalidFacts,
  kPlatform,
  kRoute,
  kProvider,
  kProviderVersion,
  kAttributeMissing,
  kAttributeType,
  kAttributeValue,
  kModel,
  kPhase,
  kRegion,
  kOperation,
  kDataType,
  kStorageFormat,
  kQuantization,
  kQuantizationMode,
  kQuantizationGroup,
};

struct RouteMatchResult final {
  RouteMismatch mismatch = RouteMismatch::kNone;
  RouteAttributeId attribute{};

  constexpr bool matched() const noexcept {
    return mismatch == RouteMismatch::kNone;
  }
  constexpr explicit operator bool() const noexcept { return matched(); }
};

// Pure, allocation-free cold matcher. It reports the first mismatch in the
// stable field order above; no diagnostic string or borrowed span survives.
HYLE_COMPUTE_API RouteMatchResult
match_route_rule(const RouteRule &rule, const RouteMatchInput &input) noexcept;

} // namespace hyle::compute
