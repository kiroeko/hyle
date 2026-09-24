#include "hyle/compute/route_policy.h"

#include <cstddef>

namespace hyle::compute {
namespace {

constexpr bool valid_platform(RoutePlatform value) noexcept {
  return value >= RoutePlatform::kUnknown && value <= RoutePlatform::kMacOS;
}

constexpr bool valid_route(NativeRoute value) noexcept {
  return value >= NativeRoute::kUnknown && value <= NativeRoute::kVulkan;
}

constexpr bool valid_phase(WorkloadPhase value) noexcept {
  return value >= WorkloadPhase::kUnspecified &&
         value <= WorkloadPhase::kDecode;
}

constexpr bool valid_operation(OperationKind value) noexcept {
  return value >= OperationKind::kUnknown && value <= OperationKind::kCustom;
}

constexpr bool valid_data_type(DataType value) noexcept {
  return value >= DataType::kUnknown && value <= DataType::kUInt64;
}

constexpr bool valid_storage_kind(RouteStorageKind value) noexcept {
  return value >= RouteStorageKind::kAny && value <= RouteStorageKind::kOpaque;
}

constexpr bool is_lower_hex_digest(std::string_view value) noexcept {
  if (value.size() != 64)
    return false;
  for (const char character : value) {
    if (!((character >= '0' && character <= '9') ||
          (character >= 'a' && character <= 'f')))
      return false;
  }
  return true;
}

constexpr bool
valid_profile_description(ModelArtifactProfileDescription profile) noexcept {
  return profile.empty() ||
         (profile.complete() &&
          is_lower_hex_digest(profile.configuration_digest) &&
          is_lower_hex_digest(profile.manifest_digest));
}

constexpr bool
valid_profile_selector(const ModelArtifactProfileSelector &profile) noexcept {
  return (profile.configuration_digest.empty() ||
          is_lower_hex_digest(profile.configuration_digest)) &&
         (profile.manifest_digest.empty() ||
          is_lower_hex_digest(profile.manifest_digest));
}

template <class Value> bool duplicates(std::span<const Value> values) noexcept {
  for (std::size_t right = 0; right < values.size(); ++right)
    for (std::size_t left = 0; left < right; ++left)
      if (values[left].id == values[right].id)
        return true;
  return false;
}

bool duplicate_ports(std::span<const RouteTensorPredicate> ports) noexcept {
  for (std::size_t right = 0; right < ports.size(); ++right)
    for (std::size_t left = 0; left < right; ++left)
      if (ports[left].direction == ports[right].direction &&
          ports[left].index == ports[right].index)
        return true;
  return false;
}

bool valid_port(const RouteTensorPredicate &port) noexcept {
  if ((port.direction != RoutePortDirection::kInput &&
       port.direction != RoutePortDirection::kOutput) ||
      !valid_data_type(port.data_type) ||
      !valid_storage_kind(port.storage_kind) ||
      port.minimum_rank > port.maximum_rank ||
      port.maximum_rank > kMaxTensorRank)
    return false;
  for (std::size_t right = 0; right < port.dimensions.size(); ++right) {
    if (!port.dimensions[right].valid())
      return false;
    for (std::size_t left = 0; left < right; ++left)
      if (port.dimensions[left].axis == port.dimensions[right].axis)
        return false;
  }
  return true;
}

bool valid_selector(const RouteSelector &selector) noexcept {
  if (!valid_platform(selector.platform) || !valid_route(selector.route) ||
      !valid_phase(selector.phase) || !valid_operation(selector.operation) ||
      !valid_profile_selector(selector.profile) ||
      (!selector.region.empty() && !selector.region.valid()) ||
      (!selector.provider_version.empty() && !selector.provider.valid()))
    return false;
  for (const auto &predicate : selector.environment)
    if (!predicate.valid())
      return false;
  for (const auto &predicate : selector.operation_facts)
    if (!predicate.valid() ||
        predicate.kind == RouteFactPredicateKind::kTextExact)
      return false;
  for (const auto &port : selector.ports)
    if (!valid_port(port))
      return false;
  return !duplicates(selector.environment) &&
         !duplicates(selector.operation_facts) &&
         !duplicate_ports(selector.ports);
}

bool valid_input(const RouteMatchInput &input) noexcept {
  if (!valid_platform(input.platform) || !valid_route(input.route) ||
      !valid_phase(input.phase) ||
      (!input.region.empty() && !input.region.valid()) ||
      !valid_profile_description(input.profile))
    return false;
  for (const auto &fact : input.environment)
    if (!fact.valid())
      return false;
  return !duplicates(input.environment);
}

RouteMatchResult
mismatch(RouteMismatch value, RouteFactId fact = {},
         RoutePortDirection direction = RoutePortDirection::kUnknown,
         std::uint32_t port = 0, std::uint8_t axis = 0) noexcept {
  return {.mismatch = value,
          .fact = fact,
          .direction = direction,
          .port = port,
          .axis = axis};
}

const RouteFact *find_fact(std::span<const RouteFact> facts,
                           RouteFactId id) noexcept {
  for (const auto &fact : facts)
    if (fact.id == id)
      return &fact;
  return nullptr;
}

RouteMatchResult
match_fact_predicates(std::span<const RouteFactPredicate> predicates,
                      std::span<const RouteFact> facts) noexcept {
  for (const auto &predicate : predicates) {
    const auto *fact = find_fact(facts, predicate.id);
    if (fact == nullptr)
      return mismatch(RouteMismatch::kEnvironmentMissing, predicate.id);
    if ((predicate.kind == RouteFactPredicateKind::kTextExact &&
         fact->kind != RouteFactKind::kText) ||
        (predicate.kind != RouteFactPredicateKind::kTextExact &&
         fact->kind != RouteFactKind::kUnsigned))
      return mismatch(RouteMismatch::kEnvironmentType, predicate.id);
    const bool matches =
        predicate.kind == RouteFactPredicateKind::kTextExact
            ? fact->text == predicate.text
        : predicate.kind == RouteFactPredicateKind::kUnsignedExact
            ? fact->unsigned_value == predicate.minimum
            : fact->unsigned_value >= predicate.minimum &&
                  fact->unsigned_value <= predicate.maximum;
    if (!matches)
      return mismatch(RouteMismatch::kEnvironmentValue, predicate.id);
  }
  return {};
}

enum class ProfileMatch : std::uint8_t { kMatch, kMissing, kDifferent };

ProfileMatch match_profile_field(std::string_view required,
                                 std::string_view actual) noexcept {
  if (required.empty())
    return ProfileMatch::kMatch;
  if (actual.empty())
    return ProfileMatch::kMissing;
  return required == actual ? ProfileMatch::kMatch : ProfileMatch::kDifferent;
}

RouteMatchResult
match_profile(const ModelArtifactProfileSelector &required,
              ModelArtifactProfileDescription actual) noexcept {
  const auto fields = {
      match_profile_field(required.model_publisher, actual.model_publisher),
      match_profile_field(required.model_repository, actual.model_repository),
      match_profile_field(required.model_family, actual.model_family),
      match_profile_field(required.model_name, actual.model_name),
      match_profile_field(required.model_revision, actual.model_revision),
      match_profile_field(required.configuration_digest,
                          actual.configuration_digest),
      match_profile_field(required.artifact_repository,
                          actual.artifact_repository),
      match_profile_field(required.artifact_revision, actual.artifact_revision),
      match_profile_field(required.variant, actual.variant),
      match_profile_field(required.manifest_digest, actual.manifest_digest),
      match_profile_field(required.format_family, actual.format_family)};
  for (const auto result : fields) {
    if (result == ProfileMatch::kMissing)
      return mismatch(RouteMismatch::kProfileMissing);
    if (result == ProfileMatch::kDifferent)
      return mismatch(RouteMismatch::kProfileValue);
  }
  return {};
}

bool operation_fact(const Operation &operation, RouteFactId id,
                    std::uint64_t *value) noexcept {
  if (id == operation_route_fact::kAccumulationDataType) {
    const auto type = operation_traits(operation).accumulation_type;
    if (type == DataType::kUnknown)
      return false;
    *value = static_cast<std::uint64_t>(type);
    return true;
  }
  const auto *quantized = operation.get_if<QuantizedMatMul>();
  if (quantized == nullptr)
    return false;
  if (id == operation_route_fact::kQuantizedWeightMode) {
    *value = static_cast<std::uint64_t>(quantized->mode);
    return true;
  }
  if (id == operation_route_fact::kQuantizationGroupSize) {
    *value = quantized->group_size;
    return true;
  }
  return false;
}

RouteMatchResult
match_operation_facts(std::span<const RouteFactPredicate> predicates,
                      const Operation &operation) noexcept {
  for (const auto &predicate : predicates) {
    std::uint64_t value = 0;
    if (!operation_fact(operation, predicate.id, &value))
      return mismatch(RouteMismatch::kOperationFactMissing, predicate.id);
    const bool matches =
        predicate.kind == RouteFactPredicateKind::kUnsignedExact
            ? value == predicate.minimum
            : value >= predicate.minimum && value <= predicate.maximum;
    if (!matches)
      return mismatch(RouteMismatch::kOperationFactValue, predicate.id);
  }
  return {};
}

const TensorDesc *find_port(const RouteTensorPredicate &predicate,
                            const RouteMatchInput &input) noexcept {
  const auto values = predicate.direction == RoutePortDirection::kInput
                          ? input.inputs
                          : input.outputs;
  return predicate.index < values.size() ? &values[predicate.index] : nullptr;
}

RouteStorageKind storage_kind(const TensorDesc &tensor) noexcept {
  switch (tensor.storage_format().kind()) {
  case StorageFormatKind::kPlain:
    return RouteStorageKind::kPlain;
  case StorageFormatKind::kBlocked:
    return RouteStorageKind::kBlocked;
  case StorageFormatKind::kOpaque:
    return RouteStorageKind::kOpaque;
  }
  return RouteStorageKind::kAny;
}

RouteMatchResult match_ports(std::span<const RouteTensorPredicate> predicates,
                             const RouteMatchInput &input) noexcept {
  for (const auto &predicate : predicates) {
    const auto *tensor = find_port(predicate, input);
    if (tensor == nullptr || !tensor->valid())
      return mismatch(RouteMismatch::kPortMissing, {}, predicate.direction,
                      predicate.index);
    if (predicate.data_type != DataType::kUnknown &&
        predicate.data_type != tensor->data_type())
      return mismatch(RouteMismatch::kDataType, {}, predicate.direction,
                      predicate.index);
    if (predicate.storage_kind != RouteStorageKind::kAny &&
        predicate.storage_kind != storage_kind(*tensor))
      return mismatch(RouteMismatch::kStorageKind, {}, predicate.direction,
                      predicate.index);
    if (predicate.storage_format.valid() &&
        predicate.storage_format != tensor->storage_format().id())
      return mismatch(RouteMismatch::kStorageFormat, {}, predicate.direction,
                      predicate.index);
    if (tensor->rank() < predicate.minimum_rank ||
        tensor->rank() > predicate.maximum_rank)
      return mismatch(RouteMismatch::kRank, {}, predicate.direction,
                      predicate.index);
    for (const auto &dimension : predicate.dimensions) {
      if (dimension.axis >= tensor->rank())
        return mismatch(RouteMismatch::kDimension, {}, predicate.direction,
                        predicate.index, dimension.axis);
      const auto extent = tensor->dimensions()[dimension.axis];
      if (extent < dimension.minimum || extent > dimension.maximum ||
          extent % dimension.multiple != 0)
        return mismatch(RouteMismatch::kDimension, {}, predicate.direction,
                        predicate.index, dimension.axis);
    }
  }
  return {};
}

} // namespace

RouteMatchResult match_route_selector(const RouteSelector &selector,
                                      const RouteMatchInput &input) noexcept {
  if (!valid_selector(selector))
    return mismatch(RouteMismatch::kInvalidSelector);
  if (!valid_input(input))
    return mismatch(RouteMismatch::kInvalidFacts);

  if (selector.platform != RoutePlatform::kUnknown &&
      selector.platform != input.platform)
    return mismatch(RouteMismatch::kPlatform);
  if (selector.route != NativeRoute::kUnknown && selector.route != input.route)
    return mismatch(RouteMismatch::kRoute);
  if (selector.provider.valid() && selector.provider != input.provider.id)
    return mismatch(RouteMismatch::kProvider);
  if (!selector.provider_version.empty() &&
      selector.provider_version != input.provider.version)
    return mismatch(RouteMismatch::kProviderVersion);

  auto result = match_fact_predicates(selector.environment, input.environment);
  if (!result)
    return result;
  result = match_profile(selector.profile, input.profile);
  if (!result)
    return result;
  if (selector.phase != WorkloadPhase::kUnspecified &&
      selector.phase != input.phase)
    return mismatch(RouteMismatch::kPhase);
  if (!selector.region.empty() && selector.region != input.region)
    return mismatch(RouteMismatch::kRegion);
  if (selector.operation != OperationKind::kUnknown &&
      selector.operation != input.operation.kind())
    return mismatch(RouteMismatch::kOperation);
  result = match_operation_facts(selector.operation_facts, input.operation);
  if (!result)
    return result;
  return match_ports(selector.ports, input);
}

RouteMatchResult match_route_policy(const RoutePolicyEntry &policy,
                                    const RouteMatchInput &input) noexcept {
  if (!policy.id.valid() || !policy.provider.valid() ||
      !policy.implementation.valid() ||
      (policy.selector.provider.valid() &&
       policy.selector.provider != policy.provider))
    return mismatch(RouteMismatch::kInvalidPolicy);
  if (input.provider.id != policy.provider)
    return mismatch(RouteMismatch::kProvider);
  return match_route_selector(policy.selector, input);
}

} // namespace hyle::compute
