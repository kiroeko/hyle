#include "hyle/compute/route_rule.h"

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

constexpr bool
valid_quantization_presence(RouteQuantizationPresence value) noexcept {
  return value >= RouteQuantizationPresence::kAny &&
         value <= RouteQuantizationPresence::kQuantizedMatMul;
}

constexpr bool valid_quantized_mode(QuantizedWeightMode value) noexcept {
  return value >= QuantizedWeightMode::kSignedSymmetric4 &&
         value <= QuantizedWeightMode::kSignedSymmetric8;
}

bool duplicate_predicates(
    std::span<const RouteAttributePredicate> predicates) noexcept {
  for (std::size_t right = 0; right < predicates.size(); ++right)
    for (std::size_t left = 0; left < right; ++left)
      if (predicates[left].id == predicates[right].id)
        return true;
  return false;
}

bool duplicate_attributes(std::span<const RouteAttribute> attributes) noexcept {
  for (std::size_t right = 0; right < attributes.size(); ++right)
    for (std::size_t left = 0; left < right; ++left)
      if (attributes[left].id == attributes[right].id)
        return true;
  return false;
}

bool valid_model_rule(ModelDescription model) noexcept {
  return !((model.family.empty() &&
            (!model.name.empty() || !model.revision.empty() ||
             !model.configuration.empty())) ||
           (model.name.empty() &&
            (!model.revision.empty() || !model.configuration.empty())));
}

bool valid_rule(const RouteRule &rule) noexcept {
  if (!valid_platform(rule.platform) || !valid_route(rule.route) ||
      !valid_phase(rule.phase) || !valid_operation(rule.operation) ||
      !valid_data_type(rule.input0_type) ||
      !valid_data_type(rule.input1_type) ||
      !valid_data_type(rule.output0_type) || !valid_model_rule(rule.model) ||
      (!rule.region.empty() && !rule.region.valid()) ||
      (!rule.provider_version.empty() && !rule.provider.valid()) ||
      !valid_quantization_presence(rule.quantization.presence) ||
      (rule.quantization.mode &&
       !valid_quantized_mode(*rule.quantization.mode)))
    return false;
  const bool quantized_fields = rule.quantization.mode.has_value() ||
                                rule.quantization.group_size != 0 ||
                                rule.quantization.weight_format.valid() ||
                                rule.quantization.scale_format.valid();
  if (quantized_fields &&
      rule.quantization.presence != RouteQuantizationPresence::kQuantizedMatMul)
    return false;
  for (const auto &predicate : rule.attributes)
    if (!predicate.valid())
      return false;
  return !duplicate_predicates(rule.attributes);
}

bool valid_facts(const RouteMatchInput &input) noexcept {
  if (!valid_platform(input.platform) || !valid_route(input.route) ||
      !valid_model_rule(input.model) || !valid_phase(input.phase) ||
      (!input.region.empty() && !input.region.valid()))
    return false;
  for (const auto &attribute : input.attributes)
    if (!attribute.valid())
      return false;
  return !duplicate_attributes(input.attributes);
}

RouteMatchResult mismatch(RouteMismatch value,
                          RouteAttributeId attribute = {}) noexcept {
  return {.mismatch = value, .attribute = attribute};
}

const RouteAttribute *find_attribute(std::span<const RouteAttribute> attributes,
                                     RouteAttributeId id) noexcept {
  for (const auto &attribute : attributes)
    if (attribute.id == id)
      return &attribute;
  return nullptr;
}

bool field_matches(std::string_view required,
                   std::string_view actual) noexcept {
  return required.empty() || (!actual.empty() && required == actual);
}

bool format_matches(StorageFormatId required,
                    const TensorDesc &actual) noexcept {
  return !required.valid() || actual.storage_format().id() == required;
}

} // namespace

RouteMatchResult match_route_rule(const RouteRule &rule,
                                  const RouteMatchInput &input) noexcept {
  if (!valid_rule(rule))
    return mismatch(RouteMismatch::kInvalidRule);
  if (!valid_facts(input))
    return mismatch(RouteMismatch::kInvalidFacts);

  if (rule.platform != RoutePlatform::kUnknown &&
      rule.platform != input.platform)
    return mismatch(RouteMismatch::kPlatform);
  if (rule.route != NativeRoute::kUnknown && rule.route != input.route)
    return mismatch(RouteMismatch::kRoute);
  if (rule.provider.valid() && rule.provider != input.provider.id)
    return mismatch(RouteMismatch::kProvider);
  if (!rule.provider_version.empty() &&
      rule.provider_version != input.provider.version)
    return mismatch(RouteMismatch::kProviderVersion);

  for (const auto &predicate : rule.attributes) {
    const auto *attribute = find_attribute(input.attributes, predicate.id);
    if (attribute == nullptr)
      return mismatch(RouteMismatch::kAttributeMissing, predicate.id);
    if ((predicate.kind == RoutePredicateKind::kTextExact &&
         attribute->kind != RouteAttributeKind::kText) ||
        (predicate.kind != RoutePredicateKind::kTextExact &&
         attribute->kind != RouteAttributeKind::kUnsigned))
      return mismatch(RouteMismatch::kAttributeType, predicate.id);
    const bool matches =
        predicate.kind == RoutePredicateKind::kTextExact
            ? attribute->text == predicate.text
        : predicate.kind == RoutePredicateKind::kUnsignedExact
            ? attribute->unsigned_value == predicate.minimum
            : attribute->unsigned_value >= predicate.minimum &&
                  attribute->unsigned_value <= predicate.maximum;
    if (!matches)
      return mismatch(RouteMismatch::kAttributeValue, predicate.id);
  }

  if (!field_matches(rule.model.family, input.model.family) ||
      !field_matches(rule.model.name, input.model.name) ||
      !field_matches(rule.model.revision, input.model.revision) ||
      !field_matches(rule.model.configuration, input.model.configuration))
    return mismatch(RouteMismatch::kModel);
  if (rule.phase != WorkloadPhase::kUnspecified && rule.phase != input.phase)
    return mismatch(RouteMismatch::kPhase);
  if (!rule.region.empty() && rule.region != input.region)
    return mismatch(RouteMismatch::kRegion);

  if (rule.operation != OperationKind::kUnknown &&
      rule.operation != input.operation.kind())
    return mismatch(RouteMismatch::kOperation);
  const auto check_type = [&](DataType required, std::size_t index,
                              std::span<const TensorDesc> values) noexcept {
    return required == DataType::kUnknown ||
           (index < values.size() && values[index].valid() &&
            values[index].data_type() == required);
  };
  if (!check_type(rule.input0_type, 0, input.inputs) ||
      !check_type(rule.input1_type, 1, input.inputs) ||
      !check_type(rule.output0_type, 0, input.outputs))
    return mismatch(RouteMismatch::kDataType);
  if ((rule.input0_format.valid() &&
       (input.inputs.empty() || !input.inputs[0].valid() ||
        !format_matches(rule.input0_format, input.inputs[0]))) ||
      (rule.input1_format.valid() &&
       (input.inputs.size() < 2 || !input.inputs[1].valid() ||
        !format_matches(rule.input1_format, input.inputs[1]))))
    return mismatch(RouteMismatch::kStorageFormat);

  const auto *quantized = input.operation.get_if<QuantizedMatMul>();
  if (rule.quantization.presence == RouteQuantizationPresence::kNone &&
      quantized != nullptr)
    return mismatch(RouteMismatch::kQuantization);
  if (rule.quantization.presence ==
      RouteQuantizationPresence::kQuantizedMatMul) {
    if (quantized == nullptr)
      return mismatch(RouteMismatch::kQuantization);
    if (rule.quantization.mode && quantized->mode != *rule.quantization.mode)
      return mismatch(RouteMismatch::kQuantizationMode);
    if (rule.quantization.group_size != 0 &&
        quantized->group_size != rule.quantization.group_size)
      return mismatch(RouteMismatch::kQuantizationGroup);
    if ((rule.quantization.weight_format.valid() &&
         (input.inputs.size() < 2 || !input.inputs[1].valid() ||
          !format_matches(rule.quantization.weight_format, input.inputs[1]))) ||
        (rule.quantization.scale_format.valid() &&
         (input.inputs.size() < 3 || !input.inputs[2].valid() ||
          !format_matches(rule.quantization.scale_format, input.inputs[2]))))
      return mismatch(RouteMismatch::kStorageFormat);
  }
  return {};
}

} // namespace hyle::compute
