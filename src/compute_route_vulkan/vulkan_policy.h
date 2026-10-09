#pragma once

#include "vulkan_program_internal.h"

namespace hyle::compute::vulkan::detail {

enum class DefaultCarrier : std::uint8_t { kGgml, kIree };

// These are mature engineering defaults, not unmeasured hardware winners.
// The original typed/native feature and resource checks remain authoritative.
inline constexpr RouteSelector kGgmlDefaultSelector{
    .route = NativeRoute::kVulkan, .provider = kGgmlCarrierId};
inline constexpr RouteSelector kIreeDefaultSelector{
    .route = NativeRoute::kVulkan, .provider = kIreeCarrierId};

inline bool matches_cold_rule(const ProgramState &program, const PlanStep &step,
                              DefaultCarrier carrier,
                              const RouteSelector &selector) noexcept {
  std::array<TensorDesc, 8> inputs;
  std::array<TensorDesc, 2> outputs;
  if (step.inputs.size() > inputs.size() || step.outputs.size() > outputs.size())
    return false;
  for (std::size_t i = 0; i < step.inputs.size(); ++i)
    inputs[i] = program.values[step.inputs[i]].description;
  for (std::size_t i = 0; i < step.outputs.size(); ++i)
    outputs[i] = program.values[step.outputs[i]].description;
  const auto &environment = program.runtime->device->environment;
  const auto facts = route_facts(environment);
  const ProviderIdentity identity = carrier == DefaultCarrier::kGgml
      ? ProviderIdentity{kGgmlCarrierId, "ggml-vulkan", environment.ggml_revision}
      : ProviderIdentity{kIreeCarrierId, "iree-vulkan", environment.iree_revision};
  const RouteMatchInput input{
      .platform = kBuildRoutePlatform,
      .route = NativeRoute::kVulkan,
      .provider = identity,
      .environment = facts.values(),
      .profile = program.workload.profile.description(),
      .phase = program.workload.phase,
      .region = program.workload.region,
      .operation = step.operation,
      .inputs = std::span(inputs).first(step.inputs.size()),
      .outputs = std::span(outputs).first(step.outputs.size())};
  return match_route_selector(selector, input).matched();
}

inline bool matches_mature_default(const ProgramState &program,
                                    const PlanStep &step,
                                    DefaultCarrier carrier) noexcept {
  return matches_cold_rule(program, step, carrier,
      carrier == DefaultCarrier::kGgml ? kGgmlDefaultSelector : kIreeDefaultSelector);
}

} // namespace hyle::compute::vulkan::detail
