#include "hyle/compute/route_profile.h"

#include <algorithm>
#include <cmath>

namespace hyle::compute {
namespace {

bool complete(const RouteProfileIdentity &key) noexcept {
  if (!key.hyle_build.valid() || !key.semantics.valid() ||
      !key.shape_domain.valid() || !key.route.valid() ||
      !key.workload_data.valid() || key.targets.empty() ||
      !key.workload.valid() ||
      (key.completion != ProfileCompletionBoundary::kOperationTerminal &&
       key.completion != ProfileCompletionBoundary::kGraphTerminal))
    return false;
  for (const auto &target : key.targets)
    if (!target.backend.valid() || !target.build.valid() ||
        !target.hardware.valid() || !target.runtime.valid() ||
        !target.configuration.valid())
      return false;
  for (const auto &provider : key.providers)
    if (!provider.provider.valid() || !provider.build.valid() ||
        !provider.configuration.valid() || !provider.implementation.valid())
      return false;
  for (const auto &input : key.inputs)
    if (!input.valid())
      return false;
  for (const auto &output : key.outputs)
    if (!output.valid())
      return false;
  return true;
}

template <class T>
bool equal(std::span<const T> left, std::span<const T> right) noexcept {
  return std::equal(left.begin(), left.end(), right.begin(), right.end());
}

} // namespace

bool ProfileTimingInterval::valid() const noexcept {
  return std::isfinite(lower_ns) && std::isfinite(upper_ns) && lower_ns >= 0 &&
         upper_ns >= lower_ns;
}

RouteProfileMatch match_route_profile(const RouteProfileIdentity &request,
                                      const RouteProfile &profile) noexcept {
  const auto &key = profile.identity;
  if (!complete(request) || !complete(key))
    return {.reason = ProfileMatchReason::kIncompleteIdentity};
  if (request.hyle_build != key.hyle_build)
    return {.reason = ProfileMatchReason::kHyleBuildMismatch};
  if (!equal(request.targets, key.targets))
    return {.reason = ProfileMatchReason::kTargetMismatch};
  if (!equal(request.providers, key.providers))
    return {.reason = ProfileMatchReason::kProviderMismatch};
  if (request.semantics != key.semantics)
    return {.reason = ProfileMatchReason::kSemanticsMismatch};
  if (request.shape_domain != key.shape_domain)
    return {.reason = ProfileMatchReason::kShapeDomainMismatch};
  if (!equal(request.inputs, key.inputs) ||
      !equal(request.outputs, key.outputs))
    return {.reason = ProfileMatchReason::kTensorMismatch};
  if (request.route != key.route)
    return {.reason = ProfileMatchReason::kRouteMismatch};
  if (request.workload.objective != key.workload.objective ||
      request.workload.concurrent_invocations !=
          key.workload.concurrent_invocations ||
      request.workload.require_determinism !=
          key.workload.require_determinism ||
      request.workload_data != key.workload_data)
    return {.reason = ProfileMatchReason::kWorkloadMismatch};
  if (request.completion != key.completion)
    return {.reason = ProfileMatchReason::kCompletionMismatch};
  if ((profile.replay && !profile.replay->valid()) ||
      (profile.compilation && !profile.compilation->valid()))
    return {.reason = ProfileMatchReason::kInvalidTiming};
  if (!profile.replay ||
      (request.workload.expected_replays && !profile.compilation))
    return {.reason = ProfileMatchReason::kMissingTiming};
  auto cost = *profile.replay;
  if (request.workload.expected_replays) {
    const double count =
        static_cast<double>(*request.workload.expected_replays);
    cost.lower_ns += profile.compilation->lower_ns / count;
    cost.upper_ns += profile.compilation->upper_ns / count;
    if (!cost.valid())
      return {.reason = ProfileMatchReason::kInvalidTiming};
  }
  return {.reason = ProfileMatchReason::kMatched, .cost = cost};
}

RouteProfileLookup
find_route_profile(const RouteProfileIdentity &request,
                   std::span<const RouteProfile> profiles) noexcept {
  RouteProfileLookup result;
  for (const auto &profile : profiles) {
    const auto match = match_route_profile(request, profile);
    if (!match.cost) {
      ++result.rejected_entries;
      continue;
    }
    ++result.matched_entries;
    if (!result.cost) {
      result.cost = match.cost;
    } else {
      result.cost->lower_ns =
          (std::min)(result.cost->lower_ns, match.cost->lower_ns);
      result.cost->upper_ns =
          (std::max)(result.cost->upper_ns, match.cost->upper_ns);
    }
  }
  return result;
}

} // namespace hyle::compute
