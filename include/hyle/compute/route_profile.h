#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

#include "hyle/compute/device.h"
#include "hyle/compute/tensor_desc.h"

namespace hyle::compute {

// Stable content/configuration identity supplied by its owner, not generated
// from pointers, display names, ProviderInstanceId or object padding. All-zero
// means unknown. A nonzero value is an assertion of identity, NOT proof that
// an experiment or its instrumentation was qualified. Hash collisions and
// changes to any represented fact are the producing owner's responsibility.
struct ProfileFingerprint final {
  std::array<std::uint64_t, 4> words{};

  constexpr bool valid() const noexcept {
    return words[0] != 0 || words[1] != 0 || words[2] != 0 || words[3] != 0;
  }
  friend constexpr bool
  operator==(const ProfileFingerprint &,
             const ProfileFingerprint &) noexcept = default;
};

enum class WorkloadObjective : std::uint8_t {
  kSteadyLatency = 0,
  kSteadyThroughput,
};

struct WorkloadIntent final {
  WorkloadObjective objective = WorkloadObjective::kSteadyLatency;
  std::size_t concurrent_invocations = 1;
  bool require_determinism = false;
  // Absent means long-lived: do not add compile time to replay time. An
  // engaged value must be positive and explicitly requests amortization.
  // This is a query policy, not part of measured workload identity: changing
  // the expected lifetime can reuse the same qualified cold/replay intervals.
  std::optional<std::uint64_t> expected_replays{};

  constexpr bool valid() const noexcept {
    return (objective == WorkloadObjective::kSteadyLatency ||
            objective == WorkloadObjective::kSteadyThroughput) &&
           concurrent_invocations != 0 &&
           (!expected_replays || *expected_replays != 0);
  }
  friend constexpr bool operator==(const WorkloadIntent &,
                                   const WorkloadIntent &) noexcept = default;
};

// Graph selection consumes only the complete Graph terminal boundary. Direct
// operation profiles remain a distinct namespace, never a cheaper substitute
// for the same kernel inside a Graph. Enqueue-only/device-event-only timings
// cannot be represented as terminal measurements.
enum class ProfileCompletionBoundary : std::uint8_t {
  kUnknown = 0,
  kOperationTerminal,
  kGraphTerminal,
};

struct ProfileTargetIdentity final {
  BackendId backend{};
  ProfileFingerprint build{};
  ProfileFingerprint hardware{};
  ProfileFingerprint runtime{}; // Driver/runtime/OS versions relevant to work.
  ProfileFingerprint configuration{}; // ISA, executor, placement, power policy.
  friend constexpr bool
  operator==(const ProfileTargetIdentity &,
             const ProfileTargetIdentity &) noexcept = default;
};

struct ProfileProviderIdentity final {
  ProviderId provider{};
  ProfileFingerprint build{};
  ProfileFingerprint configuration{};
  ProviderRouteKey implementation{};
  friend constexpr bool
  operator==(const ProfileProviderIdentity &,
             const ProfileProviderIdentity &) noexcept = default;
};

// Immutable borrowed key for ONE complete candidate route at a common HYLE
// boundary. Ordered targets/providers/tensors are canonical coordinates, not
// sets. The caller keeps these spans unchanged through a matching call.
//
// Fingerprints cover full facts, not informal labels: semantics includes
// operation/schema versions, instance attributes, graph edges, precision and
// effects; shape_domain includes bounds/guards; route includes node cover,
// target/storage/lane assignment, resource classes, transformations and replay
// mode; workload_data includes representative content/version/distribution and
// the qualified measurement contract. Custom schemas without a stable instance
// identity cannot supply a qualified persistent profile.
struct RouteProfileIdentity final {
  ProfileFingerprint hyle_build{};
  ProfileFingerprint semantics{};
  ProfileFingerprint shape_domain{};
  ProfileFingerprint route{};
  ProfileFingerprint workload_data{};
  std::span<const ProfileTargetIdentity> targets{};
  std::span<const ProfileProviderIdentity> providers{};
  std::span<const TensorDesc> inputs{};
  std::span<const TensorDesc> outputs{};
  WorkloadIntent workload{};
  ProfileCompletionBoundary completion = ProfileCompletionBoundary::kUnknown;
};

// A qualified uncertainty envelope, not a single best observation. Values are
// normalized ns per completed Invocation at the stated concurrency: latency
// for kSteadyLatency, inverse sustained throughput for kSteadyThroughput.
// Zero is an exact known zero; absence of an optional interval is unknown.
struct ProfileTimingInterval final {
  double lower_ns = 0;
  double upper_ns = 0;
  HYLE_COMPUTE_API bool valid() const noexcept;
  friend constexpr bool operator==(ProfileTimingInterval,
                                   ProfileTimingInterval) noexcept = default;
};

struct RouteProfile final {
  RouteProfileIdentity identity{};
  std::optional<ProfileTimingInterval> replay{};
  // Entire cold route construction, including packing and materialization,
  // not one Provider's private preparation call.
  std::optional<ProfileTimingInterval> compilation{};
};

enum class ProfileMatchReason : std::uint8_t {
  kMatched = 0,
  kIncompleteIdentity,
  kHyleBuildMismatch,
  kTargetMismatch,
  kProviderMismatch,
  kSemanticsMismatch,
  kShapeDomainMismatch,
  kTensorMismatch,
  kRouteMismatch,
  kWorkloadMismatch,
  kCompletionMismatch,
  kMissingTiming,
  kInvalidTiming,
};

struct RouteProfileMatch final {
  ProfileMatchReason reason = ProfileMatchReason::kIncompleteIdentity;
  // Qualified replay or explicitly amortized complete-route cost. Never a
  // sum of independently measured kernels/regions/copies.
  std::optional<ProfileTimingInterval> cost{};
};

// No allocation, callbacks, device queries or global state. Compares every
// required identity component, then validates/normalizes the timing envelope.
HYLE_COMPUTE_API RouteProfileMatch match_route_profile(
    const RouteProfileIdentity &request, const RouteProfile &profile) noexcept;

struct RouteProfileLookup final {
  std::optional<ProfileTimingInterval> cost{};
  std::size_t matched_entries = 0;
  std::size_t rejected_entries = 0;
};

// Finite linear read-only scan of the caller's span. Matching duplicate
// observations are conservatively enveloped, NOT minimized or averaged. No
// match leaves cost unknown. The caller can inspect individual mismatch
// reasons with match_route_profile(); no reference to this span is retained.
HYLE_COMPUTE_API RouteProfileLookup
find_route_profile(const RouteProfileIdentity &request,
                   std::span<const RouteProfile> profiles) noexcept;

} // namespace hyle::compute
