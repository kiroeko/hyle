#pragma once

#include <array>
#include <cstddef>
#include <span>

#include <cuda.h>

#include "device_status_kernel.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/detail/queue_sequence.h"
#include "hyle/compute/route_policy.h"

namespace hyle::compute::cuda::detail {

// Same-build Provider bridge. The caller must first establish that the
// ProviderVTable backend_state belongs to this CUDA backend. Compute keeps
// that state alive through every provider artifact and invocation.
HYLE_COMPUTE_BACKEND_CUDA_API CUcontext
borrow_primary_context(void *backend_state) noexcept;

// Borrowed immutable cold facts, with the same established-backend and
// lifetime preconditions as borrow_primary_context. Null state returns null.
HYLE_COMPUTE_BACKEND_CUDA_API const Environment *
borrow_environment(void *backend_state) noexcept;

// One cold projection of the immutable backend snapshot. Providers borrow the
// returned strings only while the Environment lives and keep no matcher state
// in prepared artifacts or submissions.
struct RouteFacts final {
  std::array<RouteFact, 8> storage;
  std::size_t count = 7;

  std::span<const RouteFact> environment() const noexcept {
    return {storage.data(), count};
  }
};

inline RouteFacts route_facts(const Environment &environment) noexcept {
  RouteFacts facts{{
      RouteFact::string(route_fact::kDeviceName,
                        environment.device_name.data()),
      RouteFact::number(route_fact::kArchitectureMajor,
                        static_cast<std::uint64_t>(environment.sm_major)),
      RouteFact::number(route_fact::kArchitectureMinor,
                        static_cast<std::uint64_t>(environment.sm_minor)),
      RouteFact::number(
          route_fact::kApiVersion,
          static_cast<std::uint64_t>(environment.driver_cuda_version)),
      RouteFact::number(
          route_fact::kRuntimeVersion,
          static_cast<std::uint64_t>(environment.runtime_version)),
      RouteFact::number(
          route_fact::kToolkitVersion,
          static_cast<std::uint64_t>(environment.build_toolkit_version)),
      RouteFact::string(route_fact::kDriverName, "nvidia"),
  }};
  if (environment.driver_release[0] != '\0') {
    facts.storage[facts.count++] = RouteFact::string(
        route_fact::kDriverVersion, environment.driver_release.data());
  }
  return facts;
}

// Inside one already-admitted QueueSequence only. Ordinary first use enqueues
// a reset; propagate its EnqueueResult even if a later command fails. A capture
// gets its own stable address, never the recording sequence's completion slot,
// and records no reset: each replay first copies its admitted outer status into
// that word, then merges the captured status back before original retirement.
// Kernels atomically retain the first nonzero DeviceStatusCode and must not
// access invalid payload. Compute/Graph's real Completion reports the code.
HYLE_COMPUTE_BACKEND_CUDA_API compute::detail::EnqueueResult
acquire_device_status(compute::detail::QueueSequence &sequence, unsigned **device_word) noexcept;

// Exceptional native work not yet joined to the admitted Queue's root stream.
// Its callback reports actual retirement separately from observation errors;
// false `wait` must remain nonblocking. State is borrowed from the provider
// invocation retained by this sequence's original Completion, including after
// both public owners are abandoned. No callback may submit further work.
struct NativeRetirement final {
  void *state = nullptr;
  Status (*observe)(void *state, bool wait, bool *retired) noexcept = nullptr;
};

// Same-build original-Completion seam for an admitted CUDA sequence.
// Ordinary failure attaches only one probe and then stops. Cold
// recording may compose probes by saving the returned previous probe in the
// same retained invocation; it must not register that invocation twice. Its
// observer first proves outer capture ended/aborted before native cleanup.
// The original Completion retains every execution/resource owner until both
// its root fence and the complete native prefix actually retire.
HYLE_COMPUTE_BACKEND_CUDA_API NativeRetirement defer_native_retirement(
    compute::detail::QueueSequence &sequence, NativeRetirement native) noexcept;

// Preserve the caller's current driver context across native SDK work. This
// is the backend's existing scope, shared by native providers without giving
// them a second context owner or access to the backend's mutable state.
class ScopedPrimaryContext final {
public:
  HYLE_COMPUTE_BACKEND_CUDA_API
  explicit ScopedPrimaryContext(CUcontext primary) noexcept;
  ScopedPrimaryContext(const ScopedPrimaryContext &) = delete;
  ScopedPrimaryContext &operator=(const ScopedPrimaryContext &) = delete;
  HYLE_COMPUTE_BACKEND_CUDA_API ~ScopedPrimaryContext();

  const Status &status() const noexcept { return status_; }
  explicit operator bool() const noexcept { return status_.ok(); }

private:
  CUcontext expected_ = nullptr;
  bool pushed_ = false;
  Status status_{};
};

} // namespace hyle::compute::cuda::detail
