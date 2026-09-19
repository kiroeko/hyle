#pragma once

#include <cstdint>
#include <string_view>

#include "hyle/compute/visibility.h"
#include "hyle/error/result.h"
#include "hyle/mem/allocator.h"

namespace hyle::compute {
namespace detail {
struct ModelIdentityState;
}

// Consumer-declared identity, not model contents or a request to load them.
// Configuration must identify every model option relevant to a rule (for
// example, the consumer's canonical configuration digest). Comparisons are
// byte-exact; a family name alone never qualifies a model-specific rule.
struct ModelDescription final {
  std::string_view family;
  std::string_view revision;
  std::string_view configuration;

  constexpr bool complete() const noexcept {
    return !family.empty() && !revision.empty() && !configuration.empty();
  }
  friend constexpr bool operator==(const ModelDescription &,
                                   const ModelDescription &) noexcept = default;
};

// Optional immutable owner. create() copies the three fields into one cold
// allocation; its input strings may expire immediately afterwards. Copies
// share that allocation and are safe to use on independent threads. Mutating
// one handle concurrently is not supported. The allocator backing must live
// until the last handle/artifact retaining this identity has been destroyed.
// Empty identity has no allocation, reference-count work or known-model claim.
class ModelIdentity final {
public:
  ModelIdentity() noexcept = default;
  HYLE_COMPUTE_API ModelIdentity(const ModelIdentity &other) noexcept;
  HYLE_COMPUTE_API ModelIdentity &
  operator=(const ModelIdentity &other) noexcept;
  HYLE_COMPUTE_API ModelIdentity(ModelIdentity &&other) noexcept;
  HYLE_COMPUTE_API ModelIdentity &operator=(ModelIdentity &&other) noexcept;
  HYLE_COMPUTE_API ~ModelIdentity();

  HYLE_COMPUTE_API static Result<ModelIdentity>
  create(ModelDescription description,
         mem::AllocatorRef allocator = mem::system_allocator()) noexcept;
  bool empty() const noexcept { return state_ == nullptr; }
  // Borrowed from this shared owner, never from create()'s input strings.
  HYLE_COMPUTE_API ModelDescription description() const noexcept;
  // A rule must supply all three exact fields. This only checks model identity;
  // operation/region semantics, hardware, precision and resources still apply.
  HYLE_COMPUTE_API bool matches(ModelDescription required) const noexcept;
  friend HYLE_COMPUTE_API bool operator==(const ModelIdentity &left,
                                         const ModelIdentity &right) noexcept;

private:
  explicit ModelIdentity(detail::ModelIdentityState *state) noexcept
      : state_(state) {}
  detail::ModelIdentityState *state_ = nullptr;
};

enum class WorkloadPhase : std::uint8_t {
  kUnspecified = 0,
  kPrefill,
  kDecode,
};

// Optional semantic identity supplied by the consumer of a complete fragment.
// Version is part of the identity. This is neither a Provider choice nor an
// alternative to inspecting the actual typed operation/DAG and state effects.
struct RegionSemanticId final {
  std::uint64_t namespace_id = 0;
  std::uint64_t region_id = 0;
  std::uint32_t version = 0;

  constexpr bool empty() const noexcept {
    return namespace_id == 0 && region_id == 0 && version == 0;
  }
  constexpr bool valid() const noexcept {
    return (namespace_id != 0 || region_id != 0) && version != 0;
  }
  friend constexpr bool operator==(RegionSemanticId,
                                   RegionSemanticId) noexcept = default;
};

// Cold preparation facts. Reuse an immutable ModelIdentity across stages;
// prepare a new artifact to change these facts. No field is consulted during
// ordinary submission. Equality is only the workload part of qualification,
// not a portable plan-cache key or proof of implementation eligibility.
struct WorkloadContext final {
  ModelIdentity model;
  WorkloadPhase phase = WorkloadPhase::kUnspecified;
  RegionSemanticId region;

  bool valid() const noexcept {
    return (phase == WorkloadPhase::kUnspecified ||
            phase == WorkloadPhase::kPrefill || phase == WorkloadPhase::kDecode) &&
           (region.empty() || region.valid());
  }
  friend bool operator==(const WorkloadContext &,
                         const WorkloadContext &) noexcept = default;
};

} // namespace hyle::compute
