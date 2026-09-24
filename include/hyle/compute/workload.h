#pragma once

#include <cstdint>
#include <string_view>

#include "hyle/compute/visibility.h"
#include "hyle/error/result.h"
#include "hyle/mem/allocator.h"

namespace hyle::compute {
namespace detail {
struct ModelArtifactProfileState;
}

// Consumer-verified identity of one immutable model artifact. HYLE does not
// open the repository or manifest and this value never claims a uniform tensor
// dtype. Digests are canonical lowercase SHA-256 hex. A nonempty description
// is exact and complete; selectors may independently constrain a subset.
struct ModelArtifactProfileDescription final {
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
  constexpr bool complete() const noexcept {
    return !model_publisher.empty() && !model_repository.empty() &&
           !model_family.empty() && !model_name.empty() &&
           !model_revision.empty() && !configuration_digest.empty() &&
           !artifact_repository.empty() && !artifact_revision.empty() &&
           !variant.empty() && !manifest_digest.empty() &&
           !format_family.empty();
  }
  friend constexpr bool
  operator==(const ModelArtifactProfileDescription &,
             const ModelArtifactProfileDescription &) noexcept = default;
};

// Optional immutable owner. create() copies all fields into one cold
// allocation; input views may expire immediately. Copies share the allocation
// and support concurrent reads. Mutating one handle remains exclusive. The
// allocator backing must outlive the final retained handle/artifact.
class ModelArtifactProfile final {
public:
  ModelArtifactProfile() noexcept = default;
  HYLE_COMPUTE_API
  ModelArtifactProfile(const ModelArtifactProfile &other) noexcept;
  HYLE_COMPUTE_API ModelArtifactProfile &
  operator=(const ModelArtifactProfile &other) noexcept;
  HYLE_COMPUTE_API ModelArtifactProfile(ModelArtifactProfile &&other) noexcept;
  HYLE_COMPUTE_API ModelArtifactProfile &
  operator=(ModelArtifactProfile &&other) noexcept;
  HYLE_COMPUTE_API ~ModelArtifactProfile();

  HYLE_COMPUTE_API static Result<ModelArtifactProfile>
  create(ModelArtifactProfileDescription description,
         mem::AllocatorRef allocator = mem::system_allocator()) noexcept;

  bool empty() const noexcept { return state_ == nullptr; }
  HYLE_COMPUTE_API ModelArtifactProfileDescription description() const noexcept;
  HYLE_COMPUTE_API bool
  matches(ModelArtifactProfileDescription expected) const noexcept;
  friend HYLE_COMPUTE_API bool
  operator==(const ModelArtifactProfile &left,
             const ModelArtifactProfile &right) noexcept;

private:
  explicit ModelArtifactProfile(
      detail::ModelArtifactProfileState *state) noexcept
      : state_(state) {}
  detail::ModelArtifactProfileState *state_ = nullptr;
};

enum class WorkloadPhase : std::uint8_t {
  kUnspecified = 0,
  kPrefill,
  kDecode,
};

// Optional semantic identity supplied by the consumer of a complete fragment.
// Version is part of the identity. This never substitutes for inspecting the
// actual typed operation/DAG, state, effects or resources.
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

// Cold preparation facts. Prepare a new artifact to change them. No field is
// consulted during ordinary submission. Equality is one qualification/cache
// dimension, not proof that a Provider can execute the actual operation/DAG.
struct WorkloadContext final {
  ModelArtifactProfile profile;
  WorkloadPhase phase = WorkloadPhase::kUnspecified;
  RegionSemanticId region;

  bool valid() const noexcept {
    return (phase == WorkloadPhase::kUnspecified ||
            phase == WorkloadPhase::kPrefill ||
            phase == WorkloadPhase::kDecode) &&
           (region.empty() || region.valid());
  }
  friend bool operator==(const WorkloadContext &,
                         const WorkloadContext &) noexcept = default;
};

} // namespace hyle::compute
