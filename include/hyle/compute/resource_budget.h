#pragma once

#include <cstddef>
#include <array>
#include <optional>
#include <span>

#include "hyle/compute/visibility.h"
#include "hyle/error/result.h"
#include "hyle/mem/quota_allocator.h"
#include "hyle/mem/resource.h"

namespace hyle::compute {
namespace detail { struct ResourceBudgetState; }

struct ResourceBudgetLimit final {
  mem::MemoryResourceRef resource{};
  std::size_t max_bytes = SIZE_MAX;
};

struct ResourceBudgetOptions final {
  std::span<const ResourceBudgetLimit> limits{};
  mem::MemoryResourceRef metadata_resource = mem::system_resource();
  mem::AllocatorRef allocator = mem::system_allocator();
  // Explicit partial accounting. Opaque Provider bounds stay separately
  // visible and never prove that a particular device's total fits its limit.
  bool allow_unattributed = false;
};

enum class ResourceBudgetCategory : unsigned char {
  kMetadata = 0, kConstant, kWorkspace, kMigration, kExternal, kCount
};
inline constexpr std::size_t kResourceBudgetCategoryCount =
    static_cast<std::size_t>(ResourceBudgetCategory::kCount);

struct ResourceBudgetUsage final {
  mem::MemoryProperties properties{};
  std::size_t live_bytes = 0, peak_bytes = 0, limit_bytes = SIZE_MAX;
  std::array<std::size_t, kResourceBudgetCategoryCount> live_by_category{};
};

struct UnattributedStorageUsage final {
  std::size_t private_upper_bound_bytes = 0;
  std::size_t unknown_private_owners = 0;
  std::size_t control_bytes = 0;
};

// A local admission scope for explicitly managed requested bytes. It reserves
// no physical device memory and controls neither drivers nor other processes.
// Native resource owners remain borrowed. Buffers and raw metadata allocations
// are charged once, not again for every alias/view, and retain this scope.
// Quota admission is thread-safe; usage counters are independent snapshots.
class ResourceBudget final {
public:
  class Reservation final {
  public:
    Reservation() noexcept = default;
    Reservation(const Reservation &) = delete;
    Reservation &operator=(const Reservation &) = delete;
    HYLE_COMPUTE_API Reservation(Reservation &&other) noexcept;
    HYLE_COMPUTE_API Reservation &operator=(Reservation &&other) noexcept;
    HYLE_COMPUTE_API ~Reservation();
    HYLE_COMPUTE_API void reset() noexcept;
  private:
    friend class ResourceBudget;
    Reservation(detail::ResourceBudgetState *state,
                mem::QuotaAllocator::Reservation charge, bool unknown,
                std::size_t row, ResourceBudgetCategory category) noexcept;
    detail::ResourceBudgetState *state_ = nullptr;
    mem::QuotaAllocator::Reservation charge_{};
    bool unknown_ = false;
    std::size_t row_ = SIZE_MAX;
    ResourceBudgetCategory category_ = ResourceBudgetCategory::kExternal;
  };

  ResourceBudget() noexcept = default;
  HYLE_COMPUTE_API ResourceBudget(const ResourceBudget &other) noexcept;
  HYLE_COMPUTE_API ResourceBudget &operator=(const ResourceBudget &other) noexcept;
  HYLE_COMPUTE_API ResourceBudget(ResourceBudget &&other) noexcept;
  HYLE_COMPUTE_API ResourceBudget &operator=(ResourceBudget &&other) noexcept;
  HYLE_COMPUTE_API ~ResourceBudget();
  HYLE_COMPUTE_API static Result<ResourceBudget>
  create(ResourceBudgetOptions options) noexcept;
  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  HYLE_COMPUTE_API std::size_t resource_count() const noexcept;
  HYLE_COMPUTE_API Result<mem::MemoryResourceRef> resource(std::size_t index) const noexcept;
  HYLE_COMPUTE_API Result<ResourceBudgetUsage>
  usage(mem::MemoryResourceRef resource) const noexcept;
  HYLE_COMPUTE_API UnattributedStorageUsage unattributed_usage() const noexcept;
  HYLE_COMPUTE_API bool allows_unattributed() const noexcept;
  HYLE_COMPUTE_API Result<Reservation>
  reserve(mem::MemoryResourceRef resource, std::size_t bytes,
          ResourceBudgetCategory category = ResourceBudgetCategory::kExternal) const noexcept;
  HYLE_COMPUTE_API Result<Reservation>
  reserve_unattributed(std::optional<std::size_t> upper_bound) const noexcept;
  // Sized-free host metadata allocator. Each successful allocation retains
  // the scope; the returned ref itself is borrowed, like every AllocatorRef.
  HYLE_COMPUTE_API Result<mem::AllocatorRef> metadata_allocator() const noexcept;
private:
  explicit ResourceBudget(detail::ResourceBudgetState *state) noexcept : state_(state) {}
  detail::ResourceBudgetState *state_ = nullptr;
};

} // namespace hyle::compute
