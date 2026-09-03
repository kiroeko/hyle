#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "hyle/error/result.h"
#include "hyle/mem/buffer.h"
#include "hyle/mem/visibility.h"

namespace hyle::mem {

inline constexpr std::uint32_t kNoLifetimeAlias = UINT32_MAX;

enum class LifetimeFlag : std::uint8_t {
  kNone = 0,
  // The placement remains unavailable after its declared interval. Persistent
  // and output are named consumer facts with the same conservative reuse rule.
  kNoReuse = 1u << 0,
  kPersistent = 1u << 1,
  kOutput = 1u << 2,
};

constexpr LifetimeFlag operator|(LifetimeFlag left,
                                 LifetimeFlag right) noexcept {
  return static_cast<LifetimeFlag>(static_cast<std::uint8_t>(left) |
                                   static_cast<std::uint8_t>(right));
}
constexpr LifetimeFlag& operator|=(LifetimeFlag& left,
                                   LifetimeFlag right) noexcept {
  left = left | right;
  return left;
}
constexpr bool has_lifetime_flag(LifetimeFlag flags,
                                 LifetimeFlag flag) noexcept {
  return (static_cast<std::uint8_t>(flags) &
          static_cast<std::uint8_t>(flag)) != 0;
}

// Input IDs are their dense positions in the request span. resource_slot IDs
// must likewise be dense [0, resource_count). Intervals are half-open. Aliases
// share their root allocation at alias_offset; alias graphs may point in either
// direction but must be acyclic and stay in one resource.
struct LifetimeRequest {
  std::uint32_t resource_slot = 0;
  std::size_t size = 0;
  std::size_t alignment = alignof(std::max_align_t);
  std::uint64_t begin = 0;
  std::uint64_t end = 1;
  std::uint32_t alias_of = kNoLifetimeAlias;
  std::size_t alias_offset = 0;
  LifetimeFlag flags = LifetimeFlag::kNone;
};

struct MemoryPlacement {
  std::uint32_t resource_slot = 0;
  // Resource-local physical backing index. `offset` is relative to this
  // backing, not to a virtual concatenation of all resource allocations.
  std::uint32_t backing_index = 0;
  std::size_t offset = 0;
  std::size_t size = 0;
  std::size_t alignment = alignof(std::max_align_t);
};

struct LifetimeResourcePlan {
  // Sum of the exact planned physical backing sizes for this resource.
  std::size_t peak_bytes = 0;
  std::size_t backing_count = 0;
};

struct LifetimeBackingPlan {
  std::size_t size = 0;
  // Minimum alignment requested from the provider. An equivalent provider may
  // return a stronger alignment when the plan is instantiated.
  std::size_t alignment = alignof(std::max_align_t);
};

class PlannedBuffers;

// Immutable placement result. Queries are O(1); instances allocate one Buffer
// per planned physical backing. The plan owns no provider, Tensor or Graph.
class LifetimePlan {
 public:
  constexpr LifetimePlan() noexcept = default;
  LifetimePlan(const LifetimePlan&) = delete;
  LifetimePlan& operator=(const LifetimePlan&) = delete;
  HYLE_MEM_API LifetimePlan(LifetimePlan&& other) noexcept;
  HYLE_MEM_API LifetimePlan& operator=(LifetimePlan&& other) noexcept;
  HYLE_MEM_API ~LifetimePlan();

  bool valid() const noexcept { return impl_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  HYLE_MEM_API std::size_t request_count() const noexcept;
  HYLE_MEM_API std::size_t resource_count() const noexcept;
  HYLE_MEM_API Result<MemoryPlacement> placement(
      std::size_t request_id) const noexcept;
  HYLE_MEM_API Result<LifetimeResourcePlan> resource_plan(
      std::size_t resource_slot) const noexcept;
  HYLE_MEM_API Result<LifetimeBackingPlan> backing_plan(
      std::size_t resource_slot, std::size_t backing_index) const noexcept;
  HYLE_MEM_API Result<PlannedBuffers> instantiate(
      std::span<const MemoryResourceRef> resources) const noexcept;

 private:
  friend class LifetimePlanner;
  explicit LifetimePlan(void* impl) noexcept : impl_(impl) {}
  void* impl_ = nullptr;
};

// Owns the instantiated per-resource backing allocations and a compact copy of
// their placements, so it remains usable after the LifetimePlan is destroyed.
// Views remain bound across owner move and are invalidated by reset/destruction.
class PlannedBuffers {
 public:
  constexpr PlannedBuffers() noexcept = default;
  PlannedBuffers(const PlannedBuffers&) = delete;
  PlannedBuffers& operator=(const PlannedBuffers&) = delete;
  HYLE_MEM_API PlannedBuffers(PlannedBuffers&& other) noexcept;
  HYLE_MEM_API PlannedBuffers& operator=(PlannedBuffers&& other) noexcept;
  HYLE_MEM_API ~PlannedBuffers();

  bool valid() const noexcept { return impl_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  HYLE_MEM_API std::size_t request_count() const noexcept;
  HYLE_MEM_API std::size_t resource_count() const noexcept;
  HYLE_MEM_API Result<std::size_t> backing_count(
      std::size_t resource_slot) const noexcept;
  HYLE_MEM_API Result<BufferView> view(std::size_t request_id) noexcept;
  HYLE_MEM_API Result<ConstBufferView> view(
      std::size_t request_id) const noexcept;
  HYLE_MEM_API Result<BufferView> backing(
      std::size_t resource_slot, std::size_t backing_index) noexcept;
  HYLE_MEM_API Result<ConstBufferView> backing(
      std::size_t resource_slot, std::size_t backing_index) const noexcept;
  HYLE_MEM_API void reset() noexcept;

 private:
  friend class LifetimePlan;
  explicit PlannedBuffers(void* impl) noexcept : impl_(impl) {}
  void* impl_ = nullptr;
};

// Cold tensor-free planner. Resource properties are borrowed only for this
// call; the returned plan can later be instantiated on an equivalent provider.
// It uses deterministic best-fit reusable slots: active intervals are released
// with a min-heap and free capacities are found with an indexed segment tree.
// New slots are split at provider allocation limits. Build is O((N+E) log N),
// scratch is O(N), and no table is proportional to bytes/pages. E is the
// alias-edge count (<= N).
class LifetimePlanner {
 public:
  HYLE_MEM_API static Result<LifetimePlan> build(
      std::span<const LifetimeRequest> requests,
      std::span<const MemoryResourceRef> resources) noexcept;
};

static_assert(sizeof(LifetimePlan) == sizeof(void*));
static_assert(sizeof(PlannedBuffers) == sizeof(void*));

}  // namespace hyle::mem
