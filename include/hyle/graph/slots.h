#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "hyle/compute/lease_slots.h"
#include "hyle/compute/operation.h"
#include "hyle/graph/executable.h"
#include "hyle/graph/visibility.h"

namespace hyle::graph::slots {

// Mathematical item geometry only. Neither the pool identity nor its ready or
// mutable permissions belong to an operation descriptor. An item is flattened
// to one plain row; padding between slots is not part of that logical row.
class Schema final {
public:
  Schema() noexcept = default;
  HYLE_GRAPH_API static Result<Schema> create(compute::TensorDesc item,
      std::uint32_t slots, std::size_t stride_bytes) noexcept;
  bool valid() const noexcept { return type_ != compute::DataType::kUnknown; }
  compute::DataType type() const noexcept { return type_; }
  std::int64_t item_elements() const noexcept { return item_elements_; }
  std::uint32_t slot_count() const noexcept { return slots_; }
  std::size_t stride_bytes() const noexcept { return stride_; }
  // [slot_count,item_elements], with the original leading byte stride. The
  // final extent stops at the last item's end, excluding its trailing padding.
  // This is a description, never a whole-pool initialized-storage permission.
  HYLE_GRAPH_API Result<compute::TensorDesc> capacity_description() const noexcept;
  friend bool operator==(const Schema &, const Schema &) noexcept = default;

private:
  compute::DataType type_ = compute::DataType::kUnknown;
  std::int64_t item_elements_ = 0;
  std::uint32_t slots_ = 0;
  std::size_t stride_ = 0;
};

// Exact same-type store of rows[M,E] to external slots selected by I32[M].
// Sources and destinations are admitted separately by their real leases; this
// descriptor carries no pool, executor or write authority. Destination slots
// must be distinct. External-effect ordering applies; a later cache reader
// additionally needs its explicit original Graph dependency on this node.
struct Store final {
  Schema schema;
  static constexpr compute::OperationId operation_id{
      0x48594c454752534cULL, 0x4f5453544f524501ULL};
  static constexpr std::string_view operation_name = "slot-store";
  constexpr compute::OperationTraits operation_traits() const noexcept {
    return {{2, 2}, 0, compute::AliasPolicy::kNone,
        compute::EffectKind::kExternalSideEffect,
        compute::PrecisionPolicy::kExact, compute::DataType::kUnknown};
  }
  HYLE_GRAPH_API Status infer_output_descriptions(
      std::span<const compute::TensorDesc>,
      std::span<compute::TensorDesc>) const noexcept;
};
static_assert(compute::detail::InlineCustomOperationDescriptor<Store>);

struct SlotBindingsOptions final {
  mem::AllocatorRef allocator = mem::system_allocator();
  compute::ResourceBudget resource_budget{};
};
namespace detail { struct SlotBindingsData; }

// Cold-owned resource transaction, not an execution or publication mechanism.
// It retains the original ready sources and exclusive destination blocks plus
// their validated coordinates. No payload or capacity-sized metadata is made.
// Creation rejects without consuming any caller writer. Resource/Context and
// allocator lifetimes retain the original lease and AllocatorRef contracts.
class SlotBindings final {
public:
  SlotBindings() noexcept = default;
  SlotBindings(const SlotBindings &) = delete;
  SlotBindings &operator=(const SlotBindings &) = delete;
  HYLE_GRAPH_API SlotBindings(SlotBindings &&) noexcept;
  HYLE_GRAPH_API SlotBindings &operator=(SlotBindings &&) noexcept;
  HYLE_GRAPH_API ~SlotBindings();
  HYLE_GRAPH_API static Result<SlotBindings> create(compute::LeaseSlotLayout,
      std::span<const compute::ReadLease> sources,
      std::span<compute::WriteLease> destinations,
      SlotBindingsOptions = {}) noexcept;

  bool valid() const noexcept { return data_ != nullptr; }
  HYLE_GRAPH_API compute::LeaseSlotLayout layout() const noexcept;
  HYLE_GRAPH_API std::size_t source_count() const noexcept;
  HYLE_GRAPH_API std::size_t destination_count() const noexcept;
  // Borrow only while this original transaction is cold and caller-owned.
  // Moving/resetting a lease invalidates its saved address fact; an adapter
  // must revalidate real handles before native admission. Coordinates alone
  // never imply readiness, permission, or a successful native outcome.
  HYLE_GRAPH_API std::span<const compute::ReadLease> sources() const noexcept;
  HYLE_GRAPH_API std::span<compute::WriteLease> destinations() noexcept;
  HYLE_GRAPH_API std::span<const compute::WriteLease> destinations() const noexcept;
  HYLE_GRAPH_API std::span<const compute::LeaseSlotAddress>
  source_addresses() const noexcept;
  HYLE_GRAPH_API std::span<const compute::LeaseSlotAddress>
  destination_addresses() const noexcept;

private:
  explicit SlotBindings(detail::SlotBindingsData *data) noexcept : data_(data) {}
  detail::SlotBindingsData *data_ = nullptr;
};

struct PrepareOptions final {
  // CUDA device storage needs a same-Context pinned/managed host resource for
  // its queued metadata upload. Native sources qualify their own cold copy
  // route; host-accessible destination controls use direct flush/close.
  mem::MemoryResourceRef host_workspace = mem::system_resource();
  mem::AllocatorRef allocator = mem::system_allocator();
  compute::ResourceBudget resource_budget{};
};
// Prepare the original selected Store slots and their small, fixed I32
// controls. Ordinary CPU/CUDA route compilation supplies the fallback Store
// automatically. No tensor payload is copied and no second executor is made.
// Shared-Context lanes require their original QueueBinding. Native execution
// lanes have no Compute Queue and use an empty span (or only shared siblings).
HYLE_GRAPH_API Status prepare(Invocation &, std::span<const QueueBinding>,
    PrepareOptions = {}) noexcept;
// Logical resource association, independent of the selected providers. The
// checked axis-zero Gather reads this Store's original pool through private
// source IDs. The arithmetic between Gather and Store remains ordinary Graph
// math and may execute as separate operations or as a fused region.
struct SourceBinding final { NodeId store, gather; };
// Prepares one private [source IDs | destination IDs] carrier per associated
// Store. The original Graph establishes geometry and input identities; actual
// selected readers still prove their bounded access during binding. Duplicate
// Stores/readers/pools and shared or published control inputs are rejected.
HYLE_GRAPH_API Status prepare(Invocation &, const Graph &,
    std::span<const QueueBinding>, std::span<const SourceBinding>,
    PrepareOptions = {}) noexcept;
// Cold binding for the selected Store. Its private destination-ID input comes
// from actual writer coordinates. A qualified fused source reader also gets
// private source IDs and the capacity's exact original ready guards; an
// independent fallback Store uses a prepared SourceBinding for its reader;
// without that association ordinary source bindings remain caller-owned.
// Preparation/binding may wait for a metadata-only upload on the original
// execution source. Failure never consumes the caller's transaction. No executor or
// provider selection is created here.
HYLE_GRAPH_API Status bind(Invocation &, const Graph &, NodeId store,
    SlotBindings &&) noexcept;
// No implicit wait or publication: returns the original transaction only
// after the whole Graph's work/results/observations have actually retired,
// including failed or closed recovery. The caller observes execution success
// before publishing writers; successful take alone is not a native outcome.
HYLE_GRAPH_API Result<SlotBindings> take(Invocation &, NodeId store) noexcept;

} // namespace hyle::graph::slots
