#pragma once

#include "hyle/graph/region_provider.h"
#include "hyle/graph/slots.h"

namespace hyle::graph::slots::detail {
// Exact selected-artifact cold bridge. This is not a native provider ABI or a
// hot dispatch table: ordinary submission uses the original prepared region.
inline constexpr compute::ProviderExtensionId kStoreBindingExtensionId{
    0x48594c45534c4f54ULL, 0x42494e44494e4701ULL, 1};

struct StoreSource final {
  NodeId gather{};
  ValueId capacity{}, indices{};
  std::uint32_t capacity_input_port = UINT32_MAX;
  std::size_t count = 0;
  bool permission_changes_preserve_commands = false;
};
struct StoreRegionFacts final {
  NodeId store{};
  Schema schema{};
  ValueId destination_indices{};
  std::uint32_t destination_input_port = UINT32_MAX;
  std::size_t destination_count = 0;
  mem::MemoryResourceRef resource{};
  StoreSource source{};
};
struct StoreConfiguration final {
  compute::ConstTensorView capacity{}, source_indices{}, destination_indices{};
};
// Borrowed only during externally serialized cold preparation or an exact
// submitted Source callback. Copy layout/coordinates or export independent
// tickets there; never retain transaction or a writer address. The capacity is an
// address carrier only. indices_guard covers just this Store's initialized
// destination-ID slice and retains no payload/whole-bank read permission.
struct StorePendingExport final {
  const SlotBindings *transaction = nullptr;
  compute::ConstTensorView capacity{}, indices{};
  compute::ReadLease indices_guard;
};
struct StoreBindingData;
struct StoreBindingAccess;
// One original region's metadata and original lease transaction. Construction
// cannot allocate; slots::prepare performs all fallible owner allocation.
class StoreBinding final {
public:
  StoreBinding() noexcept = default;
  StoreBinding(const StoreBinding &) = delete;
  StoreBinding &operator=(const StoreBinding &) = delete;
  HYLE_GRAPH_API StoreBinding(StoreBinding &&) noexcept;
  HYLE_GRAPH_API StoreBinding &operator=(StoreBinding &&) noexcept;
  HYLE_GRAPH_API ~StoreBinding();
  HYLE_GRAPH_API bool prepared() const noexcept;
  HYLE_GRAPH_API bool valid() const noexcept;
  // Shared-Queue submission only; native artifacts use their original source
  // endpoint and the same cold authority/configuration validation below.
  HYLE_GRAPH_API bool same_queue(const compute::detail::QueueSequence &) const noexcept;
  // Borrowed only in cold preparation/sealing, never permission to read the
  // capacity. Both IDs slices belong to the single private initialized carrier.
  HYLE_GRAPH_API StoreConfiguration configuration() const noexcept;
  HYLE_GRAPH_API const SlotBindings *transaction() const noexcept;
  // Pure current-transaction proof, not submission/readiness qualification.
  // The caller externally serializes cold binding preparation or holds its
  // exact submitted Source. No tickets are sealed, IDs uploaded or writers moved.
  HYLE_GRAPH_API Result<StorePendingExport> export_pending() const noexcept;
  // Checks the original parent, exact selected artifact and actual retained
  // readers/writers; reset/release cannot revive a stale transaction.
  HYLE_GRAPH_API Status validate_authority() const noexcept;
  HYLE_GRAPH_API Status validate_configuration(const StoreConfiguration &) const noexcept;
private:
  friend struct StoreBindingAccess;
  StoreBindingData *data_ = nullptr;
};
struct StoreRegionVTable final {
  Result<StoreRegionFacts> (*inspect)(const ProviderRegionPrepared &,
      void *invocation) noexcept = nullptr;
  StoreBinding *(*binding)(void *invocation) noexcept = nullptr;
  // Copies views only. No leases, uploads, publication or native work. Called
  // before moving the caller's transaction; failure leaves that owner intact.
  Status (*configure)(const ProviderRegionPrepared &, void *invocation,
      const StoreConfiguration &) noexcept = nullptr;
  // Retired/rollback cleanup, no allocation/failure/work. It releases borrowed
  // configuration without moving the common owner or changing fixed carriers.
  void (*clear_configuration)(const ProviderRegionPrepared &,
      void *invocation) noexcept = nullptr;
};

// Compile-local assembly facts. The returned artifact copies these values;
// no adapter or provider lookup is kept on the execution path.
struct StoreAdapter final {
  mem::MemoryResourceRef resource;
  compute::DeviceClass device_class = compute::DeviceClass::kCpu;
};
// Same-build module assembly, not a public provider selection API.
HYLE_GRAPH_API const RegionProviderVTable *store_provider() noexcept;
} // namespace hyle::graph::slots::detail
