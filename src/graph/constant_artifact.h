#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

#include "hyle/graph/detail/execution_source.h"
#include "hyle/compute/device.h"
#include "hyle/container/flat_hash_map.h"
#include "hyle/container/small_vector.h"
#include "hyle/graph/region_provider.h"
#include "hyle/mem/buffer.h"
#include "hyle/util/ref_counted.h"
#include "storage_budget.h"

namespace hyle::graph::detail {

struct PlanState;
struct PreparedConstantOutput final {
  ValueId value;
  compute::ConstTensorView binding;
};
using PreparedConstantOutputs = container::SmallVector<
    PreparedConstantOutput, 2, mem::AllocatorRef>;

struct CanonicalConstantKey final {
  std::uint32_t constant = 0;
  compute::TensorDesc description{};
  mem::MemoryResourceRef resource{};
  std::size_t alignment = 1;
  friend bool operator==(const CanonicalConstantKey &,
                         const CanonicalConstantKey &) noexcept = default;
};

// One immutable allocation, independently retained by exactly its selected
// Plans. The family-local cache keeps lookup references only during compile.
struct CanonicalConstantState final : util::RefCounted<CanonicalConstantState> {
  explicit CanonicalConstantState(mem::AllocatorRef owner) noexcept
      : allocator(owner) {}
  void destroy() noexcept {
    const auto owner = allocator;
    const auto retained = metadata_owner;
    this->~CanonicalConstantState();
    owner.deallocate(this, sizeof(CanonicalConstantState),
                     alignof(CanonicalConstantState));
  }
  mem::AllocatorRef allocator;
  compute::StorageOwner metadata_owner;
  compute::StorageOwner storage_owner;
  CanonicalConstantKey key;
  util::Ref<StorageBudgetState> budget;
  // Reverse destruction frees bytes before returning their reservation.
  StorageCharge charge;
  mem::Buffer buffer;
  // Mutated only by the unpublished, single-threaded family compilation.
  std::size_t last_candidate = 0;
  std::uint32_t candidate_owner_index = UINT32_MAX;
  bool retained_by_prior_plan = false;
  bool ready = false;
  CompiledStorageReport storage() const noexcept {
    return {.materialized_constant_bytes = buffer.size()};
  }
};

struct CanonicalConstantKeyHash final {
  std::uint64_t operator()(const CanonicalConstantKey *key) const noexcept;
};
struct CanonicalConstantKeyEqual final {
  bool operator()(const CanonicalConstantKey *left,
                  const CanonicalConstantKey *right) const noexcept {
    return *left == *right;
  }
};

struct CanonicalConstantAcquisition final {
  util::Ref<CanonicalConstantState> owner;
  bool created = false;
};

struct ConstantArtifactKey final {
  const void *owner = nullptr;
  const void *native_capability = nullptr;
  compute::ProviderInstanceId provider{};
  std::uint32_t constant = 0;
  std::uint32_t storage = 0;
  ConstantPackingKey packing{};
  compute::TensorDesc description{};
  std::uint32_t root = 0;
  std::size_t source_offset = 0;
  friend bool operator==(const ConstantArtifactKey &,
                         const ConstantArtifactKey &) noexcept = default;
};

struct ConstantArtifactState final : util::RefCounted<ConstantArtifactState> {
  explicit ConstantArtifactState(mem::AllocatorRef owner) noexcept
      : allocator(owner), packing_identity(owner) {}
  ~ConstantArtifactState() {
    if (payload.destroy != nullptr)
      payload.destroy(payload.state, allocator);
  }
  void destroy() noexcept {
    const auto owner = allocator;
    const auto retained = metadata_owner;
    this->~ConstantArtifactState();
    owner.deallocate(this, sizeof(ConstantArtifactState),
                     alignof(ConstantArtifactState));
  }
  mem::AllocatorRef allocator;
  compute::StorageOwner metadata_owner;
  compute::StorageOwner storage_owner;
  GraphRegionSource source;
  util::Ref<StorageBudgetState> budget;
  StorageCharge charge;
  ConstantArtifactKey key;
  // Core metadata is allocated/charged through the original Plan allocator,
  // independently of the provider's declared persistent/packed bytes.
  container::SmallVector<std::byte, 0, mem::AllocatorRef> packing_identity;
  compute::ProviderCompilationStorage private_storage;
  ConstantArtifactPayload payload;
  std::size_t packed_bytes = 0;
  std::optional<std::size_t> persistent_bytes;
  ExecutionTargetId target;
  // Mutated only by the unpublished, single-threaded family compilation.
  std::size_t last_candidate = 0;
  bool retained_by_prior_plan = false;
  CompiledStorageReport storage() const noexcept {
    return {.provider_persistent_bytes = persistent_bytes.value_or(0),
            .packed_constant_bytes = packed_bytes,
            .unknown_provider_count = persistent_bytes ? 0U : 1U};
  }
};

struct ConstantArtifactKeyHash final {
  std::uint64_t operator()(const ConstantArtifactKey *key) const noexcept;
};
struct ConstantArtifactKeyEqual final {
  bool operator()(const ConstantArtifactKey *left,
                  const ConstantArtifactKey *right) const noexcept {
    return *left == *right;
  }
};

// No allocation until a canonical copy or provider packing is requested.
// Lookup storage disappears before compile() returns; selected Plans retain
// only their used payloads.
class ConstantArtifactCache final {
public:
  HYLE_GRAPH_API Status begin_candidate() noexcept;
  HYLE_GRAPH_API void commit_plan() noexcept;
  // The slot's original binding supplies its immutable logical description;
  // physical_instance selects exactly one primary or replica owner mapping.
  // A miss reserves metadata/bytes and returns an unready allocation. The
  // caller marks it ready only after the complete copy batch terminates
  // successfully. Cache hits never expose unfinished bytes.
  HYLE_GRAPH_API Result<CanonicalConstantAcquisition>
  acquire_canonical(PlanState &plan, std::size_t slot,
                    std::uint32_t physical_instance,
                    mem::MemoryResourceRef destination,
                    std::size_t minimum_alignment) noexcept;
  compute::ProviderPrepareOutcome
  acquire(PlanState &plan,
          const GraphRegionSource &source,
          ExecutionTargetId target,
          std::span<const ConstantBinding> physical_constants, ValueId constant,
          ConstantArtifactDeclaration declaration,
          ConstantArtifactFactory factory, void *context,
          ConstantArtifactView *result, CompileOptions options,
          std::span<const RegionConstantOrigin> origins = {}) noexcept;

private:
  struct PreparationQueue final {
    const void *endpoint = nullptr;
    compute::Queue queue;
  };
  struct PreparationQueues final {
    PreparationQueues(compute::StorageOwner owner,
                      util::Ref<StorageBudgetState> retained_budget,
                      mem::AllocatorRef allocator) noexcept
        : metadata_owner(std::move(owner)), budget(std::move(retained_budget)),
          entries(allocator) {}
    // The original allocator can belong to either retained owner. Destroy
    // queue entries and their metadata before releasing either backing owner.
    compute::StorageOwner metadata_owner;
    util::Ref<StorageBudgetState> budget;
    container::SmallVector<PreparationQueue, 0, mem::AllocatorRef> entries;
  };
  using CanonicalMap = container::FlatHashMap<
      const CanonicalConstantKey *, util::Ref<CanonicalConstantState>,
      CanonicalConstantKeyHash, CanonicalConstantKeyEqual, mem::AllocatorRef>;
  using Map = container::FlatHashMap<
      const ConstantArtifactKey *, util::Ref<ConstantArtifactState>,
      ConstantArtifactKeyHash, ConstantArtifactKeyEqual, mem::AllocatorRef>;
  void prune_unused() noexcept;
  Status remember_canonical(
      PlanState &plan, std::uint32_t physical_instance,
      const util::Ref<CanonicalConstantState> &owner) noexcept;
  compute::ProviderPrepareOutcome
  remember(PlanState &plan, const util::Ref<ConstantArtifactState> &artifact,
           ConstantArtifactView *result) noexcept;
  // Packed payloads can borrow canonical bytes. Destroy/prune packing first.
  std::optional<CanonicalMap> canonical_entries_;
  std::optional<Map> entries_;
  std::size_t candidate_ = 0;
  // Compile/family-local, lazy and keyed by the actual retained Compute
  // Context, independently of provider/target slots on that Context.
  std::optional<PreparationQueues> preparation_queues_;
};

struct ConstantArtifactScope final {
  ConstantArtifactCache &cache;
  PlanState &plan;
  const GraphRegionSource &source;
  ExecutionTargetId target;
  CompileOptions options;
  std::span<const ConstantBinding> constants;
  std::span<const ValueId> allowed_outputs;
  std::span<const mem::MemoryResourceRef> output_resources;
  PreparedConstantOutputs outputs;
  bool active_factory = false;
  bool budget_declined = false;
  std::span<const RegionConstantOrigin> constant_origins{};
  ConstantArtifactAccess access() noexcept {
    return ConstantArtifactAccess(this);
  }
};

} // namespace hyle::graph::detail
