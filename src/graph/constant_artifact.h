#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

#include "hyle/compute/detail/provider_extension.h"
#include "hyle/container/flat_hash_map.h"
#include "hyle/graph/region_provider.h"
#include "hyle/util/ref_counted.h"
#include "storage_budget.h"

namespace hyle::graph::detail {

struct PlanState;

struct ConstantArtifactKey final {
  compute::ProviderInstanceId provider{};
  std::uint32_t constant = 0;
  std::uint32_t storage = 0;
  ConstantPackingKey packing{};
  compute::TensorDesc description{};
  friend bool operator==(const ConstantArtifactKey &,
                         const ConstantArtifactKey &) noexcept = default;
};

struct ConstantArtifactState final : util::RefCounted<ConstantArtifactState> {
  explicit ConstantArtifactState(mem::AllocatorRef owner) noexcept
      : allocator(owner) {}
  ~ConstantArtifactState() {
    if (payload.destroy != nullptr)
      payload.destroy(payload.state, allocator);
  }
  void destroy() noexcept {
    const auto owner = allocator;
    this->~ConstantArtifactState();
    owner.deallocate(this, sizeof(ConstantArtifactState),
                     alignof(ConstantArtifactState));
  }
  mem::AllocatorRef allocator;
  compute::detail::ProviderExtensionRef extension;
  util::Ref<StorageBudgetState> budget;
  mem::QuotaAllocator::Reservation charge;
  ConstantArtifactKey key;
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

// No allocation until a provider requests packing. Lookup storage disappears
// before compile() returns; selected Plans retain only their used payloads.
class ConstantArtifactCache final {
public:
  Status begin_candidate() noexcept;
  void commit_plan() noexcept;
  compute::ProviderPrepareOutcome
  acquire(PlanState &plan,
          const compute::detail::ProviderExtensionRef &extension,
          ExecutionTargetId target, ValueId constant,
          ConstantArtifactDeclaration declaration,
          ConstantArtifactFactory factory, void *context,
          ConstantArtifactView *result, CompileOptions options) noexcept;

private:
  using Map = container::FlatHashMap<
      const ConstantArtifactKey *, util::Ref<ConstantArtifactState>,
      ConstantArtifactKeyHash, ConstantArtifactKeyEqual, mem::AllocatorRef>;
  void prune_unused() noexcept;
  compute::ProviderPrepareOutcome
  remember(PlanState &plan, const util::Ref<ConstantArtifactState> &artifact,
           ConstantArtifactView *result) noexcept;
  std::optional<Map> entries_;
  std::size_t candidate_ = 0;
};

struct ConstantArtifactScope final {
  ConstantArtifactCache &cache;
  PlanState &plan;
  const compute::detail::ProviderExtensionRef &extension;
  ExecutionTargetId target;
  CompileOptions options;
  bool active_factory = false;
  bool budget_declined = false;
  ConstantArtifactAccess access() noexcept {
    return ConstantArtifactAccess(this);
  }
};

} // namespace hyle::graph::detail
