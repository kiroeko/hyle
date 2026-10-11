#pragma once

#include <span>
#include <utility>

#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/resource_leases.h"

namespace hyle::compute {
namespace detail { struct LeasedOperationState; }

// ReadLease inputs share a ready version. Reading through a WriteLease is an
// explicit in-place/exclusive path; the original Provider still checks aliasing.
struct LeasedInput final {
  TensorDesc description{};
  ReadLease read{};
  const WriteLease *exclusive = nullptr;
  LeasedInput(TensorDesc desc, ReadLease lease) noexcept
      : description(desc), read(std::move(lease)) {}
  LeasedInput(TensorDesc desc, const WriteLease &lease) noexcept
      : description(desc), exclusive(&lease) {}
};
struct LeasedOutput final {
  TensorDesc description{};
  const WriteLease *lease = nullptr;
  LeasedOutput(TensorDesc desc, const WriteLease &write) noexcept
      : description(desc), lease(&write) {}
};

struct LeasedOperationOptions final {
  mem::AllocatorRef allocator = mem::system_allocator();
  ResourceBudget resource_budget{};
};

// Opt-in owning binding adapter over the original prepared operation. Cold
// creation allocates fixed views/use metadata and takes the workspace lease.
// Inputs/output references are borrowed only during submit; actual accepted
// work automatically retains every lease through the original native runtime.
// Control is thread-compatible, with one concurrent run per instance. Source
// LeasePools, native resource/Context, Queue and allocator obey their lifetimes.
class LeasedOperationInvocation final {
public:
  LeasedOperationInvocation() noexcept = default;
  LeasedOperationInvocation(const LeasedOperationInvocation &) = delete;
  LeasedOperationInvocation &operator=(const LeasedOperationInvocation &) = delete;
  HYLE_COMPUTE_API LeasedOperationInvocation(LeasedOperationInvocation &&) noexcept;
  HYLE_COMPUTE_API LeasedOperationInvocation &operator=(LeasedOperationInvocation &&) noexcept;
  HYLE_COMPUTE_API ~LeasedOperationInvocation();
  HYLE_COMPUTE_API static Result<LeasedOperationInvocation>
  create(const PreparedOperation &prepared, WriteLease workspace = {},
         LeasedOperationOptions options = {}) noexcept;
  bool valid() const noexcept { return state_ != nullptr; }
  HYLE_COMPUTE_API bool in_flight() const noexcept;
  HYLE_COMPUTE_API ProviderIdentity provider() const noexcept;
  HYLE_COMPUTE_API Result<Completion>
  submit(Queue &queue, std::span<const LeasedInput> inputs,
         std::span<const LeasedOutput> outputs) noexcept;
  // Observations of this owner and its Completion are externally serialized.
  // One cold completion slot: observe/drop an old Completion before replay.
  HYLE_COMPUTE_API Status close() noexcept;
  // Observes this accepted run even while its public Completion is held.
  // Unknown retirement returns its original error and retains native uses.
  HYLE_COMPUTE_API Status drain() noexcept;
private:
  explicit LeasedOperationInvocation(detail::LeasedOperationState *state) noexcept : state_(state) {}
  detail::LeasedOperationState *state_ = nullptr;
};
} // namespace hyle::compute
