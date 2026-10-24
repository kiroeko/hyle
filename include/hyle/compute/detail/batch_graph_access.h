#pragma once
#include "hyle/compute/detail/batch_provider.h"
#include "hyle/compute/detail/context_ref.h"
namespace hyle::compute::detail {
// Borrowed only for one synchronous cold preparation. The callback runs before
// recorded native resources change; it is never retained by a batch slot.
struct BeforeCommandChange final {
  void *state = nullptr;
  Status (*call)(void *) noexcept = nullptr;
};
// Both are trusted cold Graph contracts. Explicit capacity metadata is bound
// and protected by Graph's owned configuration; its exact Q/output port extent
// can be below the schema maximum, with active rows bounded by that extent.
// Native math retains its own prepared mirror even when an authorized copy
// changes the physical carrier. Owned-layout ports remain exactly active-sized.
enum class BatchGraphContract : std::uint8_t {
  kPreparedMetadataCapacity,
  kOwnedLayoutActive,
};
// Graph's original shared-region admission/retirement owns external inputs,
// output and this same native slot. This bridge does not create a Completion,
// executor or shadow output. It may only run inside that admitted region.
struct BatchGraphAccess final {
  HYLE_COMPUTE_BATCH_API static Result<BatchProgram> prepare(const ContextRef&,
      const ProviderExtensionRef&,BatchSchema,BatchProgramOptions) noexcept;
  HYLE_COMPUTE_BATCH_API static Result<BatchInvocation> create(const BatchProgram&,Queue&) noexcept;
  HYLE_COMPUTE_BATCH_API static Result<ReadLease> metadata(const BatchInvocation&) noexcept;
  HYLE_COMPUTE_BATCH_API static Result<PreparedBatch> prepare(BatchInvocation&,
      const BatchLayoutVersion&,bool update,BeforeCommandChange) noexcept;
  // Prepare a local plan using the source's already checked-retired same-layout
  // mirror. Sharing precedes upload; each plan/workspace stays independent.
  HYLE_COMPUTE_BATCH_API static Result<PreparedBatch> prepare_shared(BatchInvocation&,
      const BatchLayoutVersion&,bool update,const BatchInvocation&,BeforeCommandChange = {}) noexcept;
  HYLE_COMPUTE_BATCH_API static void clear_shared_metadata(BatchInvocation&) noexcept;
  // Cold seal of this original prepared slot using its retained Queue owner.
  // Saves only the actual metadata carrier identity, never read permission.
  HYLE_COMPUTE_BATCH_API static Status validate_prepared(
      const PreparedBatch&,BatchGraphContract) noexcept;
  // Cold Graph owns these exact bound ports; recording executes no tensors.
  HYLE_COMPUTE_BATCH_API static Status prepare_replay(PreparedBatch&,Queue&,
      std::span<const ConstTensorView>,std::span<const TensorView>,BatchGraphContract) noexcept;
  HYLE_COMPUTE_BATCH_API static Status validate_replay(PreparedBatch&,Queue&,
      std::span<const ConstTensorView>,std::span<const TensorView>,BatchGraphContract) noexcept;
  HYLE_COMPUTE_BATCH_API static EnqueueResult enqueue(PreparedBatch&,QueueSequence&,
      std::span<const ConstTensorView>,std::span<const TensorView>,BatchGraphContract) noexcept;
  // Cold proof from the actual prepared provider slot, never from a backend
  // family or an empty logical shape alone. Unknown capabilities return false.
  HYLE_COMPUTE_BATCH_API static Result<bool> empty_cache_reads(
      const PreparedBatch&,BatchGraphContract) noexcept;
};
} // namespace hyle::compute::detail
