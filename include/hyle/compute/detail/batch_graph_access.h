#pragma once
#include <memory>
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
  kDeviceLayoutChoice,
};
// Trusted Graph-only borrowed coordinates, never a storage permission. The
// original region owns this fixed workspace through every native retirement.
// Graph's existing root use admits execution/recording. The selected provider
// must separately prove that cold planning neither reads nor writes this float
// payload (it may only compute offsets), so planning needs no float write use.
// Public/standalone Batch invocations retain their original owned workspaces.
struct BatchGraphFloatWorkspace final {
  mem::BufferView storage;
};
// Graph's original shared-region admission/retirement owns external inputs,
// output and this same native slot. This bridge does not create a Completion,
// executor or shadow output. It may only run inside that admitted region.
struct BatchGraphAccess final {
  HYLE_COMPUTE_BATCH_API static Result<BatchProgram> prepare(const ContextRef&,
      const ProviderExtensionRef&,BatchSchema,BatchProgramOptions) noexcept;
  HYLE_COMPUTE_BATCH_API static Result<BatchInvocation> create(const BatchProgram&,Queue&) noexcept;
  // Only metadata copies use this same-Context Queue. Native plan/run/capture
  // stay on the original execution Queue. The shared wrapper remains at its
  // stable address through the original buffer/native-use lifetime; uploads
  // still really wait and publish ready metadata before Graph preparation.
  HYLE_COMPUTE_BATCH_API static Result<BatchInvocation> create(const BatchProgram&,
      Queue& execution,std::shared_ptr<Queue> metadata_queue) noexcept;
  // The fixed view must cover the program's declared float hard cap. This
  // entry neither acquires a lease nor manufactures Graph/root authority.
  HYLE_COMPUTE_BATCH_API static Result<BatchInvocation> create(const BatchProgram&,
      Queue& execution,std::shared_ptr<Queue> metadata_queue,
      BatchGraphFloatWorkspace) noexcept;
  HYLE_COMPUTE_BATCH_API static Result<ReadLease> metadata(const BatchInvocation&) noexcept;
  HYLE_COMPUTE_BATCH_API static Result<PreparedBatch> prepare(BatchInvocation&,
      const BatchLayoutVersion&,bool update,BeforeCommandChange) noexcept;
  HYLE_COMPUTE_BATCH_API static Result<PreparedBatch> prepare_choices(BatchInvocation&,
      const std::array<BatchLayoutVersion,2>&,bool update,BeforeCommandChange) noexcept;
  HYLE_COMPUTE_BATCH_API static Status validate_choices(const BatchInvocation&,
      const std::array<BatchLayoutVersion,2>&) noexcept;
  // Prepare a local plan using the source's already checked-retired same-layout
  // mirror. Sharing precedes upload; each plan/workspace stays independent.
  HYLE_COMPUTE_BATCH_API static Result<PreparedBatch> prepare_shared(BatchInvocation&,
      const BatchLayoutVersion&,bool update,const BatchInvocation&,BeforeCommandChange = {}) noexcept;
  HYLE_COMPUTE_BATCH_API static void clear_shared_metadata(BatchInvocation&) noexcept;
  // Cold seal of this original prepared slot using its retained Queue owner.
  // Saves only the actual metadata carrier identity, never read permission.
  HYLE_COMPUTE_BATCH_API static Status validate_prepared(
      const PreparedBatch&,BatchGraphContract) noexcept;
  // Read-only identity of the original prepared episode, including prepares
  // that reuse one immutable layout version. No new epoch/permission is made.
  HYLE_COMPUTE_BATCH_API static Result<std::uint64_t> preparation_epoch(
      const PreparedBatch&) noexcept;
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
