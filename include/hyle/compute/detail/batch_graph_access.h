#pragma once
#include "hyle/compute/detail/batch_provider.h"
namespace hyle::compute::detail {
// Graph's original shared-region admission/retirement owns external inputs,
// output and this same native slot. This bridge does not create a Completion,
// executor or shadow output. It may only run inside that admitted region.
struct BatchGraphAccess final {
  HYLE_COMPUTE_BATCH_API static Result<BatchProgram> prepare(const Context&,BatchSchema,BatchProgramOptions,ProviderInstanceId) noexcept;
  HYLE_COMPUTE_BATCH_API static Result<BatchInvocation> create(const BatchProgram&,Queue&) noexcept;
  HYLE_COMPUTE_BATCH_API static Result<ReadLease> metadata(const BatchInvocation&) noexcept;
  // Prepare a local plan using the source's already checked-retired same-layout
  // mirror. Sharing precedes upload; each plan/workspace stays independent.
  HYLE_COMPUTE_BATCH_API static Result<PreparedBatch> prepare_shared(BatchInvocation&,
      const BatchLayoutVersion&,bool update,const BatchInvocation&) noexcept;
  HYLE_COMPUTE_BATCH_API static void clear_shared_metadata(BatchInvocation&) noexcept;
  // Cold Graph owns these exact bound ports; recording executes no tensors.
  HYLE_COMPUTE_BATCH_API static Status prepare_replay(PreparedBatch&,Queue&,
      std::span<const ConstTensorView>,std::span<const TensorView>) noexcept;
  HYLE_COMPUTE_BATCH_API static Status validate_replay(PreparedBatch&,Queue&,
      std::span<const ConstTensorView>,std::span<const TensorView>) noexcept;
  HYLE_COMPUTE_BATCH_API static EnqueueResult enqueue(PreparedBatch&,Queue&,std::span<const ConstTensorView>,std::span<const TensorView>) noexcept;
};
} // namespace hyle::compute::detail
