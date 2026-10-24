#pragma once

#include "vulkan_memory_internal.h"
#include "hyle/compute/weight_assembly.h"

namespace hyle::compute::vulkan::detail {
// Same original cold transfer pool/fence/semaphore preparation, shared by F2.
Status prepare_transfer_run(Run &run) noexcept;
struct UploadPart final {
  compute::ConstTensorView source;
  std::size_t offset = 0;
};
struct TransferAccess final {
  // Same slot owner/Run as the public factory, with an actual Runtime and
  // already allocated mapped native staging; no synthetic Context or Queue.
  static Result<TransferSlot> create(const std::shared_ptr<Runtime> &, Tensor staging,
                                    std::size_t capacity, mem::AllocatorRef) noexcept;
  // Cold coarse composition ownership only; this does not make bytes ready.
  static HYLE_COMPUTE_ROUTE_VULKAN_API Status
  retain(TransferSlot &slot, const Tensor &tensor) noexcept;
  static Result<Completion> upload(TransferSlot &slot, const Tensor &destination,
                                   std::span<const UploadPart> parts,
                                   std::size_t used_bytes,
                                   compute::WeightUploadReport *report = nullptr,
                                   bool check_source_aliases = false) noexcept;
  // Cold exact-byte device constant copy through the same staging allocation.
  // Both legs stay on the original compute queue; no host payload is copied.
  static Result<Completion> copy(TransferSlot &, const Tensor &source,
                                 const Tensor &destination) noexcept;
};
} // namespace hyle::compute::vulkan::detail
