#pragma once

#include "vulkan_memory_internal.h"

namespace hyle::compute::vulkan::detail {
// Same original cold transfer pool/fence/semaphore preparation, shared by F2.
Status prepare_transfer_run(Run &run) noexcept;
struct UploadPart final {
  compute::ConstTensorView source;
  std::size_t offset = 0;
};
struct TransferAccess final {
  // Cold coarse composition ownership only; this does not make bytes ready.
  static HYLE_COMPUTE_ROUTE_VULKAN_API Status
  retain(TransferSlot &slot, const Tensor &tensor) noexcept;
  static Result<Completion> upload(TransferSlot &slot, const Tensor &destination,
                                   std::span<const UploadPart> parts,
                                   std::size_t used_bytes) noexcept;
};
} // namespace hyle::compute::vulkan::detail
