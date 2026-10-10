#pragma once

#include "vulkan_memory_internal.h"

namespace hyle::compute::vulkan::detail {
struct UploadPart final {
  compute::ConstTensorView source;
  std::size_t offset = 0;
};
struct TransferAccess final {
  static Result<Completion> upload(TransferSlot &slot, const Tensor &destination,
                                   std::span<const UploadPart> parts,
                                   std::size_t used_bytes) noexcept;
};
} // namespace hyle::compute::vulkan::detail
