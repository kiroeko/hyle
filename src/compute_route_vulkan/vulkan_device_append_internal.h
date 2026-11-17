#pragma once

#include "vulkan_copy_internal.h"
#include "hyle/compute/detail/device_lengths_paged_append_wire.h"

namespace hyle::compute::vulkan::detail {
struct DeviceAppendPageBinding final {
  compute::detail::DeviceAppendPage page;
  mem::BufferView destination;
  mem::ConstBufferView prefix;
};
// Another native encoding of the ORIGINAL CopyRecording/Run, with the same
// completion and retirement. The arena grants addressability, not permission;
// each destination and inherited prefix must match its original COW authority.
HYLE_COMPUTE_ROUTE_VULKAN_API Status prepare_device_append_recording(CopyRecording &,
    compute::detail::DeviceLengthsPagedAppendSpec, const TensorDesc &rows,
    InvocationAllocation = {}) noexcept;
HYLE_COMPUTE_ROUTE_VULKAN_API Status configure_device_append_recording(CopyRecording &,
    mem::ConstBufferView arena, std::span<const compute::detail::DeviceAppendRequest>,
    std::span<const DeviceAppendPageBinding>) noexcept;
// Exact native rows/Q/KV carriers, cold bound before submission. Optional
// original producer error words are folded on-device before any page writes.
HYLE_COMPUTE_ROUTE_VULKAN_API Status bind_device_append_inputs(CopyRecording &,
    std::span<const Tensor>, std::span<const Tensor> source_checks = {}) noexcept;
// Borrowed device verdict; its original recording and ordered dependency must
// remain alive. This is not a ready result or another publication mechanism.
HYLE_COMPUTE_ROUTE_VULKAN_API Result<ConstTensorView> device_append_status(CopyRecording &) noexcept;
} // namespace hyle::compute::vulkan::detail
