#pragma once

#include "apple_copy_internal.h"
#include "hyle/compute/detail/device_lengths_paged_append_wire.h"

namespace hyle::compute::apple::detail {
// Cold descriptors from the original typed COW transaction. The arena is
// addressability only; each exact destination/prefix carries its own authority.
struct DeviceAppendPageBinding final {
  compute::detail::DeviceAppendPage page;
  mem::BufferView destination;
  mem::ConstBufferView prefix;
};
HYLE_COMPUTE_ROUTE_APPLE_API Status prepare_device_append_recording(CopyRecording &,
    compute::detail::DeviceLengthsPagedAppendSpec, const TensorDesc &rows) noexcept;
HYLE_COMPUTE_ROUTE_APPLE_API Status configure_device_append_recording(CopyRecording &,
    mem::ConstBufferView arena, std::span<const compute::detail::DeviceAppendRequest>,
    std::span<const DeviceAppendPageBinding>) noexcept;
// Original native inputs: packed rows, query lengths, KV lengths. Sources are
// pinned at the actual accepted epoch and resolved only by the existing FIFO.
HYLE_COMPUTE_ROUTE_APPLE_API Status bind_device_append_inputs(CopyRecording &,
    std::span<const InputBinding>) noexcept;
// Borrowed original device verdict. Readers must retain the original recording
// and import its accepted dependency through their real completion endpoint.
HYLE_COMPUTE_ROUTE_APPLE_API Result<ConstTensorView> device_append_status(CopyRecording &) noexcept;
} // namespace hyle::compute::apple::detail
