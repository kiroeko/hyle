#pragma once

#include "hyle/compute/device.h"
#include "hyle/error/result.h"
#include "hyle/mem/resource.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_COMPUTE_API
#elif defined(HYLE_COMPUTE_BUILD)
#  define HYLE_COMPUTE_API HYLE_EXPORT
#else
#  define HYLE_COMPUTE_API HYLE_IMPORT
#endif

namespace hyle::compute {

// Process-lifetime storage providers for one Device. The returned references do
// not own backend state and remain valid for the process; Buffers allocated from
// them must still outlive every in-flight use on a Stream.
//
// device_memory_resource is the device-preferred storage route (VRAM on a
// discrete GPU, ordinary host memory on the host backend). host_memory_resource
// is host-visible staging storage and carries kHostPinned only when the backend
// actually provides page-locked memory. managed_memory_resource is optional and
// returns Unimplemented when the selected backend has no unified allocation.
HYLE_COMPUTE_API Result<mem::MemoryResourceRef> device_memory_resource(
    const Device& device) noexcept;
HYLE_COMPUTE_API Result<mem::MemoryResourceRef> host_memory_resource(
    const Device& device) noexcept;
HYLE_COMPUTE_API Result<mem::MemoryResourceRef> managed_memory_resource(
    const Device& device) noexcept;

// True only for one of this Device's native providers. Domain equality alone
// is not enough to interpret an opaque native handle; typed launch/copy adapters
// use this check before handing a Mem view to a Compute backend.
HYLE_COMPUTE_API bool memory_resource_belongs_to(
    mem::MemoryResourceRef resource, const Device& device) noexcept;

}  // namespace hyle::compute
