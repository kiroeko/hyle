#pragma once

#include <cuda_runtime_api.h>

#include "hyle/compute/detail/paged_append_wire.h"
#include "hyle/compute/detail/device_lengths_paged_append_wire.h"

namespace hyle::compute::detail {struct DeviceLengthsPagedAppendVTable;}

namespace hyle::compute::cuda::detail {

const void *paged_append_kernel() noexcept;
cudaError_t launch_paged_append(
    const void *kernel, void *arena, const void *rows, const void *metadata,
    compute::detail::PagedAppendNativeSpec spec, cudaStream_t stream) noexcept;
const void *device_choice_paged_append_kernel() noexcept;
cudaError_t launch_device_choice_paged_append(const void *kernel,void *arena,
    const void *rows,const void *metadata,const void *selector,
    compute::detail::PagedAppendNativeSpec spec,cudaStream_t stream)noexcept;
const compute::detail::DeviceLengthsPagedAppendVTable*device_lengths_paged_append_interface()noexcept;
const void*device_lengths_paged_append_kernel()noexcept;
cudaError_t launch_device_lengths_paged_append(const void*,void*,const void*,const void*,
    const void*,const void*,compute::detail::DeviceLengthsPagedAppendSpec,unsigned*,cudaStream_t)noexcept;

} // namespace hyle::compute::cuda::detail
