#pragma once

#include "../compute/provider_internal.h"

namespace hyle::compute::native::internal {

// Cold routing only. The original ds4 group64/group32 routes stay unchanged.
bool is_torch_cpu_operation(const Operation &operation,
                           std::span<const TensorDesc> inputs) noexcept;
Status propose_torch_cpu(
    const detail::ProviderOperationRequest &request,
    std::span<detail::ResourceBindingRequirement> input_requirements,
    std::span<detail::ResourceBindingRequirement> output_requirements,
    detail::ProviderOperationProposal *proposal) noexcept;
ProviderPrepareOutcome prepare_torch_cpu(
    const detail::ProviderPrepareRequest &request, mem::AllocatorRef allocator,
    detail::ProviderPrepared *prepared) noexcept;

} // namespace hyle::compute::native::internal
