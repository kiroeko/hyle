#pragma once

#include "provider_internal.h"

namespace hyle::compute::cuda_native::grouped_fp8 {
inline constexpr std::uint64_t kRoute = 20;
Status propose(void *state, const detail::ProviderOperationRequest &request,
               std::span<detail::ResourceBindingRequirement> inputs,
               std::span<detail::ResourceBindingRequirement> outputs,
               detail::ProviderOperationProposal *proposal) noexcept;
ProviderPrepareOutcome prepare(void *state,
                               const detail::ProviderPrepareRequest &request,
                               mem::AllocatorRef allocator,
                               detail::ProviderPrepared *output) noexcept;
} // namespace hyle::compute::cuda_native::grouped_fp8
