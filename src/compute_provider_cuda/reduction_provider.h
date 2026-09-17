#pragma once

#include "provider_internal.h"

namespace hyle::compute::cuda_native::reduction {
inline constexpr std::uint64_t kRoute = 5;

Status propose(const detail::ProviderOperationRequest &request,
               std::span<detail::ResourceBindingRequirement> inputs,
               std::span<detail::ResourceBindingRequirement> outputs,
               detail::ProviderOperationProposal *proposal) noexcept;
ProviderPrepareOutcome prepare(void *state,
                               const detail::ProviderPrepareRequest &request,
                               mem::AllocatorRef allocator,
                               detail::ProviderPrepared *output) noexcept;
} // namespace hyle::compute::cuda_native::reduction
