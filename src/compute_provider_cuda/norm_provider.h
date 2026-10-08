#pragma once

#include "provider_internal.h"

namespace hyle::compute::cuda_native::normalization {
inline constexpr std::uint64_t kRoute = 4;

Status propose(const detail::ProviderOperationRequest &request,
               std::span<detail::ResourceBindingRequirement> inputs,
               std::span<detail::ResourceBindingRequirement> outputs,
               detail::ProviderOperationProposal *proposal) noexcept;
ProviderPrepareOutcome prepare(void *state,
                               const detail::ProviderPrepareRequest &request,
                               mem::AllocatorRef allocator,
                               detail::ProviderPrepared *output) noexcept;
void require_alignment(detail::ProviderPrepared &, std::size_t) noexcept;
Status sglang_propose(const detail::ProviderOperationRequest &request,
                      std::span<detail::ResourceBindingRequirement> inputs,
                      std::span<detail::ResourceBindingRequirement> outputs,
                      detail::ProviderOperationProposal *proposal) noexcept;
ProviderPrepareOutcome
sglang_prepare(void *state, const detail::ProviderPrepareRequest &request,
               mem::AllocatorRef allocator,
               detail::ProviderPrepared *output) noexcept;
} // namespace hyle::compute::cuda_native::normalization
