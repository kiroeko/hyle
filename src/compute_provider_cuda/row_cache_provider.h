#pragma once

#include "provider_internal.h"

namespace hyle::compute::cuda_native::row_cache {
inline constexpr std::uint64_t kRoute = 10;
bool inspect_access(const detail::ProviderPrepared &,
                    detail::OperationPreparedAccess &) noexcept;
Status propose(const detail::ProviderOperationRequest &,
               std::span<detail::ResourceBindingRequirement>,
               std::span<detail::ResourceBindingRequirement>,
               detail::ProviderOperationProposal *) noexcept;
ProviderPrepareOutcome prepare(void *, const detail::ProviderPrepareRequest &,
                               mem::AllocatorRef, detail::ProviderPrepared *) noexcept;
} // namespace hyle::compute::cuda_native::row_cache
