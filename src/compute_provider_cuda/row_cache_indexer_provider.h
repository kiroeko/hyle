#pragma once
#include "provider_internal.h"

namespace hyle::compute::cuda_native::row_cache_indexer {
inline constexpr std::uint64_t kRoute = 11;
inline constexpr std::uint64_t kDenseRoute = 21;
bool inspect_access(const detail::ProviderPrepared &,
                    detail::OperationPreparedAccess &) noexcept;
Status configure_access(const detail::ProviderPrepared &, void *, const void *) noexcept;
Status propose(const detail::ProviderOperationRequest &,
               std::span<detail::ResourceBindingRequirement>,
               std::span<detail::ResourceBindingRequirement>,
               detail::ProviderOperationProposal *) noexcept;
ProviderPrepareOutcome prepare(void *, const detail::ProviderPrepareRequest &,
                               mem::AllocatorRef,
                               detail::ProviderPrepared *) noexcept;
} // namespace hyle::compute::cuda_native::row_cache_indexer
