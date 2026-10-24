#pragma once

#include "provider_internal.h"

namespace hyle::compute::cuda_native::float8_quantize {
inline constexpr std::uint64_t kRoute = 0x4638514e5401ULL;
Status propose(const detail::ProviderOperationRequest &,
               std::span<detail::ResourceBindingRequirement>,
               std::span<detail::ResourceBindingRequirement>,
               detail::ProviderOperationProposal *) noexcept;
ProviderPrepareOutcome prepare(void *, const detail::ProviderPrepareRequest &,
                               mem::AllocatorRef,
                               detail::ProviderPrepared *) noexcept;
} // namespace hyle::compute::cuda_native::float8_quantize
