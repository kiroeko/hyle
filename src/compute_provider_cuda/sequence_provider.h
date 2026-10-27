#pragma once

#include "provider_internal.h"

namespace hyle::compute::cuda_native::sequence {
inline constexpr std::uint64_t kRoute = 0x53455155454e01ULL;
bool preserves_capture_bindings(const detail::ProviderPrepared &) noexcept;
Status propose(const detail::ProviderOperationRequest &,
               std::span<detail::ResourceBindingRequirement>,
               std::span<detail::ResourceBindingRequirement>,
               detail::ProviderOperationProposal *) noexcept;
ProviderPrepareOutcome prepare(void *, const detail::ProviderPrepareRequest &,
                               mem::AllocatorRef, detail::ProviderPrepared *) noexcept;
} // namespace hyle::compute::cuda_native::sequence
