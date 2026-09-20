#pragma once

#include "hyle/compute/providers/ggml_cpu.h"
#include "provider_internal.h"

namespace hyle::compute::ggml_cpu::standard_index {

inline constexpr std::uint64_t kRoute = 2;
bool supports(Operation, std::span<const TensorDesc>,
              std::span<const TensorDesc>) noexcept;
ProviderRuntimeProperties runtime() noexcept;
ProviderRouteCostFacts costs() noexcept;
ProviderPrepareOutcome prepare(Options, const detail::ProviderPrepareRequest &,
                               mem::AllocatorRef,
                               detail::ProviderPrepared *) noexcept;

} // namespace hyle::compute::ggml_cpu::standard_index
