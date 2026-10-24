#pragma once

#include <cstdint>
#include <optional>

#include "../compute/provider_internal.h"

namespace hyle::compute::native::internal {

// Same-build CPU facade seam; no route-owned public type crosses back here.
enum class ThreadRuntimeBackend : std::uint8_t {
  kUnavailable,
  kOpenMP,
  kNative,
};
struct ThreadRuntimeInfo final {
  ThreadRuntimeBackend backend = ThreadRuntimeBackend::kUnavailable;
  std::optional<std::uint32_t> configured_intra_op_threads{};
  std::optional<std::uint32_t> intra_op_limit{};
  std::optional<std::uint32_t> inter_op_limit{};
  bool configuration_open = false;
  bool usage_started = false;
};
HYLE_COMPUTE_PROVIDER_NATIVE_API Status
configure_thread_runtime(std::uint32_t intra_op_threads) noexcept;
HYLE_COMPUTE_PROVIDER_NATIVE_API Result<ThreadRuntimeInfo>
thread_runtime_info() noexcept;
// Cold admission only. Never held across math, callbacks or completion waits.
HYLE_COMPUTE_PROVIDER_NATIVE_API void mark_thread_runtime_used() noexcept;
bool thread_runtime_allows_qualified_default() noexcept;

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
bool inspect_torch_cpu_access(const detail::ProviderPrepared &,
                             detail::OperationPreparedAccess &) noexcept;

} // namespace hyle::compute::native::internal
