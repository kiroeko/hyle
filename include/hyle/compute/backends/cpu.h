#pragma once

#include <concepts>
#include <functional>
#include <type_traits>
#include <utility>

#include "hyle/compute/completion.h"
#include "hyle/compute/detail/queue_sequence.h"
#include "hyle/compute/device.h"
#include "hyle/mem/resource.h"

namespace hyle::compute::cpu {

inline constexpr BackendId kBackendId{0x48594c4500000001ULL,
                                      0x4350550000000001ULL};

HYLE_COMPUTE_BACKEND_CPU_API Backend backend() noexcept;
HYLE_COMPUTE_BACKEND_CPU_API Device device() noexcept;
HYLE_COMPUTE_BACKEND_CPU_API Result<mem::MemoryResourceRef>
memory_resource(Device device) noexcept;

// CPU-only typed escape hatch. The callable runs on the caller thread before
// return, is never retained, and performs no allocation. Admission is atomic
// with Queue/Context close, matching prepared submissions.
template <class Function, class... Args>
  requires std::is_nothrow_invocable_v<Function, Args...> &&
           (std::same_as<std::invoke_result_t<Function, Args...>, void> ||
            std::same_as<std::invoke_result_t<Function, Args...>, Status>)
Result<Completion> submit_host(Queue &queue, Function &&function,
                               Args &&...args) noexcept {
  auto sequence = detail::QueueSequence::begin(queue);
  if (!sequence)
    return std::unexpected(std::move(sequence).error());

  if constexpr (std::same_as<std::invoke_result_t<Function, Args...>, void>) {
    std::invoke(std::forward<Function>(function), std::forward<Args>(args)...);
    sequence->record(detail::EnqueueResult::accepted_command());
  } else {
    sequence->record(detail::EnqueueResult::accepted_command(
        std::invoke(std::forward<Function>(function),
                    std::forward<Args>(args)...)));
  }
  return sequence->finish();
}

} // namespace hyle::compute::cpu
