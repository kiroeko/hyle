#pragma once

#include <concepts>
#include <functional>
#include <type_traits>
#include <utility>

#include "hyle/compute/completion.h"
#include "hyle/compute/device.h"

namespace hyle::compute {

// CPU-only typed escape hatch for a direct engineering kernel. It executes on
// the caller thread before returning, so a capturing callable is never retained
// and no type-erased argument bag or heap allocation is introduced.
template <class Function, class... Args>
  requires std::is_nothrow_invocable_v<Function, Args...> &&
           (std::same_as<std::invoke_result_t<Function, Args...>, void> ||
            std::same_as<std::invoke_result_t<Function, Args...>, Status>)
Result<Completion> submit_host(Queue& queue, Function&& function,
                               Args&&... args) noexcept {
  if (!queue.valid()) {
    return std::unexpected(Status::FailedPrecondition("queue is invalid"));
  }
  if (!queue.is_open()) {
    return std::unexpected(Status::FailedPrecondition("queue is closed"));
  }

  if constexpr (std::same_as<std::invoke_result_t<Function, Args...>, void>) {
    std::invoke(std::forward<Function>(function),
                std::forward<Args>(args)...);
    return Completion::completed();
  } else {
    return Completion::completed(std::invoke(
        std::forward<Function>(function), std::forward<Args>(args)...));
  }
}

}  // namespace hyle::compute
