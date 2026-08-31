#pragma once

#include <concepts>
#include <cstddef>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#include "hyle/error/status.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_SCHED_API
#elif defined(HYLE_SCHED_BUILD)
#  define HYLE_SCHED_API HYLE_EXPORT
#else
#  define HYLE_SCHED_API HYLE_IMPORT
#endif

namespace hyle::sched {

class CancelToken;
class Scheduler;

// Policy for one synchronous, non-suspending bulk episode. The caller chooses
// the algorithmic grain: Sched only distributes the resulting half-open chunks.
// max_concurrency == 0 uses the Scheduler's worker count; larger values are
// clamped to it. The optional token is borrowed until bulk_execute() returns.
struct BulkExecuteOptions {
  std::size_t grain_size = 1;
  unsigned max_concurrency = 0;
  const CancelToken* cancel_token = nullptr;
};

// Process one half-open chunk [begin, end). The callback runs on a Scheduler
// provider worker. A Fiber already running on this same Scheduler may execute
// chunks inline on its own Fiber stack while worker-stack runners handle sibling
// chunks; external callers and Fibers owned by another Scheduler never execute
// the body. The callback must not suspend/block/throw and may return a shared
// batch error. The first concurrently observed non-OK Status wins; body calls
// already in flight finish while each runner stops before its next invocation.
// The same context may be called concurrently by multiple participants; mutable
// shared state is the caller's synchronization responsibility.
using BulkBodyFn =
    Status (*)(void* context, std::size_t begin, std::size_t end) noexcept;

// Executes [begin, end) as a single structured batch and returns only after all
// admitted participants are quiescent. Results belong in caller-owned storage
// that outlives this call. External-thread and cross-Scheduler Fiber calls submit
// at most O(worker-count) allocation-free runners sharing O(1) batch state. A
// same-Scheduler Fiber participates and submits at most worker-count minus one;
// it suspends only if a sibling is still running after its local share. A call
// nested inside a non-suspending worker-stack callback executes serially on that
// callback to preserve progress even for a one-worker Scheduler.
HYLE_SCHED_API Status bulk_execute(
    Scheduler& scheduler, std::size_t begin, std::size_t end, BulkBodyFn body,
    void* context, BulkExecuteOptions options = {}) noexcept;

namespace detail {

template <class Fn>
Status invoke_bulk_body(void* context, std::size_t begin,
                        std::size_t end) noexcept {
  auto* fn = static_cast<Fn*>(context);
  if constexpr (std::same_as<std::invoke_result_t<Fn&, std::size_t,
                                                  std::size_t>,
                             void>) {
    std::invoke(*fn, begin, end);
    return Status();
  } else {
    return std::invoke(*fn, begin, end);
  }
}

}  // namespace detail

// Owning-free C++ adapter. The callable is borrowed for this synchronous call,
// so move-only callables work without a callable allocation. Multiple runners
// may invoke the same object concurrently. Returning void means an infallible
// body; returning Status enables first-error stop.
template <class Fn>
  requires std::is_nothrow_invocable_v<Fn&, std::size_t, std::size_t> &&
           (std::same_as<std::invoke_result_t<Fn&, std::size_t, std::size_t>,
                         void> ||
            std::same_as<std::invoke_result_t<Fn&, std::size_t, std::size_t>,
                         Status>)
Status bulk_execute(Scheduler& scheduler, std::size_t begin, std::size_t end,
                    Fn&& body, BulkExecuteOptions options = {}) noexcept {
  using Callable = std::remove_reference_t<Fn>;
  void* context = const_cast<void*>(
      static_cast<const void*>(std::addressof(body)));
  return bulk_execute(scheduler, begin, end,
                      &detail::invoke_bulk_body<Callable>, context, options);
}

}  // namespace hyle::sched
