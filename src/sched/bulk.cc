#include "hyle/sched/bulk.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <utility>

#include "hyle/core/compiler.h"
#include "hyle/sched/cancel.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"
#include "hyle/sched/sync.h"
#include "hyle/sched/waker.h"
#include "hyle/util/spin_lock.h"

namespace hyle::sched {
namespace {

struct BulkState {
  BulkBodyFn body = nullptr;
  void* context = nullptr;
  const CancelToken* cancel_token = nullptr;
  std::size_t begin = 0;
  std::size_t span = 0;
  std::size_t grain = 1;
  std::size_t chunk_count = 0;

  std::atomic<std::size_t> next_chunk{0};
  // The initial count is a submission guard. A runner may finish while its
  // siblings are still being submitted, but completion cannot fire until the
  // caller releases this guard after the admission loop.
  std::atomic<unsigned> active{1};
  std::atomic<bool> failed{false};
  std::atomic<bool> cancelled{false};
  util::SpinLock error_mu;
  Status first_error;
  Event completed{EventResetMode::kManual};

  bool claim(std::size_t& index) noexcept {
    index = next_chunk.load(std::memory_order_relaxed);
    do {
      if (index >= chunk_count) return false;
    } while (!next_chunk.compare_exchange_weak(
        index, index + 1, std::memory_order_relaxed,
        std::memory_order_relaxed));
    return true;
  }

  void record_error(Status error) noexcept {
    bool expected = false;
    if (!failed.compare_exchange_strong(expected, true,
                                        std::memory_order_acq_rel,
                                        std::memory_order_acquire)) {
      return;
    }
    error_mu.lock();
    first_error = std::move(error);
    error_mu.unlock();
  }

  void finish_one() noexcept {
    const unsigned previous = active.fetch_sub(1, std::memory_order_acq_rel);
    HYLE_FATAL_ASSERT(previous != 0);
    if (previous == 1) completed.signal();
  }

  Status result() noexcept {
    if (failed.load(std::memory_order_acquire)) {
      error_mu.lock();
      Status result = first_error;
      error_mu.unlock();
      return result;
    }
    if (cancelled.load(std::memory_order_acquire)) {
      return Status::Cancelled("bulk execution cancelled");
    }
    return Status();
  }
};

void run_chunks(BulkState& state) noexcept {
  for (;;) {
    if (state.failed.load(std::memory_order_acquire)) return;

    std::size_t chunk = 0;
    if (!state.claim(chunk)) return;

    // Check after a successful claim. This makes cancellation observable only
    // when it actually prevents a chunk from running; a token cancelled after
    // the final chunk completed does not retroactively change success.
    if (state.cancel_token != nullptr &&
        state.cancel_token->is_cancelled()) {
      state.cancelled.store(true, std::memory_order_release);
      return;
    }

    const std::size_t offset = chunk * state.grain;
    const std::size_t length =
        (std::min)(state.grain, state.span - offset);
    Status status = state.body(state.context, state.begin + offset,
                               state.begin + offset + length);
    if (!status.ok()) {
      state.record_error(std::move(status));
      return;
    }
  }
}

void run_with_cancel_scope(BulkState& state) noexcept {
  if (state.cancel_token != nullptr && state.cancel_token->valid()) {
    CancelScope scope(*state.cancel_token);
    run_chunks(state);
    return;
  }
  run_chunks(state);
}

void run_bulk_runner(void* context) noexcept {
  auto& state = *static_cast<BulkState*>(context);
  run_with_cancel_scope(state);
  state.finish_one();
}

}  // namespace

Status bulk_execute(Scheduler& scheduler, std::size_t begin,
                    std::size_t end, BulkBodyFn body, void* context,
                    BulkExecuteOptions options) noexcept {
  if (!scheduler.valid()) {
    return Status::FailedPrecondition("bulk execution requires a valid Scheduler");
  }
  if (body == nullptr) {
    return Status::InvalidArgument("bulk execution body is null");
  }
  if (begin > end) {
    return Status::InvalidArgument("bulk execution range is reversed");
  }
  if (options.grain_size == 0) {
    return Status::InvalidArgument("bulk execution grain_size must be nonzero");
  }
  if (begin == end) return Status();
  if (options.cancel_token != nullptr &&
      options.cancel_token->is_cancelled()) {
    return Status::Cancelled("bulk execution cancelled before admission");
  }

  const std::size_t span = end - begin;
  const std::size_t chunks =
      span / options.grain_size + (span % options.grain_size != 0 ? 1 : 0);
  BulkState state;
  state.body = body;
  state.context = context;
  state.cancel_token = options.cancel_token;
  state.begin = begin;
  state.span = span;
  state.grain = options.grain_size;
  state.chunk_count = chunks;

  // schedule()/schedule_task() callbacks run on a worker stack, where waiting
  // would block the very worker needed to drain the queue (a guaranteed
  // deadlock with one worker). Preserve the full result/cancel contract with a
  // documented serial fallback. Scheduler Fibers remain on the parallel path:
  // Event::wait() suspends them instead of blocking the worker.
  if (current_worker_index() >= 0 && Fiber::current() == nullptr) {
    run_with_cancel_scope(state);
    return state.result();
  }

  const unsigned workers = scheduler.num_workers();
  if (workers == 0) {
    return Status::FailedPrecondition("bulk execution Scheduler has no workers");
  }
  unsigned concurrency = options.max_concurrency == 0
                             ? workers
                             : (std::min)(options.max_concurrency, workers);
  concurrency = static_cast<unsigned>(
      (std::min)(static_cast<std::size_t>(concurrency), chunks));

  unsigned admitted = 0;
  Status admission_error;
  for (unsigned i = 0; i < concurrency; ++i) {
    state.active.fetch_add(1, std::memory_order_relaxed);
    Status status = scheduler.schedule(&run_bulk_runner, &state);
    if (!status.ok()) {
      state.finish_one();
      admission_error = std::move(status);
      break;
    }
    ++admitted;
  }

  state.finish_one();  // release the submission guard
  if (admitted == 0) {
    return admission_error.WithContext("bulk execution admission");
  }

  // If only part of the desired runner set was admitted, every admitted runner
  // can still drain the shared cursor. Queue pressure reduces concurrency; it
  // never drops an item or changes the result contract.
  state.completed.wait();
  return state.result();
}

}  // namespace hyle::sched
