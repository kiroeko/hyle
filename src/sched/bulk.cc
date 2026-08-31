#include "hyle/sched/bulk.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <limits>
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

constexpr std::size_t kTargetReservationsPerRunner = 8;

struct ChunkReservation {
  std::size_t begin = 0;
  std::size_t end = 0;
};

struct BulkState {
  BulkBodyFn body = nullptr;
  void* context = nullptr;
  const CancelToken* cancel_token = nullptr;
  std::size_t begin = 0;
  std::size_t span = 0;
  std::size_t grain = 1;
  std::size_t chunk_count = 0;
  std::size_t reservation_size = 1;
  bool fetch_add_claims = false;

  std::atomic<std::size_t> next_chunk{0};
  // The initial count is a submission guard. A runner may finish while its
  // siblings are still being submitted, but completion cannot fire until the
  // caller releases this guard after the admission loop.
  std::atomic<unsigned> active{1};
  std::atomic<bool> failed{false};
  std::atomic<bool> cancelled{false};
  std::atomic<bool> notification_complete{false};
  util::SpinLock error_mu;
  Status first_error;
  Event completed{EventResetMode::kManual};

  bool claim(ChunkReservation& reservation) noexcept {
    if (fetch_add_claims) {
      const std::size_t first = next_chunk.fetch_add(
          reservation_size, std::memory_order_relaxed);
      if (first >= chunk_count) return false;
      reservation.begin = first;
      reservation.end = first +
                        (std::min)(reservation_size, chunk_count - first);
      return true;
    }

    std::size_t first = next_chunk.load(std::memory_order_relaxed);
    do {
      if (first >= chunk_count) return false;
      reservation.begin = first;
      reservation.end = first +
                        (std::min)(reservation_size, chunk_count - first);
    } while (!next_chunk.compare_exchange_weak(
        first, reservation.end, std::memory_order_relaxed,
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
    if (previous == 1) {
      // Event::signal() may wake the waiter before notify_all() itself has
      // returned. Publish only after the notifier has stopped touching Event;
      // the waiter treats this release as the BulkState lifetime handoff.
      completed.signal();
      notification_complete.store(true, std::memory_order_release);
    }
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
    ChunkReservation reservation;
    if (!state.claim(reservation)) return;

    for (std::size_t chunk = reservation.begin;
         chunk < reservation.end; ++chunk) {
      // A reservation is only an internal cursor optimization. Preserve the
      // public grain as the callback and cancellation unit by checking before
      // every body invocation, not merely once per reservation.
      if (state.failed.load(std::memory_order_acquire)) return;
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

  // Keep enough reservations for late load balancing while amortizing the
  // shared cursor across several public grains. This changes no callback
  // boundary: run_chunks() still invokes and checks each grain separately.
  const std::size_t maximum = (std::numeric_limits<std::size_t>::max)();
  const std::size_t concurrency_size = static_cast<std::size_t>(concurrency);
  const std::size_t target_capacity =
      concurrency_size <= maximum / kTargetReservationsPerRunner
          ? concurrency_size * kTargetReservationsPerRunner
          : maximum;
  const std::size_t target_reservations =
      (std::min)(chunks, target_capacity);
  state.reservation_size =
      chunks / target_reservations +
      (chunks % target_reservations != 0 ? 1 : 0);

  // fetch_add is the normal single-RMW claim. Near SIZE_MAX, failed terminal
  // claims could wrap the cursor, so retain a saturating CAS path for that
  // representable but impractical boundary instead of weakening correctness.
  state.fetch_add_claims =
      concurrency_size < maximum &&
      state.reservation_size <=
          (maximum - chunks) / (concurrency_size + 1);

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
  while (!state.notification_complete.load(std::memory_order_acquire)) {
    util::cpu_relax();
  }
  return state.result();
}

}  // namespace hyle::sched
