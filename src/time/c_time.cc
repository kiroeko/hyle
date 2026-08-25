#include "hyle/c/time.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

#include "../error/status_access.h"
#include "hyle/error/status.h"
#include "hyle/time/clock.h"
#include "hyle/time/common_timeout_queue.h"
#include "hyle/time/timer_queue.h"

struct hyle_time_delivery_context;

struct hyle_time_timer_o {
  hyle::time::TimerEntry entry;
  void* user_data = nullptr;
  // The low bit records the allocation policy while the remaining bits are a
  // transient pointer to the synchronous C delivery batch, if any. Delivery
  // reuses padding that the old bool already forced into the opaque object, so
  // it does not add per-timer storage or touch the native TimerEntry hot path.
  std::uintptr_t delivery_and_storage = 0;

  explicit hyle_time_timer_o(void* data, bool in_place = false) noexcept
      : entry(this),
        user_data(data),
        delivery_and_storage(in_place ? kCallerStorageBit : 0) {}

  [[nodiscard]] bool caller_storage() const noexcept {
    return (delivery_and_storage & kCallerStorageBit) != 0;
  }

  [[nodiscard]] hyle_time_delivery_context* delivery() const noexcept {
    return reinterpret_cast<hyle_time_delivery_context*>(delivery_and_storage &
                                                         ~kCallerStorageBit);
  }

  void begin_delivery(hyle_time_delivery_context* context) noexcept {
    HYLE_FATAL_ASSERT(context != nullptr);
    HYLE_FATAL_ASSERT(delivery() == nullptr);
    const auto address = reinterpret_cast<std::uintptr_t>(context);
    HYLE_FATAL_ASSERT((address & kCallerStorageBit) == 0);
    delivery_and_storage = address | (delivery_and_storage & kCallerStorageBit);
  }

  void end_delivery() noexcept { delivery_and_storage &= kCallerStorageBit; }

 private:
  static constexpr std::uintptr_t kCallerStorageBit = 1;
};

static_assert(std::is_standard_layout_v<hyle_time_timer_o>);
static_assert(sizeof(hyle_time_timer_o) % alignof(hyle_time_timer_o) == 0);
// The tagged storage/delivery state fits the same alignment slot as the former
// storage-kind byte. The native TimerEntry remains unchanged.
static_assert(sizeof(hyle_time_timer_o) <=
              sizeof(hyle::time::TimerEntry) + 2 * sizeof(void*));

struct hyle_time_delivery_context {
  hyle::time::TimerList* entries;
  const void* source;
};

static_assert(alignof(hyle_time_delivery_context) >= 2);

struct hyle_time_queue_o {
  hyle::time::TimerQueue queue;
  std::size_t active_deliveries = 0;
  explicit hyle_time_queue_o(uint64_t now_ns) noexcept : queue(now_ns) {}
};

struct hyle_time_common_queue_o {
  hyle::time::CommonTimeoutQueue queue;
  std::size_t active_deliveries = 0;
  explicit hyle_time_common_queue_o(uint64_t duration_ns) noexcept
      : queue(duration_ns) {}
};

namespace {

hyle_status steal_status(hyle::Status&& status) noexcept {
  return hyle::detail::status_to_ffi(std::move(status));
}

hyle::Status validate_pair(const void* source,
                           const hyle_time_timer* timer) noexcept {
  if (source == nullptr || timer == nullptr) {
    return hyle::Status::InvalidArgument(
        "time source and timer must be non-null");
  }
  const hyle_time_delivery_context* delivery = timer->delivery();
  if ((delivery != nullptr && delivery->source != source) ||
      (delivery == nullptr && timer->entry.scheduled() &&
       !hyle::time::detail::TimerEntryAccess::owned_by(timer->entry, source))) {
    return hyle::Status::FailedPrecondition(
        "timer is already scheduled on another source");
  }
  return hyle::Status{};
}

bool detach_pending_delivery(hyle_time_timer* timer) noexcept {
  hyle_time_delivery_context* delivery = timer->delivery();
  if (delivery == nullptr) return false;
  HYLE_FATAL_ASSERT(delivery->entries != nullptr);
  delivery->entries->remove(&timer->entry);
  timer->end_delivery();
  return true;
}

size_t visit_list(hyle::time::TimerList& entries, const void* source,
                  hyle_time_timer_visit_fn visit, void* ctx) noexcept {
  hyle_time_delivery_context delivery{&entries, source};
  entries.for_each([&](hyle::time::TimerEntry* entry) noexcept {
    auto* timer = static_cast<hyle_time_timer*>(entry->data());
    HYLE_FATAL_ASSERT(timer != nullptr);
    timer->begin_delivery(&delivery);
  });

  size_t count = 0;
  while (hyle::time::TimerEntry* entry = entries.pop_front()) {
    auto* timer = static_cast<hyle_time_timer*>(entry->data());
    HYLE_FATAL_ASSERT(timer != nullptr);
    timer->end_delivery();
    ++count;
    if (visit != nullptr) visit(timer, ctx);
  }
  return count;
}

template <class Source>
size_t advance_and_visit(Source& source, const void* source_token,
                         std::size_t& active_deliveries, uint64_t now_ns,
                         hyle_time_timer_visit_fn visit, void* ctx) noexcept {
  hyle::time::TimerList fired;
  source.advance(now_ns, fired);
  ++active_deliveries;
  const std::size_t count = visit_list(fired, source_token, visit, ctx);
  --active_deliveries;
  return count;
}

template <class Source>
size_t cancel_all_and_visit(Source& source, const void* source_token,
                            std::size_t& active_deliveries,
                            hyle_time_timer_visit_fn visit,
                            void* ctx) noexcept {
  hyle::time::TimerList cancelled;
  source.cancel_all(cancelled);
  ++active_deliveries;
  const std::size_t count = visit_list(cancelled, source_token, visit, ctx);
  --active_deliveries;
  return count;
}

}  // namespace

extern "C" {

uint64_t hyle_time_monotonic_ns(void) { return hyle::time::monotonic_ns(); }
uint64_t hyle_time_realtime_ns(void) { return hyle::time::realtime_ns(); }
void hyle_time_realtime_timestamp(hyle_unix_timestamp* out) {
  if (out != nullptr) *out = hyle::time::realtime_timestamp();
}
void hyle_time_sleep_for_ns(uint64_t ns) { hyle::time::sleep_for_ns(ns); }
void hyle_time_sleep_until_ns(uint64_t deadline_ns) {
  hyle::time::sleep_until_ns(deadline_ns);
}
uint64_t hyle_time_deadline_after_ns(uint64_t timeout_ns) {
  return hyle::time::deadline_after_ns(timeout_ns);
}
uint64_t hyle_time_remaining_ns(uint64_t deadline_ns) {
  return hyle::time::remaining_ns(deadline_ns);
}

hyle_status hyle_time_timer_create(hyle_time_timer** out, void* user_data) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("timer output must be non-null"));
  }
  *out = nullptr;
  auto* timer = new (std::nothrow) hyle_time_timer(user_data);
  if (timer == nullptr) {
    return steal_status(
        hyle::Status::ResourceExhausted("timer allocation failed"));
  }
  *out = timer;
  return steal_status(hyle::Status{});
}

hyle_status hyle_time_timer_destroy(hyle_time_timer* timer) {
  if (timer == nullptr) return steal_status(hyle::Status{});
  if (timer->delivery() != nullptr) {
    if (timer->caller_storage()) {
      return steal_status(hyle::Status::FailedPrecondition(
          "caller-storage timer must be released with timer_deinit"));
    }
    (void)detach_pending_delivery(timer);
    delete timer;
    return steal_status(hyle::Status{});
  }
  if (timer->entry.scheduled()) {
    return steal_status(hyle::Status::FailedPrecondition(
        "scheduled timer must be cancelled before destroy"));
  }
  if (timer->caller_storage()) {
    return steal_status(hyle::Status::FailedPrecondition(
        "caller-storage timer must be released with timer_deinit"));
  }
  delete timer;
  return steal_status(hyle::Status{});
}

size_t hyle_time_timer_storage_size(void) { return sizeof(hyle_time_timer); }

size_t hyle_time_timer_storage_alignment(void) {
  return alignof(hyle_time_timer);
}

hyle_status hyle_time_timer_init(hyle_time_timer** out, void* storage,
                                 size_t storage_size, void* user_data) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("timer output must be non-null"));
  }
  *out = nullptr;
  if (storage == nullptr || storage_size < sizeof(hyle_time_timer)) {
    return steal_status(hyle::Status::InvalidArgument(
        "timer storage is null or smaller than timer_storage_size"));
  }
  if (reinterpret_cast<std::uintptr_t>(storage) % alignof(hyle_time_timer) !=
      0) {
    return steal_status(hyle::Status::InvalidArgument(
        "timer storage does not satisfy timer_storage_alignment"));
  }
  *out = ::new (storage) hyle_time_timer(user_data, true);
  return steal_status(hyle::Status{});
}

hyle_status hyle_time_timer_deinit(hyle_time_timer* timer) {
  if (timer == nullptr) return steal_status(hyle::Status{});
  if (timer->delivery() != nullptr) {
    if (!timer->caller_storage()) {
      return steal_status(hyle::Status::FailedPrecondition(
          "created timer must be released with timer_destroy"));
    }
    (void)detach_pending_delivery(timer);
    std::destroy_at(timer);
    return steal_status(hyle::Status{});
  }
  if (timer->entry.scheduled()) {
    return steal_status(hyle::Status::FailedPrecondition(
        "scheduled timer must be cancelled before deinit"));
  }
  if (!timer->caller_storage()) {
    return steal_status(hyle::Status::FailedPrecondition(
        "created timer must be released with timer_destroy"));
  }
  std::destroy_at(timer);
  return steal_status(hyle::Status{});
}

void* hyle_time_timer_user_data(const hyle_time_timer* timer) {
  return timer != nullptr ? timer->user_data : nullptr;
}

void hyle_time_timer_set_user_data(hyle_time_timer* timer, void* user_data) {
  if (timer != nullptr) timer->user_data = user_data;
}

uint64_t hyle_time_timer_deadline_ns(const hyle_time_timer* timer) {
  return timer != nullptr ? timer->entry.deadline_ns() : 0;
}

int hyle_time_timer_is_scheduled(const hyle_time_timer* timer) {
  return timer != nullptr && timer->entry.scheduled() ? 1 : 0;
}

hyle_status hyle_time_queue_create_at(hyle_time_queue** out, uint64_t now_ns) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("timer queue output must be non-null"));
  }
  *out = nullptr;
  auto* queue = new (std::nothrow) hyle_time_queue(now_ns);
  if (queue == nullptr) {
    return steal_status(
        hyle::Status::ResourceExhausted("timer queue allocation failed"));
  }
  *out = queue;
  return steal_status(hyle::Status{});
}

hyle_status hyle_time_queue_create(hyle_time_queue** out) {
  return hyle_time_queue_create_at(out, hyle::time::monotonic_ns());
}

hyle_status hyle_time_queue_destroy(hyle_time_queue* queue) {
  if (queue == nullptr) return steal_status(hyle::Status{});
  if (queue->active_deliveries != 0) {
    return steal_status(hyle::Status::FailedPrecondition(
        "timer queue cannot be destroyed from an active visitor"));
  }
  if (!queue->queue.empty()) {
    return steal_status(hyle::Status::FailedPrecondition(
        "timer queue must be empty before destroy"));
  }
  delete queue;
  return steal_status(hyle::Status{});
}

hyle_status hyle_time_queue_schedule(hyle_time_queue* queue,
                                     hyle_time_timer* timer,
                                     uint64_t deadline_ns) {
  hyle::Status status = validate_pair(queue, timer);
  if (!status.ok()) return steal_status(std::move(status));
  (void)detach_pending_delivery(timer);
  queue->queue.schedule(&timer->entry, deadline_ns);
  return steal_status(hyle::Status{});
}

hyle_status hyle_time_queue_cancel(int* out_cancelled, hyle_time_queue* queue,
                                   hyle_time_timer* timer) {
  if (out_cancelled != nullptr) *out_cancelled = 0;
  hyle::Status status = validate_pair(queue, timer);
  if (!status.ok()) return steal_status(std::move(status));
  const bool cancelled =
      detach_pending_delivery(timer) || queue->queue.cancel(&timer->entry);
  if (out_cancelled != nullptr) *out_cancelled = cancelled ? 1 : 0;
  return steal_status(hyle::Status{});
}

hyle_status hyle_time_queue_advance(size_t* out_fired, hyle_time_queue* queue,
                                    uint64_t now_ns,
                                    hyle_time_timer_visit_fn visit, void* ctx) {
  if (out_fired != nullptr) *out_fired = 0;
  if (out_fired == nullptr || queue == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "timer queue and fired output must be non-null"));
  }
  if (now_ns < queue->queue.now_ns()) {
    return steal_status(hyle::Status::InvalidArgument(
        "timer queue advance timestamp must be non-decreasing"));
  }
  *out_fired = advance_and_visit(queue->queue, queue, queue->active_deliveries,
                                 now_ns, visit, ctx);
  return steal_status(hyle::Status{});
}

hyle_status hyle_time_queue_cancel_all(size_t* out_cancelled,
                                       hyle_time_queue* queue,
                                       hyle_time_timer_visit_fn visit,
                                       void* ctx) {
  if (out_cancelled != nullptr) *out_cancelled = 0;
  if (out_cancelled == nullptr || queue == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "timer queue and cancelled output must be non-null"));
  }
  *out_cancelled = cancel_all_and_visit(queue->queue, queue,
                                        queue->active_deliveries, visit, ctx);
  return steal_status(hyle::Status{});
}

uint64_t hyle_time_queue_next_deadline_ns(const hyle_time_queue* queue) {
  return queue != nullptr ? queue->queue.next_deadline_ns() : UINT64_MAX;
}
size_t hyle_time_queue_size(const hyle_time_queue* queue) {
  return queue != nullptr ? queue->queue.size() : 0;
}
int hyle_time_queue_empty(const hyle_time_queue* queue) {
  return queue == nullptr || queue->queue.empty() ? 1 : 0;
}

hyle_status hyle_time_common_queue_create(hyle_time_common_queue** out,
                                          uint64_t duration_ns) {
  if (out == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "common timeout queue output must be non-null"));
  }
  *out = nullptr;
  auto* queue = new (std::nothrow) hyle_time_common_queue(duration_ns);
  if (queue == nullptr) {
    return steal_status(hyle::Status::ResourceExhausted(
        "common timeout queue allocation failed"));
  }
  *out = queue;
  return steal_status(hyle::Status{});
}

hyle_status hyle_time_common_queue_destroy(hyle_time_common_queue* queue) {
  if (queue == nullptr) return steal_status(hyle::Status{});
  if (queue->active_deliveries != 0) {
    return steal_status(hyle::Status::FailedPrecondition(
        "common timeout queue cannot be destroyed from an active visitor"));
  }
  if (!queue->queue.empty()) {
    return steal_status(hyle::Status::FailedPrecondition(
        "common timeout queue must be empty before destroy"));
  }
  delete queue;
  return steal_status(hyle::Status{});
}

hyle_status hyle_time_common_queue_schedule(hyle_time_common_queue* queue,
                                            hyle_time_timer* timer,
                                            uint64_t deadline_ns) {
  hyle::Status status = validate_pair(queue, timer);
  if (!status.ok()) return steal_status(std::move(status));
  if (timer->delivery() != nullptr &&
      !hyle::time::detail::CommonTimeoutQueueAccess::accepts_idle_deadline(
          queue->queue, deadline_ns)) {
    return steal_status(hyle::Status::InvalidArgument(
        "common timeout deadlines must be non-decreasing"));
  }
  (void)detach_pending_delivery(timer);
  const bool ok = queue->queue.try_schedule(&timer->entry, deadline_ns);
  if (!ok) {
    return steal_status(hyle::Status::InvalidArgument(
        "common timeout deadlines must be non-decreasing"));
  }
  return steal_status(hyle::Status{});
}

hyle_status hyle_time_common_queue_schedule_after(hyle_time_common_queue* queue,
                                                  hyle_time_timer* timer,
                                                  uint64_t now_ns) {
  hyle::Status status = validate_pair(queue, timer);
  if (!status.ok()) return steal_status(std::move(status));
  const uint64_t deadline_ns =
      hyle::time::saturating_add_ns(now_ns, queue->queue.duration_ns());
  if (timer->delivery() != nullptr &&
      !hyle::time::detail::CommonTimeoutQueueAccess::accepts_idle_deadline(
          queue->queue, deadline_ns)) {
    return steal_status(hyle::Status::InvalidArgument(
        "common timeout deadlines must be non-decreasing"));
  }
  (void)detach_pending_delivery(timer);
  const bool ok = queue->queue.try_schedule(&timer->entry, deadline_ns);
  if (!ok) {
    return steal_status(hyle::Status::InvalidArgument(
        "common timeout deadlines must be non-decreasing"));
  }
  return steal_status(hyle::Status{});
}

hyle_status hyle_time_common_queue_cancel(int* out_cancelled,
                                          hyle_time_common_queue* queue,
                                          hyle_time_timer* timer) {
  if (out_cancelled != nullptr) *out_cancelled = 0;
  hyle::Status status = validate_pair(queue, timer);
  if (!status.ok()) return steal_status(std::move(status));
  const bool cancelled =
      detach_pending_delivery(timer) || queue->queue.cancel(&timer->entry);
  if (out_cancelled != nullptr) *out_cancelled = cancelled ? 1 : 0;
  return steal_status(hyle::Status{});
}

hyle_status hyle_time_common_queue_advance(size_t* out_fired,
                                           hyle_time_common_queue* queue,
                                           uint64_t now_ns,
                                           hyle_time_timer_visit_fn visit,
                                           void* ctx) {
  if (out_fired != nullptr) *out_fired = 0;
  if (out_fired == nullptr || queue == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "common timeout queue and fired output must be non-null"));
  }
  *out_fired = advance_and_visit(queue->queue, queue, queue->active_deliveries,
                                 now_ns, visit, ctx);
  return steal_status(hyle::Status{});
}

hyle_status hyle_time_common_queue_cancel_all(size_t* out_cancelled,
                                              hyle_time_common_queue* queue,
                                              hyle_time_timer_visit_fn visit,
                                              void* ctx) {
  if (out_cancelled != nullptr) *out_cancelled = 0;
  if (out_cancelled == nullptr || queue == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "common timeout queue and cancelled output must be non-null"));
  }
  *out_cancelled = cancel_all_and_visit(queue->queue, queue,
                                        queue->active_deliveries, visit, ctx);
  return steal_status(hyle::Status{});
}

uint64_t hyle_time_common_queue_next_deadline_ns(
    const hyle_time_common_queue* queue) {
  return queue != nullptr ? queue->queue.next_deadline_ns() : UINT64_MAX;
}
size_t hyle_time_common_queue_size(const hyle_time_common_queue* queue) {
  return queue != nullptr ? queue->queue.size() : 0;
}
int hyle_time_common_queue_empty(const hyle_time_common_queue* queue) {
  return queue == nullptr || queue->queue.empty() ? 1 : 0;
}
uint64_t hyle_time_common_queue_duration_ns(
    const hyle_time_common_queue* queue) {
  return queue != nullptr ? queue->queue.duration_ns() : 0;
}

}  // extern "C"
