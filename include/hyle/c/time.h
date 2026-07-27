/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary
 *
 * Pure C ABI for HYLE clocks and timer queues (close-01).
 * C++ layouts and intrusive hooks never cross this edge.
 * Docs (1:1 with this header): docs/guide/c-time.md
 */
#ifndef HYLE_C_TIME_H_
#define HYLE_C_TIME_H_

#include <stddef.h>
#include <stdint.h>

#include "hyle/c/error.h"
#include "hyle/c/time_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(HYLE_STATIC)
#  define HYLE_TIME_C_API
#elif defined(_WIN32) || defined(__CYGWIN__)
#  if defined(HYLE_TIME_BUILD)
#    define HYLE_TIME_C_API __declspec(dllexport)
#  else
#    define HYLE_TIME_C_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define HYLE_TIME_C_API __attribute__((visibility("default")))
#else
#  define HYLE_TIME_C_API
#endif

/* Clock values are nanoseconds. Deadlines/elapsed time use monotonic, never
 * realtime. profiling_clock is local-span sampling only; query availability. */
HYLE_TIME_C_API uint64_t hyle_time_monotonic_ns(void);
HYLE_TIME_C_API uint64_t hyle_time_profiling_clock_ns(void);
HYLE_TIME_C_API int hyle_time_profiling_clock_available(void);
HYLE_TIME_C_API uint64_t hyle_time_realtime_ns(void);
HYLE_TIME_C_API void hyle_time_realtime_timestamp(
    hyle_unix_timestamp* out);
HYLE_TIME_C_API uint64_t hyle_time_coarse_realtime_ns(void);
HYLE_TIME_C_API int hyle_time_coarse_realtime_available(void);
HYLE_TIME_C_API void hyle_time_sleep_for_ns(uint64_t ns);
HYLE_TIME_C_API void hyle_time_sleep_until_ns(uint64_t deadline_ns);
HYLE_TIME_C_API uint64_t hyle_time_deadline_after_ns(uint64_t timeout_ns);
HYLE_TIME_C_API uint64_t hyle_time_remaining_ns(uint64_t deadline_ns);

/* Header-inline unit conveniences. Multiplication saturates instead of wrapping. */
static inline void hyle_time_sleep_for_us(uint64_t us) {
  hyle_time_sleep_for_ns(
      us > UINT64_MAX / UINT64_C(1000) ? UINT64_MAX
                                       : us * UINT64_C(1000));
}
static inline void hyle_time_sleep_for_ms(uint64_t ms) {
  hyle_time_sleep_for_ns(
      ms > UINT64_MAX / UINT64_C(1000000) ? UINT64_MAX
                                          : ms * UINT64_C(1000000));
}

typedef struct hyle_time_timer_o hyle_time_timer;
typedef struct hyle_time_queue_o hyle_time_queue;
typedef struct hyle_time_common_queue_o hyle_time_common_queue;

/* Visitor runs synchronously on the thread calling advance. The timer has
 * already left its source and may be rescheduled by the visitor. */
typedef void (*hyle_time_timer_visit_fn)(hyle_time_timer* timer, void* ctx);

/* A reusable opaque timer node. Creation allocates once; arm/cancel/expire do
 * not allocate. For large timer sets, the storage_size/alignment + init/deinit
 * path lets callers allocate controls in one arena/slab while the layout stays
 * opaque (the same future-proof shape as libevent event_assign). Each init must
 * receive fresh or previously deinitialized storage. The storage and timer must
 * remain alive until deinit. Do not mix create/destroy with init/deinit.
 * Recoverable misuse returns owning hyle_status; free errors with
 * hyle_status_free. Destroy/deinit leave a scheduled timer alive and report
 * FAILED_PRECONDITION instead of aborting across the C boundary. */
HYLE_TIME_C_API hyle_status hyle_time_timer_create(hyle_time_timer** out,
                                                   void* user_data);
HYLE_TIME_C_API hyle_status hyle_time_timer_destroy(hyle_time_timer* timer);
HYLE_TIME_C_API size_t hyle_time_timer_storage_size(void);
HYLE_TIME_C_API size_t hyle_time_timer_storage_alignment(void);
HYLE_TIME_C_API hyle_status hyle_time_timer_init(
    hyle_time_timer** out, void* storage, size_t storage_size,
    void* user_data);
HYLE_TIME_C_API hyle_status hyle_time_timer_deinit(hyle_time_timer* timer);
HYLE_TIME_C_API void* hyle_time_timer_user_data(const hyle_time_timer* timer);
HYLE_TIME_C_API void hyle_time_timer_set_user_data(hyle_time_timer* timer,
                                                   void* user_data);
HYLE_TIME_C_API uint64_t
hyle_time_timer_deadline_ns(const hyle_time_timer* timer);
HYLE_TIME_C_API int hyle_time_timer_is_scheduled(const hyle_time_timer* timer);

/* General arbitrary-deadline radix timer queue. One source is single-thread
 * owned. Destroy requires empty. schedule reschedules an entry on this queue;
 * using the same timer with another live source returns FAILED_PRECONDITION. */
/* Normal construction samples the monotonic origin. create_at is for virtual
 * clocks, deterministic tests and event loops that already own a timestamp. */
HYLE_TIME_C_API hyle_status hyle_time_queue_create(hyle_time_queue** out);
HYLE_TIME_C_API hyle_status hyle_time_queue_create_at(hyle_time_queue** out,
                                                      uint64_t now_ns);
HYLE_TIME_C_API hyle_status hyle_time_queue_destroy(hyle_time_queue* queue);
HYLE_TIME_C_API hyle_status hyle_time_queue_schedule(
    hyle_time_queue* queue, hyle_time_timer* timer, uint64_t deadline_ns);
static inline hyle_status hyle_time_queue_schedule_after(
    hyle_time_queue* queue, hyle_time_timer* timer, uint64_t timeout_ns) {
  return hyle_time_queue_schedule(
      queue, timer, hyle_time_deadline_after_ns(timeout_ns));
}
/* out_cancelled receives 1 when an armed timer was detached, 0 when it was
 * already idle. It may be NULL when the distinction is irrelevant. */
HYLE_TIME_C_API hyle_status hyle_time_queue_cancel(
    int* out_cancelled, hyle_time_queue* queue, hyle_time_timer* timer);
/* now_ns must be non-decreasing across calls. Due timers are visited
 * synchronously; order across different radix buckets is unspecified. */
HYLE_TIME_C_API hyle_status hyle_time_queue_advance(
    size_t* out_fired, hyle_time_queue* queue, uint64_t now_ns,
    hyle_time_timer_visit_fn visit, void* ctx);
/* Detach all timers without firing them; visitor order is unspecified. */
HYLE_TIME_C_API hyle_status hyle_time_queue_cancel_all(
    size_t* out_cancelled, hyle_time_queue* queue,
    hyle_time_timer_visit_fn visit, void* ctx);
HYLE_TIME_C_API uint64_t
hyle_time_queue_next_deadline_ns(const hyle_time_queue* queue);
/* UINT64_MAX is both the empty sentinel and a valid far deadline; call empty()
 * when that distinction matters. */
HYLE_TIME_C_API size_t hyle_time_queue_size(const hyle_time_queue* queue);
HYLE_TIME_C_API int hyle_time_queue_empty(const hyle_time_queue* queue);

/* Fixed-duration ordered queue. schedule_after(now) is the normal entry point.
 * schedule(deadline) accepts an explicit absolute deadline but returns
 * INVALID_ARGUMENT when it would violate non-decreasing order; a rejected
 * reschedule preserves the timer's prior arm. Destroy requires empty. */
HYLE_TIME_C_API hyle_status hyle_time_common_queue_create(
    hyle_time_common_queue** out, uint64_t duration_ns);
HYLE_TIME_C_API hyle_status
hyle_time_common_queue_destroy(hyle_time_common_queue* queue);
HYLE_TIME_C_API hyle_status hyle_time_common_queue_schedule(
    hyle_time_common_queue* queue, hyle_time_timer* timer,
    uint64_t deadline_ns);
HYLE_TIME_C_API hyle_status hyle_time_common_queue_schedule_after(
    hyle_time_common_queue* queue, hyle_time_timer* timer, uint64_t now_ns);
static inline hyle_status hyle_time_common_queue_schedule_now(
    hyle_time_common_queue* queue, hyle_time_timer* timer) {
  return hyle_time_common_queue_schedule_after(
      queue, timer, hyle_time_monotonic_ns());
}
HYLE_TIME_C_API hyle_status hyle_time_common_queue_cancel(
    int* out_cancelled, hyle_time_common_queue* queue,
    hyle_time_timer* timer);
HYLE_TIME_C_API hyle_status hyle_time_common_queue_advance(
    size_t* out_fired, hyle_time_common_queue* queue, uint64_t now_ns,
    hyle_time_timer_visit_fn visit, void* ctx);
HYLE_TIME_C_API hyle_status hyle_time_common_queue_cancel_all(
    size_t* out_cancelled, hyle_time_common_queue* queue,
    hyle_time_timer_visit_fn visit, void* ctx);
HYLE_TIME_C_API uint64_t hyle_time_common_queue_next_deadline_ns(
    const hyle_time_common_queue* queue);
HYLE_TIME_C_API size_t
hyle_time_common_queue_size(const hyle_time_common_queue* queue);
HYLE_TIME_C_API int
hyle_time_common_queue_empty(const hyle_time_common_queue* queue);
HYLE_TIME_C_API uint64_t
hyle_time_common_queue_duration_ns(const hyle_time_common_queue* queue);

#ifdef __cplusplus
}
#endif

#endif /* HYLE_C_TIME_H_ */
