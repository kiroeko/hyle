/* SPDX: same as HYLE.
 *
 * Pure C ABI for HYLE clocks and timer sources (close-01).
 * C++ layouts, intrusive hooks and TimerSource concepts never cross this edge.
 * Docs (1:1 with this header): docs/guide/c-time.md
 */
#ifndef HYLE_C_TIME_H_
#define HYLE_C_TIME_H_

#include <stddef.h>
#include <stdint.h>

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

/* Thread-local diagnostic for the most recent recoverable C-boundary failure.
 * Successful fallible calls clear stale text. Never free the returned pointer. */
HYLE_TIME_C_API const char* hyle_time_last_error(void);

/* Clock values are nanoseconds. Deadlines/elapsed time use monotonic, never
 * realtime. fast_monotonic is profiling-only; query availability first. */
HYLE_TIME_C_API uint64_t hyle_time_monotonic_ns(void);
HYLE_TIME_C_API uint64_t hyle_time_fast_monotonic_ns(void);
HYLE_TIME_C_API int hyle_time_fast_monotonic_available(void);
HYLE_TIME_C_API uint64_t hyle_time_realtime_ns(void);
HYLE_TIME_C_API uint64_t hyle_time_fast_realtime_ns(void);
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
typedef struct hyle_time_wheel_o hyle_time_wheel;
typedef struct hyle_time_common_queue_o hyle_time_common_queue;

/* Visitor runs synchronously on the thread calling advance. The timer has
 * already left its source and may be rescheduled by the visitor. */
typedef void (*hyle_time_timer_visit_fn)(hyle_time_timer* timer, void* ctx);

/* A reusable opaque timer node. Creation allocates once; arm/cancel/expire do
 * not allocate. A timer must be unscheduled before destroy. */
HYLE_TIME_C_API hyle_time_timer* hyle_time_timer_create(void* user_data);
HYLE_TIME_C_API void hyle_time_timer_destroy(hyle_time_timer* timer);
HYLE_TIME_C_API void* hyle_time_timer_user_data(const hyle_time_timer* timer);
HYLE_TIME_C_API void hyle_time_timer_set_user_data(hyle_time_timer* timer,
                                                   void* user_data);
HYLE_TIME_C_API uint64_t
hyle_time_timer_deadline_ns(const hyle_time_timer* timer);
HYLE_TIME_C_API int hyle_time_timer_is_scheduled(const hyle_time_timer* timer);

/* General arbitrary-deadline timer wheel. One source is single-thread owned.
 * Destroy requires empty. schedule reschedules an entry already on this wheel;
 * using the same timer with another live source returns 0 + last_error. */
HYLE_TIME_C_API hyle_time_wheel* hyle_time_wheel_create(uint64_t start_ns);
HYLE_TIME_C_API void hyle_time_wheel_destroy(hyle_time_wheel* wheel);
HYLE_TIME_C_API int hyle_time_wheel_schedule(hyle_time_wheel* wheel,
                                             hyle_time_timer* timer,
                                             uint64_t deadline_ns);
static inline int hyle_time_wheel_schedule_after(hyle_time_wheel* wheel,
                                                 hyle_time_timer* timer,
                                                 uint64_t timeout_ns) {
  return hyle_time_wheel_schedule(
      wheel, timer, hyle_time_deadline_after_ns(timeout_ns));
}
HYLE_TIME_C_API int hyle_time_wheel_cancel(hyle_time_wheel* wheel,
                                           hyle_time_timer* timer);
HYLE_TIME_C_API size_t hyle_time_wheel_advance(
    hyle_time_wheel* wheel, uint64_t now_ns, hyle_time_timer_visit_fn visit,
    void* ctx);
HYLE_TIME_C_API uint64_t
hyle_time_wheel_next_deadline_ns(const hyle_time_wheel* wheel);
HYLE_TIME_C_API size_t hyle_time_wheel_size(const hyle_time_wheel* wheel);
HYLE_TIME_C_API int hyle_time_wheel_empty(const hyle_time_wheel* wheel);

/* Fixed-duration ordered queue. schedule_after(now) is the normal entry point.
 * schedule(deadline) accepts an explicit absolute deadline but returns 0 when
 * it would violate non-decreasing order; a rejected reschedule preserves the
 * timer's prior arm. Destroy requires empty. */
HYLE_TIME_C_API hyle_time_common_queue* hyle_time_common_queue_create(
    uint64_t start_ns, uint64_t duration_ns);
HYLE_TIME_C_API void
hyle_time_common_queue_destroy(hyle_time_common_queue* queue);
HYLE_TIME_C_API int hyle_time_common_queue_schedule(
    hyle_time_common_queue* queue, hyle_time_timer* timer,
    uint64_t deadline_ns);
HYLE_TIME_C_API int hyle_time_common_queue_schedule_after(
    hyle_time_common_queue* queue, hyle_time_timer* timer, uint64_t now_ns);
static inline int hyle_time_common_queue_schedule_now(
    hyle_time_common_queue* queue, hyle_time_timer* timer) {
  return hyle_time_common_queue_schedule_after(
      queue, timer, hyle_time_monotonic_ns());
}
HYLE_TIME_C_API int hyle_time_common_queue_cancel(
    hyle_time_common_queue* queue, hyle_time_timer* timer);
HYLE_TIME_C_API size_t hyle_time_common_queue_advance(
    hyle_time_common_queue* queue, uint64_t now_ns,
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
