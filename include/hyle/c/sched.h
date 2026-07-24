/* SPDX: same as HYLE.
 *
 * Pure C ABI for the HYLE scheduler. C++ fibers, coroutines and templates never
 * cross this boundary; callbacks and opaque handles are the type-erased seam.
 * Docs: docs/guide/c-sched.md
 */
#ifndef HYLE_C_SCHED_H_
#define HYLE_C_SCHED_H_

#include <stddef.h>
#include <stdint.h>

#include "hyle/c/mem.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(HYLE_STATIC)
#  define HYLE_SCHED_C_API
#elif defined(_WIN32) || defined(__CYGWIN__)
#  if defined(HYLE_SCHED_BUILD)
#    define HYLE_SCHED_C_API __declspec(dllexport)
#  else
#    define HYLE_SCHED_C_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define HYLE_SCHED_C_API __attribute__((visibility("default")))
#else
#  define HYLE_SCHED_C_API
#endif

HYLE_SCHED_C_API const char* hyle_sched_last_error(void);

typedef struct hyle_sched_scheduler_o hyle_sched_scheduler;
typedef struct hyle_sched_waker_o hyle_sched_waker;
typedef struct hyle_sched_mutex_o hyle_sched_mutex;
typedef struct hyle_sched_cv_o hyle_sched_cv;
typedef struct hyle_sched_wait_group_o hyle_sched_wait_group;
typedef struct hyle_sched_latch_o hyle_sched_latch;
typedef struct hyle_sched_semaphore_o hyle_sched_semaphore;
typedef struct hyle_sched_rwlock_o hyle_sched_rwlock;
typedef struct hyle_sched_channel_o hyle_sched_channel;
typedef struct hyle_sched_oneshot_sender_o hyle_sched_oneshot_sender;
typedef struct hyle_sched_oneshot_receiver_o hyle_sched_oneshot_receiver;
typedef struct hyle_sched_watch_sender_o hyle_sched_watch_sender;
typedef struct hyle_sched_watch_receiver_o hyle_sched_watch_receiver;
typedef struct hyle_sched_broadcast_sender_o hyle_sched_broadcast_sender;
typedef struct hyle_sched_broadcast_receiver_o hyle_sched_broadcast_receiver;
typedef struct hyle_sched_cancel_token_o hyle_sched_cancel_token;
typedef struct hyle_sched_cancel_callback_o hyle_sched_cancel_callback;
typedef struct hyle_sched_blocking_pool_o hyle_sched_blocking_pool;

typedef void (*hyle_sched_task_fn)(void* ctx);
typedef void (*hyle_sched_cancel_fn)(void* ctx);
typedef void (*hyle_sched_blocking_done_fn)(void* ctx, int cancelled);

typedef struct hyle_sched_options {
  uint32_t struct_size;
  uint32_t num_workers; /* 0 = process-available CPUs, capped at 1024 */
  size_t queue_capacity;
  size_t fiber_stack_size;
  hyle_allocator backing;
} hyle_sched_options;

#define HYLE_SCHED_OPTIONS_INIT                                                \
  { (uint32_t)sizeof(hyle_sched_options), 0, 1024, 64 * 1024,                 \
    { NULL, NULL, NULL, NULL } }

HYLE_SCHED_C_API hyle_sched_options hyle_sched_options_default(void);
HYLE_SCHED_C_API hyle_sched_scheduler* hyle_sched_scheduler_create(
    const hyle_sched_options* options);
HYLE_SCHED_C_API void hyle_sched_scheduler_destroy(
    hyle_sched_scheduler* scheduler);
/* 1 = admitted, 0 = closed/invalid. The callback and ctx must remain valid until
 * the callback returns; HYLE owns the backing Fiber record. */
HYLE_SCHED_C_API int hyle_sched_spawn(hyle_sched_scheduler* scheduler,
                                      hyle_sched_task_fn task, void* ctx);
/* Lightweight callback that runs on a worker stack and therefore MUST NOT call
 * yield/suspend/sleep or a blocking fiber wait. */
HYLE_SCHED_C_API int hyle_sched_schedule(hyle_sched_scheduler* scheduler,
                                         hyle_sched_task_fn task, void* ctx);
HYLE_SCHED_C_API int hyle_sched_spawn_on(hyle_sched_scheduler* scheduler,
                                         uint32_t worker,
                                         hyle_sched_task_fn task, void* ctx);
HYLE_SCHED_C_API uint32_t hyle_sched_num_workers(
    const hyle_sched_scheduler* scheduler);
HYLE_SCHED_C_API int hyle_sched_accepting(
    const hyle_sched_scheduler* scheduler);
HYLE_SCHED_C_API void hyle_sched_begin_shutdown(
    hyle_sched_scheduler* scheduler);
HYLE_SCHED_C_API void hyle_sched_drain(hyle_sched_scheduler* scheduler);
HYLE_SCHED_C_API int hyle_sched_drain_for(hyle_sched_scheduler* scheduler,
                                          uint64_t timeout_ns);
HYLE_SCHED_C_API void hyle_sched_shutdown(hyle_sched_scheduler* scheduler);

/* Current-fiber operations. yield/suspend/sleep require a scheduler task
 * callback. Capture and publish a fresh waker before calling suspend. */
HYLE_SCHED_C_API int hyle_sched_current_worker_index(void);
HYLE_SCHED_C_API int hyle_sched_shutdown_requested(void);
HYLE_SCHED_C_API void hyle_sched_wait_for_shutdown(void);
HYLE_SCHED_C_API void hyle_sched_yield(void);
HYLE_SCHED_C_API void hyle_sched_suspend(void);
HYLE_SCHED_C_API void hyle_sched_sleep_for_ns(uint64_t ns);
HYLE_SCHED_C_API void hyle_sched_sleep_until_ns(uint64_t deadline_ns);

/* Capture inside a task, pass to an async backend, then wake exactly once.
 * Destroy after wake; destroying an unwoken waker does not resume its fiber. */
HYLE_SCHED_C_API hyle_sched_waker* hyle_sched_waker_capture(
    hyle_allocator backing);
HYLE_SCHED_C_API int hyle_sched_waker_wake(hyle_sched_waker* waker);
HYLE_SCHED_C_API void hyle_sched_waker_destroy(hyle_sched_waker* waker);

/* Mutex + condition variable. cv_wait atomically unlocks/relocks the mutex. */
HYLE_SCHED_C_API hyle_sched_mutex* hyle_sched_mutex_create(
    hyle_allocator backing);
HYLE_SCHED_C_API void hyle_sched_mutex_destroy(hyle_sched_mutex* mutex);
HYLE_SCHED_C_API void hyle_sched_mutex_lock(hyle_sched_mutex* mutex);
HYLE_SCHED_C_API int hyle_sched_mutex_try_lock(hyle_sched_mutex* mutex);
HYLE_SCHED_C_API void hyle_sched_mutex_unlock(hyle_sched_mutex* mutex);
HYLE_SCHED_C_API hyle_sched_cv* hyle_sched_cv_create(hyle_allocator backing);
HYLE_SCHED_C_API void hyle_sched_cv_destroy(hyle_sched_cv* cv);
HYLE_SCHED_C_API void hyle_sched_cv_wait(hyle_sched_cv* cv,
                                         hyle_sched_mutex* mutex);
HYLE_SCHED_C_API int hyle_sched_cv_wait_for(hyle_sched_cv* cv,
                                            hyle_sched_mutex* mutex,
                                            uint64_t timeout_ns);
HYLE_SCHED_C_API void hyle_sched_cv_notify_one(hyle_sched_cv* cv);
HYLE_SCHED_C_API void hyle_sched_cv_notify_all(hyle_sched_cv* cv);

HYLE_SCHED_C_API hyle_sched_wait_group* hyle_sched_wait_group_create(
    hyle_allocator backing, long initial);
HYLE_SCHED_C_API void hyle_sched_wait_group_destroy(
    hyle_sched_wait_group* group);
HYLE_SCHED_C_API int hyle_sched_wait_group_add(hyle_sched_wait_group* group,
                                               long count);
HYLE_SCHED_C_API int hyle_sched_wait_group_done(
    hyle_sched_wait_group* group);
HYLE_SCHED_C_API void hyle_sched_wait_group_wait(
    hyle_sched_wait_group* group);

HYLE_SCHED_C_API hyle_sched_latch* hyle_sched_latch_create(
    hyle_allocator backing, long count);
HYLE_SCHED_C_API void hyle_sched_latch_destroy(hyle_sched_latch* latch);
HYLE_SCHED_C_API int hyle_sched_latch_count_down(hyle_sched_latch* latch,
                                                 long count);
HYLE_SCHED_C_API int hyle_sched_latch_try_wait(const hyle_sched_latch* latch);
HYLE_SCHED_C_API void hyle_sched_latch_wait(hyle_sched_latch* latch);

HYLE_SCHED_C_API hyle_sched_semaphore* hyle_sched_semaphore_create(
    hyle_allocator backing, size_t permits);
HYLE_SCHED_C_API void hyle_sched_semaphore_destroy(
    hyle_sched_semaphore* semaphore);
HYLE_SCHED_C_API int hyle_sched_semaphore_acquire(
    hyle_sched_semaphore* semaphore, size_t permits);
HYLE_SCHED_C_API int hyle_sched_semaphore_try_acquire(
    hyle_sched_semaphore* semaphore, size_t permits);
HYLE_SCHED_C_API int hyle_sched_semaphore_release(
    hyle_sched_semaphore* semaphore, size_t permits);
HYLE_SCHED_C_API void hyle_sched_semaphore_close(
    hyle_sched_semaphore* semaphore);
HYLE_SCHED_C_API size_t hyle_sched_semaphore_available(
    const hyle_sched_semaphore* semaphore);

HYLE_SCHED_C_API hyle_sched_rwlock* hyle_sched_rwlock_create(
    hyle_allocator backing);
HYLE_SCHED_C_API void hyle_sched_rwlock_destroy(hyle_sched_rwlock* lock);
HYLE_SCHED_C_API void hyle_sched_rwlock_read_lock(hyle_sched_rwlock* lock);
HYLE_SCHED_C_API int hyle_sched_rwlock_try_read_lock(hyle_sched_rwlock* lock);
HYLE_SCHED_C_API void hyle_sched_rwlock_read_unlock(hyle_sched_rwlock* lock);
HYLE_SCHED_C_API void hyle_sched_rwlock_write_lock(hyle_sched_rwlock* lock);
HYLE_SCHED_C_API int hyle_sched_rwlock_try_write_lock(hyle_sched_rwlock* lock);
HYLE_SCHED_C_API void hyle_sched_rwlock_write_unlock(hyle_sched_rwlock* lock);

/* Bounded MPMC pointer channel. HYLE never owns/dereferences payload pointers. */
HYLE_SCHED_C_API hyle_sched_channel* hyle_sched_channel_create(
    hyle_allocator backing, size_t capacity);
HYLE_SCHED_C_API void hyle_sched_channel_destroy(hyle_sched_channel* channel);
HYLE_SCHED_C_API int hyle_sched_channel_send(hyle_sched_channel* channel,
                                             void* value);
HYLE_SCHED_C_API int hyle_sched_channel_recv(hyle_sched_channel* channel,
                                             void** out);
HYLE_SCHED_C_API void hyle_sched_channel_close(hyle_sched_channel* channel);

/* Type-erased pointer-value messaging. Payload ownership always remains with the
 * caller; HYLE only copies pointer values. */
typedef enum hyle_sched_recv_result {
  HYLE_SCHED_RECV_LAGGED = -2,
  HYLE_SCHED_RECV_CLOSED = -1,
  HYLE_SCHED_RECV_EMPTY = 0,
  HYLE_SCHED_RECV_VALUE = 1
} hyle_sched_recv_result;
typedef enum hyle_sched_send_result {
  HYLE_SCHED_SEND_CLOSED = -1,
  HYLE_SCHED_SEND_FULL = 0,
  HYLE_SCHED_SEND_SENT = 1
} hyle_sched_send_result;

HYLE_SCHED_C_API hyle_sched_send_result hyle_sched_channel_try_send(
    hyle_sched_channel* channel, void* value);
HYLE_SCHED_C_API hyle_sched_send_result hyle_sched_channel_send_for(
    hyle_sched_channel* channel, void* value, uint64_t timeout_ns);
HYLE_SCHED_C_API hyle_sched_recv_result hyle_sched_channel_try_recv(
    hyle_sched_channel* channel, void** out);
HYLE_SCHED_C_API hyle_sched_recv_result hyle_sched_channel_recv_for(
    hyle_sched_channel* channel, void** out, uint64_t timeout_ns);

HYLE_SCHED_C_API int hyle_sched_oneshot_create(
    hyle_allocator backing, hyle_sched_oneshot_sender** out_sender,
    hyle_sched_oneshot_receiver** out_receiver);
HYLE_SCHED_C_API void hyle_sched_oneshot_sender_destroy(
    hyle_sched_oneshot_sender* sender);
HYLE_SCHED_C_API void hyle_sched_oneshot_receiver_destroy(
    hyle_sched_oneshot_receiver* receiver);
HYLE_SCHED_C_API int hyle_sched_oneshot_send(
    hyle_sched_oneshot_sender* sender, void* value);
HYLE_SCHED_C_API hyle_sched_recv_result hyle_sched_oneshot_recv(
    hyle_sched_oneshot_receiver* receiver, void** out);
HYLE_SCHED_C_API hyle_sched_recv_result hyle_sched_oneshot_try_recv(
    hyle_sched_oneshot_receiver* receiver, void** out);
HYLE_SCHED_C_API int hyle_sched_oneshot_sender_is_closed(
    const hyle_sched_oneshot_sender* sender);
HYLE_SCHED_C_API void hyle_sched_oneshot_sender_wait_closed(
    hyle_sched_oneshot_sender* sender);

HYLE_SCHED_C_API int hyle_sched_watch_create(
    hyle_allocator backing, void* initial,
    hyle_sched_watch_sender** out_sender,
    hyle_sched_watch_receiver** out_receiver);
HYLE_SCHED_C_API hyle_sched_watch_sender* hyle_sched_watch_sender_clone(
    const hyle_sched_watch_sender* sender, hyle_allocator backing);
HYLE_SCHED_C_API void hyle_sched_watch_sender_destroy(
    hyle_sched_watch_sender* sender);
HYLE_SCHED_C_API hyle_sched_watch_receiver* hyle_sched_watch_receiver_clone(
    const hyle_sched_watch_receiver* receiver, hyle_allocator backing);
HYLE_SCHED_C_API hyle_sched_watch_receiver* hyle_sched_watch_subscribe(
    const hyle_sched_watch_sender* sender, hyle_allocator backing);
HYLE_SCHED_C_API void hyle_sched_watch_receiver_destroy(
    hyle_sched_watch_receiver* receiver);
HYLE_SCHED_C_API int hyle_sched_watch_send(
    hyle_sched_watch_sender* sender, void* value);
HYLE_SCHED_C_API void* hyle_sched_watch_send_replace(
    hyle_sched_watch_sender* sender, void* value);
HYLE_SCHED_C_API void* hyle_sched_watch_sender_borrow(
    const hyle_sched_watch_sender* sender);
HYLE_SCHED_C_API void* hyle_sched_watch_receiver_borrow(
    const hyle_sched_watch_receiver* receiver);
HYLE_SCHED_C_API void* hyle_sched_watch_receiver_borrow_and_update(
    hyle_sched_watch_receiver* receiver);
HYLE_SCHED_C_API int hyle_sched_watch_changed(
    hyle_sched_watch_receiver* receiver);
HYLE_SCHED_C_API int hyle_sched_watch_has_changed(
    const hyle_sched_watch_receiver* receiver);
HYLE_SCHED_C_API size_t hyle_sched_watch_receiver_count(
    const hyle_sched_watch_sender* sender);

HYLE_SCHED_C_API int hyle_sched_broadcast_create(
    hyle_allocator backing, size_t capacity,
    hyle_sched_broadcast_sender** out_sender,
    hyle_sched_broadcast_receiver** out_receiver);
HYLE_SCHED_C_API hyle_sched_broadcast_sender*
hyle_sched_broadcast_sender_clone(
    const hyle_sched_broadcast_sender* sender, hyle_allocator backing);
HYLE_SCHED_C_API void hyle_sched_broadcast_sender_destroy(
    hyle_sched_broadcast_sender* sender);
HYLE_SCHED_C_API hyle_sched_broadcast_receiver*
hyle_sched_broadcast_receiver_clone(
    const hyle_sched_broadcast_receiver* receiver, hyle_allocator backing);
HYLE_SCHED_C_API hyle_sched_broadcast_receiver*
hyle_sched_broadcast_subscribe(
    const hyle_sched_broadcast_sender* sender, hyle_allocator backing);
HYLE_SCHED_C_API void hyle_sched_broadcast_receiver_destroy(
    hyle_sched_broadcast_receiver* receiver);
HYLE_SCHED_C_API int hyle_sched_broadcast_send(
    hyle_sched_broadcast_sender* sender, void* value);
HYLE_SCHED_C_API hyle_sched_recv_result hyle_sched_broadcast_recv(
    hyle_sched_broadcast_receiver* receiver, void** out);
HYLE_SCHED_C_API hyle_sched_recv_result hyle_sched_broadcast_try_recv(
    hyle_sched_broadcast_receiver* receiver, void** out);
HYLE_SCHED_C_API size_t hyle_sched_broadcast_sender_count(
    const hyle_sched_broadcast_sender* sender);
HYLE_SCHED_C_API size_t hyle_sched_broadcast_receiver_count(
    const hyle_sched_broadcast_sender* sender);

/* Cancellation tree topology/callback registration is owner-worker-only; the
 * atomic is_cancelled query may be called from any thread. */
HYLE_SCHED_C_API hyle_sched_cancel_token* hyle_sched_cancel_root(
    hyle_allocator backing);
HYLE_SCHED_C_API hyle_sched_cancel_token* hyle_sched_cancel_derive(
    hyle_sched_cancel_token* parent, hyle_allocator handle_backing);
HYLE_SCHED_C_API void hyle_sched_cancel_token_destroy(
    hyle_sched_cancel_token* token);
HYLE_SCHED_C_API void hyle_sched_cancel(hyle_sched_cancel_token* token);
HYLE_SCHED_C_API int hyle_sched_is_cancelled(
    const hyle_sched_cancel_token* token);
HYLE_SCHED_C_API hyle_sched_cancel_callback* hyle_sched_cancel_callback_create(
    const hyle_sched_cancel_token* token, hyle_sched_cancel_fn fn, void* ctx,
    hyle_allocator backing);
HYLE_SCHED_C_API void hyle_sched_cancel_callback_destroy(
    hyle_sched_cancel_callback* callback);

/* Separate blocking pool. done(ctx, cancelled) runs exactly once: on the pool
 * thread after a started task returns, or on the shutdown/submission caller when
 * an unstarted job is rejected or dropped. */
HYLE_SCHED_C_API hyle_sched_blocking_pool* hyle_sched_blocking_pool_create(
    hyle_allocator backing, uint32_t max_threads, uint64_t keep_alive_ns);
HYLE_SCHED_C_API void hyle_sched_blocking_pool_destroy(
    hyle_sched_blocking_pool* pool);
HYLE_SCHED_C_API int hyle_sched_blocking_submit(
    hyle_sched_blocking_pool* pool, hyle_sched_task_fn task,
    hyle_sched_blocking_done_fn done, void* ctx);
HYLE_SCHED_C_API void hyle_sched_blocking_shutdown(
    hyle_sched_blocking_pool* pool);
HYLE_SCHED_C_API uint32_t hyle_sched_blocking_live_threads(
    const hyle_sched_blocking_pool* pool);
HYLE_SCHED_C_API size_t hyle_sched_blocking_retained_thread_handles(
    const hyle_sched_blocking_pool* pool);

#ifdef __cplusplus
}
#endif

#endif /* HYLE_C_SCHED_H_ */
