#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "hyle/c/sched.h"

#define CHECK(expr)                  \
  do {                               \
    if (!(expr)) return __LINE__;    \
  } while (0)

typedef struct task_ctx {
  hyle_sched_mutex* mutex;
  hyle_sched_wait_group* done;
  int* count;
} task_ctx;

static void count_task(void* raw) {
  task_ctx* ctx = (task_ctx*)raw;
  hyle_sched_mutex_lock(ctx->mutex);
  ++*ctx->count;
  hyle_sched_mutex_unlock(ctx->mutex);
  hyle_sched_yield();
  hyle_sched_wait_group_done(ctx->done);
}

static void count_inline_task(void* raw) {
  task_ctx* ctx = (task_ctx*)raw;
  hyle_sched_mutex_lock(ctx->mutex);
  ++*ctx->count;
  hyle_sched_mutex_unlock(ctx->mutex);
  hyle_sched_wait_group_done(ctx->done);
}

typedef struct fiber_ops_ctx {
  hyle_sched_wait_group* done;
  int worker;
  int saw_shutdown;
} fiber_ops_ctx;

static void fiber_ops_task(void* raw) {
  fiber_ops_ctx* ctx = (fiber_ops_ctx*)raw;
  ctx->worker = hyle_sched_current_worker_index();
  ctx->saw_shutdown = hyle_sched_shutdown_requested();
  hyle_sched_sleep_for_ns(UINT64_C(1000000));
  hyle_sched_sleep_until_ns(0);
  hyle_sched_wait_group_done(ctx->done);
}

typedef struct cv_ctx {
  hyle_sched_mutex* mutex;
  hyle_sched_cv* cv;
  hyle_sched_wait_group* done;
  int ready;
} cv_ctx;

static void cv_notifier(void* raw) {
  cv_ctx* ctx = (cv_ctx*)raw;
  hyle_sched_sleep_for_ns(UINT64_C(1000000));
  hyle_sched_mutex_lock(ctx->mutex);
  ctx->ready = 1;
  hyle_sched_cv_notify_one(ctx->cv);
  hyle_sched_mutex_unlock(ctx->mutex);
  hyle_sched_wait_group_done(ctx->done);
}

typedef struct latch_ctx {
  hyle_sched_latch* latch;
} latch_ctx;

static void latch_task(void* raw) {
  hyle_sched_latch_count_down(((latch_ctx*)raw)->latch, 1);
}

typedef struct blocking_ctx {
  hyle_sched_wait_group* done;
  int ran;
  int cancelled;
} blocking_ctx;

static void blocking_task(void* raw) { ((blocking_ctx*)raw)->ran = 1; }
static void blocking_done(void* raw, int cancelled) {
  blocking_ctx* ctx = (blocking_ctx*)raw;
  ctx->cancelled = cancelled;
  hyle_sched_wait_group_done(ctx->done);
}
static void cancelled(void* raw) { ++*(int*)raw; }

typedef struct tracked_allocator_state {
  hyle_allocator base;
  size_t live;
  int bad_free;
} tracked_allocator_state;

static void* tracked_alloc(void* raw, size_t size, size_t alignment) {
  tracked_allocator_state* state = (tracked_allocator_state*)raw;
  void* p = state->base.alloc(state->base.state, size, alignment);
  if (p != NULL) ++state->live;
  return p;
}
static void tracked_free(void* raw, void* ptr, size_t size,
                         size_t alignment) {
  tracked_allocator_state* state = (tracked_allocator_state*)raw;
  if (ptr != NULL) {
    if (state->live == 0) {
      state->bad_free = 1;
    } else {
      --state->live;
    }
  }
  state->base.free(state->base.state, ptr, size, alignment);
}
static void* tracked_realloc(void* raw, void* ptr, size_t old_size,
                             size_t new_size, size_t alignment) {
  tracked_allocator_state* state = (tracked_allocator_state*)raw;
  void* out = state->base.realloc(state->base.state, ptr, old_size, new_size,
                                  alignment);
  if (ptr == NULL && out != NULL) ++state->live;
  return out;
}

typedef struct suspend_ctx {
  hyle_allocator alloc;
  hyle_sched_wait_group* ready;
  hyle_sched_wait_group* done;
  hyle_sched_waker* waker;
} suspend_ctx;

static void suspend_task(void* raw) {
  suspend_ctx* ctx = (suspend_ctx*)raw;
  ctx->waker = hyle_sched_waker_capture(ctx->alloc);
  hyle_sched_wait_group_done(ctx->ready);
  hyle_sched_suspend();
  hyle_sched_wait_group_done(ctx->done);
}

typedef struct shutdown_ctx {
  hyle_sched_wait_group* ready;
  hyle_sched_wait_group* done;
  int observed;
} shutdown_ctx;

static void shutdown_task(void* raw) {
  shutdown_ctx* ctx = (shutdown_ctx*)raw;
  hyle_sched_wait_group_done(ctx->ready);
  hyle_sched_wait_for_shutdown();
  ctx->observed = hyle_sched_shutdown_requested();
  hyle_sched_wait_group_done(ctx->done);
}

int main(void) {
  hyle_allocator alloc = hyle_system_allocator();
  tracked_allocator_state tracked = {alloc, 0, 0};
  hyle_allocator state_alloc = {
      tracked_alloc, tracked_free, tracked_realloc, &tracked};

  /* Versioned options: an older prefix keeps defaults for absent fields. */
  hyle_sched_options prefix = HYLE_SCHED_OPTIONS_INIT;
  prefix.struct_size = (uint32_t)offsetof(hyle_sched_options, queue_capacity);
  prefix.num_workers = 1;
  hyle_sched_scheduler* prefix_scheduler =
      hyle_sched_scheduler_create(&prefix);
  CHECK(prefix_scheduler != NULL);
  CHECK(hyle_sched_num_workers(prefix_scheduler) == 1);
  hyle_sched_scheduler_destroy(prefix_scheduler);

  hyle_sched_options automatic = HYLE_SCHED_OPTIONS_INIT;
  hyle_sched_scheduler* automatic_scheduler =
      hyle_sched_scheduler_create(&automatic);
  CHECK(automatic_scheduler != NULL);
  CHECK(hyle_sched_num_workers(automatic_scheduler) >= 1);
  hyle_sched_scheduler_destroy(automatic_scheduler);

  hyle_sched_options invalid = HYLE_SCHED_OPTIONS_INIT;
  invalid.struct_size = 0;
  CHECK(hyle_sched_scheduler_create(&invalid) == NULL);
  CHECK(hyle_sched_last_error() != NULL);
  CHECK(strlen(hyle_sched_last_error()) != 0);

  hyle_sched_options options = hyle_sched_options_default();
  options.num_workers = 2;
  options.queue_capacity = 256;
  hyle_sched_scheduler* scheduler = hyle_sched_scheduler_create(&options);
  CHECK(scheduler != NULL);
  CHECK(hyle_sched_num_workers(scheduler) == 2);
  CHECK(hyle_sched_accepting(scheduler));
  CHECK(strlen(hyle_sched_last_error()) == 0);

  hyle_sched_mutex* mutex = hyle_sched_mutex_create(alloc);
  hyle_sched_wait_group* group = hyle_sched_wait_group_create(alloc, 32);
  CHECK(mutex != NULL && group != NULL);
  CHECK(hyle_sched_mutex_try_lock(mutex));
  hyle_sched_mutex_unlock(mutex);
  int count = 0;
  task_ctx task = {mutex, group, &count};
  for (int i = 0; i < 32; ++i)
    CHECK(hyle_sched_spawn(scheduler, count_task, &task));
  hyle_sched_wait_group_wait(group);
  hyle_sched_drain(scheduler);
  CHECK(count == 32);

  hyle_sched_wait_group* inline_group =
      hyle_sched_wait_group_create(alloc, 32);
  task_ctx inline_task = {mutex, inline_group, &count};
  CHECK(inline_group != NULL);
  for (int i = 0; i < 32; ++i)
    CHECK(hyle_sched_schedule(scheduler, count_inline_task, &inline_task));
  hyle_sched_wait_group_wait(inline_group);
  hyle_sched_drain(scheduler);
  CHECK(count == 64);

  hyle_sched_wait_group* pinned_done =
      hyle_sched_wait_group_create(alloc, 1);
  fiber_ops_ctx fiber_ops = {pinned_done, -1, -1};
  CHECK(hyle_sched_spawn_on(scheduler, 0, fiber_ops_task, &fiber_ops));
  hyle_sched_wait_group_wait(pinned_done);
  CHECK(fiber_ops.worker == 0 && fiber_ops.saw_shutdown == 0);
  CHECK(hyle_sched_drain_for(scheduler, UINT64_C(1000000000)));

  hyle_sched_wait_group* suspend_ready =
      hyle_sched_wait_group_create(alloc, 1);
  hyle_sched_wait_group* suspend_done =
      hyle_sched_wait_group_create(alloc, 1);
  suspend_ctx suspended = {alloc, suspend_ready, suspend_done, NULL};
  CHECK(hyle_sched_spawn(scheduler, suspend_task, &suspended));
  hyle_sched_wait_group_wait(suspend_ready);
  CHECK(suspended.waker != NULL);
  CHECK(hyle_sched_waker_wake(suspended.waker));
  CHECK(!hyle_sched_waker_wake(suspended.waker));
  hyle_sched_wait_group_wait(suspend_done);
  hyle_sched_waker_destroy(suspended.waker);

  hyle_sched_cv* cv = hyle_sched_cv_create(alloc);
  hyle_sched_wait_group* cv_done = hyle_sched_wait_group_create(alloc, 1);
  cv_ctx cv_state = {mutex, cv, cv_done, 0};
  CHECK(cv != NULL && cv_done != NULL);
  hyle_sched_mutex_lock(mutex);
  CHECK(hyle_sched_cv_wait_for(cv, mutex, UINT64_C(1000000)) == 1);
  hyle_sched_mutex_unlock(mutex);
  hyle_sched_mutex_lock(mutex);
  CHECK(hyle_sched_spawn(scheduler, cv_notifier, &cv_state));
  while (!cv_state.ready) hyle_sched_cv_wait(cv, mutex);
  hyle_sched_mutex_unlock(mutex);
  hyle_sched_wait_group_wait(cv_done);
  hyle_sched_cv_notify_all(cv);

  hyle_sched_wait_group* add_group =
      hyle_sched_wait_group_create(alloc, 0);
  CHECK(add_group != NULL);
  CHECK(hyle_sched_wait_group_add(add_group, 1));
  CHECK(hyle_sched_wait_group_done(add_group));
  CHECK(!hyle_sched_wait_group_done(add_group));
  CHECK(!hyle_sched_wait_group_add(add_group, -1));
  hyle_sched_wait_group_wait(add_group);

  hyle_sched_latch* latch = hyle_sched_latch_create(alloc, 1);
  latch_ctx latch_state = {latch};
  CHECK(latch != NULL && !hyle_sched_latch_try_wait(latch));
  CHECK(hyle_sched_spawn(scheduler, latch_task, &latch_state));
  hyle_sched_latch_wait(latch);
  CHECK(hyle_sched_latch_try_wait(latch));

  hyle_sched_semaphore* semaphore =
      hyle_sched_semaphore_create(alloc, 2);
  CHECK(semaphore != NULL);
  CHECK(hyle_sched_semaphore_available(semaphore) == 2);
  CHECK(hyle_sched_semaphore_try_acquire(semaphore, 1));
  CHECK(hyle_sched_semaphore_acquire(semaphore, 1));
  CHECK(!hyle_sched_semaphore_try_acquire(semaphore, 1));
  CHECK(hyle_sched_semaphore_release(semaphore, 2));
  hyle_sched_semaphore_close(semaphore);
  CHECK(!hyle_sched_semaphore_acquire(semaphore, 1));

  hyle_sched_rwlock* rwlock = hyle_sched_rwlock_create(alloc);
  CHECK(rwlock != NULL);
  CHECK(hyle_sched_rwlock_try_read_lock(rwlock));
  hyle_sched_rwlock_read_unlock(rwlock);
  hyle_sched_rwlock_read_lock(rwlock);
  CHECK(!hyle_sched_rwlock_try_write_lock(rwlock));
  hyle_sched_rwlock_read_unlock(rwlock);
  hyle_sched_rwlock_write_lock(rwlock);
  CHECK(!hyle_sched_rwlock_try_read_lock(rwlock));
  hyle_sched_rwlock_write_unlock(rwlock);

  hyle_sched_channel* channel = hyle_sched_channel_create(alloc, 2);
  int a = 1;
  int b = 2;
  void* out = NULL;
  CHECK(channel != NULL);
  CHECK(hyle_sched_channel_try_recv(channel, &out) ==
        HYLE_SCHED_RECV_EMPTY);
  CHECK(hyle_sched_channel_try_send(channel, &a) ==
        HYLE_SCHED_SEND_SENT);
  CHECK(hyle_sched_channel_try_send(channel, &b) ==
        HYLE_SCHED_SEND_SENT);
  CHECK(hyle_sched_channel_try_send(channel, &b) ==
        HYLE_SCHED_SEND_FULL);
  CHECK(hyle_sched_channel_send_for(channel, &b, UINT64_C(1000000)) ==
        HYLE_SCHED_SEND_FULL);
  CHECK(hyle_sched_channel_recv_for(channel, &out, UINT64_C(1000000)) ==
            HYLE_SCHED_RECV_VALUE &&
        out == &a);
  CHECK(hyle_sched_channel_recv(channel, &out) && out == &b);
  hyle_sched_channel_close(channel);
  CHECK(hyle_sched_channel_try_recv(channel, &out) ==
        HYLE_SCHED_RECV_CLOSED);

  hyle_sched_oneshot_sender* oneshot_tx = NULL;
  hyle_sched_oneshot_receiver* oneshot_rx = NULL;
  CHECK(hyle_sched_oneshot_create(state_alloc, &oneshot_tx, &oneshot_rx));
  CHECK(hyle_sched_oneshot_try_recv(oneshot_rx, &out) ==
        HYLE_SCHED_RECV_EMPTY);
  CHECK(!hyle_sched_oneshot_sender_is_closed(oneshot_tx));
  CHECK(hyle_sched_oneshot_send(oneshot_tx, &a));
  CHECK(hyle_sched_oneshot_recv(oneshot_rx, &out) ==
        HYLE_SCHED_RECV_VALUE);
  CHECK(out == &a);
  hyle_sched_oneshot_sender_destroy(oneshot_tx);
  hyle_sched_oneshot_receiver_destroy(oneshot_rx);
  oneshot_tx = NULL;
  oneshot_rx = NULL;
  CHECK(hyle_sched_oneshot_create(state_alloc, &oneshot_tx, &oneshot_rx));
  hyle_sched_oneshot_receiver_destroy(oneshot_rx);
  CHECK(hyle_sched_oneshot_sender_is_closed(oneshot_tx));
  hyle_sched_oneshot_sender_wait_closed(oneshot_tx);
  hyle_sched_oneshot_sender_destroy(oneshot_tx);

  hyle_sched_watch_sender* watch_tx = NULL;
  hyle_sched_watch_receiver* watch_rx = NULL;
  CHECK(hyle_sched_watch_create(
      state_alloc, &a, &watch_tx, &watch_rx));
  CHECK(hyle_sched_watch_sender_borrow(watch_tx) == &a);
  CHECK(hyle_sched_watch_receiver_borrow(watch_rx) == &a);
  CHECK(!hyle_sched_watch_has_changed(watch_rx));
  hyle_sched_watch_sender* watch_tx2 =
      hyle_sched_watch_sender_clone(watch_tx, state_alloc);
  hyle_sched_watch_receiver* watch_rx2 =
      hyle_sched_watch_receiver_clone(watch_rx, state_alloc);
  hyle_sched_watch_receiver* watch_sub =
      hyle_sched_watch_subscribe(watch_tx, state_alloc);
  CHECK(watch_tx2 != NULL && watch_rx2 != NULL && watch_sub != NULL);
  CHECK(hyle_sched_watch_receiver_count(watch_tx) == 3);
  CHECK(hyle_sched_watch_send(watch_tx, &b));
  CHECK(hyle_sched_watch_changed(watch_rx));
  CHECK(hyle_sched_watch_receiver_borrow_and_update(watch_rx) == &b);
  CHECK(hyle_sched_watch_send_replace(watch_tx, &a) == &b);
  CHECK(hyle_sched_watch_sender_borrow(watch_tx2) == &a);
  hyle_sched_watch_receiver_destroy(watch_sub);
  hyle_sched_watch_receiver_destroy(watch_rx2);
  hyle_sched_watch_sender_destroy(watch_tx2);
  hyle_sched_watch_receiver_destroy(watch_rx);
  hyle_sched_watch_sender_destroy(watch_tx);

  hyle_sched_broadcast_sender* broadcast_tx = NULL;
  hyle_sched_broadcast_receiver* broadcast_rx = NULL;
  CHECK(hyle_sched_broadcast_create(
      state_alloc, 2, &broadcast_tx, &broadcast_rx));
  hyle_sched_broadcast_sender* broadcast_tx2 =
      hyle_sched_broadcast_sender_clone(broadcast_tx, state_alloc);
  hyle_sched_broadcast_receiver* broadcast_rx2 =
      hyle_sched_broadcast_receiver_clone(broadcast_rx, state_alloc);
  hyle_sched_broadcast_receiver* broadcast_sub =
      hyle_sched_broadcast_subscribe(broadcast_tx, state_alloc);
  CHECK(broadcast_tx2 != NULL && broadcast_rx2 != NULL &&
        broadcast_sub != NULL);
  CHECK(hyle_sched_broadcast_sender_count(broadcast_tx) == 2);
  CHECK(hyle_sched_broadcast_receiver_count(broadcast_tx) == 3);
  CHECK(hyle_sched_broadcast_try_recv(broadcast_rx, &out) ==
        HYLE_SCHED_RECV_EMPTY);
  CHECK(hyle_sched_broadcast_send(broadcast_tx, &a));
  CHECK(hyle_sched_broadcast_try_recv(broadcast_rx, &out) ==
        HYLE_SCHED_RECV_VALUE);
  CHECK(out == &a);
  CHECK(hyle_sched_broadcast_recv(broadcast_rx2, &out) ==
        HYLE_SCHED_RECV_VALUE);
  CHECK(hyle_sched_broadcast_recv(broadcast_sub, &out) ==
        HYLE_SCHED_RECV_VALUE);
  hyle_sched_broadcast_receiver_destroy(broadcast_sub);
  hyle_sched_broadcast_receiver_destroy(broadcast_rx2);
  hyle_sched_broadcast_sender_destroy(broadcast_tx2);
  hyle_sched_broadcast_receiver_destroy(broadcast_rx);
  hyle_sched_broadcast_sender_destroy(broadcast_tx);

  hyle_sched_cancel_token* root = hyle_sched_cancel_root(state_alloc);
  hyle_sched_cancel_token* child =
      hyle_sched_cancel_derive(root, state_alloc);
  int fired = 0;
  hyle_sched_cancel_callback* callback =
      hyle_sched_cancel_callback_create(
          child, cancelled, &fired, state_alloc);
  CHECK(root != NULL && child != NULL && callback != NULL);
  hyle_sched_cancel(root);
  CHECK(fired == 1 && hyle_sched_is_cancelled(child));
  hyle_sched_cancel_callback_destroy(callback);
  hyle_sched_cancel_token_destroy(child);
  hyle_sched_cancel_token_destroy(root);
  CHECK(!tracked.bad_free && tracked.live == 0);

  hyle_sched_blocking_pool* pool =
      hyle_sched_blocking_pool_create(
          alloc, 2, UINT64_C(1000000000));
  hyle_sched_wait_group* blocking_group =
      hyle_sched_wait_group_create(alloc, 1);
  blocking_ctx blocking = {blocking_group, 0, -1};
  CHECK(pool != NULL && blocking_group != NULL);
  CHECK(hyle_sched_blocking_submit(
      pool, blocking_task, blocking_done, &blocking));
  hyle_sched_wait_group_wait(blocking_group);
  CHECK(blocking.ran == 1 && blocking.cancelled == 0);
  CHECK(hyle_sched_blocking_live_threads(pool) <= 2);
  CHECK(hyle_sched_blocking_retained_thread_handles(pool) <= 2);
  hyle_sched_blocking_shutdown(pool);
  CHECK(hyle_sched_blocking_live_threads(pool) == 0);

  hyle_sched_wait_group* shutdown_ready =
      hyle_sched_wait_group_create(alloc, 1);
  hyle_sched_wait_group* shutdown_done =
      hyle_sched_wait_group_create(alloc, 1);
  shutdown_ctx shutdown_state = {shutdown_ready, shutdown_done, 0};
  CHECK(hyle_sched_spawn(scheduler, shutdown_task, &shutdown_state));
  hyle_sched_wait_group_wait(shutdown_ready);
  hyle_sched_begin_shutdown(scheduler);
  hyle_sched_wait_group_wait(shutdown_done);
  CHECK(shutdown_state.observed);
  CHECK(!hyle_sched_accepting(scheduler));
  CHECK(!hyle_sched_spawn(scheduler, count_task, &task));
  CHECK(hyle_sched_drain_for(scheduler, UINT64_C(1000000000)));
  hyle_sched_shutdown(scheduler);

  /* Every opaque-handle function is NULL-safe at the FFI system boundary. */
  hyle_sched_scheduler_destroy(NULL);
  hyle_sched_waker_destroy(NULL);
  hyle_sched_mutex_lock(NULL);
  CHECK(!hyle_sched_mutex_try_lock(NULL));
  hyle_sched_mutex_unlock(NULL);
  hyle_sched_cv_wait(NULL, NULL);
  CHECK(!hyle_sched_cv_wait_for(NULL, NULL, 0));
  hyle_sched_cv_notify_one(NULL);
  hyle_sched_cv_notify_all(NULL);
  CHECK(!hyle_sched_wait_group_add(NULL, 0));
  CHECK(!hyle_sched_wait_group_done(NULL));
  hyle_sched_wait_group_wait(NULL);
  CHECK(!hyle_sched_latch_count_down(NULL, 0));
  CHECK(!hyle_sched_latch_try_wait(NULL));
  hyle_sched_latch_wait(NULL);
  CHECK(!hyle_sched_semaphore_acquire(NULL, 1));
  CHECK(!hyle_sched_semaphore_try_acquire(NULL, 1));
  CHECK(!hyle_sched_semaphore_release(NULL, 0));
  hyle_sched_semaphore_close(NULL);
  CHECK(hyle_sched_semaphore_available(NULL) == 0);
  hyle_sched_rwlock_read_lock(NULL);
  CHECK(!hyle_sched_rwlock_try_read_lock(NULL));
  hyle_sched_rwlock_read_unlock(NULL);
  hyle_sched_rwlock_write_lock(NULL);
  CHECK(!hyle_sched_rwlock_try_write_lock(NULL));
  hyle_sched_rwlock_write_unlock(NULL);
  hyle_sched_channel_close(NULL);
  hyle_sched_oneshot_sender_destroy(NULL);
  hyle_sched_oneshot_receiver_destroy(NULL);
  CHECK(!hyle_sched_oneshot_send(NULL, NULL));
  CHECK(hyle_sched_oneshot_recv(NULL, NULL) == HYLE_SCHED_RECV_CLOSED);
  CHECK(hyle_sched_oneshot_try_recv(NULL, NULL) ==
        HYLE_SCHED_RECV_CLOSED);
  CHECK(hyle_sched_oneshot_sender_is_closed(NULL));
  hyle_sched_oneshot_sender_wait_closed(NULL);
  hyle_sched_watch_sender_destroy(NULL);
  hyle_sched_watch_receiver_destroy(NULL);
  CHECK(!hyle_sched_watch_send(NULL, NULL));
  CHECK(hyle_sched_watch_send_replace(NULL, NULL) == NULL);
  CHECK(hyle_sched_watch_sender_borrow(NULL) == NULL);
  CHECK(hyle_sched_watch_receiver_borrow(NULL) == NULL);
  CHECK(hyle_sched_watch_receiver_borrow_and_update(NULL) == NULL);
  CHECK(!hyle_sched_watch_changed(NULL));
  CHECK(!hyle_sched_watch_has_changed(NULL));
  CHECK(hyle_sched_watch_receiver_count(NULL) == 0);
  hyle_sched_broadcast_sender_destroy(NULL);
  hyle_sched_broadcast_receiver_destroy(NULL);
  CHECK(!hyle_sched_broadcast_send(NULL, NULL));
  CHECK(hyle_sched_broadcast_recv(NULL, NULL) == HYLE_SCHED_RECV_CLOSED);
  CHECK(hyle_sched_broadcast_sender_count(NULL) == 0);
  CHECK(hyle_sched_broadcast_receiver_count(NULL) == 0);
  hyle_sched_cancel(NULL);
  CHECK(!hyle_sched_is_cancelled(NULL));
  hyle_sched_blocking_shutdown(NULL);
  CHECK(hyle_sched_blocking_live_threads(NULL) == 0);
  CHECK(hyle_sched_blocking_retained_thread_handles(NULL) == 0);

  hyle_sched_blocking_pool_destroy(pool);
  hyle_sched_wait_group_destroy(shutdown_done);
  hyle_sched_wait_group_destroy(shutdown_ready);
  hyle_sched_wait_group_destroy(blocking_group);
  hyle_sched_cancel_callback_destroy(NULL);
  hyle_sched_cancel_token_destroy(NULL);
  hyle_sched_channel_destroy(channel);
  hyle_sched_rwlock_destroy(rwlock);
  hyle_sched_semaphore_destroy(semaphore);
  hyle_sched_latch_destroy(latch);
  hyle_sched_wait_group_destroy(add_group);
  hyle_sched_wait_group_destroy(cv_done);
  hyle_sched_cv_destroy(cv);
  hyle_sched_wait_group_destroy(suspend_done);
  hyle_sched_wait_group_destroy(suspend_ready);
  hyle_sched_wait_group_destroy(pinned_done);
  hyle_sched_wait_group_destroy(inline_group);
  hyle_sched_wait_group_destroy(group);
  hyle_sched_mutex_destroy(mutex);
  hyle_sched_scheduler_destroy(scheduler);
  return 0;
}
