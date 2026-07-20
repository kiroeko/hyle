#include "hyle/c/sched.h"

#include <atomic>
#include <bit>
#include <cstddef>
#include <cstring>
#include <new>
#include <type_traits>
#include <utility>

#include "hyle/mem/allocator.h"
#include "hyle/platform/thread.h"
#include "hyle/sched/blocking.h"
#include "hyle/sched/cancel.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"
#include "hyle/sched/sync.h"
#include "hyle/sched/timer.h"
#include "hyle/sched/waker.h"

namespace {

thread_local const char* g_last_error = "";
void clear_error() noexcept { g_last_error = ""; }
void set_error(const char* message) noexcept { g_last_error = message; }

bool valid_backing(const hyle_allocator& a) noexcept {
  return a.alloc != nullptr && a.free != nullptr && a.realloc != nullptr;
}
bool empty_backing(const hyle_allocator& a) noexcept {
  return a.state == nullptr && a.alloc == nullptr && a.free == nullptr &&
         a.realloc == nullptr;
}

void* c_alloc(void* state, std::size_t size, std::size_t align) noexcept {
  const auto* a = static_cast<const hyle_allocator*>(state);
  return a->alloc(a->state, size, align);
}
void c_free(void* state, void* ptr, std::size_t size,
            std::size_t align) noexcept {
  const auto* a = static_cast<const hyle_allocator*>(state);
  a->free(a->state, ptr, size, align);
}
void* c_realloc(void* state, void* ptr, std::size_t old_size,
                std::size_t new_size, std::size_t align) noexcept {
  const auto* a = static_cast<const hyle_allocator*>(state);
  return a->realloc(a->state, ptr, old_size, new_size, align);
}
hyle::mem::Allocator to_cpp(const hyle_allocator* a) noexcept {
  return hyle::mem::Allocator(&c_alloc, &c_free, &c_realloc,
                              const_cast<hyle_allocator*>(a));
}

// Shared C++ states (oneshot/watch/broadcast/cancel) retain their Allocator and
// may outlive the create call. Pointing that Allocator at a stack-local
// hyle_allocator would leave a dangling callback state. This bridge owns a copy
// until the final allocation made through it is released.
struct CAllocatorBridge {
  hyle_allocator backing;
  std::atomic<std::size_t> refs{1};  // factory lease + live allocations
};

void bridge_release(CAllocatorBridge* bridge) noexcept {
  if (bridge->refs.fetch_sub(1, std::memory_order_acq_rel) != 1) return;
  const hyle_allocator backing = bridge->backing;
  bridge->~CAllocatorBridge();
  backing.free(backing.state, bridge, sizeof(CAllocatorBridge),
               alignof(CAllocatorBridge));
}

void* bridge_alloc(void* state, std::size_t size, std::size_t align) noexcept {
  auto* bridge = static_cast<CAllocatorBridge*>(state);
  bridge->refs.fetch_add(1, std::memory_order_relaxed);
  void* p = bridge->backing.alloc(bridge->backing.state, size, align);
  if (p == nullptr) bridge_release(bridge);
  return p;
}
void bridge_free(void* state, void* ptr, std::size_t size,
                 std::size_t align) noexcept {
  auto* bridge = static_cast<CAllocatorBridge*>(state);
  bridge->backing.free(bridge->backing.state, ptr, size, align);
  bridge_release(bridge);
}
void* bridge_realloc(void* state, void* ptr, std::size_t old_size,
                     std::size_t new_size, std::size_t align) noexcept {
  auto* bridge = static_cast<CAllocatorBridge*>(state);
  // realloc(NULL, n) creates a new live allocation and therefore needs the same
  // bridge lease as bridge_alloc(). realloc(p, 0) releases one. A failed resize
  // of an existing block keeps both the block and its lease, per realloc.
  if (ptr == nullptr && new_size != 0) {
    bridge->refs.fetch_add(1, std::memory_order_relaxed);
    void* out = bridge->backing.realloc(
        bridge->backing.state, nullptr, 0, new_size, align);
    if (out == nullptr) bridge_release(bridge);
    return out;
  }
  void* out = bridge->backing.realloc(bridge->backing.state, ptr, old_size,
                                      new_size, align);
  if (ptr != nullptr && new_size == 0) bridge_release(bridge);
  return out;
}

CAllocatorBridge* make_bridge(const hyle_allocator& backing) noexcept {
  void* p = backing.alloc(backing.state, sizeof(CAllocatorBridge),
                          alignof(CAllocatorBridge));
  if (p == nullptr) {
    set_error("allocator bridge allocation failed");
    return nullptr;
  }
  return new (p) CAllocatorBridge{backing};
}

hyle::mem::Allocator bridge_allocator(CAllocatorBridge* bridge) noexcept {
  return hyle::mem::Allocator(&bridge_alloc, &bridge_free, &bridge_realloc,
                              bridge);
}

bool normalize(hyle_allocator* a) noexcept {
  if (empty_backing(*a)) {
    *a = hyle_system_allocator();
    return true;
  }
  if (valid_backing(*a)) return true;
  set_error("allocator must be zero-initialized or provide alloc/free/realloc");
  return false;
}

template <class T, class... Args>
T* make_handle(hyle_allocator backing, Args&&... args) noexcept {
  if (!normalize(&backing)) return nullptr;
  void* p = backing.alloc(backing.state, sizeof(T), alignof(T));
  if (p == nullptr) {
    set_error("sched handle allocation failed");
    return nullptr;
  }
  try {
    T* out = new (p) T(backing, std::forward<Args>(args)...);
    clear_error();
    return out;
  } catch (...) {
    backing.free(backing.state, p, sizeof(T), alignof(T));
    set_error("sched handle construction failed");
    return nullptr;
  }
}

template <class T>
void drop_handle(T* p) noexcept {
  if (p == nullptr) return;
  hyle_allocator backing = p->backing;
  p->~T();
  backing.free(backing.state, p, sizeof(T), alignof(T));
}

}  // namespace

static_assert(std::is_standard_layout_v<hyle_sched_options>);
static_assert(std::is_trivially_copyable_v<hyle_sched_options>);
static_assert(offsetof(hyle_sched_options, struct_size) == 0);

struct hyle_sched_scheduler_o {
  hyle_allocator backing;
  hyle::Status init_status;
  hyle::sched::Scheduler scheduler;
  std::size_t stack_size;
  hyle_sched_scheduler_o(hyle_allocator b, unsigned workers,
                         std::size_t capacity, std::size_t stack)
      : backing(b),
        scheduler(workers, to_cpp(&backing), capacity, &init_status),
        stack_size(stack) {}
};
struct hyle_sched_waker_o {
  hyle_allocator backing;
  hyle::sched::Waker waker;
  std::atomic<bool> fired{false};
  hyle_sched_waker_o(hyle_allocator b, hyle::sched::Waker w)
      : backing(b), waker(w) {}
};
#define HYLE_C_SIMPLE_HANDLE(name, type)             \
  struct name {                                      \
    hyle_allocator backing;                          \
    type value;                                      \
    template <class... Args>                         \
    name(hyle_allocator b, Args&&... args)           \
        : backing(b), value(std::forward<Args>(args)...) {} \
  }
HYLE_C_SIMPLE_HANDLE(hyle_sched_mutex_o, hyle::sched::Mutex);
HYLE_C_SIMPLE_HANDLE(hyle_sched_cv_o, hyle::sched::ConditionVariable);
HYLE_C_SIMPLE_HANDLE(hyle_sched_wait_group_o, hyle::sched::WaitGroup);
HYLE_C_SIMPLE_HANDLE(hyle_sched_latch_o, hyle::sched::Latch);
HYLE_C_SIMPLE_HANDLE(hyle_sched_semaphore_o, hyle::sched::Semaphore);
HYLE_C_SIMPLE_HANDLE(hyle_sched_rwlock_o, hyle::sched::RwLock);

struct hyle_sched_channel_o {
  hyle_allocator backing;
  hyle::sched::Channel<void*> value;
  hyle_sched_channel_o(hyle_allocator b, std::size_t capacity)
      : backing(b), value(to_cpp(&backing), capacity) {}
};
struct hyle_sched_oneshot_sender_o {
  hyle_allocator backing;
  hyle::sched::OneshotSender<void*> value;
  bool spent = false;
  hyle_sched_oneshot_sender_o(hyle_allocator b,
                              hyle::sched::OneshotSender<void*> sender)
      : backing(b), value(std::move(sender)) {}
};
struct hyle_sched_oneshot_receiver_o {
  hyle_allocator backing;
  hyle::sched::OneshotReceiver<void*> value;
  bool spent = false;
  hyle_sched_oneshot_receiver_o(hyle_allocator b,
                                hyle::sched::OneshotReceiver<void*> receiver)
      : backing(b), value(std::move(receiver)) {}
};
struct hyle_sched_watch_sender_o {
  hyle_allocator backing;
  hyle::sched::WatchSender<void*> value;
  hyle_sched_watch_sender_o(hyle_allocator b,
                            hyle::sched::WatchSender<void*> sender)
      : backing(b), value(std::move(sender)) {}
};
struct hyle_sched_watch_receiver_o {
  hyle_allocator backing;
  hyle::sched::WatchReceiver<void*> value;
  hyle_sched_watch_receiver_o(hyle_allocator b,
                              hyle::sched::WatchReceiver<void*> receiver)
      : backing(b), value(std::move(receiver)) {}
};
struct hyle_sched_broadcast_sender_o {
  hyle_allocator backing;
  hyle::sched::BroadcastSender<void*> value;
  hyle_sched_broadcast_sender_o(hyle_allocator b,
                                hyle::sched::BroadcastSender<void*> sender)
      : backing(b), value(std::move(sender)) {}
};
struct hyle_sched_broadcast_receiver_o {
  hyle_allocator backing;
  hyle::sched::BroadcastReceiver<void*> value;
  hyle_sched_broadcast_receiver_o(
      hyle_allocator b, hyle::sched::BroadcastReceiver<void*> receiver)
      : backing(b), value(std::move(receiver)) {}
};
struct hyle_sched_cancel_token_o {
  hyle_allocator backing;
  hyle::sched::CancelToken value;
  hyle_sched_cancel_token_o(hyle_allocator b, hyle::sched::CancelToken token)
      : backing(b), value(std::move(token)) {}
};
struct hyle_sched_cancel_callback_o {
  hyle_allocator backing;
  hyle_sched_cancel_fn fn;
  void* ctx;
  hyle::sched::CancelCallback value;
  hyle_sched_cancel_callback_o(hyle_allocator b,
                               const hyle::sched::CancelToken& token,
                               hyle_sched_cancel_fn callback, void* context)
      : backing(b),
        fn(callback),
        ctx(context),
        value(token, &invoke, this) {}
  static void invoke(void* p) noexcept {
    auto* self = static_cast<hyle_sched_cancel_callback_o*>(p);
    self->fn(self->ctx);
  }
};
struct hyle_sched_blocking_pool_o {
  hyle_allocator backing;
  hyle::sched::BlockingPool value;
  hyle_sched_blocking_pool_o(hyle_allocator b, unsigned max_threads,
                             std::uint64_t keep_alive)
      : backing(b), value(to_cpp(&backing), max_threads, keep_alive) {}
};

namespace {

int spawn_c(hyle_sched_scheduler* s, unsigned worker, bool pinned,
            hyle_sched_task_fn fn, void* ctx) noexcept {
  if (s == nullptr || fn == nullptr) {
    set_error("scheduler and task callback must be non-null");
    return 0;
  }
  if (pinned && worker >= s->scheduler.num_workers()) {
    set_error("worker index is out of range");
    return 0;
  }
  const hyle::Status status =
      pinned ? s->scheduler.spawn_detached_on(
                   worker, fn, ctx, s->stack_size)
             : s->scheduler.spawn_detached(fn, ctx, s->stack_size);
  if (!status.ok()) {
    set_error(status.IsCancelled() ? "scheduler admission is closed"
                                   : "detached task allocation failed");
    return 0;
  }
  clear_error();
  return 1;
}

hyle_sched_send_result channel_send_result(
    hyle::sched::Channel<void*>::SendStatus status) noexcept {
  using Status = hyle::sched::Channel<void*>::SendStatus;
  if (status == Status::kSent) return HYLE_SCHED_SEND_SENT;
  if (status == Status::kClosed) return HYLE_SCHED_SEND_CLOSED;
  return HYLE_SCHED_SEND_FULL;
}
hyle_sched_recv_result channel_recv_result(
    hyle::sched::Channel<void*>::RecvStatus status) noexcept {
  using Status = hyle::sched::Channel<void*>::RecvStatus;
  if (status == Status::kValue) return HYLE_SCHED_RECV_VALUE;
  if (status == Status::kClosed) return HYLE_SCHED_RECV_CLOSED;
  return HYLE_SCHED_RECV_EMPTY;
}

struct CBlockingJob {
  hyle::sched::detail::BlockingJob header;
  hyle::mem::Allocator alloc;
  hyle_sched_task_fn task;
  hyle_sched_blocking_done_fn done;
  void* ctx;

  CBlockingJob(hyle::mem::Allocator a, hyle_sched_task_fn t,
               hyle_sched_blocking_done_fn d, void* c) noexcept
      : alloc(a), task(t), done(d), ctx(c) {
    header.ctx = this;
    header.run_fn = &run;
    header.cancel_fn = &cancel;
    header.destroy_fn = &destroy;
  }
  static void run(void* p) noexcept {
    auto* self = static_cast<CBlockingJob*>(p);
    self->task(self->ctx);
    if (self->done != nullptr) self->done(self->ctx, 0);
  }
  static void cancel(void* p) noexcept {
    auto* self = static_cast<CBlockingJob*>(p);
    if (self->done != nullptr) self->done(self->ctx, 1);
  }
  static void destroy(void* p) noexcept {
    auto* self = static_cast<CBlockingJob*>(p);
    hyle::mem::Allocator a = self->alloc;
    self->~CBlockingJob();
    a.deallocate(self, sizeof(CBlockingJob), alignof(CBlockingJob));
  }
};

}  // namespace

extern "C" {

const char* hyle_sched_last_error(void) { return g_last_error; }

hyle_sched_options hyle_sched_options_default(void) {
  hyle_sched_options o = HYLE_SCHED_OPTIONS_INIT;
  const unsigned detected = hyle::platform::hardware_concurrency();
  o.num_workers = detected == 0 ? 1 : (detected > 1024 ? 1024 : detected);
  o.backing = hyle_system_allocator();
  return o;
}

hyle_sched_scheduler* hyle_sched_scheduler_create(
    const hyle_sched_options* options) {
  hyle_sched_options o = hyle_sched_options_default();
  if (options != nullptr) {
    if (options->struct_size < sizeof(options->struct_size)) {
      set_error("sched options struct_size is too small");
      return nullptr;
    }
    const std::size_t readable =
        options->struct_size < sizeof(o) ? options->struct_size : sizeof(o);
    // Start from current defaults and overlay only the prefix the caller knows.
    // This accepts older options structs while allowing future callers to append
    // fields without making this library read past its known layout.
    std::memcpy(&o, options, readable);
  }
  if (o.num_workers == 0) {
    const unsigned detected = hyle::platform::hardware_concurrency();
    o.num_workers = detected == 0 ? 1 : (detected > 1024 ? 1024 : detected);
  }
  if (o.num_workers > 1024 ||
      o.queue_capacity < 2 || o.queue_capacity > (size_t{1} << 31) ||
      !std::has_single_bit(o.queue_capacity) ||
      o.fiber_stack_size < hyle::mem::kFiberStackSmall) {
    set_error("invalid workers, power-of-two queue capacity, or fiber stack size");
    return nullptr;
  }
  auto* out = make_handle<hyle_sched_scheduler_o>(
      o.backing, o.num_workers, o.queue_capacity, o.fiber_stack_size);
  if (out != nullptr && !out->scheduler.valid()) {
    const bool unavailable = out->init_status.IsUnavailable();
    drop_handle(out);
    set_error(unavailable ? "scheduler worker thread creation failed"
                          : "scheduler allocation failed");
    return nullptr;
  }
  if (out != nullptr) clear_error();
  return out;
}
void hyle_sched_scheduler_destroy(hyle_sched_scheduler* s) { drop_handle(s); }
int hyle_sched_spawn(hyle_sched_scheduler* s, hyle_sched_task_fn fn, void* ctx) {
  return spawn_c(s, 0, false, fn, ctx);
}
int hyle_sched_schedule(hyle_sched_scheduler* s, hyle_sched_task_fn fn,
                        void* ctx) {
  if (s == nullptr || fn == nullptr) {
    set_error("scheduler and task callback must be non-null");
    return 0;
  }
  const hyle::Status status = s->scheduler.schedule(fn, ctx);
  if (!status.ok()) {
    set_error(status.IsCancelled() ? "scheduler admission is closed"
                                   : "scheduled task allocation failed");
    return 0;
  }
  clear_error();
  return 1;
}
int hyle_sched_spawn_on(hyle_sched_scheduler* s, uint32_t worker,
                        hyle_sched_task_fn fn, void* ctx) {
  return spawn_c(s, worker, true, fn, ctx);
}
uint32_t hyle_sched_num_workers(const hyle_sched_scheduler* s) {
  return s != nullptr ? s->scheduler.num_workers() : 0;
}
int hyle_sched_accepting(const hyle_sched_scheduler* s) {
  return s != nullptr && s->scheduler.accepting() ? 1 : 0;
}
void hyle_sched_begin_shutdown(hyle_sched_scheduler* s) {
  if (s) s->scheduler.begin_shutdown();
}
void hyle_sched_drain(hyle_sched_scheduler* s) {
  if (s) s->scheduler.drain();
}
int hyle_sched_drain_for(hyle_sched_scheduler* s, uint64_t ns) {
  return s != nullptr && s->scheduler.drain_for(ns) ? 1 : 0;
}
void hyle_sched_shutdown(hyle_sched_scheduler* s) {
  if (s) s->scheduler.shutdown();
}
int hyle_sched_current_worker_index(void) {
  return hyle::sched::current_worker_index();
}
int hyle_sched_shutdown_requested(void) {
  return hyle::sched::shutdown_requested() ? 1 : 0;
}
void hyle_sched_wait_for_shutdown(void) {
  hyle::sched::wait_for_shutdown();
}
void hyle_sched_yield(void) { hyle::sched::Fiber::yield(); }
void hyle_sched_suspend(void) { hyle::sched::suspend(); }
void hyle_sched_sleep_for_ns(uint64_t ns) { hyle::sched::sleep_for_ns(ns); }
void hyle_sched_sleep_until_ns(uint64_t ns) { hyle::sched::sleep_until_ns(ns); }

hyle_sched_waker* hyle_sched_waker_capture(hyle_allocator backing) {
  hyle::sched::Waker w = hyle::sched::current_waker();
  if (!w.valid()) {
    set_error("waker capture requires a running scheduler fiber");
    return nullptr;
  }
  auto* out = make_handle<hyle_sched_waker_o>(backing, w);
  if (out) clear_error();
  return out;
}
int hyle_sched_waker_wake(hyle_sched_waker* w) {
  if (w == nullptr) return 0;
  bool expected = false;
  if (!w->fired.compare_exchange_strong(expected, true,
                                        std::memory_order_acq_rel))
    return 0;
  w->waker.wake();
  return 1;
}
void hyle_sched_waker_destroy(hyle_sched_waker* w) { drop_handle(w); }

#define HYLE_C_CREATE_DROP(prefix, handle)                                    \
  handle* prefix##_create(hyle_allocator b) { return make_handle<handle>(b); } \
  void prefix##_destroy(handle* p) { drop_handle(p); }
HYLE_C_CREATE_DROP(hyle_sched_mutex, hyle_sched_mutex_o)
HYLE_C_CREATE_DROP(hyle_sched_cv, hyle_sched_cv_o)
HYLE_C_CREATE_DROP(hyle_sched_rwlock, hyle_sched_rwlock_o)

void hyle_sched_mutex_lock(hyle_sched_mutex* p) {
  if (p) p->value.lock();
}
int hyle_sched_mutex_try_lock(hyle_sched_mutex* p) {
  return p != nullptr && p->value.try_lock() ? 1 : 0;
}
void hyle_sched_mutex_unlock(hyle_sched_mutex* p) {
  if (p) p->value.unlock();
}
void hyle_sched_cv_wait(hyle_sched_cv* cv, hyle_sched_mutex* mu) {
  if (cv && mu) cv->value.wait(mu->value);
}
int hyle_sched_cv_wait_for(hyle_sched_cv* cv, hyle_sched_mutex* mu,
                           uint64_t ns) {
  if (cv == nullptr || mu == nullptr) return 0;
  return cv->value.wait_for(mu->value, ns) ==
                 hyle::sched::ConditionVariable::CvStatus::kTimeout
             ? 1
             : 0;
}
void hyle_sched_cv_notify_one(hyle_sched_cv* cv) {
  if (cv) cv->value.notify_one();
}
void hyle_sched_cv_notify_all(hyle_sched_cv* cv) {
  if (cv) cv->value.notify_all();
}

hyle_sched_wait_group* hyle_sched_wait_group_create(hyle_allocator b, long n) {
  if (n < 0) {
    set_error("wait group initial count must be non-negative");
    return nullptr;
  }
  return make_handle<hyle_sched_wait_group_o>(b, n);
}
void hyle_sched_wait_group_destroy(hyle_sched_wait_group* p) { drop_handle(p); }
int hyle_sched_wait_group_add(hyle_sched_wait_group* p, long n) {
  if (p == nullptr || !p->value.try_add(n)) {
    set_error("invalid wait group add");
    return 0;
  }
  clear_error();
  return 1;
}
int hyle_sched_wait_group_done(hyle_sched_wait_group* p) {
  if (p == nullptr || !p->value.try_done()) {
    set_error("wait group is already drained");
    return 0;
  }
  clear_error();
  return 1;
}
void hyle_sched_wait_group_wait(hyle_sched_wait_group* p) {
  if (p) p->value.wait();
}

hyle_sched_latch* hyle_sched_latch_create(hyle_allocator b, long n) {
  if (n < 0) {
    set_error("latch count must be non-negative");
    return nullptr;
  }
  return make_handle<hyle_sched_latch_o>(b, n);
}
void hyle_sched_latch_destroy(hyle_sched_latch* p) { drop_handle(p); }
int hyle_sched_latch_count_down(hyle_sched_latch* p, long n) {
  if (p == nullptr || !p->value.try_count_down(n)) {
    set_error("latch count_down exceeds the remaining count");
    return 0;
  }
  clear_error();
  return 1;
}
int hyle_sched_latch_try_wait(const hyle_sched_latch* p) {
  return p != nullptr && p->value.try_wait() ? 1 : 0;
}
void hyle_sched_latch_wait(hyle_sched_latch* p) {
  if (p) p->value.wait();
}

hyle_sched_semaphore* hyle_sched_semaphore_create(hyle_allocator b, size_t n) {
  return make_handle<hyle_sched_semaphore_o>(b, n);
}
void hyle_sched_semaphore_destroy(hyle_sched_semaphore* p) { drop_handle(p); }
int hyle_sched_semaphore_acquire(hyle_sched_semaphore* p, size_t n) {
  return p != nullptr && p->value.acquire(n) ? 1 : 0;
}
int hyle_sched_semaphore_try_acquire(hyle_sched_semaphore* p, size_t n) {
  return p != nullptr && p->value.try_acquire(n) ? 1 : 0;
}
int hyle_sched_semaphore_release(hyle_sched_semaphore* p, size_t n) {
  if (p == nullptr || !p->value.try_release(n)) {
    set_error("semaphore permit count overflow");
    return 0;
  }
  clear_error();
  return 1;
}
void hyle_sched_semaphore_close(hyle_sched_semaphore* p) {
  if (p) p->value.close();
}
size_t hyle_sched_semaphore_available(const hyle_sched_semaphore* p) {
  return p != nullptr ? p->value.available() : 0;
}

void hyle_sched_rwlock_read_lock(hyle_sched_rwlock* p) {
  if (p) p->value.lock_read();
}
int hyle_sched_rwlock_try_read_lock(hyle_sched_rwlock* p) {
  return p != nullptr && p->value.try_lock_read() ? 1 : 0;
}
void hyle_sched_rwlock_read_unlock(hyle_sched_rwlock* p) {
  if (p) p->value.unlock_read();
}
void hyle_sched_rwlock_write_lock(hyle_sched_rwlock* p) {
  if (p) p->value.lock_write();
}
int hyle_sched_rwlock_try_write_lock(hyle_sched_rwlock* p) {
  return p != nullptr && p->value.try_lock_write() ? 1 : 0;
}
void hyle_sched_rwlock_write_unlock(hyle_sched_rwlock* p) {
  if (p) p->value.unlock_write();
}

hyle_sched_channel* hyle_sched_channel_create(hyle_allocator b, size_t cap) {
  if (cap == 0 || cap > SIZE_MAX / sizeof(void*)) {
    set_error("invalid channel capacity");
    return nullptr;
  }
  auto* out = make_handle<hyle_sched_channel_o>(b, cap);
  if (out != nullptr && !out->value.valid()) {
    drop_handle(out);
    set_error("channel buffer allocation failed");
    return nullptr;
  }
  return out;
}
void hyle_sched_channel_destroy(hyle_sched_channel* p) { drop_handle(p); }
int hyle_sched_channel_send(hyle_sched_channel* p, void* v) {
  return p != nullptr && p->value.send(v) ? 1 : 0;
}
int hyle_sched_channel_recv(hyle_sched_channel* p, void** out) {
  if (p == nullptr || out == nullptr) return 0;
  return p->value.recv(*out) ? 1 : 0;
}
void hyle_sched_channel_close(hyle_sched_channel* p) {
  if (p) p->value.close();
}

hyle_sched_send_result hyle_sched_channel_try_send(
    hyle_sched_channel* p, void* value) {
  return p != nullptr ? channel_send_result(p->value.try_send(value))
                      : HYLE_SCHED_SEND_CLOSED;
}
hyle_sched_send_result hyle_sched_channel_send_for(
    hyle_sched_channel* p, void* value, uint64_t timeout_ns) {
  return p != nullptr
             ? channel_send_result(p->value.send_for(value, timeout_ns))
             : HYLE_SCHED_SEND_CLOSED;
}
hyle_sched_recv_result hyle_sched_channel_try_recv(
    hyle_sched_channel* p, void** out) {
  if (p == nullptr || out == nullptr) return HYLE_SCHED_RECV_CLOSED;
  return channel_recv_result(p->value.try_recv(*out));
}
hyle_sched_recv_result hyle_sched_channel_recv_for(
    hyle_sched_channel* p, void** out, uint64_t timeout_ns) {
  if (p == nullptr || out == nullptr) return HYLE_SCHED_RECV_CLOSED;
  return channel_recv_result(p->value.recv_for(*out, timeout_ns));
}

int hyle_sched_oneshot_create(
    hyle_allocator b, hyle_sched_oneshot_sender** out_sender,
    hyle_sched_oneshot_receiver** out_receiver) {
  if (out_sender == nullptr || out_receiver == nullptr) {
    set_error("oneshot output pointers must be non-null");
    return 0;
  }
  *out_sender = nullptr;
  *out_receiver = nullptr;
  if (!normalize(&b)) return 0;
  CAllocatorBridge* bridge = make_bridge(b);
  if (bridge == nullptr) return 0;
  auto pair =
      hyle::sched::try_make_oneshot<void*>(bridge_allocator(bridge));
  bridge_release(bridge);
  if (!pair.has_value()) {
    set_error("oneshot state allocation failed");
    return 0;
  }
  auto* sender = make_handle<hyle_sched_oneshot_sender_o>(
      b, std::move(pair->first));
  if (sender == nullptr) return 0;
  auto* receiver = make_handle<hyle_sched_oneshot_receiver_o>(
      b, std::move(pair->second));
  if (receiver == nullptr) {
    drop_handle(sender);
    return 0;
  }
  *out_sender = sender;
  *out_receiver = receiver;
  clear_error();
  return 1;
}
void hyle_sched_oneshot_sender_destroy(hyle_sched_oneshot_sender* p) {
  drop_handle(p);
}
void hyle_sched_oneshot_receiver_destroy(hyle_sched_oneshot_receiver* p) {
  drop_handle(p);
}
int hyle_sched_oneshot_send(hyle_sched_oneshot_sender* p, void* value) {
  if (p == nullptr || p->spent) return 0;
  p->spent = true;
  return p->value.send(value).has_value() ? 1 : 0;
}
hyle_sched_recv_result hyle_sched_oneshot_recv(
    hyle_sched_oneshot_receiver* p, void** out) {
  if (p == nullptr || out == nullptr || p->spent)
    return HYLE_SCHED_RECV_CLOSED;
  p->spent = true;
  auto result = p->value.recv();
  if (!result.has_value()) return HYLE_SCHED_RECV_CLOSED;
  *out = *result;
  return HYLE_SCHED_RECV_VALUE;
}
hyle_sched_recv_result hyle_sched_oneshot_try_recv(
    hyle_sched_oneshot_receiver* p, void** out) {
  if (p == nullptr || out == nullptr || p->spent)
    return HYLE_SCHED_RECV_CLOSED;
  auto result = p->value.try_recv();
  if (result.has_value()) {
    *out = *result;
    return HYLE_SCHED_RECV_VALUE;
  }
  return result.error().IsNotFound() ? HYLE_SCHED_RECV_EMPTY
                                     : HYLE_SCHED_RECV_CLOSED;
}
int hyle_sched_oneshot_sender_is_closed(
    const hyle_sched_oneshot_sender* p) {
  return p == nullptr || p->spent || p->value.is_closed() ? 1 : 0;
}
void hyle_sched_oneshot_sender_wait_closed(hyle_sched_oneshot_sender* p) {
  if (p != nullptr && !p->spent) p->value.closed();
}

int hyle_sched_watch_create(
    hyle_allocator b, void* initial, hyle_sched_watch_sender** out_sender,
    hyle_sched_watch_receiver** out_receiver) {
  if (out_sender == nullptr || out_receiver == nullptr) {
    set_error("watch output pointers must be non-null");
    return 0;
  }
  *out_sender = nullptr;
  *out_receiver = nullptr;
  if (!normalize(&b)) return 0;
  CAllocatorBridge* bridge = make_bridge(b);
  if (bridge == nullptr) return 0;
  auto pair =
      hyle::sched::try_make_watch<void*>(bridge_allocator(bridge), initial);
  bridge_release(bridge);
  if (!pair.has_value()) {
    set_error("watch state allocation failed");
    return 0;
  }
  auto* sender = make_handle<hyle_sched_watch_sender_o>(
      b, std::move(pair->first));
  if (sender == nullptr) return 0;
  auto* receiver = make_handle<hyle_sched_watch_receiver_o>(
      b, std::move(pair->second));
  if (receiver == nullptr) {
    drop_handle(sender);
    return 0;
  }
  *out_sender = sender;
  *out_receiver = receiver;
  clear_error();
  return 1;
}
hyle_sched_watch_sender* hyle_sched_watch_sender_clone(
    const hyle_sched_watch_sender* p, hyle_allocator b) {
  if (p == nullptr) {
    set_error("watch sender must be non-null");
    return nullptr;
  }
  return make_handle<hyle_sched_watch_sender_o>(b, p->value);
}
void hyle_sched_watch_sender_destroy(hyle_sched_watch_sender* p) {
  drop_handle(p);
}
hyle_sched_watch_receiver* hyle_sched_watch_receiver_clone(
    const hyle_sched_watch_receiver* p, hyle_allocator b) {
  if (p == nullptr) {
    set_error("watch receiver must be non-null");
    return nullptr;
  }
  return make_handle<hyle_sched_watch_receiver_o>(b, p->value);
}
hyle_sched_watch_receiver* hyle_sched_watch_subscribe(
    const hyle_sched_watch_sender* p, hyle_allocator b) {
  if (p == nullptr) {
    set_error("watch sender must be non-null");
    return nullptr;
  }
  return make_handle<hyle_sched_watch_receiver_o>(b, p->value.subscribe());
}
void hyle_sched_watch_receiver_destroy(hyle_sched_watch_receiver* p) {
  drop_handle(p);
}
int hyle_sched_watch_send(hyle_sched_watch_sender* p, void* value) {
  return p != nullptr && p->value.send(value).has_value() ? 1 : 0;
}
void* hyle_sched_watch_send_replace(hyle_sched_watch_sender* p, void* value) {
  return p != nullptr ? p->value.send_replace(value) : nullptr;
}
void* hyle_sched_watch_sender_borrow(const hyle_sched_watch_sender* p) {
  return p != nullptr ? p->value.borrow() : nullptr;
}
void* hyle_sched_watch_receiver_borrow(const hyle_sched_watch_receiver* p) {
  return p != nullptr ? p->value.borrow() : nullptr;
}
void* hyle_sched_watch_receiver_borrow_and_update(
    hyle_sched_watch_receiver* p) {
  return p != nullptr ? p->value.borrow_and_update() : nullptr;
}
int hyle_sched_watch_changed(hyle_sched_watch_receiver* p) {
  return p != nullptr && p->value.changed().has_value() ? 1 : 0;
}
int hyle_sched_watch_has_changed(const hyle_sched_watch_receiver* p) {
  return p != nullptr && p->value.has_changed() ? 1 : 0;
}
size_t hyle_sched_watch_receiver_count(const hyle_sched_watch_sender* p) {
  return p != nullptr ? p->value.receiver_count() : 0;
}

int hyle_sched_broadcast_create(
    hyle_allocator b, size_t capacity,
    hyle_sched_broadcast_sender** out_sender,
    hyle_sched_broadcast_receiver** out_receiver) {
  if (out_sender == nullptr || out_receiver == nullptr) {
    set_error("broadcast output pointers must be non-null");
    return 0;
  }
  *out_sender = nullptr;
  *out_receiver = nullptr;
  if (!normalize(&b)) return 0;
  CAllocatorBridge* bridge = make_bridge(b);
  if (bridge == nullptr) return 0;
  auto pair = hyle::sched::try_make_broadcast<void*>(
      bridge_allocator(bridge), capacity);
  bridge_release(bridge);
  if (!pair.has_value()) {
    set_error(capacity == 0 ? "invalid broadcast capacity"
                            : "broadcast allocation failed");
    return 0;
  }
  auto* sender = make_handle<hyle_sched_broadcast_sender_o>(
      b, std::move(pair->first));
  if (sender == nullptr) return 0;
  auto* receiver = make_handle<hyle_sched_broadcast_receiver_o>(
      b, std::move(pair->second));
  if (receiver == nullptr) {
    drop_handle(sender);
    return 0;
  }
  *out_sender = sender;
  *out_receiver = receiver;
  clear_error();
  return 1;
}
hyle_sched_broadcast_sender* hyle_sched_broadcast_sender_clone(
    const hyle_sched_broadcast_sender* p, hyle_allocator b) {
  if (p == nullptr) {
    set_error("broadcast sender must be non-null");
    return nullptr;
  }
  return make_handle<hyle_sched_broadcast_sender_o>(b, p->value);
}
void hyle_sched_broadcast_sender_destroy(hyle_sched_broadcast_sender* p) {
  drop_handle(p);
}
hyle_sched_broadcast_receiver* hyle_sched_broadcast_receiver_clone(
    const hyle_sched_broadcast_receiver* p, hyle_allocator b) {
  if (p == nullptr) {
    set_error("broadcast receiver must be non-null");
    return nullptr;
  }
  return make_handle<hyle_sched_broadcast_receiver_o>(b, p->value);
}
hyle_sched_broadcast_receiver* hyle_sched_broadcast_subscribe(
    const hyle_sched_broadcast_sender* p, hyle_allocator b) {
  if (p == nullptr) {
    set_error("broadcast sender must be non-null");
    return nullptr;
  }
  return make_handle<hyle_sched_broadcast_receiver_o>(
      b, p->value.subscribe());
}
void hyle_sched_broadcast_receiver_destroy(
    hyle_sched_broadcast_receiver* p) {
  drop_handle(p);
}
int hyle_sched_broadcast_send(hyle_sched_broadcast_sender* p, void* value) {
  if (p == nullptr) return 0;
  p->value.send(value);
  return 1;
}
hyle_sched_recv_result hyle_sched_broadcast_recv(
    hyle_sched_broadcast_receiver* p, void** out) {
  if (p == nullptr || out == nullptr) return HYLE_SCHED_RECV_CLOSED;
  auto result = p->value.recv();
  if (result.has_value()) {
    *out = *result;
    return HYLE_SCHED_RECV_VALUE;
  }
  return result.error().IsUnavailable() ? HYLE_SCHED_RECV_LAGGED
                                        : HYLE_SCHED_RECV_CLOSED;
}
hyle_sched_recv_result hyle_sched_broadcast_try_recv(
    hyle_sched_broadcast_receiver* p, void** out) {
  if (p == nullptr || out == nullptr) return HYLE_SCHED_RECV_CLOSED;
  auto result = p->value.try_recv();
  if (result.has_value()) {
    *out = *result;
    return HYLE_SCHED_RECV_VALUE;
  }
  if (result.error().IsNotFound()) return HYLE_SCHED_RECV_EMPTY;
  return result.error().IsUnavailable() ? HYLE_SCHED_RECV_LAGGED
                                        : HYLE_SCHED_RECV_CLOSED;
}
size_t hyle_sched_broadcast_sender_count(
    const hyle_sched_broadcast_sender* p) {
  return p != nullptr ? p->value.sender_count() : 0;
}
size_t hyle_sched_broadcast_receiver_count(
    const hyle_sched_broadcast_sender* p) {
  return p != nullptr ? p->value.receiver_count() : 0;
}

hyle_sched_cancel_token* hyle_sched_cancel_root(hyle_allocator b) {
  if (!normalize(&b)) return nullptr;
  CAllocatorBridge* bridge = make_bridge(b);
  if (bridge == nullptr) return nullptr;
  auto token =
      hyle::sched::CancelToken::root(bridge_allocator(bridge));
  bridge_release(bridge);
  if (!token.valid()) {
    set_error("cancel token allocation failed");
    return nullptr;
  }
  return make_handle<hyle_sched_cancel_token_o>(b, std::move(token));
}
hyle_sched_cancel_token* hyle_sched_cancel_derive(
    hyle_sched_cancel_token* parent, hyle_allocator b) {
  if (parent == nullptr) {
    set_error("cancel parent must be non-null");
    return nullptr;
  }
  auto child = parent->value.derive();
  if (!child.valid()) {
    set_error("cancel token allocation failed");
    return nullptr;
  }
  return make_handle<hyle_sched_cancel_token_o>(b, std::move(child));
}
void hyle_sched_cancel_token_destroy(hyle_sched_cancel_token* p) {
  drop_handle(p);
}
void hyle_sched_cancel(hyle_sched_cancel_token* p) {
  if (p) p->value.cancel();
}
int hyle_sched_is_cancelled(const hyle_sched_cancel_token* p) {
  return p != nullptr && p->value.is_cancelled() ? 1 : 0;
}
hyle_sched_cancel_callback* hyle_sched_cancel_callback_create(
    const hyle_sched_cancel_token* token, hyle_sched_cancel_fn fn, void* ctx,
    hyle_allocator b) {
  if (token == nullptr || fn == nullptr) {
    set_error("cancel token and callback must be non-null");
    return nullptr;
  }
  return make_handle<hyle_sched_cancel_callback_o>(b, token->value, fn, ctx);
}
void hyle_sched_cancel_callback_destroy(hyle_sched_cancel_callback* p) {
  drop_handle(p);
}

hyle_sched_blocking_pool* hyle_sched_blocking_pool_create(
    hyle_allocator b, uint32_t max_threads, uint64_t keep_alive) {
  if (max_threads == 0) {
    set_error("blocking pool max_threads must be non-zero");
    return nullptr;
  }
  return make_handle<hyle_sched_blocking_pool_o>(b, max_threads, keep_alive);
}
void hyle_sched_blocking_pool_destroy(hyle_sched_blocking_pool* p) {
  drop_handle(p);
}
int hyle_sched_blocking_submit(hyle_sched_blocking_pool* p,
                               hyle_sched_task_fn task,
                               hyle_sched_blocking_done_fn done, void* ctx) {
  if (p == nullptr || task == nullptr) {
    set_error("blocking pool and task callback must be non-null");
    return 0;
  }
  hyle::mem::Allocator alloc = to_cpp(&p->backing);
  void* storage = alloc.allocate(sizeof(CBlockingJob), alignof(CBlockingJob));
  if (storage == nullptr) {
    set_error("blocking task record allocation failed");
    return 0;
  }
  auto* job = new (storage) CBlockingJob(alloc, task, done, ctx);
  if (!p->value.submit(&job->header)) {
    job->header.cancel();
    job->header.destroy();
    set_error("blocking pool is shutting down");
    return 0;
  }
  clear_error();
  return 1;
}
void hyle_sched_blocking_shutdown(hyle_sched_blocking_pool* p) {
  if (p) p->value.shutdown();
}
uint32_t hyle_sched_blocking_live_threads(const hyle_sched_blocking_pool* p) {
  return p != nullptr ? p->value.live_threads() : 0;
}
size_t hyle_sched_blocking_retained_thread_handles(
    const hyle_sched_blocking_pool* p) {
  return p != nullptr ? p->value.retained_thread_handles() : 0;
}

}  // extern "C"
