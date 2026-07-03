#include "hyle/sched/join.h"

#include <atomic>

#include "hyle/sched/fiber.h"

namespace hyle::sched {
namespace detail {

// The cancel scope lives ON THE FIBER, not in thread_local storage. A task fiber
// suspends at a co_await and may be re-injected on a DIFFERENT worker thread, so a
// thread-local scope set on the first worker would be invisible after migration --
// is_cancel_requested() would read false on the new thread and silently miss an
// abort() (root cause of the Join.AbortIsCooperative flake, 2026-07-03). Storing
// the flag on the Fiber makes it travel with the execution (like the TSan fiber
// handle). Each spawn_task owns its own fiber, so there is exactly one cancel
// scope per fiber -- no nesting stack needed (nested spawn_task uses a NEW fiber,
// which carries its own flag). push sets it, pop clears it.
void push_cancel_scope(std::atomic<bool>* flag) noexcept {
  if (Fiber* self = Fiber::current(); self != nullptr) {
    self->set_cancel_flag(flag);
  }
}

void pop_cancel_scope() noexcept {
  if (Fiber* self = Fiber::current(); self != nullptr) {
    self->set_cancel_flag(nullptr);
  }
}

std::atomic<bool>* current_cancel_flag() noexcept {
  Fiber* self = Fiber::current();
  if (self == nullptr) return nullptr;
  return static_cast<std::atomic<bool>*>(self->cancel_flag());
}

}  // namespace detail

bool is_cancel_requested() noexcept {
  std::atomic<bool>* f = detail::current_cancel_flag();
  return f != nullptr && f->load(std::memory_order_acquire);
}

}  // namespace hyle::sched
