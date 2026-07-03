#include "hyle/sched/join.h"

#include <atomic>

namespace hyle::sched {
namespace detail {
namespace {

// The cancel flag of the task currently running on THIS worker thread. A stack
// of pointers because a task may spawn/await nested work on the same fiber stack;
// the innermost running task's flag is the one is_cancel_requested() reports.
// Depth is tiny (nested spawn_task on one fiber is rare) so a fixed small array
// avoids any allocation on the hot suspend path.
constexpr int kMaxCancelDepth = 16;
thread_local std::atomic<bool>* t_cancel_stack[kMaxCancelDepth] = {};
thread_local int t_cancel_depth = 0;

}  // namespace

void push_cancel_scope(std::atomic<bool>* flag) noexcept {
  // Overflow just stops nesting deeper (queries fall back to the outer scope);
  // 16 deep is already pathological. Never crash on it.
  if (t_cancel_depth < kMaxCancelDepth) {
    t_cancel_stack[t_cancel_depth] = flag;
  }
  ++t_cancel_depth;
}

void pop_cancel_scope() noexcept {
  --t_cancel_depth;
  if (t_cancel_depth >= 0 && t_cancel_depth < kMaxCancelDepth) {
    t_cancel_stack[t_cancel_depth] = nullptr;
  }
}

std::atomic<bool>* current_cancel_flag() noexcept {
  int d = t_cancel_depth;
  if (d > kMaxCancelDepth) d = kMaxCancelDepth;
  if (d <= 0) return nullptr;
  return t_cancel_stack[d - 1];
}

}  // namespace detail

bool is_cancel_requested() noexcept {
  std::atomic<bool>* f = detail::current_cancel_flag();
  return f != nullptr && f->load(std::memory_order_acquire);
}

}  // namespace hyle::sched
