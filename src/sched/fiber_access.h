#pragma once

#include "hyle/sched/fiber.h"

namespace hyle::sched::detail {

// Source-only access to Fiber's scheduler state machine. This header stays in
// src/sched and is never part of source/static/shared user delivery.
struct FiberAccess {
  using FinishFn = Fiber::FinishFn;

  static Fiber* next(Fiber* fiber) noexcept { return fiber->ready_next_; }
  static void set_next(Fiber* fiber, Fiber* next) noexcept {
    fiber->ready_next_ = next;
  }
  static void request_wait(Fiber* fiber) noexcept {
    fiber->request_wait();
  }
  static bool consume_wait_request(Fiber* fiber) noexcept {
    return fiber->consume_wait_request();
  }
  static int owner_worker(const Fiber* fiber) noexcept {
    return fiber->owner_worker();
  }
  static void set_owner_worker(Fiber* fiber, int worker) noexcept {
    fiber->set_owner_worker(worker);
  }
  static void set_on_finish(Fiber* fiber, FinishFn fn,
                            void* context) noexcept {
    fiber->set_on_finish(fn, context);
  }
  static FinishFn on_finish(const Fiber* fiber) noexcept {
    return fiber->on_finish();
  }
  static void* on_finish_context(const Fiber* fiber) noexcept {
    return fiber->on_finish_ctx();
  }
  static bool wake_rendezvous(Fiber* fiber) noexcept {
    return fiber->wake_rendezvous();
  }
};

}  // namespace hyle::sched::detail
