#pragma once

#include <cstdint>
#include <utility>

#include "hyle/io/detail/net.h"
#include "hyle/io/detail/net_timeout.h"
#include "hyle/io/operation.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/cancel.h"

namespace hyle::io::detail {

// Shared state for the public one-budget awaitables. The default/no-options
// branch stays on the suspended fiber's stack. A deadline or cancellation
// source uses a heap slot to arbitrate the extra wake source and waits for the
// kernel's mandatory terminal completion before returning borrowed storage.
struct BudgetState {
  Proactor* proactor = nullptr;
  mem::AllocatorRef allocator;
  OperationOptions options;
  IoOperation ordinary{};
  TimedSlot* bounded = nullptr;
  std::int64_t immediate = 0;
  std::uint32_t completion_flags = 0;
  bool immediate_ready = false;

  bool await_ready() noexcept {
    if (options.cancel != nullptr && options.cancel->is_cancelled()) {
      immediate = kCancelled;
      immediate_ready = true;
      return true;
    }
    if (options.deadline.has_value() &&
        *options.deadline <= time::monotonic_now()) {
      immediate = kDeadlineExceeded;
      immediate_ready = true;
      return true;
    }
    return false;
  }

  template <class Submit>
  bool await_suspend(Submit&& submit) noexcept {
    const bool bounded_operation =
        options.deadline.has_value() || options.cancel != nullptr;
    if (!bounded_operation) {
      ordinary.waker = sched::current_waker();
      if (!std::forward<Submit>(submit)(to_token(&ordinary))) {
        ordinary.result = kSubmitFailed;
        return false;
      }
      sched::suspend();
      return false;
    }

    bounded = make_budget_slot(allocator, proactor, options.deadline);
    if (bounded == nullptr) {
      immediate = kSubmitFailed;
      immediate_ready = true;
      return false;
    }
    if (!std::forward<Submit>(submit)(to_timed_token(bounded))) {
      on_submit_failed(bounded);
      return false;
    }

    if (options.cancel != nullptr) {
      sched::CancelCallback callback{*options.cancel, &cancel_timed_slot,
                                     bounded};
      sched::suspend();
      await_cancel_retirement(bounded);
    } else {
      sched::suspend();
    }
    return false;
  }

  std::int64_t await_resume() noexcept {
    if (immediate_ready) return immediate;
    completion_flags =
        bounded != nullptr ? bounded->io.flags : ordinary.flags;
    return bounded != nullptr ? resume_timed(bounded) : ordinary.result;
  }
};

}  // namespace hyle::io::detail

namespace hyle::io::detail {

struct BudgetRecvOp {
  Proactor* proactor = nullptr;
  std::intptr_t fd = -1;
  void* buffer = nullptr;
  std::uint32_t length = 0;
  mem::AllocatorRef allocator;
  OperationOptions options{};
  std::uint32_t flags = 0;
  BudgetState state{};

  bool await_ready() noexcept {
    state = BudgetState{proactor, allocator, options};
    return state.await_ready();
  }
  bool await_suspend(std::coroutine_handle<>) noexcept {
    return state.await_suspend([&](std::uint64_t token) noexcept {
      return proactor->submit_recv(fd, buffer, length, flags, token);
    });
  }
  std::int64_t await_resume() noexcept { return state.await_resume(); }
};

struct BudgetSendOp {
  Proactor* proactor = nullptr;
  std::intptr_t fd = -1;
  const void* buffer = nullptr;
  std::uint32_t length = 0;
  mem::AllocatorRef allocator;
  OperationOptions options{};
  std::uint32_t flags = 0;
  BudgetState state{};

  bool await_ready() noexcept {
    state = BudgetState{proactor, allocator, options};
    return state.await_ready();
  }
  bool await_suspend(std::coroutine_handle<>) noexcept {
    return state.await_suspend([&](std::uint64_t token) noexcept {
      return proactor->submit_send(fd, buffer, length, flags, token);
    });
  }
  std::int64_t await_resume() noexcept { return state.await_resume(); }
};

struct BudgetRecvBuffersOp {
  Proactor* proactor = nullptr;
  std::intptr_t fd = -1;
  NativeBuffer* buffers = nullptr;
  std::uint32_t count = 0;
  mem::AllocatorRef allocator;
  OperationOptions options{};
  BudgetState state{};

  bool await_ready() noexcept {
    state = BudgetState{proactor, allocator, options};
    return state.await_ready();
  }
  bool await_suspend(std::coroutine_handle<>) noexcept {
    return state.await_suspend([&](std::uint64_t token) noexcept {
      return proactor->submit_recv_buffers(fd, buffers, count, token);
    });
  }
  std::int64_t await_resume() noexcept { return state.await_resume(); }
};

struct BudgetSendBuffersOp {
  Proactor* proactor = nullptr;
  std::intptr_t fd = -1;
  NativeBuffer* buffers = nullptr;
  std::uint32_t count = 0;
  mem::AllocatorRef allocator;
  OperationOptions options{};
  BudgetState state{};

  bool await_ready() noexcept {
    state = BudgetState{proactor, allocator, options};
    return state.await_ready();
  }
  bool await_suspend(std::coroutine_handle<>) noexcept {
    return state.await_suspend([&](std::uint64_t token) noexcept {
      return proactor->submit_send_buffers(fd, buffers, count, token);
    });
  }
  std::int64_t await_resume() noexcept { return state.await_resume(); }
};

struct BudgetAcceptOp {
  Proactor* proactor = nullptr;
  std::intptr_t fd = -1;
  mem::AllocatorRef allocator;
  OperationOptions options{};
  BudgetState state{};

  bool await_ready() noexcept {
    state = BudgetState{proactor, allocator, options};
    return state.await_ready();
  }
  bool await_suspend(std::coroutine_handle<>) noexcept {
    return state.await_suspend([&](std::uint64_t token) noexcept {
      return proactor->submit_accept(fd, nullptr, nullptr, token);
    });
  }
  std::int64_t await_resume() noexcept { return state.await_resume(); }
};

struct BudgetConnectOp {
  Proactor* proactor = nullptr;
  std::intptr_t fd = -1;
  const void* address = nullptr;
  std::uint32_t address_length = 0;
  mem::AllocatorRef allocator;
  OperationOptions options{};
  BudgetState state{};

  bool await_ready() noexcept {
    state = BudgetState{proactor, allocator, options};
    return state.await_ready();
  }
  bool await_suspend(std::coroutine_handle<>) noexcept {
    return state.await_suspend([&](std::uint64_t token) noexcept {
      return proactor->submit_connect(fd, address, address_length, token);
    });
  }
  std::int64_t await_resume() noexcept { return state.await_resume(); }
};

struct BudgetRecvFromOp {
  Proactor* proactor = nullptr;
  std::intptr_t fd = -1;
  void* buffer = nullptr;
  std::uint32_t length = 0;
  void* address = nullptr;
  std::uint32_t address_capacity = 0;
  mem::AllocatorRef allocator;
  OperationOptions options{};
  BudgetState state{};
  alignas(16) unsigned char scratch[Proactor::kDatagramScratch]{};

  bool await_ready() noexcept {
    state = BudgetState{proactor, allocator, options};
    return state.await_ready();
  }
  bool await_suspend(std::coroutine_handle<>) noexcept {
    return state.await_suspend([&](std::uint64_t token) noexcept {
      return proactor->submit_recvfrom(fd, buffer, length, address,
                                       address_capacity, scratch, token);
    });
  }
  std::int64_t await_resume() noexcept { return state.await_resume(); }
};

struct BudgetSendToOp {
  Proactor* proactor = nullptr;
  std::intptr_t fd = -1;
  const void* buffer = nullptr;
  std::uint32_t length = 0;
  const void* address = nullptr;
  std::uint32_t address_length = 0;
  mem::AllocatorRef allocator;
  OperationOptions options{};
  BudgetState state{};
  alignas(16) unsigned char scratch[Proactor::kDatagramScratch]{};

  bool await_ready() noexcept {
    state = BudgetState{proactor, allocator, options};
    return state.await_ready();
  }
  bool await_suspend(std::coroutine_handle<>) noexcept {
    return state.await_suspend([&](std::uint64_t token) noexcept {
      return proactor->submit_sendto(fd, buffer, length, address,
                                     address_length, scratch, token);
    });
  }
  std::int64_t await_resume() noexcept { return state.await_resume(); }
};

struct BudgetRecvFromBuffersOp {
  Proactor* proactor = nullptr;
  std::intptr_t fd = -1;
  NativeBuffer* buffers = nullptr;
  std::uint32_t count = 0;
  void* address = nullptr;
  std::uint32_t address_capacity = 0;
  mem::AllocatorRef allocator;
  OperationOptions options{};
  BudgetState state{};
  alignas(16) unsigned char scratch[Proactor::kDatagramScratch]{};

  bool await_ready() noexcept {
    state = BudgetState{proactor, allocator, options};
    return state.await_ready();
  }
  bool await_suspend(std::coroutine_handle<>) noexcept {
    return state.await_suspend([&](std::uint64_t token) noexcept {
      return proactor->submit_recvfrom_buffers(
          fd, buffers, count, address, address_capacity, scratch, token);
    });
  }
  std::int64_t await_resume() noexcept { return state.await_resume(); }
};

struct BudgetSendToBuffersOp {
  Proactor* proactor = nullptr;
  std::intptr_t fd = -1;
  NativeBuffer* buffers = nullptr;
  std::uint32_t count = 0;
  const void* address = nullptr;
  std::uint32_t address_length = 0;
  mem::AllocatorRef allocator;
  OperationOptions options{};
  BudgetState state{};
  alignas(16) unsigned char scratch[Proactor::kDatagramScratch]{};

  bool await_ready() noexcept {
    state = BudgetState{proactor, allocator, options};
    return state.await_ready();
  }
  bool await_suspend(std::coroutine_handle<>) noexcept {
    return state.await_suspend([&](std::uint64_t token) noexcept {
      return proactor->submit_sendto_buffers(
          fd, buffers, count, address, address_length, scratch, token);
    });
  }
  std::int64_t await_resume() noexcept { return state.await_resume(); }
};

}  // namespace hyle::io::detail
