#include "hyle/io/proactor.h"

#include <new>

#include "hyle/macros.h"
#include "hyle/platform.h"

// io-01 proactor backend. This file provides the per-OS Impl behind proactor.h.
// io-02 wires the real submission ops (read/write/accept); io-01 establishes the
// completion-reaping seam and proves it drives the sched Waker spine. Only the
// Windows IOCP path is built+tested on this machine right now; the Linux
// io_uring path lands next (borrowing only liburing's syscall shim, io-design).

#if defined(HYLE_OS_WINDOWS)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>

namespace hyle::io {

// IOCP is a true proactor: an operation posts a completion to the port when the
// kernel finishes it, carrying the completion key + OVERLAPPED we attached. We
// use the completion KEY as the 64-bit user_data (the Waker token). The worker
// reaps with GetQueuedCompletionStatusEx (batch), the same shape as io_uring's
// for_each_cqe. (Mechanism heartbeat-verified before this was written.)
struct Proactor::Impl {
  HANDLE port = nullptr;
};

Proactor::~Proactor() {
  if (impl_ != nullptr) {
    if (impl_->port != nullptr) ::CloseHandle(impl_->port);
    delete impl_;
    impl_ = nullptr;
  }
}

bool Proactor::init(unsigned /*entries*/) noexcept {
  if (impl_ != nullptr) return true;  // idempotent
  // NumberOfConcurrentThreads=0 -> IOCP allows as many concurrently-running
  // completion threads as there are cores (we drive it per-worker anyway).
  HANDLE p = ::CreateIoCompletionPort(INVALID_HANDLE_VALUE, nullptr, 0, 0);
  if (p == nullptr) return false;
  impl_ = new (std::nothrow) Impl{};
  if (impl_ == nullptr) {
    ::CloseHandle(p);
    return false;
  }
  impl_->port = p;
  return true;
}

std::size_t Proactor::reap(Completion* out, std::size_t max,
                           std::uint64_t timeout_ns) noexcept {
  if (impl_ == nullptr || max == 0) return 0;

  // ns -> ms for the OS wait. 0 stays 0 (non-blocking poll); UINT64_MAX ->
  // INFINITE (block until a completion); otherwise round UP so a sub-ms deadline
  // never truncates to a busy poll.
  DWORD ms;
  if (timeout_ns == 0) {
    ms = 0;
  } else if (timeout_ns == UINT64_MAX) {
    ms = INFINITE;
  } else {
    std::uint64_t rounded = (timeout_ns + 999'999ull) / 1'000'000ull;
    ms = rounded >= INFINITE ? (INFINITE - 1) : static_cast<DWORD>(rounded);
  }

  // Batch dequeue -- one syscall pulls up to `max` finished ops (io_uring
  // for_each_cqe analogue). Cap the batch at a stack buffer chunk.
  constexpr ULONG kMaxBatch = 128;
  OVERLAPPED_ENTRY entries[kMaxBatch];
  ULONG want = static_cast<ULONG>(max < kMaxBatch ? max : kMaxBatch);
  ULONG removed = 0;
  BOOL ok = ::GetQueuedCompletionStatusEx(impl_->port, entries, want, &removed,
                                          ms, FALSE);
  if (!ok) {
    // WAIT_TIMEOUT (nothing ready within `ms`) is the normal empty case, not an
    // error -- report zero reaped. Any other failure is also surfaced as zero
    // (the caller re-polls; a truly broken port would fail init).
    return 0;
  }

  for (ULONG i = 0; i < removed; ++i) {
    out[i].user_data = static_cast<std::uint64_t>(entries[i].lpCompletionKey);
    // dwNumberOfBytesTransferred is the op's byte count; io error mapping to a
    // negative result is an op-submission concern (io-02), so io-01 reports the
    // raw byte count. flags/buffer_id stay 0 (IOCP has no CQE flags).
    out[i].result =
        static_cast<std::int64_t>(entries[i].dwNumberOfBytesTransferred);
    out[i].flags = 0;
    out[i].buffer_id = 0;
  }
  return static_cast<std::size_t>(removed);
}

bool Proactor::post(std::uint64_t user_data, std::int64_t result) noexcept {
  if (impl_ == nullptr) return false;
  // Post a synthetic completion: key = user_data (the Waker token), bytes =
  // result. Wakes a blocked reap() from any thread AND drives tests without a
  // real fd. OVERLAPPED is unused here (nullptr) -- io-02's real ops attach one.
  return ::PostQueuedCompletionStatus(
             impl_->port,
             static_cast<DWORD>(static_cast<std::uint64_t>(result)),
             static_cast<ULONG_PTR>(user_data), nullptr) != 0;
}

}  // namespace hyle::io

#elif defined(HYLE_OS_LINUX)

// io_uring backend lands next (io-02): borrow liburing's 3 syscalls + mmap'd
// ring layout, write get_sqe/submit/reap inline with the exact barriers
// (load_acquire ktail / store_release khead). Until then, Linux has no proactor
// backend -- init() reports failure rather than pretending.
namespace hyle::io {

struct Proactor::Impl {};
Proactor::~Proactor() { delete impl_; }
bool Proactor::init(unsigned) noexcept { return false; }  // io-02 pending
std::size_t Proactor::reap(Completion*, std::size_t, std::uint64_t) noexcept {
  return 0;
}
bool Proactor::post(std::uint64_t, std::int64_t) noexcept { return false; }

}  // namespace hyle::io

#else
#  error "HYLE io/proactor: unsupported OS"
#endif
