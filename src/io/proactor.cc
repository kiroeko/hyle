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

// IOCP real-IO ops (io-02) are NOT yet implemented: unlike io_uring (fd goes in
// the SQE, token in user_data), IOCP needs a per-op OVERLAPPED whose lifetime
// spans the in-flight op, and the per-op token must ride that OVERLAPPED (the
// completion KEY is per-handle, not per-op). That request-lifetime contract is
// designed with net/ (io-04) so it isn't guessed here. io_uring carries the real
// read/write now (io-02 primary, WSL-tested); IOCP real ops follow. Report false
// rather than pretend.
bool Proactor::submit_read(std::intptr_t, void*, std::uint32_t, std::uint64_t,
                           std::uint64_t) noexcept {
  return false;  // io-02 IOCP real-op pending (io_uring path is done)
}
bool Proactor::submit_write(std::intptr_t, const void*, std::uint32_t,
                            std::uint64_t, std::uint64_t) noexcept {
  return false;  // io-02 IOCP real-op pending
}
bool Proactor::register_handle(std::intptr_t fd) noexcept {
  if (impl_ == nullptr) return false;
  // Associate the handle with the port so its ops complete here. Key unused for
  // now (per-op token rides the OVERLAPPED once real ops land). Harmless to call.
  return ::CreateIoCompletionPort(reinterpret_cast<HANDLE>(fd), impl_->port, 0,
                                  0) != nullptr;
}

}  // namespace hyle::io

#elif defined(HYLE_OS_LINUX)

// io_uring backend (io-02). io-design says borrow ONLY the syscall shim, not
// liburing's helpers -- and the mechanism heartbeat proved we need even less:
// the 3 io_uring syscalls + the uapi header + our own inline ring arithmetic.
// The fast-path barriers are replicated exactly from liburing v2.15 (verified):
//   submit publish  -> store_release on SQ ktail  (pairs with kernel acquire)
//   reap read       -> load_acquire  on CQ ktail  (pairs with kernel release)
//   reap advance    -> store_release on CQ khead  (kernel won't overwrite unread)
// Only io_uring_enter is a syscall on the hot path; submit/reap ring writes are
// pure userspace. No SQPOLL here (io-01/02 baseline); that is a setup-flag
// upgrade the interface already accommodates (submit stays syscall-optional).
#  include <linux/io_uring.h>
#  include <sys/mman.h>
#  include <sys/syscall.h>
#  include <unistd.h>

#  include <atomic>
#  include <cerrno>
#  include <cstring>

namespace hyle::io {
namespace {

int sys_io_uring_setup(unsigned entries, io_uring_params* p) noexcept {
  return static_cast<int>(::syscall(__NR_io_uring_setup, entries, p));
}
// enter with the EXT_ARG timeout path (modern kernels): a timespec passed via
// io_uring_getevents_arg on the SAME syscall -- no extra timeout SQE.
int sys_io_uring_enter(int fd, unsigned to_submit, unsigned min_complete,
                       unsigned flags, void* arg, std::size_t argsz) noexcept {
  return static_cast<int>(::syscall(__NR_io_uring_enter, fd, to_submit,
                                    min_complete, flags, arg, argsz));
}

// atomic view over a kernel-shared u32 ring counter (barriers per liburing).
inline std::atomic<unsigned>* as_atomic(void* p) noexcept {
  return reinterpret_cast<std::atomic<unsigned>*>(p);
}

}  // namespace

struct Proactor::Impl {
  int fd = -1;
  void* sq_ptr = nullptr;
  std::size_t sq_sz = 0;
  void* cq_ptr = nullptr;        // == sq_ptr when SINGLE_MMAP
  std::size_t cq_sz = 0;
  io_uring_sqe* sqes = nullptr;
  std::size_t sqes_sz = 0;
  unsigned sq_entries = 0;
  bool single_mmap = false;
  // cached ring field pointers
  unsigned* sq_tail = nullptr;
  unsigned* sq_ring_mask = nullptr;
  unsigned* sq_array = nullptr;
  unsigned* cq_head = nullptr;
  unsigned* cq_tail = nullptr;
  unsigned* cq_ring_mask = nullptr;
  io_uring_cqe* cqes = nullptr;
};

Proactor::~Proactor() {
  if (impl_ == nullptr) return;
  if (impl_->sqes != nullptr) ::munmap(impl_->sqes, impl_->sqes_sz);
  if (impl_->cq_ptr != nullptr && !impl_->single_mmap)
    ::munmap(impl_->cq_ptr, impl_->cq_sz);
  if (impl_->sq_ptr != nullptr) ::munmap(impl_->sq_ptr, impl_->sq_sz);
  if (impl_->fd >= 0) ::close(impl_->fd);
  delete impl_;
  impl_ = nullptr;
}

bool Proactor::init(unsigned entries) noexcept {
  if (impl_ != nullptr) return true;  // idempotent
  io_uring_params p;
  std::memset(&p, 0, sizeof(p));
  int fd = sys_io_uring_setup(entries, &p);
  if (fd < 0) return false;

  auto* impl = new (std::nothrow) Impl{};
  if (impl == nullptr) {
    ::close(fd);
    return false;
  }
  impl->fd = fd;
  impl->sq_entries = p.sq_entries;
  impl->single_mmap = (p.features & IORING_FEAT_SINGLE_MMAP) != 0;

  std::size_t sqring = p.sq_off.array + p.sq_entries * sizeof(unsigned);
  std::size_t cqring = p.cq_off.cqes + p.cq_entries * sizeof(io_uring_cqe);
  if (impl->single_mmap) {
    if (cqring > sqring) sqring = cqring;
    cqring = sqring;
  }
  impl->sq_sz = sqring;
  impl->cq_sz = cqring;

  void* sq = ::mmap(nullptr, sqring, PROT_READ | PROT_WRITE,
                    MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQ_RING);
  if (sq == MAP_FAILED) {
    ::close(fd);
    delete impl;
    return false;
  }
  impl->sq_ptr = sq;
  void* cq = impl->single_mmap
                 ? sq
                 : ::mmap(nullptr, cqring, PROT_READ | PROT_WRITE,
                          MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_CQ_RING);
  if (cq == MAP_FAILED) {
    ::munmap(sq, sqring);
    ::close(fd);
    delete impl;
    return false;
  }
  impl->cq_ptr = cq;

  impl->sqes_sz = p.sq_entries * sizeof(io_uring_sqe);
  void* sqes = ::mmap(nullptr, impl->sqes_sz, PROT_READ | PROT_WRITE,
                      MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQES);
  if (sqes == MAP_FAILED) {
    if (!impl->single_mmap) ::munmap(cq, cqring);
    ::munmap(sq, sqring);
    ::close(fd);
    delete impl;
    return false;
  }
  impl->sqes = static_cast<io_uring_sqe*>(sqes);

  auto* B = static_cast<char*>(sq);
  auto* C = static_cast<char*>(cq);
  impl->sq_tail = reinterpret_cast<unsigned*>(B + p.sq_off.tail);
  impl->sq_ring_mask = reinterpret_cast<unsigned*>(B + p.sq_off.ring_mask);
  impl->sq_array = reinterpret_cast<unsigned*>(B + p.sq_off.array);
  impl->cq_head = reinterpret_cast<unsigned*>(C + p.cq_off.head);
  impl->cq_tail = reinterpret_cast<unsigned*>(C + p.cq_off.tail);
  impl->cq_ring_mask = reinterpret_cast<unsigned*>(C + p.cq_off.ring_mask);
  impl->cqes = reinterpret_cast<io_uring_cqe*>(C + p.cq_off.cqes);

  impl_ = impl;
  return true;
}

std::size_t Proactor::reap(Completion* out, std::size_t max,
                           std::uint64_t timeout_ns) noexcept {
  if (impl_ == nullptr || max == 0) return 0;

  // If the CQ is already non-empty, skip the syscall entirely (pure userspace
  // reap -- the io_uring fast path). Otherwise enter to wait for >=1 completion,
  // passing the timeout via EXT_ARG so no separate timeout SQE is consumed.
  unsigned tail = as_atomic(impl_->cq_tail)->load(std::memory_order_acquire);
  unsigned head = *impl_->cq_head;
  if (head == tail) {
    if (timeout_ns == 0) return 0;  // non-blocking poll, nothing ready
    struct __kernel_timespec ts;
    struct io_uring_getevents_arg arg;
    std::memset(&arg, 0, sizeof(arg));
    unsigned flags = IORING_ENTER_GETEVENTS;
    if (timeout_ns != UINT64_MAX) {
      ts.tv_sec = static_cast<long long>(timeout_ns / 1'000'000'000ull);
      ts.tv_nsec = static_cast<long long>(timeout_ns % 1'000'000'000ull);
      arg.ts = reinterpret_cast<std::uint64_t>(&ts);
      flags |= IORING_ENTER_EXT_ARG;
    }
    int r = sys_io_uring_enter(impl_->fd, 0, 1, flags,
                               (flags & IORING_ENTER_EXT_ARG) ? &arg : nullptr,
                               (flags & IORING_ENTER_EXT_ARG) ? sizeof(arg) : 0);
    if (r < 0 && errno != ETIME && errno != EINTR) return 0;
    tail = as_atomic(impl_->cq_tail)->load(std::memory_order_acquire);
    head = *impl_->cq_head;
  }

  std::size_t n = 0;
  const unsigned mask = *impl_->cq_ring_mask;
  while (head != tail && n < max) {
    const io_uring_cqe& c = impl_->cqes[head & mask];
    out[n].user_data = c.user_data;
    out[n].result = static_cast<std::int64_t>(c.res);  // <0 = -errno (io-02 ops)
    // Map io_uring CQE flags onto our neutral flags.
    out[n].flags = 0;
    if (c.flags & IORING_CQE_F_MORE) out[n].flags |= Completion::kMore;
    if (c.flags & IORING_CQE_F_BUFFER) {
      out[n].flags |= Completion::kBuffer;
      out[n].buffer_id = c.flags >> IORING_CQE_BUFFER_SHIFT;
    } else {
      out[n].buffer_id = 0;
    }
    ++head;
    ++n;
  }
  // Publish the consumed head once for the whole batch (store_release pairs with
  // the kernel; one release-store amortized over n completions).
  as_atomic(impl_->cq_head)->store(head, std::memory_order_release);
  return n;
}

namespace {
// Fill one SQE and submit it (one io_uring_enter). Pure ring mechanics -- takes
// the mapped ring fields directly so it needn't name the private Impl type.
// Shared by post/read/write. addr/len/off are 0 for NOP.
bool submit_one(int ring_fd, unsigned* sq_tail, unsigned sq_mask,
                unsigned* sq_array, io_uring_sqe* sqes, std::uint8_t opcode,
                std::int32_t fd, std::uint64_t addr, std::uint32_t len,
                std::uint64_t off, std::uint64_t user_data) noexcept {
  unsigned tail = *sq_tail;
  unsigned index = tail & sq_mask;
  io_uring_sqe& s = sqes[index];
  std::memset(&s, 0, sizeof(s));
  s.opcode = opcode;
  s.fd = fd;
  s.off = off;
  s.addr = addr;
  s.len = len;
  s.user_data = user_data;
  sq_array[index] = index;
  // publish SQ tail (store_release pairs with kernel load_acquire)
  as_atomic(sq_tail)->store(tail + 1, std::memory_order_release);
  int r = sys_io_uring_enter(ring_fd, 1, 0, 0, nullptr, 0);
  return r >= 0;
}
}  // namespace

bool Proactor::post(std::uint64_t user_data, std::int64_t result) noexcept {
  if (impl_ == nullptr) return false;
  // NOP SQE carrying user_data: backend-independent "wake/inject" hook. NOP res
  // is always 0, so `result` is not delivered on Linux -- post() is a wake hook,
  // not a data channel (IOCP can carry bytes; asymmetry documented in header).
  (void)result;
  return submit_one(impl_->fd, impl_->sq_tail, *impl_->sq_ring_mask,
                    impl_->sq_array, impl_->sqes, IORING_OP_NOP, -1, 0, 0, 0,
                    user_data);
}

bool Proactor::submit_read(std::intptr_t fd, void* buf, std::uint32_t len,
                           std::uint64_t offset,
                           std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  return submit_one(impl_->fd, impl_->sq_tail, *impl_->sq_ring_mask,
                    impl_->sq_array, impl_->sqes, IORING_OP_READ,
                    static_cast<std::int32_t>(fd),
                    reinterpret_cast<std::uint64_t>(buf), len, offset, user_data);
}

bool Proactor::submit_write(std::intptr_t fd, const void* buf,
                            std::uint32_t len, std::uint64_t offset,
                            std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  return submit_one(impl_->fd, impl_->sq_tail, *impl_->sq_ring_mask,
                    impl_->sq_array, impl_->sqes, IORING_OP_WRITE,
                    static_cast<std::int32_t>(fd),
                    reinterpret_cast<std::uint64_t>(buf), len, offset, user_data);
}

bool Proactor::register_handle(std::intptr_t /*fd*/) noexcept {
  // io_uring needs no per-fd association (the fd goes straight in each SQE).
  return impl_ != nullptr;
}

}  // namespace hyle::io

#else
#  error "HYLE io/proactor: unsupported OS"
#endif
