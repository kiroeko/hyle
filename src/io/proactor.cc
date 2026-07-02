#include "hyle/io/proactor.h"

#include <new>

#include "hyle/macros.h"
#include "hyle/platform.h"

// io proactor backend: the per-OS Impl behind proactor.h. Both platforms carry
// real IO: Windows IOCP (ReadFile/WriteFile/WSARecv/WSASend/AcceptEx/ConnectEx
// with per-op OVERLAPPED) and Linux io_uring (raw syscalls, no liburing link) +
// an epoll reactor fallback. The proactor is the completion-reaping seam that
// drives the sched Waker spine -- io/ only moves completion tokens, sched/ owns
// the wake. Mechanism heartbeat-verified per backend before each was written.

#if defined(HYLE_OS_WINDOWS)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
// winsock2 MUST precede windows.h (else the older winsock.h leaks in and clashes).
#  include <winsock2.h>
#  include <mswsock.h>   // AcceptEx/ConnectEx LPFN types + WSAID_* GUIDs
#  include <ws2tcpip.h>  // sockaddr_storage / IN6ADDR
#  include <windows.h>

#  include <cstring>

namespace hyle::io {
namespace {
// Map a Win32/WSA error to the negative-result convention the header promises
// (same shape as io_uring CQE res < 0). Callers store this in Completion.result.
inline std::int64_t neg_err(DWORD e) noexcept {
  return -static_cast<std::int64_t>(e);
}
// Lazily fetch a Winsock extension function (AcceptEx/ConnectEx) via WSAIoctl.
// `slot` caches it. Takes only public types so it need not name Proactor::Impl.
bool load_ext_fn(SOCKET s, GUID guid, void** slot) noexcept {
  if (*slot != nullptr) return true;
  DWORD got = 0;
  return ::WSAIoctl(s, SIO_GET_EXTENSION_FUNCTION_POINTER, &guid, sizeof(guid),
                    slot, sizeof(void*), &got, nullptr, nullptr) == 0;
}
}  // namespace

// IOCP is a true proactor: a submitted op posts a completion to the port when the
// kernel finishes it, carrying the OVERLAPPED we attached. Unlike io_uring (token
// rides the SQE user_data), the per-op token must ride a per-op OVERLAPPED whose
// lifetime spans the in-flight op -- so each op allocates an `IoOp` (OVERLAPPED as
// its FIRST member, so the completion's lpOverlapped casts straight back to it).
// The proactor OWNS that OVERLAPPED end to end; net/ never sees it. post() injects
// a completion with lpOverlapped==nullptr, which reap uses to tell the two apart.
// (Mechanism heartbeat-verified before this was written.)
struct Proactor::Impl {
  HANDLE port = nullptr;
  bool wsa_started = false;
  LPFN_ACCEPTEX acceptex = nullptr;
  LPFN_CONNECTEX connectex = nullptr;

  enum OpKind : int { kRead, kWrite, kRecv, kSend, kAccept, kConnect };
  struct IoOp {
    OVERLAPPED ov{};                      // MUST be first (lpOverlapped -> IoOp*)
    std::uint64_t user_data = 0;
    HANDLE handle = nullptr;              // op's handle/socket (GetOverlappedResult)
    int kind = 0;
    WSABUF wsabuf{};                      // recv/send descriptor (outlives the op)
    SOCKET accept_sock = INVALID_SOCKET;  // AcceptEx: the accepted socket (result)
    IoOp* prev = nullptr;
    IoOp* next = nullptr;
    // AcceptEx writes local+remote addrs here (needs >= 2*(sockaddr+16)).
    alignas(16) char addr_buf[2 * (sizeof(SOCKADDR_STORAGE) + 16)] = {};
  };
  IoOp* inflight = nullptr;  // intrusive doubly-linked list of in-flight ops

  void link(IoOp* op) noexcept {
    op->prev = nullptr;
    op->next = inflight;
    if (inflight != nullptr) inflight->prev = op;
    inflight = op;
  }
  void unlink(IoOp* op) noexcept {
    if (op->prev != nullptr)
      op->prev->next = op->next;
    else
      inflight = op->next;
    if (op->next != nullptr) op->next->prev = op->prev;
  }
};

Proactor::~Proactor() {
  if (impl_ == nullptr) return;
  // Cancel every in-flight op, then DRAIN the port before freeing the IoOps.
  // The per-worker drain model means the loop normally reaps to empty before
  // teardown, so `inflight` is usually empty and none of this runs; this handles
  // the case where it isn't. CancelIoEx is asynchronous and a port-associated
  // OVERLAPPED always completes via the PORT (not the OVERLAPPED's hEvent, which
  // we never set) -- so the cancelled op's completion is posted to the port, and
  // we must collect it there before freeing the IoOp, or we'd free an OVERLAPPED
  // the kernel is still writing (a shutdown UAF).
  for (Impl::IoOp* op = impl_->inflight; op != nullptr; op = op->next) {
    if (op->handle != nullptr) ::CancelIoEx(op->handle, &op->ov);
  }
  OVERLAPPED_ENTRY ents[64];
  while (impl_->inflight != nullptr) {
    ULONG removed = 0;
    BOOL ok = ::GetQueuedCompletionStatusEx(impl_->port, ents, 64, &removed,
                                            1000, FALSE);
    if (!ok || removed == 0) break;  // timed out -> stop (don't hang teardown)
    for (ULONG i = 0; i < removed; ++i) {
      if (ents[i].lpOverlapped == nullptr) continue;  // a stray post(), ignore
      auto* op = reinterpret_cast<Impl::IoOp*>(ents[i].lpOverlapped);
      impl_->unlink(op);
      if (op->accept_sock != INVALID_SOCKET) ::closesocket(op->accept_sock);
      delete op;
    }
  }
  // Free the port FIRST to sever the kernel's path to any residual OVERLAPPED,
  // then free any op that never came back (unreachable in practice: a cancelled
  // op's completion is always posted -- this only guards a wedged kernel so
  // teardown can't hang or leak).
  if (impl_->port != nullptr) ::CloseHandle(impl_->port);
  for (Impl::IoOp* op = impl_->inflight; op != nullptr;) {
    Impl::IoOp* nx = op->next;
    if (op->accept_sock != INVALID_SOCKET) ::closesocket(op->accept_sock);
    delete op;
    op = nx;
  }
  if (impl_->wsa_started) ::WSACleanup();
  delete impl_;
  impl_ = nullptr;
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
  // Winsock is needed only for the socket ops; a file-only proactor never touches
  // it. Startup here is cheap and ref-counted, so do it eagerly and keep the
  // socket-submit paths free of a per-call init check. Non-fatal if it fails --
  // file read/write still work.
  WSADATA wsa;
  if (::WSAStartup(MAKEWORD(2, 2), &wsa) == 0) impl_->wsa_started = true;
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
    if (entries[i].lpOverlapped == nullptr) {
      // A post() synthetic completion: key = the Waker token, bytes = the posted
      // result. (Real ops always carry an OVERLAPPED; this is how they differ.)
      out[i].user_data = static_cast<std::uint64_t>(entries[i].lpCompletionKey);
      out[i].result =
          static_cast<std::int64_t>(entries[i].dwNumberOfBytesTransferred);
      out[i].flags = 0;
      out[i].buffer_id = 0;
      continue;
    }
    // A real op: OVERLAPPED was the first member of the IoOp, so cast straight
    // back. Its lifetime ends here -- unlink from the in-flight list and free.
    auto* op = reinterpret_cast<Impl::IoOp*>(entries[i].lpOverlapped);
    impl_->unlink(op);
    out[i].user_data = op->user_data;
    out[i].flags = 0;
    out[i].buffer_id = 0;
    DWORD xfer = 0;
    // GetOverlappedResult (no wait -- the op is already complete) is the
    // documented way to read a per-op error out of a batch-dequeued completion:
    // dwNumberOfBytesTransferred alone can't distinguish 0-byte EOF from failure.
    BOOL rok = ::GetOverlappedResult(op->handle, &op->ov, &xfer, FALSE);
    if (!rok) {
      out[i].result = neg_err(::GetLastError());  // negative Win32 error
      if (op->accept_sock != INVALID_SOCKET) ::closesocket(op->accept_sock);
    } else if (op->kind == Impl::kAccept) {
      // Finalize the accepted socket so it inherits the listener's properties,
      // then hand its fd back as the result (>= 0). Ownership passes to caller.
      SOCKET ls = reinterpret_cast<SOCKET>(op->handle);
      ::setsockopt(op->accept_sock, SOL_SOCKET, SO_UPDATE_ACCEPT_CONTEXT,
                   reinterpret_cast<char*>(&ls), sizeof(ls));
      out[i].result = static_cast<std::int64_t>(op->accept_sock);
    } else if (op->kind == Impl::kConnect) {
      ::setsockopt(reinterpret_cast<SOCKET>(op->handle), SOL_SOCKET,
                   SO_UPDATE_CONNECT_CONTEXT, nullptr, 0);
      out[i].result = 0;  // connected
    } else {
      out[i].result = static_cast<std::int64_t>(xfer);  // bytes transferred
    }
    delete op;
  }
  return static_cast<std::size_t>(removed);
}

bool Proactor::post(std::uint64_t user_data, std::int64_t result) noexcept {
  if (impl_ == nullptr) return false;
  // Post a synthetic completion: key = user_data (the Waker token), bytes =
  // result. Wakes a blocked reap() from any thread AND drives tests without a
  // real fd. OVERLAPPED is nullptr -- reap() uses that to tell it from a real op.
  return ::PostQueuedCompletionStatus(
             impl_->port,
             static_cast<DWORD>(static_cast<std::uint64_t>(result)),
             static_cast<ULONG_PTR>(user_data), nullptr) != 0;
}

bool Proactor::submit_read(std::intptr_t fd, void* buf, std::uint32_t len,
                           std::uint64_t offset,
                           std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  auto* op = new (std::nothrow) Impl::IoOp{};
  if (op == nullptr) return false;
  op->user_data = user_data;
  op->handle = reinterpret_cast<HANDLE>(fd);
  op->kind = Impl::kRead;
  op->ov.Offset = static_cast<DWORD>(offset & 0xffffffffull);
  op->ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
  impl_->link(op);
  BOOL ok = ::ReadFile(op->handle, buf, len, nullptr, &op->ov);
  // Async success queues a completion too (we don't set SKIP_ON_SUCCESS), so only
  // a synchronous failure that ISN'T "pending" means the op won't complete.
  if (ok || ::GetLastError() == ERROR_IO_PENDING) return true;
  impl_->unlink(op);
  delete op;
  return false;
}

bool Proactor::submit_write(std::intptr_t fd, const void* buf,
                            std::uint32_t len, std::uint64_t offset,
                            std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  auto* op = new (std::nothrow) Impl::IoOp{};
  if (op == nullptr) return false;
  op->user_data = user_data;
  op->handle = reinterpret_cast<HANDLE>(fd);
  op->kind = Impl::kWrite;
  op->ov.Offset = static_cast<DWORD>(offset & 0xffffffffull);
  op->ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
  impl_->link(op);
  BOOL ok = ::WriteFile(op->handle, buf, len, nullptr, &op->ov);
  if (ok || ::GetLastError() == ERROR_IO_PENDING) return true;
  impl_->unlink(op);
  delete op;
  return false;
}

bool Proactor::submit_recv(std::intptr_t fd, void* buf, std::uint32_t len,
                           std::uint32_t flags,
                           std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  auto* op = new (std::nothrow) Impl::IoOp{};
  if (op == nullptr) return false;
  op->user_data = user_data;
  op->handle = reinterpret_cast<HANDLE>(fd);
  op->kind = Impl::kRecv;
  op->wsabuf.buf = static_cast<CHAR*>(buf);  // buffer descriptor outlives the op
  op->wsabuf.len = len;
  impl_->link(op);
  DWORD f = flags;
  int r = ::WSARecv(static_cast<SOCKET>(fd), &op->wsabuf, 1, nullptr, &f,
                    &op->ov, nullptr);
  if (r == 0 || ::WSAGetLastError() == WSA_IO_PENDING) return true;
  impl_->unlink(op);
  delete op;
  return false;
}

bool Proactor::submit_send(std::intptr_t fd, const void* buf, std::uint32_t len,
                           std::uint32_t flags,
                           std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  auto* op = new (std::nothrow) Impl::IoOp{};
  if (op == nullptr) return false;
  op->user_data = user_data;
  op->handle = reinterpret_cast<HANDLE>(fd);
  op->kind = Impl::kSend;
  op->wsabuf.buf = static_cast<CHAR*>(const_cast<void*>(buf));
  op->wsabuf.len = len;
  impl_->link(op);
  int r = ::WSASend(static_cast<SOCKET>(fd), &op->wsabuf, 1, nullptr, flags,
                    &op->ov, nullptr);
  if (r == 0 || ::WSAGetLastError() == WSA_IO_PENDING) return true;
  impl_->unlink(op);
  delete op;
  return false;
}

bool Proactor::submit_accept(std::intptr_t fd, void* /*addr*/, void* /*addrlen*/,
                             std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  SOCKET ls = static_cast<SOCKET>(fd);
  if (!load_ext_fn(ls, WSAID_ACCEPTEX,
                   reinterpret_cast<void**>(&impl_->acceptex)))
    return false;
  // AcceptEx needs a pre-created accept socket of the same family as the listener
  // (learn it via getsockname). The peer address the caller passed is ignored on
  // Windows -- AcceptEx writes both addrs into the op's own addr_buf; net/ reads
  // the peer with getpeername on the accepted socket after completion.
  SOCKADDR_STORAGE ss;
  int slen = static_cast<int>(sizeof(ss));
  int fam = AF_INET;
  if (::getsockname(ls, reinterpret_cast<sockaddr*>(&ss), &slen) == 0)
    fam = ss.ss_family;
  SOCKET as = ::WSASocketW(fam, SOCK_STREAM, 0, nullptr, 0, WSA_FLAG_OVERLAPPED);
  if (as == INVALID_SOCKET) return false;
  auto* op = new (std::nothrow) Impl::IoOp{};
  if (op == nullptr) {
    ::closesocket(as);
    return false;
  }
  op->user_data = user_data;
  op->handle = reinterpret_cast<HANDLE>(fd);  // listener (for SO_UPDATE + result)
  op->kind = Impl::kAccept;
  op->accept_sock = as;
  impl_->link(op);
  constexpr DWORD kAddr = sizeof(SOCKADDR_STORAGE) + 16;
  DWORD got = 0;
  BOOL ok = impl_->acceptex(ls, as, op->addr_buf, 0, kAddr, kAddr, &got, &op->ov);
  if (ok || ::WSAGetLastError() == WSA_IO_PENDING) return true;
  impl_->unlink(op);
  ::closesocket(as);
  delete op;
  return false;
}

bool Proactor::submit_connect(std::intptr_t fd, const void* addr,
                              std::uint32_t addrlen,
                              std::uint64_t user_data) noexcept {
  if (impl_ == nullptr || addr == nullptr) return false;
  SOCKET s = static_cast<SOCKET>(fd);
  if (!load_ext_fn(s, WSAID_CONNECTEX,
                   reinterpret_cast<void**>(&impl_->connectex)))
    return false;
  // ConnectEx requires the socket be already bound. Bind to the wildcard address
  // of the target's family (a no-op error if it is already bound).
  auto family = reinterpret_cast<const sockaddr*>(addr)->sa_family;
  if (family == AF_INET6) {
    sockaddr_in6 any{};
    any.sin6_family = AF_INET6;
    ::bind(s, reinterpret_cast<sockaddr*>(&any), sizeof(any));
  } else {
    sockaddr_in any{};
    any.sin_family = AF_INET;
    ::bind(s, reinterpret_cast<sockaddr*>(&any), sizeof(any));
  }
  auto* op = new (std::nothrow) Impl::IoOp{};
  if (op == nullptr) return false;
  op->user_data = user_data;
  op->handle = reinterpret_cast<HANDLE>(fd);
  op->kind = Impl::kConnect;
  impl_->link(op);
  BOOL ok = impl_->connectex(s, reinterpret_cast<const sockaddr*>(addr),
                             static_cast<int>(addrlen), nullptr, 0, nullptr,
                             &op->ov);
  if (ok || ::WSAGetLastError() == WSA_IO_PENDING) return true;
  impl_->unlink(op);
  delete op;
  return false;
}

bool Proactor::register_handle(std::intptr_t fd) noexcept {
  if (impl_ == nullptr) return false;
  // Associate the handle/socket with the port so its ops complete here. The
  // per-op token rides the OVERLAPPED, so the completion KEY is unused (0).
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
#  include <sys/epoll.h>
#  include <sys/mman.h>
#  include <sys/socket.h>
#  include <sys/syscall.h>
#  include <unistd.h>

#  include <atomic>
#  include <cerrno>
#  include <cstdlib>
#  include <cstring>
#  include <unordered_map>
#  include <vector>

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
  // Backend chosen at init: io_uring is the SOTA proactor (default, always tried
  // first); epoll is the REACTOR fallback used ONLY when io_uring is unavailable
  // (old kernel ENOSYS, or disabled/blocked by sysctl kernel.io_uring_disabled /
  // container seccomp -> EPERM). Production takes io_uring wherever it exists;
  // epoll keeps HYLE runnable where io_uring is turned off (common in hardened
  // containers). Tests can force epoll to exercise the fallback path.
  enum Backend { kUring, kEpoll };
  Backend backend = kUring;

  // ---- io_uring backend state ----
  int fd = -1;
  void* sq_ptr = nullptr;
  std::size_t sq_sz = 0;
  void* cq_ptr = nullptr;        // == sq_ptr when SINGLE_MMAP
  std::size_t cq_sz = 0;
  io_uring_sqe* sqes = nullptr;
  std::size_t sqes_sz = 0;
  unsigned sq_entries = 0;
  bool single_mmap = false;
  unsigned* sq_tail = nullptr;
  unsigned* sq_ring_mask = nullptr;
  unsigned* sq_array = nullptr;
  unsigned* cq_head = nullptr;
  unsigned* cq_tail = nullptr;
  unsigned* cq_ring_mask = nullptr;
  io_uring_cqe* cqes = nullptr;

  // ---- epoll backend state (reactor -> completion emulation) ----
  // epoll only signals READINESS; the proactor must remember each armed op and
  // perform the IO itself when the fd becomes ready, then synthesize a
  // Completion. One pending op per fd (EPOLLONESHOT), which suffices for the
  // one-in-flight-op-per-fd model io-04 net uses. `opcode` reuses the IORING_OP_*
  // tags (READ/WRITE for byte IO, ACCEPT/CONNECT for the socket handshakes) so
  // reap() can branch on which syscall to run when the fd fires.
  int epfd = -1;
  struct PendingOp {
    std::uint8_t opcode;   // IORING_OP_READ/WRITE/ACCEPT/CONNECT as the op tag
    void* buf;
    std::uint32_t len;
    std::uint64_t user_data;
  };
  std::unordered_map<int, PendingOp> pending;  // fd -> armed op
  // post() has no fd (a bare wake token) and epoll has no
  // PostQueuedCompletionStatus, so injected wakes queue here; reap drains them
  // first. std in the .cc impl is allowed (conventions §1: internal, non-hot).
  struct Posted {
    std::uint64_t user_data;
    std::int64_t result;
  };
  std::vector<Posted> posted;
};

Proactor::~Proactor() {
  if (impl_ == nullptr) return;
  if (impl_->backend == Impl::kEpoll) {
    if (impl_->epfd >= 0) ::close(impl_->epfd);
  } else {
    if (impl_->sqes != nullptr) ::munmap(impl_->sqes, impl_->sqes_sz);
    if (impl_->cq_ptr != nullptr && !impl_->single_mmap)
      ::munmap(impl_->cq_ptr, impl_->cq_sz);
    if (impl_->sq_ptr != nullptr) ::munmap(impl_->sq_ptr, impl_->sq_sz);
    if (impl_->fd >= 0) ::close(impl_->fd);
  }
  delete impl_;
  impl_ = nullptr;
}

namespace {
// (io_uring ring helpers live here; backend-init + epoll-arm are Proactor static
// members below since they must name the private Impl.)
}  // namespace

// Try to stand up the io_uring backend (the SOTA path). Returns the filled Impl
// or nullptr if io_uring is unavailable (old kernel ENOSYS / disabled EPERM /
// mmap failure) -- caller then falls back to epoll.
Proactor::Impl* Proactor::try_init_uring(unsigned entries) noexcept {
  io_uring_params p;
  std::memset(&p, 0, sizeof(p));
  int fd = sys_io_uring_setup(entries, &p);
  if (fd < 0) return nullptr;  // ENOSYS(old kernel) / EPERM(sysctl/seccomp disabled)

  auto* impl = new (std::nothrow) Proactor::Impl{};
  if (impl == nullptr) {
    ::close(fd);
    return nullptr;
  }
  impl->backend = Proactor::Impl::kUring;
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
    return nullptr;
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
    return nullptr;
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
    return nullptr;
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
  return impl;
}

// Stand up the epoll fallback backend (reactor). Returns nullptr on failure.
Proactor::Impl* Proactor::try_init_epoll() noexcept {
  int ep = ::epoll_create1(EPOLL_CLOEXEC);
  if (ep < 0) return nullptr;
  auto* impl = new (std::nothrow) Proactor::Impl{};
  if (impl == nullptr) {
    ::close(ep);
    return nullptr;
  }
  impl->backend = Proactor::Impl::kEpoll;
  impl->epfd = ep;
  return impl;
}

bool Proactor::init(unsigned entries) noexcept {
  if (impl_ != nullptr) return true;  // idempotent
  // SOTA-first: always try io_uring; production runs on it wherever it exists.
  // Fall back to epoll ONLY when io_uring is truly unavailable (old kernel /
  // sysctl-disabled / seccomp-blocked). A test-only env var forces the epoll
  // path so the fallback is exercised even where io_uring works (this machine).
  const char* force = ::getenv("HYLE_IO_FORCE_EPOLL");
  if (force == nullptr || force[0] == '0' || force[0] == '\0') {
    Impl* u = try_init_uring(entries);
    if (u != nullptr) {
      impl_ = u;
      return true;
    }
    // io_uring unavailable -> fall through to epoll (the whole point of io-03).
  }
  Impl* e = try_init_epoll();
  if (e == nullptr) return false;
  impl_ = e;
  return true;
}

std::size_t Proactor::reap(Completion* out, std::size_t max,
                           std::uint64_t timeout_ns) noexcept {
  if (impl_ == nullptr || max == 0) return 0;

  if (impl_->backend == Impl::kEpoll) {
    // Reactor -> completion: drain injected post()s first (they need no fd),
    // then epoll_wait for ready fds and perform each armed op inline, turning
    // "ready" into a synthesized Completion (bytes, or negative errno).
    std::size_t n = 0;
    while (n < max && !impl_->posted.empty()) {
      Impl::Posted pd = impl_->posted.back();
      impl_->posted.pop_back();
      out[n].user_data = pd.user_data;
      out[n].result = pd.result;
      out[n].flags = 0;
      out[n].buffer_id = 0;
      ++n;
    }
    if (n > 0) return n;  // deliver wakes promptly; caller loops for more

    int ms;
    if (timeout_ns == 0) {
      ms = 0;
    } else if (timeout_ns == UINT64_MAX) {
      ms = -1;  // block
    } else {
      std::uint64_t r = (timeout_ns + 999'999ull) / 1'000'000ull;
      ms = r > 2'000'000'000ull ? 2'000'000'000 : static_cast<int>(r);
    }
    epoll_event evs[64];
    int want = static_cast<int>(max < 64 ? max : 64);
    int got = ::epoll_wait(impl_->epfd, evs, want, ms);
    if (got <= 0) return 0;  // timeout / EINTR -> nothing this round

    for (int i = 0; i < got && n < max; ++i) {
      int fd = static_cast<int>(evs[i].data.u64);  // we store fd in data.u64
      auto it = impl_->pending.find(fd);
      if (it == impl_->pending.end()) continue;  // stale (already served)
      Impl::PendingOp op = it->second;
      impl_->pending.erase(it);
      ::epoll_ctl(impl_->epfd, EPOLL_CTL_DEL, fd, nullptr);  // one-shot done
      // Perform the IO inline now that the fd is ready. The op tag says which.
      std::int64_t res;
      if (op.opcode == IORING_OP_WRITE) {
        ssize_t w = ::write(fd, op.buf, op.len);
        res = (w < 0) ? -errno : static_cast<std::int64_t>(w);
      } else if (op.opcode == IORING_OP_ACCEPT) {
        // Listener readable -> a connection is queued; accept it. Result is the
        // accepted fd (>= 0) or a negative errno, matching io_uring ACCEPT.
        int as = ::accept(fd, nullptr, nullptr);
        res = (as < 0) ? -errno : static_cast<std::int64_t>(as);
      } else if (op.opcode == IORING_OP_CONNECT) {
        // Non-blocking connect signalled writable -> read SO_ERROR to learn the
        // real outcome (0 = connected, else the errno). Result 0 or -errno.
        int soerr = 0;
        socklen_t sl = sizeof(soerr);
        ::getsockopt(fd, SOL_SOCKET, SO_ERROR, &soerr, &sl);
        res = (soerr == 0) ? 0 : -soerr;
      } else {  // READ
        ssize_t r = ::read(fd, op.buf, op.len);
        res = (r < 0) ? -errno : static_cast<std::int64_t>(r);
      }
      out[n].user_data = op.user_data;
      out[n].result = res;
      out[n].flags = 0;
      out[n].buffer_id = 0;
      ++n;
    }
    return n;
  }

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
// Shared by post/read/write/recv/send/accept/connect. addr/len/off are 0 for
// NOP. `op_flags` sets the SQE's flags union (rw_flags aliases msg_flags for
// recv/send and accept_flags for accept -- same 32 bits); 0 for read/write/NOP.
bool submit_one(int ring_fd, unsigned* sq_tail, unsigned sq_mask,
                unsigned* sq_array, io_uring_sqe* sqes, std::uint8_t opcode,
                std::int32_t fd, std::uint64_t addr, std::uint32_t len,
                std::uint64_t off, std::uint64_t user_data,
                std::uint32_t op_flags = 0) noexcept {
  unsigned tail = *sq_tail;
  unsigned index = tail & sq_mask;
  io_uring_sqe& s = sqes[index];
  std::memset(&s, 0, sizeof(s));
  s.opcode = opcode;
  s.fd = fd;
  s.off = off;
  s.addr = addr;
  s.len = len;
  s.rw_flags = static_cast<int>(op_flags);  // union: msg_flags/accept_flags too
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
  if (impl_->backend == Impl::kEpoll) {
    // epoll has no PostQueuedCompletionStatus: queue a software completion that
    // reap() drains first. (Single-threaded per-worker use; cross-thread post to
    // an epoll proactor is not the model -- io_uring/IOCP are the MT-wake paths.)
    impl_->posted.push_back({user_data, result});
    return true;
  }
  // NOP SQE carrying user_data: backend-independent "wake/inject" hook. NOP res
  // is always 0, so `result` is not delivered on Linux -- post() is a wake hook,
  // not a data channel (IOCP can carry bytes; asymmetry documented in header).
  (void)result;
  return submit_one(impl_->fd, impl_->sq_tail, *impl_->sq_ring_mask,
                    impl_->sq_array, impl_->sqes, IORING_OP_NOP, -1, 0, 0, 0,
                    user_data);
}

// epoll fallback: arm a one-shot readiness watch for `fd` and remember the op;
// reap() does the actual read/write when the fd fires. `events` is EPOLLIN for
// read, EPOLLOUT for write. Stores fd in data.u64 so reap recovers it.
bool Proactor::epoll_arm(Impl* impl, int fd, unsigned events,
                         unsigned char opcode, void* buf, unsigned len,
                         std::uint64_t user_data) noexcept {
  impl->pending[fd] = Impl::PendingOp{opcode, buf, len, user_data};
  epoll_event ev{};
  ev.events = events | EPOLLONESHOT;
  ev.data.u64 = static_cast<std::uint64_t>(fd);
  // ADD, or MOD if this fd is already registered from a prior op.
  if (::epoll_ctl(impl->epfd, EPOLL_CTL_ADD, fd, &ev) < 0) {
    if (errno == EEXIST) {
      if (::epoll_ctl(impl->epfd, EPOLL_CTL_MOD, fd, &ev) < 0) {
        impl->pending.erase(fd);
        return false;
      }
    } else {
      impl->pending.erase(fd);
      return false;
    }
  }
  return true;
}

bool Proactor::submit_read(std::intptr_t fd, void* buf, std::uint32_t len,
                           std::uint64_t offset,
                           std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  if (impl_->backend == Impl::kEpoll) {
    return epoll_arm(impl_, static_cast<int>(fd), EPOLLIN, IORING_OP_READ, buf,
                     len, user_data);
  }
  return submit_one(impl_->fd, impl_->sq_tail, *impl_->sq_ring_mask,
                    impl_->sq_array, impl_->sqes, IORING_OP_READ,
                    static_cast<std::int32_t>(fd),
                    reinterpret_cast<std::uint64_t>(buf), len, offset, user_data);
}

bool Proactor::submit_write(std::intptr_t fd, const void* buf,
                            std::uint32_t len, std::uint64_t offset,
                            std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  if (impl_->backend == Impl::kEpoll) {
    return epoll_arm(impl_, static_cast<int>(fd), EPOLLOUT, IORING_OP_WRITE,
                     const_cast<void*>(buf), len, user_data);
  }
  return submit_one(impl_->fd, impl_->sq_tail, *impl_->sq_ring_mask,
                    impl_->sq_array, impl_->sqes, IORING_OP_WRITE,
                    static_cast<std::int32_t>(fd),
                    reinterpret_cast<std::uint64_t>(buf), len, offset, user_data);
}

bool Proactor::submit_recv(std::intptr_t fd, void* buf, std::uint32_t len,
                           std::uint32_t flags,
                           std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  if (impl_->backend == Impl::kEpoll) {
    // On a socket, read() == recv(flags=0); the epoll reactor does read() when
    // EPOLLIN fires. Non-zero recv flags aren't modeled on the epoll fallback.
    return epoll_arm(impl_, static_cast<int>(fd), EPOLLIN, IORING_OP_READ, buf,
                     len, user_data);
  }
  return submit_one(impl_->fd, impl_->sq_tail, *impl_->sq_ring_mask,
                    impl_->sq_array, impl_->sqes, IORING_OP_RECV,
                    static_cast<std::int32_t>(fd),
                    reinterpret_cast<std::uint64_t>(buf), len, 0, user_data,
                    flags);
}

bool Proactor::submit_send(std::intptr_t fd, const void* buf, std::uint32_t len,
                           std::uint32_t flags,
                           std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  if (impl_->backend == Impl::kEpoll) {
    return epoll_arm(impl_, static_cast<int>(fd), EPOLLOUT, IORING_OP_WRITE,
                     const_cast<void*>(buf), len, user_data);
  }
  return submit_one(impl_->fd, impl_->sq_tail, *impl_->sq_ring_mask,
                    impl_->sq_array, impl_->sqes, IORING_OP_SEND,
                    static_cast<std::int32_t>(fd),
                    reinterpret_cast<std::uint64_t>(buf), len, 0, user_data,
                    flags);
}

bool Proactor::submit_accept(std::intptr_t fd, void* addr, void* addrlen,
                             std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  if (impl_->backend == Impl::kEpoll) {
    // Reactor accept: watch the listener for EPOLLIN (a queued connection makes
    // it readable); reap() runs ::accept when it fires and returns the fd. The
    // peer addr is read by net/ via getpeername on the accepted fd, so buf/len
    // stay 0 (the caller's addr/addrlen aren't populated on the epoll path).
    return epoll_arm(impl_, static_cast<int>(fd), EPOLLIN, IORING_OP_ACCEPT,
                     nullptr, 0, user_data);
  }
  // accept: addr -> sqe.addr (sockaddr*), addrlen -> sqe.addr2 (socklen_t*).
  // submit_one puts its `off` arg into sqe.off which unions with addr2, so pass
  // addrlen as the off argument.
  return submit_one(impl_->fd, impl_->sq_tail, *impl_->sq_ring_mask,
                    impl_->sq_array, impl_->sqes, IORING_OP_ACCEPT,
                    static_cast<std::int32_t>(fd),
                    reinterpret_cast<std::uint64_t>(addr), 0,
                    reinterpret_cast<std::uint64_t>(addrlen), user_data);
}

bool Proactor::submit_connect(std::intptr_t fd, const void* addr,
                              std::uint32_t addrlen,
                              std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  if (impl_->backend == Impl::kEpoll) {
    // Reactor connect: kick off the non-blocking connect now. If it completes
    // synchronously (rare, e.g. UNIX socket), synthesize the completion via
    // post() so the caller still reaps one. Otherwise EINPROGRESS -> watch
    // EPOLLOUT; reap() reads SO_ERROR when writable to learn the outcome.
    int r = ::connect(static_cast<int>(fd),
                      reinterpret_cast<const sockaddr*>(addr),
                      static_cast<socklen_t>(addrlen));
    if (r == 0) return post(user_data, 0);            // connected immediately
    if (errno != EINPROGRESS) return post(user_data, -errno);  // hard failure
    return epoll_arm(impl_, static_cast<int>(fd), EPOLLOUT, IORING_OP_CONNECT,
                     nullptr, 0, user_data);
  }
  // connect: addr -> sqe.addr (sockaddr*), addrlen -> sqe.off.
  return submit_one(impl_->fd, impl_->sq_tail, *impl_->sq_ring_mask,
                    impl_->sq_array, impl_->sqes, IORING_OP_CONNECT,
                    static_cast<std::int32_t>(fd),
                    reinterpret_cast<std::uint64_t>(addr), 0,
                    static_cast<std::uint64_t>(addrlen), user_data);
}

bool Proactor::register_handle(std::intptr_t /*fd*/) noexcept {
  // Neither Linux backend needs a pre-registration step: io_uring puts the fd in
  // each SQE; epoll adds the fd at submit time (with EPOLLONESHOT).
  return impl_ != nullptr;
}

}  // namespace hyle::io

#else
#  error "HYLE io/proactor: unsupported OS"
#endif
