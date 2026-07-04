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

  enum OpKind : int {
    kRead, kWrite, kRecv, kSend, kAccept, kConnect, kRecvFrom, kSendTo,
    kFileMeta  // io-05: open/close/statx/fsync/fallocate done SYNCHRONOUSLY here
               // (no async form on IOCP) then posted with the result already in
               // syn_result -- reap skips GetOverlappedResult for these.
  };
  struct IoOp {
    OVERLAPPED ov{};                      // MUST be first (lpOverlapped -> IoOp*)
    std::uint64_t user_data = 0;
    HANDLE handle = nullptr;              // op's handle/socket (GetOverlappedResult)
    int kind = 0;
    WSABUF wsabuf{};                      // recv/send descriptor (outlives the op)
    SOCKET accept_sock = INVALID_SOCKET;  // AcceptEx: the accepted socket (result)
    // Multishot accept emulation (io-07): IOCP has no native multishot, so reap
    // re-issues AcceptEx on `handle` (the listener) after each completion and
    // stamps Completion::kMore -- mirroring io_uring's multishot upward. Set only
    // on a multishot accept op.
    bool multishot = false;
    // io-05 synthetic-completion path: a kFileMeta op runs its syscall at submit
    // time (open/stat/fsync/fallocate are synchronous on Windows), stores the
    // outcome here (a fd/handle >= 0, or 0, or a negative errno -- 64-bit so an
    // opened HANDLE isn't truncated the way post()'s DWORD result would be), then
    // posts itself to the port. reap sees kFileMeta and hands back syn_result
    // verbatim instead of calling GetOverlappedResult.
    std::int64_t syn_result = 0;
    // Datagram (WSARecvFrom/WSASendTo) only: WSARecvFrom writes the source addr
    // into the CALLER's addr buffer, and writes its length into `fromlen` -- both
    // must outlive the async op, so fromlen lives here (the addr buffer is caller-
    // owned on the fiber stack). Unused (0) for stream ops.
    int fromlen = 0;
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
    if (op->kind == Impl::kFileMeta) {
      // io-05: the syscall already ran at submit; hand back its stored 64-bit
      // result (fd/0/-errno) untouched -- no GetOverlappedResult (there was no
      // real overlapped IO). This is the open/close/statx/fsync/fallocate path.
      out[i].result = op->syn_result;
      delete op;
      continue;
    }
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
    // Multishot accept emulation: on a successful accept, re-arm AcceptEx on the
    // listener and stamp kMore (more connections coming, same user_data) --
    // mirrors io_uring native multishot upward. On failure leave kMore clear =
    // terminal (the multishot ended). Re-arm failure also ends it (kMore clear).
    if (op->multishot && op->kind == Impl::kAccept && out[i].result >= 0) {
      if (iocp_arm_accept(impl_, reinterpret_cast<std::intptr_t>(op->handle),
                          op->user_data, /*multishot=*/true))
        out[i].flags |= Completion::kMore;
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

bool Proactor::wake() noexcept {
  if (impl_ == nullptr) return false;
  // PostQueuedCompletionStatus IS the thread-safe cross-thread wake on IOCP: post
  // the reserved wake token with a nullptr OVERLAPPED. reap() sees lpOverlapped ==
  // nullptr (a synthetic completion) and, because the key is kWakeToken, the
  // completion router (io_context/net) drops it -- it only breaks the blocking
  // GetQueuedCompletionStatusEx so the worker re-checks its run queue.
  return ::PostQueuedCompletionStatus(
             impl_->port, 0, static_cast<ULONG_PTR>(kWakeToken), nullptr) != 0;
}

bool Proactor::in_flight() const noexcept {
  // A real op links an IoOp onto `inflight` for its lifetime; the list being
  // non-empty means the port still owes ≥1 completion. IOCP surfaces completions
  // via the port whether or not the owner blocks, so this gate is not the strict
  // correctness lever it is on io_uring -- but the scheduler uses the same signal
  // on both platforms (block in the port when ops are outstanding, else park for
  // an interruptible wake), so report it uniformly.
  return impl_ != nullptr && impl_->inflight != nullptr;
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

// Arm one AcceptEx on listener `fd`. Shared by submit_accept (one-shot),
// submit_accept_multishot, and reap's multishot re-arm. `multishot` is stamped on
// the IoOp so reap knows to re-arm + set kMore after this one completes.
bool Proactor::iocp_arm_accept(Impl* impl, std::intptr_t fd,
                               std::uint64_t user_data, bool multishot) noexcept {
  SOCKET ls = static_cast<SOCKET>(fd);
  if (!load_ext_fn(ls, WSAID_ACCEPTEX,
                   reinterpret_cast<void**>(&impl->acceptex)))
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
  auto* op = new (std::nothrow) Proactor::Impl::IoOp{};
  if (op == nullptr) {
    ::closesocket(as);
    return false;
  }
  op->user_data = user_data;
  op->handle = reinterpret_cast<HANDLE>(fd);  // listener (for SO_UPDATE + result)
  op->kind = Proactor::Impl::kAccept;
  op->accept_sock = as;
  op->multishot = multishot;
  impl->link(op);
  constexpr DWORD kAddr = sizeof(SOCKADDR_STORAGE) + 16;
  DWORD got = 0;
  BOOL ok = impl->acceptex(ls, as, op->addr_buf, 0, kAddr, kAddr, &got, &op->ov);
  if (ok || ::WSAGetLastError() == WSA_IO_PENDING) return true;
  impl->unlink(op);
  ::closesocket(as);
  delete op;
  return false;
}

bool Proactor::submit_accept(std::intptr_t fd, void* /*addr*/, void* /*addrlen*/,
                             std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  return iocp_arm_accept(impl_, fd, user_data, /*multishot=*/false);
}

bool Proactor::submit_accept_multishot(std::intptr_t fd,
                                       std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  // IOCP has no native multishot: arm one AcceptEx flagged multishot; reap
  // re-arms after each completion and stamps kMore (emulation, uniform contract).
  return iocp_arm_accept(impl_, fd, user_data, /*multishot=*/true);
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

bool Proactor::submit_recvfrom(std::intptr_t fd, void* buf, std::uint32_t len,
                               void* addr, std::uint32_t addr_cap,
                               void* /*scratch*/,
                               std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  auto* op = new (std::nothrow) Impl::IoOp{};
  if (op == nullptr) return false;
  op->user_data = user_data;
  op->handle = reinterpret_cast<HANDLE>(fd);
  op->kind = Impl::kRecvFrom;
  op->wsabuf.buf = static_cast<CHAR*>(buf);
  op->wsabuf.len = len;
  op->fromlen = static_cast<int>(addr_cap);  // WSARecvFrom reads cap / writes len
  impl_->link(op);
  DWORD f = 0;
  // WSARecvFrom writes the source address into the caller's `addr` and its length
  // into op->fromlen; both outlive the async op (addr on the fiber stack, fromlen
  // in the IoOp). scratch (the io_uring msghdr) is unused on IOCP.
  int r = ::WSARecvFrom(static_cast<SOCKET>(fd), &op->wsabuf, 1, nullptr, &f,
                        static_cast<sockaddr*>(addr),
                        addr ? &op->fromlen : nullptr, &op->ov, nullptr);
  if (r == 0 || ::WSAGetLastError() == WSA_IO_PENDING) return true;
  impl_->unlink(op);
  delete op;
  return false;
}

bool Proactor::submit_sendto(std::intptr_t fd, const void* buf, std::uint32_t len,
                             const void* addr, std::uint32_t addrlen,
                             void* /*scratch*/,
                             std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  auto* op = new (std::nothrow) Impl::IoOp{};
  if (op == nullptr) return false;
  op->user_data = user_data;
  op->handle = reinterpret_cast<HANDLE>(fd);
  op->kind = Impl::kSendTo;
  op->wsabuf.buf = static_cast<CHAR*>(const_cast<void*>(buf));
  op->wsabuf.len = len;
  impl_->link(op);
  int r = ::WSASendTo(static_cast<SOCKET>(fd), &op->wsabuf, 1, nullptr, 0,
                      static_cast<const sockaddr*>(addr),
                      static_cast<int>(addrlen), &op->ov, nullptr);
  if (r == 0 || ::WSAGetLastError() == WSA_IO_PENDING) return true;
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

// ---- file ops (io-05) ------------------------------------------------------
// On Windows none of open/close/statx/fsync/fallocate has an async form (even a
// FILE_FLAG_OVERLAPPED handle opens synchronously), so we run the syscall NOW and
// synthesize a completion: allocate a kFileMeta IoOp, stash the 64-bit result, and
// post it to the port with its OWN OVERLAPPED so reap routes it like any real op
// (user_data intact) but reads syn_result instead of GetOverlappedResult. This is
// the same reactor->completion synthesis epoll uses for accept/connect.
bool Proactor::iocp_post_meta(Impl* impl, std::int64_t result,
                              std::uint64_t user_data) noexcept {
  auto* op = new (std::nothrow) Impl::IoOp{};
  if (op == nullptr) return false;
  op->user_data = user_data;
  op->kind = Impl::kFileMeta;
  op->syn_result = result;
  impl->link(op);
  // Post with the op's own OVERLAPPED (non-null) so reap treats it as a real op
  // (a null OVERLAPPED is a bare wake/post). Zero bytes / key -- reap uses kind.
  if (::PostQueuedCompletionStatus(impl->port, 0, 0, &op->ov) != 0) return true;
  impl->unlink(op);
  delete op;
  return false;
}

bool Proactor::submit_open(const char* path, std::uint32_t flags,
                           std::uint32_t /*mode*/,
                           std::uint64_t user_data) noexcept {
  if (impl_ == nullptr || path == nullptr) return false;
  // Map neutral OpenFlags onto CreateFileW access + disposition. FILE_FLAG_
  // OVERLAPPED so subsequent submit_read/write on this handle are truly async.
  DWORD access = 0;
  if (flags & kRead) access |= GENERIC_READ;
  if (flags & kWrite) access |= GENERIC_WRITE;
  if (flags & kAppend) access |= FILE_APPEND_DATA;
  DWORD disposition;
  if (flags & kCreate)
    disposition = (flags & kTruncate) ? CREATE_ALWAYS : OPEN_ALWAYS;
  else
    disposition = (flags & kTruncate) ? TRUNCATE_EXISTING : OPEN_EXISTING;
  // Widen the UTF-8 path to UTF-16 for CreateFileW (a short stack buffer covers
  // typical paths; longer ones fail with a negative result rather than truncate).
  wchar_t wpath[1024];
  int wn = ::MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath, 1024);
  if (wn == 0) return iocp_post_meta(impl_, -static_cast<std::int64_t>(::GetLastError()), user_data);
  HANDLE h = ::CreateFileW(wpath, access,
                           FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                           disposition, FILE_FLAG_OVERLAPPED, nullptr);
  if (h == INVALID_HANDLE_VALUE)
    return iocp_post_meta(impl_, -static_cast<std::int64_t>(::GetLastError()), user_data);
  // Associate with the port so this handle's later async read/write complete here.
  if (::CreateIoCompletionPort(h, impl_->port, 0, 0) == nullptr) {
    DWORD e = ::GetLastError();
    ::CloseHandle(h);
    return iocp_post_meta(impl_, -static_cast<std::int64_t>(e), user_data);
  }
  return iocp_post_meta(impl_, static_cast<std::int64_t>(reinterpret_cast<std::intptr_t>(h)),
                        user_data);
}

bool Proactor::submit_close(std::intptr_t fd,
                            std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  BOOL ok = ::CloseHandle(reinterpret_cast<HANDLE>(fd));
  return iocp_post_meta(impl_, ok ? 0 : -static_cast<std::int64_t>(::GetLastError()),
                        user_data);
}

bool Proactor::submit_statx(std::intptr_t fd, FileStat* out,
                            std::uint64_t user_data) noexcept {
  if (impl_ == nullptr || out == nullptr) return false;
  BY_HANDLE_FILE_INFORMATION info;
  if (::GetFileInformationByHandle(reinterpret_cast<HANDLE>(fd), &info) == 0)
    return iocp_post_meta(impl_, -static_cast<std::int64_t>(::GetLastError()), user_data);
  *out = FileStat{};
  out->size = (static_cast<std::uint64_t>(info.nFileSizeHigh) << 32) |
              info.nFileSizeLow;
  // FILETIME is 100ns ticks since 1601-01-01; convert to ns since Unix epoch.
  std::uint64_t ft = (static_cast<std::uint64_t>(info.ftLastWriteTime.dwHighDateTime) << 32) |
                     info.ftLastWriteTime.dwLowDateTime;
  constexpr std::uint64_t kEpochDiff100ns = 116444736000000000ull;  // 1601->1970
  out->mtime_ns = (ft > kEpochDiff100ns) ? (ft - kEpochDiff100ns) * 100ull : 0;
  out->is_dir = (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
  out->mode = 0;  // no POSIX mode on Windows
  return iocp_post_meta(impl_, 0, user_data);
}

bool Proactor::submit_fsync(std::intptr_t fd, bool /*datasync*/,
                            std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  // Windows has no data-only flush; FlushFileBuffers does the full sync either way.
  BOOL ok = ::FlushFileBuffers(reinterpret_cast<HANDLE>(fd));
  return iocp_post_meta(impl_, ok ? 0 : -static_cast<std::int64_t>(::GetLastError()),
                        user_data);
}

bool Proactor::submit_fallocate(std::intptr_t fd, std::uint32_t /*mode*/,
                                std::uint64_t offset, std::uint64_t len,
                                std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  // Windows equivalent: seek to offset+len and SetEndOfFile to extend/allocate.
  // (No sparse/keep-size mode distinction -- mode is ignored, as documented.)
  HANDLE h = reinterpret_cast<HANDLE>(fd);
  LARGE_INTEGER end;
  end.QuadPart = static_cast<LONGLONG>(offset + len);
  LARGE_INTEGER cur;
  cur.QuadPart = 0;
  // Save current position, set EOF at offset+len, restore position.
  LARGE_INTEGER zero;
  zero.QuadPart = 0;
  ::SetFilePointerEx(h, zero, &cur, FILE_CURRENT);
  BOOL ok = ::SetFilePointerEx(h, end, nullptr, FILE_BEGIN) &&
            ::SetEndOfFile(h);
  ::SetFilePointerEx(h, cur, nullptr, FILE_BEGIN);
  return iocp_post_meta(impl_, ok ? 0 : -static_cast<std::int64_t>(::GetLastError()),
                        user_data);
}


// Provided buffer ring (io-07b) is an io_uring-only mechanism -- IOCP has no
// equivalent (WSARecv uses caller-posted buffers). net/ detects the false return
// and falls back to a pre-posted-buffer recv on Windows. Honest not-supported.
bool Proactor::register_buf_ring(int, unsigned, unsigned) noexcept {
  return false;
}
bool Proactor::submit_recv_multishot(std::intptr_t, int,
                                     std::uint64_t) noexcept {
  return false;
}
void* Proactor::buf_ring_addr(int, std::uint32_t) noexcept { return nullptr; }
bool Proactor::buf_ring_return(int, std::uint32_t) noexcept { return false; }

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
#  include <sys/eventfd.h>
#  include <sys/mman.h>
#  include <sys/socket.h>
#  include <sys/stat.h>   // io-05: fstat + struct stat (statx synthesis)
#  include <sys/syscall.h>
#  include <fcntl.h>      // io-05: O_* open flags
#  include <unistd.h>

#  include <atomic>
#  include <cerrno>
#  include <cstdio>
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
// register op (io-07b provided buffer ring uses IORING_REGISTER_PBUF_RING). Raw
// syscall, no liburing link -- same policy as setup/enter above.
int sys_io_uring_register(int fd, unsigned opcode, void* arg,
                          unsigned nr_args) noexcept {
  return static_cast<int>(
      ::syscall(__NR_io_uring_register, fd, opcode, arg, nr_args));
}

// atomic view over a kernel-shared u32 ring counter (barriers per liburing).
inline std::atomic<unsigned>* as_atomic(void* p) noexcept {
  return reinterpret_cast<std::atomic<unsigned>*>(p);
}
// atomic view over the u16 buffer-ring tail (io-07b): the provided-buffer ring's
// tail is a __u16, published to the kernel with a release-store (liburing
// io_uring_buf_ring_advance uses io_uring_smp_store_release on this 16-bit field).
inline std::atomic<std::uint16_t>* as_atomic16(void* p) noexcept {
  return reinterpret_cast<std::atomic<std::uint16_t>*>(p);
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
  // Provided buffer ring (io-07b, kBuffer half). ONE ring supported for now. The
  // kernel picks a buffer from `buf_ring` for each multishot recv arrival and
  // reports its id in the CQE (IORING_CQE_F_BUFFER). We own the backing buffer
  // pool `buf_pool` (nentries * buf_size, one contiguous block) and re-hand a
  // consumed buffer to the ring via buf_ring_return (add + release-store tail).
  // Kernel-allocated ring (IOU_PBUF_RING_MMAP) mmap'd at IORING_OFF_PBUF_RING.
  io_uring_buf_ring* buf_ring = nullptr;
  std::size_t buf_ring_sz = 0;   // mmap length (nentries * sizeof(io_uring_buf))
  void* buf_pool = nullptr;      // backing storage for all buffers (own mmap)
  std::size_t buf_pool_sz = 0;
  unsigned buf_ring_entries = 0; // power-of-two count (== ring mask + 1)
  unsigned buf_ring_mask = 0;    // entries - 1
  unsigned buf_size = 0;         // bytes per buffer
  int buf_bgid = -1;             // buffer group id (-1 = no ring registered)
  // Submit batch (io-04 opt): SQEs filled by submit_* but not yet flushed to the
  // kernel. The worker loop now reaps only when its run-queue drains, so many
  // fibers submit between two reaps -- these accumulate here and the next reap
  // folds them into ITS enter (to_submit = sq_pending). Owner-thread-only (the
  // per-core worker); post() no longer touches the SQ, so no atomic needed.
  unsigned sq_pending = 0;
  // Count of io ops submitted-but-not-yet-completed (in-flight). Lets the worker
  // decide whether an idle reap should BLOCK (harvest via a GETEVENTS enter) or
  // return immediately: if ops are in flight, the worker MUST do a blocking
  // (GETEVENTS) reap to harvest them -- a non-blocking reap(0) never enters with
  // GETEVENTS, so completions requiring a kernel harvest never surface and the
  // waiting fibers hang (the >ring-depth deadlock, root-caused 2026-07-03).
  // Owner-thread-only. Bumped on each submit_*, decremented per real completion.
  // The persistent wake-read (see wake_efd) is NOT counted here: it is permanent
  // infrastructure, not a user op, so it must not make in_flight() report true for
  // a worker that has no real io (else that worker would never park on the futex).
  std::uint64_t inflight = 0;
  // Cross-thread wake channel (io-04 park merge). wake() writes 8 bytes to this
  // eventfd from ANY thread (write() is atomic -- no single-producer SQ race); a
  // persistent IORING_OP_READ armed on it (user_data = kWakeToken) completes on
  // that write, breaking the owner worker's blocking GETEVENTS reap. reap detects
  // the wake completion, drops it (never a user completion), and re-arms the read.
  // -1 if eventfd/arming failed (park merge then falls back to a bounded reap).
  int wake_efd = -1;
  std::uint64_t wake_buf = 0;  // 8-byte sink the eventfd read writes into
  // Bench instrumentation (gated by HYLE_IO_BENCH_STATS env): count enters, the
  // SQEs submitted through them, and completions reaped -- to report the average
  // submit batch size (SQEs/enters, the direct evidence batching happens) and a
  // sqes-vs-reaped sanity diff (should net to ~0). Zero cost when the env is unset.
  std::uint64_t stat_enters = 0;
  std::uint64_t stat_sqes = 0;
  std::uint64_t stat_reaped = 0;

  // ---- epoll backend state (reactor -> completion emulation) ----
  // epoll only signals READINESS; the proactor must remember each armed op and
  // perform the IO itself when the fd becomes ready, then synthesize a
  // Completion. One pending op per fd (EPOLLONESHOT), which suffices for the
  // one-in-flight-op-per-fd model io-04 net uses. `opcode` reuses the IORING_OP_*
  // tags (READ/WRITE for byte IO, ACCEPT/CONNECT for the socket handshakes) so
  // reap() can branch on which syscall to run when the fd fires.
  int epfd = -1;
  struct PendingOp {
    std::uint8_t opcode;   // IORING_OP_READ/WRITE/ACCEPT/CONNECT/RECVMSG/SENDMSG
    void* buf;
    std::uint32_t len;
    std::uint64_t user_data;
    // Datagram (RECVMSG/SENDMSG) only: peer address buffer. recvfrom fills it (up
    // to addr_cap) and reports the length back via the sockaddr family; sendto
    // reads it as the destination. Null/0 for stream ops.
    void* addr = nullptr;
    std::uint32_t addr_cap = 0;
    // Multishot accept emulation (io-07): if set, reap re-arms this ACCEPT watch
    // after each delivery and stamps Completion::kMore, mirroring io_uring's
    // native multishot upward. Only meaningful for IORING_OP_ACCEPT.
    bool multishot = false;
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
    if (impl_->wake_efd >= 0) ::close(impl_->wake_efd);
    if (impl_->epfd >= 0) ::close(impl_->epfd);
  } else {
    // Bench stats (gated): report the average submit batch size = SQEs / enters.
    // >1 means batching happened; ~1 means every op still paid its own enter.
    if (impl_->stat_enters > 0 && ::getenv("HYLE_IO_BENCH_STATS") != nullptr) {
      std::fprintf(stderr,
                   "[io] enters=%llu sqes=%llu avg_batch=%.2f reaped=%llu "
                   "diff(sqes-reaped)=%lld\n",
                   static_cast<unsigned long long>(impl_->stat_enters),
                   static_cast<unsigned long long>(impl_->stat_sqes),
                   static_cast<double>(impl_->stat_sqes) /
                       static_cast<double>(impl_->stat_enters),
                   static_cast<unsigned long long>(impl_->stat_reaped),
                   static_cast<long long>(impl_->stat_sqes) -
                       static_cast<long long>(impl_->stat_reaped));
    }
    if (impl_->wake_efd >= 0) ::close(impl_->wake_efd);
    // Provided buffer ring (io-07b): unregister (kernel drops its ref) then unmap
    // our ring view + the backing pool. Order: unregister before unmap so the
    // kernel isn't left pointing at a freed mapping.
    if (impl_->buf_bgid >= 0) {
      io_uring_buf_reg reg;
      std::memset(&reg, 0, sizeof(reg));
      reg.bgid = static_cast<std::uint16_t>(impl_->buf_bgid);
      sys_io_uring_register(impl_->fd, IORING_UNREGISTER_PBUF_RING, &reg, 1);
      if (impl_->buf_ring != nullptr)
        ::munmap(impl_->buf_ring, impl_->buf_ring_sz);
      if (impl_->buf_pool != nullptr) ::munmap(impl_->buf_pool, impl_->buf_pool_sz);
    }
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

  // Cross-thread wake channel (io-04 park merge): an eventfd + a persistent read
  // armed on it. wake() writes the eventfd from any thread (no SQ race); the read
  // completes with kWakeToken, breaking a blocking reap so the worker can park
  // truly-asleep (no 1ms re-poll) yet still respond to a foreign spawn/steal.
  // Non-fatal if it fails -- the proactor still serves io; the scheduler then
  // falls back to a bounded (kIoPollNs) reap instead of an infinite one.
  impl->wake_efd = ::eventfd(0, EFD_CLOEXEC);
  if (impl->wake_efd >= 0) {
    // Arm the wake-read directly (not submit_one): it must NOT bump inflight --
    // it's permanent infra, not a user op that in_flight() should count.
    unsigned tail = *impl->sq_tail;
    unsigned index = tail & *impl->sq_ring_mask;
    io_uring_sqe& s = impl->sqes[index];
    std::memset(&s, 0, sizeof(s));
    s.opcode = IORING_OP_READ;
    s.fd = impl->wake_efd;
    s.addr = reinterpret_cast<std::uint64_t>(&impl->wake_buf);
    s.len = sizeof(impl->wake_buf);
    s.user_data = Proactor::kWakeToken;
    impl->sq_array[index] = index;
    as_atomic(impl->sq_tail)->store(tail + 1, std::memory_order_release);
    sys_io_uring_enter(fd, 1, 0, 0, nullptr, 0);  // submit the wake-read now
  }
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
  // Cross-thread wake channel (park merge), epoll flavour: a persistent (level-
  // triggered, NO EPOLLONESHOT) eventfd in the watch set. wake() writes it; the
  // blocking epoll_wait returns; reap drains it (re-reads to reset the counter,
  // then keeps it armed) and never emits it as a user completion. `data.u64 ==
  // wake_efd` distinguishes it from armed io fds. Non-fatal on failure.
  impl->wake_efd = ::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  if (impl->wake_efd >= 0) {
    epoll_event ev{};
    ev.events = EPOLLIN;  // level-triggered + persistent: stays armed across waits
    ev.data.u64 = static_cast<std::uint64_t>(impl->wake_efd);
    if (::epoll_ctl(ep, EPOLL_CTL_ADD, impl->wake_efd, &ev) < 0) {
      ::close(impl->wake_efd);
      impl->wake_efd = -1;
    }
  }
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
      if (fd == impl_->wake_efd) {
        // Cross-thread wake (park merge): drain the counter so the level-triggered
        // watch stops firing, and emit NO completion -- it only breaks the block.
        std::uint64_t sink;
        while (::read(impl_->wake_efd, &sink, sizeof(sink)) == sizeof(sink)) {}
        continue;
      }
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
      } else if (op.opcode == IORING_OP_RECVMSG) {
        // Datagram readable -> recvfrom fills the peer addr the submit stashed.
        socklen_t al = static_cast<socklen_t>(op.addr_cap);
        ssize_t r = ::recvfrom(fd, op.buf, op.len, 0,
                               static_cast<sockaddr*>(op.addr),
                               op.addr ? &al : nullptr);
        res = (r < 0) ? -errno : static_cast<std::int64_t>(r);
      } else if (op.opcode == IORING_OP_SENDMSG) {
        // Datagram writable -> sendto the destination the submit stashed.
        ssize_t w = ::sendto(fd, op.buf, op.len, 0,
                             static_cast<const sockaddr*>(op.addr),
                             static_cast<socklen_t>(op.addr_cap));
        res = (w < 0) ? -errno : static_cast<std::int64_t>(w);
      } else {  // READ
        ssize_t r = ::read(fd, op.buf, op.len);
        res = (r < 0) ? -errno : static_cast<std::int64_t>(r);
      }
      out[n].user_data = op.user_data;
      out[n].result = res;
      out[n].flags = 0;
      out[n].buffer_id = 0;
      // Multishot accept emulation: on a successful accept, re-arm the listener
      // watch and stamp kMore (more connections coming, same user_data) -- mirrors
      // io_uring native multishot. On failure, leave kMore clear = terminal (the
      // multishot ended), matching io_uring's behaviour on a fatal accept error.
      if (op.multishot && res >= 0) {
        out[n].flags |= Completion::kMore;
        epoll_event rev{};
        rev.events = EPOLLIN | EPOLLONESHOT;
        rev.data.u64 = static_cast<std::uint64_t>(fd);
        if (::epoll_ctl(impl_->epfd, EPOLL_CTL_ADD, fd, &rev) == 0) {
          impl_->pending[fd] = op;  // keep multishot=true for the next delivery
        }
      }
      ++n;
    }
    return n;
  }

  // io-05: drain any synchronously-synthesized completions first (statx runs its
  // fstat at submit time -- it's a pure metadata query, not a disk-io path, so it
  // needn't burn an SQE). These carry their result directly, like epoll's posted
  // queue. Deliver them before touching the ring so they aren't starved.
  {
    std::size_t np = 0;
    while (np < max && !impl_->posted.empty()) {
      Impl::Posted pd = impl_->posted.back();
      impl_->posted.pop_back();
      out[np].user_data = pd.user_data;
      out[np].result = pd.result;
      out[np].flags = 0;
      out[np].buffer_id = 0;
      ++np;
    }
    if (np > 0) return np;  // deliver promptly; caller loops for ring completions
  }

  // Flush the batched SQEs (io-04 submit batch): the worker queued many via
  // submit_* since its last drain; fold to_submit = sq_pending into this reap's
  // enter so N submits + this reap cost ONE syscall. Reset the count up front.
  unsigned submit = impl_->sq_pending;
  impl_->sq_pending = 0;

  // If the CQ is already non-empty we normally skip the syscall (pure userspace
  // reap). But pending SQEs must still be handed to the kernel -- a submit-only
  // enter (no wait) does that, then we reap the ready completions in userspace.
  unsigned tail = as_atomic(impl_->cq_tail)->load(std::memory_order_acquire);
  unsigned head = *impl_->cq_head;
  if (head == tail) {
    if (timeout_ns == 0) {
      // Non-blocking poll, CQ empty: flush the batch (no wait) so it reaches the
      // kernel, then return -- the completions will show up on a later reap.
      if (submit != 0) {
        sys_io_uring_enter(impl_->fd, submit, 0, 0, nullptr, 0);
        impl_->stat_enters += 1;
        impl_->stat_sqes += submit;
      }
      return 0;
    }
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
    // One enter both submits the batched SQEs AND waits for >=1 completion.
    int r = sys_io_uring_enter(impl_->fd, submit, 1, flags,
                               (flags & IORING_ENTER_EXT_ARG) ? &arg : nullptr,
                               (flags & IORING_ENTER_EXT_ARG) ? sizeof(arg) : 0);
    impl_->stat_enters += 1;
    impl_->stat_sqes += submit;
    if (r < 0 && errno != ETIME && errno != EINTR) return 0;
    tail = as_atomic(impl_->cq_tail)->load(std::memory_order_acquire);
    head = *impl_->cq_head;
  } else if (submit != 0) {
    // CQ non-empty but SQEs pending: submit-only enter (no wait) to flush them,
    // then reap the ready completions below.
    sys_io_uring_enter(impl_->fd, submit, 0, 0, nullptr, 0);
    impl_->stat_enters += 1;
    impl_->stat_sqes += submit;
  }

  std::size_t n = 0;
  bool wake_fired = false;
  const unsigned mask = *impl_->cq_ring_mask;
  while (head != tail && n < max) {
    const io_uring_cqe& c = impl_->cqes[head & mask];
    if (c.user_data == Proactor::kWakeToken) {
      // The cross-thread wake-read fired (park merge): consume it here -- it is
      // NOT a user completion (no out[] slot, no inflight decrement -- it was
      // never counted). Mark for re-arm after the loop so the next wake works.
      wake_fired = true;
      ++head;
      continue;
    }
    out[n].user_data = c.user_data;
    out[n].result = static_cast<std::int64_t>(c.res);  // <0 = -errno (io-02 ops)
    // Map io_uring CQE flags onto our neutral flags.
    out[n].flags = 0;
    if (c.flags & IORING_CQE_F_MORE) {
      out[n].flags |= Completion::kMore;
    } else if (impl_->inflight != 0) {
      // Final completion for this op (multishot's last, or an ordinary 1:1 op) --
      // the kernel no longer owes us a CQE for it. kMore-set completions leave the
      // count alone (more coming for the same submit). Guarded != 0 so a synthetic
      // reap (all post() wakes, which don't submit real ops) can't underflow.
      --impl_->inflight;
    }
    if (c.flags & IORING_CQE_F_BUFFER) {
      out[n].flags |= Completion::kBuffer;
      out[n].buffer_id = c.flags >> IORING_CQE_BUFFER_SHIFT;
    } else {
      out[n].buffer_id = 0;
    }
    ++head;
    ++n;
  }
  // (stat_reaped is tallied once at function end = total n.)
  // Publish the consumed head once for the whole batch (store_release pairs with
  // the kernel; one release-store amortized over n completions).
  as_atomic(impl_->cq_head)->store(head, std::memory_order_release);

  // Re-arm the one-shot wake-read if it fired (the read cleared the eventfd, so a
  // fresh read is needed for the next wake). Owner thread here (reap runs on the
  // worker), so touching the SQ is safe; batch it (flushed by the next enter).
  // NOT counted in inflight -- it is permanent infra, not a user op.
  if (wake_fired && impl_->wake_efd >= 0) {
    unsigned t = *impl_->sq_tail;
    unsigned ix = t & *impl_->sq_ring_mask;
    io_uring_sqe& s = impl_->sqes[ix];
    std::memset(&s, 0, sizeof(s));
    s.opcode = IORING_OP_READ;
    s.fd = impl_->wake_efd;
    s.addr = reinterpret_cast<std::uint64_t>(&impl_->wake_buf);
    s.len = sizeof(impl_->wake_buf);
    s.user_data = Proactor::kWakeToken;
    impl_->sq_array[ix] = ix;
    as_atomic(impl_->sq_tail)->store(t + 1, std::memory_order_release);
    ++impl_->sq_pending;  // flushed by the next reap's enter
  }

  // A full CQ overflows into the kernel's NODROP backlog (IORING_FEAT_NODROP sets
  // IORING_SQ_CQ_OVERFLOW rather than dropping). No explicit drain loop is needed
  // here: the scheduler keeps this worker doing bounded GETEVENTS reaps while
  // in_flight() != 0, and EVERY GETEVENTS io_uring_enter flushes that backlog into
  // the (now head-advanced, so free) CQ -- so the next reap harvests it. Draining
  // in ≤`max` chunks across successive reaps converges; a single-call recovery
  // loop would only save loop iterations, at the cost of double-counting the
  // in-flight tally. (The >ring-depth deadlock was NEVER overflow -- overflow=0
  // measured; it was the worker never doing a GETEVENTS enter at all.)
  impl_->stat_reaped += n;  // total real+wake completions handed back this reap
  return n;
}

bool Proactor::in_flight() const noexcept {
  if (impl_ == nullptr) return false;
  if (impl_->backend == Impl::kEpoll) {
    // Reactor: an op is "in flight" while its fd is armed (awaiting readiness) or
    // a synthetic completion is queued. Either means a reap could still yield one.
    return !impl_->pending.empty() || !impl_->posted.empty();
  }
  // io_uring: the kernel owes `inflight` completions, PLUS any io-05 synchronously
  // synthesized completions still queued (statx) -- both must keep the worker
  // reaping (a queued posted completion a parked worker never drains = a hang).
  return impl_->inflight != 0 || !impl_->posted.empty();
}

namespace {
// Fill one SQE + advance the SQ tail + count it pending. Normally does NOT enter
// -- the worker's drain-time reap flushes the whole batch in one enter (io-04
// submit batch). EXCEPTION: if the batch would overflow the SQ ring (pending
// reaches sq_entries), flush it now with a submit-only enter (this is tokio-
// uring's "SQ full -> submit" trigger). Takes the mapped fields + &sq_pending +
// ring_fd/sq_entries so it needn't name the private Impl type. Shared by
// post/read/write/recv/send/accept/connect. `op_flags` sets the SQE flags union
// (rw_flags/accept_flags/msg_flags). `ioprio` sets the SQE ioprio field, which
// carries the multishot bits (IORING_ACCEPT_MULTISHOT / IORING_RECV_MULTISHOT) --
// 0 for a normal one-shot op.
bool submit_one(int ring_fd, unsigned sq_entries, unsigned* sq_pending,
                std::uint64_t* inflight, unsigned* sq_tail, unsigned sq_mask,
                unsigned* sq_array, io_uring_sqe* sqes, std::uint8_t opcode,
                std::int32_t fd, std::uint64_t addr, std::uint32_t len,
                std::uint64_t off, std::uint64_t user_data,
                std::uint32_t op_flags = 0, std::uint16_t ioprio = 0) noexcept {
  if (*sq_pending >= sq_entries) {  // ring full of un-flushed SQEs -> flush now
    sys_io_uring_enter(ring_fd, *sq_pending, 0, 0, nullptr, 0);
    *sq_pending = 0;
  }
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
  s.ioprio = ioprio;                        // multishot bits (accept/recv), else 0
  s.user_data = user_data;
  sq_array[index] = index;
  // publish SQ tail (store_release pairs with kernel load_acquire); the enter
  // that tells the kernel to consume these happens in reap() (batched).
  as_atomic(sq_tail)->store(tail + 1, std::memory_order_release);
  ++*sq_pending;
  // One more op the kernel owes us a completion for. reap() decrements per CQE.
  // The worker reads this (Proactor::in_flight) to decide it must BLOCK in a
  // GETEVENTS reap rather than the futex parker, or those completions never
  // surface (the >ring-depth hang). Owner-thread-only, no atomic needed.
  ++*inflight;
  return true;
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
  // NOP SQE carrying user_data: the cross-thread wake / inject hook. Unlike the
  // batched submit_* ops (flushed by the worker's drain reap), post MUST enter
  // NOW: it wakes a reap() blocked on ANOTHER thread, which cannot flush this
  // SQE for us. Enter submits all pending SQEs (this NOP + any batched).
  (void)result;
  submit_one(impl_->fd, impl_->sq_entries, &impl_->sq_pending, &impl_->inflight, impl_->sq_tail, *impl_->sq_ring_mask,
             impl_->sq_array, impl_->sqes, IORING_OP_NOP, -1, 0, 0, 0, user_data);
  int r = sys_io_uring_enter(impl_->fd, impl_->sq_pending, 0, 0, nullptr, 0);
  impl_->sq_pending = 0;
  return r >= 0;
}

bool Proactor::wake() noexcept {
  if (impl_ == nullptr || impl_->wake_efd < 0) return false;
  // Write 8 bytes to the wake eventfd from ANY thread. eventfd write is atomic
  // and NOT the io_uring SQ, so this never races the single-producer submission
  // queue (that was the TSan bug that killed the earlier NOP-SQE wake). The
  // armed IORING_OP_READ (io_uring) or the level-triggered EPOLLIN watch (epoll)
  // then completes / fires, breaking the owner worker's blocking reap. reap drops
  // the wake token and (io_uring) re-arms the read.
  std::uint64_t one = 1;
  ssize_t w = ::write(impl_->wake_efd, &one, sizeof(one));
  return w == static_cast<ssize_t>(sizeof(one));
}

// epoll fallback: arm a one-shot readiness watch for `fd` and remember the op;
// reap() does the actual read/write when the fd fires. `events` is EPOLLIN for
// read, EPOLLOUT for write. Stores fd in data.u64 so reap recovers it. The
// datagram variant additionally stashes the peer-address buffer (recvfrom fills
// it / sendto reads it); the stream variant passes addr=null.
bool Proactor::epoll_arm_dgram(Impl* impl, int fd, unsigned events,
                               unsigned char opcode, void* buf, unsigned len,
                               void* addr, unsigned addr_cap,
                               std::uint64_t user_data,
                               std::uint64_t offset) noexcept {
  impl->pending[fd] =
      Impl::PendingOp{opcode, buf, len, user_data, addr, addr_cap};
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
    } else if (errno == EPERM) {
      // EPERM = fd is a REGULAR FILE (or otherwise not epoll-able). epoll cannot
      // watch regular files -- they are always "ready" but read()/write() may block
      // on disk (io-05: only io_uring/IOCP async a regular file; tokio offloads to
      // a thread pool for exactly this). So do the byte IO SYNCHRONOUSLY now, AT the
      // file offset (pread/pwrite -- the proactor read/write contract is offset-
      // based, unlike a stream), and synthesize a completion via the posted queue
      // (the same synthesis the metadata file ops use). Stream/datagram sockets are
      // never regular files, so this branch is file-only.
      impl->pending.erase(fd);
      std::int64_t res;
      if (opcode == IORING_OP_WRITE) {
        ssize_t w = ::pwrite(fd, buf, len, static_cast<off_t>(offset));
        res = (w < 0) ? -errno : static_cast<std::int64_t>(w);
      } else {  // IORING_OP_READ
        ssize_t r = ::pread(fd, buf, len, static_cast<off_t>(offset));
        res = (r < 0) ? -errno : static_cast<std::int64_t>(r);
      }
      impl->posted.push_back({user_data, res});
      return true;
    } else {
      impl->pending.erase(fd);
      return false;
    }
  }
  return true;
}

bool Proactor::epoll_arm(Impl* impl, int fd, unsigned events,
                         unsigned char opcode, void* buf, unsigned len,
                         std::uint64_t user_data, std::uint64_t offset) noexcept {
  return epoll_arm_dgram(impl, fd, events, opcode, buf, len, nullptr, 0,
                         user_data, offset);
}

bool Proactor::submit_read(std::intptr_t fd, void* buf, std::uint32_t len,
                           std::uint64_t offset,
                           std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  if (impl_->backend == Impl::kEpoll) {
    return epoll_arm(impl_, static_cast<int>(fd), EPOLLIN, IORING_OP_READ, buf,
                     len, user_data, offset);
  }
  return submit_one(impl_->fd, impl_->sq_entries, &impl_->sq_pending, &impl_->inflight, impl_->sq_tail, *impl_->sq_ring_mask,
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
                     const_cast<void*>(buf), len, user_data, offset);
  }
  return submit_one(impl_->fd, impl_->sq_entries, &impl_->sq_pending, &impl_->inflight, impl_->sq_tail, *impl_->sq_ring_mask,
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
  return submit_one(impl_->fd, impl_->sq_entries, &impl_->sq_pending, &impl_->inflight, impl_->sq_tail, *impl_->sq_ring_mask,
                    impl_->sq_array, impl_->sqes, IORING_OP_RECV,
                    static_cast<std::int32_t>(fd),
                    reinterpret_cast<std::uint64_t>(buf), len, 0, user_data,
                    flags);
}

// ---- provided buffer ring + multishot recv (io-07b) -----------------------

bool Proactor::register_buf_ring(int bgid, unsigned nentries,
                                 unsigned buf_size) noexcept {
  if (impl_ == nullptr || impl_->backend != Impl::kUring) return false;
  if (impl_->buf_bgid >= 0) return false;  // one ring supported for now
  if (nentries == 0 || buf_size == 0) return false;
  // Round entries up to a power of two (the ring mask requires it).
  unsigned n = 1;
  while (n < nentries) n <<= 1;

  // Register a KERNEL-allocated buffer ring (IOU_PBUF_RING_MMAP): the kernel owns
  // the ring memory; we mmap it at IORING_OFF_PBUF_RING | (bgid << shift).
  io_uring_buf_reg reg;
  std::memset(&reg, 0, sizeof(reg));
  reg.ring_entries = n;
  reg.bgid = static_cast<std::uint16_t>(bgid);
  reg.flags = IOU_PBUF_RING_MMAP;
  if (sys_io_uring_register(impl_->fd, IORING_REGISTER_PBUF_RING, &reg, 1) < 0)
    return false;

  std::size_t ring_sz = static_cast<std::size_t>(n) * sizeof(io_uring_buf);
  off_t off = IORING_OFF_PBUF_RING |
              (static_cast<off_t>(bgid) << IORING_OFF_PBUF_SHIFT);
  void* br = ::mmap(nullptr, ring_sz, PROT_READ | PROT_WRITE,
                    MAP_SHARED | MAP_POPULATE, impl_->fd, off);
  if (br == MAP_FAILED) {
    reg.flags = 0;
    sys_io_uring_register(impl_->fd, IORING_UNREGISTER_PBUF_RING, &reg, 1);
    return false;
  }

  // Own the backing buffer pool (n * buf_size, one block).
  std::size_t pool_sz = static_cast<std::size_t>(n) * buf_size;
  void* pool = ::mmap(nullptr, pool_sz, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (pool == MAP_FAILED) {
    ::munmap(br, ring_sz);
    reg.flags = 0;
    sys_io_uring_register(impl_->fd, IORING_UNREGISTER_PBUF_RING, &reg, 1);
    return false;
  }

  impl_->buf_ring = static_cast<io_uring_buf_ring*>(br);
  impl_->buf_ring_sz = ring_sz;
  impl_->buf_pool = pool;
  impl_->buf_pool_sz = pool_sz;
  impl_->buf_ring_entries = n;
  impl_->buf_ring_mask = n - 1;
  impl_->buf_size = buf_size;
  impl_->buf_bgid = bgid;

  // Init the ring (tail = 0) then publish ALL n buffers to the kernel: fill each
  // slot with (addr, len, bid) at offset i, then one release-store of tail = n.
  impl_->buf_ring->tail = 0;
  for (unsigned i = 0; i < n; ++i) {
    io_uring_buf& b = impl_->buf_ring->bufs[i & impl_->buf_ring_mask];
    b.addr = reinterpret_cast<std::uint64_t>(
        static_cast<char*>(pool) + static_cast<std::size_t>(i) * buf_size);
    b.len = buf_size;
    b.bid = static_cast<std::uint16_t>(i);
  }
  as_atomic16(&impl_->buf_ring->tail)->store(static_cast<std::uint16_t>(n),
                                             std::memory_order_release);
  return true;
}

void* Proactor::buf_ring_addr(int bgid, std::uint32_t buffer_id) noexcept {
  if (impl_ == nullptr || impl_->buf_bgid != bgid) return nullptr;
  if (buffer_id >= impl_->buf_ring_entries) return nullptr;
  return static_cast<char*>(impl_->buf_pool) +
         static_cast<std::size_t>(buffer_id) * impl_->buf_size;
}

bool Proactor::buf_ring_return(int bgid, std::uint32_t buffer_id) noexcept {
  if (impl_ == nullptr || impl_->buf_bgid != bgid) return false;
  if (buffer_id >= impl_->buf_ring_entries) return false;
  // Re-add the buffer at the current tail slot, then release-store tail+1 so the
  // kernel sees it available again (liburing buf_ring_add + buf_ring_advance(1)).
  std::uint16_t tail =
      as_atomic16(&impl_->buf_ring->tail)->load(std::memory_order_relaxed);
  io_uring_buf& b = impl_->buf_ring->bufs[tail & impl_->buf_ring_mask];
  b.addr = reinterpret_cast<std::uint64_t>(
      static_cast<char*>(impl_->buf_pool) +
      static_cast<std::size_t>(buffer_id) * impl_->buf_size);
  b.len = impl_->buf_size;
  b.bid = static_cast<std::uint16_t>(buffer_id);
  as_atomic16(&impl_->buf_ring->tail)
      ->store(static_cast<std::uint16_t>(tail + 1), std::memory_order_release);
  return true;
}

bool Proactor::submit_recv_multishot(std::intptr_t fd, int bgid,
                                     std::uint64_t user_data) noexcept {
  if (impl_ == nullptr || impl_->backend != Impl::kUring) return false;
  if (impl_->buf_bgid != bgid) return false;  // ring must be registered first
  // Multishot recv drawing from the provided buffer ring: IORING_RECV_MULTISHOT
  // in ioprio + IOSQE_BUFFER_SELECT in the SQE flags + buf_group = bgid. No caller
  // buffer (addr/len 0): the kernel picks one per arrival and reports its id in
  // IORING_CQE_F_BUFFER. submit_one doesn't set sqe.flags/buf_group, so fill the
  // SQE directly here.
  unsigned tail = *impl_->sq_tail;
  unsigned index = tail & *impl_->sq_ring_mask;
  io_uring_sqe& s = impl_->sqes[index];
  std::memset(&s, 0, sizeof(s));
  s.opcode = IORING_OP_RECV;
  s.fd = static_cast<std::int32_t>(fd);
  s.ioprio = IORING_RECV_MULTISHOT;
  s.flags = IOSQE_BUFFER_SELECT;
  s.buf_group = static_cast<std::uint16_t>(bgid);
  s.user_data = user_data;
  impl_->sq_array[index] = index;
  as_atomic(impl_->sq_tail)->store(tail + 1, std::memory_order_release);
  ++impl_->sq_pending;
  ++impl_->inflight;
  return true;
}

bool Proactor::submit_send(std::intptr_t fd, const void* buf, std::uint32_t len,
                           std::uint32_t flags,
                           std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  if (impl_->backend == Impl::kEpoll) {
    return epoll_arm(impl_, static_cast<int>(fd), EPOLLOUT, IORING_OP_WRITE,
                     const_cast<void*>(buf), len, user_data);
  }
  return submit_one(impl_->fd, impl_->sq_entries, &impl_->sq_pending, &impl_->inflight, impl_->sq_tail, *impl_->sq_ring_mask,
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
  return submit_one(impl_->fd, impl_->sq_entries, &impl_->sq_pending, &impl_->inflight, impl_->sq_tail, *impl_->sq_ring_mask,
                    impl_->sq_array, impl_->sqes, IORING_OP_ACCEPT,
                    static_cast<std::int32_t>(fd),
                    reinterpret_cast<std::uint64_t>(addr), 0,
                    reinterpret_cast<std::uint64_t>(addrlen), user_data);
}

bool Proactor::submit_accept_multishot(std::intptr_t fd,
                                       std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  if (impl_->backend == Impl::kEpoll) {
    // epoll has no native multishot. Arm a one-shot accept watch, then flag it
    // multishot so reap() re-arms after each delivery and stamps kMore -- the
    // caller sees the same continuous kMore-set stream as io_uring. Peer addr is
    // read by net/ via getpeername on the accepted fd (buf/len stay 0).
    if (!epoll_arm(impl_, static_cast<int>(fd), EPOLLIN, IORING_OP_ACCEPT,
                   nullptr, 0, user_data))
      return false;
    impl_->pending[static_cast<int>(fd)].multishot = true;
    return true;
  }
  // io_uring native: IORING_OP_ACCEPT with IORING_ACCEPT_MULTISHOT in ioprio.
  // addr/addrlen null -- multishot cannot fill a per-connection addr into one
  // fixed buffer (net/ uses getpeername on the accepted fd). One submit yields a
  // CQE per connection with IORING_CQE_F_MORE set until cancelled/errored.
  return submit_one(impl_->fd, impl_->sq_entries, &impl_->sq_pending, &impl_->inflight, impl_->sq_tail, *impl_->sq_ring_mask,
                    impl_->sq_array, impl_->sqes, IORING_OP_ACCEPT,
                    static_cast<std::int32_t>(fd), 0, 0, 0, user_data,
                    /*op_flags=*/0, /*ioprio=*/IORING_ACCEPT_MULTISHOT);
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
  return submit_one(impl_->fd, impl_->sq_entries, &impl_->sq_pending, &impl_->inflight, impl_->sq_tail, *impl_->sq_ring_mask,
                    impl_->sq_array, impl_->sqes, IORING_OP_CONNECT,
                    static_cast<std::int32_t>(fd),
                    reinterpret_cast<std::uint64_t>(addr), 0,
                    static_cast<std::uint64_t>(addrlen), user_data);
}

namespace {
// Lay a msghdr + a single iovec into the caller's `scratch` (>= kDatagramScratch),
// pointing the iovec at (buf,len) and msg_name at (addr,addr_len). io_uring
// RECVMSG/SENDMSG read this msghdr WHEN THE OP COMPLETES, so it must outlive the
// submit -- scratch lives on the awaiting fiber's frozen stack. Returns the msghdr
// address (goes into the SQE's addr field). One iovec suffices for datagrams.
msghdr* build_msghdr(void* scratch, void* buf, std::uint32_t len, void* addr,
                     std::uint32_t addr_len) noexcept {
  static_assert(sizeof(msghdr) + sizeof(iovec) <= Proactor::kDatagramScratch,
                "kDatagramScratch too small for msghdr + iovec");
  auto* mh = static_cast<msghdr*>(scratch);
  auto* iov = reinterpret_cast<iovec*>(static_cast<char*>(scratch) + sizeof(msghdr));
  std::memset(mh, 0, sizeof(*mh));
  iov->iov_base = buf;
  iov->iov_len = len;
  mh->msg_iov = iov;
  mh->msg_iovlen = 1;
  mh->msg_name = addr;         // recv: kernel fills; send: kernel reads dest
  mh->msg_namelen = addr_len;  // recv: capacity; send: dest length
  return mh;
}
}  // namespace

bool Proactor::submit_recvfrom(std::intptr_t fd, void* buf, std::uint32_t len,
                               void* addr, std::uint32_t addr_cap, void* scratch,
                               std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  if (impl_->backend == Impl::kEpoll) {
    // Reactor: arm EPOLLIN; reap() runs ::recvfrom when readable, filling addr.
    return epoll_arm_dgram(impl_, static_cast<int>(fd), EPOLLIN,
                           IORING_OP_RECVMSG, buf, len, addr, addr_cap, user_data);
  }
  // RECVMSG: sqe.addr = &msghdr, sqe.len = 1 (liburing io_uring_prep_recvmsg).
  msghdr* mh = build_msghdr(scratch, buf, len, addr, addr_cap);
  return submit_one(impl_->fd, impl_->sq_entries, &impl_->sq_pending, &impl_->inflight, impl_->sq_tail, *impl_->sq_ring_mask,
                    impl_->sq_array, impl_->sqes, IORING_OP_RECVMSG,
                    static_cast<std::int32_t>(fd),
                    reinterpret_cast<std::uint64_t>(mh), 1, 0, user_data);
}

bool Proactor::submit_sendto(std::intptr_t fd, const void* buf, std::uint32_t len,
                             const void* addr, std::uint32_t addrlen,
                             void* scratch, std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  if (impl_->backend == Impl::kEpoll) {
    // Reactor: arm EPOLLOUT; reap() runs ::sendto to `addr` when writable.
    return epoll_arm_dgram(impl_, static_cast<int>(fd), EPOLLOUT,
                           IORING_OP_SENDMSG, const_cast<void*>(buf), len,
                           const_cast<void*>(addr), addrlen, user_data);
  }
  // SENDMSG: same shape as RECVMSG; msg_name is the destination (kernel reads it).
  msghdr* mh = build_msghdr(scratch, const_cast<void*>(buf), len,
                            const_cast<void*>(addr), addrlen);
  return submit_one(impl_->fd, impl_->sq_entries, &impl_->sq_pending, &impl_->inflight, impl_->sq_tail, *impl_->sq_ring_mask,
                    impl_->sq_array, impl_->sqes, IORING_OP_SENDMSG,
                    static_cast<std::int32_t>(fd),
                    reinterpret_cast<std::uint64_t>(mh), 1, 0, user_data);
}

bool Proactor::register_handle(std::intptr_t /*fd*/) noexcept {
  // Neither Linux backend needs a pre-registration step: io_uring puts the fd in
  // each SQE; epoll adds the fd at submit time (with EPOLLONESHOT).
  return impl_ != nullptr;
}

// ---- file ops (io-05) ------------------------------------------------------
// Map neutral OpenFlags onto POSIX O_* for openat.
namespace {
int open_flags_to_posix(std::uint32_t flags) noexcept {
  int f = 0;
  const bool rd = (flags & Proactor::kRead) != 0;
  const bool wr = (flags & Proactor::kWrite) != 0;
  if (rd && wr) f = O_RDWR;
  else if (wr) f = O_WRONLY;
  else f = O_RDONLY;
  if (flags & Proactor::kCreate) f |= O_CREAT;
  if (flags & Proactor::kTruncate) f |= O_TRUNC;
  if (flags & Proactor::kAppend) f |= O_APPEND;
  return f;
}
}  // namespace

bool Proactor::submit_open(const char* path, std::uint32_t flags,
                           std::uint32_t mode,
                           std::uint64_t user_data) noexcept {
  if (impl_ == nullptr || path == nullptr) return false;
  int oflags = open_flags_to_posix(flags);
  if (impl_->backend == Impl::kEpoll) {
    // No async openat on epoll -> open synchronously, synthesize a completion.
    int fd = ::open(path, oflags, static_cast<mode_t>(mode));
    impl_->posted.push_back(
        {user_data, fd < 0 ? -static_cast<std::int64_t>(errno)
                           : static_cast<std::int64_t>(fd)});
    return true;
  }
  // io_uring IORING_OP_OPENAT: fd = dirfd (AT_FDCWD), addr = path pointer, len =
  // mode, off = 0, open_flags in the rw_flags union. The caller guarantees `path`
  // outlives the completion (proactor.h contract). Result = opened fd or -errno.
  return submit_one(impl_->fd, impl_->sq_entries, &impl_->sq_pending,
                    &impl_->inflight, impl_->sq_tail, *impl_->sq_ring_mask,
                    impl_->sq_array, impl_->sqes, IORING_OP_OPENAT, AT_FDCWD,
                    reinterpret_cast<std::uint64_t>(path), mode, 0, user_data,
                    static_cast<std::uint32_t>(oflags));
}

bool Proactor::submit_close(std::intptr_t fd,
                            std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  if (impl_->backend == Impl::kEpoll) {
    int r = ::close(static_cast<int>(fd));
    impl_->posted.push_back(
        {user_data, r < 0 ? -static_cast<std::int64_t>(errno) : 0});
    return true;
  }
  // IORING_OP_CLOSE: only fd matters (addr/len/off = 0). Result 0 or -errno.
  return submit_one(impl_->fd, impl_->sq_entries, &impl_->sq_pending,
                    &impl_->inflight, impl_->sq_tail, *impl_->sq_ring_mask,
                    impl_->sq_array, impl_->sqes, IORING_OP_CLOSE,
                    static_cast<std::int32_t>(fd), 0, 0, 0, user_data);
}

bool Proactor::submit_statx(std::intptr_t fd, FileStat* out,
                            std::uint64_t user_data) noexcept {
  if (impl_ == nullptr || out == nullptr) return false;
  // statx is a pure metadata query (no disk-io wait), so run fstat synchronously
  // on BOTH backends and synthesize a completion -- it needn't burn an SQE, and
  // this fills `out` at submit time (no per-op statx buffer to carry to the CQE,
  // no struct-statx leak into the neutral header). The posted queue delivers it.
  struct stat st;
  int r = ::fstat(static_cast<int>(fd), &st);
  if (r == 0) {
    *out = FileStat{};
    out->size = static_cast<std::uint64_t>(st.st_size);
    out->mtime_ns = static_cast<std::uint64_t>(st.st_mtim.tv_sec) * 1'000'000'000ull +
                    static_cast<std::uint64_t>(st.st_mtim.tv_nsec);
    out->mode = static_cast<std::uint32_t>(st.st_mode);
    out->is_dir = S_ISDIR(st.st_mode);
  }
  impl_->posted.push_back(
      {user_data, r < 0 ? -static_cast<std::int64_t>(errno) : 0});
  return true;
}

bool Proactor::submit_fsync(std::intptr_t fd, bool datasync,
                            std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  if (impl_->backend == Impl::kEpoll) {
    int r = datasync ? ::fdatasync(static_cast<int>(fd))
                     : ::fsync(static_cast<int>(fd));
    impl_->posted.push_back(
        {user_data, r < 0 ? -static_cast<std::int64_t>(errno) : 0});
    return true;
  }
  // IORING_OP_FSYNC: fd + fsync_flags (IORING_FSYNC_DATASYNC bit for fdatasync).
  // fsync_flags shares the rw_flags union submit_one fills via op_flags.
  return submit_one(impl_->fd, impl_->sq_entries, &impl_->sq_pending,
                    &impl_->inflight, impl_->sq_tail, *impl_->sq_ring_mask,
                    impl_->sq_array, impl_->sqes, IORING_OP_FSYNC,
                    static_cast<std::int32_t>(fd), 0, 0, 0, user_data,
                    datasync ? IORING_FSYNC_DATASYNC : 0u);
}

bool Proactor::submit_fallocate(std::intptr_t fd, std::uint32_t mode,
                                std::uint64_t offset, std::uint64_t len,
                                std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  if (impl_->backend == Impl::kEpoll) {
    int r = ::posix_fallocate(static_cast<int>(fd),
                              static_cast<off_t>(offset),
                              static_cast<off_t>(len));
    // posix_fallocate returns the errno directly (not via errno).
    impl_->posted.push_back({user_data, r == 0 ? 0 : -static_cast<std::int64_t>(r)});
    return true;
  }
  // IORING_OP_FALLOCATE has a deliberately non-obvious field mapping (verified
  // against liburing io_uring_prep_fallocate): addr = LEN (byte count), len = MODE,
  // off = OFFSET. So pass through submit_one as addr=len, len=mode, off=offset.
  return submit_one(impl_->fd, impl_->sq_entries, &impl_->sq_pending,
                    &impl_->inflight, impl_->sq_tail, *impl_->sq_ring_mask,
                    impl_->sq_array, impl_->sqes, IORING_OP_FALLOCATE,
                    static_cast<std::int32_t>(fd), len, mode, offset, user_data);
}


}  // namespace hyle::io

#else
#  error "HYLE io/proactor: unsupported OS"
#endif
