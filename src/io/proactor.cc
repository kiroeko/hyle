#include "hyle/io/proactor.h"

#include <limits>
#include <new>
#include <vector>

#include "hyle/container/swiss_table.h"
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
    void* accept_addr = nullptr;           // one-shot caller's peer sockaddr
    void* accept_addrlen = nullptr;        // one-shot caller's int*
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
  IoOp* free_ops = nullptr;  // owner-thread cache: no per-I/O heap churn

  IoOp* acquire() noexcept {
    if (free_ops != nullptr) {
      IoOp* op = free_ops;
      free_ops = free_ops->next;
      *op = IoOp{};
      return op;
    }
    return new (std::nothrow) IoOp{};
  }
  void recycle(IoOp* op) noexcept {
    op->next = free_ops;
    free_ops = op;
  }
  void release_cache() noexcept {
    while (free_ops != nullptr) {
      IoOp* op = free_ops;
      free_ops = free_ops->next;
      delete op;
    }
  }

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
                                            INFINITE, FALSE);
    if (!ok) break;  // broken/closed port: preserve residual IoOps below
    for (ULONG i = 0; i < removed; ++i) {
      if (ents[i].lpOverlapped == nullptr) continue;  // a stray post(), ignore
      auto* op = reinterpret_cast<Impl::IoOp*>(ents[i].lpOverlapped);
      impl_->unlink(op);
      if (op->accept_sock != INVALID_SOCKET) ::closesocket(op->accept_sock);
      impl_->recycle(op);
    }
  }
  // A successfully-cancelled OVERLAPPED operation is guaranteed to enqueue its
  // final completion, so the normal path reaches an empty list. Closing an IOCP
  // does NOT make it safe to free an OVERLAPPED still owned by a device. If the
  // port itself failed, leak those impossible-to-prove-retired IoOps rather than
  // manufacture a teardown UAF.
  if (impl_->port != nullptr) ::CloseHandle(impl_->port);
  if (impl_->wsa_started) ::WSACleanup();
  impl_->release_cache();
  delete impl_;
  impl_ = nullptr;
}

bool Proactor::init(unsigned /*entries*/, bool /*defer_worker_thread*/) noexcept {
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

BackendKind Proactor::backend_kind() const noexcept {
  return impl_ != nullptr ? HYLE_IO_BACKEND_IOCP : HYLE_IO_BACKEND_NONE;
}
const char* Proactor::backend_name() const noexcept {
  return impl_ != nullptr ? "iocp" : "none";
}
ProactorCapabilities Proactor::capabilities() const noexcept {
  std::uint64_t bits = 0;
  if (impl_ != nullptr) {
    bits |= HYLE_IO_CAP_EMULATED_MULTISHOT_ACCEPT |
            HYLE_IO_CAP_ASYNC_FILE_DATA;
  }
#if defined(HYLE_IO_TLS)
  bits |= HYLE_IO_CAP_TLS;
#endif
#if defined(HYLE_IO_HTTP2)
  bits |= HYLE_IO_CAP_HTTP2;
#endif
  return {sizeof(ProactorCapabilities), backend_kind(), bits};
}
bool Proactor::has_async_file_data() const noexcept {
  return impl_ != nullptr;
}
bool Proactor::has_async_file_metadata() const noexcept {
  return false;
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
      impl_->recycle(op);
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
      // Finalize the accepted socket so it inherits the listener's properties, then
      // hand its fd back as the result (>= 0). Ownership passes to caller. CHECK the
      // SO_UPDATE_ACCEPT_CONTEXT result (red-team L6): without it succeeding, the
      // accepted socket is half-configured -- getpeername/getsockname/shutdown and
      // inherited options don't work (MSDN). Handing such a socket back is worse than
      // failing, so on error close it and report the failure instead of a bad fd.
      SOCKET ls = reinterpret_cast<SOCKET>(op->handle);
      if (::setsockopt(op->accept_sock, SOL_SOCKET, SO_UPDATE_ACCEPT_CONTEXT,
                       reinterpret_cast<char*>(&ls), sizeof(ls)) != 0) {
        out[i].result = neg_err(::WSAGetLastError());
        ::closesocket(op->accept_sock);
      } else if (op->accept_addr != nullptr) {
        int* addrlen = static_cast<int*>(op->accept_addrlen);
        if (::getpeername(op->accept_sock,
                          static_cast<sockaddr*>(op->accept_addr),
                          addrlen) != 0) {
          out[i].result = neg_err(::WSAGetLastError());
          ::closesocket(op->accept_sock);
        } else {
          out[i].result = static_cast<std::int64_t>(op->accept_sock);
        }
      } else {
        out[i].result = static_cast<std::int64_t>(op->accept_sock);
      }
    } else if (op->kind == Impl::kConnect) {
      // Same for connect: SO_UPDATE_CONNECT_CONTEXT must succeed or the connected
      // socket won't behave for getpeername/shutdown (red-team L6). On error report
      // the failure rather than a spurious "connected" (0).
      if (::setsockopt(reinterpret_cast<SOCKET>(op->handle), SOL_SOCKET,
                       SO_UPDATE_CONNECT_CONTEXT, nullptr, 0) != 0) {
        out[i].result = neg_err(::WSAGetLastError());
      } else {
        out[i].result = 0;  // connected
      }
    } else {
      out[i].result = static_cast<std::int64_t>(xfer);  // bytes transferred
    }
    // Multishot accept emulation: on a successful accept, re-arm AcceptEx on the
    // listener and stamp kMore (more connections coming, same user_data) --
    // mirrors io_uring native multishot upward. On failure leave kMore clear =
    // terminal (the multishot ended). Re-arm failure also ends it (kMore clear).
    if (op->multishot && op->kind == Impl::kAccept && out[i].result >= 0) {
      if (iocp_arm_accept(impl_, reinterpret_cast<std::intptr_t>(op->handle),
                          nullptr, nullptr, op->user_data,
                          /*multishot=*/true))
        out[i].flags |= Completion::kMore;
    }
    impl_->recycle(op);
  }
  return static_cast<std::size_t>(removed);
}

bool Proactor::post(std::uint64_t user_data, std::int64_t result) noexcept {
  if (impl_ == nullptr) return false;
  // Post a synthetic completion: key = user_data (the Waker token), bytes =
  // result. Wakes a blocked reap() from any thread AND drives tests without a
  // real fd. OVERLAPPED is nullptr -- reap() uses that to tell it from a real op.
  //
  // CONTRACT: `result` must be a NON-NEGATIVE, <=32-bit value on IOCP. It rides
  // dwNumberOfBytesTransferred (a 32-bit DWORD) and reap() zero-extends it back, so
  // a negative result (e.g. -errno) would be mangled into a large positive value
  // and a >2^32 result truncated. This is fine for post()'s uses (a wake token, a
  // byte count); anything needing a negative/64-bit outcome must NOT go through
  // post() -- a real op carries its int64 result via GetOverlappedResult/neg_err,
  // and the io-05 file path uses a 64-bit syn_result IoOp instead. Assert the
  // contract at runtime so an untrusted C caller cannot turn a range mistake
  // into process termination or silent truncation.
  if (result < 0 || result > 0xFFFFFFFFll) return false;
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
  auto* op = impl_->acquire();
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
  impl_->recycle(op);
  return false;
}

bool Proactor::submit_write(std::intptr_t fd, const void* buf,
                            std::uint32_t len, std::uint64_t offset,
                            std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  auto* op = impl_->acquire();
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
  impl_->recycle(op);
  return false;
}

bool Proactor::submit_recv(std::intptr_t fd, void* buf, std::uint32_t len,
                           std::uint32_t flags,
                           std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  auto* op = impl_->acquire();
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
  impl_->recycle(op);
  return false;
}

bool Proactor::submit_send(std::intptr_t fd, const void* buf, std::uint32_t len,
                           std::uint32_t flags,
                           std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  auto* op = impl_->acquire();
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
  impl_->recycle(op);
  return false;
}

// Arm one AcceptEx on listener `fd`. Shared by submit_accept (one-shot),
// submit_accept_multishot, and reap's multishot re-arm. `multishot` is stamped on
// the IoOp so reap knows to re-arm + set kMore after this one completes.
bool Proactor::iocp_arm_accept(Impl* impl, std::intptr_t fd,
                               void* addr, void* addrlen,
                               std::uint64_t user_data, bool multishot) noexcept {
  SOCKET ls = static_cast<SOCKET>(fd);
  if (!load_ext_fn(ls, WSAID_ACCEPTEX,
                   reinterpret_cast<void**>(&impl->acceptex)))
    return false;
  // AcceptEx needs a pre-created accept socket of the same family as the listener
  // (learn it via getsockname). AcceptEx writes both addresses into its private
  // buffer; after completion the one-shot path copies the peer address into the
  // caller's buffer with getpeername. Multishot has no per-result caller buffer.
  SOCKADDR_STORAGE ss;
  int slen = static_cast<int>(sizeof(ss));
  int fam = AF_INET;
  if (::getsockname(ls, reinterpret_cast<sockaddr*>(&ss), &slen) == 0)
    fam = ss.ss_family;
  SOCKET as = ::WSASocketW(fam, SOCK_STREAM, 0, nullptr, 0, WSA_FLAG_OVERLAPPED);
  if (as == INVALID_SOCKET) return false;
  auto* op = impl->acquire();
  if (op == nullptr) {
    ::closesocket(as);
    return false;
  }
  op->user_data = user_data;
  op->handle = reinterpret_cast<HANDLE>(fd);  // listener (for SO_UPDATE + result)
  op->kind = Proactor::Impl::kAccept;
  op->accept_sock = as;
  op->accept_addr = addr;
  op->accept_addrlen = addrlen;
  op->multishot = multishot;
  impl->link(op);
  constexpr DWORD kAddr = sizeof(SOCKADDR_STORAGE) + 16;
  DWORD got = 0;
  BOOL ok = impl->acceptex(ls, as, op->addr_buf, 0, kAddr, kAddr, &got, &op->ov);
  if (ok || ::WSAGetLastError() == WSA_IO_PENDING) return true;
  impl->unlink(op);
  ::closesocket(as);
  impl->recycle(op);
  return false;
}

bool Proactor::submit_accept(std::intptr_t fd, void* addr, void* addrlen,
                             std::uint64_t user_data) noexcept {
  if (impl_ == nullptr || ((addr == nullptr) != (addrlen == nullptr)))
    return false;
  return iocp_arm_accept(impl_, fd, addr, addrlen, user_data,
                         /*multishot=*/false);
}

bool Proactor::submit_accept_multishot(std::intptr_t fd,
                                       std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  // IOCP has no native multishot: arm one AcceptEx flagged multishot; reap
  // re-arms after each completion and stamps kMore (emulation, uniform contract).
  return iocp_arm_accept(impl_, fd, nullptr, nullptr, user_data,
                         /*multishot=*/true);
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
  auto* op = impl_->acquire();
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
  impl_->recycle(op);
  return false;
}

bool Proactor::submit_recvfrom(std::intptr_t fd, void* buf, std::uint32_t len,
                               void* addr, std::uint32_t addr_cap,
                               void* /*scratch*/,
                               std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  auto* op = impl_->acquire();
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
  impl_->recycle(op);
  return false;
}

bool Proactor::submit_sendto(std::intptr_t fd, const void* buf, std::uint32_t len,
                             const void* addr, std::uint32_t addrlen,
                             void* /*scratch*/,
                             std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  auto* op = impl_->acquire();
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
  impl_->recycle(op);
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
  auto* op = impl->acquire();
  if (op == nullptr) return false;
  op->user_data = user_data;
  op->kind = Impl::kFileMeta;
  op->syn_result = result;
  impl->link(op);
  // Post with the op's own OVERLAPPED (non-null) so reap treats it as a real op
  // (a null OVERLAPPED is a bare wake/post). Zero bytes / key -- reap uses kind.
  if (::PostQueuedCompletionStatus(impl->port, 0, 0, &op->ov) != 0) return true;
  impl->unlink(op);
  impl->recycle(op);
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

bool Proactor::submit_fallocate(std::intptr_t fd, std::uint32_t mode,
                                std::uint64_t offset, std::uint64_t len,
                                std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  if (mode != 0)
    return iocp_post_meta(
        impl_, -static_cast<std::int64_t>(ERROR_NOT_SUPPORTED), user_data);
  constexpr auto kMaxOffset =
      static_cast<std::uint64_t>((std::numeric_limits<LONGLONG>::max)());
  if (len > kMaxOffset || offset > kMaxOffset - len)
    return iocp_post_meta(
        impl_, -static_cast<std::int64_t>(ERROR_ARITHMETIC_OVERFLOW),
        user_data);
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

// io-06 cancel (IOCP): find the in-flight IoOp whose token matches, then CancelIoEx
// it. The op still completes with ERROR_OPERATION_ABORTED (posted to the port), so
// the caller's orphan table must keep the slot alive until that lands. cancel_user_
// data is unused on IOCP (no separate cancel completion -- CancelIoEx is synchronous
// in its request; only the cancelled op's own ABORTED completion arrives). Returns
// false if no matching in-flight op (already completed -> nothing to cancel).
bool Proactor::submit_cancel(std::uint64_t target_user_data,
                             std::uint64_t /*cancel_user_data*/) noexcept {
  if (impl_ == nullptr) return false;
  for (Impl::IoOp* op = impl_->inflight; op != nullptr; op = op->next) {
    if (op->user_data == target_user_data && op->handle != nullptr) {
      // CancelIoEx targets THIS specific overlapped op on its handle. Async: the
      // op reap()s later with GetLastError()==ERROR_OPERATION_ABORTED.
      return ::CancelIoEx(op->handle, &op->ov) != 0;
    }
  }
  return false;  // not found: already completed, nothing to cancel
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
// pure userspace. Setup flags (COOP_TASKRUN + SUBMIT_ALL) are enabled with a
// kernel-version fallback ladder in try_init_uring below -- they cut completion
// task_work overhead for the single-consumer-per-ring design. SQPOLL is NOT used
// and NOT wired (no reserved param): a kernel poller thread per core costs idle CPU
// and is mutually exclusive with DEFER_TASKRUN -- if a latency-bound workload ever
// justifies it, add a rung to the flag ladder then, not a dead interface now.
#  include <linux/io_uring.h>
#  include <linux/stat.h>
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
  // DEFER_TASKRUN active: completions run their task_work ONLY on an io_uring_enter
  // carrying IORING_ENTER_GETEVENTS -- a submit-only enter never surfaces them. So a
  // non-blocking reap(0) with ops in flight must still enter with GETEVENTS (min_
  // complete=0, no wait) to run the deferred work + harvest. Currently ALWAYS FALSE:
  // DEFER needs SINGLE_ISSUER, which needs per-worker ring enablement not yet done
  // (see try_init_uring). The field + its reap branch are the tested foundation kept
  // in place for when that ring-lifecycle change lands.
  bool defer_taskrun = false;
  // R_DISABLED: the ring starts disabled; the first reap (on the owner worker thread)
  // must call ENABLE_RINGS -- that makes the worker the SINGLE_ISSUER (not the creating
  // thread) -- and only then arm the wake-read (its enter must be on the issuer). Set
  // when the DEFER rung (which includes R_DISABLED) wins; cleared after the first reap
  // enables the ring. See reap()'s needs_enable block + arm_wake_read().
  bool needs_enable = false;
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
  // Registered (fixed) file table (io-07c). A sparse table registered with the
  // kernel; slots hold raw fds so submit_*_fixed can reference a slot INDEX with
  // IOSQE_FIXED_FILE (kernel skips per-op fget/fput). `file_slots` mirrors the kernel
  // table for our own free-slot bookkeeping (-1 = free, -2 = a kernel-allocated
  // direct descriptor). Owner-thread-only.
  std::vector<std::intptr_t> file_slots;  // empty = table not registered
  struct PendingDirectClose {
    std::uint64_t user_data;
    int slot;
  };
  // Direct accept asks the kernel to choose a free table slot. Track its token
  // until the CQE tells us which slot became occupied; direct close keeps that
  // slot reserved until its own successful CQE.
  std::vector<std::uint64_t> pending_direct_accepts;
  std::vector<PendingDirectClose> pending_direct_closes;
  // Native STATX needs stable kernel output storage until its CQE. Reap converts
  // each Linux structure into the public POD, then frees the record.
  struct PendingStatx {
    std::uint64_t user_data;
    FileStat* output;
    struct statx native;
  };
  std::vector<PendingStatx*> pending_statx;
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
  // Completion. TWO pending ops per fd -- one read-side (EPOLLIN: READ/RECVMSG/
  // ACCEPT) and one write-side (EPOLLOUT: WRITE/SENDMSG/CONNECT) -- so a full-duplex
  // socket (one fiber reading, another writing the SAME fd) does not lose an op
  // (the earlier single map<fd,op> had the second submit overwrite the first,
  // stranding a fiber forever -- red-team L3). `opcode` reuses the IORING_OP_* tags
  // so reap() can branch on which syscall to run when the fd fires.
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
    // One-shot accept's socklen_t*, or null. Socket recv/send message flags are
    // retained separately so readiness backends do not silently execute flags=0.
    void* addrlen = nullptr;
    std::uint32_t flags = 0;
    // Multishot accept emulation (io-07): if set, reap re-arms this ACCEPT watch
    // after each delivery and stamps Completion::kMore, mirroring io_uring's
    // native multishot upward. Only meaningful for IORING_OP_ACCEPT.
    bool multishot = false;
  };
  // Split by direction so read+write on ONE fd coexist. A fd may appear in both.
  // The registered epoll event mask for an fd is the UNION of whichever sides are
  // armed (EPOLLIN|EPOLLOUT when both), all under EPOLLONESHOT.
  // Flat HYLE SwissTables keep the readiness hot path allocation-free between
  // growths and make io/ a real consumer of container/'s SIMD control-byte probe.
  container::SwissTable<int, PendingOp> pending_in{mem::system_allocator()};
  container::SwissTable<int, PendingOp> pending_out{mem::system_allocator()};
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
    for (Impl::PendingStatx* pending : impl_->pending_statx) delete pending;
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
Proactor::Impl* Proactor::try_init_uring(unsigned entries,
                                         bool defer_worker_thread) noexcept {
  // Setup-flag ladder (liburing echo-server.c / proxy.c "preferred setup"), enabled
  // with a kernel-version fallback: the kernel rejects unsupported flags with -EINVAL,
  // so try the richest supported rung and step down. Correctness is identical across
  // rungs; only completion-delivery overhead differs.
  //   DEFER_TASKRUN + SINGLE_ISSUER + R_DISABLED (>=6.1): the best setup for a ring
  //     with ONE consumer thread (HYLE's per-core model). Completions run task_work
  //     only on the owner's own GETEVENTS enter (no IPI); SINGLE_ISSUER lets the kernel
  //     skip issuer locking. SINGLE_ISSUER pins the issuer task -- normally the ring
  //     CREATOR, but HYLE creates the ring on one thread (IoContext ctor) and reaps on
  //     another (the owner worker). R_DISABLED breaks that: the ring starts disabled and
  //     the task that later calls ENABLE_RINGS becomes the issuer -- so the FIRST reap
  //     (on the owner worker) enables it (see reap()'s needs_enable path), making the
  //     worker the issuer. Requires the reap path to enter GETEVENTS whenever ops are in
  //     flight -- the defer_taskrun branch in reap() does the non-blocking peek. Only
  //     offered when defer_worker_thread is set (the cross-thread IoContext case).
  //   COOP_TASKRUN (>=5.19): weaker cousin, no issuer/enable dance -- the step-down when
  //     defer isn't wanted or the kernel is <6.1. Completions still land without an IPI.
  //   SUBMIT_ALL (>=5.18): a bad SQE in a batch doesn't abort the whole submit -- HYLE
  //     folds many fibers' SQEs into one enter, so one bad fd must not drop its peers.
#ifndef IORING_SETUP_SUBMIT_ALL
#  define IORING_SETUP_SUBMIT_ALL (1u << 7)
#endif
#ifndef IORING_SETUP_COOP_TASKRUN
#  define IORING_SETUP_COOP_TASKRUN (1u << 8)
#endif
#ifndef IORING_SETUP_SINGLE_ISSUER
#  define IORING_SETUP_SINGLE_ISSUER (1u << 12)
#endif
#ifndef IORING_SETUP_DEFER_TASKRUN
#  define IORING_SETUP_DEFER_TASKRUN (1u << 13)
#endif
#ifndef IORING_SETUP_R_DISABLED
#  define IORING_SETUP_R_DISABLED (1u << 6)
#endif
#ifndef IORING_REGISTER_ENABLE_RINGS
#  define IORING_REGISTER_ENABLE_RINGS 12u
#endif
  // The DEFER rung is FIRST only when the caller opts in (cross-thread reap). Its
  // R_DISABLED means the ring needs a worker-thread ENABLE_RINGS before first use.
  const unsigned kDeferFlags = IORING_SETUP_DEFER_TASKRUN |
                               IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_R_DISABLED |
                               IORING_SETUP_SUBMIT_ALL;
  const unsigned kLadderWithDefer[] = {
      kDeferFlags,                                          // >=6.1 (best, R_DISABLED)
      IORING_SETUP_COOP_TASKRUN | IORING_SETUP_SUBMIT_ALL,  // >=5.19
      IORING_SETUP_SUBMIT_ALL,                              // >=5.18
      0u,                                                   // io-01 baseline
  };
  const unsigned kLadderNoDefer[] = {
      IORING_SETUP_COOP_TASKRUN | IORING_SETUP_SUBMIT_ALL,  // >=5.19
      IORING_SETUP_SUBMIT_ALL,                              // >=5.18
      0u,                                                   // io-01 baseline
  };
  const unsigned* kFlagLadder = defer_worker_thread ? kLadderWithDefer : kLadderNoDefer;
  const unsigned kLadderLen = defer_worker_thread ? 4u : 3u;
  io_uring_params p;
  int fd = -1;
  unsigned chosen_flags = 0;
  for (unsigned li = 0; li < kLadderLen; ++li) {
    unsigned flags = kFlagLadder[li];
    std::memset(&p, 0, sizeof(p));
    p.flags = flags;
    fd = sys_io_uring_setup(entries, &p);
    if (fd >= 0) { chosen_flags = flags; break; }
    // -EINVAL: this flag combo unsupported on this kernel -> try the next rung.
    // Any OTHER error (ENOSYS old kernel / EPERM sysctl-disabled) is fatal for all
    // rungs -> bail to the epoll fallback immediately.
    if (errno != EINVAL) return nullptr;
  }
  if (fd < 0) return nullptr;

  auto* impl = new (std::nothrow) Proactor::Impl{};
  if (impl == nullptr) {
    ::close(fd);
    return nullptr;
  }
  impl->backend = Proactor::Impl::kUring;
  impl->fd = fd;
  impl->sq_entries = p.sq_entries;
  impl->single_mmap = (p.features & IORING_FEAT_SINGLE_MMAP) != 0;
  // DEFER_TASKRUN chosen -> reap(0) must enter with GETEVENTS to surface deferred
  // completions (see the defer_taskrun branch in reap()).
  impl->defer_taskrun = (chosen_flags & IORING_SETUP_DEFER_TASKRUN) != 0;
  // R_DISABLED chosen -> the ring is disabled; the FIRST reap (on the owner worker)
  // must ENABLE_RINGS to become the single issuer, and only THEN arm the wake-read.
  impl->needs_enable = (chosen_flags & IORING_SETUP_R_DISABLED) != 0;

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
  if (impl->wake_efd >= 0 && !impl->needs_enable) {
    // Fill + submit the wake-read now. But NOT when R_DISABLED (needs_enable): the
    // ring is disabled (submit would fail) AND this is the creating thread, which is
    // NOT the issuer under SINGLE_ISSUER -- entering here would bind the wrong issuer.
    // In that case the first reap (on the owner worker) enables the ring THEN arms this
    // (see reap()'s needs_enable block), so the issuer is the worker.
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
Proactor::Impl* Proactor::try_init_epoll(unsigned entries) noexcept {
  int ep = ::epoll_create1(EPOLL_CLOEXEC);
  if (ep < 0) return nullptr;
  auto* impl = new (std::nothrow) Proactor::Impl{};
  if (impl == nullptr) {
    ::close(ep);
    return nullptr;
  }
  impl->backend = Proactor::Impl::kEpoll;
  impl->epfd = ep;
  if (!impl->pending_in.reserve(entries) ||
      !impl->pending_out.reserve(entries)) {
    delete impl;
    ::close(ep);
    return nullptr;
  }
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

bool Proactor::init(unsigned entries, bool defer_worker_thread) noexcept {
  if (impl_ != nullptr) return true;  // idempotent
  // SOTA-first: always try io_uring; production runs on it wherever it exists.
  // Fall back to epoll ONLY when io_uring is truly unavailable (old kernel /
  // sysctl-disabled / seccomp-blocked). A test-only env var forces the epoll
  // path so the fallback is exercised even where io_uring works (this machine).
  const char* force = ::getenv("HYLE_IO_FORCE_EPOLL");
  if (force == nullptr || force[0] == '0' || force[0] == '\0') {
    Impl* u = try_init_uring(entries, defer_worker_thread);
    if (u != nullptr) {
      impl_ = u;
      return true;
    }
    // io_uring unavailable -> fall through to epoll (the whole point of io-03).
  }
  Impl* e = try_init_epoll(entries);
  if (e == nullptr) return false;
  impl_ = e;
  return true;
}

BackendKind Proactor::backend_kind() const noexcept {
  if (impl_ == nullptr) return HYLE_IO_BACKEND_NONE;
  return impl_->backend == Impl::kUring ? HYLE_IO_BACKEND_IO_URING
                                        : HYLE_IO_BACKEND_EPOLL;
}
const char* Proactor::backend_name() const noexcept {
  if (impl_ == nullptr) return "none";
  return impl_->backend == Impl::kUring ? "io_uring" : "epoll";
}
ProactorCapabilities Proactor::capabilities() const noexcept {
  std::uint64_t bits = 0;
  if (impl_ != nullptr) {
    if (impl_->backend == Impl::kUring) {
      bits |= HYLE_IO_CAP_NATIVE_MULTISHOT_ACCEPT |
              HYLE_IO_CAP_PROVIDED_BUFFER_RING |
              HYLE_IO_CAP_FIXED_FILES | HYLE_IO_CAP_DIRECT_ACCEPT |
              HYLE_IO_CAP_ASYNC_FILE_DATA |
              HYLE_IO_CAP_ASYNC_FILE_METADATA;
    } else {
      bits |= HYLE_IO_CAP_EMULATED_MULTISHOT_ACCEPT;
    }
  }
#if defined(HYLE_IO_TLS)
  bits |= HYLE_IO_CAP_TLS;
#endif
#if defined(HYLE_IO_HTTP2)
  bits |= HYLE_IO_CAP_HTTP2;
#endif
  return {sizeof(ProactorCapabilities), backend_kind(), bits};
}
bool Proactor::has_async_file_data() const noexcept {
  return impl_ != nullptr && impl_->backend == Impl::kUring;
}
bool Proactor::has_async_file_metadata() const noexcept {
  return impl_ != nullptr && impl_->backend == Impl::kUring;
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

    // Perform the ready op inline and synthesize its result. The op tag says
    // which syscall. Shared by both directions (read-side / write-side).
    auto perform = [](int fd, const Impl::PendingOp& op) -> std::int64_t {
      if (op.opcode == IORING_OP_WRITE) {
        ssize_t w = ::write(fd, op.buf, op.len);
        return (w < 0) ? -errno : static_cast<std::int64_t>(w);
      } else if (op.opcode == IORING_OP_SEND) {
        ssize_t w = ::send(fd, op.buf, op.len, static_cast<int>(op.flags));
        return (w < 0) ? -errno : static_cast<std::int64_t>(w);
      } else if (op.opcode == IORING_OP_ACCEPT) {
        int as = ::accept4(fd, static_cast<sockaddr*>(op.addr),
                           static_cast<socklen_t*>(op.addrlen),
                           SOCK_NONBLOCK | SOCK_CLOEXEC);
        return (as < 0) ? -errno : static_cast<std::int64_t>(as);
      } else if (op.opcode == IORING_OP_CONNECT) {
        int soerr = 0;
        socklen_t sl = sizeof(soerr);
        ::getsockopt(fd, SOL_SOCKET, SO_ERROR, &soerr, &sl);
        return (soerr == 0) ? 0 : -soerr;
      } else if (op.opcode == IORING_OP_RECVMSG) {
        socklen_t al = static_cast<socklen_t>(op.addr_cap);
        ssize_t r = ::recvfrom(fd, op.buf, op.len, 0,
                               static_cast<sockaddr*>(op.addr),
                               op.addr ? &al : nullptr);
        return (r < 0) ? -errno : static_cast<std::int64_t>(r);
      } else if (op.opcode == IORING_OP_SENDMSG) {
        ssize_t w = ::sendto(fd, op.buf, op.len, 0,
                             static_cast<const sockaddr*>(op.addr),
                             static_cast<socklen_t>(op.addr_cap));
        return (w < 0) ? -errno : static_cast<std::int64_t>(w);
      } else if (op.opcode == IORING_OP_RECV) {
        ssize_t r = ::recv(fd, op.buf, op.len, static_cast<int>(op.flags));
        return (r < 0) ? -errno : static_cast<std::int64_t>(r);
      } else {  // READ
        ssize_t r = ::read(fd, op.buf, op.len);
        return (r < 0) ? -errno : static_cast<std::int64_t>(r);
      }
    };

    for (int i = 0; i < got && n < max; ++i) {
      int fd = static_cast<int>(evs[i].data.u64);  // we store fd in data.u64
      if (fd == impl_->wake_efd) {
        // Cross-thread wake (park merge): drain the counter so the level-triggered
        // watch stops firing, and emit NO completion -- it only breaks the block.
        std::uint64_t sink;
        while (::read(impl_->wake_efd, &sink, sizeof(sink)) == sizeof(sink)) {}
        continue;
      }
      // EPOLLONESHOT disarmed the whole fd. Serve whichever side(s) fired, then
      // (below) either re-arm a still-pending other side or DEL the fd. HUP/ERR
      // wake both sides so a half-closed peer resolves both a pending read and
      // write instead of stranding one (red-team L3: full-duplex fd).
      const bool err = (evs[i].events & (EPOLLHUP | EPOLLERR)) != 0;
      const bool rd_ready =
          (evs[i].events & (EPOLLIN | EPOLLPRI)) != 0 || err;
      const bool wr_ready = (evs[i].events & (EPOLLOUT)) != 0 || err;
      bool multishot_rearmed = false;

      // A LEVEL/oneshot readiness can be SPURIOUS: epoll says "readable" but by the
      // time we run the inline syscall the data/connection is gone, and read/recv/
      // accept returns -EAGAIN/-EWOULDBLOCK. A real proactor (io_uring/IOCP) NEVER
      // surfaces EAGAIN -- the op just stays in flight. So on a spurious EAGAIN we
      // must NOT deliver it as the op's result (the fiber would see a bogus error
      // and likely drop the connection); instead re-arm the op and keep waiting for a
      // TRUE readiness. We do that by re-inserting the op into its pending map and
      // NOT emitting a completion -- the unified re-arm below then re-watches the fd.
      // Guarded by !err: if the fd genuinely HUP/ERR'd, deliver perform's result
      // (an EAGAIN there is terminal, not spurious -- avoid a re-arm spin). (L4.)
      auto is_spurious = [&](std::int64_t res) noexcept {
        return !err && (res == -EAGAIN || res == -EWOULDBLOCK);
      };

      // ---- read-side (recv/accept/recvfrom) ----
      if (rd_ready && n < max) {
        if (Impl::PendingOp* pending = impl_->pending_in.find(fd)) {
          Impl::PendingOp op = *pending;
          impl_->pending_in.erase(fd);
          std::int64_t res = perform(fd, op);
          if (is_spurious(res)) {
            impl_->pending_in.insert(fd, op);  // false alarm: re-watch, deliver nothing
          } else {
            out[n].user_data = op.user_data;
            out[n].result = res;
            out[n].flags = 0;
            out[n].buffer_id = 0;
            // Multishot accept emulation: re-arm + stamp kMore ONLY if the re-arm
            // succeeds (a failed re-arm is terminal -- stamping kMore would hang the
            // caller waiting for a stream that ended; matches IOCP). We re-insert into
            // pending_in and let the unified re-arm below register the fd's watch.
            if (op.multishot && res >= 0) {
              impl_->pending_in.insert(fd, op);  // keep multishot for next delivery
              out[n].flags |= Completion::kMore;
              multishot_rearmed = true;
            }
            ++n;
          }
        }
      }
      // ---- write-side (send/connect/sendto) ----
      if (wr_ready && n < max) {
        if (Impl::PendingOp* pending = impl_->pending_out.find(fd)) {
          Impl::PendingOp op = *pending;
          impl_->pending_out.erase(fd);
          std::int64_t res = perform(fd, op);
          if (is_spurious(res)) {
            impl_->pending_out.insert(fd, op);  // false alarm: re-watch, deliver nothing
          } else {
            out[n].user_data = op.user_data;
            out[n].result = res;
            out[n].flags = 0;
            out[n].buffer_id = 0;
            ++n;
          }
        }
      }
      // Re-arm the fd for whatever side(s) remain pending (the other side we didn't
      // serve, and/or a just-re-armed multishot accept). If NOTHING is left, DEL the
      // fd. If the re-arm MOD fails, the stranded side's fiber would hang -- but a
      // MOD on a live epfd+fd effectively never fails; DEL-on-empty is the norm.
      unsigned mask = 0;
      if (Impl::PendingOp* in = impl_->pending_in.find(fd)) {
        mask |= (in->opcode == IORING_OP_RECV &&
                 (in->flags & MSG_OOB) != 0)
                    ? EPOLLPRI
                    : EPOLLIN;
      }
      if (impl_->pending_out.contains(fd)) mask |= EPOLLOUT;
      if (mask != 0) {
        epoll_event rev{};
        rev.events = mask | EPOLLONESHOT;
        rev.data.u64 = static_cast<std::uint64_t>(fd);
        // The fd is still registered (EPOLLONESHOT only disables events, not the
        // registration), so MOD. A multishot that was DEL'd in a prior life uses ADD
        // -- but here the fd was armed this round, so it is registered: MOD.
        if (::epoll_ctl(impl_->epfd, EPOLL_CTL_MOD, fd, &rev) < 0) {
          // Re-arm failed: drop the stranded side(s) so we don't leak the map entry.
          // A multishot that failed re-arm must also clear its kMore (already
          // emitted this round) -- but MOD essentially never fails on a live fd, so
          // this is a defensive cleanup, not an expected path.
          if (multishot_rearmed) { /* kMore already sent; nothing more we can do */ }
          impl_->pending_in.erase(fd);
          impl_->pending_out.erase(fd);
          ::epoll_ctl(impl_->epfd, EPOLL_CTL_DEL, fd, nullptr);
        }
      } else {
        ::epoll_ctl(impl_->epfd, EPOLL_CTL_DEL, fd, nullptr);  // both sides done
      }
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

  // R_DISABLED lifecycle (DEFER_TASKRUN path): the ring was created disabled on the
  // constructing thread; enable it HERE, on the FIRST reap, which runs on the owner
  // WORKER thread -- so THIS thread becomes the SINGLE_ISSUER (the ENABLE_RINGS caller
  // is the issuer, per io_uring_setup.2). Only then arm the wake-read (its submit must
  // be by the issuer). One-time: cleared after. If ENABLE_RINGS fails, fall back to a
  // usable state by clearing the flags (the ring can't be used disabled, but we avoid
  // a hard hang -- the scheduler's reap just returns nothing and the io stalls visibly
  // rather than corrupting). This block is the whole reason DEFER_TASKRUN is safe on
  // HYLE's create-on-one-thread / reap-on-another model.
  if (impl_->needs_enable) {
    impl_->needs_enable = false;  // once, regardless of outcome
    if (sys_io_uring_register(impl_->fd, IORING_REGISTER_ENABLE_RINGS, nullptr, 0) < 0) {
      // Enable failed: we can't use the ring. Nothing to reap; return empty.
      return 0;
    }
    // Now the issuer -- arm the persistent wake-read (deferred from init). Fill the
    // SQE; it flushes to the kernel via THIS reap's own enter below (sq_pending++).
    if (impl_->wake_efd >= 0) {
      unsigned wt = *impl_->sq_tail;
      unsigned wi = wt & *impl_->sq_ring_mask;
      io_uring_sqe& ws = impl_->sqes[wi];
      std::memset(&ws, 0, sizeof(ws));
      ws.opcode = IORING_OP_READ;
      ws.fd = impl_->wake_efd;
      ws.addr = reinterpret_cast<std::uint64_t>(&impl_->wake_buf);
      ws.len = sizeof(impl_->wake_buf);
      ws.user_data = Proactor::kWakeToken;
      impl_->sq_array[wi] = wi;
      as_atomic(impl_->sq_tail)->store(wt + 1, std::memory_order_release);
      ++impl_->sq_pending;  // flushed by this reap's enter (wake-read isn't inflight)
    }
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
      // Non-blocking poll, CQ empty. Normally: flush the batch (no wait) so it
      // reaches the kernel, then return -- completions show up on a later reap.
      // BUT under DEFER_TASKRUN a completion's task_work runs only on a GETEVENTS
      // enter, so a submit-only enter would leave it stuck in task_work forever
      // (the drain-time reap(0) never surfaces it -> fibers hang). So when defer is
      // on, enter with GETEVENTS + min_complete=0: run the deferred work and harvest
      // whatever is ready WITHOUT blocking (the io_uring_peek_cqe idiom under defer).
      if (impl_->defer_taskrun) {
        int r = sys_io_uring_enter(impl_->fd, submit, 0, IORING_ENTER_GETEVENTS,
                                   nullptr, 0);
        impl_->stat_enters += 1;
        impl_->stat_sqes += submit;
        if (r < 0 && errno != EINTR) return 0;
        tail = as_atomic(impl_->cq_tail)->load(std::memory_order_acquire);
        head = *impl_->cq_head;
        // fall through to the userspace harvest loop below with the fresh tail
      } else {
        if (submit != 0) {
          sys_io_uring_enter(impl_->fd, submit, 0, 0, nullptr, 0);
          impl_->stat_enters += 1;
          impl_->stat_sqes += submit;
        }
        return 0;
      }
    } else {
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
    }
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
    // A successful direct accept returns the kernel-selected fixed-file slot.
    // Reserve it in our mirror before handing the completion to the caller, so
    // assign_fixed_file cannot overwrite the live direct descriptor.
    for (std::size_t i = 0; i < impl_->pending_direct_accepts.size(); ++i) {
      if (impl_->pending_direct_accepts[i] != c.user_data) continue;
      if (c.res >= 0 &&
          static_cast<std::size_t>(c.res) < impl_->file_slots.size()) {
        impl_->file_slots[static_cast<std::size_t>(c.res)] = -2;
      }
      impl_->pending_direct_accepts[i] =
          impl_->pending_direct_accepts.back();
      impl_->pending_direct_accepts.pop_back();
      break;
    }
    // Keep a direct slot reserved while CLOSE is in flight. Only a successful
    // CQE makes it locally reusable; a failed close still names a live/unknown
    // kernel slot and must not be silently recycled.
    for (std::size_t i = 0; i < impl_->pending_direct_closes.size(); ++i) {
      const Impl::PendingDirectClose pending =
          impl_->pending_direct_closes[i];
      if (pending.user_data != c.user_data) continue;
      if (c.res >= 0 && pending.slot >= 0 &&
          static_cast<std::size_t>(pending.slot) < impl_->file_slots.size()) {
        impl_->file_slots[static_cast<std::size_t>(pending.slot)] = -1;
      }
      impl_->pending_direct_closes[i] =
          impl_->pending_direct_closes.back();
      impl_->pending_direct_closes.pop_back();
      break;
    }
    // STATX writes a Linux-only structure. Convert it to the stable public POD
    // before the completion wakes its caller, then release the native storage.
    for (std::size_t i = 0; i < impl_->pending_statx.size(); ++i) {
      Impl::PendingStatx* pending = impl_->pending_statx[i];
      if (pending->user_data != c.user_data) continue;
      if (c.res >= 0) {
        FileStat value{};
        value.size = pending->native.stx_size;
        value.mtime_ns =
            static_cast<std::uint64_t>(pending->native.stx_mtime.tv_sec) *
                1'000'000'000ull +
            static_cast<std::uint32_t>(pending->native.stx_mtime.tv_nsec);
        value.mode = pending->native.stx_mode;
        value.is_dir = (pending->native.stx_mode & S_IFMT) == S_IFDIR;
        *pending->output = value;
      }
      delete pending;
      impl_->pending_statx[i] = impl_->pending_statx.back();
      impl_->pending_statx.pop_back();
      break;
    }
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
    return !impl_->pending_in.empty() || !impl_->pending_out.empty() ||
           !impl_->posted.empty();
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
bool reserve_sqe(int ring_fd, unsigned sq_entries, unsigned* sq_pending,
                 unsigned* sq_tail, unsigned sq_mask, unsigned* sq_array,
                 io_uring_sqe* sqes, io_uring_sqe** sqe_out) noexcept {
  while (*sq_pending >= sq_entries) {
    const int submitted =
        sys_io_uring_enter(ring_fd, *sq_pending, 0, 0, nullptr, 0);
    if (submitted <= 0) return false;
    *sq_pending -= static_cast<unsigned>(submitted);
  }
  unsigned tail = *sq_tail;
  unsigned index = tail & sq_mask;
  io_uring_sqe& s = sqes[index];
  std::memset(&s, 0, sizeof(s));
  sq_array[index] = index;
  *sqe_out = &s;
  return true;
}

void commit_sqe(unsigned* sq_pending, std::uint64_t* inflight,
                unsigned* sq_tail, unsigned tail) noexcept {
  as_atomic(sq_tail)->store(tail + 1, std::memory_order_release);
  ++*sq_pending;
  ++*inflight;
}

bool submit_one(int ring_fd, unsigned sq_entries, unsigned* sq_pending,
                std::uint64_t* inflight, unsigned* sq_tail, unsigned sq_mask,
                unsigned* sq_array, io_uring_sqe* sqes, std::uint8_t opcode,
                std::int32_t fd, std::uint64_t addr, std::uint32_t len,
                std::uint64_t off, std::uint64_t user_data,
                std::uint32_t op_flags = 0, std::uint16_t ioprio = 0) noexcept {
  io_uring_sqe* sqe = nullptr;
  if (!reserve_sqe(ring_fd, sq_entries, sq_pending, sq_tail, sq_mask,
                   sq_array, sqes, &sqe))
    return false;
  const unsigned tail = *sq_tail;
  io_uring_sqe& s = *sqe;
  s.opcode = opcode;
  s.fd = fd;
  s.off = off;
  s.addr = addr;
  s.len = len;
  s.rw_flags = static_cast<int>(op_flags);  // union: msg_flags/accept_flags too
  s.ioprio = ioprio;                        // multishot bits (accept/recv), else 0
  s.user_data = user_data;
  // publish SQ tail (store_release pairs with kernel load_acquire); the enter
  // that tells the kernel to consume these happens in reap() (batched).
  commit_sqe(sq_pending, inflight, sq_tail, tail);
  // One more op the kernel owes us a completion for. reap() decrements per CQE.
  // The worker reads this (Proactor::in_flight) to decide it must BLOCK in a
  // GETEVENTS reap rather than the futex parker, or those completions never
  // surface (the >ring-depth hang). Owner-thread-only, no atomic needed.
  return true;
}
}  // namespace

bool Proactor::post(std::uint64_t user_data, std::int64_t result) noexcept {
  if (impl_ == nullptr) return false;
  // post() is owner-thread-only; wake() is the cross-thread primitive. Keep the
  // synthetic completion in user space for BOTH io_uring and epoll so its signed
  // 64-bit result is preserved. An io_uring NOP CQE always reports res=0, which
  // previously discarded `result` and made the Linux C ABI differ from IOCP.
  impl_->posted.push_back({user_data, result});
  return true;
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
//
// FULL-DUPLEX (red-team L3): read-side ops go in pending_in, write-side in
// pending_out, so a concurrent read+write on one fd don't clobber each other. The
// single epoll registration for the fd carries the UNION of both sides' events
// (EPOLLIN|EPOLLOUT when both are armed), under EPOLLONESHOT. When one side fires,
// EPOLLONESHOT disarms the whole fd; reap serves the fired side(s) and re-arms any
// side still pending (see reap's epoll branch).
bool Proactor::epoll_arm_dgram(Impl* impl, int fd, unsigned events,
                               unsigned char opcode, void* buf, unsigned len,
                               void* addr, unsigned addr_cap, void* addrlen,
                               unsigned flags,
                               std::uint64_t user_data,
                               std::uint64_t offset) noexcept {
  const bool is_read = (events & (EPOLLIN | EPOLLPRI)) != 0;
  auto& side = is_read ? impl->pending_in : impl->pending_out;
  auto& other = is_read ? impl->pending_out : impl->pending_in;
  // Was this fd already registered with epoll (either side armed) before us?
  const bool side_pending = side.contains(fd);
  const bool other_pending = other.contains(fd);
  const bool fd_already_watched = side_pending || other_pending;
  // A readiness backend can keep exactly one operation per fd+direction.
  // Replacing it would orphan the first awaiter forever. Make possible table
  // growth explicit as well, so OOM returns submit=false instead of reaching
  // SwissTable's infallible-insert assertion inside this noexcept API.
  if (side_pending || !side.try_prepare_insert()) return false;
  side.insert(fd, Impl::PendingOp{opcode, buf, len, user_data, addr, addr_cap,
                                  addrlen, flags});

  epoll_event ev{};
  // Union of armed directions: our side + whatever the other side already has.
  ev.events = events | EPOLLONESHOT;
  if (other_pending) {
    if (is_read) {
      ev.events |= EPOLLOUT;
    } else if (Impl::PendingOp* in = impl->pending_in.find(fd)) {
      ev.events |= (in->opcode == IORING_OP_RECV &&
                    (in->flags & MSG_OOB) != 0)
                       ? EPOLLPRI
                       : EPOLLIN;
    }
  }
  ev.data.u64 = static_cast<std::uint64_t>(fd);
  const int op_ctl = fd_already_watched ? EPOLL_CTL_MOD : EPOLL_CTL_ADD;
  if (::epoll_ctl(impl->epfd, op_ctl, fd, &ev) < 0) {
    if (op_ctl == EPOLL_CTL_ADD && errno == EEXIST) {
      // Raced/stale registration for this fd -> MOD instead.
      if (::epoll_ctl(impl->epfd, EPOLL_CTL_MOD, fd, &ev) < 0) {
        side.erase(fd);
        return false;
      }
    } else if (op_ctl == EPOLL_CTL_ADD && errno == EPERM) {
      // EPERM = fd is a REGULAR FILE (or otherwise not epoll-able). epoll cannot
      // watch regular files -- they are always "ready" but read()/write() may block
      // on disk (io-05: only io_uring/IOCP async a regular file; tokio offloads to
      // a thread pool for exactly this). So do the byte IO SYNCHRONOUSLY now, AT the
      // file offset (pread/pwrite -- the proactor read/write contract is offset-
      // based, unlike a stream), and synthesize a completion via the posted queue
      // (the same synthesis the metadata file ops use). Stream/datagram sockets are
      // never regular files, so this branch is file-only (and a regular file is
      // never in `other`, so the union logic above is a no-op here).
      side.erase(fd);
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
      side.erase(fd);
      return false;
    }
  }
  return true;
}

bool Proactor::epoll_arm(Impl* impl, int fd, unsigned events,
                         unsigned char opcode, void* buf, unsigned len,
                         std::uint64_t user_data, std::uint64_t offset) noexcept {
  return epoll_arm_dgram(impl, fd, events, opcode, buf, len, nullptr, 0,
                         nullptr, 0, user_data, offset);
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
    const unsigned events = (flags & MSG_OOB) != 0 ? EPOLLPRI : EPOLLIN;
    return epoll_arm_dgram(impl_, static_cast<int>(fd), events,
                           IORING_OP_RECV, buf, len, nullptr, 0, nullptr, flags,
                           user_data);
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
  // R_DISABLED-pending (DEFER path): the ring isn't enabled yet -- it enables on the
  // first reap, on the OWNER WORKER thread. register_buf_ring is called on the CREATING
  // thread before the scheduler runs, so we can neither register (ring disabled) nor
  // enable here (that would bind the wrong SINGLE_ISSUER). So a provided-buffer ring is
  // incompatible with the DEFER path as currently wired -> return false (callers, which
  // are tests/benches, already self-skip on false). Provided-buf-ring + DEFER on one
  // proactor is a documented mutual exclusion (a rare combo: DEFER is the throughput
  // default, buf-ring is the multishot/zero-copy path -- wiring both needs the ring
  // registered from a worker fiber, deferred).
  if (impl_->needs_enable) return false;
  if (impl_->buf_bgid >= 0) return false;  // one ring supported for now
  if (bgid < 0 || bgid > UINT16_MAX || nentries == 0 ||
      nentries > 32768 || buf_size == 0)
    return false;
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

// ---- registered (fixed) files (io-07c) ------------------------------------
#ifndef IORING_REGISTER_FILES2
#  define IORING_REGISTER_FILES2 13u
#endif
#ifndef IORING_REGISTER_FILES_UPDATE
#  define IORING_REGISTER_FILES_UPDATE 6u
#endif
#ifndef IORING_RSRC_REGISTER_SPARSE
#  define IORING_RSRC_REGISTER_SPARSE (1u << 0)
#endif

bool Proactor::register_file_table(unsigned nr) noexcept {
  if (impl_ == nullptr || impl_->backend != Impl::kUring) return false;
  if (nr == 0 || !impl_->file_slots.empty()) return false;  // once
  if (impl_->needs_enable) return false;  // R_DISABLED-pending: can't register yet
  // Register a SPARSE table (all slots empty). io_uring_rsrc_register: nr slots,
  // SPARSE flag, no data/tags. The kernel allocates the table; we fill slots later
  // via FILES_UPDATE (assign_fixed_slot).
  io_uring_rsrc_register rr;
  std::memset(&rr, 0, sizeof(rr));
  rr.nr = nr;
  rr.flags = IORING_RSRC_REGISTER_SPARSE;
  if (sys_io_uring_register(impl_->fd, IORING_REGISTER_FILES2, &rr, sizeof(rr)) < 0)
    return false;
  impl_->file_slots.assign(nr, -1);  // mirror for free-slot bookkeeping (-1 = free)
  impl_->pending_direct_accepts.reserve(nr);
  impl_->pending_direct_closes.reserve(nr);
  return true;
}

int Proactor::assign_fixed_slot(std::intptr_t fd) noexcept {
  if (impl_ == nullptr || impl_->backend != Impl::kUring) return -1;
  if (impl_->file_slots.empty()) return -1;  // table not registered
  // Find a free slot (linear scan -- tables are small; a free-list is overkill).
  int slot = -1;
  for (std::size_t i = 0; i < impl_->file_slots.size(); ++i) {
    if (impl_->file_slots[i] == -1) { slot = static_cast<int>(i); break; }
  }
  if (slot < 0) return -1;  // table full
  // Point the kernel table slot at this fd (FILES_UPDATE at offset=slot, one fd).
  std::int32_t raw = static_cast<std::int32_t>(fd);
  io_uring_rsrc_update ru;
  std::memset(&ru, 0, sizeof(ru));
  ru.offset = static_cast<std::uint32_t>(slot);
  ru.data = reinterpret_cast<std::uint64_t>(&raw);
  if (sys_io_uring_register(impl_->fd, IORING_REGISTER_FILES_UPDATE, &ru, 1) < 0)
    return -1;
  impl_->file_slots[static_cast<std::size_t>(slot)] = fd;
  return slot;
}

bool Proactor::release_fixed_slot(int index) noexcept {
  if (impl_ == nullptr || impl_->backend != Impl::kUring) return false;
  if (index < 0 || static_cast<std::size_t>(index) >= impl_->file_slots.size())
    return false;
  // Kernel-allocated direct descriptors are closed with submit_close_direct;
  // FILES_UPDATE release is only valid for assign_fixed_file mappings.
  if (impl_->file_slots[static_cast<std::size_t>(index)] < 0) return false;
  // Clear the kernel table entry (FILES_UPDATE with fd = -1) so a stale slot can't
  // resolve to a closed fd, then mark it free for reuse.
  std::int32_t empty = -1;
  io_uring_rsrc_update ru;
  std::memset(&ru, 0, sizeof(ru));
  ru.offset = static_cast<std::uint32_t>(index);
  ru.data = reinterpret_cast<std::uint64_t>(&empty);
  if (sys_io_uring_register(impl_->fd, IORING_REGISTER_FILES_UPDATE, &ru, 1) < 0)
    return false;
  impl_->file_slots[static_cast<std::size_t>(index)] = -1;
  return true;
}

bool Proactor::submit_recv_fixed(int slot, void* buf, std::uint32_t len,
                                 std::uint32_t flags,
                                 std::uint64_t user_data) noexcept {
  if (impl_ == nullptr || impl_->backend != Impl::kUring) return false;
  if (slot < 0 || static_cast<std::size_t>(slot) >= impl_->file_slots.size() ||
      impl_->file_slots[static_cast<std::size_t>(slot)] == -1)
    return false;
  // Fill the SQE directly (submit_one doesn't set sqe.flags): IOSQE_FIXED_FILE +
  // sqe.fd = the SLOT INDEX (the kernel resolves the real fd from its table).
  io_uring_sqe* sqe = nullptr;
  if (!reserve_sqe(impl_->fd, impl_->sq_entries, &impl_->sq_pending,
                   impl_->sq_tail, *impl_->sq_ring_mask, impl_->sq_array,
                   impl_->sqes, &sqe))
    return false;
  const unsigned tail = *impl_->sq_tail;
  io_uring_sqe& s = *sqe;
  s.opcode = IORING_OP_RECV;
  s.fd = slot;
  s.flags = IOSQE_FIXED_FILE;
  s.addr = reinterpret_cast<std::uint64_t>(buf);
  s.len = len;
  s.rw_flags = static_cast<int>(flags);
  s.user_data = user_data;
  commit_sqe(&impl_->sq_pending, &impl_->inflight, impl_->sq_tail, tail);
  return true;
}

bool Proactor::submit_send_fixed(int slot, const void* buf, std::uint32_t len,
                                 std::uint32_t flags,
                                 std::uint64_t user_data) noexcept {
  if (impl_ == nullptr || impl_->backend != Impl::kUring) return false;
  if (slot < 0 || static_cast<std::size_t>(slot) >= impl_->file_slots.size() ||
      impl_->file_slots[static_cast<std::size_t>(slot)] == -1)
    return false;
  io_uring_sqe* sqe = nullptr;
  if (!reserve_sqe(impl_->fd, impl_->sq_entries, &impl_->sq_pending,
                   impl_->sq_tail, *impl_->sq_ring_mask, impl_->sq_array,
                   impl_->sqes, &sqe))
    return false;
  const unsigned tail = *impl_->sq_tail;
  io_uring_sqe& s = *sqe;
  s.opcode = IORING_OP_SEND;
  s.fd = slot;
  s.flags = IOSQE_FIXED_FILE;
  s.addr = reinterpret_cast<std::uint64_t>(buf);
  s.len = len;
  s.rw_flags = static_cast<int>(flags);
  s.user_data = user_data;
  commit_sqe(&impl_->sq_pending, &impl_->inflight, impl_->sq_tail, tail);
  return true;
}

#ifndef IORING_FILE_INDEX_ALLOC
#  define IORING_FILE_INDEX_ALLOC (~0u)
#endif

bool Proactor::submit_accept_direct(std::intptr_t fd,
                                    std::uint64_t user_data) noexcept {
  if (impl_ == nullptr || impl_->backend != Impl::kUring) return false;
  if (impl_->file_slots.empty()) return false;  // needs a registered table
  std::size_t free_slots = 0;
  for (const std::intptr_t value : impl_->file_slots)
    if (value == -1) ++free_slots;
  if (free_slots <= impl_->pending_direct_accepts.size()) return false;
  // ACCEPT with file_index = ALLOC: the kernel accepts + places the new connection in
  // a FREE table slot as a direct descriptor, and the CQE result is the chosen slot
  // index. No addr/addrlen (peer address not needed here). submit_one doesn't set
  // sqe.file_index, so fill the SQE directly.
  io_uring_sqe* sqe = nullptr;
  if (!reserve_sqe(impl_->fd, impl_->sq_entries, &impl_->sq_pending,
                   impl_->sq_tail, *impl_->sq_ring_mask, impl_->sq_array,
                   impl_->sqes, &sqe))
    return false;
  const unsigned tail = *impl_->sq_tail;
  io_uring_sqe& s = *sqe;
  s.opcode = IORING_OP_ACCEPT;
  s.fd = static_cast<std::int32_t>(fd);
  s.accept_flags = SOCK_NONBLOCK | SOCK_CLOEXEC;
  s.file_index = IORING_FILE_INDEX_ALLOC;  // kernel picks a free slot
  s.user_data = user_data;
  impl_->pending_direct_accepts.push_back(user_data);
  commit_sqe(&impl_->sq_pending, &impl_->inflight, impl_->sq_tail, tail);
  return true;
}

bool Proactor::submit_close_direct(int slot, std::uint64_t user_data) noexcept {
  if (impl_ == nullptr || impl_->backend != Impl::kUring) return false;
  if (slot < 0 || static_cast<std::size_t>(slot) >= impl_->file_slots.size() ||
      impl_->file_slots[static_cast<std::size_t>(slot)] != -2)
    return false;
  for (const Impl::PendingDirectClose& pending :
       impl_->pending_direct_closes)
    if (pending.slot == slot) return false;
  // CLOSE with file_index (fd unused): closes the direct descriptor in that
  // slot. Keep the mirror reserved until the successful close CQE.
  io_uring_sqe* sqe = nullptr;
  if (!reserve_sqe(impl_->fd, impl_->sq_entries, &impl_->sq_pending,
                   impl_->sq_tail, *impl_->sq_ring_mask, impl_->sq_array,
                   impl_->sqes, &sqe))
    return false;
  const unsigned tail = *impl_->sq_tail;
  io_uring_sqe& s = *sqe;
  s.opcode = IORING_OP_CLOSE;
  s.file_index = static_cast<std::uint32_t>(slot);
  s.user_data = user_data;
  impl_->pending_direct_closes.push_back({user_data, slot});
  commit_sqe(&impl_->sq_pending, &impl_->inflight, impl_->sq_tail, tail);
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
  io_uring_sqe* sqe = nullptr;
  if (!reserve_sqe(impl_->fd, impl_->sq_entries, &impl_->sq_pending,
                   impl_->sq_tail, *impl_->sq_ring_mask, impl_->sq_array,
                   impl_->sqes, &sqe))
    return false;
  const unsigned tail = *impl_->sq_tail;
  io_uring_sqe& s = *sqe;
  s.opcode = IORING_OP_RECV;
  s.fd = static_cast<std::int32_t>(fd);
  s.ioprio = IORING_RECV_MULTISHOT;
  s.flags = IOSQE_BUFFER_SELECT;
  s.buf_group = static_cast<std::uint16_t>(bgid);
  s.user_data = user_data;
  commit_sqe(&impl_->sq_pending, &impl_->inflight, impl_->sq_tail, tail);
  return true;
}

bool Proactor::submit_send(std::intptr_t fd, const void* buf, std::uint32_t len,
                           std::uint32_t flags,
                           std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  if (impl_->backend == Impl::kEpoll) {
    return epoll_arm_dgram(impl_, static_cast<int>(fd), EPOLLOUT,
                           IORING_OP_SEND, const_cast<void*>(buf), len, nullptr,
                           0, nullptr, flags, user_data);
  }
  return submit_one(impl_->fd, impl_->sq_entries, &impl_->sq_pending, &impl_->inflight, impl_->sq_tail, *impl_->sq_ring_mask,
                    impl_->sq_array, impl_->sqes, IORING_OP_SEND,
                    static_cast<std::int32_t>(fd),
                    reinterpret_cast<std::uint64_t>(buf), len, 0, user_data,
                    flags);
}

bool Proactor::submit_accept(std::intptr_t fd, void* addr, void* addrlen,
                             std::uint64_t user_data) noexcept {
  if (impl_ == nullptr || ((addr == nullptr) != (addrlen == nullptr)))
    return false;
  if (impl_->backend == Impl::kEpoll) {
    // Reactor accept: watch the listener for EPOLLIN (a queued connection makes
    // it readable); reap() runs ::accept when it fires and returns the fd. The
    // peer addr is read by net/ via getpeername on the accepted fd, so buf/len
    // stay 0 (the caller's addr/addrlen aren't populated on the epoll path).
    return epoll_arm_dgram(impl_, static_cast<int>(fd), EPOLLIN,
                           IORING_OP_ACCEPT, nullptr, 0, addr, 0, addrlen, 0,
                           user_data);
  }
  // accept: addr -> sqe.addr (sockaddr*), addrlen -> sqe.addr2 (socklen_t*).
  // submit_one puts its `off` arg into sqe.off which unions with addr2, so pass
  // addrlen as the off argument.
  return submit_one(impl_->fd, impl_->sq_entries, &impl_->sq_pending, &impl_->inflight, impl_->sq_tail, *impl_->sq_ring_mask,
                    impl_->sq_array, impl_->sqes, IORING_OP_ACCEPT,
                    static_cast<std::int32_t>(fd),
                    reinterpret_cast<std::uint64_t>(addr), 0,
                    reinterpret_cast<std::uint64_t>(addrlen), user_data,
                    /*op_flags=*/SOCK_NONBLOCK | SOCK_CLOEXEC);
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
    impl_->pending_in.find(static_cast<int>(fd))->multishot = true;  // read-side
    return true;
  }
  // io_uring native: IORING_OP_ACCEPT with IORING_ACCEPT_MULTISHOT in ioprio.
  // addr/addrlen null -- multishot cannot fill a per-connection addr into one
  // fixed buffer (net/ uses getpeername on the accepted fd). One submit yields a
  // CQE per connection with IORING_CQE_F_MORE set until cancelled/errored.
  return submit_one(impl_->fd, impl_->sq_entries, &impl_->sq_pending, &impl_->inflight, impl_->sq_tail, *impl_->sq_ring_mask,
                    impl_->sq_array, impl_->sqes, IORING_OP_ACCEPT,
                    static_cast<std::int32_t>(fd), 0, 0, 0, user_data,
                    /*op_flags=*/SOCK_NONBLOCK | SOCK_CLOEXEC,
                    /*ioprio=*/IORING_ACCEPT_MULTISHOT);
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
                           IORING_OP_RECVMSG, buf, len, addr, addr_cap, nullptr,
                           0, user_data);
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
                           const_cast<void*>(addr), addrlen, nullptr, 0,
                           user_data);
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
  int f = O_CLOEXEC;
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
  if (impl_->backend == Impl::kEpoll) {
    struct stat st;
    int r = ::fstat(static_cast<int>(fd), &st);
    if (r == 0) {
      *out = FileStat{};
      out->size = static_cast<std::uint64_t>(st.st_size);
      out->mtime_ns =
          static_cast<std::uint64_t>(st.st_mtim.tv_sec) * 1'000'000'000ull +
          static_cast<std::uint64_t>(st.st_mtim.tv_nsec);
      out->mode = static_cast<std::uint32_t>(st.st_mode);
      out->is_dir = S_ISDIR(st.st_mode);
    }
    impl_->posted.push_back(
        {user_data, r < 0 ? -static_cast<std::int64_t>(errno) : 0});
    return true;
  }

  // The user_data identifies an in-flight operation everywhere in Proactor; a
  // duplicate would make cancellation/completion routing ambiguous.
  for (const Impl::PendingStatx* pending : impl_->pending_statx)
    if (pending->user_data == user_data) return false;
  auto* pending = new (std::nothrow) Impl::PendingStatx{};
  if (pending == nullptr) return false;
  pending->user_data = user_data;
  pending->output = out;
  impl_->pending_statx.push_back(pending);

  // liburing io_uring_prep_statx mapping: fd=dfd, addr=path, len=mask,
  // off=statx buffer and statx_flags=flags. AT_EMPTY_PATH queries `fd`.
  static constexpr char kEmptyPath[] = "";
  const bool submitted = submit_one(
      impl_->fd, impl_->sq_entries, &impl_->sq_pending, &impl_->inflight,
      impl_->sq_tail, *impl_->sq_ring_mask, impl_->sq_array, impl_->sqes,
      IORING_OP_STATX, static_cast<std::int32_t>(fd),
      reinterpret_cast<std::uint64_t>(kEmptyPath), STATX_BASIC_STATS,
      reinterpret_cast<std::uint64_t>(&pending->native), user_data,
      AT_EMPTY_PATH);
  if (!submitted) {
    impl_->pending_statx.pop_back();
    delete pending;
  }
  return submitted;
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
  const auto max_offset =
      static_cast<std::uint64_t>((std::numeric_limits<off_t>::max)());
  if (len > max_offset || offset > max_offset - len) {
    impl_->posted.push_back({user_data, -EOVERFLOW});
    return true;
  }
  if (impl_->backend == Impl::kEpoll) {
    if (mode != 0) {
      impl_->posted.push_back({user_data, -ENOTSUP});
      return true;
    }
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

// io-06 cancel (Linux). io_uring: an IORING_OP_ASYNC_CANCEL SQE keyed on the target
// op's user_data (cancel_flags 0 == match-on-user_data, the documented default).
// The cancelled op reaps a -ECANCELED CQE (or completes normally if it raced); the
// cancel op itself reaps its OWN CQE under cancel_user_data (0 ok / -ENOENT not
// found / -EALREADY in progress) -- the router must drop that token (like a wake),
// which the orphan layer (net_timeout.h) handles. Slot lifetime is the caller's
// responsibility (keep alive until the -ECANCELED lands). epoll: readiness model,
// no in-flight kernel op -- just drop the fd's armed watch + pending entry; no late
// completion, nothing to orphan.
bool Proactor::submit_cancel(std::uint64_t target_user_data,
                             std::uint64_t cancel_user_data) noexcept {
  if (impl_ == nullptr) return false;
  if (impl_->backend == Impl::kEpoll) {
    // Find the armed op carrying target_user_data in either direction, drop it, and
    // re-arm the fd for the OTHER side if it is still pending (full-duplex, L3) --
    // else DEL the fd. Then synthesize a -ECANCELED completion (see below).
    auto try_cancel = [&](auto& side) -> bool {
      int fd = -1;
      side.for_each([&](const int& candidate, const Impl::PendingOp& op) {
        if (fd < 0 && op.user_data == target_user_data) fd = candidate;
      });
      if (fd >= 0) {
        side.erase(fd);
        // Re-arm for whatever remains on this fd (the other direction), else DEL.
        unsigned mask = 0;
        if (impl_->pending_in.contains(fd)) mask |= EPOLLIN;
        if (impl_->pending_out.contains(fd)) mask |= EPOLLOUT;
        if (mask != 0) {
          epoll_event rev{};
          rev.events = mask | EPOLLONESHOT;
          rev.data.u64 = static_cast<std::uint64_t>(fd);
          if (::epoll_ctl(impl_->epfd, EPOLL_CTL_MOD, fd, &rev) < 0) {
            impl_->pending_in.erase(fd);
            impl_->pending_out.erase(fd);
            ::epoll_ctl(impl_->epfd, EPOLL_CTL_DEL, fd, nullptr);
          }
        } else {
          ::epoll_ctl(impl_->epfd, EPOLL_CTL_DEL, fd, nullptr);
        }
        // Synthesize the cancellation completion. Unlike io_uring/IOCP, epoll's
        // cancel produces NO late kernel CQE -- but the timed-op layer (net_timeout)
        // is backend-neutral and RELIES on a final completion arriving to drop the
        // op's kernel-ref (else the heap slot leaks). So post a -ECANCELED for the
        // target here: complete_timed sees timer_fired and releases the kernel ref,
        // exactly as the real -ECANCELED CQE would on io_uring. (-125 = ECANCELED.)
        impl_->posted.push_back({target_user_data, -125});
        return true;
      }
      return false;
    };
    if (try_cancel(impl_->pending_in)) return true;
    if (try_cancel(impl_->pending_out)) return true;
    return false;  // not armed (already fired) -> nothing to cancel
  }
  // io_uring ASYNC_CANCEL: addr = target user_data, cancel_flags in the rw_flags
  // union (0 = default user_data match). The cancel op is itself in-flight and
  // reaps under cancel_user_data.
  return submit_one(impl_->fd, impl_->sq_entries, &impl_->sq_pending,
                    &impl_->inflight, impl_->sq_tail, *impl_->sq_ring_mask,
                    impl_->sq_array, impl_->sqes, IORING_OP_ASYNC_CANCEL,
                    -1, target_user_data, 0, 0, cancel_user_data);
}


}  // namespace hyle::io

#elif defined(HYLE_OS_MACOS)

// kqueue backend (io-03 mac). kqueue is the SAME readiness-reactor model as Linux's
// epoll fallback -- this block is a MECHANICAL MIRROR of that epoll backend, with the
// epoll primitives swapped for their kqueue equivalents (verified 1:1 against the
// Linux #elif above). There is no io_uring analogue on macOS, so this is the ONLY
// backend here: every submit_* arms a one-shot readiness watch (or, for a regular
// file / metadata op, runs the syscall synchronously and synthesizes a completion),
// and reap() turns "ready" into a performed-inline Completion. io_uring-only
// facilities are not part of this backend or the portable Proactor surface; they
// are reachable only through a successfully acquired IoUringExtension.
//
// Key kqueue-vs-epoll semantic difference exploited below: epoll registers ONE fd
// with a UNION event mask (EPOLLIN|EPOLLOUT) and EPOLLONESHOT disarms the WHOLE fd,
// so epoll must re-arm the still-pending other side each reap. kqueue registers the
// two directions as INDEPENDENT filters (EVFILT_READ / EVFILT_WRITE); EV_ONESHOT
// auto-deletes ONLY the filter that fired, leaving the other side armed untouched.
// So the "re-arm the union" dance collapses: we re-arm ONLY on a spurious EAGAIN or
// a multishot-accept re-delivery, and cancel/DEL touches ONLY the one fired-or-armed
// filter. The cross-thread wake is EVFILT_USER (NOTE_TRIGGER), which needs no fd --
// it replaces epoll's eventfd entirely.
#  include <sys/event.h>
#  include <sys/types.h>
#  include <sys/socket.h>
#  include <sys/stat.h>
#  include <netinet/in.h>
#  include <unistd.h>
#  include <fcntl.h>

#  include <cerrno>
#  include <cstdint>
#  include <cstring>
#  include <ctime>
#  include <vector>

namespace hyle::io {
namespace {

// PendingOp.opcode reuses the io_uring op names as PLAIN TAGS telling perform()
// which syscall to run when the fd fires -- the same role they play in the epoll
// branch. <linux/io_uring.h> doesn't exist here, so define the small subset we
// actually branch on as distinct integer constants (values are arbitrary tags).
enum : std::uint8_t {
  IORING_OP_READ = 1,
  IORING_OP_WRITE = 2,
  IORING_OP_ACCEPT = 3,
  IORING_OP_CONNECT = 4,
  IORING_OP_RECVMSG = 5,
  IORING_OP_SENDMSG = 6,
  IORING_OP_RECV = 7,
  IORING_OP_SEND = 8,
};

// EVFILT_USER wake ident (io-04 park merge). A fixed, fd-uncollidable value: the
// wake is distinguished by filter==EVFILT_USER, but a distinct ident keeps it clear
// of the fd/EVFILT_READ-WRITE key space regardless. (kqueue keys on (ident,filter).)
constexpr std::uintptr_t kWakeIdent = static_cast<std::uintptr_t>(0xFFFFFFFFu);

// Map neutral OpenFlags onto POSIX O_* for open (mirror of the Linux helper).
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

// Lay a msghdr + one iovec into `scratch` for a datagram op. Identical to the Linux
// build_msghdr, but kqueue's reactor runs recvfrom/sendto INLINE (it doesn't hand a
// msghdr to a kernel op), so this is unused on macOS -- the neutral submit_recvfrom/
// sendto take `scratch` for signature parity and we simply don't touch it.

}  // namespace

struct Proactor::Impl {
  // kqueue is the sole backend on macOS (no io_uring). The kq descriptor + the two
  // per-direction pending maps + the software `posted` queue are the whole state --
  // a direct rename of the epoll fallback's (epfd, pending_in, pending_out, posted),
  // minus wake_efd (EVFILT_USER needs no fd).
  int kq = -1;
  // TWO pending ops per fd -- read-side (EVFILT_READ: READ/RECVMSG/ACCEPT) and
  // write-side (EVFILT_WRITE: WRITE/SENDMSG/CONNECT) -- so a full-duplex socket (one
  // fiber reading, another writing the SAME fd) never loses an op (red-team L3).
  // `opcode` reuses the IORING_OP_* tags so reap() knows which syscall to run.
  struct PendingOp {
    std::uint8_t opcode;   // IORING_OP_READ/WRITE/ACCEPT/CONNECT/RECVMSG/SENDMSG
    void* buf;
    std::uint32_t len;
    std::uint64_t user_data;
    // Datagram (RECVMSG/SENDMSG) only: peer address buffer. recvfrom fills it (up to
    // addr_cap); sendto reads it as the destination. Null/0 for stream ops.
    void* addr = nullptr;
    std::uint32_t addr_cap = 0;
    // One-shot accept's socklen_t*, or null; recv/send message flags.
    void* addrlen = nullptr;
    std::uint32_t flags = 0;
    // Multishot accept emulation (io-07): if set, reap re-arms this ACCEPT watch
    // after each delivery and stamps Completion::kMore. Only for IORING_OP_ACCEPT.
    bool multishot = false;
  };
  // Split by direction so read+write on ONE fd coexist as INDEPENDENT kqueue filters
  // (unlike epoll's single union registration -- see the block header note).
  container::SwissTable<int, PendingOp> pending_in{mem::system_allocator()};
  container::SwissTable<int, PendingOp> pending_out{mem::system_allocator()};
  // post() has no fd (a bare wake token) and there is no kqueue "inject completion",
  // so injected wakes + synchronously-synthesized file-op results queue here; reap
  // drains them first. std in the .cc impl is allowed (conventions §1: non-hot).
  struct Posted {
    std::uint64_t user_data;
    std::int64_t result;
  };
  std::vector<Posted> posted;
};

Proactor::~Proactor() {
  if (impl_ == nullptr) return;
  if (impl_->kq >= 0) ::close(impl_->kq);
  delete impl_;
  impl_ = nullptr;
}

// Stand up the kqueue backend (reactor). Named try_init_epoll to reuse the header's
// private-static declaration (it names the private Impl, which a free function can't).
// Returns nullptr on failure. Mirror of the Linux try_init_epoll.
Proactor::Impl* Proactor::try_init_epoll(unsigned entries) noexcept {
  int kq = ::kqueue();
  if (kq < 0) return nullptr;
  ::fcntl(kq, F_SETFD, FD_CLOEXEC);  // kqueue() has no CLOEXEC flag; set it explicitly
  auto* impl = new (std::nothrow) Proactor::Impl{};
  if (impl == nullptr) {
    ::close(kq);
    return nullptr;
  }
  impl->kq = kq;
  if (!impl->pending_in.reserve(entries) ||
      !impl->pending_out.reserve(entries)) {
    delete impl;
    ::close(kq);
    return nullptr;
  }
  // Cross-thread wake channel (park merge), kqueue flavour: a persistent EVFILT_USER
  // registered once. wake() fires it with NOTE_TRIGGER; the blocking kevent returns;
  // reap detects filter==EVFILT_USER and emits NO user completion (it only breaks the
  // block). EV_CLEAR auto-resets the trigger after delivery. No fd needed (this is why
  // macOS drops the eventfd the epoll fallback carries). Non-fatal on failure.
  struct kevent kev;
  EV_SET(&kev, kWakeIdent, EVFILT_USER, EV_ADD | EV_CLEAR, 0, 0, nullptr);
  ::kevent(kq, &kev, 1, nullptr, 0, nullptr);
  return impl;
}

bool Proactor::init(unsigned entries, bool /*defer_worker_thread*/) noexcept {
  if (impl_ != nullptr) return true;  // idempotent
  // No io_uring on macOS: kqueue is the only backend. (entries/defer are io_uring
  // ring-lifecycle knobs with no kqueue analogue -- ignored, like IOCP ignores them.)
  Impl* k = try_init_epoll(entries);
  if (k == nullptr) return false;
  impl_ = k;
  return true;
}

BackendKind Proactor::backend_kind() const noexcept {
  return impl_ != nullptr ? HYLE_IO_BACKEND_KQUEUE : HYLE_IO_BACKEND_NONE;
}
const char* Proactor::backend_name() const noexcept {
  return impl_ != nullptr ? "kqueue" : "none";
}
ProactorCapabilities Proactor::capabilities() const noexcept {
  std::uint64_t bits =
      impl_ != nullptr ? HYLE_IO_CAP_EMULATED_MULTISHOT_ACCEPT : 0;
#if defined(HYLE_IO_TLS)
  bits |= HYLE_IO_CAP_TLS;
#endif
#if defined(HYLE_IO_HTTP2)
  bits |= HYLE_IO_CAP_HTTP2;
#endif
  return {sizeof(ProactorCapabilities), backend_kind(), bits};
}
bool Proactor::has_async_file_data() const noexcept {
  return false;
}
bool Proactor::has_async_file_metadata() const noexcept {
  return false;
}

std::size_t Proactor::reap(Completion* out, std::size_t max,
                           std::uint64_t timeout_ns) noexcept {
  if (impl_ == nullptr || max == 0) return 0;

  // Reactor -> completion: drain injected post()s + synchronously-synthesized file-op
  // results first (they need no fd), then kevent() for ready fds and perform each armed
  // op inline, turning "ready" into a synthesized Completion (bytes, or negative errno).
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
  if (n > 0) return n;  // deliver wakes/synth promptly; caller loops for more

  // kevent timeout: nullptr timespec = block forever; {0,0} = poll and return.
  struct timespec ts;
  struct timespec* pts;
  if (timeout_ns == UINT64_MAX) {
    pts = nullptr;  // block
  } else {
    ts.tv_sec = static_cast<time_t>(timeout_ns / 1'000'000'000ull);
    ts.tv_nsec = static_cast<long>(timeout_ns % 1'000'000'000ull);
    pts = &ts;  // timeout_ns==0 -> {0,0} -> non-blocking poll
  }
  struct kevent evs[64];
  int want = static_cast<int>(max < 64 ? max : 64);
  int got = ::kevent(impl_->kq, nullptr, 0, evs, want, pts);
  if (got <= 0) return 0;  // timeout / EINTR -> nothing this round

  // Perform the ready op inline and synthesize its result. The op tag says which
  // syscall. Shared by both directions -- IDENTICAL to the epoll branch's perform.
  auto perform = [](int fd, const Impl::PendingOp& op) -> std::int64_t {
    if (op.opcode == IORING_OP_WRITE) {
      ssize_t w = ::write(fd, op.buf, op.len);
      return (w < 0) ? -errno : static_cast<std::int64_t>(w);
    } else if (op.opcode == IORING_OP_SEND) {
      ssize_t w = ::send(fd, op.buf, op.len, static_cast<int>(op.flags));
      return (w < 0) ? -errno : static_cast<std::int64_t>(w);
    } else if (op.opcode == IORING_OP_ACCEPT) {
      int as = ::accept(fd, static_cast<sockaddr*>(op.addr),
                        static_cast<socklen_t*>(op.addrlen));
      if (as >= 0) {
        const int flags = ::fcntl(as, F_GETFL, 0);
        if (flags < 0 || ::fcntl(as, F_SETFL, flags | O_NONBLOCK) != 0) {
          const int e = errno;
          ::close(as);
          return -e;
        }
      }
      return (as < 0) ? -errno : static_cast<std::int64_t>(as);
    } else if (op.opcode == IORING_OP_CONNECT) {
      int soerr = 0;
      socklen_t sl = sizeof(soerr);
      ::getsockopt(fd, SOL_SOCKET, SO_ERROR, &soerr, &sl);
      return (soerr == 0) ? 0 : -soerr;
    } else if (op.opcode == IORING_OP_RECVMSG) {
      socklen_t al = static_cast<socklen_t>(op.addr_cap);
      ssize_t r = ::recvfrom(fd, op.buf, op.len, 0,
                             static_cast<sockaddr*>(op.addr),
                             op.addr ? &al : nullptr);
      return (r < 0) ? -errno : static_cast<std::int64_t>(r);
    } else if (op.opcode == IORING_OP_SENDMSG) {
      ssize_t w = ::sendto(fd, op.buf, op.len, 0,
                           static_cast<const sockaddr*>(op.addr),
                           static_cast<socklen_t>(op.addr_cap));
      return (w < 0) ? -errno : static_cast<std::int64_t>(w);
    } else if (op.opcode == IORING_OP_RECV) {
      ssize_t r = ::recv(fd, op.buf, op.len, static_cast<int>(op.flags));
      return (r < 0) ? -errno : static_cast<std::int64_t>(r);
    } else {  // READ
      ssize_t r = ::read(fd, op.buf, op.len);
      return (r < 0) ? -errno : static_cast<std::int64_t>(r);
    }
  };

  // Re-arm ONE filter (EV_ONESHOT auto-deleted it when it fired). Used only for a
  // spurious-EAGAIN re-watch and a multishot-accept re-delivery -- never for the
  // "other side", which (unlike epoll) stays independently armed.
  auto rearm = [&](int fd, short filter) -> bool {
    struct kevent rk;
    EV_SET(&rk, static_cast<std::uintptr_t>(fd), filter, EV_ADD | EV_ONESHOT, 0, 0,
           reinterpret_cast<void*>(static_cast<std::intptr_t>(fd)));
    return ::kevent(impl_->kq, &rk, 1, nullptr, 0, nullptr) >= 0;
  };

  for (int i = 0; i < got && n < max; ++i) {
    if (evs[i].filter == EVFILT_USER) {
      // Cross-thread wake (park merge): emit NO completion -- it only breaks the
      // block. EV_CLEAR already reset the trigger; nothing to drain (vs epoll's
      // eventfd read). Mirrors the epoll wake_efd branch's "continue".
      continue;
    }
    int fd = static_cast<int>(evs[i].ident);  // we set ident = fd at arm time
    // EV_EOF (peer half-close/HUP) and EV_ERROR (errno in evs[i].data) are the kqueue
    // analogues of EPOLLHUP|EPOLLERR: a genuine terminal condition, so we do NOT treat
    // a resulting EAGAIN as spurious (avoid a re-arm spin). One kevent carries exactly
    // one filter, so -- unlike epoll's single-fd HUP that wakes both sides -- each
    // direction's EV_EOF arrives as its own event and serves its own side.
    const bool err = (evs[i].flags & (EV_EOF | EV_ERROR)) != 0;
    const bool is_read_ev =
        evs[i].filter == EVFILT_READ || evs[i].filter == EVFILT_EXCEPT;

    // A readiness can be SPURIOUS: kqueue says "readable" but by the time we run the
    // inline syscall the data/connection is gone -> -EAGAIN/-EWOULDBLOCK. A real
    // proactor never surfaces EAGAIN; so we re-arm the op and deliver NOTHING, waiting
    // for a TRUE readiness (L4). Guarded by !err: on a real EOF/ERROR that EAGAIN is
    // terminal, not spurious.
    auto is_spurious = [&](std::int64_t res) noexcept {
      return !err && (res == -EAGAIN || res == -EWOULDBLOCK);
    };

    if (is_read_ev) {
      // ---- read-side (recv/accept/recvfrom) ----
      if (Impl::PendingOp* pending = impl_->pending_in.find(fd)) {
        Impl::PendingOp op = *pending;
        impl_->pending_in.erase(fd);
        std::int64_t res = perform(fd, op);
        if (is_spurious(res)) {
          // False alarm: re-watch the read filter, deliver nothing. If the re-arm
          // fails (essentially never on a live kq+fd) the op is dropped, not leaked.
          const short filter =
              op.opcode == IORING_OP_RECV && (op.flags & MSG_OOB) != 0
                  ? EVFILT_EXCEPT
                  : EVFILT_READ;
          if (rearm(fd, filter)) impl_->pending_in.insert(fd, op);
        } else {
          out[n].user_data = op.user_data;
          out[n].result = res;
          out[n].flags = 0;
          out[n].buffer_id = 0;
          // Multishot accept emulation: re-arm + stamp kMore ONLY if BOTH the accept
          // succeeded and the re-arm succeeds (a failed re-arm is terminal -- stamping
          // kMore would hang the caller waiting for a stream that ended; matches IOCP).
          if (op.multishot && res >= 0 && rearm(fd, EVFILT_READ)) {
            impl_->pending_in.insert(fd, op);  // keep multishot for next delivery
            out[n].flags |= Completion::kMore;
          }
          ++n;
        }
      }
    } else {
      // ---- write-side (send/connect/sendto) ---- (EVFILT_WRITE; no multishot)
      if (Impl::PendingOp* pending = impl_->pending_out.find(fd)) {
        Impl::PendingOp op = *pending;
        impl_->pending_out.erase(fd);
        std::int64_t res = perform(fd, op);
        if (is_spurious(res)) {
          if (rearm(fd, EVFILT_WRITE)) impl_->pending_out.insert(fd, op);
        } else {
          out[n].user_data = op.user_data;
          out[n].result = res;
          out[n].flags = 0;
          out[n].buffer_id = 0;
          ++n;
        }
      }
    }
    // NOTE: no unified re-arm/DEL here (the epoll branch's tail). EV_ONESHOT already
    // auto-deleted the fired filter, and the OTHER direction's filter -- if armed --
    // is a SEPARATE kqueue registration still live and untouched. So there is nothing
    // to re-arm or DEL for a normal completion; the map erase above is the whole
    // teardown. This is the structural simplification kqueue's per-filter model buys.
  }
  return n;
}

bool Proactor::in_flight() const noexcept {
  if (impl_ == nullptr) return false;
  // Reactor: an op is "in flight" while its fd is armed (awaiting readiness) or a
  // synthetic completion is queued. Either means a reap could still yield one.
  return !impl_->pending_in.empty() || !impl_->pending_out.empty() ||
         !impl_->posted.empty();
}

bool Proactor::post(std::uint64_t user_data, std::int64_t result) noexcept {
  if (impl_ == nullptr) return false;
  // kqueue has no "inject completion": queue a software completion that reap() drains
  // first. (Single-threaded per-worker use; cross-thread post is not the model.)
  impl_->posted.push_back({user_data, result});
  return true;
}

bool Proactor::wake() noexcept {
  if (impl_ == nullptr) return false;
  // Fire the persistent EVFILT_USER with NOTE_TRIGGER from ANY thread. kevent() is a
  // syscall (kernel-serialized), so this never races the owner's arming -- the kqueue
  // analogue of epoll's atomic eventfd write. The blocking reap's kevent then returns;
  // reap drops the EVFILT_USER event (emits no completion). Returns false if the wake
  // filter was never registered (ENOENT) -- benign, matches epoll's failure return.
  struct kevent kev;
  EV_SET(&kev, kWakeIdent, EVFILT_USER, 0, NOTE_TRIGGER, 0, nullptr);
  return ::kevent(impl_->kq, &kev, 1, nullptr, 0, nullptr) >= 0;
}

// Arm a one-shot readiness watch for `fd` and remember the op; reap() does the actual
// read/write when the fd fires. `events` carries the kqueue filter (EVFILT_READ for a
// read-side op, EVFILT_WRITE for a write-side op). The datagram variant additionally
// stashes the peer-address buffer (recvfrom fills it / sendto reads it); the stream
// variant passes addr=null. Named epoll_arm_dgram to reuse the header's private-static
// declaration (it names the private Impl). Mirror of the Linux epoll_arm_dgram.
//
// FULL-DUPLEX (red-team L3): read-side ops go in pending_in, write-side in pending_out,
// so a concurrent read+write on one fd don't clobber each other. Unlike epoll (one fd,
// union mask) kqueue registers each direction as an INDEPENDENT filter, so there is no
// union to compute -- each side arms its own EV_ONESHOT filter.
bool Proactor::epoll_arm_dgram(Impl* impl, int fd, unsigned events,
                               unsigned char opcode, void* buf, unsigned len,
                               void* addr, unsigned addr_cap, void* addrlen,
                               unsigned flags,
                               std::uint64_t user_data,
                               std::uint64_t offset) noexcept {
  const bool is_read =
      events == static_cast<unsigned>(EVFILT_READ) ||
      events == static_cast<unsigned>(EVFILT_EXCEPT);
  // REGULAR-FILE sync fallback. epoll gets EPERM when arming a regular file and takes a
  // synchronous pread/pwrite+posted path; kqueue would instead register the vnode and
  // report it always-ready, then read()/write() would BLOCK the worker on disk. So
  // detect a regular file via fstat S_ISREG AT ARM TIME (only READ/WRITE can hit a
  // regular file -- sockets never do) and do the byte IO synchronously at the file
  // offset, synthesizing a completion via posted -- the identical outcome to epoll's
  // EPERM branch, just triggered by S_ISREG since kqueue never hands us EPERM.
  if (opcode == IORING_OP_READ || opcode == IORING_OP_WRITE) {
    struct stat st;
    if (::fstat(fd, &st) == 0 && S_ISREG(st.st_mode)) {
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
    }
  }
  auto& side = is_read ? impl->pending_in : impl->pending_out;
  if (side.contains(fd) || !side.try_prepare_insert()) return false;
  side.insert(fd, Impl::PendingOp{opcode, buf, len, user_data, addr, addr_cap,
                                  addrlen, flags});
  struct kevent kev;
  const short filter = static_cast<short>(events);
  EV_SET(&kev, static_cast<std::uintptr_t>(fd), filter, EV_ADD | EV_ONESHOT, 0, 0,
         reinterpret_cast<void*>(static_cast<std::intptr_t>(fd)));
  if (::kevent(impl->kq, &kev, 1, nullptr, 0, nullptr) < 0) {
    side.erase(fd);
    return false;
  }
  return true;
}

bool Proactor::epoll_arm(Impl* impl, int fd, unsigned events,
                         unsigned char opcode, void* buf, unsigned len,
                         std::uint64_t user_data, std::uint64_t offset) noexcept {
  return epoll_arm_dgram(impl, fd, events, opcode, buf, len, nullptr, 0,
                         nullptr, 0, user_data, offset);
}

bool Proactor::submit_read(std::intptr_t fd, void* buf, std::uint32_t len,
                           std::uint64_t offset,
                           std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  return epoll_arm(impl_, static_cast<int>(fd), EVFILT_READ, IORING_OP_READ, buf,
                   len, user_data, offset);
}

bool Proactor::submit_write(std::intptr_t fd, const void* buf, std::uint32_t len,
                            std::uint64_t offset,
                            std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  return epoll_arm(impl_, static_cast<int>(fd), EVFILT_WRITE, IORING_OP_WRITE,
                   const_cast<void*>(buf), len, user_data, offset);
}

bool Proactor::submit_recv(std::intptr_t fd, void* buf, std::uint32_t len,
                           std::uint32_t flags,
                           std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  const unsigned events =
      (flags & MSG_OOB) != 0 ? EVFILT_EXCEPT : EVFILT_READ;
  return epoll_arm_dgram(impl_, static_cast<int>(fd), events,
                         IORING_OP_RECV, buf, len, nullptr, 0, nullptr, flags,
                         user_data);
}

bool Proactor::submit_send(std::intptr_t fd, const void* buf, std::uint32_t len,
                           std::uint32_t flags,
                           std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  return epoll_arm_dgram(impl_, static_cast<int>(fd), EVFILT_WRITE,
                         IORING_OP_SEND, const_cast<void*>(buf), len, nullptr, 0,
                         nullptr, flags, user_data);
}

bool Proactor::submit_accept(std::intptr_t fd, void* addr, void* addrlen,
                             std::uint64_t user_data) noexcept {
  if (impl_ == nullptr || ((addr == nullptr) != (addrlen == nullptr)))
    return false;
  return epoll_arm_dgram(impl_, static_cast<int>(fd), EVFILT_READ,
                         IORING_OP_ACCEPT, nullptr, 0, addr, 0, addrlen, 0,
                         user_data);
}

bool Proactor::submit_accept_multishot(std::intptr_t fd,
                                       std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  // No native multishot: arm a one-shot accept watch, then flag it multishot so reap()
  // re-arms after each delivery and stamps kMore -- the caller sees the same continuous
  // kMore-set stream as io_uring. (Identical to the epoll emulation.)
  if (!epoll_arm(impl_, static_cast<int>(fd), EVFILT_READ, IORING_OP_ACCEPT, nullptr,
                 0, user_data))
    return false;
  impl_->pending_in.find(static_cast<int>(fd))->multishot = true;  // read-side
  return true;
}

bool Proactor::submit_connect(std::intptr_t fd, const void* addr,
                              std::uint32_t addrlen,
                              std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  // Reactor connect: kick off the non-blocking connect now. If it completes
  // synchronously (rare, e.g. UNIX socket) synthesize via post(); otherwise
  // EINPROGRESS -> watch EVFILT_WRITE; reap() reads SO_ERROR when writable.
  int r = ::connect(static_cast<int>(fd),
                    reinterpret_cast<const sockaddr*>(addr),
                    static_cast<socklen_t>(addrlen));
  if (r == 0) return post(user_data, 0);                     // connected immediately
  if (errno != EINPROGRESS) return post(user_data, -errno);  // hard failure
  return epoll_arm(impl_, static_cast<int>(fd), EVFILT_WRITE, IORING_OP_CONNECT,
                   nullptr, 0, user_data);
}

bool Proactor::submit_recvfrom(std::intptr_t fd, void* buf, std::uint32_t len,
                               void* addr, std::uint32_t addr_cap, void* /*scratch*/,
                               std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  // Reactor: arm EVFILT_READ; reap() runs ::recvfrom when readable, filling addr.
  return epoll_arm_dgram(impl_, static_cast<int>(fd), EVFILT_READ, IORING_OP_RECVMSG,
                         buf, len, addr, addr_cap, nullptr, 0, user_data);
}

bool Proactor::submit_sendto(std::intptr_t fd, const void* buf, std::uint32_t len,
                             const void* addr, std::uint32_t addrlen,
                             void* /*scratch*/, std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  // Reactor: arm EVFILT_WRITE; reap() runs ::sendto to `addr` when writable.
  return epoll_arm_dgram(impl_, static_cast<int>(fd), EVFILT_WRITE, IORING_OP_SENDMSG,
                         const_cast<void*>(buf), len, const_cast<void*>(addr),
                         addrlen, nullptr, 0, user_data);
}

bool Proactor::register_handle(std::intptr_t /*fd*/) noexcept {
  // kqueue adds the fd at submit time (EV_ADD), like epoll -- no pre-registration.
  return impl_ != nullptr;
}

bool Proactor::submit_open(const char* path, std::uint32_t flags,
                           std::uint32_t mode, std::uint64_t user_data) noexcept {
  if (impl_ == nullptr || path == nullptr) return false;
  // No async open on kqueue -> open synchronously, synthesize a completion.
  int oflags = open_flags_to_posix(flags);
  int fd = ::open(path, oflags, static_cast<mode_t>(mode));
  impl_->posted.push_back(
      {user_data, fd < 0 ? -static_cast<std::int64_t>(errno)
                         : static_cast<std::int64_t>(fd)});
  return true;
}

bool Proactor::submit_close(std::intptr_t fd, std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  int r = ::close(static_cast<int>(fd));
  impl_->posted.push_back(
      {user_data, r < 0 ? -static_cast<std::int64_t>(errno) : 0});
  return true;
}

bool Proactor::submit_statx(std::intptr_t fd, FileStat* out,
                            std::uint64_t user_data) noexcept {
  if (impl_ == nullptr || out == nullptr) return false;
  // Pure metadata query (no disk-io wait): run fstat synchronously and synthesize a
  // completion. macOS `struct stat` uses st_mtimespec (BSD), not Linux's st_mtim.
  struct stat st;
  int r = ::fstat(static_cast<int>(fd), &st);
  if (r == 0) {
    *out = FileStat{};
    out->size = static_cast<std::uint64_t>(st.st_size);
    out->mtime_ns = static_cast<std::uint64_t>(st.st_mtimespec.tv_sec) * 1'000'000'000ull +
                    static_cast<std::uint64_t>(st.st_mtimespec.tv_nsec);
    out->mode = static_cast<std::uint32_t>(st.st_mode);
    out->is_dir = S_ISDIR(st.st_mode);
  }
  impl_->posted.push_back(
      {user_data, r < 0 ? -static_cast<std::int64_t>(errno) : 0});
  return true;
}

bool Proactor::submit_fsync(std::intptr_t fd, bool /*datasync*/,
                            std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  // macOS has no fdatasync (Linux-only): ::fsync is the correct superset for both the
  // datasync and full-sync request, so ignore `datasync` and always fsync. Synchronous
  // + synthesized (no async form). (F_FULLFSYNC would force a device-cache flush; plain
  // fsync matches the neutral contract "flush data to disk" without that extra cost.)
  int r = ::fsync(static_cast<int>(fd));
  impl_->posted.push_back(
      {user_data, r < 0 ? -static_cast<std::int64_t>(errno) : 0});
  return true;
}

bool Proactor::submit_fallocate(std::intptr_t fd, std::uint32_t mode,
                                std::uint64_t offset, std::uint64_t len,
                                std::uint64_t user_data) noexcept {
  if (impl_ == nullptr) return false;
  if (mode != 0) {
    impl_->posted.push_back({user_data, -ENOTSUP});
    return true;
  }
  const auto max_offset =
      static_cast<std::uint64_t>((std::numeric_limits<off_t>::max)());
  if (len > max_offset || offset > max_offset - len) {
    impl_->posted.push_back({user_data, -EOVERFLOW});
    return true;
  }
  // macOS has no posix_fallocate (Linux/glibc): the documented equivalent is
  // fcntl(F_PREALLOCATE) to reserve space (contiguous, falling back to any), then
  // ftruncate to actually grow the file to offset+len. Synchronous + synthesized,
  // mirroring epoll's posix_fallocate path. `mode` (Linux fallocate mode) is ignored,
  // as it is on the epoll path (posix_fallocate takes no mode either).
  off_t want = static_cast<off_t>(offset + len);
  fstore_t fst;
  fst.fst_flags = F_ALLOCATECONTIG;
  fst.fst_posmode = F_PEOFPOSMODE;
  fst.fst_offset = 0;
  fst.fst_length = want;
  fst.fst_bytesalloc = 0;
  int r = ::fcntl(static_cast<int>(fd), F_PREALLOCATE, &fst);
  if (r < 0) {
    fst.fst_flags = F_ALLOCATEALL;  // retry allowing a non-contiguous allocation
    r = ::fcntl(static_cast<int>(fd), F_PREALLOCATE, &fst);
  }
  std::int64_t res;
  if (r < 0) {
    res = -errno;
  } else {
    res = (::ftruncate(static_cast<int>(fd), want) < 0) ? -errno : 0;
  }
  impl_->posted.push_back({user_data, res});
  return true;
}

// io-06 cancel (kqueue): readiness model -- no in-flight kernel op. Find the armed op
// carrying target_user_data in either direction, drop it, EV_DELETE its still-armed
// filter (the other direction is an independent filter -- left untouched, unlike
// epoll's re-arm-the-union), then synthesize a -ECANCELED completion. The timed-op
// layer (net_timeout) is backend-neutral and RELIES on a final completion arriving to
// drop the op's kernel-ref, so we post -125 (ECANCELED) for the target, exactly as the
// real -ECANCELED CQE would on io_uring.
bool Proactor::submit_cancel(std::uint64_t target_user_data,
                             std::uint64_t /*cancel_user_data*/) noexcept {
  if (impl_ == nullptr) return false;
  auto try_cancel = [&](auto& side, short filter) -> bool {
    int fd = -1;
    side.for_each([&](const int& candidate, const Impl::PendingOp& op) {
      if (fd < 0 && op.user_data == target_user_data) fd = candidate;
    });
    if (fd >= 0) {
      side.erase(fd);
      // Delete ONLY this direction's still-armed filter (EV_ONESHOT hasn't fired, so
      // it is registered). The other direction, if pending, stays independently armed.
      struct kevent dk;
      EV_SET(&dk, static_cast<std::uintptr_t>(fd), filter, EV_DELETE, 0, 0, nullptr);
      ::kevent(impl_->kq, &dk, 1, nullptr, 0, nullptr);
      impl_->posted.push_back({target_user_data, -125});  // -125 = ECANCELED
      return true;
    }
    return false;
  };
  if (try_cancel(impl_->pending_in, EVFILT_READ)) return true;
  if (try_cancel(impl_->pending_out, EVFILT_WRITE)) return true;
  return false;  // not armed (already fired) -> nothing to cancel
}

}  // namespace hyle::io

#else
#  error "HYLE io/proactor: unsupported OS"
#endif
