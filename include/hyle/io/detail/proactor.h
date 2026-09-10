#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/core/compiler.h"
#include "hyle/core/environment.h"
#include "hyle/io/buffer.h"
#include "hyle/io/capabilities.h"
#include "hyle/time/unix_timestamp.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_IO_API
#elif defined(HYLE_IO_CORE_BUILD)
#  define HYLE_IO_API HYLE_EXPORT
#else
#  define HYLE_IO_API HYLE_IMPORT
#endif

// Proactor: the platform-neutral COMPLETION-reaping seam (io-design "模块形状",
// sched-design §3.2/§3.3). One proactor per worker. It is NOT the socket/file
// API (that is io-04 net/); it is the thin layer that (a) lets a backend record
// "operation X, when it completes, should wake THIS token" and (b) lets the
// worker loop reap finished operations and hand their tokens back so the caller
// wakes the fibers.
//
// The completion->fiber mapping is io-design §3.3's physical mechanism, verified
// on this machine before this interface was written: every backend carries a
// 64-bit user_data with the submitted op (io_uring SQE user_data / IOCP
// completion key+OVERLAPPED), and returns it verbatim on completion. We store a
// `Completion` token there. sched/ owns the Waker; io/ only moves tokens.
//
// Backends (io-02+): io_uring (Linux, per-core ring), IOCP (Windows, per-worker
// port), epoll/kqueue (reactor -> completion emulation). This header is what
// they all present upward; the consumer (a sched worker) is backend-agnostic.

namespace hyle::io::detail {

using BackendKind = IoBackend;
using ProactorCapability = IoCapability;
using ProactorCapabilities = IoCapabilities;

struct FileStat {
  std::uint64_t size = 0;
  time::UnixTimestamp modified_time{};
  std::uint32_t mode = 0;
  bool is_directory = false;
};

// A finished operation handed back by the proactor. `user_data` is the exact
// 64-bit value the submitter attached (typically a Waker pointer / id, cast to
// u64); `result` is the op's outcome (bytes transferred, or a negative errno on
// failure -- the same convention io_uring CQE `res` uses, IOCP maps onto it).
//
// `flags` mirrors io_uring's `cqe->flags` (IOCP synthesizes it). It is NOT
// optional dressing -- the io_uring SOTA mechanisms REQUIRE it (verified against
// liburing v2.15):
//   - kMore: this op will produce MORE completions (multishot accept/recv set
//     IORING_CQE_F_MORE). One submit -> many completions sharing one user_data;
//     the caller must NOT recycle the token/Waker until a completion arrives
//     with kMore CLEAR. A "1 submit = 1 completion" abstraction would corrupt
//     multishot -- so the flag lives here from day one.
//   - kBuffer + buffer_id: the kernel picked a provided/ring buffer (zero-copy
//     recv); IORING_CQE_F_BUFFER, id in the high bits. Surfaced so net/ can
//     recover which buffer without a side table.
// Backends that lack a notion (IOCP) just leave flags 0 / buffer_id 0.
// user_data must identify one concurrently in-flight operation; multishot is the
// deliberate exception where one operation emits multiple CQEs with its token.
struct Completion {
  std::uint64_t user_data = 0;
  std::int64_t result = 0;
  std::uint32_t flags = 0;      // backend completion flags (see kMore/kBuffer)
  std::uint32_t buffer_id = 0;  // provided-buffer id when (flags & kBuffer)

  enum : std::uint32_t {
    kMore = 1u << 0,    // more completions coming for this user_data (multishot)
    kBuffer = 1u << 1,  // buffer_id is valid (kernel-selected zero-copy buffer)
    kDatagramTruncated =
        1u << 2, // receive copied a prefix and discarded the datagram tail
  };
};

// Per-worker proactor. Not thread-safe across workers by design: each worker
// owns one (its own IOCP port / io_uring ring), submits and reaps on its own
// thread -- the per-core philosophy that matches the run-queue (sched-design
// §3.2). Cross-thread submission (a foreign thread arming io) is a backend
// concern handled where it arises, not by locking this object.
//
// API macro on the methods (not the class): backends hold OS handles / std
// members; method-level export keeps the layout backend-side and the seam clean
// (same pattern as Parker/Scheduler).
class Proactor {
 public:
  Proactor() noexcept = default;
  Proactor(const Proactor&) = delete;
  Proactor& operator=(const Proactor&) = delete;
  Proactor(Proactor&&) = delete;
  Proactor& operator=(Proactor&&) = delete;
  HYLE_IO_API ~Proactor();

  // Create the backend's completion object (IOCP port / io_uring ring). Returns
  // false if the OS refused (out of fds, unsupported kernel). Call once before use.
  //
  // `defer_worker_thread`: set true when this ring is created on one thread but will
  // be submitted+reaped on a DIFFERENT owner-worker thread (the IoContext shard-per-
  // core case). It opts into io_uring's DEFER_TASKRUN+SINGLE_ISSUER (the best setup
  // for one-consumer-per-ring) via the R_DISABLED lifecycle: the ring starts disabled
  // and the FIRST reap (on the owner worker) enables it, making that worker the single
  // issuer (else the creating thread would be the issuer and the worker's enter fails
  // EEXIST). Left false (default) for a bare same-thread Proactor -- create+reap on one
  // thread needs no deferral and keeps the plain setup. io_uring only; no-op elsewhere.
  HYLE_IO_API bool init(unsigned entries = 256,
                        bool defer_worker_thread = false) noexcept;
  HYLE_IO_API BackendKind backend_kind() const noexcept;
  HYLE_IO_API const char* backend_name() const noexcept;
  // Bits describe the selected backend/extension surface, not a guarantee that
  // the running kernel will accept every optional registration or operation.
  HYLE_IO_API ProactorCapabilities capabilities() const noexcept;
  HYLE_IO_API bool has_async_file_data() const noexcept;
  HYLE_IO_API bool has_async_file_metadata() const noexcept;
  // ---- submit real IO ops (io-02) -------------------------------------------
  // Queue a read/write of `len` bytes on `fd` at file `offset` (offset ignored
  // for streams; use 0 or the pipe/socket convention). `buf` is caller-owned and
  // MUST stay valid until the matching completion arrives -- the kernel writes
  // into it directly (io_uring true zero-copy; IOCP OVERLAPPED read). `user_data`
  // is the token returned verbatim on completion (a Waker). Returns false only if
  // the op could not be queued (ring full / bad handle); a false here is a submit
  // failure, whereas an IO error (EPIPE, short read) comes back as a NEGATIVE
  // Completion.result at reap time -- the io_uring CQE res convention, IOCP maps
  // GetLastError onto it. On Linux this fills an IORING_OP_READ/WRITE SQE; on
  // Windows a ReadFile/WriteFile with an OVERLAPPED bound to the port.
  //
  // fd is intptr_t so the same signature carries a POSIX int fd AND a Windows
  // HANDLE (which is a pointer-width handle) without a per-OS type in the header.
  HYLE_IO_API bool submit_read(std::intptr_t fd, void* buf, std::uint32_t len,
                               std::uint64_t offset,
                               std::uint64_t user_data) noexcept;
  HYLE_IO_API bool submit_write(std::intptr_t fd, const void* buf,
                                std::uint32_t len, std::uint64_t offset,
                                std::uint64_t user_data) noexcept;
  // Vectored positional IO. `buffers` is an awaitable-owned native descriptor
  // array and must remain alive with the referenced bytes until completion.
  // One call maps to one native vectored submission on POSIX. Windows regular
  // files have no general arbitrary-buffer vectored primitive; the typed File
  // facade uses one staged ReadFile/WriteFile submission there instead.
  HYLE_IO_API bool submit_read_buffers(
      std::intptr_t fd, detail::NativeBuffer* buffers,
      std::uint32_t count, std::uint64_t offset,
      std::uint64_t user_data) noexcept;
  HYLE_IO_API bool submit_write_buffers(
      std::intptr_t fd, detail::NativeBuffer* buffers,
      std::uint32_t count, std::uint64_t offset,
      std::uint64_t user_data) noexcept;

  // ---- socket ops (io-04 net) -----------------------------------------------
  // recv/send are read/write-shaped for a socket fd (no file offset; `flags` is
  // the recv/send msg flags, usually 0). `buf` is caller-owned and must outlive
  // the completion (cancel-safe: valid until the CQE arrives, even on cancel).
  HYLE_IO_API bool submit_recv(std::intptr_t fd, void* buf, std::uint32_t len,
                               std::uint32_t flags,
                               std::uint64_t user_data) noexcept;
  HYLE_IO_API bool submit_send(std::intptr_t fd, const void* buf,
                               std::uint32_t len, std::uint32_t flags,
                               std::uint64_t user_data) noexcept;
  // Stream scatter/gather. The typed facade fixes flags at zero so every
  // backend can use its true readv/writev or WSARecv/WSASend operation.
  HYLE_IO_API bool submit_recv_buffers(
      std::intptr_t fd, detail::NativeBuffer* buffers,
      std::uint32_t count, std::uint64_t user_data) noexcept;
  HYLE_IO_API bool submit_send_buffers(
      std::intptr_t fd, detail::NativeBuffer* buffers,
      std::uint32_t count, void* message_scratch,
      std::uint64_t user_data) noexcept;
  // accept a connection on listening `fd`. `addr`/`addrlen` receive the peer
  // address (may be null to ignore); both must outlive the completion. The
  // completion result is the accepted fd (>=0) or a negative errno.
  HYLE_IO_API bool submit_accept(std::intptr_t fd, void* addr, void* addrlen,
                                 std::uint64_t user_data) noexcept;

  // Multishot accept (io-07): ONE submit that keeps producing a completion per
  // incoming connection until it is cancelled or errors -- no re-arm per accept.
  // Each completion carries the accepted fd (>=0) in result and `user_data`
  // VERBATIM (the same token every time), with Completion::kMore SET while the
  // accept remains armed; a completion with kMore CLEAR is the terminal one (the
  // multishot ended -- cancelled, or a fatal error in result). The caller must
  // therefore NOT recycle `user_data`'s waker/slot until it sees kMore clear, and
  // must be ready for many completions from one submit. Backends without a native
  // multishot (IOCP, epoll fallback) emulate it by re-arming a one-shot accept
  // after each delivery, presenting the SAME kMore-set stream upward -- so net/
  // sees one uniform contract. On io_uring this sets IORING_ACCEPT_MULTISHOT.
  // Returns false if the op could not be queued.
  HYLE_IO_API bool submit_accept_multishot(std::intptr_t fd,
                                           std::uint64_t user_data) noexcept;
  // connect `fd` to `addr` (sockaddr*, `addrlen` bytes). Completion result is 0
  // on success or a negative errno.
  HYLE_IO_API bool submit_connect(std::intptr_t fd, const void* addr,
                                  std::uint32_t addrlen,
                                  std::uint64_t user_data) noexcept;
#if HYLE_OS_WINDOWS
  // Internal named-pipe listener primitive. The handle must be an overlapped
  // CreateNamedPipe instance already associated with this proactor's IOCP.
  HYLE_IO_API bool submit_pipe_accept(
      std::intptr_t pipe, std::uint64_t user_data) noexcept;

  // Internal directory-change primitive for io_watch. `directory`
  // must be a FILE_LIST_DIRECTORY handle opened with FILE_FLAG_OVERLAPPED and
  // already associated with this proactor. The caller-owned buffer remains
  // borrowed until the terminal completion and receives FILE_NOTIFY_INFORMATION
  // records. The fixed filter covers entry names, metadata, size and writes;
  // `watch_subtree` maps directly to ReadDirectoryChangesW's bWatchSubtree;
  // portable event normalization and re-arm policy belong to io_watch.
  HYLE_IO_API bool submit_directory_changes(
      std::intptr_t directory, void* buffer, std::uint32_t length,
      bool watch_subtree, std::uint64_t user_data) noexcept;
#endif

  // ---- datagram ops (UDP) ---------------------------------------------------
  // Scratch bytes the caller MUST provide for the datagram ops below. On io_uring
  // recvfrom/sendto go through RECVMSG/SENDMSG, which need a `struct msghdr` +
  // `struct iovec` that OUTLIVE the submit (the kernel reads them when the op
  // completes, not at submit time). The backend builds those in `scratch`; it must
  // therefore live until the completion arrives -- put it on the awaiting fiber's
  // frozen stack (like the RecvOp/SendOp slots). IOCP (WSARecvFrom/WSASendTo) and
  // epoll (inline recvfrom/sendto) don't use it, but the neutral signature takes it
  // so net/ allocates it uniformly. 128 bytes covers msghdr(56)+iovec(16) with
  // room and 16-byte alignment headroom.
  static constexpr std::size_t kDatagramScratch = 128;

  // recvfrom: receive one datagram on `fd` into `buf` (len bytes); the sender's
  // address is written into `addr` (a sockaddr buffer of `addr_cap` bytes, e.g.
  // sockaddr_storage). Both `buf` and `addr` are caller-owned and must outlive the
  // completion; `scratch` (kDatagramScratch bytes) likewise. Result = bytes
  // received (>=0) or negative errno. The peer address length is not returned
  // separately -- read it from addr->sa_family (AF_INET => 16, AF_INET6 => 28).
  HYLE_IO_API bool submit_recvfrom(std::intptr_t fd, void* buf, std::uint32_t len,
                                   void* addr, std::uint32_t addr_cap,
                                   void* scratch,
                                   std::uint64_t user_data) noexcept;
  // sendto: send `buf` (len bytes) on `fd` to `addr` (sockaddr*, `addrlen` bytes).
  // `buf`, `addr`, and `scratch` (kDatagramScratch bytes) must outlive the
  // completion. Result = bytes sent (>=0) or negative errno.
  HYLE_IO_API bool submit_sendto(std::intptr_t fd, const void* buf,
                                 std::uint32_t len, const void* addr,
                                 std::uint32_t addrlen, void* scratch,
                                 std::uint64_t user_data) noexcept;
  HYLE_IO_API bool submit_recvfrom_buffers(
      std::intptr_t fd, detail::NativeBuffer* buffers,
      std::uint32_t count, void* addr, std::uint32_t addr_cap,
      void* scratch, std::uint64_t user_data) noexcept;
  HYLE_IO_API bool submit_sendto_buffers(
      std::intptr_t fd, detail::NativeBuffer* buffers,
      std::uint32_t count, const void* addr, std::uint32_t addrlen,
      void* scratch, std::uint64_t user_data) noexcept;

  // Windows only: an OVERLAPPED handle must be associated with the port ONCE
  // before submit_read/write on it (io_uring needs no such step). No-op / returns
  // true on Linux. Kept in the neutral interface so net/ (io-04) can register a
  // socket the same way on both platforms.
  HYLE_IO_API bool register_handle(std::intptr_t fd) noexcept;

  // ---- file ops (io-05, capability C5) --------------------------------------
  // Neutral, backend-filled file metadata (the statx subset a file server needs).
  // Bytes/times are the cross-platform intersection; `mode` is the POSIX st_mode
  // (0 on Windows -- use is_directory). One struct both backends fill on a stat op.
  using FileStat = hyle::io::detail::FileStat;

  // Neutral open flags (mapped to O_* on POSIX, to CreateFileW's disposition/access
  // on Windows). A file server needs read, write, create, truncate, append. `mode`
  // (POSIX permission bits for a freshly created file) is ignored on Windows.
  enum OpenFlags : std::uint32_t {
    kRead = 1u << 0,      // open for reading
    kWrite = 1u << 1,     // open for writing
    kCreate = 1u << 2,    // create if absent
    kTruncate = 1u << 3,  // truncate to empty on open
    kAppend = 1u << 4,    // writes append at end
  };

  // Open `path` with neutral `flags` (OpenFlags bitmask). The completion result is
  // the opened fd/handle as an intptr_t (>= 0) or a negative errno. `path` MUST
  // outlive the completion (the kernel reads it when the op runs, not at submit --
  // io_uring openat contract). On io_uring this is a real async IORING_OP_OPENAT;
  // on IOCP/epoll it is done synchronously (open has no async form on either) and a
  // completion is synthesized -- the caller sees one uniform completion either way.
  HYLE_IO_API bool submit_open(const char* path, std::uint32_t flags,
                               std::uint32_t mode,
                               std::uint64_t user_data) noexcept;
  // Close `fd`. Completion result 0 or negative errno. Real async on io_uring
  // (IORING_OP_CLOSE); synchronous + synthesized on IOCP/epoll.
  HYLE_IO_API bool submit_close(std::intptr_t fd,
                                std::uint64_t user_data) noexcept;
  // Stat `fd` into `*out` (caller-owned; MUST outlive the completion). Result 0 or
  // negative errno. io_uring issues a real IORING_OP_STATX and converts its native
  // result to this POD before delivery. IOCP/epoll/kqueue synthesize a completion
  // from a synchronous metadata query; the ordinary File facade offloads those
  // paths so a scheduler worker is not blocked. Raw Proactor users can inspect
  // has_async_file_metadata() or the capability bit.
  HYLE_IO_API bool submit_statx(std::intptr_t fd, FileStat* out,
                                std::uint64_t user_data) noexcept;
  // Flush `fd`'s data to disk. `datasync` = fdatasync (data only, skip metadata) vs
  // fsync (data + metadata). Result 0 or negative errno. io_uring IORING_OP_FSYNC
  // (real async); IOCP FlushFileBuffers / epoll fsync, synchronous + synthesized.
  HYLE_IO_API bool submit_fsync(std::intptr_t fd, bool datasync,
                                std::uint64_t user_data) noexcept;
  // Preallocate/extend `fd`: ensure `len` bytes are allocated starting at `offset`
  // (`mode` = POSIX fallocate mode, usually 0). Result 0 or negative errno.
  // io_uring IORING_OP_FALLOCATE (real async); on IOCP the SetFileValidData/
  // SetEndOfFile equivalent, on epoll posix_fallocate, synchronous + synthesized.
  HYLE_IO_API bool submit_fallocate(std::intptr_t fd, std::uint32_t mode,
                                    std::uint64_t offset, std::uint64_t len,
                                    std::uint64_t user_data) noexcept;

  // ---- cancel an in-flight op (io-06, capability C2) ------------------------
  // Cancel the in-flight op whose completion token is `target_user_data` (the same
  // value passed as an op's user_data). Used by timed io when the deadline fires
  // first: the timer won, so the in-flight recv/accept/connect must be torn down.
  //
  // Cancellation is ASYNCHRONOUS and DOES NOT free anything here. The cancelled op
  // still delivers a FINAL completion (io_uring: a -ECANCELED CQE; IOCP: an
  // ERROR_OPERATION_ABORTED completion; the cancel may also race the op completing
  // normally -> a normal result). The caller MUST keep the op's slot alive until
  // that final completion arrives (see net_timeout.h's orphan table) -- a slot on a
  // fiber stack that resumed on timeout would be a use-after-free when the late
  // completion lands. `cancel_user_data` is the token for the CANCEL op's OWN
  // completion (io_uring emits one; the router must drop it, like kWakeToken).
  //
  // Backends: io_uring fills an IORING_OP_ASYNC_CANCEL SQE (addr = target_user_data,
  // IORING_ASYNC_CANCEL_USERDATA); IOCP finds the in-flight IoOp by token and calls
  // CancelIoEx(handle, &overlapped); epoll drops the fd's armed watch + pending
  // entry (readiness model -- no late kernel completion, so nothing to orphan).
  // Returns false if the op could not be found / the cancel could not be queued
  // (e.g. it already completed) -- a false is benign (nothing to cancel).
  HYLE_IO_API bool submit_cancel(std::uint64_t target_user_data,
                                 std::uint64_t cancel_user_data) noexcept;

  // Reap up to `max` finished operations into `out` (caller-provided array),
  // returning how many were written. `timeout_ns` bounds the wait: 0 = poll and
  // return immediately (the worker's non-blocking loop step), UINT64_MAX = block
  // until at least one completes. This is GetQueuedCompletionStatusEx / io_uring
  // cq peek behind one signature. The caller then wakes each completion's fiber.
  HYLE_IO_API std::size_t reap(Completion* out, std::size_t max,
                               std::uint64_t timeout_ns) noexcept;

  // Manually inject a completion carrying `user_data` (result defaults to 0) --
  // used to drive tests without a real fd, and (io-04) to synthesize a completion
  // for an op that finished synchronously. SAME-THREAD on io_uring: it fills a
  // NOP SQE, and the submission queue is single-producer, so post must run on the
  // ring's owner thread (a foreign-thread post races the owner). IOCP's
  // PostQueuedCompletionStatus is thread-safe, but treat post as same-thread for
  // portability. (Cross-thread wake of a blocked reap is wake(), below.)
  HYLE_IO_API bool post(std::uint64_t user_data, std::int64_t result = 0) noexcept;

  // Cross-thread wake: unblock a reap() that is blocking on THIS proactor from ANY
  // thread, WITHOUT racing the (single-producer) submission queue. This is the
  // scheduler's park-merge primitive -- an idle worker with io in flight blocks IN
  // reap (a GETEVENTS io_uring_enter / GetQueuedCompletionStatusEx), and a foreign
  // thread that hands it a fiber (spawn/steal) or asks it to stop calls wake() to
  // break that block promptly instead of waiting out a poll timeout. Mechanism per
  // backend: io_uring writes a dedicated eventfd (a persistent IORING_OP_READ on it
  // completes with the reserved kWakeToken; reap drains + re-arms it, and never
  // hands it out as a user completion); IOCP posts kWakeToken via the thread-safe
  // PostQueuedCompletionStatus; epoll writes an eventfd in its watch set. The woken
  // reap returns (perhaps with zero user completions); the caller re-checks its
  // queues. Idempotent and cheap; safe to call when nothing is blocked.
  HYLE_IO_API bool wake() noexcept;

  // Reserved user_data/token for wake() completions. Never a valid IoOperation*
  // ( ~0 is kernel-space, never a heap object ), so the completion router can tell
  // a wake apart from a real op and drop it. Consumers (io_context routing,
  // net::complete) MUST skip a completion carrying this token.
  static constexpr std::uint64_t kWakeToken = ~std::uint64_t{0};

  // Whether the ring has ops SUBMITTED-but-not-yet-COMPLETED (the kernel still
  // owes ≥1 completion). The scheduler integration reads this at a worker's park
  // boundary to decide it MUST block in a GETEVENTS reap (io_uring surfaces those
  // completions only on a kernel enter) rather than the futex parker -- else the
  // completions never arrive and their fibers hang. False when nothing is
  // outstanding, so the worker can park for an immediately-interruptible wake.
  HYLE_IO_API bool in_flight() const noexcept;

  bool valid() const noexcept { return impl_ != nullptr; }

 private:
  // Linux-only primitives stay behind the typed public extension and internal
  // mechanism tests.  They never form a portable raw-submission API.
#if HYLE_OS_LINUX
 public:
  HYLE_IO_API bool register_buf_ring(int bgid, unsigned nentries,
                                     unsigned buf_size) noexcept;
  HYLE_IO_API bool register_file_table(unsigned nr) noexcept;
  HYLE_IO_API int assign_fixed_slot(std::intptr_t fd) noexcept;
  HYLE_IO_API bool release_fixed_slot(int index) noexcept;
  HYLE_IO_API bool submit_recv_fixed(int slot, void* buf, std::uint32_t len,
                                     std::uint32_t flags,
                                     std::uint64_t user_data) noexcept;
  HYLE_IO_API bool submit_send_fixed(int slot, const void* buf, std::uint32_t len,
                                     std::uint32_t flags,
                                     std::uint64_t user_data) noexcept;
  HYLE_IO_API bool submit_accept_direct(std::intptr_t fd,
                                        std::uint64_t user_data) noexcept;
  HYLE_IO_API bool submit_close_direct(int slot,
                                       std::uint64_t user_data) noexcept;
  HYLE_IO_API bool submit_recv_multishot(std::intptr_t fd, int bgid,
                                         std::uint64_t user_data) noexcept;
  HYLE_IO_API void* buf_ring_addr(int bgid,
                                  std::uint32_t buffer_id) noexcept;
  HYLE_IO_API bool buf_ring_return(int bgid,
                                   std::uint32_t buffer_id) noexcept;
#endif

 private:
  struct Impl;
  // Backend init + epoll-arm helpers need to touch the private Impl, so they are
  // private statics (defined per-OS in proactor.cc) rather than free functions
  // in the .cc anonymous namespace (which cannot name a private nested type --
  // GCC rightly rejects that; only some compilers let it slide).
  static Impl* try_init_uring(unsigned entries, bool defer_worker_thread) noexcept;
  static Impl* try_init_epoll(unsigned entries) noexcept;
  static bool epoll_arm(Impl* impl, int fd, unsigned events, unsigned char opcode,
                        void* buf, unsigned len, std::uint64_t user_data,
                        std::uint64_t offset = 0) noexcept;
  // Datagram variant: also stashes the peer-address buffer so reap()'s inline
  // recvfrom/sendto can fill/read it (epoll fallback only). `offset` is the file
  // offset for the regular-file synchronous-fallback path (EPERM); 0 for sockets.
  static bool epoll_arm_dgram(Impl* impl, int fd, unsigned events,
                              unsigned char opcode, void* buf, unsigned len,
                              void* addr, unsigned addr_cap, void* addrlen,
                              unsigned flags,
                              std::uint64_t user_data,
                              std::uint64_t offset = 0) noexcept;
  // IOCP only: arm one AcceptEx on listener `fd` (shared by submit_accept, its
  // multishot variant, and reap's multishot re-arm). Private static since it
  // names the private Impl. `multishot` is stamped on the IoOp so reap re-arms +
  // stamps kMore. Defined only in the Windows branch of proactor.cc.
  static bool iocp_arm_accept(Impl* impl, std::intptr_t fd,
                              void* addr, void* addrlen,
                              std::uint64_t user_data, bool multishot) noexcept;
  // IOCP only (io-05): synthesize a completion carrying an already-computed 64-bit
  // result for a file metadata op run synchronously (open/close/statx/fsync/
  // fallocate have no async form on Windows). Private static since it names the
  // private Impl. Defined only in the Windows branch of proactor.cc.
  static bool iocp_post_meta(Impl* impl, std::int64_t result,
                             std::uint64_t user_data) noexcept;

  Impl* impl_ = nullptr;  // backend state (IOCP port / io_uring ring), pimpl
};

}  // namespace hyle::io::detail
