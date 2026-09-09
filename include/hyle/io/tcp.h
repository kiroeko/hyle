#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string_view>

#include "hyle/core/assert.h"
#include "hyle/error/result.h"
#include "hyle/io/buffer.h"
#include "hyle/io/detail/io_context_access.h"
#include "hyle/io/detail/net.h"
#include "hyle/io/detail/net_budget.h"
#include "hyle/io/endpoint.h"
#include "hyle/io/io_context.h"
#if HYLE_OS_LINUX
#  include "hyle/io/io_uring.h"
#endif
#include "hyle/io/operation.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_IO_API
#elif defined(HYLE_IO_CORE_BUILD)
#  define HYLE_IO_API HYLE_EXPORT
#else
#  define HYLE_IO_API HYLE_IMPORT
#endif

// Scheduler-bound RAII TCP resources. TcpStream owns a connected socket and
// exposes typed span-based asynchronous transfers;
// TcpListener binds/listens and exposes co_await accept -> TcpStream. Both are
// shard-per-core aware: a socket is bound to an owner worker (its ring), and the
// fiber servicing it must be pinned there (Scheduler::spawn_on) so submit + reap
// stay on one core. The wrappers carry the owner index so accept() can hand a new
// connection to the right core.
//
// Native descriptors, backend completion tokens, and negative platform error
// codes stay behind this facade.

namespace hyle::io {

struct TcpListenOptions {
  int backlog = 128;
  bool reuse_address = true;
  bool reuse_port = false;
  // Applies only to IPv6 listeners. false accepts mapped IPv4 connections too
  // where the operating system supports a dual-stack socket.
  bool ipv6_only = false;
};

struct TcpConnectOptions {
  OperationOptions operation{};
  // RFC 8305-style delay before starting another address while an earlier
  // attempt is still pending. A prompt failure accelerates the next attempt.
  time::Duration attempt_delay = std::chrono::milliseconds(250);
  // Caps simultaneously open connection attempts and their socket resources.
  unsigned max_parallel_attempts = 2;
};

struct TcpKeepaliveOptions {
  bool enabled = true;
  // Empty values preserve the operating-system defaults. Non-empty durations
  // are rounded up to whole seconds because that is the portable TCP contract.
  std::optional<time::Duration> idle;
  std::optional<time::Duration> interval;
  std::optional<unsigned> probe_count;
};

// A connected TCP socket. Move-only; closes the socket on destruction. Transfer
// operations MUST be awaited from the fiber pinned to owner_worker().
class TcpStream {
 public:
  TcpStream() noexcept = default;
  HYLE_IO_API ~TcpStream();
  HYLE_IO_API TcpStream(TcpStream&& o) noexcept;
  HYLE_IO_API TcpStream& operator=(TcpStream&& o) noexcept;
  TcpStream(const TcpStream&) = delete;
  TcpStream& operator=(const TcpStream&) = delete;

  HYLE_IO_API static Result<TcpStream> open(
      IoContext& io, unsigned owner_worker,
      AddressFamily family = AddressFamily::kIpv4) noexcept;
  // Adopt ownership of a socket supplied by an embedding host and associate it
  // with this context's owner shard. HYLE normalizes the owned POSIX socket to
  // nonblocking mode (and suppresses per-write SIGPIPE where the OS requires
  // a socket option); the caller must not continue using the transferred
  // handle. On failure the native socket is closed.
  HYLE_IO_API static Result<TcpStream> adopt(IoContext& io,
                                             unsigned owner_worker,
                                             std::intptr_t native_socket) noexcept;

  bool valid() const noexcept { return fd_ >= 0; }
  explicit operator bool() const noexcept { return valid(); }
  std::intptr_t native_handle() const noexcept { return fd_; }
  HYLE_IO_API std::intptr_t release_native_handle() noexcept;
  unsigned owner_worker() const noexcept { return owner_; }
  HYLE_IO_API Status set_nodelay(bool enabled = true) noexcept;
  HYLE_IO_API Status set_keepalive(bool enabled = true) noexcept;
  HYLE_IO_API Status set_keepalive(
      const TcpKeepaliveOptions& options) noexcept;
  HYLE_IO_API Status shutdown_write() noexcept;
  HYLE_IO_API Result<Endpoint> local_endpoint() const noexcept;
  HYLE_IO_API Result<Endpoint> peer_endpoint() const noexcept;

  struct ConnectAwaitable {
    TcpStream* self = nullptr;
    Endpoint endpoint;
    OperationOptions options;
    alignas(16) std::array<std::byte, 128> native_address{};
    std::uint32_t native_length = 0;
    detail::BudgetConnectOp inner{};
    Status preparation{};

    HYLE_IO_API bool await_ready() noexcept;
    bool await_suspend(std::coroutine_handle<> handle) noexcept {
      return inner.await_suspend(handle);
    }
    HYLE_IO_API Status await_resume() noexcept;
  };

  ConnectAwaitable connect(const Endpoint& endpoint,
                           OperationOptions options = {}) noexcept {
    HYLE_FATAL_ASSERT(valid() &&
                      sched::current_worker_index() == static_cast<int>(owner_));
    return ConnectAwaitable{this, endpoint, options};
  }

  struct ReadSomeAwaitable {
    detail::RecvOp inner;
    bool empty = false;

    bool await_ready() noexcept { return empty || inner.await_ready(); }
    bool await_suspend(std::coroutine_handle<> handle) noexcept {
      return inner.await_suspend(handle);
    }
    TransferResult await_resume() noexcept {
      if (empty) return {};
      return detail::transfer_from_native(inner.await_resume(),
                                          detail::kNetworkNativeDomain,
                                          "TCP read", true);
    }
  };

  struct WriteSomeAwaitable {
    detail::SendOp inner;
    bool empty = false;

    bool await_ready() noexcept { return empty || inner.await_ready(); }
    bool await_suspend(std::coroutine_handle<> handle) noexcept {
      return inner.await_suspend(handle);
    }
    TransferResult await_resume() noexcept {
      if (empty) return {};
      return detail::transfer_from_native(inner.await_resume(),
                                          detail::kNetworkNativeDomain,
                                          "TCP write", false);
    }
  };

  struct BudgetReadSomeAwaitable {
    detail::BudgetRecvOp inner;
    bool empty = false;

    bool await_ready() noexcept { return empty || inner.await_ready(); }
    bool await_suspend(std::coroutine_handle<> handle) noexcept {
      return inner.await_suspend(handle);
    }
    TransferResult await_resume() noexcept {
      if (empty) return {};
      return detail::transfer_from_native(inner.await_resume(),
                                          detail::kNetworkNativeDomain,
                                          "TCP read", true);
    }
  };

  struct BudgetWriteSomeAwaitable {
    detail::BudgetSendOp inner;
    bool empty = false;

    bool await_ready() noexcept { return empty || inner.await_ready(); }
    bool await_suspend(std::coroutine_handle<> handle) noexcept {
      return inner.await_suspend(handle);
    }
    TransferResult await_resume() noexcept {
      if (empty) return {};
      return detail::transfer_from_native(inner.await_resume(),
                                          detail::kNetworkNativeDomain,
                                          "TCP write", false);
    }
  };

  struct ReadBuffersAwaitable {
    TcpStream* self = nullptr;
    detail::BufferSequenceStorage<MutableBuffer> buffers;
    OperationOptions options{};
    detail::BudgetRecvBuffersOp inner{};
    BufferSequenceResult immediate{};
    bool immediate_ready = false;

    HYLE_IO_API bool await_ready() noexcept;
    bool await_suspend(std::coroutine_handle<> handle) noexcept {
      return inner.await_suspend(handle);
    }
    HYLE_IO_API BufferSequenceResult await_resume() noexcept;
  };

  struct WriteBuffersAwaitable {
    TcpStream* self = nullptr;
    detail::BufferSequenceStorage<ConstBuffer> buffers;
    OperationOptions options{};
    detail::BudgetSendBuffersOp inner{};
    BufferSequenceResult immediate{};
    bool immediate_ready = false;

    HYLE_IO_API bool await_ready() noexcept;
    bool await_suspend(std::coroutine_handle<> handle) noexcept {
      return inner.await_suspend(handle);
    }
    HYLE_IO_API BufferSequenceResult await_resume() noexcept;
  };

  // Ordinary typed fast path: one borrowed buffer, no allocation, one native
  // submission and at most one scheduler wake. Options/cancellation are added
  // by the unified operation-budget overload in IOCORE-F03.
  ReadSomeAwaitable read_some(std::span<std::byte> buffer) noexcept {
    HYLE_FATAL_ASSERT(valid() &&
                      sched::current_worker_index() == static_cast<int>(owner_));
    HYLE_FATAL_ASSERT(buffer.size() <=
                      (std::numeric_limits<std::uint32_t>::max)());
    const bool empty = buffer.empty();
    return ReadSomeAwaitable{
        detail::RecvOp{&detail::IoContextAccess::proactor_for_current(*io_), fd_, buffer.data(),
               static_cast<std::uint32_t>(buffer.size()), 0},
        empty};
  }

  WriteSomeAwaitable write_some(std::span<const std::byte> buffer) noexcept {
    HYLE_FATAL_ASSERT(valid() &&
                      sched::current_worker_index() == static_cast<int>(owner_));
    HYLE_FATAL_ASSERT(buffer.size() <=
                      (std::numeric_limits<std::uint32_t>::max)());
    const bool empty = buffer.empty();
    return WriteSomeAwaitable{
        detail::SendOp{&detail::IoContextAccess::proactor_for_current(*io_), fd_, buffer.data(),
               static_cast<std::uint32_t>(buffer.size()), 0},
        empty};
  }

  BudgetReadSomeAwaitable read_some(
      std::span<std::byte> buffer, OperationOptions options) noexcept {
    HYLE_FATAL_ASSERT(valid() &&
                      sched::current_worker_index() == static_cast<int>(owner_));
    HYLE_FATAL_ASSERT(buffer.size() <=
                      (std::numeric_limits<std::uint32_t>::max)());
    const bool empty = buffer.empty();
    detail::Proactor* proactor =
        &detail::IoContextAccess::proactor_for_current(*io_);
    return BudgetReadSomeAwaitable{
        detail::BudgetRecvOp{proactor, fd_, buffer.data(),
                     static_cast<std::uint32_t>(buffer.size()),
                     detail::IoContextAccess::operation_allocator(*io_), options},
        empty};
  }

  BudgetWriteSomeAwaitable write_some(
      std::span<const std::byte> buffer, OperationOptions options) noexcept {
    HYLE_FATAL_ASSERT(valid() &&
                      sched::current_worker_index() == static_cast<int>(owner_));
    HYLE_FATAL_ASSERT(buffer.size() <=
                      (std::numeric_limits<std::uint32_t>::max)());
    const bool empty = buffer.empty();
    detail::Proactor* proactor =
        &detail::IoContextAccess::proactor_for_current(*io_);
    return BudgetWriteSomeAwaitable{
        detail::BudgetSendOp{proactor, fd_, buffer.data(),
                     static_cast<std::uint32_t>(buffer.size()),
                     detail::IoContextAccess::operation_allocator(*io_), options},
        empty};
  }

  // Scatter/gather descriptors are copied synchronously into the returned
  // awaitable; the pointed-to bytes remain borrowed until co_await completes.
  // A previous result's cursor resumes a partial transfer without the caller
  // rebuilding subspans. One call maps to one native vectored submission.
  ReadBuffersAwaitable read_some(
      std::span<const MutableBuffer> buffers,
      OperationOptions options = {},
      BufferSequenceCursor cursor = {}) noexcept {
    HYLE_FATAL_ASSERT(valid() &&
                      sched::current_worker_index() == static_cast<int>(owner_));
    return ReadBuffersAwaitable{
        this, detail::BufferSequenceStorage<MutableBuffer>(buffers, cursor),
        options};
  }

  WriteBuffersAwaitable write_some(
      std::span<const ConstBuffer> buffers,
      OperationOptions options = {},
      BufferSequenceCursor cursor = {}) noexcept {
    HYLE_FATAL_ASSERT(valid() &&
                      sched::current_worker_index() == static_cast<int>(owner_));
    return WriteBuffersAwaitable{
        this, detail::BufferSequenceStorage<ConstBuffer>(buffers, cursor),
        options};
  }

#if HYLE_OS_LINUX
  // Explicit provided-buffer/multishot receive route. Acquire and configure the
  // owner worker's extension through IoContext first. A mismatched context or
  // worker is reported as a typed setup error instead of exposing a raw driver.
  HYLE_IO_API Result<IoUringReceiveStream> read_multishot(
      IoUringExtension extension, int group_id,
      mem::AllocatorRef allocator) noexcept;
#endif

 private:
  friend class TcpListener;

  HYLE_IO_API TcpStream(IoContext& io, std::intptr_t fd,
                        unsigned owner_worker) noexcept;

  IoContext* io_ = nullptr;
  std::intptr_t fd_ = -1;
  unsigned owner_ = 0;
};

namespace detail {

// Protocol-engine seam: attempt one receive on an already-owned socket without
// waiting or allocating. Ordinary users should use TcpStream::read_some(); this
// exists so a full-duplex protocol can drain bytes that are already queued before
// it blocks on the opposite direction.
HYLE_IO_API ImmediateTransferResult
tcp_try_read_some(TcpStream &stream, std::span<std::byte> buffer) noexcept;
HYLE_IO_API ImmediateTransferResult
tcp_try_write_some(TcpStream &stream,
                   std::span<const std::byte> buffer) noexcept;

} // namespace detail

// A listening TCP socket. co_await accept() yields the next connection as a
// TcpStream bound to `accept_owner` (round-robin across workers by default so new
// connections shard across cores). Move-only; closes on destruction.
class TcpListener {
 public:
  TcpListener() noexcept = default;
  HYLE_IO_API ~TcpListener();
  HYLE_IO_API TcpListener(TcpListener&& o) noexcept;
  HYLE_IO_API TcpListener& operator=(TcpListener&& o) noexcept;
  TcpListener(const TcpListener&) = delete;
  TcpListener& operator=(const TcpListener&) = delete;

  HYLE_IO_API static Result<TcpListener> bind(
      IoContext& io, unsigned owner_worker, const Endpoint& endpoint,
      const TcpListenOptions& options = {}) noexcept;

  bool valid() const noexcept { return fd_ >= 0; }
  explicit operator bool() const noexcept { return valid(); }
  std::intptr_t native_handle() const noexcept { return fd_; }
  unsigned owner_worker() const noexcept { return owner_; }
  HYLE_IO_API Result<Endpoint> local_endpoint() const noexcept;

  // An awaitable that co_awaits AcceptOp then wraps the accepted fd (binding it to
  // `assign_owner`'s ring) into a TcpStream. Returned by accept(); co_await it.
  struct AcceptAwaitable {
    TcpListener* self;
    unsigned assign_owner;
    detail::BudgetAcceptOp inner;

    bool await_ready() noexcept { return inner.await_ready(); }
    bool await_suspend(std::coroutine_handle<> h) noexcept {
      return inner.await_suspend(h);
    }
    HYLE_IO_API Result<TcpStream> await_resume() noexcept;
  };

  // co_await listener.accept() -> TcpStream. The new connection is bound to
  // `assign_owner` (its fibers must be spawn_on(assign_owner) to service it on
  // that core). accept() itself is awaited from the fiber pinned to the
  // listener's owner_worker().
  AcceptAwaitable accept(unsigned assign_owner,
                         OperationOptions options = {}) noexcept {
    HYLE_FATAL_ASSERT(valid() &&
                      sched::current_worker_index() == static_cast<int>(owner_));
    return AcceptAwaitable{
        this, assign_owner,
        detail::BudgetAcceptOp{
            &detail::IoContextAccess::proactor_for_current(*io_), fd_,
            detail::IoContextAccess::operation_allocator(*io_), options}};
  }

 private:
  HYLE_IO_API TcpListener(IoContext& io, std::intptr_t fd,
                          unsigned owner_worker) noexcept;

  Result<TcpStream> finish_accept(std::int64_t result,
                                  unsigned assign_owner) noexcept;
  IoContext* io_ = nullptr;
  std::intptr_t fd_ = -1;
  unsigned owner_ = 0;
};

// Composed stream operations preserve partial progress together with the
// terminal status/EOF. One absolute OperationOptions budget is reused across
// every partial syscall; it is never reset per chunk.
HYLE_IO_API sched::Task<TransferResult> read_exact(
    mem::AllocatorRef coroutine_allocator, TcpStream& stream,
    std::span<std::byte> buffer, OperationOptions options = {}) noexcept;
HYLE_IO_API sched::Task<TransferResult> write_all(
    mem::AllocatorRef coroutine_allocator, TcpStream& stream,
    std::span<const std::byte> buffer, OperationOptions options = {}) noexcept;
HYLE_IO_API sched::Task<BufferSequenceResult> read_exact(
    mem::AllocatorRef coroutine_allocator, TcpStream& stream,
    std::span<const MutableBuffer> buffers,
    OperationOptions options = {}) noexcept;
HYLE_IO_API sched::Task<BufferSequenceResult> write_all(
    mem::AllocatorRef coroutine_allocator, TcpStream& stream,
    std::span<const ConstBuffer> buffers,
    OperationOptions options = {}) noexcept;

// Connect one of an endpoint sequence using bounded RFC 8305-style staggered
// attempts. Inputs are copied before this function returns, so caller storage
// need not outlive the returned Task. Address families are interleaved while
// preserving resolver order within each family. Run the Task on owner_worker.
HYLE_IO_API sched::Task<Result<TcpStream>> connect_tcp(
    mem::AllocatorRef coroutine_allocator, IoContext& io,
    unsigned owner_worker, std::span<const Endpoint> endpoints,
    TcpConnectOptions options = {}) noexcept;

// Owning hostname convenience: copies `host` before returning, resolves without
// blocking a Scheduler worker, then applies the same total operation budget and
// bounded Happy-Eyeballs policy as the endpoint-sequence overload.
HYLE_IO_API sched::Task<Result<TcpStream>> connect_tcp(
    mem::AllocatorRef coroutine_allocator, IoContext& io,
    unsigned owner_worker, std::string_view host, std::uint16_t port,
    TcpConnectOptions options = {}) noexcept;

}  // namespace hyle::io
