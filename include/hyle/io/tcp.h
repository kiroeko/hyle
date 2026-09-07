#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string_view>

#include "hyle/core/assert.h"
#include "hyle/error/result.h"
#include "hyle/io/endpoint.h"
#include "hyle/io/io_context.h"
#include "hyle/io/net.h"
#include "hyle/io/net_budget.h"
#include "hyle/io/net_cancel.h"
#include "hyle/io/net_multishot.h"
#include "hyle/io/net_timeout.h"
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
  HYLE_IO_API TcpStream(IoContext& io, std::intptr_t fd,
                        unsigned owner_worker) noexcept;
  HYLE_IO_API ~TcpStream();
  HYLE_IO_API TcpStream(TcpStream&& o) noexcept;
  HYLE_IO_API TcpStream& operator=(TcpStream&& o) noexcept;
  TcpStream(const TcpStream&) = delete;
  TcpStream& operator=(const TcpStream&) = delete;

  HYLE_IO_API static Result<TcpStream> open(
      IoContext& io, unsigned owner_worker,
      AddressFamily family = AddressFamily::kIpv4) noexcept;
  // Adopt ownership of a socket supplied by an embedding host and associate it
  // with this context's owner shard. On failure the native socket is closed.
  HYLE_IO_API static Result<TcpStream> adopt(IoContext& io,
                                             unsigned owner_worker,
                                             std::intptr_t native_socket) noexcept;

  bool valid() const noexcept { return fd_ >= 0; }
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
    BudgetConnectOp inner{};
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
    RecvOp inner;
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
    SendOp inner;
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
    BudgetRecvOp inner;
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
    BudgetSendOp inner;
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
        RecvOp{&io_->proactor_for_current(), fd_, buffer.data(),
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
        SendOp{&io_->proactor_for_current(), fd_, buffer.data(),
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
    Proactor* proactor = &io_->proactor_for_current();
    return BudgetReadSomeAwaitable{
        BudgetRecvOp{proactor, fd_, buffer.data(),
                     static_cast<std::uint32_t>(buffer.size()),
                     io_->operation_allocator(), options},
        empty};
  }

  BudgetWriteSomeAwaitable write_some(
      std::span<const std::byte> buffer, OperationOptions options) noexcept {
    HYLE_FATAL_ASSERT(valid() &&
                      sched::current_worker_index() == static_cast<int>(owner_));
    HYLE_FATAL_ASSERT(buffer.size() <=
                      (std::numeric_limits<std::uint32_t>::max)());
    const bool empty = buffer.empty();
    Proactor* proactor = &io_->proactor_for_current();
    return BudgetWriteSomeAwaitable{
        BudgetSendOp{proactor, fd_, buffer.data(),
                     static_cast<std::uint32_t>(buffer.size()),
                     io_->operation_allocator(), options},
        empty};
  }

  // Explicit io_uring zero-copy multishot receive track. The caller must first
  // acquire `io.proactor(owner).io_uring_extension()` and register `bgid` on it;
  // requiring that handle prevents a non-Linux caller from entering an API that
  // cannot honor its zero-copy provided-buffer contract.
  MultishotRecvStream read_multishot(IoUringExtension uring, int bgid,
                                     mem::AllocatorRef alloc) noexcept {
    HYLE_FATAL_ASSERT(valid() &&
                      sched::current_worker_index() == static_cast<int>(owner_) &&
                      uring.valid() &&
                      uring.proactor() == &io_->proactor_for_current());
    return MultishotRecvStream{uring, fd_, bgid, alloc};
  }

 private:
  IoContext* io_ = nullptr;
  std::intptr_t fd_ = -1;
  unsigned owner_ = 0;
};

// A listening TCP socket. co_await accept() yields the next connection as a
// TcpStream bound to `accept_owner` (round-robin across workers by default so new
// connections shard across cores). Move-only; closes on destruction.
class TcpListener {
 public:
  TcpListener() noexcept = default;
  HYLE_IO_API TcpListener(IoContext& io, std::intptr_t fd,
                          unsigned owner_worker) noexcept;
  HYLE_IO_API ~TcpListener();
  HYLE_IO_API TcpListener(TcpListener&& o) noexcept;
  HYLE_IO_API TcpListener& operator=(TcpListener&& o) noexcept;
  TcpListener(const TcpListener&) = delete;
  TcpListener& operator=(const TcpListener&) = delete;

  HYLE_IO_API static Result<TcpListener> bind(
      IoContext& io, unsigned owner_worker, const Endpoint& endpoint,
      const TcpListenOptions& options = {}) noexcept;

  bool valid() const noexcept { return fd_ >= 0; }
  std::intptr_t native_handle() const noexcept { return fd_; }
  unsigned owner_worker() const noexcept { return owner_; }
  HYLE_IO_API Result<Endpoint> local_endpoint() const noexcept;

  // An awaitable that co_awaits AcceptOp then wraps the accepted fd (binding it to
  // `assign_owner`'s ring) into a TcpStream. Returned by accept(); co_await it.
  struct AcceptAwaitable {
    TcpListener* self;
    unsigned assign_owner;
    BudgetAcceptOp inner;

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
        BudgetAcceptOp{&io_->proactor_for_current(), fd_,
                       io_->operation_allocator(), options}};
  }

 private:
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
