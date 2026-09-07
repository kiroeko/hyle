#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

#include "hyle/core/assert.h"
#include "hyle/error/result.h"
#include "hyle/io/buffer.h"
#include "hyle/io/detail/net_budget.h"
#include "hyle/io/endpoint.h"
#include "hyle/io/io_context.h"
#include "hyle/io/operation.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_IO_API
#elif defined(HYLE_IO_CORE_BUILD)
#  define HYLE_IO_API HYLE_EXPORT
#else
#  define HYLE_IO_API HYLE_IMPORT
#endif

// io-04 net, datagram half: RAII UDP socket wrapper over the raw co_await ops in
// net.h. Unlike TCP there is no connection/accept -- a single UdpSocket both
// receives (recv_from, learning each datagram's sender) and sends (send_to, to an
// explicit destination). Shard-per-core aware exactly like TcpStream: the socket
// is bound to an owner worker's ring, and the fiber servicing it must be pinned
// there (Scheduler::spawn_on) so submit + reap stay on one core.
//
// Setup, transfer, endpoint conversion and native error mapping stay behind the
// typed owner. Native sockaddr and negative errno values never cross this API.

namespace hyle::io {

struct UdpSocketOptions {
  bool reuse_address = true;
  bool reuse_port = false;
  bool broadcast = false;
  bool ipv6_only = false;
};

struct [[nodiscard]] DatagramResult {
  std::size_t bytes = 0;
  Endpoint peer;
  Status status{};
  bool truncated = false;

  bool ok() const noexcept { return status.ok(); }
  explicit operator bool() const noexcept { return ok(); }
};

struct [[nodiscard]] BufferDatagramResult {
  BufferSequenceProgress progress{};
  Endpoint peer;
  Status status{};
  bool truncated = false;

  bool ok() const noexcept { return status.ok(); }
  explicit operator bool() const noexcept { return ok(); }
};

// A bound (or unbound-but-sendable) UDP socket. Move-only; closes the fd on
// destruction. recv_from()/send_to() are co_await-able and MUST be awaited from
// the fiber pinned to owner_worker().
class UdpSocket {
 public:
  UdpSocket() noexcept = default;
  HYLE_IO_API UdpSocket(IoContext& io, std::intptr_t fd,
                        unsigned owner_worker) noexcept;
  HYLE_IO_API ~UdpSocket();
  HYLE_IO_API UdpSocket(UdpSocket&& o) noexcept;
  HYLE_IO_API UdpSocket& operator=(UdpSocket&& o) noexcept;
  UdpSocket(const UdpSocket&) = delete;
  UdpSocket& operator=(const UdpSocket&) = delete;

  HYLE_IO_API static Result<UdpSocket> open(
      IoContext& io, unsigned owner_worker,
      AddressFamily family = AddressFamily::kIpv4,
      const UdpSocketOptions& options = {}) noexcept;
  HYLE_IO_API static Result<UdpSocket> bind(
      IoContext& io, unsigned owner_worker, const Endpoint& endpoint,
      const UdpSocketOptions& options = {}) noexcept;
  HYLE_IO_API static Result<UdpSocket> adopt(
      IoContext& io, unsigned owner_worker,
      std::intptr_t native_socket) noexcept;

  bool valid() const noexcept { return fd_ >= 0; }
  std::intptr_t native_handle() const noexcept { return fd_; }
  HYLE_IO_API std::intptr_t release_native_handle() noexcept;
  unsigned owner_worker() const noexcept { return owner_; }
  HYLE_IO_API Result<Endpoint> local_endpoint() const noexcept;

  struct ReceiveAwaitable {
    UdpSocket* self = nullptr;
    std::span<std::byte> buffer;
    OperationOptions options;
    alignas(16) std::array<std::byte, 128> native_peer{};
    detail::BudgetRecvFromOp inner{};

    HYLE_IO_API bool await_ready() noexcept;
    bool await_suspend(std::coroutine_handle<> handle) noexcept {
      return inner.await_suspend(handle);
    }
    HYLE_IO_API DatagramResult await_resume() noexcept;
  };

  struct SendAwaitable {
    UdpSocket* self = nullptr;
    std::span<const std::byte> buffer;
    Endpoint destination;
    OperationOptions options;
    alignas(16) std::array<std::byte, 128> native_destination{};
    std::uint32_t native_length = 0;
    detail::BudgetSendToOp inner{};
    Status preparation{};

    HYLE_IO_API bool await_ready() noexcept;
    bool await_suspend(std::coroutine_handle<> handle) noexcept {
      return inner.await_suspend(handle);
    }
    HYLE_IO_API TransferResult await_resume() noexcept;
  };

  struct ReceiveBuffersAwaitable {
    UdpSocket* self = nullptr;
    detail::BufferSequenceStorage<MutableBuffer> buffers;
    OperationOptions options{};
    alignas(16) std::array<std::byte, 128> native_peer{};
    detail::BudgetRecvFromBuffersOp inner{};
    BufferDatagramResult immediate{};
    bool immediate_ready = false;

    HYLE_IO_API bool await_ready() noexcept;
    bool await_suspend(std::coroutine_handle<> handle) noexcept {
      return inner.await_suspend(handle);
    }
    HYLE_IO_API BufferDatagramResult await_resume() noexcept;
  };

  struct SendBuffersAwaitable {
    UdpSocket* self = nullptr;
    detail::BufferSequenceStorage<ConstBuffer> buffers;
    Endpoint destination;
    OperationOptions options{};
    alignas(16) std::array<std::byte, 128> native_destination{};
    std::uint32_t native_length = 0;
    detail::BudgetSendToBuffersOp inner{};
    BufferSequenceResult immediate{};
    bool immediate_ready = false;

    HYLE_IO_API bool await_ready() noexcept;
    bool await_suspend(std::coroutine_handle<> handle) noexcept {
      return inner.await_suspend(handle);
    }
    HYLE_IO_API BufferSequenceResult await_resume() noexcept;
  };

  ReceiveAwaitable receive_from(
      std::span<std::byte> buffer, OperationOptions options = {}) noexcept {
    HYLE_FATAL_ASSERT(valid() &&
                      sched::current_worker_index() == static_cast<int>(owner_));
    HYLE_FATAL_ASSERT(buffer.size() <=
                      (std::numeric_limits<std::uint32_t>::max)());
    return ReceiveAwaitable{this, buffer, options};
  }

  SendAwaitable send_to(
      std::span<const std::byte> buffer, const Endpoint& destination,
      OperationOptions options = {}) noexcept {
    HYLE_FATAL_ASSERT(valid() &&
                      sched::current_worker_index() == static_cast<int>(owner_));
    HYLE_FATAL_ASSERT(buffer.size() <=
                      (std::numeric_limits<std::uint32_t>::max)());
    return SendAwaitable{this, buffer, destination, options};
  }

  ReceiveBuffersAwaitable receive_from(
      std::span<const MutableBuffer> buffers,
      OperationOptions options = {}) noexcept {
    HYLE_FATAL_ASSERT(valid() &&
                      sched::current_worker_index() == static_cast<int>(owner_));
    return ReceiveBuffersAwaitable{
        this, detail::BufferSequenceStorage<MutableBuffer>(buffers), options};
  }

  SendBuffersAwaitable send_to(
      std::span<const ConstBuffer> buffers, const Endpoint& destination,
      OperationOptions options = {}) noexcept {
    HYLE_FATAL_ASSERT(valid() &&
                      sched::current_worker_index() == static_cast<int>(owner_));
    return SendBuffersAwaitable{
        this, detail::BufferSequenceStorage<ConstBuffer>(buffers), destination,
        options};
  }

 private:
  IoContext* io_ = nullptr;
  std::intptr_t fd_ = -1;
  unsigned owner_ = 0;
};

}  // namespace hyle::io
