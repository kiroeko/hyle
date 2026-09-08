#include "hyle/io/tcp.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstring>
#include <utility>

#include "hyle/core/environment.h"  // HYLE_OS_* (must precede the per-OS socket headers)
#include "hyle/error/system_error.h"
#include "hyle/io/detail/io_context_access.h"
#if HYLE_OS_LINUX
#  include "hyle/io/detail/io_uring_access.h"
#endif
#include "hyle/io/dns.h"
#include "hyle/sched/cancel.h"
#include "hyle/sched/sync.h"
#include "hyle/sched/timer.h"
#include "endpoint_native.h"

#if HYLE_OS_WINDOWS
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  include <windows.h>
#else
#  include <arpa/inet.h>
#  include <fcntl.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <sys/socket.h>
#  include <unistd.h>
#endif

// io-04 net convenience layer implementation: RAII fd management + cross-platform
// TCP socket setup. The co_await methods themselves are inline in tcp.h (they
// just build net.h awaitables); this file owns fd lifetime and the socket/bind/
// listen syscalls behind tcp_listen / tcp_socket.

namespace hyle::io {
namespace {

// Close a socket fd on either platform.
void close_fd(std::intptr_t fd) noexcept {
  if (fd < 0) return;
#if HYLE_OS_WINDOWS
  ::closesocket(static_cast<SOCKET>(fd));
#else
  ::close(static_cast<int>(fd));
#endif
}

// Create an overlapped/nonblocking TCP socket, or -1 on failure.
int native_family(AddressFamily family) noexcept {
  return family == AddressFamily::kIpv6 ? AF_INET6 : AF_INET;
}

std::intptr_t make_tcp_socket(AddressFamily family) noexcept {
  const int af = native_family(family);
#if HYLE_OS_WINDOWS
  SOCKET s = ::WSASocketW(af, SOCK_STREAM, 0, nullptr, 0, WSA_FLAG_OVERLAPPED);
  return s == INVALID_SOCKET ? -1 : static_cast<std::intptr_t>(s);
#else
  int s = ::socket(af, SOCK_STREAM, 0);
  if (s < 0) return -1;
#  if HYLE_OS_MACOS
  // macOS has no MSG_NOSIGNAL. Without SO_NOSIGPIPE a normal peer-close race
  // during TLS close_notify can terminate the entire process instead of
  // completing the pending write with EPIPE.
  const int no_sigpipe = 1;
  if (::setsockopt(s, SOL_SOCKET, SO_NOSIGPIPE, &no_sigpipe,
                   sizeof(no_sigpipe)) != 0) {
    ::close(s);
    return -1;
  }
#  endif
  const int flags = ::fcntl(s, F_GETFL, 0);
  if (flags < 0 || ::fcntl(s, F_SETFL, flags | O_NONBLOCK) != 0) {
    ::close(s);
    return -1;
  }
  return static_cast<std::intptr_t>(s);
#endif
}

Status socket_error(const char* operation) noexcept {
#if HYLE_OS_WINDOWS
  return status_from_winsock(::WSAGetLastError(), operation);
#else
  return status_from_errno(errno, operation);
#endif
}

Status handle_error(const char* operation) noexcept {
#if HYLE_OS_WINDOWS
  return status_from_win32(::GetLastError(), operation);
#else
  return status_from_errno(errno, operation);
#endif
}

Status configure_adopted_socket(std::intptr_t fd) noexcept {
#if HYLE_OS_WINDOWS
  (void)fd;
  return {};
#else
#  if HYLE_OS_MACOS
  const int no_sigpipe = 1;
  if (::setsockopt(static_cast<int>(fd), SOL_SOCKET, SO_NOSIGPIPE,
                   &no_sigpipe, sizeof(no_sigpipe)) != 0) {
    return socket_error("set SO_NOSIGPIPE on adopted TCP socket");
  }
#  endif
  const int flags = ::fcntl(static_cast<int>(fd), F_GETFL, 0);
  if (flags < 0 ||
      ::fcntl(static_cast<int>(fd), F_SETFL, flags | O_NONBLOCK) != 0) {
    return socket_error("set adopted TCP socket nonblocking");
  }
  return {};
#endif
}

Status set_socket_integer(std::intptr_t fd, int level, int option, int value,
                          const char* operation) noexcept {
#if HYLE_OS_WINDOWS
  if (::setsockopt(static_cast<SOCKET>(fd), level, option,
                   reinterpret_cast<const char*>(&value), sizeof(value)) != 0)
#else
  if (::setsockopt(static_cast<int>(fd), level, option, &value,
                   sizeof(value)) != 0)
#endif
    return socket_error(operation);
  return Status{};
}

Result<int> keepalive_seconds(time::Duration duration,
                              const char* field) noexcept {
  if (duration <= time::Duration::zero())
    return make_unexpected(
        Status::InvalidArgument(field).WithContext(
            "TCP keepalive duration must be positive"));
  auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
  if (seconds < duration) ++seconds;
  if (seconds.count() > INT_MAX)
    return make_unexpected(
        Status::OutOfRange(field).WithContext(
            "TCP keepalive duration exceeds the native range"));
  return static_cast<int>(seconds.count());
}

struct ConnectAttemptOutcome {
  Result<TcpStream> result;
};

using ConnectOutcomeChannel = sched::Channel<ConnectAttemptOutcome>;

sched::Task<Result<TcpStream>> connect_error_task(mem::AllocatorRef,
                                                   Status status) noexcept {
  co_return make_unexpected(std::move(status));
}

Result<EndpointList> interleave_endpoints(
    mem::AllocatorRef allocator,
    std::span<const Endpoint> endpoints) noexcept {
  EndpointList primary(allocator);
  EndpointList secondary(allocator);
  EndpointList ordered(allocator);
  if (endpoints.empty())
    return make_unexpected(
        Status::InvalidArgument("TCP connect requires at least one endpoint"));
  const AddressFamily first_family = endpoints.front().family();
  for (const Endpoint& endpoint : endpoints) {
    if (!endpoint.valid())
      return make_unexpected(
          Status::InvalidArgument("TCP connect endpoint is invalid"));
    EndpointList& destination = endpoint.family() == first_family
                                    ? primary
                                    : secondary;
    if (!destination.try_push_back(endpoint))
      return make_unexpected(Status::ResourceExhausted(
          "TCP endpoint ordering allocation failed"));
  }
  if (!ordered.try_reserve(endpoints.size()))
    return make_unexpected(Status::ResourceExhausted(
        "TCP endpoint ordering allocation failed"));
  std::size_t primary_index = 0;
  std::size_t secondary_index = 0;
  while (primary_index < primary.size() ||
         secondary_index < secondary.size()) {
    if (primary_index < primary.size() &&
        !ordered.try_push_back(primary[primary_index++]))
      return make_unexpected(Status::ResourceExhausted(
          "TCP endpoint ordering allocation failed"));
    if (secondary_index < secondary.size() &&
        !ordered.try_push_back(secondary[secondary_index++]))
      return make_unexpected(Status::ResourceExhausted(
          "TCP endpoint ordering allocation failed"));
  }
  return ordered;
}

sched::Task<void> run_connect_attempt(
    mem::AllocatorRef, IoContext* io, unsigned owner_worker,
    Endpoint endpoint, OperationOptions operation,
    sched::CancelToken cancellation,
    ConnectOutcomeChannel* outcomes) noexcept {
  Result<TcpStream> opened =
      TcpStream::open(*io, owner_worker, endpoint.family());
  if (!opened.has_value()) {
    auto sent = outcomes->send(ConnectAttemptOutcome{
        make_unexpected(std::move(opened).error())});
    HYLE_FATAL_ASSERT(sent.has_value());
    co_return;
  }

  TcpStream stream = std::move(*opened);
  operation.cancel = &cancellation;
  Status connected = co_await stream.connect(endpoint, operation);
  auto sent = connected.ok()
                  ? outcomes->send(
                        ConnectAttemptOutcome{std::move(stream)})
                  : outcomes->send(ConnectAttemptOutcome{
                        make_unexpected(std::move(connected))});
  HYLE_FATAL_ASSERT(sent.has_value());
}

Status start_connect_attempt(IoContext& io, unsigned owner_worker,
                             mem::AllocatorRef allocator,
                             const Endpoint& endpoint,
                             OperationOptions operation,
                             const sched::CancelToken& cancellation,
                             ConnectOutcomeChannel* outcomes) noexcept {
  auto task = run_connect_attempt(allocator, &io, owner_worker, endpoint,
                                  operation, cancellation, outcomes);
  if (!task.valid())
    return Status::ResourceExhausted(
        "TCP connection attempt coroutine allocation failed");
  return detail::IoContextAccess::scheduler(io).spawn_detached_on(
      owner_worker,
      [task = std::move(task)]() mutable noexcept { task.resume(); });
}

sched::Task<Result<TcpStream>> connect_ordered(
    mem::AllocatorRef allocator, IoContext& io, unsigned owner_worker,
    EndpointList endpoints, TcpConnectOptions options) noexcept {
  if (!io.valid())
    co_return make_unexpected(
        Status::FailedPrecondition("TCP connect requires a valid IoContext"));
  if (owner_worker >= io.num_workers())
    co_return make_unexpected(
        Status::OutOfRange("TCP owner worker is out of range"));
  if (options.attempt_delay < time::Duration::zero())
    co_return make_unexpected(
        Status::InvalidArgument("TCP attempt delay cannot be negative"));
  if (options.max_parallel_attempts == 0)
    co_return make_unexpected(Status::InvalidArgument(
        "TCP max parallel attempts must be at least one"));
  HYLE_FATAL_ASSERT(sched::current_worker_index() ==
                    static_cast<int>(owner_worker));

  const std::size_t capacity =
      (std::min)(endpoints.size(),
                 static_cast<std::size_t>(options.max_parallel_attempts));
  Status channel_status;
  ConnectOutcomeChannel outcomes(allocator, capacity, &channel_status);
  if (!channel_status.ok())
    co_return make_unexpected(std::move(channel_status).WithContext(
        "create TCP connection-attempt channel"));

  Result<sched::CancelToken> cancellation_result =
      options.operation.cancel != nullptr
          ? options.operation.cancel->try_derive()
          : sched::CancelToken::try_create_root(allocator);
  if (!cancellation_result.has_value())
    co_return make_unexpected(std::move(cancellation_result).error());
  sched::CancelToken cancellation = std::move(*cancellation_result);

  auto budget_status = [&]() noexcept -> Status {
    if (options.operation.cancel != nullptr &&
        options.operation.cancel->is_cancelled())
      return Status::Cancelled("TCP connection cancelled");
    if (options.operation.deadline.has_value() &&
        *options.operation.deadline <= time::monotonic_now())
      return Status::DeadlineExceeded("TCP connection deadline exceeded");
    return Status{};
  };

  std::size_t next_endpoint = 0;
  std::size_t active = 0;
  Status last_error = Status::Unavailable("all TCP connection attempts failed");
  auto launch_one = [&]() noexcept {
    while (next_endpoint < endpoints.size()) {
      Status started = start_connect_attempt(
          io, owner_worker, allocator, endpoints[next_endpoint++],
          options.operation, cancellation, &outcomes);
      if (started.ok()) {
        ++active;
        return true;
      }
      last_error = std::move(started);
    }
    return false;
  };

  auto cancel_and_drain = [&]() noexcept -> Status {
    cancellation.cancel();
    while (active != 0) {
      Result<ConnectAttemptOutcome> outcome = outcomes.recv();
      if (!outcome.has_value())
        return std::move(outcome).error().WithContext(
            "drain TCP connection attempts");
      --active;
    }
    return Status{};
  };

  Status initial_budget = budget_status();
  if (!initial_budget.ok())
    co_return make_unexpected(std::move(initial_budget));
  (void)launch_one();
  time::MonotonicTimePoint next_launch =
      time::deadline_after(options.attempt_delay);
  while (active != 0 || next_endpoint < endpoints.size()) {
    if (active == 0) {
      Status expired = budget_status();
      if (!expired.ok()) co_return make_unexpected(std::move(expired));
      (void)launch_one();
      next_launch = time::deadline_after(options.attempt_delay);
      continue;
    }

    Result<ConnectAttemptOutcome> outcome =
        next_endpoint < endpoints.size() &&
                active < options.max_parallel_attempts
            ? outcomes.recv_for(time::remaining(next_launch))
            : outcomes.recv();
    if (!outcome.has_value() && outcome.error().IsDeadlineExceeded()) {
      Status expired = budget_status();
      if (!expired.ok()) {
        Status drained = cancel_and_drain();
        if (!drained.ok()) co_return make_unexpected(std::move(drained));
        co_return make_unexpected(std::move(expired));
      }
      (void)launch_one();
      next_launch = time::deadline_after(options.attempt_delay);
      continue;
    }
    if (!outcome.has_value())
      co_return make_unexpected(std::move(outcome).error().WithContext(
          "receive TCP connection attempt"));

    --active;
    if (outcome->result.has_value()) {
      TcpStream winner = std::move(*outcome->result);
      cancellation.cancel();
      while (active != 0) {
        Result<ConnectAttemptOutcome> loser = outcomes.recv();
        if (!loser.has_value())
          co_return make_unexpected(std::move(loser).error().WithContext(
              "drain TCP connection attempts"));
        --active;
      }
      co_return winner;
    }

    last_error = std::move(outcome->result).error();
    Status expired = budget_status();
    if (!expired.ok()) {
      Status drained = cancel_and_drain();
      if (!drained.ok()) co_return make_unexpected(std::move(drained));
      co_return make_unexpected(std::move(expired));
    }
    if (next_endpoint < endpoints.size() &&
        active < options.max_parallel_attempts) {
      (void)launch_one();
      next_launch = time::deadline_after(options.attempt_delay);
    }
  }
  co_return make_unexpected(
      std::move(last_error).WithContext("connect TCP endpoint set"));
}

struct OwnedHost {
  explicit OwnedHost(mem::AllocatorRef allocator) noexcept : bytes(allocator) {}
  container::SmallVector<char, 256, mem::AllocatorRef> bytes;
};

sched::Task<Result<TcpStream>> connect_host(
    mem::AllocatorRef allocator, IoContext& io, unsigned owner_worker,
    OwnedHost host, std::uint16_t port, TcpConnectOptions options) noexcept {
  ResolveHandle resolving = resolve(io, allocator,
                                    std::string_view(host.bytes.data(),
                                                     host.bytes.size()),
                                    port);
  Result<EndpointList> resolved = resolving.join(options.operation);
  if (!resolved.has_value())
    co_return make_unexpected(std::move(resolved).error());
  Result<EndpointList> ordered = interleave_endpoints(
      allocator, std::span<const Endpoint>(resolved->data(), resolved->size()));
  if (!ordered.has_value())
    co_return make_unexpected(std::move(ordered).error());
  auto task = connect_ordered(allocator, io, owner_worker,
                              std::move(*ordered), options);
  if (!task.valid())
    co_return make_unexpected(Status::ResourceExhausted(
        "TCP connection task allocation failed"));
  co_return co_await task;
}

Result<Endpoint> query_endpoint(std::intptr_t fd, bool peer) noexcept {
  sockaddr_storage address{};
#if HYLE_OS_WINDOWS
  int size = sizeof(address);
  const int result = peer ? ::getpeername(static_cast<SOCKET>(fd),
                                          reinterpret_cast<sockaddr*>(&address),
                                          &size)
                          : ::getsockname(static_cast<SOCKET>(fd),
                                         reinterpret_cast<sockaddr*>(&address),
                                         &size);
#else
  socklen_t size = sizeof(address);
  const int result = peer ? ::getpeername(static_cast<int>(fd),
                                          reinterpret_cast<sockaddr*>(&address),
                                          &size)
                          : ::getsockname(static_cast<int>(fd),
                                         reinterpret_cast<sockaddr*>(&address),
                                         &size);
#endif
  if (result != 0)
    return make_unexpected(socket_error(peer ? "getpeername" : "getsockname"));
  return EndpointNativeAccess::decode(&address,
                                      static_cast<std::uint32_t>(size));
}

}  // namespace

// ---- TcpStream -------------------------------------------------------------

TcpStream::TcpStream(IoContext& io, std::intptr_t fd, unsigned owner) noexcept
    : io_(&io), fd_(fd), owner_(owner) {}

TcpStream::~TcpStream() { close_fd(fd_); }

TcpStream::TcpStream(TcpStream&& o) noexcept
    : io_(o.io_), fd_(std::exchange(o.fd_, -1)), owner_(o.owner_) {}

TcpStream& TcpStream::operator=(TcpStream&& o) noexcept {
  if (this != &o) {
    close_fd(fd_);
    io_ = o.io_;
    fd_ = std::exchange(o.fd_, -1);
    owner_ = o.owner_;
  }
  return *this;
}

Result<TcpStream> TcpStream::open(IoContext& io, unsigned owner_worker,
                                  AddressFamily family) noexcept {
  if (!io.valid())
    return make_unexpected(
        Status::FailedPrecondition("TCP open requires a valid IoContext"));
  if (owner_worker >= io.num_workers())
    return make_unexpected(Status::OutOfRange("TCP owner worker is out of range"));
  std::intptr_t fd = make_tcp_socket(family);
  if (fd < 0) return make_unexpected(socket_error("create TCP socket"));
  if (!detail::IoContextAccess::bind_handle(io, owner_worker, fd)) {
    Status status = handle_error("associate TCP socket with IO shard");
    close_fd(fd);
    return make_unexpected(std::move(status));
  }
  return TcpStream{io, fd, owner_worker};
}

Result<TcpStream> TcpStream::adopt(IoContext& io, unsigned owner_worker,
                                    std::intptr_t native_socket) noexcept {
  if (native_socket < 0)
    return make_unexpected(Status::InvalidArgument("invalid native TCP socket"));
  if (!io.valid()) {
    close_fd(native_socket);
    return make_unexpected(
        Status::FailedPrecondition("TCP adopt requires a valid IoContext"));
  }
  if (owner_worker >= io.num_workers()) {
    close_fd(native_socket);
    return make_unexpected(Status::OutOfRange("TCP owner worker is out of range"));
  }
  Status configured = configure_adopted_socket(native_socket);
  if (!configured.ok()) {
    close_fd(native_socket);
    return make_unexpected(std::move(configured));
  }
  if (!detail::IoContextAccess::bind_handle(io, owner_worker, native_socket)) {
    Status status = handle_error("associate adopted TCP socket with IO shard");
    close_fd(native_socket);
    return make_unexpected(std::move(status));
  }
  return TcpStream{io, native_socket, owner_worker};
}

std::intptr_t TcpStream::release_native_handle() noexcept {
  io_ = nullptr;
  return std::exchange(fd_, -1);
}

detail::ImmediateTransferResult
detail::tcp_try_read_some(TcpStream &stream,
                          std::span<std::byte> buffer) noexcept {
  HYLE_FATAL_ASSERT(stream.valid() &&
                    sched::current_worker_index() ==
                        static_cast<int>(stream.owner_worker()));
  if (buffer.empty())
    return {};

#if HYLE_OS_WINDOWS
  // HYLE accepts host-adopted sockets and therefore cannot assume FIONBIO. A
  // zero-time readiness query makes the following recv nonblocking without
  // changing the embedding host's socket mode. With the single-reader contract
  // no competing consumer can drain the readiness between these two calls.
  WSAPOLLFD descriptor{};
  descriptor.fd = static_cast<SOCKET>(stream.native_handle());
  descriptor.events = POLLRDNORM;
  const int ready = ::WSAPoll(&descriptor, 1, 0);
  if (ready == 0) {
    return {{}, ImmediateTransferState::kWouldBlock};
  }
  if (ready == SOCKET_ERROR) {
    const int error = ::WSAGetLastError();
    return {transfer_from_native(-static_cast<std::int64_t>(error),
                                 kNetworkNativeDomain,
                                 "TCP immediate read readiness", true),
            ImmediateTransferState::kReady};
  }
  const int requested = static_cast<int>((std::min)(
      buffer.size(), static_cast<std::size_t>((std::numeric_limits<int>::max)())));
  const int received = ::recv(descriptor.fd,
                              reinterpret_cast<char *>(buffer.data()),
                              requested, 0);
  if (received == SOCKET_ERROR) {
    const int error = ::WSAGetLastError();
    if (error == WSAEWOULDBLOCK)
      return {{}, ImmediateTransferState::kWouldBlock};
    return {transfer_from_native(-static_cast<std::int64_t>(error),
                                 kNetworkNativeDomain,
                                 "TCP immediate read", true),
            ImmediateTransferState::kReady};
  }
  return {transfer_from_native(received, kNetworkNativeDomain,
                               "TCP immediate read", true),
          ImmediateTransferState::kReady};
#else
  ssize_t received;
  do {
    received = ::recv(static_cast<int>(stream.native_handle()), buffer.data(),
                      buffer.size(), MSG_DONTWAIT);
  } while (received < 0 && errno == EINTR);
  if (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
    return {{}, ImmediateTransferState::kWouldBlock};
  const std::int64_t native =
      received < 0 ? -static_cast<std::int64_t>(errno)
                   : static_cast<std::int64_t>(received);
  return {transfer_from_native(native, kNetworkNativeDomain,
                               "TCP immediate read", true),
          ImmediateTransferState::kReady};
#endif
}

detail::ImmediateTransferResult
detail::tcp_try_write_some(TcpStream &stream,
                           std::span<const std::byte> buffer) noexcept {
  HYLE_FATAL_ASSERT(stream.valid() &&
                    sched::current_worker_index() ==
                        static_cast<int>(stream.owner_worker()));
  if (buffer.empty())
    return {};

#if HYLE_OS_WINDOWS
  WSAPOLLFD descriptor{};
  descriptor.fd = static_cast<SOCKET>(stream.native_handle());
  descriptor.events = POLLWRNORM;
  const int ready = ::WSAPoll(&descriptor, 1, 0);
  if (ready == 0)
    return {{}, ImmediateTransferState::kWouldBlock};
  if (ready == SOCKET_ERROR) {
    const int error = ::WSAGetLastError();
    return {transfer_from_native(-static_cast<std::int64_t>(error),
                                 kNetworkNativeDomain,
                                 "TCP immediate write readiness", false),
            ImmediateTransferState::kReady};
  }
  const int requested = static_cast<int>((std::min)(
      buffer.size(), static_cast<std::size_t>((std::numeric_limits<int>::max)())));
  const int written = ::send(descriptor.fd,
                             reinterpret_cast<const char *>(buffer.data()),
                             requested, 0);
  if (written == SOCKET_ERROR) {
    const int error = ::WSAGetLastError();
    if (error == WSAEWOULDBLOCK)
      return {{}, ImmediateTransferState::kWouldBlock};
    return {transfer_from_native(-static_cast<std::int64_t>(error),
                                 kNetworkNativeDomain,
                                 "TCP immediate write", false),
            ImmediateTransferState::kReady};
  }
  return {transfer_from_native(written, kNetworkNativeDomain,
                               "TCP immediate write", false),
          ImmediateTransferState::kReady};
#else
  int flags = MSG_DONTWAIT;
#  if defined(MSG_NOSIGNAL)
  flags |= MSG_NOSIGNAL;
#  endif
  ssize_t written;
  do {
    written = ::send(static_cast<int>(stream.native_handle()), buffer.data(),
                     buffer.size(), flags);
  } while (written < 0 && errno == EINTR);
  if (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
    return {{}, ImmediateTransferState::kWouldBlock};
  const std::int64_t native =
      written < 0 ? -static_cast<std::int64_t>(errno)
                  : static_cast<std::int64_t>(written);
  return {transfer_from_native(native, kNetworkNativeDomain,
                               "TCP immediate write", false),
          ImmediateTransferState::kReady};
#endif
}

Status TcpStream::set_nodelay(bool enabled) noexcept {
  if (!valid()) return Status::FailedPrecondition("TCP stream is not open");
  const int value = enabled ? 1 : 0;
#if HYLE_OS_WINDOWS
  if (::setsockopt(static_cast<SOCKET>(fd_), IPPROTO_TCP, TCP_NODELAY,
                   reinterpret_cast<const char*>(&value), sizeof(value)) != 0)
    return socket_error("set TCP_NODELAY");
#else
  if (::setsockopt(static_cast<int>(fd_), IPPROTO_TCP, TCP_NODELAY, &value,
                   sizeof(value)) != 0)
    return socket_error("set TCP_NODELAY");
#endif
  return Status{};
}

Status TcpStream::set_keepalive(bool enabled) noexcept {
  TcpKeepaliveOptions options;
  options.enabled = enabled;
  return set_keepalive(options);
}

Status TcpStream::set_keepalive(
    const TcpKeepaliveOptions& options) noexcept {
  if (!valid()) return Status::FailedPrecondition("TCP stream is not open");
  if (!options.enabled &&
      (options.idle.has_value() || options.interval.has_value() ||
       options.probe_count.has_value()))
    return Status::InvalidArgument(
        "disabled TCP keepalive cannot carry probe tuning");

  std::optional<int> idle;
  std::optional<int> interval;
  if (options.idle.has_value()) {
    auto value = keepalive_seconds(*options.idle, "idle");
    if (!value.has_value()) return std::move(value).error();
    idle = *value;
  }
  if (options.interval.has_value()) {
    auto value = keepalive_seconds(*options.interval, "interval");
    if (!value.has_value()) return std::move(value).error();
    interval = *value;
  }
  if (options.probe_count.has_value() && *options.probe_count == 0)
    return Status::InvalidArgument(
        "TCP keepalive probe count must be at least one");

  Status status = set_socket_integer(fd_, SOL_SOCKET, SO_KEEPALIVE,
                                     options.enabled ? 1 : 0,
                                     "set SO_KEEPALIVE");
  if (!status.ok() || !options.enabled) return status;

  if (idle.has_value()) {
#if HYLE_OS_MACOS
    status = set_socket_integer(fd_, IPPROTO_TCP, TCP_KEEPALIVE, *idle,
                                "set TCP_KEEPALIVE");
#elif defined(TCP_KEEPIDLE)
    status = set_socket_integer(fd_, IPPROTO_TCP, TCP_KEEPIDLE, *idle,
                                "set TCP_KEEPIDLE");
#else
    return Status::Unimplemented(
        "TCP keepalive idle tuning is not supported by this platform");
#endif
    if (!status.ok()) return status;
  }
  if (interval.has_value()) {
#if defined(TCP_KEEPINTVL)
    status = set_socket_integer(fd_, IPPROTO_TCP, TCP_KEEPINTVL, *interval,
                                "set TCP_KEEPINTVL");
#else
    return Status::Unimplemented(
        "TCP keepalive interval tuning is not supported by this platform");
#endif
    if (!status.ok()) return status;
  }
  if (options.probe_count.has_value()) {
#if defined(TCP_KEEPCNT)
    if (*options.probe_count > static_cast<unsigned>(INT_MAX))
      return Status::OutOfRange(
          "TCP keepalive probe count exceeds the native range");
    status = set_socket_integer(fd_, IPPROTO_TCP, TCP_KEEPCNT,
                                static_cast<int>(*options.probe_count),
                                "set TCP_KEEPCNT");
#else
    return Status::Unimplemented(
        "TCP keepalive probe-count tuning is not supported by this platform");
#endif
  }
  return status;
}

Status TcpStream::shutdown_write() noexcept {
  if (!valid()) return Status::FailedPrecondition("TCP stream is not open");
#if HYLE_OS_WINDOWS
  if (::shutdown(static_cast<SOCKET>(fd_), SD_SEND) != 0)
    return socket_error("shutdown TCP write side");
#else
  if (::shutdown(static_cast<int>(fd_), SHUT_WR) != 0)
    return socket_error("shutdown TCP write side");
#endif
  return Status{};
}

Result<Endpoint> TcpStream::local_endpoint() const noexcept {
  if (!valid())
    return make_unexpected(Status::FailedPrecondition("TCP stream is not open"));
  return query_endpoint(fd_, false);
}

Result<Endpoint> TcpStream::peer_endpoint() const noexcept {
  if (!valid())
    return make_unexpected(Status::FailedPrecondition("TCP stream is not open"));
  return query_endpoint(fd_, true);
}

#if HYLE_OS_LINUX
Result<IoUringReceiveStream> TcpStream::read_multishot(
    IoUringExtension extension, int group_id,
    mem::AllocatorRef allocator) noexcept {
  if (!valid()) {
    return make_unexpected(
        Status::FailedPrecondition("multishot receive requires a valid stream"));
  }
  if (!allocator.valid()) {
    return make_unexpected(Status::InvalidArgument(
        "multishot receive requires a complete allocator"));
  }
  if (group_id < 0 || group_id > UINT16_MAX) {
    return make_unexpected(
        Status::InvalidArgument("invalid io_uring receive-ring group id"));
  }
  if (!detail::IoUringAccess::belongs_to(extension, *io_, owner_)) {
    return make_unexpected(Status::FailedPrecondition(
        "io_uring extension does not own this stream"));
  }
  return IoUringReceiveStream(
      detail::IoUringAccess::proactor(extension), fd_, group_id, allocator);
}
#endif

bool TcpStream::ConnectAwaitable::await_ready() noexcept {
  auto native = EndpointNativeAccess::encode(endpoint);
  if (!native.has_value()) {
    preparation = native.error();
    return true;
  }
  std::memcpy(native_address.data(), native->storage, native->size);
  native_length = native->size;
  inner.proactor =
      &detail::IoContextAccess::proactor_for_current(*self->io_);
  inner.fd = self->fd_;
  inner.address = native_address.data();
  inner.address_length = native_length;
  inner.allocator = detail::IoContextAccess::operation_allocator(*self->io_);
  inner.options = options;
  return inner.await_ready();
}

Status TcpStream::ConnectAwaitable::await_resume() noexcept {
  if (!preparation.ok()) return preparation;
  return detail::transfer_from_native(inner.await_resume(),
                                      detail::kNetworkNativeDomain,
                                      "TCP connect", false)
      .status;
}

// ---- TcpListener -----------------------------------------------------------

TcpListener::TcpListener(IoContext& io, std::intptr_t fd, unsigned owner) noexcept
    : io_(&io), fd_(fd), owner_(owner) {}

TcpListener::~TcpListener() { close_fd(fd_); }

TcpListener::TcpListener(TcpListener&& o) noexcept
    : io_(o.io_), fd_(std::exchange(o.fd_, -1)), owner_(o.owner_) {}

TcpListener& TcpListener::operator=(TcpListener&& o) noexcept {
  if (this != &o) {
    close_fd(fd_);
    io_ = o.io_;
    fd_ = std::exchange(o.fd_, -1);
    owner_ = o.owner_;
  }
  return *this;
}

Result<TcpListener> TcpListener::bind(IoContext& io, unsigned owner_worker,
                                      const Endpoint& endpoint,
                                      const TcpListenOptions& options) noexcept {
  if (!io.valid())
    return make_unexpected(
        Status::FailedPrecondition("TCP bind requires a valid IoContext"));
  if (owner_worker >= io.num_workers())
    return make_unexpected(Status::OutOfRange("TCP owner worker is out of range"));
  if (options.backlog < 0)
    return make_unexpected(Status::InvalidArgument("TCP backlog cannot be negative"));
  auto native = EndpointNativeAccess::encode(endpoint);
  if (!native.has_value()) return make_unexpected(native.error());
  std::intptr_t fd = make_tcp_socket(endpoint.family());
  if (fd < 0) return make_unexpected(socket_error("create TCP listener"));

  const int reuse = options.reuse_address ? 1 : 0;
#if HYLE_OS_WINDOWS
  if (::setsockopt(static_cast<SOCKET>(fd), SOL_SOCKET, SO_REUSEADDR,
                   reinterpret_cast<const char*>(&reuse), sizeof(reuse)) != 0) {
#else
  if (::setsockopt(static_cast<int>(fd), SOL_SOCKET, SO_REUSEADDR, &reuse,
                   sizeof(reuse)) != 0) {
#endif
    Status status = socket_error("set SO_REUSEADDR");
    close_fd(fd);
    return make_unexpected(std::move(status));
  }

  if (endpoint.family() == AddressFamily::kIpv6) {
    const int ipv6_only = options.ipv6_only ? 1 : 0;
#if HYLE_OS_WINDOWS
    if (::setsockopt(static_cast<SOCKET>(fd), IPPROTO_IPV6, IPV6_V6ONLY,
                     reinterpret_cast<const char*>(&ipv6_only),
                     sizeof(ipv6_only)) != 0) {
#else
    if (::setsockopt(static_cast<int>(fd), IPPROTO_IPV6, IPV6_V6ONLY,
                     &ipv6_only, sizeof(ipv6_only)) != 0) {
#endif
      Status status = socket_error("set IPV6_V6ONLY");
      close_fd(fd);
      return make_unexpected(std::move(status));
    }
  }

  if (options.reuse_port) {
#if defined(SO_REUSEPORT)
    const int one = 1;
#  if HYLE_OS_WINDOWS
    const int result = ::setsockopt(
        static_cast<SOCKET>(fd), SOL_SOCKET, SO_REUSEPORT,
        reinterpret_cast<const char*>(&one), sizeof(one));
#  else
    const int result = ::setsockopt(static_cast<int>(fd), SOL_SOCKET,
                                    SO_REUSEPORT, &one, sizeof(one));
#  endif
    if (result != 0) {
      Status status = socket_error("set SO_REUSEPORT");
      close_fd(fd);
      return make_unexpected(std::move(status));
    }
#else
    close_fd(fd);
    return make_unexpected(Status::Unimplemented(
        "reuse-port is not supported by this operating system"));
#endif
  }

#if HYLE_OS_WINDOWS
  if (::bind(static_cast<SOCKET>(fd),
             reinterpret_cast<const sockaddr*>(native->storage),
             static_cast<int>(native->size)) != 0 ||
      ::listen(static_cast<SOCKET>(fd), options.backlog) != 0) {
#else
  if (::bind(static_cast<int>(fd),
             reinterpret_cast<const sockaddr*>(native->storage),
             static_cast<socklen_t>(native->size)) != 0 ||
      ::listen(static_cast<int>(fd), options.backlog) != 0) {
#endif
    Status status = socket_error("bind/listen TCP socket");
    close_fd(fd);
    return make_unexpected(std::move(status));
  }
  if (!detail::IoContextAccess::bind_handle(io, owner_worker, fd)) {
    Status status = handle_error("associate TCP listener with IO shard");
    close_fd(fd);
    return make_unexpected(std::move(status));
  }
  return TcpListener{io, fd, owner_worker};
}

Result<Endpoint> TcpListener::local_endpoint() const noexcept {
  if (!valid())
    return make_unexpected(Status::FailedPrecondition("TCP listener is not open"));
  return query_endpoint(fd_, false);
}

Result<TcpStream> TcpListener::AcceptAwaitable::await_resume() noexcept {
  return self->finish_accept(inner.await_resume(), assign_owner);
}

Result<TcpStream> TcpListener::finish_accept(std::int64_t result,
                                             unsigned assign_owner) noexcept {
  if (result < 0) {
    return make_unexpected(
        detail::transfer_from_native(result, detail::kNetworkNativeDomain,
                                     "TCP accept", false)
            .status);
  }
  std::intptr_t new_fd = static_cast<std::intptr_t>(result);
#if HYLE_OS_MACOS
  const int no_sigpipe = 1;
  if (::setsockopt(static_cast<int>(new_fd), SOL_SOCKET, SO_NOSIGPIPE,
                   &no_sigpipe, sizeof(no_sigpipe)) != 0) {
    close_fd(new_fd);
    return make_unexpected(
        socket_error("set SO_NOSIGPIPE on accepted TCP socket"));
  }
#endif
  // Bind the accepted fd to its assigned owner's ring (IOCP association; no-op on
  // io_uring/epoll). If binding fails, drop the fd and return invalid.
  if (!detail::IoContextAccess::bind_handle(*io_, assign_owner, new_fd)) {
    Status status = handle_error("associate accepted TCP socket with IO shard");
    close_fd(new_fd);
    return make_unexpected(std::move(status));
  }
  return TcpStream{*io_, new_fd, assign_owner};
}

bool TcpStream::ReadBuffersAwaitable::await_ready() noexcept {
  if (!buffers.ok()) {
    immediate.status = buffers.status();
    immediate_ready = true;
    return true;
  }
  if (buffers.empty()) {
    immediate.progress = buffers.progress(0);
    immediate_ready = true;
    return true;
  }
  inner.proactor =
      &detail::IoContextAccess::proactor_for_current(*self->io_);
  inner.fd = self->fd_;
  inner.buffers = buffers.native_data();
  inner.count = static_cast<std::uint32_t>(buffers.native_count());
  inner.allocator = detail::IoContextAccess::operation_allocator(*self->io_);
  inner.options = options;
  return inner.await_ready();
}

BufferSequenceResult TcpStream::ReadBuffersAwaitable::await_resume() noexcept {
  if (immediate_ready) return std::move(immediate);
  TransferResult transfer = detail::transfer_from_native(
      inner.await_resume(), detail::kNetworkNativeDomain, "TCP read", true);
  if (transfer.bytes > buffers.total_size()) {
    return BufferSequenceResult{
        {}, Status::Internal("TCP vectored read exceeded requested bytes"),
        false};
  }
  return BufferSequenceResult{buffers.progress(transfer.bytes),
                              std::move(transfer.status), transfer.eof};
}

bool TcpStream::WriteBuffersAwaitable::await_ready() noexcept {
  if (!buffers.ok()) {
    immediate.status = buffers.status();
    immediate_ready = true;
    return true;
  }
  if (buffers.empty()) {
    immediate.progress = buffers.progress(0);
    immediate_ready = true;
    return true;
  }
  inner.proactor =
      &detail::IoContextAccess::proactor_for_current(*self->io_);
  inner.fd = self->fd_;
  inner.buffers = buffers.native_data();
  inner.count = static_cast<std::uint32_t>(buffers.native_count());
  inner.allocator = detail::IoContextAccess::operation_allocator(*self->io_);
  inner.options = options;
  return inner.await_ready();
}

BufferSequenceResult TcpStream::WriteBuffersAwaitable::await_resume() noexcept {
  if (immediate_ready) return std::move(immediate);
  TransferResult transfer = detail::transfer_from_native(
      inner.await_resume(), detail::kNetworkNativeDomain, "TCP write", false);
  if (transfer.bytes > buffers.total_size()) {
    return BufferSequenceResult{
        {}, Status::Internal("TCP vectored write exceeded requested bytes"),
        false};
  }
  return BufferSequenceResult{buffers.progress(transfer.bytes),
                              std::move(transfer.status), false};
}

sched::Task<TransferResult> read_exact(mem::AllocatorRef, TcpStream& stream,
                                      std::span<std::byte> buffer,
                                      OperationOptions options) noexcept {
  std::size_t done = 0;
  while (done < buffer.size()) {
    TransferResult part =
        co_await stream.read_some(buffer.subspan(done), options);
    done += part.bytes;
    if (!part.ok() || part.eof) {
      part.bytes = done;
      co_return part;
    }
    if (part.bytes == 0) {
      co_return TransferResult{
          done, Status::Unavailable("TCP read made no forward progress"), false};
    }
  }
  co_return TransferResult{done, Status{}, false};
}

sched::Task<TransferResult> write_all(mem::AllocatorRef, TcpStream& stream,
                                     std::span<const std::byte> buffer,
                                     OperationOptions options) noexcept {
  std::size_t done = 0;
  while (done < buffer.size()) {
    TransferResult part =
        co_await stream.write_some(buffer.subspan(done), options);
    done += part.bytes;
    if (!part.ok()) {
      part.bytes = done;
      co_return part;
    }
    if (part.bytes == 0) {
      co_return TransferResult{
          done, Status::Unavailable("TCP write made no forward progress"), false};
    }
  }
  co_return TransferResult{done, Status{}, false};
}

namespace {

sched::Task<BufferSequenceResult> read_exact_buffers(
    mem::AllocatorRef, TcpStream& stream,
    detail::BufferSequenceStorage<MutableBuffer> buffers,
    OperationOptions options) noexcept {
  if (!buffers.ok())
    co_return BufferSequenceResult{{}, buffers.status(), false};
  BufferSequenceProgress progress = buffers.progress(0);
  while (!progress.complete) {
    BufferSequenceResult part =
        co_await stream.read_some(buffers.sequence(), options, progress.next);
    progress.bytes += part.progress.bytes;
    progress.next = part.progress.next;
    progress.complete = part.progress.complete;
    if (!part.ok()) {
      part.progress = progress;
      co_return part;
    }
    if (part.eof) {
      co_return BufferSequenceResult{
          progress,
          Status::DataLoss("EOF before exact TCP vectored read completed"),
          true};
    }
    if (part.progress.bytes == 0) {
      co_return BufferSequenceResult{
          progress,
          Status::Unavailable("TCP vectored read made no forward progress"),
          false};
    }
  }
  co_return BufferSequenceResult{progress, Status{}, false};
}

sched::Task<BufferSequenceResult> write_all_buffers(
    mem::AllocatorRef, TcpStream& stream,
    detail::BufferSequenceStorage<ConstBuffer> buffers,
    OperationOptions options) noexcept {
  if (!buffers.ok())
    co_return BufferSequenceResult{{}, buffers.status(), false};
  BufferSequenceProgress progress = buffers.progress(0);
  while (!progress.complete) {
    BufferSequenceResult part =
        co_await stream.write_some(buffers.sequence(), options, progress.next);
    progress.bytes += part.progress.bytes;
    progress.next = part.progress.next;
    progress.complete = part.progress.complete;
    if (!part.ok()) {
      part.progress = progress;
      co_return part;
    }
    if (part.progress.bytes == 0) {
      co_return BufferSequenceResult{
          progress,
          Status::Unavailable("TCP vectored write made no forward progress"),
          false};
    }
  }
  co_return BufferSequenceResult{progress, Status{}, false};
}

}  // namespace

sched::Task<BufferSequenceResult> read_exact(
    mem::AllocatorRef coroutine_allocator, TcpStream& stream,
    std::span<const MutableBuffer> buffers, OperationOptions options) noexcept {
  return read_exact_buffers(
      coroutine_allocator, stream,
      detail::BufferSequenceStorage<MutableBuffer>(buffers), options);
}

sched::Task<BufferSequenceResult> write_all(
    mem::AllocatorRef coroutine_allocator, TcpStream& stream,
    std::span<const ConstBuffer> buffers, OperationOptions options) noexcept {
  return write_all_buffers(
      coroutine_allocator, stream,
      detail::BufferSequenceStorage<ConstBuffer>(buffers), options);
}

sched::Task<Result<TcpStream>> connect_tcp(
    mem::AllocatorRef coroutine_allocator, IoContext& io,
    unsigned owner_worker, std::span<const Endpoint> endpoints,
    TcpConnectOptions options) noexcept {
  Result<EndpointList> ordered =
      interleave_endpoints(coroutine_allocator, endpoints);
  if (!ordered.has_value())
    return connect_error_task(coroutine_allocator,
                              std::move(ordered).error());
  return connect_ordered(coroutine_allocator, io, owner_worker,
                         std::move(*ordered), options);
}

sched::Task<Result<TcpStream>> connect_tcp(
    mem::AllocatorRef coroutine_allocator, IoContext& io,
    unsigned owner_worker, std::string_view host, std::uint16_t port,
    TcpConnectOptions options) noexcept {
  if (!coroutine_allocator.valid())
    return connect_error_task(
        coroutine_allocator,
        Status::InvalidArgument("TCP connect requires a complete allocator"));
  if (host.empty() || host.find('\0') != std::string_view::npos)
    return connect_error_task(
        coroutine_allocator,
        Status::InvalidArgument("TCP connect requires one non-empty host name"));
  OwnedHost owned(coroutine_allocator);
  if (!owned.bytes.try_reserve(host.size()))
    return connect_error_task(
        coroutine_allocator,
        Status::ResourceExhausted("TCP host name allocation failed"));
  for (char byte : host) {
    if (!owned.bytes.try_push_back(byte))
      return connect_error_task(
          coroutine_allocator,
          Status::ResourceExhausted("TCP host name allocation failed"));
  }
  return connect_host(coroutine_allocator, io, owner_worker,
                      std::move(owned), port, options);
}

}  // namespace hyle::io
