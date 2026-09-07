#include "hyle/io/dns.h"

#include <string_view>

#include "hyle/core/environment.h"
#include "endpoint_native.h"

#if HYLE_OS_WINDOWS
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  include <windows.h>
#else
#  include <netdb.h>
#  include <netinet/in.h>
#  include <sys/socket.h>
#  include <cstdio>
#endif

namespace hyle::io {
namespace {

Status resolver_status(int code) noexcept {
#if HYLE_OS_WINDOWS
  return status_from_winsock(code, "resolve host");
#else
  StatusCode canonical = StatusCode::Unknown;
  switch (code) {
    case EAI_AGAIN:
      canonical = StatusCode::Unavailable;
      break;
    case EAI_NONAME:
      canonical = StatusCode::NotFound;
      break;
    case EAI_MEMORY:
      canonical = StatusCode::ResourceExhausted;
      break;
    case EAI_BADFLAGS:
    case EAI_FAMILY:
    case EAI_SOCKTYPE:
    case EAI_SERVICE:
      canonical = StatusCode::InvalidArgument;
      break;
    default:
      canonical = StatusCode::Unknown;
      break;
  }
  const char* message = ::gai_strerror(code);
  return Status(canonical,
                message != nullptr ? std::string_view(message)
                                   : std::string_view("name resolution failed"));
#endif
}

struct ResolveJob {
  explicit ResolveJob(mem::AllocatorRef allocator) noexcept
      : allocator(allocator), host(allocator) {}

  ResolveJob(ResolveJob&&) noexcept = default;
  ResolveJob& operator=(ResolveJob&&) noexcept = default;
  ResolveJob(const ResolveJob&) = delete;
  ResolveJob& operator=(const ResolveJob&) = delete;

  detail::ResolveOutcome operator()() const noexcept {
    return detail::resolve_blocking(allocator, host.data(), port);
  }

  mem::AllocatorRef allocator;
  container::SmallVector<char, 256, mem::AllocatorRef> host;
  std::uint16_t port = 0;
};

}  // namespace

namespace detail {

ResolveOutcome resolve_blocking(mem::AllocatorRef allocator, const char* host,
                                std::uint16_t port) noexcept {
  ResolveOutcome outcome(allocator);
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  char service[8]{};
#if HYLE_OS_WINDOWS
  ::_itoa_s(static_cast<int>(port), service, sizeof(service), 10);
#else
  std::snprintf(service, sizeof(service), "%u", static_cast<unsigned>(port));
#endif

  addrinfo* addresses = nullptr;
  const int result = ::getaddrinfo(host, service, &hints, &addresses);
  if (result != 0) {
    outcome.status = resolver_status(result);
    return outcome;
  }

  for (addrinfo* current = addresses; current != nullptr;
       current = current->ai_next) {
    if (current->ai_family != AF_INET && current->ai_family != AF_INET6)
      continue;
    auto endpoint = EndpointNativeAccess::decode(
        current->ai_addr, static_cast<std::uint32_t>(current->ai_addrlen));
    if (!endpoint.has_value()) continue;
    if (!outcome.endpoints.try_push_back(std::move(*endpoint))) {
      outcome.status =
          Status::ResourceExhausted("resolved endpoint list allocation failed");
      break;
    }
  }
  ::freeaddrinfo(addresses);
  if (outcome.status.ok() && outcome.endpoints.empty())
    outcome.status = Status::NotFound("resolver returned no IPv4/IPv6 endpoint");
  return outcome;
}

}  // namespace detail

Result<EndpointList> ResolveHandle::join() noexcept {
  auto joined = handle_.join();
  if (!joined.has_value()) return make_unexpected(joined.error());
  detail::ResolveOutcome outcome = std::move(*joined);
  if (!outcome.status.ok()) return make_unexpected(std::move(outcome.status));
  return std::move(outcome.endpoints);
}

ResolveHandle resolve(sched::BlockingPool& pool, mem::AllocatorRef allocator,
                      std::string_view host, std::uint16_t port) noexcept {
  if (!allocator.valid()) {
    return ResolveHandle(sched::JoinHandle<detail::ResolveOutcome>(
        Status::InvalidArgument("resolve requires a complete allocator")));
  }
  if (host.empty() || host.find('\0') != std::string_view::npos) {
    return ResolveHandle(sched::JoinHandle<detail::ResolveOutcome>(
        Status::InvalidArgument("resolve requires one non-empty host name")));
  }

  ResolveJob job(allocator);
  if (!job.host.try_reserve(host.size() + 1)) {
    return ResolveHandle(sched::JoinHandle<detail::ResolveOutcome>(
        Status::ResourceExhausted("host name allocation failed")));
  }
  for (char byte : host) {
    if (!job.host.try_push_back(byte)) {
      return ResolveHandle(sched::JoinHandle<detail::ResolveOutcome>(
          Status::ResourceExhausted("host name allocation failed")));
    }
  }
  if (!job.host.try_push_back('\0')) {
    return ResolveHandle(sched::JoinHandle<detail::ResolveOutcome>(
        Status::ResourceExhausted("host name allocation failed")));
  }
  job.port = port;
  return ResolveHandle(sched::spawn_blocking(pool, allocator, std::move(job)));
}

}  // namespace hyle::io
