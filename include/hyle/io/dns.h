#pragma once

#include <cstdint>
#include <string_view>
#include <utility>

#include "hyle/container/small_vector.h"
#include "hyle/error/result.h"
#include "hyle/io/endpoint.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/blocking.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_IO_API
#elif defined(HYLE_IO_CORE_BUILD)
#  define HYLE_IO_API HYLE_EXPORT
#else
#  define HYLE_IO_API HYLE_IMPORT
#endif

namespace hyle::io {

// Most host names resolve to one or two endpoints, so eight inline slots avoid
// an allocation on the common path without imposing the old hard eight-address
// truncation. Growth uses the allocator explicitly supplied to resolve().
using EndpointList =
    container::SmallVector<Endpoint, 8, mem::AllocatorRef>;

namespace detail {

struct ResolveOutcome {
  explicit ResolveOutcome(mem::AllocatorRef allocator) noexcept
      : endpoints(allocator) {}

  EndpointList endpoints;
  Status status;
};

HYLE_IO_API ResolveOutcome resolve_blocking(
    mem::AllocatorRef allocator, const char* host,
    std::uint16_t port) noexcept;

}  // namespace detail

// Domain-specific wrapper flattens BlockingPool's task failure and resolver's
// own failure into one Result<EndpointList>. Cancellation is best effort: a job
// not yet started is cancelled; portable getaddrinfo cannot be force-stopped
// once inside the OS, so an already-running job completes and releases itself.
class ResolveHandle {
 public:
  ResolveHandle() noexcept = default;
  explicit ResolveHandle(sched::JoinHandle<detail::ResolveOutcome> handle) noexcept
      : handle_(std::move(handle)) {}

  ResolveHandle(ResolveHandle&&) noexcept = default;
  ResolveHandle& operator=(ResolveHandle&&) noexcept = default;
  ResolveHandle(const ResolveHandle&) = delete;
  ResolveHandle& operator=(const ResolveHandle&) = delete;

  HYLE_IO_API Result<EndpointList> join() noexcept;
  void request_cancel() noexcept { handle_.request_cancel(); }
  bool is_finished() const noexcept { return handle_.is_finished(); }
  bool valid() const noexcept { return handle_.valid(); }
  void detach() noexcept { handle_.detach(); }

 private:
  sched::JoinHandle<detail::ResolveOutcome> handle_;
};

// Copy `host` before returning, then resolve it on the blocking pool so an NSS,
// hosts-file, or network lookup never blocks a scheduler worker. Host names are
// not truncated and all returned IPv4/IPv6 endpoints are retained until an
// allocator failure is reported.
HYLE_IO_API ResolveHandle resolve(sched::BlockingPool& pool,
                                  mem::AllocatorRef allocator,
                                  std::string_view host,
                                  std::uint16_t port) noexcept;

}  // namespace hyle::io
