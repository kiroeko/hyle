#pragma once

#include <cstdint>
#include <string_view>
#include <utility>

#include "hyle/container/small_vector.h"
#include "hyle/error/result.h"
#include "hyle/io/endpoint.h"
#include "hyle/io/operation.h"
#include "hyle/mem/allocator.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_IO_API
#elif defined(HYLE_IO_CORE_BUILD)
#  define HYLE_IO_API HYLE_EXPORT
#else
#  define HYLE_IO_API HYLE_IMPORT
#endif

namespace hyle::io {

class IoContext;
class ResolveHandle;

// Copy `host` before returning, then resolve it on IoContext's blocking pool so
// NSS, hosts-file, or network lookup never blocks a scheduler worker. Host names
// are not truncated and all returned IPv4/IPv6 endpoints are retained until an
// allocator failure is reported.
HYLE_IO_API ResolveHandle resolve(IoContext& io,
                                  mem::AllocatorRef allocator,
                                  std::string_view host,
                                  std::uint16_t port) noexcept;

// Most host names resolve to one or two endpoints, so eight inline slots avoid
// an allocation on the common path without imposing the old hard eight-address
// truncation. Growth uses the allocator explicitly supplied to resolve().
using EndpointList =
    container::SmallVector<Endpoint, 8, mem::AllocatorRef>;

namespace detail {
struct ResolveAccess;
struct ResolveState;
}  // namespace detail

// Owning result handle for one asynchronous name lookup. join() is fiber-aware
// when called on a Scheduler worker and parks a foreign thread otherwise. A
// deadline or cancellation returns promptly; portable getaddrinfo cannot be
// force-stopped once inside the OS, so an already-running lookup safely finishes
// against its own state after the caller has returned.
class ResolveHandle {
 public:
  ResolveHandle() noexcept = default;
  HYLE_IO_API ~ResolveHandle();
  HYLE_IO_API ResolveHandle(ResolveHandle&& other) noexcept;
  HYLE_IO_API ResolveHandle& operator=(ResolveHandle&& other) noexcept;
  ResolveHandle(const ResolveHandle&) = delete;
  ResolveHandle& operator=(const ResolveHandle&) = delete;

  HYLE_IO_API Result<EndpointList> join(
      OperationOptions options = {}) noexcept;
  HYLE_IO_API void request_cancel() noexcept;
  HYLE_IO_API bool is_finished() const noexcept;
  HYLE_IO_API bool valid() const noexcept;
  HYLE_IO_API void detach() noexcept;

 private:
  friend struct detail::ResolveAccess;
  explicit ResolveHandle(detail::ResolveState* state) noexcept : state_(state) {}
  explicit ResolveHandle(Status status) noexcept
      : immediate_(std::move(status)) {}

  detail::ResolveState* state_ = nullptr;
  Status immediate_;
};

}  // namespace hyle::io
