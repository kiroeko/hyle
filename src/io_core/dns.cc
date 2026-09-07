#include "hyle/io/dns.h"

#include <atomic>
#include <cstdio>
#include <new>
#include <string_view>
#include <utility>

#include "hyle/core/environment.h"
#include "hyle/io/io_context.h"
#include "hyle/sched/blocking.h"
#include "hyle/sched/cancel.h"
#include "hyle/sched/sync.h"
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
#endif

namespace hyle::io {
namespace {

struct ResolveOutcome {
  explicit ResolveOutcome(mem::AllocatorRef allocator) noexcept
      : endpoints(allocator) {}

  EndpointList endpoints;
  Status status;
};

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

}  // namespace

namespace detail {

// One allocation owns the copied host, result, completion event and the
// BlockingPool job header. The handle and pool each retain a reference. This is
// what makes an early deadline/cancellation safe even though portable
// getaddrinfo cannot be interrupted after entering the operating system.
struct ResolveState {
  explicit ResolveState(mem::AllocatorRef source_allocator) noexcept
      : allocator(source_allocator),
        host(source_allocator),
        outcome(source_allocator),
        ready(sched::EventResetMode::kManual) {
    job.ctx = this;
    job.run_fn = &ResolveState::run_job;
    job.cancel_fn = &ResolveState::cancel_job;
    job.destroy_fn = &ResolveState::destroy_job;
  }

  void retain() noexcept { refs.fetch_add(1, std::memory_order_relaxed); }

  void release() noexcept {
    if (refs.fetch_sub(1, std::memory_order_acq_rel) != 1) return;
    mem::AllocatorRef source_allocator = allocator;
    this->~ResolveState();
    source_allocator.deallocate(this, sizeof(ResolveState),
                                alignof(ResolveState));
  }

  void publish(ResolveOutcome resolved) noexcept {
    outcome = std::move(resolved);
    completed.store(true, std::memory_order_release);
    ready.signal();
  }

  static void run_job(void* context) noexcept {
    auto* state = static_cast<ResolveState*>(context);
    if (state->cancel_requested.load(std::memory_order_acquire)) {
      ResolveOutcome cancelled(state->allocator);
      cancelled.status =
          Status::Cancelled("name resolution cancelled before start");
      state->publish(std::move(cancelled));
      return;
    }
    state->publish(
        resolve_blocking(state->allocator, state->host.data(), state->port));
  }

  static void cancel_job(void* context) noexcept {
    auto* state = static_cast<ResolveState*>(context);
    ResolveOutcome cancelled(state->allocator);
    cancelled.status = Status::Cancelled("resolver blocking pool shut down");
    state->publish(std::move(cancelled));
  }

  static void destroy_job(void* context) noexcept {
    static_cast<ResolveState*>(context)->release();
  }

  sched::detail::BlockingJob job{};
  mem::AllocatorRef allocator;
  container::SmallVector<char, 256, mem::AllocatorRef> host;
  std::uint16_t port = 0;
  ResolveOutcome outcome;
  sched::Event ready;
  std::atomic<unsigned> refs{1};
  std::atomic<bool> completed{false};
  std::atomic<bool> cancel_requested{false};
};

struct ResolveAccess {
  static ResolveHandle from_state(ResolveState* state) noexcept {
    return ResolveHandle(state);
  }

  static ResolveHandle from_status(Status status) noexcept {
    return ResolveHandle(std::move(status));
  }
};

}  // namespace detail

namespace {

void cancel_resolve_wait(void* context) noexcept {
  auto* state = static_cast<detail::ResolveState*>(context);
  state->cancel_requested.store(true, std::memory_order_release);
  state->ready.signal();
}

}  // namespace

ResolveHandle::~ResolveHandle() { detach(); }

ResolveHandle::ResolveHandle(ResolveHandle&& other) noexcept
    : state_(std::exchange(other.state_, nullptr)),
      immediate_(std::move(other.immediate_)) {
  other.immediate_ = Status{};
}

ResolveHandle& ResolveHandle::operator=(ResolveHandle&& other) noexcept {
  if (this == &other) return *this;
  detach();
  state_ = std::exchange(other.state_, nullptr);
  immediate_ = std::move(other.immediate_);
  other.immediate_ = Status{};
  return *this;
}

Result<EndpointList> ResolveHandle::join(OperationOptions options) noexcept {
  if (state_ == nullptr) {
    if (immediate_.ok())
      return make_unexpected(
          Status::FailedPrecondition("resolver handle is not joinable"));
    Status status = std::move(immediate_);
    immediate_ = Status{};
    return make_unexpected(std::move(status));
  }

  auto consume = [this]() noexcept -> Result<EndpointList> {
    detail::ResolveState* state = std::exchange(state_, nullptr);
    Status status = std::move(state->outcome.status);
    if (!status.ok()) {
      state->release();
      return make_unexpected(std::move(status));
    }
    EndpointList endpoints = std::move(state->outcome.endpoints);
    state->release();
    return endpoints;
  };

  if (state_->completed.load(std::memory_order_acquire)) return consume();

  const bool externally_cancelled =
      options.cancel != nullptr && options.cancel->is_cancelled();
  if (externally_cancelled ||
      state_->cancel_requested.load(std::memory_order_acquire)) {
    request_cancel();
    detach();
    return make_unexpected(Status::Cancelled("name resolution cancelled"));
  }
  if (options.deadline.has_value() &&
      *options.deadline <= time::monotonic_now()) {
    request_cancel();
    detach();
    return make_unexpected(
        Status::DeadlineExceeded("name resolution deadline exceeded"));
  }

  bool signaled = true;
  if (options.cancel != nullptr) {
    sched::CancelCallback callback(*options.cancel, &cancel_resolve_wait,
                                   state_);
    if (options.deadline.has_value())
      signaled = state_->ready.wait_until(*options.deadline);
    else
      state_->ready.wait();
    callback.unregister();
  } else if (options.deadline.has_value()) {
    signaled = state_->ready.wait_until(*options.deadline);
  } else {
    state_->ready.wait();
  }

  if (state_->completed.load(std::memory_order_acquire)) return consume();

  const bool cancelled =
      state_->cancel_requested.load(std::memory_order_acquire) ||
      (options.cancel != nullptr && options.cancel->is_cancelled());
  request_cancel();
  detach();
  if (cancelled)
    return make_unexpected(Status::Cancelled("name resolution cancelled"));
  if (!signaled)
    return make_unexpected(
        Status::DeadlineExceeded("name resolution deadline exceeded"));
  return make_unexpected(Status::Cancelled("name resolution cancelled"));
}

void ResolveHandle::request_cancel() noexcept {
  if (state_ == nullptr) return;
  state_->cancel_requested.store(true, std::memory_order_release);
  state_->ready.signal();
}

bool ResolveHandle::is_finished() const noexcept {
  return state_ != nullptr
             ? state_->completed.load(std::memory_order_acquire)
             : !immediate_.ok();
}

bool ResolveHandle::valid() const noexcept {
  return state_ != nullptr || !immediate_.ok();
}

void ResolveHandle::detach() noexcept {
  if (state_ != nullptr) {
    detail::ResolveState* state = std::exchange(state_, nullptr);
    state->release();
  }
  immediate_ = Status{};
}

ResolveHandle resolve(IoContext& io, mem::AllocatorRef allocator,
                      std::string_view host, std::uint16_t port) noexcept {
  if (!io.valid())
    return detail::ResolveAccess::from_status(
        Status::FailedPrecondition("resolve requires a valid IoContext"));
  if (!allocator.valid())
    return detail::ResolveAccess::from_status(
        Status::InvalidArgument("resolve requires a complete allocator"));
  if (host.empty() || host.find('\0') != std::string_view::npos)
    return detail::ResolveAccess::from_status(Status::InvalidArgument(
        "resolve requires one non-empty host name"));

  void* storage = allocator.allocate(sizeof(detail::ResolveState),
                                     alignof(detail::ResolveState));
  if (storage == nullptr)
    return detail::ResolveAccess::from_status(
        Status::ResourceExhausted("resolver state allocation failed"));
  auto* state = new (storage) detail::ResolveState(allocator);
  if (!state->host.try_reserve(host.size() + 1)) {
    state->release();
    return detail::ResolveAccess::from_status(
        Status::ResourceExhausted("host name allocation failed"));
  }
  for (char byte : host) {
    if (!state->host.try_push_back(byte)) {
      state->release();
      return detail::ResolveAccess::from_status(
          Status::ResourceExhausted("host name allocation failed"));
    }
  }
  if (!state->host.try_push_back('\0')) {
    state->release();
    return detail::ResolveAccess::from_status(
        Status::ResourceExhausted("host name allocation failed"));
  }
  state->port = port;

  state->retain();  // pool-side reference
  if (!io.blocking_pool().submit(&state->job)) {
    detail::ResolveState::cancel_job(state);
    detail::ResolveState::destroy_job(state);
  }
  return detail::ResolveAccess::from_state(state);
}

}  // namespace hyle::io
