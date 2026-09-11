#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "hyle/compute/visibility.h"
#include "hyle/compute/provider.h"
#include "hyle/error/result.h"
#include "hyle/mem/allocator.h"

namespace hyle::compute {

enum class BackendKind : std::uint8_t {
  kUnknown = 0,
  kCpu,
  kCuda,
  kMetal,
  kVulkan,
  kHip,
};

HYLE_COMPUTE_API std::string_view backend_name(BackendKind backend) noexcept;

// Cheap, non-owning identity. A backend appears here only as a vocabulary
// value; Device factories/enumeration expose only implementations compiled
// into the selected product.
class Device final {
public:
  constexpr Device() noexcept = default;

  static constexpr Device cpu() noexcept {
    return Device(BackendKind::kCpu, 0);
  }

  constexpr bool valid() const noexcept {
    return backend_ != BackendKind::kUnknown;
  }
  constexpr explicit operator bool() const noexcept { return valid(); }
  constexpr BackendKind backend() const noexcept { return backend_; }
  constexpr std::uint32_t ordinal() const noexcept { return ordinal_; }

  friend constexpr bool operator==(Device, Device) noexcept = default;

private:
  constexpr Device(BackendKind backend, std::uint32_t ordinal) noexcept
      : backend_(backend), ordinal_(ordinal) {}

  BackendKind backend_ = BackendKind::kUnknown;
  std::uint32_t ordinal_ = 0;
};

struct DeviceProperties final {
  Device device{};
  std::string_view name{};
  bool unified_memory = false;
  bool asynchronous_queues = false;
};

// Buffer-fill discovery: returns the full count and writes at most
// output.size() entries. An empty span is a count-only query, so truncation is
// always visible as returned_count > output.size().
HYLE_COMPUTE_API Result<std::size_t>
enumerate_devices(std::span<Device> output = {}) noexcept;
HYLE_COMPUTE_API Result<DeviceProperties> query_device(Device device) noexcept;

struct ContextOptions final {
  // Empty selects mem::system_allocator(). This allocation is cold context
  // metadata, never a submit-path fallback. A custom allocator backing must
  // outlive the Context and every Queue/PreparedOperation/Graph Executable or
  // Invocation whose metadata was created from it.
  mem::AllocatorRef metadata_allocator{};
  // Ordered, explicit cold-path provider choices. Only Unimplemented advances
  // to the next entry; HYLE-native is the unconditional final fallback.
  ProviderList providers{};
};

struct QueueOptions final {};

namespace detail {
struct ContextState;
struct ContextAccess;
} // namespace detail

class Queue;
class PreparedOperation;
class OperationInvocation;

// Owns one backend runtime/session. Destroying or closing the owner stops new
// submissions through all borrowed/retained handles. A retained prepared
// artifact may still create its per-run state, but that state cannot submit
// after the Context is closed. The current CPU provider has no asynchronous
// in-flight work to drain and never hides a wait here. close()/destruction must
// not race submission through another handle; ordinary submissions on distinct
// queues may run concurrently.
class HYLE_COMPUTE_API Context final {
public:
  Context() noexcept = default;
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;
  Context(Context &&other) noexcept;
  Context &operator=(Context &&other) noexcept;
  ~Context();

  static Result<Context> create(Device device,
                                ContextOptions options = {}) noexcept;
  // Natural single-provider convenience. Use ContextOptions when provider
  // order or a metadata allocator must be configured explicitly.
  static Result<Context> create(Device device, Provider provider) noexcept;

  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  bool is_open() const noexcept;
  Device device() const noexcept;
  mem::AllocatorRef metadata_allocator() const noexcept;
  Result<Queue> create_queue(QueueOptions options = {}) const noexcept;
  Status close() noexcept;

private:
  explicit Context(detail::ContextState *state) noexcept : state_(state) {}

  friend struct detail::ContextAccess;
  detail::ContextState *state_ = nullptr;
};

// Move-only ordered submission domain. The current CPU provider executes on the
// caller thread; this is a semantic queue, not a hidden worker or process-global
// scheduler.
class HYLE_COMPUTE_API Queue final {
public:
  Queue() noexcept = default;
  Queue(const Queue &) = delete;
  Queue &operator=(const Queue &) = delete;
  Queue(Queue &&other) noexcept;
  Queue &operator=(Queue &&other) noexcept;
  ~Queue();

  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  bool is_open() const noexcept;
  Device device() const noexcept;
  Status close() noexcept;

private:
  explicit Queue(detail::ContextState *state) noexcept
      : state_(state), open_(true) {}

  friend class Context;
  friend class PreparedOperation;
  friend class OperationInvocation;
  friend struct detail::ContextAccess;
  detail::ContextState *state_ = nullptr;
  bool open_ = false;
};

} // namespace hyle::compute
