#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "hyle/compute/provider.h"
#include "hyle/compute/visibility.h"
#include "hyle/error/result.h"
#include "hyle/mem/allocator.h"

namespace hyle::compute {

struct BackendId final {
  std::uint64_t namespace_id = 0;
  std::uint64_t backend_id = 0;

  constexpr bool valid() const noexcept {
    return namespace_id != 0 || backend_id != 0;
  }
  friend constexpr bool operator==(BackendId, BackendId) noexcept = default;
};

enum class DeviceClass : std::uint8_t {
  kUnknown = 0,
  kCpu,
  kGpu,
  kAccelerator,
};

struct DeviceProperties final {
  BackendId backend{};
  DeviceClass device_class = DeviceClass::kUnknown;
  std::string_view backend_name{};
  std::string_view name{};
  bool unified_memory = false;
  bool asynchronous_queues = false;
};

class Backend;
class Device;
class Context;
class Queue;
class PreparedOperation;
class OperationInvocation;

namespace detail {
struct BackendAccess;
struct BackendVTable;
struct ContextAccess;
struct ContextState;
struct DeviceAccess;
struct ProviderExtensionAccess;
struct QueueState;
} // namespace detail

// Explicit backend factory value. Backends enumerate only their own devices;
// HYLE has no process-global registry and selecting a product does not scan
// every compiled implementation.
class Backend final {
public:
  constexpr Backend() noexcept = default;

  constexpr bool valid() const noexcept { return vtable_ != nullptr; }
  constexpr explicit operator bool() const noexcept { return valid(); }
  HYLE_COMPUTE_API BackendId id() const noexcept;
  HYLE_COMPUTE_API std::string_view name() const noexcept;
  HYLE_COMPUTE_API DeviceClass device_class() const noexcept;
  HYLE_COMPUTE_API Result<std::size_t> device_count() const noexcept;
  HYLE_COMPUTE_API Result<Device> device(std::size_t index) const noexcept;

private:
  friend struct detail::BackendAccess;
  static constexpr std::size_t kInlineBytes = sizeof(void *) * 4;

  const detail::BackendVTable *vtable_ = nullptr;
  alignas(void *) std::array<std::byte, kInlineBytes> payload_{};
};

// Small copyable process-local device identity. The inline payload is owned by
// its typed backend factory; querying/copying a Device never allocates.
class Device final {
public:
  constexpr Device() noexcept = default;

  constexpr bool valid() const noexcept { return vtable_ != nullptr; }
  constexpr explicit operator bool() const noexcept { return valid(); }
  HYLE_COMPUTE_API BackendId backend_id() const noexcept;
  HYLE_COMPUTE_API std::string_view backend_name() const noexcept;
  HYLE_COMPUTE_API DeviceClass device_class() const noexcept;
  HYLE_COMPUTE_API Result<DeviceProperties> properties() const noexcept;

  friend HYLE_COMPUTE_API bool operator==(const Device &left,
                                          const Device &right) noexcept;

private:
  friend struct detail::DeviceAccess;
  static constexpr std::size_t kInlineBytes = sizeof(void *) * 6;

  const detail::BackendVTable *vtable_ = nullptr;
  alignas(void *) std::array<std::byte, kInlineBytes> payload_{};
};

struct ContextOptions final {
  // Empty selects mem::system_allocator(). This allocation is cold context
  // metadata, never a submit-path fallback. The backing must outlive Context
  // and every retained Queue/artifact/invocation created from it.
  mem::AllocatorRef metadata_allocator{};
  // Borrowed only for create(). Context copies the exact sequence into one
  // cold allocation. An empty sequence is valid for raw backend/host work but
  // cannot prepare an Operation.
  std::span<const Provider> providers{};
};

struct QueueOptions final {};

// Owns one backend session. close() stops admission but never hides a wait;
// drain() explicitly waits backend work accepted before its synchronization
// point. Queue and Completion retain the backend state they require.
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
  static Result<Context> create(Device device, Provider provider) noexcept;

  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  bool is_open() const noexcept;
  Device device() const noexcept;
  mem::AllocatorRef metadata_allocator() const noexcept;
  std::size_t provider_count() const noexcept;
  Result<ProviderIdentity> provider_identity(std::size_t index) const noexcept;
  Result<Queue> create_queue(QueueOptions options = {}) const noexcept;
  Status close() noexcept;
  Status drain() noexcept;

private:
  explicit Context(detail::ContextState *state) noexcept : state_(state) {}

  friend struct detail::ContextAccess;
  detail::ContextState *state_ = nullptr;
};

// Move-only ordered submission domain. close() is local to this Queue;
// Context::close() stops every queue in the context. Destruction closes and
// releases without waiting; call drain() when quiescence is required.
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
  Status drain() noexcept;

private:
  explicit Queue(detail::QueueState *state) noexcept : state_(state) {}

  Status begin_submission() noexcept;
  void end_submission() noexcept;

  friend class Context;
  friend class OperationInvocation;
  friend struct detail::ContextAccess;
  friend struct detail::ProviderExtensionAccess;
  friend struct CpuQueueAccess;
  detail::QueueState *state_ = nullptr;
};

// Private implementation bridge used by the typed CPU header. It is named in
// the public C++ binary surface only because the callable itself remains in the
// consumer TU; users do not construct this type.
struct CpuQueueAccess final {
  static Status begin(Queue &queue) noexcept {
    return queue.begin_submission();
  }
  static void end(Queue &queue) noexcept { queue.end_submission(); }
};

} // namespace hyle::compute
