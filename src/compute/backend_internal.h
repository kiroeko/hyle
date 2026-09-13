#pragma once

#include <cstring>
#include <type_traits>

#include "hyle/compute/completion.h"
#include "hyle/compute/detail/queue_sequence.h"
#include "hyle/compute/device.h"
#include "hyle/compute/transfer.h"
#include "hyle/mem/resource.h"

namespace hyle::compute::detail {

struct BackendVTable final {
  BackendId id{};
  std::string_view name{};
  DeviceClass device_class = DeviceClass::kUnknown;
  Result<std::size_t> (*device_count)(const Backend &backend) noexcept =
      nullptr;
  Result<Device> (*device)(const Backend &backend,
                           std::size_t index) noexcept = nullptr;
  bool (*same_device)(const Device &left,
                      const Device &right) noexcept = nullptr;
  Result<DeviceProperties> (*properties)(const Device &device) noexcept =
      nullptr;
  Status (*validate_resource)(
      const Device &device, mem::MemoryResourceRef resource) noexcept = nullptr;
  Status (*create_context)(const Device &device, mem::AllocatorRef allocator,
                           void **context) noexcept = nullptr;
  void (*destroy_context)(void *context,
                          mem::AllocatorRef allocator) noexcept = nullptr;
  Status (*create_queue)(void *context, QueueOptions options,
                         mem::AllocatorRef allocator,
                         void **queue) noexcept = nullptr;
  void (*destroy_queue)(void *queue,
                        mem::AllocatorRef allocator) noexcept = nullptr;
  // Every accepted direct submission or Graph Queue segment is recorded
  // inside one sequence and finalized exactly once. begin must reserve all
  // state needed by finish before any provider command can be accepted.
  Status (*begin_sequence)(void *queue, void **sequence) noexcept = nullptr;
  Completion (*finish_sequence)(void *queue, void *sequence, bool accepted,
                                Status terminal) noexcept = nullptr;
  Status (*reserve_completions)(void *queue,
                                std::size_t minimum) noexcept = nullptr;
  Result<TransferProperties> (*transfer_properties)(
      void *context, mem::BufferView destination,
      mem::ConstBufferView source) noexcept = nullptr;
  EnqueueResult (*copy_async)(void *queue, mem::BufferView destination,
                              mem::ConstBufferView source) noexcept = nullptr;
  EnqueueResult (*fill_async)(void *queue, mem::BufferView destination,
                              std::byte value) noexcept = nullptr;
  EnqueueResult (*wait_completion)(void *queue,
                                   const CompletionVTable *completion_vtable,
                                   void *completion_state) noexcept = nullptr;
  // Cold endpoint for completions emitted by this exact Queue. Synchronous
  // queues may return an invalid endpoint because their prefixes need no wait.
  CompletionEndpoint (*completion_endpoint)(void *queue) noexcept = nullptr;
  // Returning true commits wait_completion() to accepting a pending
  // Completion produced by `source`; false lets Graph expose/reject a host
  // wait. The endpoint keeps backend/provider types out of the common ABI.
  bool (*can_wait_endpoint)(void *destination_queue,
                            CompletionEndpoint source) noexcept = nullptr;
  const void *(*query_extension)(void *context,
                                 BackendExtensionId id) noexcept = nullptr;
  Status (*drain_context)(void *context) noexcept = nullptr;
  Status (*drain_queue)(void *queue) noexcept = nullptr;
};

struct BackendAccess final {
  template <class Payload>
    requires(std::is_trivially_copyable_v<Payload> &&
             sizeof(Payload) <= Backend::kInlineBytes &&
             alignof(Payload) <= alignof(void *))
  static Backend make(const BackendVTable *vtable,
                      const Payload &payload) noexcept {
    Backend result;
    result.vtable_ = vtable;
    std::memcpy(result.payload_.data(), &payload, sizeof(Payload));
    return result;
  }

  template <class Payload>
    requires(std::is_trivially_copyable_v<Payload> &&
             sizeof(Payload) <= Backend::kInlineBytes &&
             alignof(Payload) <= alignof(void *))
  static Payload payload(const Backend &backend) noexcept {
    Payload result{};
    std::memcpy(&result, backend.payload_.data(), sizeof(Payload));
    return result;
  }

  static const BackendVTable *vtable(const Backend &backend) noexcept {
    return backend.vtable_;
  }
};

struct DeviceAccess final {
  template <class Payload>
    requires(std::is_trivially_copyable_v<Payload> &&
             sizeof(Payload) <= Device::kInlineBytes &&
             alignof(Payload) <= alignof(void *))
  static Device make(const BackendVTable *vtable,
                     const Payload &payload) noexcept {
    Device result;
    result.vtable_ = vtable;
    std::memcpy(result.payload_.data(), &payload, sizeof(Payload));
    return result;
  }

  template <class Payload>
    requires(std::is_trivially_copyable_v<Payload> &&
             sizeof(Payload) <= Device::kInlineBytes &&
             alignof(Payload) <= alignof(void *))
  static Payload payload(const Device &device) noexcept {
    Payload result{};
    std::memcpy(&result, device.payload_.data(), sizeof(Payload));
    return result;
  }

  static const BackendVTable *vtable(const Device &device) noexcept {
    return device.vtable_;
  }
};

} // namespace hyle::compute::detail
