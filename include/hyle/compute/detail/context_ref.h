#pragma once

#include "hyle/compute/device.h"

namespace hyle::compute::detail {
class ProviderExtensionRef;
class QueueRef;

// Internal copyable lease for compiled artifacts that must retain and compare
// an exact Context without exposing backend state or making public Context
// copyable. It never reopens admission after Context::close().
class ContextRef final {
public:
  ContextRef() noexcept = default;
  HYLE_COMPUTE_API ContextRef(const ContextRef &other) noexcept;
  HYLE_COMPUTE_API ContextRef &operator=(const ContextRef &other) noexcept;
  HYLE_COMPUTE_API ContextRef(ContextRef &&other) noexcept;
  HYLE_COMPUTE_API ContextRef &operator=(ContextRef &&other) noexcept;
  HYLE_COMPUTE_API ~ContextRef();

  HYLE_COMPUTE_API static Result<ContextRef>
  acquire(const Context &context) noexcept;

  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  HYLE_COMPUTE_API Device device() const noexcept;
  HYLE_COMPUTE_API bool is_open() const noexcept;
  // Stable only while this retained owner lives. Identifies the actual
  // Context, independently of public-handle moves and provider slots.
  const void *identity_key() const noexcept { return state_; }
  HYLE_COMPUTE_API mem::AllocatorRef metadata_allocator() const noexcept;
  HYLE_COMPUTE_API std::size_t provider_count() const noexcept;
  HYLE_COMPUTE_API Result<ProviderIdentity>
  provider_identity(std::size_t) const noexcept;
  HYLE_COMPUTE_API Result<std::uint32_t>
  provider_preference(std::size_t) const noexcept;
  HYLE_COMPUTE_API Result<ProviderInstanceId>
  provider_instance_id(std::size_t) const noexcept;
  // Cold/native composition only. The shared Queue's admitted fast path
  // continues to call its selected prepared implementation directly.
  HYLE_COMPUTE_API Status call(Status (*)(void *) noexcept, void *) const noexcept;
  HYLE_COMPUTE_API Result<Queue> create_queue(QueueOptions = {}) const noexcept;
  HYLE_COMPUTE_API bool owns(const Queue &queue) const noexcept;
  HYLE_COMPUTE_API bool owns(const QueueRef &queue) const noexcept;
  HYLE_COMPUTE_API bool owns(const ProviderExtensionRef &extension) const noexcept;
  HYLE_COMPUTE_API Result<bool>
  has_backend_extension(BackendExtensionId id) const noexcept;
  HYLE_COMPUTE_API Status
  validate_resource(mem::MemoryResourceRef resource) const noexcept;

private:
  friend struct ContextAccess;
  explicit ContextRef(ContextState *state) noexcept : state_(state) {}
  ContextState *state_ = nullptr;
};

} // namespace hyle::compute::detail
