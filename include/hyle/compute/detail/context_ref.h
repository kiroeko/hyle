#pragma once

#include "hyle/compute/device.h"

namespace hyle::compute::detail {

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
  HYLE_COMPUTE_API bool owns(const Queue &queue) const noexcept;
  HYLE_COMPUTE_API Result<bool>
  has_backend_extension(BackendExtensionId id) const noexcept;
  HYLE_COMPUTE_API Status
  validate_resource(mem::MemoryResourceRef resource) const noexcept;

private:
  explicit ContextRef(ContextState *state) noexcept : state_(state) {}
  ContextState *state_ = nullptr;
};

} // namespace hyle::compute::detail
