#pragma once

#include <utility>

#include "hyle/compute/detail/queue_sequence.h"
#include "hyle/compute/device.h"

namespace hyle::compute::detail {

struct ContextState;
struct BackendExtensionAccess;

// Copyable internal capability that retains the exact Context whose backend
// supplied a versioned interface table. Vendor types stay in the owning
// backend header; this bridge only carries identity, lifetime and admission.
class BackendExtensionRef final {
public:
  BackendExtensionRef() noexcept = default;
  HYLE_COMPUTE_API BackendExtensionRef(const BackendExtensionRef &other) noexcept;
  HYLE_COMPUTE_API BackendExtensionRef &
  operator=(const BackendExtensionRef &other) noexcept;
  HYLE_COMPUTE_API BackendExtensionRef(BackendExtensionRef &&other) noexcept;
  HYLE_COMPUTE_API BackendExtensionRef &
  operator=(BackendExtensionRef &&other) noexcept;
  HYLE_COMPUTE_API ~BackendExtensionRef();

  bool valid() const noexcept {
    return owner_ != nullptr && interface_ != nullptr && id_.valid();
  }
  explicit operator bool() const noexcept { return valid(); }
  constexpr BackendExtensionId id() const noexcept { return id_; }
  constexpr BackendId backend() const noexcept { return backend_; }

private:
  BackendExtensionRef(ContextState *owner, const void *interface_value,
                      BackendExtensionId id, BackendId backend) noexcept;

  friend struct BackendExtensionAccess;
  ContextState *owner_ = nullptr;
  const void *interface_ = nullptr;
  BackendExtensionId id_{};
  BackendId backend_{};
};

using BackendExtensionCallFn = Status (*)(void *backend_context,
                                          const void *interface_value,
                                          void *call_context) noexcept;
using BackendExtensionEnqueueFn = EnqueueResult (*)(
    void *backend_context, const void *interface_value, void *backend_queue,
    void *call_context) noexcept;

struct BackendExtensionAccess final {
  HYLE_COMPUTE_API static Result<BackendExtensionRef>
  query(const Context &context, BackendExtensionId id) noexcept;
  HYLE_COMPUTE_API static Result<BackendExtensionRef>
  query(const Queue &queue, BackendExtensionId id) noexcept;

  HYLE_COMPUTE_API static Status call(const BackendExtensionRef &extension,
                                      BackendExtensionCallFn function,
                                      void *call_context) noexcept;

  HYLE_COMPUTE_API static Result<Completion>
  submit(const BackendExtensionRef &extension, Queue &queue,
         BackendExtensionEnqueueFn function, void *call_context) noexcept;

  HYLE_COMPUTE_API static EnqueueResult enqueue_admitted(
      const BackendExtensionRef &extension, QueueSequence &sequence,
      BackendExtensionEnqueueFn function, void *call_context) noexcept;
};

} // namespace hyle::compute::detail
