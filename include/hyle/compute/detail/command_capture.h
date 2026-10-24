#pragma once

#include <utility>

#include "hyle/compute/detail/backend_extension.h"
#include "hyle/mem/allocator.h"

namespace hyle::compute::detail {

// Backend-neutral command capture contract used by Graph. Vendor graph types
// stay inside their backend factor; the core runtime owns only an opaque,
// move-only replay artifact. Version changes never reinterpret an old table.
inline constexpr BackendExtensionId kCommandCaptureExtensionId{
    0x48594c45434f4d50ULL, 0x4341505455524531ULL, 3};

struct CommandCaptureVTable final {
  Status (*begin)(void *backend_context,
                  void *backend_queue) noexcept = nullptr;
  Status (*end)(void *backend_context, void *backend_queue,
                mem::AllocatorRef allocator, void *rebuild_storage,
                void **captured_commands) noexcept = nullptr;
  Status (*abort)(void *backend_context,
                  void *backend_queue) noexcept = nullptr;
  void (*destroy)(void *captured_commands,
                  mem::AllocatorRef allocator) noexcept = nullptr;
  EnqueueResult (*launch)(void *backend_context, void *backend_queue,
                          void *captured_commands) noexcept = nullptr;
  // Optional. Called only after all native uses have actually retired. On
  // success the artifact becomes permanently nonlaunchable; its idle native
  // allocation may survive released binding/provider owners. It must never
  // access those old arguments again. end() may reuse it with a newly captured
  // graph, returning a distinct owned artifact, or instantiate normally.
  // Returning false leaves ownership unchanged and the caller destroys it.
  bool (*recycle_after_retirement)(void *captured_commands) noexcept = nullptr;
  // For this backend's admitted resources, equal complete TensorView
  // descriptions/resource/token/offset/span after rebinding still denote the
  // same recorded physical arguments. Opaque indirection alone cannot prove
  // this. Native imports and mutable private resources need their own gates.
  bool tensor_bindings_have_stable_identity = false;
};

class CapturedCommands final {
public:
  CapturedCommands() noexcept = default;
  CapturedCommands(const CapturedCommands &) = delete;
  CapturedCommands &operator=(const CapturedCommands &) = delete;
  HYLE_COMPUTE_API CapturedCommands(CapturedCommands &&other) noexcept;
  HYLE_COMPUTE_API CapturedCommands &
  operator=(CapturedCommands &&other) noexcept;
  HYLE_COMPUTE_API ~CapturedCommands();

  bool valid() const noexcept {
    return extension_.valid() && table_ != nullptr && state_ != nullptr;
  }
  explicit operator bool() const noexcept { return valid(); }
  bool tensor_bindings_have_stable_identity() const noexcept {
    return valid() && table_->tensor_bindings_have_stable_identity;
  }
  HYLE_COMPUTE_API EnqueueResult enqueue(QueueSequence &sequence) noexcept;
  HYLE_COMPUTE_API void reset() noexcept;

private:
  friend struct CommandCaptureAccess;
  CapturedCommands(BackendExtensionRef extension,
                   const CommandCaptureVTable *table, void *state,
                   mem::AllocatorRef allocator) noexcept
      : extension_(std::move(extension)), table_(table), state_(state),
        allocator_(allocator) {}

  BackendExtensionRef extension_{};
  const CommandCaptureVTable *table_ = nullptr;
  void *state_ = nullptr;
  mem::AllocatorRef allocator_{};
};

// A backend-authorized idle allocation, with no enqueue API or old binding
// authority. At most one is needed for rebuilding an immutable capture range.
class CaptureRebuildStorage final {
public:
  CaptureRebuildStorage() noexcept = default;
  CaptureRebuildStorage(const CaptureRebuildStorage &) = delete;
  CaptureRebuildStorage &operator=(const CaptureRebuildStorage &) = delete;
  CaptureRebuildStorage(CaptureRebuildStorage &&) noexcept = default;
  CaptureRebuildStorage &operator=(CaptureRebuildStorage &&) noexcept = default;
  bool valid() const noexcept { return storage_.valid(); }
  void reset() noexcept { storage_.reset(); }
private:
  friend struct CommandCaptureAccess;
  CapturedCommands storage_{};
};

struct CommandCaptureAccess final {
  HYLE_COMPUTE_API static Result<BackendExtensionRef>
  query(const Queue &queue) noexcept;
  HYLE_COMPUTE_API static Status begin(const BackendExtensionRef &extension,
                                       QueueSequence &sequence) noexcept;
  HYLE_COMPUTE_API static Result<CapturedCommands>
  end(const BackendExtensionRef &extension, QueueSequence &sequence,
      mem::AllocatorRef allocator, CaptureRebuildStorage rebuild = {}) noexcept;
  // The caller must prove actual native retirement, independently of public
  // Completion destruction or reference uniqueness. Failure to recycle simply
  // destroys the old artifact and preserves ordinary fresh capture behavior.
  HYLE_COMPUTE_API static CaptureRebuildStorage
  recycle_after_retirement(CapturedCommands commands) noexcept;
  HYLE_COMPUTE_API static Status abort(const BackendExtensionRef &extension,
                                       QueueSequence &sequence) noexcept;
};

} // namespace hyle::compute::detail
