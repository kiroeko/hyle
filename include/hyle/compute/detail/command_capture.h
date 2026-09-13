#pragma once

#include <utility>

#include "hyle/compute/detail/backend_extension.h"
#include "hyle/mem/allocator.h"

namespace hyle::compute::detail {

// Backend-neutral command capture contract used by Graph. Vendor graph types
// stay inside their backend factor; the core runtime owns only an opaque,
// move-only replay artifact. Version changes never reinterpret an old table.
inline constexpr BackendExtensionId kCommandCaptureExtensionId{
    0x48594c45434f4d50ULL, 0x4341505455524531ULL, 1};

struct CommandCaptureVTable final {
  Status (*begin)(void *backend_context,
                  void *backend_queue) noexcept = nullptr;
  Status (*end)(void *backend_context, void *backend_queue,
                mem::AllocatorRef allocator,
                void **captured_commands) noexcept = nullptr;
  Status (*abort)(void *backend_context,
                  void *backend_queue) noexcept = nullptr;
  void (*destroy)(void *captured_commands,
                  mem::AllocatorRef allocator) noexcept = nullptr;
  EnqueueResult (*launch)(void *backend_context, void *backend_queue,
                          void *captured_commands) noexcept = nullptr;
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

struct CommandCaptureAccess final {
  HYLE_COMPUTE_API static Result<BackendExtensionRef>
  query(const Queue &queue) noexcept;
  HYLE_COMPUTE_API static Status begin(const BackendExtensionRef &extension,
                                       QueueSequence &sequence) noexcept;
  HYLE_COMPUTE_API static Result<CapturedCommands>
  end(const BackendExtensionRef &extension, QueueSequence &sequence,
      mem::AllocatorRef allocator) noexcept;
  HYLE_COMPUTE_API static Status abort(const BackendExtensionRef &extension,
                                       QueueSequence &sequence) noexcept;
};

} // namespace hyle::compute::detail
