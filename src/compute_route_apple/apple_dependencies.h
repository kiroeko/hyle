#pragma once

#include <memory>
#include <vector>
#include <mlx/event.h>
#include "hyle/compute/detail/completion_access.h"
#include "../compute_provider_ggml_metal/native_program.h"

namespace hyle::compute::apple::detail {
struct Runtime;
struct InvocationState;
struct CopyRecording;
struct DeviceCheckInputs;

// Inline metadata in an original math/copy Job. References pin the original
// accepted episodes; this owns no execution queue, tensor or permission ledger.
class NativeDependencies final {
public:
  NativeDependencies() = default;
  NativeDependencies(const NativeDependencies &) = delete;
  NativeDependencies &operator=(const NativeDependencies &) = delete;
  ~NativeDependencies();
  Status reserve(std::size_t capacity) noexcept;
  Status capture(const std::shared_ptr<Runtime> &,
      std::span<const compute::detail::CompletionDependency>,
      InvocationState *consumer = nullptr, CopyRecording *copy_consumer = nullptr) noexcept;
  Status append(const std::shared_ptr<Runtime> &,
      std::span<const compute::detail::CompletionDependency>,
      InvocationState *consumer = nullptr, CopyRecording *copy_consumer = nullptr) noexcept;
  Status device_checks(DeviceCheckInputs &) noexcept;
  Status resolve() noexcept; // Original FIFO owner only.
  // Raw copies on this same Runtime already import their blit memory fence.
  // Keep the original producer/error owners without duplicating that wait.
  Status resolve_copy() noexcept;
  Status observe() noexcept;
  void encode(mlx::core::Stream, const mlx::core::Event *ordered = nullptr);
  void clear() noexcept; // Only after actual retirement or failed admission.
  bool empty() const noexcept { return sources_.empty(); }
  auto waits() const noexcept {
    return native_encoded_ ? std::span<const ggml_metal::NativeWait>{}
                           : std::span<const ggml_metal::NativeWait>(waits_);
  }
  // All original ggml regions use the same real device queue. After its first
  // accepted import, that queue orders later regions behind the same frontier.
  void native_submitted() noexcept { native_encoded_ = true; }
  auto events() noexcept { return std::span<mlx::core::Event>(events_); }
private:
  Status capture_impl(const std::shared_ptr<Runtime> &,
      std::span<const compute::detail::CompletionDependency>,
      InvocationState *,CopyRecording *,bool deduplicate) noexcept;
  Status resolve_impl(bool blit_stream) noexcept;
  struct Source final {
    InvocationState *native = nullptr;
    std::shared_ptr<CopyRecording> copy;
    std::uint64_t epoch = 0;
  };
  std::vector<Source> sources_;
  std::vector<mlx::core::Event> events_;        // Native queue -> original MLX stream.
  std::vector<mlx::core::Event> native_events_; // MLX stream -> original native queue.
  std::vector<ggml_metal::NativeWait> waits_;   // Borrows native_events_ until retirement.
  bool encoded_ = false;
  bool native_encoded_ = false;
};
} // namespace hyle::compute::apple::detail
