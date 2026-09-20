#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "native_device.h"

namespace hyle::compute::iree_vulkan {
enum class ProgramKind {
  ScanInclusive,
  ScanExclusive,
  Float8Block128,
  Float8Group32
};
struct ProgramSpec final {
  ProgramKind kind = ProgramKind::ScanInclusive;
  std::int64_t rows = 0;
  std::int64_t columns = 0;
  std::int64_t depth = 0; // MatMul reduction dimension; ignored by Scan.
};
struct BufferSpan final {
  VkBuffer buffer = VK_NULL_HANDLE;
  std::uint64_t offset = 0;
  std::size_t bytes = 0;
};
struct WorkState final {
  Status status;
  bool pending = false;
};
using CompletionNotify = void (*)(void *, const Status &) noexcept;

// One original VM context and native input storage per invocation. Host calls
// are thread-compatible. GPU input copies must be enqueued on the owner's
// queue before invoke. Outputs remain private until retirement and, for Scan,
// the caller's overflow-flag readback. They cannot be published on failure.
class NativeProgram final {
public:
  NativeProgram(const NativeProgram &) = delete;
  NativeProgram &operator=(const NativeProgram &) = delete;
  HYLE_IREE_VULKAN_INTERNAL_API ~NativeProgram();
  HYLE_IREE_VULKAN_INTERNAL_API static Result<std::unique_ptr<NativeProgram>>
  create(NativeDevice device, ProgramSpec spec) noexcept;
  HYLE_IREE_VULKAN_INTERNAL_API std::size_t input_count() const noexcept;
  HYLE_IREE_VULKAN_INTERNAL_API Result<BufferSpan>
  input(std::size_t index) const noexcept;
  // The optional notification runs on the original proactor after GPU output
  // readiness. It may inspect output, but must not wait, destroy or re-invoke
  // this slot. wait/poll include the notification's return in retirement.
  HYLE_IREE_VULKAN_INTERNAL_API WorkState
  invoke(CompletionNotify notify = nullptr, void *user_data = nullptr) noexcept;
  // Only from this slot's completion notification, after enqueueing native
  // commands on the shared physical queue. Extends the slot through that
  // accepted native batch and a second notification on the original proactor.
  // This is not a drain proof for unsubmitted/deferred provider work.
  HYLE_IREE_VULKAN_INTERNAL_API Status
  after_native_submissions(CompletionNotify notify, void *user_data) noexcept;
  HYLE_IREE_VULKAN_INTERNAL_API WorkState poll() noexcept;
  HYLE_IREE_VULKAN_INTERNAL_API WorkState wait() noexcept;
  HYLE_IREE_VULKAN_INTERNAL_API Result<BufferSpan>
  output(std::size_t index) const noexcept;

private:
  struct State;
  explicit NativeProgram(std::unique_ptr<State> state) noexcept;
  WorkState observe(bool wait) noexcept;
  std::unique_ptr<State> state_;
};
} // namespace hyle::compute::iree_vulkan
