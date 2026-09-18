#pragma once

#include "native_device.h"

namespace hyle::compute::ggml_vulkan::detail {

// One cold, thread-compatible resident graph per execution slot. The private
// semantic builder provides a sufficiently sized arena, legal graph topology,
// and input/output flags. Borrowed tensors require a strong imports owner.
// gallocr keeps upstream liveness/in-place reuse; allocation is never repeated
// by submit. NativeWork retains lease() through native completion. Public
// result retention/reuse is enforced by the route's owning invocation.
class NativeGraph final {
public:
  NativeGraph() noexcept = default;
  NativeGraph(const NativeGraph &) = delete;
  NativeGraph &operator=(const NativeGraph &) = delete;
  NativeGraph(NativeGraph &&) noexcept = default;
  NativeGraph &operator=(NativeGraph &&) noexcept = default;
  HYLE_GGML_VULKAN_INTERNAL_API static Result<NativeGraph>
  allocate(NativeDevice, MetadataArena, ggml_cgraph *,
           std::shared_ptr<const void> imports = {}) noexcept;
  HYLE_GGML_VULKAN_INTERNAL_API ggml_cgraph *get() const noexcept;
  HYLE_GGML_VULKAN_INTERNAL_API std::size_t storage_bytes() const noexcept;
  std::shared_ptr<const void> lease() const noexcept { return state_; }

private:
  struct State;
  std::shared_ptr<State> state_;
};
} // namespace hyle::compute::ggml_vulkan::detail
