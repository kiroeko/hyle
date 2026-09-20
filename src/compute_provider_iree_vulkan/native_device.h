#pragma once

#include <memory>

#include "../compute_provider_ggml_vulkan/native_device.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#define HYLE_IREE_VULKAN_INTERNAL_API
#elif defined(HYLE_COMPUTE_PROVIDER_IREE_VULKAN_BUILD)
#define HYLE_IREE_VULKAN_INTERNAL_API HYLE_EXPORT
#else
#define HYLE_IREE_VULKAN_INTERNAL_API HYLE_IMPORT
#endif

namespace hyle::compute::iree_vulkan {
class NativeProgram;

// Lazy, independent original runtime owner. Copies retain the ggml native
// device directly, never its parent route. Quota, if supplied, outlives all
// copies and invocations. Shared queue access uses that native owner's lock.
class NativeDevice final {
public:
  NativeDevice() noexcept = default;
  HYLE_IREE_VULKAN_INTERNAL_API static Result<NativeDevice>
  create(ggml_vulkan::detail::NativeDevice native,
         mem::QuotaAllocator *quota = nullptr) noexcept;
  explicit operator bool() const noexcept { return state_ != nullptr; }

private:
  friend class NativeProgram;
  struct State;
  std::shared_ptr<State> state_;
};
} // namespace hyle::compute::iree_vulkan
