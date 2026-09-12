#pragma once

#include "hyle/compute/completion.h"
#include "hyle/compute/device.h"
#include "hyle/compute/tensor.h"

namespace hyle::compute::detail {

struct ContextState;
struct ProviderExtensionAccess;

// Internal copyable capability handle. It retains the exact Context/provider
// state so a compiled artifact cannot dangle when the caller releases its
// Context handle. Retention does not authorize new work after Context::close.
class ProviderExtensionRef final {
public:
  ProviderExtensionRef() noexcept = default;
  HYLE_COMPUTE_API
  ProviderExtensionRef(const ProviderExtensionRef &other) noexcept;
  HYLE_COMPUTE_API ProviderExtensionRef &
  operator=(const ProviderExtensionRef &other) noexcept;
  HYLE_COMPUTE_API ProviderExtensionRef(ProviderExtensionRef &&other) noexcept;
  HYLE_COMPUTE_API ProviderExtensionRef &
  operator=(ProviderExtensionRef &&other) noexcept;
  HYLE_COMPUTE_API ~ProviderExtensionRef();

  bool valid() const noexcept {
    return owner_ != nullptr && interface_ != nullptr && id_.valid();
  }
  explicit operator bool() const noexcept { return valid(); }
  constexpr ProviderExtensionId id() const noexcept { return id_; }
  constexpr ProviderIdentity provider() const noexcept { return provider_; }

private:
  ProviderExtensionRef(ContextState *owner, void *provider_context,
                       const void *interface_value, ProviderExtensionId id,
                       ProviderIdentity provider) noexcept;

  friend struct ProviderExtensionAccess;
  ContextState *owner_ = nullptr;
  void *provider_context_ = nullptr;
  const void *interface_ = nullptr;
  ProviderExtensionId id_{};
  ProviderIdentity provider_{};
};

// Internal cross-factor bridge used by typed extensions such as Graph region
// providers. The provider-specific interface and call payload remain typed in
// the owning factor; Compute only supplies lifetime, admission and exact Queue
// ownership checks.
using ProviderExtensionCallFn = Status (*)(void *provider_context,
                                           const void *interface_value,
                                           void *call_context) noexcept;

using ProviderExtensionSubmitFn =
    Result<Completion> (*)(void *provider_context, const void *interface_value,
                           void *call_context, Queue &queue) noexcept;

struct ProviderExtensionAccess final {
  HYLE_COMPUTE_API static Result<ProviderExtensionRef>
  query(const Context &context, std::size_t provider_index,
        ProviderExtensionId id) noexcept;

  HYLE_COMPUTE_API static Status call(const ProviderExtensionRef &extension,
                                      ProviderExtensionCallFn function,
                                      void *call_context) noexcept;

  HYLE_COMPUTE_API static Result<Completion>
  submit(const ProviderExtensionRef &extension, Queue &queue,
         std::span<const ConstTensorView> inputs,
         std::span<const TensorView> outputs,
         ProviderExtensionSubmitFn function, void *call_context) noexcept;
};

} // namespace hyle::compute::detail
