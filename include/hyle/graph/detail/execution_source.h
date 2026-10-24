#pragma once

#include "hyle/graph/execution_source.h"
#include "hyle/compute/detail/provider_extension.h"

namespace hyle::graph::detail {
// One region capability on its real source. Native capabilities have no
// Compute provider index/instance. The ordinary shared path retains the real
// ProviderExtensionRef and its original admission checks.
class GraphRegionSource final {
public:
  GraphRegionSource() noexcept = default;
  HYLE_GRAPH_API static Result<GraphRegionSource> shared(ExecutionSource,
      compute::detail::ProviderExtensionRef) noexcept;
  HYLE_GRAPH_API static Result<GraphRegionSource> native(ExecutionSource,
      const void *interface_value) noexcept;
  // Built-in ordered transfer on the actual shared Context, independent of
  // numerical providers. No provider slot or provider identity is invented.
  HYLE_GRAPH_API static Result<GraphRegionSource> shared_queue(ExecutionSource,
      const void *interface_value) noexcept;
  bool valid() const noexcept { return source_.valid() && (extension_.valid() || interface_ != nullptr); }
  explicit operator bool() const noexcept { return valid(); }
  HYLE_GRAPH_API bool is_native() const noexcept;
  HYLE_GRAPH_API bool is_shared_queue() const noexcept;
  const ExecutionSource &source() const noexcept { return source_; }
  const void *native_interface() const noexcept { return interface_; }
  const compute::detail::ProviderExtensionRef *shared_extension() const noexcept {
    return extension_.valid() ? &extension_ : nullptr;
  }
  compute::ProviderIdentity provider() const noexcept { return extension_.provider(); }
  compute::ProviderInstanceId provider_instance_id() const noexcept { return extension_.provider_instance_id(); }
  compute::ProviderExtensionId id() const noexcept { return extension_.id(); }
  HYLE_GRAPH_API Status call(compute::detail::ProviderExtensionCallFn, void *) const noexcept;

private:
  ExecutionSource source_;
  compute::detail::ProviderExtensionRef extension_;
  const void *interface_ = nullptr;
};
} // namespace hyle::graph::detail
