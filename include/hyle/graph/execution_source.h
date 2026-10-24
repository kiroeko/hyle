#pragma once

#include "hyle/compute/device.h"
#include "hyle/graph/visibility.h"
#include "hyle/mem/resource.h"

namespace hyle::graph {
namespace detail {
struct ExecutionSourceState;
struct ExecutionSourceAccess;
}

// Retains one actual execution endpoint. A source neither creates another
// Context nor owns a scheduler. Closing the original endpoint still stops
// admission; retained programs/results keep their real backing owners.
class ExecutionSource final {
public:
  ExecutionSource() noexcept = default;
  HYLE_GRAPH_API ExecutionSource(const ExecutionSource &) noexcept;
  HYLE_GRAPH_API ExecutionSource &operator=(const ExecutionSource &) noexcept;
  HYLE_GRAPH_API ExecutionSource(ExecutionSource &&) noexcept;
  HYLE_GRAPH_API ExecutionSource &operator=(ExecutionSource &&) noexcept;
  HYLE_GRAPH_API ~ExecutionSource();

  HYLE_GRAPH_API static Result<ExecutionSource>
  acquire(const compute::Context &) noexcept;

  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  HYLE_GRAPH_API bool is_open() const noexcept;
  HYLE_GRAPH_API bool same_owner(const ExecutionSource &) const noexcept;
  HYLE_GRAPH_API bool same_device(const ExecutionSource &) const noexcept;
  HYLE_GRAPH_API compute::BackendId backend_id() const noexcept;
  HYLE_GRAPH_API compute::DeviceClass device_class() const noexcept;
  HYLE_GRAPH_API Result<compute::DeviceProperties> device_properties() const noexcept;
  HYLE_GRAPH_API mem::AllocatorRef metadata_allocator() const noexcept;
  HYLE_GRAPH_API Status validate_resource(mem::MemoryResourceRef) const noexcept;

private:
  friend struct detail::ExecutionSourceAccess;
  explicit ExecutionSource(detail::ExecutionSourceState *state) noexcept : state_(state) {}
  detail::ExecutionSourceState *state_ = nullptr;
};
} // namespace hyle::graph
