#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/compute/completion.h"
#include "hyle/compute/device.h"
#include "hyle/compute/tensor.h"
#include "hyle/error/result.h"
#include "hyle/graph/graph.h"
#include "hyle/graph/visibility.h"

namespace hyle::graph {

namespace detail {
struct PlanState;
struct InvocationState;
struct ExecutableAccess;
struct InvocationAccess;
} // namespace detail

class Invocation;

// Cold-path facts for the exact artifact returned by compile(). CPU v2 never
// hides copies, synchronization or fallback behind this report.
struct CompileReport final {
  compute::BackendKind backend = compute::BackendKind::kUnknown;
  compute::ProviderKind provider = compute::ProviderKind::kHyleNative;
  std::size_t workspace_bytes = 0;
  std::size_t workspace_backing_count = 0;
  std::size_t node_count = 0;
  std::size_t copy_bytes = 0;
  std::uint32_t synchronization_count = 0;
  bool used_fallback = false;
};

// Immutable, shareable compiled artifact. Copies share one allocator-owned
// plan. Mutable bindings and workspace never live here: every concurrent run
// creates a distinct Invocation.
class Executable final {
public:
  Executable() noexcept = default;
  HYLE_GRAPH_API Executable(const Executable &other) noexcept;
  HYLE_GRAPH_API Executable &operator=(const Executable &other) noexcept;
  HYLE_GRAPH_API Executable(Executable &&other) noexcept;
  HYLE_GRAPH_API Executable &operator=(Executable &&other) noexcept;
  HYLE_GRAPH_API ~Executable();

  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  HYLE_GRAPH_API std::size_t input_count() const noexcept;
  HYLE_GRAPH_API std::size_t output_count() const noexcept;
  HYLE_GRAPH_API std::size_t node_count() const noexcept;
  HYLE_GRAPH_API std::size_t workspace_bytes() const noexcept;
  HYLE_GRAPH_API std::size_t workspace_backing_count() const noexcept;
  HYLE_GRAPH_API Result<ValueId> input_id(std::size_t index = 0) const noexcept;
  HYLE_GRAPH_API Result<ValueId>
  output_id(std::size_t index = 0) const noexcept;
  HYLE_GRAPH_API Result<Invocation> create_invocation() const noexcept;

private:
  friend struct detail::ExecutableAccess;
  explicit Executable(detail::PlanState *state) noexcept : state_(state) {}
  detail::PlanState *state_ = nullptr;
};

// Exclusive mutable run state: external bindings plus one Mem lifetime-plan
// instance. Creation may allocate; bind/reset/submit never grow metadata,
// search a provider, JIT, pack weights, or allocate workspace.
class Invocation final {
public:
  Invocation() noexcept = default;
  Invocation(const Invocation &) = delete;
  Invocation &operator=(const Invocation &) = delete;
  HYLE_GRAPH_API Invocation(Invocation &&other) noexcept;
  HYLE_GRAPH_API Invocation &operator=(Invocation &&other) noexcept;
  HYLE_GRAPH_API ~Invocation();

  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  HYLE_GRAPH_API Status bind(ValueId value,
                             compute::ConstTensorView tensor) noexcept;
  HYLE_GRAPH_API Status bind(ValueId value,
                             compute::TensorView tensor) noexcept;
  HYLE_GRAPH_API void reset() noexcept;
  HYLE_GRAPH_API Result<compute::Completion>
  submit(compute::Queue &queue) const noexcept;

private:
  friend struct detail::InvocationAccess;
  explicit Invocation(detail::InvocationState *state) noexcept
      : state_(state) {}
  detail::InvocationState *state_ = nullptr;
};

struct CompiledGraph final {
  Executable executable{};
  CompileReport report{};
};

HYLE_GRAPH_API Result<CompiledGraph>
compile(const Graph &graph, const compute::Context &context) noexcept;

} // namespace hyle::graph
