#pragma once

#include <array>
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
struct ExecutableAccess;
}  // namespace detail

class Invocation;

struct CompileReport final {
  compute::BackendKind backend = compute::BackendKind::kUnknown;
  compute::ProviderKind provider = compute::ProviderKind::kHyleNative;
  std::size_t workspace_bytes = 0;
  std::size_t copy_bytes = 0;
  std::uint32_t synchronization_count = 0;
  bool used_fallback = false;
};

// Immutable, shareable compiled artifact. Copies share one allocator-owned
// intrusive plan state; each concurrent execution uses a distinct Invocation.
class Executable final {
 public:
  Executable() noexcept = default;
  HYLE_GRAPH_API Executable(const Executable& other) noexcept;
  HYLE_GRAPH_API Executable& operator=(const Executable& other) noexcept;
  HYLE_GRAPH_API Executable(Executable&& other) noexcept;
  HYLE_GRAPH_API Executable& operator=(Executable&& other) noexcept;
  HYLE_GRAPH_API ~Executable();

  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  HYLE_GRAPH_API std::size_t input_count() const noexcept;
  HYLE_GRAPH_API Result<ValueId> input_id(
      std::size_t index = 0) const noexcept;
  HYLE_GRAPH_API Result<ValueId> output_id(
      std::size_t index = 0) const noexcept;
  HYLE_GRAPH_API Result<Invocation> create_invocation() const noexcept;

 private:
  friend struct detail::ExecutableAccess;
  explicit Executable(detail::PlanState* state) noexcept : state_(state) {}
  detail::PlanState* state_ = nullptr;
};

// Exclusive mutable binding set. CPU v1 completes synchronously, so submit()
// returns only after the bindings are reusable; asynchronous providers will
// retain the same no-reset-before-completion rule at this boundary.
class Invocation final {
 public:
  Invocation() noexcept = default;
  Invocation(const Invocation&) = delete;
  Invocation& operator=(const Invocation&) = delete;
  HYLE_GRAPH_API Invocation(Invocation&& other) noexcept;
  HYLE_GRAPH_API Invocation& operator=(Invocation&& other) noexcept;
  HYLE_GRAPH_API ~Invocation();

  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  HYLE_GRAPH_API Status bind(ValueId value,
                             compute::ConstTensorView tensor) noexcept;
  HYLE_GRAPH_API Status bind(ValueId value,
                             compute::TensorView tensor) noexcept;
  HYLE_GRAPH_API void reset() noexcept;
  HYLE_GRAPH_API Result<compute::Completion> submit(
      compute::Queue& queue) const noexcept;

 private:
  friend class Executable;
  explicit Invocation(detail::PlanState* state) noexcept;

  detail::PlanState* state_ = nullptr;
  std::array<compute::ConstTensorView, 2> inputs_{};
  compute::TensorView output_{};
  std::uint8_t input_mask_ = 0;
  bool output_bound_ = false;
};

struct CompiledGraph final {
  Executable executable{};
  CompileReport report{};
};

HYLE_GRAPH_API Result<CompiledGraph> compile(
    const Graph& graph, const compute::Context& context) noexcept;

}  // namespace hyle::graph
