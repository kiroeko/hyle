#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

#include "hyle/compute/completion.h"
#include "hyle/compute/device.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/tensor.h"
#include "hyle/error/result.h"
#include "hyle/graph/graph.h"
#include "hyle/graph/visibility.h"

namespace hyle::graph {

namespace detail {
struct PlanState;
struct InvocationState;
struct ReportState;
struct CompileReportAccess;
struct ExecutableAccess;
struct InvocationAccess;
} // namespace detail

class Invocation;

enum class CompileStepKind : std::uint8_t {
  kOperationFallback = 0,
  kProviderRegion,
  kCopy,
  kRepack,
  kBarrier,
};

enum class CompileFallbackReason : std::uint8_t {
  kNone = 0,
  kRegionSelectionDisabled,
  kNoEligibleRegion,
  kRegionNotSelected,
};

// One immutable step in the exact emitted execution plan. Copy/repack/barrier
// kinds are reserved by the accepted plan contract and appear only when an
// implementation actually emits such a step. Optional facts distinguish an
// opaque provider's unknown value from an exact zero.
struct CompileStepReport final {
  CompileStepKind kind = CompileStepKind::kOperationFallback;
  CompileFallbackReason fallback_reason = CompileFallbackReason::kNone;
  compute::ProviderIdentity provider{};
  std::uint32_t node_begin = 0;
  std::uint32_t node_count = 0;
  compute::WorkspaceRequirement workspace{};
  std::optional<std::size_t> persistent_bytes{};
  std::size_t borrowed_constant_count = 0;
  std::size_t packed_constant_count = 0;
  std::size_t packed_constant_bytes = 0;
  std::size_t copy_bytes = 0;
  std::uint32_t synchronization_count = 0;
  // Backend execution episodes per submit: one synchronous CPU executor
  // fan-out or one device command launch counts as one. Workers/tasks/chunks
  // do not count separately. A non-materializing metadata step reports zero.
  std::uint32_t dispatch_count = 1;
  bool blocks_host = false;
  bool uses_private_threads = false;
  bool supports_concurrent_invocations = true;
};

// Copyable cold-path report that owns a compact immutable copy of the emitted
// plan facts independently of the Executable and its heavyweight artifacts.
// Aggregate queries and step(index) are O(1); step returns a value, so no
// diagnostic view dangles after either handle moves.
class CompileReport final {
public:
  CompileReport() noexcept = default;
  HYLE_GRAPH_API CompileReport(const CompileReport &other) noexcept;
  HYLE_GRAPH_API CompileReport &operator=(const CompileReport &other) noexcept;
  HYLE_GRAPH_API CompileReport(CompileReport &&other) noexcept;
  HYLE_GRAPH_API CompileReport &operator=(CompileReport &&other) noexcept;
  HYLE_GRAPH_API ~CompileReport();

  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  HYLE_GRAPH_API compute::BackendId backend() const noexcept;
  HYLE_GRAPH_API compute::ProviderIdentity provider() const noexcept;
  HYLE_GRAPH_API std::size_t workspace_bytes() const noexcept;
  HYLE_GRAPH_API std::size_t workspace_backing_count() const noexcept;
  HYLE_GRAPH_API std::size_t node_count() const noexcept;
  HYLE_GRAPH_API std::size_t step_count() const noexcept;
  HYLE_GRAPH_API std::size_t region_count() const noexcept;
  HYLE_GRAPH_API std::size_t fallback_operation_count() const noexcept;
  HYLE_GRAPH_API std::size_t constant_count() const noexcept;
  HYLE_GRAPH_API std::size_t borrowed_constant_count() const noexcept;
  HYLE_GRAPH_API std::size_t packed_constant_count() const noexcept;
  HYLE_GRAPH_API std::size_t packed_constant_bytes() const noexcept;
  HYLE_GRAPH_API std::size_t invocation_owned_output_count() const noexcept;
  HYLE_GRAPH_API std::size_t copy_bytes() const noexcept;
  HYLE_GRAPH_API std::uint32_t synchronization_count() const noexcept;
  HYLE_GRAPH_API bool mixed_providers() const noexcept;
  HYLE_GRAPH_API Result<CompileStepReport>
  step(std::size_t index) const noexcept;

private:
  friend struct detail::CompileReportAccess;
  explicit CompileReport(detail::ReportState *state) noexcept : state_(state) {}
  detail::ReportState *state_ = nullptr;
};

// A constant binding is conservatively borrowed by the resulting Executable.
// A provider may additionally own a packed representation reported by
// CompileReport, but v3 does not use that aggregate fact to shorten the source
// lifetime. The source storage must outlive the Executable and every
// completion submitted from it.
struct ConstantBinding final {
  ValueId value{};
  compute::ConstTensorView tensor{};
};

enum class RegionSelectionPolicy : std::uint8_t {
  kDisabled = 0,
  // Accept only proposals with a positive provider-declared net saving after
  // boundary copies, synchronization, dispatch and thread handoff.
  kConservative,
};

enum class OperationFallbackPolicy : std::uint8_t {
  kAllow = 0,
  // Compile fails unless provider regions cover every logical node. This is
  // useful when a caller requires a captured/fused artifact and would rather
  // reject a specialization than silently execute single-operation steps.
  kReject,
};

enum class OutputOwnership : std::uint8_t {
  // A materialized output uses a distinct caller-provided mutable binding.
  kCaller = 0,
  // A published input, constant or required View aliases storage already
  // supplied by its source binding. The output cannot be rebound; retrieve
  // the derived borrowed view from Invocation::output() after completion.
  kSourceAlias,
  // The Invocation owns the backing through its workspace. This is used when
  // a published zero-copy view is rooted in private graph storage; callers can
  // add an explicit Copy node when they require their own destination.
  kInvocation,
};

struct CompileOptions final {
  std::span<const ConstantBinding> constants{};
  RegionSelectionPolicy region_selection = RegionSelectionPolicy::kConservative;
  OperationFallbackPolicy operation_fallback = OperationFallbackPolicy::kAllow;
  // Hard cold-path budget across all configured providers. Exhaustion fails
  // compilation instead of silently changing the selected plan.
  std::size_t max_region_proposals = 4096;
  // Independent cap offered to each provider. The effective writer capacity
  // is min(this value, remaining global budget); either limit is a hard error
  // rather than a silent truncation.
  std::size_t max_region_proposals_per_provider = 1024;
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
  HYLE_GRAPH_API std::size_t step_count() const noexcept;
  HYLE_GRAPH_API std::size_t region_count() const noexcept;
  HYLE_GRAPH_API std::size_t workspace_bytes() const noexcept;
  HYLE_GRAPH_API std::size_t workspace_backing_count() const noexcept;
  HYLE_GRAPH_API Result<ValueId> input_id(std::size_t index = 0) const noexcept;
  HYLE_GRAPH_API Result<ValueId>
  output_id(std::size_t index = 0) const noexcept;
  HYLE_GRAPH_API Result<OutputOwnership>
  output_ownership(std::size_t index = 0) const noexcept;
  HYLE_GRAPH_API Result<Invocation> create_invocation() const noexcept;

private:
  friend struct detail::ExecutableAccess;
  explicit Executable(detail::PlanState *state) noexcept : state_(state) {}
  detail::PlanState *state_ = nullptr;
};

// Exclusive mutable run state: external bindings plus one Mem lifetime-plan
// instance. Creation may allocate; bind/reset/submit never grow metadata,
// search a provider, JIT, pack weights, or allocate workspace. bind() checks
// the candidate against already-bound external outputs (and, for a new output,
// external inputs), so all bindings together cost O(inputs*outputs+outputs^2)
// in the worst case while repeated submit remains O(emitted steps+bindings).
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
  HYLE_GRAPH_API bool in_flight() const noexcept;
  HYLE_GRAPH_API Status reset() noexcept;
  // Returns a borrowed published output only after the latest submission has
  // reached a successful terminal state. Invocation-owned views remain valid
  // until reset, the next submit, or Invocation destruction.
  HYLE_GRAPH_API Result<compute::ConstTensorView>
  output(ValueId value) const noexcept;
  HYLE_GRAPH_API Result<compute::Completion>
  submit(compute::Queue &queue) noexcept;

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
compile(const Graph &graph, const compute::Context &context,
        CompileOptions options = {}) noexcept;

} // namespace hyle::graph
