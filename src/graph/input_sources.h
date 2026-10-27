#pragma once

#include <cstddef>
#include <span>

#include "hyle/graph/executable.h"

namespace hyle::graph::detail {
struct InvocationState;
struct InputSourcesState;
struct InputSourceBinding;
class InputSource;
struct InputSourceVTable;

// The original readonly binding checks and commit, shared by ordinary single
// binds and the atomic immutable-input bundle. Candidate ranges only override
// incoming inputs during alias validation; no physical Plan is duplicated.
HYLE_GRAPH_API Status validate_read_binding(InvocationState &, ValueId,
    compute::ConstTensorView, std::span<const InputSourceBinding> = {}) noexcept;
HYLE_GRAPH_API void commit_read_binding(InvocationState &, ValueId,
    compute::ConstTensorView) noexcept;

// Optional cold attachment. Its immutable source owners outlive provider
// slots; per-run facts describe only uses of the original physical Plan.
class InputSourcesOwner final {
public:
  InputSourcesOwner() noexcept = default;
  explicit InputSourcesOwner(InputSourcesState *state) noexcept : state_(state) {}
  InputSourcesOwner(const InputSourcesOwner &) = delete;
  InputSourcesOwner &operator=(const InputSourcesOwner &) = delete;
  HYLE_GRAPH_API InputSourcesOwner(InputSourcesOwner &&) noexcept;
  HYLE_GRAPH_API InputSourcesOwner &operator=(InputSourcesOwner &&) noexcept;
  HYLE_GRAPH_API ~InputSourcesOwner();
  InputSourcesState *get() const noexcept { return state_; }
  explicit operator bool() const noexcept { return state_ != nullptr; }
  HYLE_GRAPH_API void reset() noexcept;

private:
  InputSourcesState *state_ = nullptr;
};

HYLE_GRAPH_API Status prepare_input_sources(InvocationState &,
    std::span<const QueueBinding>, HostWaitPolicy) noexcept;
HYLE_GRAPH_API void commit_input_sources(InvocationState &) noexcept;
HYLE_GRAPH_API void begin_input_sources(InvocationState &) noexcept;
HYLE_GRAPH_API Result<bool> input_sources_ready(
    InvocationState &, std::size_t segment) noexcept;
HYLE_GRAPH_API Status wait_input_sources(
    InvocationState &, std::size_t segment) noexcept;
HYLE_GRAPH_API Status enqueue_input_sources(InvocationState &, std::size_t segment,
    compute::detail::QueueSequence &) noexcept;
HYLE_GRAPH_API Result<bool> input_results_ready(InvocationState &) noexcept;
HYLE_GRAPH_API Status wait_input_results(InvocationState &) noexcept;
HYLE_GRAPH_API void note_borrowed_input_source(InvocationState &, ValueId) noexcept;
HYLE_GRAPH_API void reset_input_source_preparation(InvocationState &) noexcept;
HYLE_GRAPH_API void reset_input_source_bindings(InvocationState &) noexcept;
HYLE_GRAPH_API bool has_input_source(const InvocationState &, ValueId) noexcept;
HYLE_GRAPH_API bool has_input_source_table(const InvocationState &,
    const InputSourceVTable *) noexcept;
// Copy only an actual binding of this exact adapter table. No producer-name
// inference, new dependency, admission or generic opaque-state cast is allowed.
HYLE_GRAPH_API InputSource bound_input_source(const InvocationState &, ValueId,
    const InputSourceVTable *expected_table) noexcept;
// Commit after complete exact-owner/retirement preflight. Other original
// source bindings survive; this never submits or changes a resource guard.
HYLE_GRAPH_API void release_bound_input_sources(InvocationState &,
    std::span<const ValueId>, std::span<const NodeId>) noexcept;
// Exact retained adapter binding; producer deduplication instead uses same_owner.
HYLE_GRAPH_API bool input_source_matches(const InvocationState &, ValueId,
    const InputSource &) noexcept;
// Exact retained adapter at an original implicit-effect site, independent of
// producer identity deduplication and of its separately registered read rights.
HYLE_GRAPH_API bool effect_source_matches(const InvocationState &, NodeId,
    const InputSource &) noexcept;
HYLE_GRAPH_API Status reconcile_input_sources(
    InvocationState &, const Status &graph_terminal) noexcept;
} // namespace hyle::graph::detail
