#pragma once

#include <cstdint>
#include <span>

#include "hyle/error/result.h"
#include "hyle/graph/visibility.h"

namespace hyle::compute { class ReadLease; class ConstTensorView; }

namespace hyle::graph {
class ValueId;
class Invocation;
namespace detail {
struct InvocationState;
struct ResourceEpisode;

// Cold opt-in owned bindings, permissions and typed payloads in the original
// InvocationState. Submission retirement is independent; provider invocations
// are destroyed before these external resource owners and payloads.
class ResourceEpisodeOwner final {
public:
  ResourceEpisodeOwner() noexcept = default;
  explicit ResourceEpisodeOwner(ResourceEpisode *state) noexcept : state_(state) {}
  ResourceEpisodeOwner(const ResourceEpisodeOwner &) = delete;
  ResourceEpisodeOwner &operator=(const ResourceEpisodeOwner &) = delete;
  HYLE_GRAPH_API ResourceEpisodeOwner(ResourceEpisodeOwner &&) noexcept;
  HYLE_GRAPH_API ResourceEpisodeOwner &operator=(ResourceEpisodeOwner &&) noexcept;
  HYLE_GRAPH_API ~ResourceEpisodeOwner();
  ResourceEpisode *get() const noexcept { return state_; }
  explicit operator bool() const noexcept { return state_ != nullptr; }
  HYLE_GRAPH_API void reset() noexcept;

private:
  ResourceEpisode *state_ = nullptr;
};

HYLE_GRAPH_API Status resource_control_ready(const InvocationState *) noexcept;
HYLE_GRAPH_API Status reset_resource_payload(Invocation &) noexcept;
HYLE_GRAPH_API void reset_resource_bindings(InvocationState &) noexcept;
// Clears cold configuration/projected owners and views, retaining caller ports.
HYLE_GRAPH_API void reset_configuration_bindings(InvocationState &) noexcept;
// Physical TensorView metadata only. Current input completeness, permissions,
// source preparation and selected native seals remain the next episode's work.
HYLE_GRAPH_API void remember_configuration_spans(InvocationState &) noexcept;
HYLE_GRAPH_API bool reusable_configuration_spans(const InvocationState &) noexcept;
HYLE_GRAPH_API void discard_configuration_spans(InvocationState &) noexcept;
// Ordinary replacement must not detach immutable metadata from the selected
// cold configuration. This check precedes all binding/cache/source mutation.
HYLE_GRAPH_API Status check_binding_replacement(
    const InvocationState &, ValueId) noexcept;
// Called only after a raw TensorView binding succeeds. Its storage remains
// explicitly borrowed; a later owned bind replaces this marker with its lease.
HYLE_GRAPH_API void note_borrowed_binding(InvocationState &, ValueId) noexcept;
// Borrow only a genuinely owned ready read covering this exact physical view.
// Native cold adapters may copy it into an independent array owner.
// Candidates belong to one guarded binding-preparation call. They preserve
// binding order, not permissions or successful range matches; clear on exit.
HYLE_GRAPH_API Result<std::span<const std::uint32_t>>
collect_input_lease_candidates(InvocationState &) noexcept;
HYLE_GRAPH_API void clear_input_lease_candidates(InvocationState &) noexcept;
HYLE_GRAPH_API const compute::ReadLease *input_lease(
    const InvocationState &, compute::ConstTensorView,
    std::span<const std::uint32_t> candidates) noexcept;
} // namespace detail
} // namespace hyle::graph
