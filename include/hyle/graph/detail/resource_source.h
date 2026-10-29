#pragma once

#include "hyle/compute/detail/lease_access.h"
#include "hyle/graph/detail/region_access.h"

namespace hyle::graph { class SubmissionSource; }

namespace hyle::graph::detail {
struct ResourceSourceState;
class ResourceSource;
struct ResourceSourceCounts final {
  std::size_t ready = 0, full = 0, append = 0, inherited = 0;
};
struct ResourceFullRange final {
  const compute::WriteLease *writer = nullptr;
  mem::ConstBufferView range{};
  std::size_t readable_bytes = 0;
};
struct ResourceAppendRange final {
  const compute::detail::AppendLease *appender = nullptr;
  mem::ConstBufferView prefix{};
};
// Only the bounded permission is inherited. Capture flattens it to its original
// resource owner; it never retains the preceding Source or terminal observer.
struct ResourceInheritance final {
  const ResourceSource *source = nullptr;
  mem::ConstBufferView prefix{};
};
// Call-scoped original resource registrations. The typed module proves its
// layout and initialized prefixes; core independently proves this exact
// selected writer and every actual registered ready/full/tail permission.
struct ResourceContribution final {
  ProjectedWriter writer{};
  ValueId written_projection{};
  compute::ConstTensorView capacity{};
  std::span<const compute::ReadLease> ready{};
  std::span<const ResourceFullRange> full{};
  std::span<const ResourceAppendRange> append{};
  std::span<const ResourceInheritance> inherited{};
};
class ResourceSource final {
public:
  ResourceSource() noexcept = default;
  ResourceSource(const ResourceSource &) = delete;
  ResourceSource &operator=(const ResourceSource &) = delete;
  HYLE_GRAPH_API ResourceSource(ResourceSource &&) noexcept;
  HYLE_GRAPH_API ResourceSource &operator=(ResourceSource &&) noexcept;
  HYLE_GRAPH_API ~ResourceSource();
  // Fixed, budgeted metadata only. No movable transaction or writer is borrowed.
  HYLE_GRAPH_API static Result<ResourceSource> prepare(Invocation &,
      ResourceSourceCounts, const compute::StorageOwner &, ResourceOptions = {}) noexcept;
  bool prepared() const noexcept { return state_ != nullptr; }
  HYLE_GRAPH_API bool valid() const noexcept;
  // After the original submission, during its metadata-only visit. All
  // authority checks precede sealing; no caller pointer survives this call.
  HYLE_GRAPH_API Status capture(const SubmissionSource &, ResourceContribution) noexcept;
  HYLE_GRAPH_API compute::ConstTensorView capacity() const noexcept;
  HYLE_GRAPH_API std::span<const compute::ReadLease> ready_pages() const noexcept;
  HYLE_GRAPH_API std::span<const compute::detail::PendingReadTicket> full_tickets() const noexcept;
  HYLE_GRAPH_API std::span<const compute::detail::PendingAppendReadTicket> append_tickets() const noexcept;
  // Typed publication must additionally preserve its own layout/frontier
  // contract. These tickets alone never establish whole-producer success.
  HYLE_GRAPH_API Status successful_outcome() const noexcept;
  // After successful actual retirement, record the real canonical ready guards.
  // Own origins are updated; inherited guards are checked without widening them.
  // Idempotent, and never waits for a pending producer.
  HYLE_GRAPH_API Status publish_ready(std::span<const compute::ReadLease>) const noexcept;
  // Bounded initialized/prospective prefix of this original contribution.
  // Pure identity/range query, never a ready view or new work admission.
  HYLE_GRAPH_API Status covers_range(mem::ConstBufferView) const noexcept;
private:
  friend struct OutputSourceAccess;
  explicit ResourceSource(ResourceSourceState *state) noexcept : state_(state) {}
  ResourceSourceState *state_ = nullptr;
};
struct ResourceReadyInput final {
  ValueId input{};
  compute::ConstTensorView tensor{};
  compute::ReadLease read{};
  // Exactly one of input or region identifies the original ready control.
  // Private controls belong to the selected region itself, not a synthetic
  // Graph input. Their actual guard is re-queried at bind/prepare/submit.
  NodeId region{};
  std::uint32_t ordinal = 0;
};
} // namespace hyle::graph::detail
