#pragma once

#include <span>
#include <utility>

#include "hyle/compute/storage_owner.h"
#include "hyle/graph/detail/resource_source.h"
#include "hyle/graph/executable.h"

namespace hyle::graph {
class SubmissionSource;
class OutputSource;
class StoreSource;
class SourceBinding;

namespace detail {
struct SubmissionSourceState;
struct OutputSourceState;
struct StoreSourceState;
struct OutputSourceAccess;
struct SourceBindingAccess;

// Borrowed call descriptors for typed modules. Only their factories expose a
// SourceBinding; these selected-resource proofs are not engine input policy.
struct SourceBindingData final {
  ValueId input{};
  const OutputSource *output = nullptr;
  const StoreSource *store = nullptr;
  ResourceProjection projection{};
  const Graph *graph = nullptr;
  const ResourceSource *resource = nullptr;
  std::span<const ProjectedReader> readers{};
  std::span<const ResourceReadyInput> ready_inputs{};
  // Actual implicit reads of one selected effect, including an empty set.
  ProjectedWriter effect{};
  std::span<const mem::ConstBufferView> effect_reads{};
  // The real complete Store publication in captured destination order.
  // Empty retains the pending-ticket path.
  std::span<const compute::ReadLease> store_ready{};
};
} // namespace detail

// A typed factory's borrowed binding intent, not readiness or permission.
// Combine values for one atomic bind. Successful binding retains the original
// views and copies required descriptors; this value can then leave scope.
class SourceBinding final {
public:
  SourceBinding() noexcept = default;

private:
  friend struct detail::SourceBindingAccess;
  explicit SourceBinding(detail::SourceBindingData data) noexcept
      : data_(std::move(data)) {}
  detail::SourceBindingData data_{};
};

// One exact original Graph submission: one proxy, Queue and whole verdict.
// Typed views share this identity, but keep their own actual resource proofs.
// Control calls are externally serialized; source observations use one mutex.
// Requires one original shared producer Queue: nonblocking prepared commands
// or complete actual command capture, retaining the original provider runtime.
class SubmissionSource final {
public:
  SubmissionSource() noexcept = default;
  SubmissionSource(const SubmissionSource &) = delete;
  SubmissionSource &operator=(const SubmissionSource &) = delete;
  HYLE_GRAPH_API SubmissionSource(SubmissionSource &&) noexcept;
  HYLE_GRAPH_API SubmissionSource &operator=(SubmissionSource &&) noexcept;
  HYLE_GRAPH_API ~SubmissionSource();

  // Success-only transfer of the exact unobserved submit proxy. The owner
  // retains this metadata's allocator through actual native retirement.
  // Arbitrary Graph metadata keeps its original caller lifetime contract.
  HYLE_GRAPH_API static Result<SubmissionSource> create(Invocation &,
      compute::Completion &&, const compute::StorageOwner &metadata_owner) noexcept;
  bool valid() const noexcept { return state_ != nullptr; }
  HYLE_GRAPH_API Result<compute::CompletionPoll> poll() noexcept;
  HYLE_GRAPH_API Status wait() noexcept;
  // Capability/control errors do not change the poll/wait mathematical verdict
  // or establish retirement of the one original Completion.
  HYLE_GRAPH_API Status request_cancel() noexcept;
  // Closes this generation's exports after all views stopped borrowing and the
  // producer actually retired. Success is a control outcome, not mathematical
  // success. poll/wait retain the original whole verdict after detach.
  // Never waits on pending work; final owner release has ordinary cleanup cost.
  HYLE_GRAPH_API Status detach() noexcept;
  // Same-build typed-module metadata borrow of the actual submitted region.
  // No execution or payload access; no source mutex is held during the visit.
  HYLE_GRAPH_API Status visit(NodeId,
      Status (*)(void *, const detail::RegionAccessView &) noexcept, void *) const noexcept;

private:
  friend class OutputSource;
  friend class StoreSource;
  friend class detail::ResourceSource;
  explicit SubmissionSource(detail::SubmissionSourceState *state) noexcept : state_(state) {}
  detail::SubmissionSourceState *state_ = nullptr;
};

// One fixed nonempty plain dense output of a SubmissionSource. Its exact
// TensorDesc and complete caller-owned writer are preserved independently of
// other outputs. It exposes no pending bytes and creates no second observer.
class OutputSource final {
public:
  OutputSource() noexcept = default;
  OutputSource(const OutputSource &) = delete;
  OutputSource &operator=(const OutputSource &) = delete;
  HYLE_GRAPH_API OutputSource(OutputSource &&) noexcept;
  HYLE_GRAPH_API OutputSource &operator=(OutputSource &&) noexcept;
  HYLE_GRAPH_API ~OutputSource();

  // The original compiled StorageDomain retains this same actual resource
  // owner. The pool proves this full writer's domain, not its other bytes.
  HYLE_GRAPH_API static Result<OutputSource> create(const SubmissionSource &,
      ValueId output, const compute::StorageOwner &, const compute::LeasePool &) noexcept;
  bool valid() const noexcept { return state_ != nullptr; }
  // Borrowed tensor-input intent. bind_sources checks its actual consumer,
  // source owner, full writer, description and range atomically with its peers.
  HYLE_GRAPH_API SourceBinding reader_binding(ValueId input) const noexcept;
  HYLE_GRAPH_API Result<compute::CompletionPoll> poll() noexcept;
  HYLE_GRAPH_API Status wait() noexcept;
  // Only whole success AND actual producer retirement publish the original
  // read lease. Already admitted ordered readers may remain in flight.
  // Pending/error retains the borrow; SubmissionSource::detach closes exports.
  HYLE_GRAPH_API Status detach() noexcept;

private:
  friend struct detail::OutputSourceAccess;
  explicit OutputSource(detail::OutputSourceState *state) noexcept : state_(state) {}
  detail::OutputSourceState *state_ = nullptr;
};

// Consecutive original destination ordinals, not physical slot numbers.
// A nonempty range is selected before the producer submits.
struct StoreDestinationRange final {
  std::size_t first = 0, count = 0;
};

// Independent original Store destination permissions, not another bank/pool
// or ready frame. The SubmissionSource owns the sole terminal contract.
class StoreSource final {
public:
  StoreSource() noexcept = default;
  StoreSource(const StoreSource &) = delete;
  StoreSource &operator=(const StoreSource &) = delete;
  HYLE_GRAPH_API StoreSource(StoreSource &&) noexcept;
  HYLE_GRAPH_API StoreSource &operator=(StoreSource &&) noexcept;
  HYLE_GRAPH_API ~StoreSource();

  // Reserve fixed metadata before producer submission using the original
  // budget and actual StorageOwner allocator. No writer is retained or sealed.
  HYLE_GRAPH_API static Result<StoreSource> prepare(Invocation &, NodeId store,
      const compute::StorageOwner &, ResourceOptions = {}) noexcept;
  // Retain only this consecutive read footprint while revalidating the whole
  // original transaction. I32 IDs remain a bounded original ready subview.
  HYLE_GRAPH_API static Result<StoreSource> prepare(Invocation &, NodeId store,
      const compute::StorageOwner &, StoreDestinationRange, ResourceOptions) noexcept;
  bool prepared() const noexcept { return state_ != nullptr; }
  HYLE_GRAPH_API bool valid() const noexcept;
  // After original submit, revalidate transaction, controls, coordinates and
  // owner before sealing destination tickets. No allocation. A late rejection
  // drops unescaped tickets; original writers remain for retired recovery.
  HYLE_GRAPH_API Status capture(const SubmissionSource &) noexcept;
  // Selected destination count, not the original ordinal upper bound.
  HYLE_GRAPH_API std::size_t destination_count() const noexcept;
  // The prepared imported Store proves its real Gather projection. A ready
  // span must contain this selected range's exact complete publications in
  // original order; nonconsecutive selections are not supported.
  HYLE_GRAPH_API Result<SourceBinding> reader_binding(Invocation &, const Graph &,
      NodeId store, std::span<const compute::ReadLease> ready = {}) const noexcept;
  // A standalone checked Gather can read the same original Store without a
  // consumer Store. Currently selects one real logical reader; its actual
  // input/ID association and selected implementation are still validated.
  // The reader span is borrowed until bind_sources succeeds.
  HYLE_GRAPH_API Result<SourceBinding> reader_binding(Invocation &, const Graph &,
      std::span<const NodeId> readers,
      std::span<const compute::ReadLease> ready = {}) const noexcept;
  // Whole success and actual retirement authorize the returned original
  // writer's exact destination ticket. Already admitted readers may remain.
  // Destination is always an original ordinal; rejection preserves the writer.
  HYLE_GRAPH_API Result<compute::ReadLease> publish(std::size_t destination,
      compute::WriteLease &returned_writer) const noexcept;

private:
  friend struct detail::OutputSourceAccess;
  explicit StoreSource(detail::StoreSourceState *state) noexcept : state_(state) {}
  detail::StoreSourceState *state_ = nullptr;
};

// Atomically bind distinct typed inputs/effects from the same submission.
// This retains all actual views even when their terminal dependency is shared.
// A view binds one original consumer bundle once; its submission may serve
// other independent views. Rejection preserves old bindings and view identity.
// After success no descriptor span or caller proof mirror is retained. The
// Invocation's existing bundle owns the exact facts used by later operations.
// Binding alone does not certify replay or grant future resource admission.
HYLE_GRAPH_API Status bind_sources(Invocation &, std::span<const SourceBinding>) noexcept;

// Prepare nonblocking Commands or reuse complete existing capture coverage.
// Pending sources cannot record or fall back to managed Commands. A successful
// retired source with real ready outputs/Store/pages can make its first capture.
// Every call still checks the complete current physical tuple and real Queue.
HYLE_GRAPH_API Status prepare_sources(Invocation &, std::span<const QueueBinding>) noexcept;
// Submit this Invocation's actual retained source bundle once, through the
// original ordered Queue and exact resource permissions. Original producer
// errors and real retirement remain part of the complete terminal outcome.
HYLE_GRAPH_API Result<compute::Completion> submit_sources(Invocation &,
    std::span<const QueueBinding>) noexcept;
// Retired cleanup, including failed/closed episodes. Take affected original
// writers first. Withdraw only the actual complete bundle; missing, replaced
// or still-live attachments reject. Qualified dormant captures may survive,
// with no remaining read authority and no implicit publication.
HYLE_GRAPH_API Status release_sources(Invocation &) noexcept;
} // namespace hyle::graph
