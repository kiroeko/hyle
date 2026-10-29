#pragma once

#include <array>
#include <cstddef>
#include <expected>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

#include "hyle/graph/detail/paged_append_source.h"
#include "hyle/graph/detail/store_source.h"
#include "hyle/graph/slots.h"

namespace hyle::inference {

// Typed views of one original producer submission. The enclosing episode
// keeps its existing owner and recovery protocol; these views introduce no
// independent Completion, publication authority or submission path.
struct SubmissionViews {
  graph::detail::SubmissionSource submission;
  // First selected tensor, such as a generated ID or an inter-program hidden
  // value. Its original description determines type; no token dtype is implied.
  graph::detail::OutputSource primary_output;
  std::vector<graph::detail::StoreSource> states;
  std::vector<graph::batch::detail::PagedAppendSource> pages;
  std::vector<graph::detail::OutputSource> extra_outputs;

  // Borrowed only for capture. Each output keeps its own actual storage owner;
  // its pool proves that complete original writer. Graph checks the owner's
  // resource and Context against the output and submission metadata owner.
  struct OutputCapture final {
    graph::ValueId output;
    const compute::StorageOwner &storage;
    const compute::LeasePool &pool;
  };
  struct StoreCapture final {
    graph::NodeId store;
    const compute::StorageOwner &storage;
  };
  struct PageCapture final {
    graph::NodeId append;
    const compute::StorageOwner &storage;
  };

  // Cold metadata only. Keep the original reserve and preparation order;
  // allocation exceptions and partially prepared views retain their meaning.
  // Ranges and their borrowed descriptors are consumed only during this call.
  template<std::ranges::sized_range Stores = std::span<const StoreCapture>,
      std::ranges::sized_range Pages = std::span<const PageCapture>>
  Status prepare(graph::Invocation &run, Stores &&stores, Pages &&appends,
      const compute::ResourceBudget &budget, std::size_t extra_output_count = 0) {
    if (submission.valid() || primary_output.valid() || !states.empty() || !pages.empty() || !extra_outputs.empty())
      return Status::FailedPrecondition("submission view preparation requires a fresh original bundle");
    states.reserve(std::ranges::size(stores));
    pages.reserve(std::ranges::size(appends));
    extra_outputs.resize(extra_output_count);
    for (const auto &store : stores) {
      HYLE_ASSIGN_OR_RETURN(auto view, graph::detail::StoreSource::prepare(
          run, store.store, store.storage, {.resource_budget = budget}));
      states.push_back(std::move(view));
    }
    for (const auto &append : appends) {
      HYLE_ASSIGN_OR_RETURN(auto view, graph::batch::detail::PagedAppendSource::prepare(
          run, append.append, append.storage, {.resource_budget = budget}));
      pages.push_back(std::move(view));
    }
    return Status::Ok();
  }

  Result<bool> capture(graph::Invocation &run, compute::Completion &completion,
      graph::ValueId output, const compute::StorageOwner &storage,
      const compute::LeasePool &pool, bool optional = false) noexcept {
    const std::array outputs{OutputCapture{output, storage, pool}};
    return capture_outputs(run, completion, outputs, storage, optional);
  }

  // The first output occupies primary_output; the rest occupy fixed cold slots.
  // Capture never grows the cohort or copies an owner/pool descriptor.
  Result<bool> capture_outputs(graph::Invocation &run, compute::Completion &completion,
      std::span<const OutputCapture> outputs, const compute::StorageOwner &metadata_owner,
      bool optional = false) noexcept {
    if (submission.valid() || primary_output.valid())
      return std::unexpected(Status::FailedPrecondition("submission views already captured their original producer"));
    for (const auto &output : extra_outputs)
      if (output.valid())
        return std::unexpected(Status::FailedPrecondition("submission views already captured an extra output"));
    if (outputs.empty() || outputs.size() - 1 != extra_outputs.size())
      return std::unexpected(Status::InvalidArgument("output capture count differs from its prepared submission views"));
    auto exported = graph::detail::SubmissionSource::create(run, std::move(completion), metadata_owner);
    if (!exported) {
      const auto &error = exported.error();
      // Only create's success-only transfer may decline. Once transferred,
      // every later failure keeps this bundle for the original recovery.
      if (optional && !error.has_native_error() &&
          (error.IsUnimplemented() || error.IsUnavailable())) return false;
      return std::unexpected(error);
    }
    submission = std::move(*exported);
    HYLE_ASSIGN_OR_RETURN(primary_output, graph::detail::OutputSource::create(
        submission, outputs[0].output, outputs[0].storage, outputs[0].pool));
    for (std::size_t i = 0; i < extra_outputs.size(); ++i) {
      HYLE_ASSIGN_OR_RETURN(extra_outputs[i], graph::detail::OutputSource::create(
          submission, outputs[i + 1].output, outputs[i + 1].storage, outputs[i + 1].pool));
    }
    for (auto &state : states) HYLE_RETURN_IF_ERROR(state.capture(submission));
    for (auto &page : pages) HYLE_RETURN_IF_ERROR(page.capture(submission));
    return true;
  }

  // Only whole-producer success and actual retirement may publish these
  // original writers. A partial failure leaves every remaining view intact.
  Status detach_outputs() noexcept {
    HYLE_RETURN_IF_ERROR(primary_output.detach());
    for (auto &output : extra_outputs) HYLE_RETURN_IF_ERROR(output.detach());
    return Status::Ok();
  }

  // Recovery drops all tensor borrows before SubmissionSource::detach().
  // Keep the cold cohort's size and capacity; this does not prove retirement.
  void clear_outputs() noexcept {
    primary_output = {};
    for (auto &output : extra_outputs) output = {};
  }
};

} // namespace hyle::inference
