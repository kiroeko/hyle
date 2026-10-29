#pragma once

#include <array>
#include <cstddef>
#include <expected>
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

  // Borrowed only for capture. Each pool proves its own complete original
  // writer; all outputs share this submission's actual storage owner.
  struct OutputCapture final {
    graph::ValueId output;
    const compute::LeasePool &pool;
  };

  // Cold metadata only. Keep the original reserve and preparation order;
  // allocation exceptions and partially prepared views retain their meaning.
  Status prepare(graph::Invocation &run,
      std::span<const graph::slots::SourceBinding> stores,
      std::span<const graph::NodeId> appends,
      const compute::StorageOwner &storage,
      const compute::ResourceBudget &budget, std::size_t extra_output_count = 0) {
    if (submission.valid() || primary_output.valid() || !states.empty() || !pages.empty() || !extra_outputs.empty())
      return Status::FailedPrecondition("submission view preparation requires a fresh original bundle");
    states.reserve(stores.size());
    pages.reserve(appends.size());
    extra_outputs.resize(extra_output_count);
    for (const auto &store : stores) {
      HYLE_ASSIGN_OR_RETURN(auto view, graph::detail::StoreSource::prepare(
          run, store.store, storage, {.resource_budget = budget}));
      states.push_back(std::move(view));
    }
    for (const auto append : appends) {
      HYLE_ASSIGN_OR_RETURN(auto view, graph::batch::detail::PagedAppendSource::prepare(
          run, append, storage, {.resource_budget = budget}));
      pages.push_back(std::move(view));
    }
    return Status::Ok();
  }

  Result<bool> capture(graph::Invocation &run, compute::Completion &completion,
      graph::ValueId output, const compute::StorageOwner &storage,
      const compute::LeasePool &pool, bool optional = false) noexcept {
    const std::array outputs{OutputCapture{output, pool}};
    return capture_outputs(run, completion, outputs, storage, optional);
  }

  // The first output occupies primary_output; the rest occupy fixed cold slots.
  // Capture never grows the cohort or copies an owner/pool descriptor.
  Result<bool> capture_outputs(graph::Invocation &run, compute::Completion &completion,
      std::span<const OutputCapture> outputs, const compute::StorageOwner &storage,
      bool optional = false) noexcept {
    if (submission.valid() || primary_output.valid())
      return std::unexpected(Status::FailedPrecondition("submission views already captured their original producer"));
    for (const auto &output : extra_outputs)
      if (output.valid())
        return std::unexpected(Status::FailedPrecondition("submission views already captured an extra output"));
    if (outputs.empty() || outputs.size() - 1 != extra_outputs.size())
      return std::unexpected(Status::InvalidArgument("output capture count differs from its prepared submission views"));
    auto exported = graph::detail::SubmissionSource::create(run, std::move(completion), storage);
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
        submission, outputs[0].output, storage, outputs[0].pool));
    for (std::size_t i = 0; i < extra_outputs.size(); ++i) {
      HYLE_ASSIGN_OR_RETURN(extra_outputs[i], graph::detail::OutputSource::create(
          submission, outputs[i + 1].output, storage, outputs[i + 1].pool));
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
