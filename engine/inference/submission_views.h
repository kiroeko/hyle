#pragma once

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
  graph::detail::OutputSource token;
  std::vector<graph::detail::StoreSource> states;
  std::vector<graph::batch::detail::PagedAppendSource> pages;

  // Cold metadata only. Keep the original reserve and preparation order;
  // allocation exceptions and partially prepared views retain their meaning.
  Status prepare(graph::Invocation &run,
      std::span<const graph::slots::SourceBinding> stores,
      std::span<const graph::NodeId> appends,
      const compute::StorageOwner &storage,
      const compute::ResourceBudget &budget) {
    if (submission.valid() || token.valid() || !states.empty() || !pages.empty())
      return Status::FailedPrecondition("submission view preparation requires a fresh original bundle");
    states.reserve(stores.size());
    pages.reserve(appends.size());
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
      graph::ValueId token_output, const compute::StorageOwner &storage,
      const compute::LeasePool &token_pool, bool optional = false) noexcept {
    if (submission.valid() || token.valid())
      return std::unexpected(Status::FailedPrecondition("submission views already captured their original producer"));
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
    HYLE_ASSIGN_OR_RETURN(token, graph::detail::OutputSource::create(
        submission, token_output, storage, token_pool));
    for (auto &state : states) HYLE_RETURN_IF_ERROR(state.capture(submission));
    for (auto &page : pages) HYLE_RETURN_IF_ERROR(page.capture(submission));
    return true;
  }
};

} // namespace hyle::inference
