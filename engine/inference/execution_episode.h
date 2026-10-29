#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <span>
#include <utility>
#include <vector>

#include "hyle/container/small_vector.h"
#include "hyle/core/assert.h"
#include "hyle/graph/batch.h"
#include "submission_views.h"
#include "append_transactions.h"
#include "program_carrier.h"
#include "slot_transactions.h"

namespace hyle::inference {

template<class Sources> struct IncomingExecutionSources final {
  std::shared_ptr<Sources> owner;
  std::vector<graph::detail::OutputBinding> bindings;
  std::vector<graph::ValueId> outside_inputs;
  bool bound = false;
};

// The resource part of one real Graph episode. Embed it after the model's
// participant pins so those pins outlive its resources. Sources is the model's
// existing submission-view bundle (submission and token); it may also carry
// model-specific boundary facts. No new executor, permission or retirement
// authority is introduced. The carrier, Store cohort and admission count are
// borrowed from their original owners.
template<class Scope, class Sources, std::size_t InlineBanks>
class ExecutionEpisode final {
 public:
  container::SmallVector<std::vector<compute::PagedPageWriter>, InlineBanks> dirty;
  container::SmallVector<container::SmallVector<compute::PagedPageAppender, 1>, InlineBanks> tails;
  container::SmallVector<graph::batch::PagedAppendPages, InlineBanks> appends;
  container::SmallVector<graph::NodeId, InlineBanks> nodes;
  container::SmallVector<std::uint8_t, InlineBanks> bound;
  graph::Invocation *run = nullptr;
  ProgramCarrier<Scope> *carrier = nullptr;
  compute::Completion completion;
  std::shared_ptr<Sources> sources;
  std::unique_ptr<IncomingExecutionSources<Sources>> incoming;
  // An aliasing shared_ptr pins the original model episode, not a new owner
  // or allocation. Only its original Invocation is consulted during cleanup.
  std::shared_ptr<ExecutionEpisode> predecessor_episode;
  Status terminal;
  SlotTransactions *state_transactions = nullptr;
  bool cleanup_pending = false;
  std::size_t *candidate_admission = nullptr;

  void configure(std::size_t banks) {
    if (!dirty.try_resize(banks) || !tails.try_resize(banks) || !appends.try_resize(banks) ||
        !nodes.try_resize(banks) || !bound.try_resize(banks)) throw std::bad_alloc{};
  }
  AppendTransactions append_transactions() noexcept {
    return {{nodes.data(), nodes.size()}, {appends.data(), appends.size()}, {bound.data(), bound.size()}};
  }
  bool has_completion() const noexcept {
    return completion.valid() || (sources && sources->submission.valid());
  }
  Result<compute::CompletionPoll> poll() noexcept {
    return sources && sources->submission.valid() ? sources->submission.poll() : completion.poll();
  }
  Status wait() noexcept {
    return sources && sources->submission.valid() ? sources->submission.wait() : completion.wait();
  }
  Status request_cancel() noexcept {
    return sources && sources->submission.valid() ? sources->submission.request_cancel() : completion.request_cancel();
  }

  // Keep these steps separate: imported Source bindings depend on the actual
  // prepared Append cohort, and model IO may be bound before replay admission.
  Status prepare_appends(const graph::Graph &logical,
      std::span<const graph::batch::NodeLayout> layouts,
      std::size_t first, std::size_t count,
      std::span<const graph::ValueId> cache_inputs,
      graph::batch::PrepareOptions options = {}) noexcept {
    if (!run || !carrier)
      return Status::FailedPrecondition("execution episode has no original carrier");
    if (first > nodes.size() || count > nodes.size() - first ||
        first > appends.size() || count > appends.size() - first ||
        first > bound.size() || count > bound.size() - first)
      return Status::OutOfRange("append preparation exceeds the original episode cohort");
    HYLE_RETURN_IF_ERROR(graph::batch::prepare(*run, carrier->queues(), layouts, options));
    return append_transactions().subspan(first, count).bind(*run, logical, cache_inputs);
  }
  Status bind_consumers(const graph::Graph &state_graph) noexcept {
    if (!run || !carrier)
      return Status::FailedPrecondition("execution episode has no original carrier");
    if (incoming) {
      HYLE_RETURN_IF_ERROR(graph::detail::bind_outputs(*run, incoming->bindings));
      incoming->bound = true;
    }
    return state_transactions ? state_transactions->bind(*run, state_graph) : Status::Ok();
  }
  Status prepare_replay(graph::ReplayOptions options = {}) noexcept {
    if (!run || !carrier)
      return Status::FailedPrecondition("execution episode has no original carrier");
    return incoming ? graph::detail::prepare_outputs(*run, incoming->bindings, carrier->queues())
                    : run->prepare_replay(carrier->queues(), options);
  }

  // All routes submit through the same already selected original carrier.
  // Model policy still decides when to submit and when an outcome may publish.
  Status submit(const std::shared_ptr<Scope> &scope) noexcept {
    if (!run || !carrier)
      return Status::FailedPrecondition("execution episode has no original carrier");
    HYLE_ASSIGN_OR_RETURN(completion, carrier->submit(scope,
        [&](graph::Invocation &invocation, std::span<const graph::QueueBinding> queues) noexcept {
          return incoming ? graph::detail::submit_outputs(invocation, incoming->bindings, queues)
                          : invocation.submit(queues);
        }));
    return Status::Ok();
  }
  Status wait_success() noexcept {
    HYLE_RETURN_IF_ERROR(wait());
    completion = {};
    return Status::Ok();
  }

  // Withdraw only the original complete Source bundle. Recovery may need
  // this after close, so it uses Graph's retired-control operation directly,
  // never a new carrier admission. Record successful withdrawal immediately:
  // a later ordinary-input failure must not retry an already released bundle.
  Status release_source_bindings() noexcept {
    if (!run || !carrier)
      return Status::FailedPrecondition("execution episode has no original carrier");
    if (incoming && incoming->bound) {
      HYLE_RETURN_IF_ERROR(graph::detail::release_outputs(*run, incoming->bindings));
      incoming->bound = false;
    }
    return Status::Ok();
  }

  // Call only after the original outputs have been retained and Store/Append
  // transactions taken. Model policy decides when this may precede commit.
  // Keep the incoming owner and predecessor until both withdrawals succeed;
  // a failed preparation with no Source binding must not invent one to release.
  Status release_consumers(std::span<const graph::ValueId> ordinary_inputs) noexcept {
    if (!run || !carrier)
      return Status::FailedPrecondition("execution episode has no original carrier");
    if (!incoming) return carrier->release_inputs(ordinary_inputs);
    HYLE_RETURN_IF_ERROR(release_source_bindings());
    HYLE_RETURN_IF_ERROR(carrier->release_inputs(incoming->outside_inputs));
    incoming.reset(); predecessor_episode.reset();
    return Status::Ok();
  }

  // This drives only the original Backend/carrier. Its Status and its actual
  // retirement are distinct; neither observation grants publication rights.
  template<class Environment>
  RetirementObservation retire(Environment &environment) noexcept {
    if (!run) return {true, {}};
    if (!carrier) return {false, Status::FailedPrecondition("execution episode lost its original carrier")};
    const auto observed = carrier->retire(environment);
    record_retirement_status(terminal, observed.status);
    return observed;
  }
  void release_admission() noexcept {
    if (candidate_admission) {
      auto *count = std::exchange(candidate_admission, nullptr);
      HYLE_FATAL_ASSERT(*count); --*count;
    }
  }

  // Nonblocking recovery of the original transactions, not an alternate
  // rollback system. Keep every remaining fact on a partial return. Recycle
  // is the model's nonthrowing final candidate discard, never publication.
  template<class Recycle>
  Result<bool> recover(Recycle &&recycle) noexcept {
    static_assert(noexcept(std::forward<Recycle>(recycle)()));
    // Destroying an unsubmitted successor's capture may synchronize its
    // context. Preserve the original predecessor gate before any withdrawal.
    if (cleanup_pending && predecessor_episode && predecessor_episode->run &&
        predecessor_episode->run->in_flight()) return false;
    if (has_completion()) {
      auto observed = poll();
      if (observed && *observed == compute::CompletionPoll::kPending) return false;
      if (!observed) record_retirement_status(terminal, observed.error());
      completion = {}; // Native guards survive an error or abandonment.
    }
    if (sources) {
      if (sources->submission.valid() && sources.use_count() != 1) return false;
      sources->clear_outputs();
      if (sources->submission.valid()) {
        const auto detached = sources->submission.detach();
        if (detached.IsUnavailable() || detached.IsFailedPrecondition()) return false;
        HYLE_RETURN_IF_ERROR(detached);
      }
      sources.reset();
    }
    if (run) {
      if (!carrier)
        return std::unexpected(Status::FailedPrecondition("execution episode lost its original carrier"));
      if (run->in_flight() || carrier->pending()) return false;
      HYLE_ASSIGN_OR_RETURN(const bool appends_returned, append_transactions().recover(*run));
      if (!appends_returned) return false;
      if (state_transactions) {
        HYLE_ASSIGN_OR_RETURN(const bool recovered, state_transactions->recover(*run));
        if (!recovered) return false;
      }
      HYLE_RETURN_IF_ERROR(release_source_bindings());
      incoming.reset(); predecessor_episode.reset();
      HYLE_RETURN_IF_ERROR(carrier->reset_retired());
    }
    HYLE_ASSIGN_OR_RETURN(const bool appends_discarded, append_transactions().discard());
    if (!appends_discarded) return false;
    for (auto &bank : tails) for (auto &tail : bank) {
      const auto rolled_back = AppendTransactions::rollback_tail(tail);
      if (rolled_back.IsUnavailable()) return false;
      HYLE_RETURN_IF_ERROR(rolled_back);
    }
    for (auto &bank : dirty) for (auto &writer : bank) {
      const auto recovered = AppendTransactions::discard_page(writer);
      if (recovered.IsUnavailable()) return false;
      HYLE_RETURN_IF_ERROR(recovered);
    }
    for (auto &append : appends) append = {};
    for (auto &bank : dirty) bank.clear();
    for (auto &bank : tails) bank.clear();
    std::forward<Recycle>(recycle)();
    run = nullptr; carrier = nullptr; state_transactions = nullptr;
    release_admission();
    return true;
  }
};

} // namespace hyle::inference
