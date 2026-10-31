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
#include "hyle/graph/detail/replay_access.h"
#include "submission_views.h"
#include "append_transactions.h"
#include "program_carrier.h"
#include "slot_transactions.h"

namespace hyle::inference {

template<class Sources> struct IncomingExecutionSources final {
  std::shared_ptr<Sources> owner;
  // Call descriptors live only until atomic binding succeeds. The original
  // Invocation then retains the accepted facts; capacity stays reusable.
  std::vector<graph::SourceBinding> bindings;
  bool bound = false;
};

// The resource part of one real Graph episode. Embed it after the model's
// participant pins so those pins outlive its resources. Sources is the model's
// existing submission-view bundle (submission and selected outputs); it may carry
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
      HYLE_RETURN_IF_ERROR(graph::bind_sources(*run, incoming->bindings));
      incoming->bound = true;
      incoming->bindings.clear();
    }
    return state_transactions ? state_transactions->bind(*run, state_graph) : Status::Ok();
  }
  Status prepare_replay(graph::ReplayOptions options = {}) noexcept {
    if (!run || !carrier)
      return Status::FailedPrecondition("execution episode has no original carrier");
    return incoming ? graph::prepare_sources(*run, carrier->queues())
                    : run->prepare_replay(carrier->queues(), options);
  }

  // A pending speculative consumer can only reuse recorded coverage. A ready
  // Source retains ordinary preparation and may record, exactly as before.
  // This typed false comes from Graph's exact final pending coverage test,
  // after its original Source validation; no Status code/string infers it.
  Status prepare_successor_replay() noexcept {
    if (!run || !carrier || !incoming || !incoming->bound ||
        !predecessor_episode || !predecessor_episode->run ||
        has_completion() || sources || unsubmitted_reuse_declined_)
      return Status::FailedPrecondition("successor replay requires one original unsubmitted incoming episode");
    HYLE_ASSIGN_OR_RETURN(auto original, carrier->borrow());
    if (original != run)
      return Status::InvalidArgument("successor replay belongs to another original Invocation");
    HYLE_ASSIGN_OR_RETURN(const bool prepared,
        graph::detail::ReplayAccess::try_prepare_sources(*run, carrier->queues()));
    if (!prepared) {
      // Recovery waits for A before withdrawing B's original transactions.
      // This is an episode-local failure origin, never a future reuse hint.
      unsubmitted_reuse_declined_ = true;
      cleanup_pending = true;
      return Status::Unavailable("pending outputs have no complete existing capture for their final physical tuple");
    }
    return Status::Ok();
  }
  bool unsubmitted_reuse_declined() const noexcept {
    return unsubmitted_reuse_declined_;
  }

  // All routes submit through the same already selected original carrier.
  // Model policy still decides when to submit and when an outcome may publish.
  Status submit(const std::shared_ptr<Scope> &scope) noexcept {
    if (!run || !carrier)
      return Status::FailedPrecondition("execution episode has no original carrier");
    if (unsubmitted_reuse_declined_)
      return Status::FailedPrecondition("typed reuse refusal cannot submit its original episode");
    HYLE_ASSIGN_OR_RETURN(completion, carrier->submit(scope,
        [&](graph::Invocation &invocation, std::span<const graph::QueueBinding> queues) noexcept {
          return incoming ? graph::submit_sources(invocation, queues)
                          : invocation.submit(queues);
        }));
    return Status::Ok();
  }
  Status wait_success() noexcept {
    HYLE_RETURN_IF_ERROR(wait());
    completion = {};
    return Status::Ok();
  }

  // Complete one original successful Graph stage, before any model Head is
  // published or its Invocation is reset. A nonthrowing completion observer
  // keeps native-success accounting before any output withdrawal can fail.
  // The two synchronous model actions then observe/retain actual outputs and
  // interpret returned transactions. They may allocate or throw as before;
  // this algorithm stores no callback,
  // adds no owner and never retries a partially completed stage. A refusal or
  // exception goes to the caller's existing recover() with every remaining
  // transaction, Source and predecessor still in its original container.
  template<class OnCompletion, class Observe, class Complete>
  Status finish_success(std::size_t first_append, std::size_t append_count,
      std::span<const graph::ValueId> ordinary_inputs,
      OnCompletion &&on_completion, Observe &&observe, Complete &&complete) {
    HYLE_RETURN_IF_ERROR(check_finish_success(first_append, append_count));
    return finish_success_impl(first_append, append_count, ordinary_inputs,
        std::forward<OnCompletion>(on_completion), std::forward<Observe>(observe),
        std::forward<Complete>(complete));
  }

  // Observe the same original Completion without waiting. Pending preserves
  // all callbacks, transactions and Source owners for the next observation.
  // Once complete, its settled terminal is consumed by the ordinary finish
  // algorithm; model callbacks remain synchronous and may throw as before.
  template<class OnCompletion, class Observe, class Complete>
  Result<bool> try_finish_success(std::size_t first_append, std::size_t append_count,
      std::span<const graph::ValueId> ordinary_inputs,
      OnCompletion &&on_completion, Observe &&observe, Complete &&complete) {
    HYLE_RETURN_IF_ERROR(check_finish_success(first_append, append_count));
    HYLE_ASSIGN_OR_RETURN(auto observed, poll());
    if (observed == compute::CompletionPoll::kPending) return false;
    HYLE_RETURN_IF_ERROR(finish_success_impl(first_append, append_count, ordinary_inputs,
        std::forward<OnCompletion>(on_completion), std::forward<Observe>(observe),
        std::forward<Complete>(complete)));
    return true;
  }

  // Withdraw only the original complete Source bundle. Recovery may need
  // this after close, so it uses Graph's retired-control operation directly,
  // never a new carrier admission. Record successful withdrawal immediately:
  // a later ordinary-input failure must not retry an already released bundle.
  Status release_source_bindings() noexcept {
    if (!run || !carrier)
      return Status::FailedPrecondition("execution episode has no original carrier");
    if (incoming && incoming->bound) {
      HYLE_RETURN_IF_ERROR(graph::release_sources(*run));
      incoming->bound = false;
    }
    return Status::Ok();
  }

  // Call only after the original outputs have been retained and Store/Append
  // transactions taken. Model policy decides when this may precede commit.
  // The caller's cold ordinary-input inventory excludes Source projections;
  // no episode-local copy or permission inference is needed here.
  // Keep the incoming owner and predecessor until both withdrawals succeed;
  // a failed preparation with no Source binding must not invent one to release.
  Status release_consumers(std::span<const graph::ValueId> ordinary_inputs) noexcept {
    if (!run || !carrier)
      return Status::FailedPrecondition("execution episode has no original carrier");
    if (!incoming) return carrier->release_inputs(ordinary_inputs);
    HYLE_RETURN_IF_ERROR(release_source_bindings());
    HYLE_RETURN_IF_ERROR(carrier->release_inputs(ordinary_inputs));
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
    unsubmitted_reuse_declined_ = false;
    return recover_impl(RecoveryMode::kDestructive, {}, std::forward<Recycle>(recycle));
  }

  // Discard unpublished work only after observing its original whole-Graph
  // success. Keep the open carrier and fixed IO; the next preparation still
  // acquires every dynamic permission and validates its complete physical tuple.
  // A partial return retains the success observation and every remaining owner.
  // Errors leave those facts available to the original destructive recovery.
  template<class Recycle>
  Result<bool> discard_successful(std::span<const graph::ValueId> ordinary_inputs,
      Recycle &&recycle) noexcept {
    return recover_impl(RecoveryMode::kSuccessfulDiscard, ordinary_inputs, std::forward<Recycle>(recycle));
  }

  // Only the typed Graph coverage refusal above can select this path. Keep
  // the open original Invocation/fixed IO, while recovering its real append,
  // Store, Source and unpublished candidate capabilities. A cleanup error
  // escalates to the original destructive recovery, without hiding its Status.
  template<class Recycle>
  Result<bool> recover_unsubmitted_reuse(std::span<const graph::ValueId> ordinary_inputs,
      Recycle &&recycle) noexcept {
    if (!unsubmitted_reuse_declined_)
      return std::unexpected(Status::FailedPrecondition("episode has no typed unsubmitted reuse refusal"));
    auto recovered = recover_impl(RecoveryMode::kUnsubmittedReuse, ordinary_inputs,
        std::forward<Recycle>(recycle));
    if (!recovered) {
      record_retirement_status(terminal, recovered.error());
      unsubmitted_reuse_declined_ = false;
    }
    return recovered;
  }

 private:
  Status check_finish_success(std::size_t first_append, std::size_t append_count) const noexcept {
    if (!run || !carrier || !has_completion())
      return Status::FailedPrecondition("successful stage requires its original carrier and completion");
    if (first_append > nodes.size() || append_count > nodes.size() - first_append ||
        first_append > appends.size() || append_count > appends.size() - first_append ||
        first_append > bound.size() || append_count > bound.size() - first_append)
      return Status::OutOfRange("successful stage append range exceeds its original cohort");
    return Status::Ok();
  }

  template<class OnCompletion, class Observe, class Complete>
  Status finish_success_impl(std::size_t first_append, std::size_t append_count,
      std::span<const graph::ValueId> ordinary_inputs,
      OnCompletion &&on_completion, Observe &&observe, Complete &&complete) {
    static_assert(noexcept(std::forward<OnCompletion>(on_completion)()));
    HYLE_RETURN_IF_ERROR(wait_success());
    std::forward<OnCompletion>(on_completion)();
    HYLE_ASSIGN_OR_RETURN(auto original, carrier->borrow());
    if (original != run)
      return Status::InvalidArgument("successful stage belongs to another original Invocation");
    if (sources) HYLE_RETURN_IF_ERROR(sources->detach_outputs());
    HYLE_RETURN_IF_ERROR(std::forward<Observe>(observe)());
    // An outgoing SubmissionSource pins this Graph's result observation.
    // Close its export before take/unbind, including a middle stage that also
    // consumes its predecessor. Keep the bundle itself alive for successors.
    if (sources) HYLE_RETURN_IF_ERROR(sources->submission.detach());
    if (append_count)
      HYLE_RETURN_IF_ERROR(append_transactions().subspan(first_append, append_count).take(*run));
    if (state_transactions) HYLE_RETURN_IF_ERROR(state_transactions->take_all(*run));
    HYLE_RETURN_IF_ERROR(std::forward<Complete>(complete)());
    return release_consumers(ordinary_inputs);
  }

  enum class RecoveryMode : std::uint8_t {
    kDestructive, kSuccessfulDiscard, kUnsubmittedReuse
  };
  bool unsubmitted_reuse_declined_ = false;
  // A default terminal Status is not evidence of successful execution. Keep
  // this observation after the original Completion/Source has been released,
  // and remember withdrawal if later page rollback needs another attempt.
  enum class SuccessfulDiscard : std::uint8_t {
    kUnobserved, kSuccessObserved, kConsumersReleased
  };
  SuccessfulDiscard successful_discard_ = SuccessfulDiscard::kUnobserved;

  template<class Recycle>
  Result<bool> recover_impl(RecoveryMode mode,
      std::span<const graph::ValueId> ordinary_inputs, Recycle &&recycle) noexcept {
    static_assert(noexcept(std::forward<Recycle>(recycle)()));
    const bool keep_carrier = mode != RecoveryMode::kDestructive;
    // Destroying an unsubmitted successor's capture may synchronize its
    // context. Preserve the original predecessor gate before any withdrawal.
    if (cleanup_pending && predecessor_episode && predecessor_episode->run &&
        predecessor_episode->run->in_flight()) return false;
    if (keep_carrier) {
      if (!run || !carrier || !carrier->initialized() || carrier->closing())
        return std::unexpected(Status::FailedPrecondition("carrier-preserving recovery requires its original open carrier"));
      if (!terminal.ok()) return std::unexpected(terminal);
      if (mode == RecoveryMode::kUnsubmittedReuse) {
        if (!unsubmitted_reuse_declined_ || has_completion() || sources ||
            (successful_discard_ != SuccessfulDiscard::kConsumersReleased &&
                (!incoming || !incoming->bound)))
          return std::unexpected(Status::FailedPrecondition("reuse refusal has submitted work or lost its original incoming bundle"));
      } else if (successful_discard_ == SuccessfulDiscard::kUnobserved) {
        if (!has_completion())
          return std::unexpected(Status::FailedPrecondition("successful discard requires its original completion"));
        const auto observed = poll();
        if (!observed) return std::unexpected(observed.error());
        if (*observed == compute::CompletionPoll::kPending) return false;
        successful_discard_ = SuccessfulDiscard::kSuccessObserved;
        completion = {};
      }
      // Completion success and the carrier's real native/retention retirement
      // are separate facts. Neither alone permits withdrawal or slot reuse.
      if (run->in_flight() || carrier->pending()) return false;
    } else if (has_completion()) {
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
      if (keep_carrier) {
        if (successful_discard_ != SuccessfulDiscard::kConsumersReleased) {
          HYLE_RETURN_IF_ERROR(release_consumers(ordinary_inputs));
          successful_discard_ = SuccessfulDiscard::kConsumersReleased;
        }
      } else {
        HYLE_RETURN_IF_ERROR(release_source_bindings());
        incoming.reset(); predecessor_episode.reset();
        // Destructive recovery remains destructive even when it takes over
        // the remaining facts of a partially completed successful discard.
        HYLE_RETURN_IF_ERROR(carrier->reset_retired());
      }
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
    successful_discard_ = SuccessfulDiscard::kUnobserved;
    unsubmitted_reuse_declined_ = false;
    release_admission();
    return true;
  }
};

} // namespace hyle::inference
