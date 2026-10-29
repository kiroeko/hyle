#pragma once

#include <cstddef>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <utility>

#include "hyle/graph/progress.h"
#include "hyle/graph/slots.h"
#include "execution_resources.h"
#include "prepared_output.h"
#include "program_io.h"
#include "retirement_observation.h"

namespace hyle::inference {

// Synchronous, borrowed installation recipe for one selected program. The
// original IO, logical Graph and progress runtimes keep their existing owners;
// no part of this recipe is retained after initialize(). Slot presence is
// explicit because a destination-only Store can have no paired sources.
struct ProgramInitialization final {
  struct Progress final {
    sched::Scheduler *scheduler = nullptr;
    sched::BlockingPool *workers = nullptr;
  };
  struct Slots final {
    const hyle::graph::Graph &graph;
    std::span<const hyle::graph::slots::SourceBinding> sources;
    hyle::graph::slots::PrepareOptions options{};
  };

  ProgramIo &io;
  graph::InvocationOptions invocation{};
  Progress progress{};
  std::optional<Slots> slots{};
};

// The original accepted Completion owns the one retention attachment. Submit
// is a statically dispatched call to Graph (including its Source-bundle form),
// not another execution route or a stored callback.
template<class Scope, class Submit>
Result<compute::Completion> submit_retained(RetainedResources<Scope> &retention,
    const std::shared_ptr<Scope> &scope, Submit &&submit) noexcept {
  static_assert(noexcept(std::forward<Submit>(submit)()));
  HYLE_RETURN_IF_ERROR(retention.reserve(scope));
  auto accepted = std::forward<Submit>(submit)();
  if (!accepted) {
    retention.reject();
    return std::unexpected(std::move(accepted).error());
  }
  HYLE_RETURN_IF_ERROR(retention.attach(*accepted));
  return std::move(*accepted);
}

// One stable, embedded carrier for an already selected concrete Graph program.
// Its borrowed CompiledGraph must outlive it: declare an owned program before
// this member, or borrow a stable immutable child from the original inventory.
// Only the existing accepted retention node pins its Scope. This object adds
// no payload owner, Completion, transaction or release permission.
template<class Scope> class ProgramCarrier final {
public:
  ProgramCarrier() = default;
  ProgramCarrier(const ProgramCarrier &) = delete;
  ProgramCarrier &operator=(const ProgramCarrier &) = delete;
  ProgramCarrier(ProgramCarrier &&) = delete;
  ProgramCarrier &operator=(ProgramCarrier &&) = delete;

  Status prepare(const graph::CompiledGraph &compiled,
      std::span<const graph::QueueBinding> available) noexcept {
    if (handed_off_ || compiled_ || run_ || pending())
      return Status::FailedPrecondition("program carrier is already cold prepared");
    HYLE_RETURN_IF_ERROR(detail::check_output_route(compiled));
    HYLE_ASSIGN_OR_RETURN(auto selected, prepare_queue_bindings(available, compiled.report));
    queues_ = std::move(selected);
    compiled_ = &compiled;
    return Status::Ok();
  }

  bool initialized() const noexcept { return run_.valid(); }
  bool closing() const noexcept { return closing_; }
  bool pending() const noexcept { return retirement_.pending(); }
  std::span<const graph::QueueBinding> queues() const noexcept { return queue_span(queues_); }

  // The complete installation order is shared by target, draft and private
  // programs. A partial bind never publishes a half-initialized run. Original
  // Graph destruction returns any fixed writer consumed by a failed candidate.
  Status initialize(const ProgramInitialization &installation) {
    if (handed_off_ || !compiled_)
      return Status::FailedPrecondition("program carrier has no prepared program");
    if (pending() || run_.in_flight())
      return Status::Unavailable("program carrier has not actually retired");
    if (closing_ || run_)
      return Status::FailedPrecondition("program carrier needs its original retired reset before initialization");
    HYLE_RETURN_IF_ERROR(installation.io.check_program(*compiled_));
    const auto &progress = installation.progress;
    if ((progress.scheduler == nullptr) != (progress.workers == nullptr))
      return Status::InvalidArgument("program progress requires both original host runtimes");
    HYLE_ASSIGN_OR_RETURN(auto candidate, compiled_->executable.create_invocation(installation.invocation));
    // Selection is already fixed by prepare(). Compare the original complete
    // canonical identity, including derived dimensions, rather than asking a
    // model to restate a family ordinal and a subset of its shape bindings.
    const auto selected = compiled_->executable.specialization();
    if (candidate.specialization() != selected)
      return Status::FailedPrecondition("program carrier specialization differs from its selected program");
    if (progress.scheduler)
      HYLE_RETURN_IF_ERROR(graph::prepare_progress(candidate, *progress.scheduler, *progress.workers));
    HYLE_RETURN_IF_ERROR(installation.io.bind_fixed(candidate));
    if (installation.slots) {
      const auto &slots = *installation.slots;
      HYLE_RETURN_IF_ERROR(graph::slots::prepare(candidate, slots.graph, queues(), slots.sources, slots.options));
    }
    run_ = std::move(candidate);
    return Status::Ok();
  }

  // Exposes the original Invocation for model-specific dynamic IO and its
  // existing Store/Append transactions. A closed handle is never reusable just
  // because operator bool() remains true or a previous observation succeeded.
  Result<graph::Invocation *> borrow() noexcept {
    if (pending() || run_.in_flight())
      return std::unexpected(Status::Unavailable("program carrier has not actually retired"));
    if (closing_ || !run_)
      return std::unexpected(Status::FailedPrecondition("program carrier is not initialized for admission"));
    return &run_;
  }

  Result<compute::Completion> submit(const std::shared_ptr<Scope> &scope) noexcept {
    return submit(scope, [](graph::Invocation &run,
        std::span<const graph::QueueBinding> queues) noexcept { return run.submit(queues); });
  }

  // Source-backed consumers use Graph's same submission API with their exact
  // original bundle. This direct, statically dispatched action is not stored;
  // reservation and attachment remain identical to an ordinary submission.
  template<class Submit>
  Result<compute::Completion> submit(const std::shared_ptr<Scope> &scope,
      Submit &&submit_graph) noexcept {
    HYLE_ASSIGN_OR_RETURN(auto run, borrow());
    static_assert(noexcept(std::forward<Submit>(submit_graph)(*run, queues())));
    return inference::submit_retained(retirement_, scope,
        [&]() noexcept { return std::forward<Submit>(submit_graph)(*run, queues()); });
  }

  Status release_inputs(std::span<const graph::ValueId> inputs) noexcept {
    HYLE_ASSIGN_OR_RETURN(auto run, borrow());
    return run->unbind(inputs);
  }

  // Environment is the existing native Backend, borrowed only for this direct
  // call. close/drain errors and actual retirement are independent. Keep the
  // same closed run when retirement is unknown, including a partial submit.
  template<class Environment>
  RetirementObservation retire(Environment &environment) noexcept {
    closing_ = true;
    Status status;
    if (run_) status = run_.close();
    // Graph close has already stopped admission when it reports this one
    // nonnative Unavailable for outstanding work. That busy hint is not the
    // terminal outcome and must not hide drain's native failure or success.
    if (status.IsUnavailable() && !status.has_native_error()) status = Status::Ok();
    record_retirement_status(status, environment.drain_retired(run_));
    return {!run_.in_flight() && !pending(), status};
  }

  // The caller first takes/recovers any original Graph-owned transactions.
  // Target and NextN retain their original Step guards. This check neither
  // takes a transaction nor treats an error Status as permission to discard.
  Status reset_retired() noexcept {
    if (handed_off_)
      return Status::FailedPrecondition("program carrier was handed off for teardown");
    if (run_.in_flight() || pending())
      return Status::Unavailable("program carrier has not actually retired");
    run_ = {};
    closing_ = false;
    return Status::Ok();
  }

  // Teardown only: after the Scope has handed off its original Backend, drop
  // the application's Graph root to break scope->Graph->scope. Accepted native
  // work and the original retention still own their real retirement lifetime.
  void release_invocation_root() noexcept {
    handed_off_ = true;
    closing_ = true;
    run_ = {};
  }

private:
  const graph::CompiledGraph *compiled_ = nullptr;
  QueueBindings queues_;
  graph::Invocation run_;
  RetainedResources<Scope> retirement_;
  bool closing_ = false;
  bool handed_off_ = false;
};

} // namespace hyle::inference
