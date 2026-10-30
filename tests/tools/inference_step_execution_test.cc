// Scripted host control only. Retirement reports below exercise propagation;
// they are not device observations, native failures or fabricated Completions.
#include "../../engine/inference/step_execution.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <exception>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <string_view>

namespace {
namespace i = hyle::inference;
void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }

enum class Event {
  kPrepareCurrent, kSubmitCurrent, kCheckCancel, kCancelCurrent, kCancelBoth,
  kPrepareSuccessor, kSubmitSuccessor, kFinishCurrent, kFinishSuccessor,
  kFinishRefusal, kKeepSuccessor, kDiscardSuccessor, kBindSuccessor,
  kCarryCurrent, kCarrySuccessor, kCarryDeferred, kCarryUnavailable, kFinishReuse,
  kRetireTyped, kRetireUnknown, kRecoverRetired, kRecoverPending,
};
using E = Event;
struct Log final {
  std::array<Event, 32> values{};
  std::size_t size = 0;
  bool overflow = false;
  void push(Event value) noexcept {
    if (size == values.size()) { overflow = true; return; }
    values[size++] = value;
  }
  void expect(std::initializer_list<Event> expected) const {
    require(!overflow && std::ranges::equal(std::span(values).first(size), expected),
        "shared step driver changed the original control order");
  }
};
enum class Fault { kNone, kTyped, kUnknown };
struct ModelSignal final { const void *origin; int payload; };

struct CommonAdapter {
  Log log;
  unsigned cancellation_checks = 0, cancel_on_check = 0;
  bool cancelled_with_successor = false;
  std::size_t committed_records = 0, uncommitted_records = 1;
  bool held_call_metadata = true;
  Event fault_at = Event::kPrepareCurrent;
  Fault fault = Fault::kNone;
  i::RetirementObservation retirement_report{true, {}};
  i::EvaluationFailure *failure_seen = nullptr;
  std::exception_ptr exception_seen;
  const ModelSignal *signal_seen = nullptr;

  void hit(Event event) {
    log.push(event);
    if (event != fault_at) return;
    if (fault == Fault::kTyped)
      throw i::EvaluationFailure(hyle::Status::OutOfRange("original host selection refused"), 1);
    if (fault == Fault::kUnknown) throw ModelSignal{this, 73};
  }
  void prepare(i::StepPosition position) {
    require(position == i::StepPosition::kCurrent, "successor preparation used the current-step action");
    hit(E::kPrepareCurrent);
  }
  void submit(i::StepPosition position) {
    hit(position == i::StepPosition::kCurrent ? E::kSubmitCurrent : E::kSubmitSuccessor);
  }
  bool cancelled() {
    hit(E::kCheckCancel);
    return ++cancellation_checks == cancel_on_check;
  }
  void cancel(bool submitted) {
    cancelled_with_successor = submitted;
    hit(submitted ? E::kCancelBoth : E::kCancelCurrent);
  }
  void finish(i::StepPosition position) {
    hit(position == i::StepPosition::kCurrent ? E::kFinishCurrent : E::kFinishSuccessor);
    require(uncommitted_records != 0, "host adapter finished a nonexistent original record");
    --uncommitted_records; ++committed_records;
  }
  i::RetirementObservation retire_failure(i::EvaluationFailure *failure) noexcept {
    log.push(failure ? E::kRetireTyped : E::kRetireUnknown);
    failure_seen = failure;
    exception_seen = std::current_exception();
    // Rethrow names the active original exception. current_exception() may
    // retain a copy, so exception_ptr equality is not an identity contract.
    if (!failure && exception_seen) {
      try { throw; }
      catch (const ModelSignal &signal) { signal_seen = &signal; }
      catch (...) {}
    }
    return retirement_report;
  }
  void recover_records(bool retired) noexcept {
    log.push(retired ? E::kRecoverRetired : E::kRecoverPending);
    if (retired) { uncommitted_records = 0; held_call_metadata = false; }
  }
};

// Deliberately has no successor, carry, refusal, bind or reuse methods.
struct SynchronousAdapter final : CommonAdapter {
  static constexpr bool kSupportsSuccessor = false;
};
struct ContinuousAdapter final : CommonAdapter {
  static constexpr bool kSupportsSuccessor = true;
  i::StepSuccessor successor = i::StepSuccessor::kPrepared;
  bool keep = true, deferred = false;
  i::StepSuccessor prepare_successor() {
    hit(E::kPrepareSuccessor);
    if (successor == i::StepSuccessor::kPrepared) ++uncommitted_records;
    return successor;
  }
  void finish_refusal() { hit(E::kFinishRefusal); }
  bool keep_successor() { hit(E::kKeepSuccessor); return keep; }
  void discard_successor() { hit(E::kDiscardSuccessor); --uncommitted_records; }
  void bind_successor() { hit(E::kBindSuccessor); }
  void carry(i::StepCarry disposition) {
    switch (disposition) {
    case i::StepCarry::kCurrent: hit(E::kCarryCurrent); break;
    case i::StepCarry::kSuccessor: hit(E::kCarrySuccessor); break;
    case i::StepCarry::kDeferred: hit(E::kCarryDeferred); deferred = true; break;
    case i::StepCarry::kUnavailable: hit(E::kCarryUnavailable); break;
    }
  }
  void finish_reuse() { hit(E::kFinishReuse); }
};

void synchronous_adapters_and_resume_need_no_successor_protocol() {
  SynchronousAdapter fresh;
  require(i::drive_steps(fresh) == i::StepDriveResult::kCompleted && fresh.committed_records == 1,
      "synchronous step failed to complete its original current record");
  fresh.log.expect({E::kPrepareCurrent, E::kSubmitCurrent, E::kCheckCancel, E::kFinishCurrent});
  SynchronousAdapter resumed;
  require(i::drive_steps(resumed, {.resume = true}) == i::StepDriveResult::kCompleted &&
      resumed.committed_records == 1, "resumed synchronous step lost its existing record");
  resumed.log.expect({E::kPrepareCurrent, E::kCheckCancel, E::kFinishCurrent});
  SynchronousAdapter cancelled;
  cancelled.cancel_on_check = 1;
  require(i::drive_steps(cancelled) == i::StepDriveResult::kCompleted && !cancelled.committed_records,
      "cancelled synchronous current step was still finished");
  cancelled.log.expect({E::kPrepareCurrent, E::kSubmitCurrent, E::kCheckCancel, E::kCancelCurrent});
}

void continuous_completion_and_no_successor_keep_the_original_order() {
  ContinuousAdapter pair;
  require(i::drive_steps(pair) == i::StepDriveResult::kCompleted && pair.committed_records == 2 &&
      !pair.uncommitted_records, "prepared successor did not complete exactly two original records");
  pair.log.expect({E::kPrepareCurrent, E::kSubmitCurrent, E::kCheckCancel, E::kPrepareSuccessor,
      E::kSubmitSuccessor, E::kCheckCancel, E::kFinishCurrent, E::kKeepSuccessor,
      E::kBindSuccessor, E::kFinishSuccessor, E::kCarrySuccessor, E::kFinishReuse});
  ContinuousAdapter current;
  current.successor = i::StepSuccessor::kNone;
  require(i::drive_steps(current, {.resume = true}) == i::StepDriveResult::kCompleted &&
      current.committed_records == 1, "resumed continuous current step was submitted or finished twice");
  current.log.expect({E::kPrepareCurrent, E::kCheckCancel, E::kPrepareSuccessor, E::kCheckCancel,
      E::kFinishCurrent, E::kCarryCurrent, E::kFinishReuse});
}

void refused_and_unavailable_successors_finish_current_first() {
  ContinuousAdapter refused;
  refused.successor = i::StepSuccessor::kRefused;
  require(i::drive_steps(refused) == i::StepDriveResult::kCompleted && refused.committed_records == 1,
      "optional successor refusal prevented the original current publication");
  refused.log.expect({E::kPrepareCurrent, E::kSubmitCurrent, E::kCheckCancel, E::kPrepareSuccessor,
      E::kCheckCancel, E::kFinishCurrent, E::kFinishRefusal});
  ContinuousAdapter unavailable;
  unavailable.successor = i::StepSuccessor::kUnavailable;
  require(i::drive_steps(unavailable) == i::StepDriveResult::kCompleted && unavailable.committed_records == 1,
      "unavailable continuation lost its completed original current step");
  unavailable.log.expect({E::kPrepareCurrent, E::kSubmitCurrent, E::kCheckCancel, E::kPrepareSuccessor,
      E::kCheckCancel, E::kFinishCurrent, E::kCarryUnavailable});
}

void cancellation_discard_and_defer_preserve_original_stage_boundaries() {
  ContinuousAdapter early;
  early.cancel_on_check = 1;
  require(i::drive_steps(early) == i::StepDriveResult::kCompleted && !early.committed_records,
      "first cancellation still prepared a successor or published current");
  early.log.expect({E::kPrepareCurrent, E::kSubmitCurrent, E::kCheckCancel, E::kCancelCurrent});
  for (const auto next : {i::StepSuccessor::kPrepared, i::StepSuccessor::kNone}) {
    ContinuousAdapter cancelled;
    cancelled.successor = next; cancelled.cancel_on_check = 2;
    require(i::drive_steps(cancelled) == i::StepDriveResult::kCompleted && !cancelled.committed_records &&
        cancelled.cancelled_with_successor == (next == i::StepSuccessor::kPrepared),
        "second cancellation confused prepared/actually submitted successor ownership");
    if (next == i::StepSuccessor::kPrepared)
      cancelled.log.expect({E::kPrepareCurrent, E::kSubmitCurrent, E::kCheckCancel, E::kPrepareSuccessor,
          E::kSubmitSuccessor, E::kCheckCancel, E::kCancelBoth});
    else
      cancelled.log.expect({E::kPrepareCurrent, E::kSubmitCurrent, E::kCheckCancel, E::kPrepareSuccessor,
          E::kCheckCancel, E::kCancelCurrent});
  }
  ContinuousAdapter discarded;
  discarded.keep = false;
  require(i::drive_steps(discarded) == i::StepDriveResult::kCompleted && discarded.committed_records == 1 &&
      !discarded.uncommitted_records, "discarding successor changed current's committed record");
  discarded.log.expect({E::kPrepareCurrent, E::kSubmitCurrent, E::kCheckCancel, E::kPrepareSuccessor,
      E::kSubmitSuccessor, E::kCheckCancel, E::kFinishCurrent, E::kKeepSuccessor, E::kDiscardSuccessor});
  ContinuousAdapter deferred;
  require(i::drive_steps(deferred, {.defer_successor = true}) == i::StepDriveResult::kDeferred &&
      deferred.deferred && deferred.committed_records == 1 && deferred.uncommitted_records == 1,
      "deferring successor finished, lost or republished its original pending record");
  deferred.log.expect({E::kPrepareCurrent, E::kSubmitCurrent, E::kCheckCancel, E::kPrepareSuccessor,
      E::kSubmitSuccessor, E::kCheckCancel, E::kFinishCurrent, E::kKeepSuccessor,
      E::kBindSuccessor, E::kCarryDeferred});
}

void typed_failure_keeps_status_identity_and_unretired_records() {
  SynchronousAdapter early;
  early.fault = Fault::kTyped;
  early.retirement_report = {false, hyle::Status::Unavailable("host owner still retains its call")};
  try { (void)i::drive_steps(early); throw std::runtime_error("prepare failure disappeared"); }
  catch (const i::EvaluationFailure &failure) {
    require(&failure == early.failure_seen && failure.status().IsOutOfRange() &&
        failure.status().message() == "original host selection refused" && failure.participant() == 1 &&
        failure.retirement_observed() && !failure.retirement().retired &&
        failure.retirement().status.IsUnavailable() && early.uncommitted_records == 1 && early.held_call_metadata,
        "prepare failure changed identity/status or erased unretired original records");
  }
  early.log.expect({E::kPrepareCurrent, E::kRetireTyped, E::kRecoverPending});
  for (const bool retired : {false, true}) {
    ContinuousAdapter late;
    late.fault = Fault::kTyped; late.fault_at = E::kFinishSuccessor;
    late.retirement_report = {retired, hyle::Status::Internal("original host retirement observation")};
    try { (void)i::drive_steps(late); throw std::runtime_error("successor finish failure disappeared"); }
    catch (const i::EvaluationFailure &failure) {
      require(&failure == late.failure_seen && failure.status().IsOutOfRange() &&
          failure.status().message() == "original host selection refused" && failure.participant() == 1 &&
          failure.retirement_observed() && failure.retirement().retired == retired &&
          failure.retirement().status.IsInternal() && late.committed_records == 1 &&
          late.uncommitted_records == (retired ? 0u : 1u) && late.held_call_metadata == !retired,
          "late failure rolled back current or treated a retirement Status as release permission");
    }
    require(late.log.size == 12 && late.log.values[10] == E::kRetireTyped &&
        late.log.values[11] == (retired ? E::kRecoverRetired : E::kRecoverPending),
        "typed failure did not retire before recovering original records");
  }
}

void unknown_failure_is_rethrown_after_original_retirement_and_recovery() {
  ContinuousAdapter late;
  late.fault = Fault::kUnknown; late.fault_at = E::kFinishReuse;
  try { (void)i::drive_steps(late); throw std::runtime_error("unknown model exception disappeared"); }
  catch (const ModelSignal &signal) {
    require(signal.origin == static_cast<CommonAdapter *>(&late) && signal.payload == 73 &&
        late.signal_seen == &signal && late.exception_seen && !late.failure_seen &&
        late.committed_records == 2 && !late.uncommitted_records && !late.held_call_metadata,
        "unknown model exception was replaced or prior committed records were rolled back");
  }
  late.log.expect({E::kPrepareCurrent, E::kSubmitCurrent, E::kCheckCancel, E::kPrepareSuccessor,
      E::kSubmitSuccessor, E::kCheckCancel, E::kFinishCurrent, E::kKeepSuccessor,
      E::kBindSuccessor, E::kFinishSuccessor, E::kCarrySuccessor, E::kFinishReuse,
      E::kRetireUnknown, E::kRecoverRetired});
}
} // namespace

int main() {
  try {
    synchronous_adapters_and_resume_need_no_successor_protocol();
    continuous_completion_and_no_successor_keep_the_original_order();
    refused_and_unavailable_successors_finish_current_first();
    cancellation_discard_and_defer_preserve_original_stage_boundaries();
    typed_failure_keeps_status_identity_and_unretired_records();
    unknown_failure_is_rethrown_after_original_retirement_and_recovery();
    std::puts("{\"status\":\"PASS\",\"groups\":6,\"device_execution\":false}");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "inference step execution: %s\n", error.what());
    return 1;
  } catch (...) {
    std::fputs("inference step execution: unexpected exception\n", stderr);
    return 1;
  }
}
