#pragma once

#include <algorithm>
#include <cstddef>
#include <exception>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "output_delivery.h"
#include "request_progress.h"
#include "request_window.h"
#include "retirement_observation.h"

namespace hyle::inference {

enum class RequestRetirement { kCompleted, kEvicted, kCancelled };

// reserved is already charged to the original in-flight work; additional is
// the most this call may additionally reserve or publish for this member.
struct RequestCreditGrant final {
  std::size_t reserved = 0, additional = 0;
  bool cancelled = false;
};

struct RequestWave final {
  std::span<const RequestKey> keys;
  std::span<const RequestCreditGrant> credits;
  bool observe_only = false;
  // Finish already accepted work but do not submit another successor. Unlike
  // observe_only, its successful observations may still advance logical work.
  bool drain_requested = false;
};

struct RequestWaveResult final {
  // Logical progress, not a native submission count: even a complete prefix
  // hit may advance a wave without launching any device work.
  bool advanced = false;
  RetirementObservation remaining{true, {}};
  std::exception_ptr failure{};
};

struct RequestPumpResult final {
  std::size_t admitted = 0, completed = 0, cancelled = 0;
  bool executed = false, waiting_retirement = false, waiting_output = false;
};

// A logical request identity, not an index in a model's submitted batch.
// Errors and release permission remain independent, including retired errors.
class RequestLoopFailure final : public std::runtime_error {
public:
  RequestLoopFailure(RequestKey key, RetirementObservation observed)
      : std::runtime_error(std::string(observed.status.message())), key_(key),
        retirement_(std::move(observed)) {}
  const Status &status() const noexcept { return retirement_.status; }
  RequestKey key() const noexcept { return key_; }
  const RetirementObservation &retirement() const noexcept { return retirement_; }
private:
  RequestKey key_;
  RetirementObservation retirement_;
};

// One serialized request owner. It owns identities and host turn scheduling,
// never a model head, output cursor, device episode or resource retirement.
// prepare() reserves all owner metadata; valid hot transitions allocate nothing.
// The statically bound adapter supplies only complete model-facing stages:
//   progress(key) -> RequestProgress&, resident(key) -> bool,
//   admission_preflight(key), admit(key, active),
//   execute_wave(RequestWave, span<size_t> remaining_reserved) -> RequestWaveResult,
//   publish_turn(key), retire(key, RequestRetirement) -> RetirementObservation.
// cancel() additionally uses stop(key) -> Status; ordinary pump-only consumers
// need no stop method. A bound OutputDelivery outlives its occupied request key
// and belongs to that request alone. Null means synchronous output consumption.
// admit restores the complete head and begins an unstarted turn using its real
// pending. execute_wave resolves ALL lookups before publishing any prompt. Its
// original Step/Source alone owns native completion. It reports actually
// observed outputs and the maximum still-publishable output count per member
// in remaining_reserved, including after a later failure. While work remains,
// subsequent pumps call the same adapter with the same ordered cohort: it may
// submit N+1 before observing N and return with N+1 still in flight. A failure
// switches that cohort to observe_only (no new work), until actual retirement.
// All borrowed spans are cold-prepared owner storage, valid until that cohort
// retires; the adapter does not retain them after retirement or across prepare.
// retire releases native/session resources only when retired is true. The
// request's committed output record remains intact until logical key release;
// an eviction does not release that key or imply versions/bytes became free.
// Exceptions propagate without retrying or pretending to roll back model work.
class RequestLoop final {
public:
  RequestLoop() = default;
  RequestLoop(const RequestLoop &) = delete;
  RequestLoop &operator=(const RequestLoop &) = delete;

  void prepare(std::size_t slots, std::size_t resident_limit) {
    if (executing_ || !cohort_.empty())
      throw std::logic_error("request cannot prepare while its original execution wave is in flight");
    std::vector<Slot> prepared(slots);
    std::vector<Terminal> retiring, output; retiring.reserve(slots); output.reserve(slots);
    std::vector<RequestKey> cohort; cohort.reserve(slots);
    std::vector<RequestCreditGrant> credits; credits.reserve(slots);
    std::vector<WaveProgress> before; before.reserve(slots);
    std::vector<std::size_t> remaining; remaining.reserve(slots);
    window_.prepare(slots, resident_limit);
    slots_.swap(prepared); retiring_.swap(retiring); waiting_output_.swap(output); resident_limit_ = resident_limit;
    cohort_.swap(cohort); wave_credits_.swap(credits); wave_before_.swap(before); wave_reserved_.swap(remaining);
    pending_eviction_.reset(); recover_turns_ = collect_cancelled_ = false; credit_waiters_ = 0;
    wave_observe_only_ = yield_requested_ = false; reported_failure_ = {}; wave_remaining_ = {true, {}};
  }

  RequestKey enqueue(std::size_t slot, OutputDelivery *delivery = nullptr) {
    const auto key = window_.enqueue(slot);
    slots_[slot] = {key, 0, delivery, false, false};
    return key;
  }

  // Caller has already observed retirement of the active episode. Pausing
  // keeps the same request's progress, pending, payload and delivery authority.
  std::size_t pause(RequestKey key) {
    if (executing_ || std::find(cohort_.begin(), cohort_.end(), key) != cohort_.end())
      throw std::logic_error("request cannot pause before its original execution wave retires");
    const auto order = window_.pause(key);
    slots_[key.slot].idle_order = order;
    clear_credit_wait(key);
    return order;
  }
  void wake(RequestKey key) { window_.wake(key); clear_credit_wait(key); }
  bool matches(RequestKey key) const noexcept { return window_.matches(key); }
  std::span<const RequestKey> active() const noexcept { return window_.active(); }
  std::optional<RequestKey> waiting_front() const noexcept { return window_.waiting_front(); }
  bool cancelled(RequestKey key) const noexcept { return matches(key) && slots_[key.slot].cancel_requested; }
  // Serialized policy intent only, including inside execute_wave. The active
  // cohort and its spans stay intact until original native retirement; then
  // unfinished survivors rejoin the existing FIFO without changing their turn.
  void request_yield() noexcept { yield_requested_ = true; }
  std::size_t credit(RequestKey key, std::size_t produced) const {
    if (!matches(key)) throw std::out_of_range("request credit query has a stale or foreign key");
    const auto *delivery = slots_[key.slot].delivery;
    const auto available = delivery ? delivery->available(produced) : (std::numeric_limits<std::size_t>::max)();
    const auto reserved = slots_[key.slot].pending_reserved;
    if (reserved > available) throw std::out_of_range("in-flight request outputs exceed its delivery credit");
    return available - reserved;
  }
  // Only the existing delivery cursor changes. Final delivery releases its
  // terminal key in the next pump, where completed/cancelled report the event.
  template<class Adapter>
  bool acknowledge(RequestKey key, std::size_t through, Adapter &adapter) {
    if (!matches(key) || !slots_[key.slot].delivery) return false;
    auto &slot = slots_[key.slot]; const auto produced = adapter.progress(key).generated.size();
    slot.delivery->acknowledge_prefix(produced, through);
    if (slot.credit_paused && slot.delivery->available(produced)) wake(key);
    return true;
  }
  // Single-owner ingress, including a stop requested from execute_wave after
  // original native submission. That case must not invalidate its active span:
  // the original model finish observes survivors before this owner retires keys.
  template<class Adapter>
  Status cancel(RequestKey key, Adapter &adapter) {
    if (!matches(key)) return Status(StatusCode::NotFound);
    auto &slot = slots_[key.slot]; slot.cancel_requested = true;
    bool terminal = false;
    for (auto &entry : retiring_) if (entry.key == key) { entry.reason = RequestRetirement::kCancelled; terminal = true; }
    for (auto &entry : waiting_output_) if (entry.key == key) { entry.reason = RequestRetirement::kCancelled; terminal = true; }
    if (!terminal && window_.cancel_waiting(key)) {
      clear_credit_wait(key);
      retiring_.push_back({key, RequestRetirement::kCancelled});
      // The same original owner is now a cancelled terminal, not two polls
      // with different reset/eviction counters or two retirement authorities.
      if (pending_eviction_ == key) pending_eviction_.reset();
    } else if (!terminal) collect_cancelled_ = true;
    return adapter.stop(key);
  }
  std::size_t idle_order(RequestKey key) const {
    if (!matches(key)) throw std::out_of_range("request idle query has a stale or foreign key");
    return slots_[key.slot].idle_order;
  }
  template<class Resident>
  std::optional<std::size_t> idle_victim(Resident resident) const {
    return window_.idle_victim(std::move(resident), [&](auto slot) { return slots_[slot].idle_order; });
  }

  template<class Adapter>
  RequestPumpResult pump_once(Adapter &adapter) {
    if (executing_) throw std::logic_error("request cannot recursively pump its borrowed execution wave");
    RequestPumpResult result;
    // The cohort owns the active identities until the original episode really
    // retires. ACK/cancel/input intents may change, but admission, turn changes,
    // eviction and active reordering must not invalidate that episode's spans.
    if (!cohort_.empty()) {
      if (!advance_wave(adapter, result)) return waiting_native(result);
      finish_wave(adapter, result);
      result.waiting_retirement = !retiring_.empty();
      result.waiting_output = !waiting_output_.empty() || credit_waiters_ != 0;
      return result;
    }
    for (std::size_t at = 0; at < waiting_output_.size();) {
      const auto entry = waiting_output_[at];
      if (!output_drained(adapter, entry.key)) { ++at; continue; }
      release_terminal(entry, result);
      waiting_output_.erase(waiting_output_.begin() + static_cast<std::ptrdiff_t>(at));
    }
    if (collect_cancelled_) {
      for (std::size_t at = 0; at < active().size();) {
        const auto key = active()[at];
        if (!cancelled(key)) { ++at; continue; }
        window_.retire_active(key); retiring_.push_back({key, RequestRetirement::kCancelled});
      }
      collect_cancelled_ = false;
    }
    if (pending_eviction_ && !finish_eviction(adapter, *pending_eviction_)) {
      return waiting_native(result);
    }
    // A terminal key is no longer active. Poll its original owner, never its
    // old input, and never free the slot merely because a previous poll failed.
    for (std::size_t at = 0; at < retiring_.size();) {
      if (!finish_retirement(adapter, at, result)) ++at;
    }
    if (!retiring_.empty()) return waiting_native(result);
    // execute_wave may have committed/observed a turn before a later observer
    // failed. Finish that boundary before considering another model execution.
    if (recover_turns_) {
      finish_turns(adapter, result); recover_turns_ = false;
      if (!retiring_.empty()) return waiting_native(result);
    }
    yield_active();
    while (const auto next = window_.waiting_front()) {
      const auto key = *next;
      auto &progress = adapter.progress(key);
      const bool finished = progress.turn_started() && progress.turn_done && !progress.more_turns();
      if (!finished && slots_[key.slot].delivery && !credit(key, progress.generated.size())) {
        window_.admitted(key); pause_for_credit(key); continue;
      }
      adapter.admission_preflight(key); // Before any victim or progress mutation.
      if (finished) {
        // A paused, already published open turn may be woken just to close
        // input. It needs no head restoration, publication or second output.
        window_.admitted(key); ++result.admitted;
        if (progress.waiting_for_input()) (void)pause(key);
        else {
          (void)window_.release_turn(key, false); retiring_.push_back({key, RequestRetirement::kCompleted});
          (void)finish_retirement(adapter, retiring_.size() - 1, result);
          if (!retiring_.empty()) return waiting_native(result);
        }
        continue;
      }
      if (!adapter.resident(key)) {
        const auto reserved = static_cast<std::size_t>(std::count_if(active().begin(), active().end(),
            [&](auto member) { return !adapter.resident(member); }));
        while (resident_count(adapter) + reserved >= resident_limit_) {
          const auto victim = idle_victim([&](auto slot) { return adapter.resident(slots_[slot].key); });
          if (!victim) {
            throw std::logic_error("request admission has no idle resident owner to evict");
          }
          const auto old = slots_[*victim].key;
          if (!finish_eviction(adapter, old)) return waiting_native(result);
        }
      }
      // A finished open-input turn was paused without a known successor.
      // wake itself never changes it; only actual later admission advances it.
      if (progress.turn_started() && progress.turn_done && progress.more_turns()) progress.advance_turn();
      adapter.admit(key, active());
      window_.admitted(key); ++result.admitted;
    }
    if (!active().empty()) {
      recover_turns_ = true;
      wave_remaining_ = {true, {}};
      cohort_.assign(active().begin(), active().end());
      wave_credits_.resize(cohort_.size()); wave_before_.resize(cohort_.size()); wave_reserved_.resize(cohort_.size());
      if (!advance_wave(adapter, result)) return waiting_native(result);
      finish_wave(adapter, result);
    }
    result.waiting_retirement = !retiring_.empty();
    result.waiting_output = !waiting_output_.empty() || credit_waiters_ != 0;
    return result;
  }

private:
  struct Slot final {
    RequestKey key{}; std::size_t idle_order = 0; OutputDelivery *delivery = nullptr;
    bool cancel_requested = false, credit_paused = false;
    std::size_t pending_reserved = 0;
  };
  struct Terminal final { RequestKey key{}; RequestRetirement reason = RequestRetirement::kCompleted; };
  struct WaveProgress final {
    std::size_t produced = 0, commits = 0, turn = 0, turn_begin = 0;
  };
  void release_wave() noexcept {
    for (const auto key : cohort_) slots_[key.slot].pending_reserved = 0;
    cohort_.clear(); wave_credits_.clear(); wave_before_.clear(); wave_reserved_.clear();
    wave_observe_only_ = false; reported_failure_ = {};
  }
  // Remaining quota and delivery credit are distinct bounds. An early EOS can
  // make remaining quota zero while an already submitted successor still owns
  // a reservation; that reservation lasts until its real discard/retirement.
  template<class Adapter>
  void prepare_wave_credits(Adapter &adapter) {
    for (std::size_t at = 0; at < cohort_.size(); ++at) {
      const auto key = cohort_[at]; auto &progress = adapter.progress(key);
      const auto held = slots_[key.slot].pending_reserved;
      const auto available = credit(key, progress.generated.size());
      const auto quota = progress.remaining_outputs();
      const auto additional = wave_observe_only_ || yield_requested_ || cancelled(key) ? 0 :
          (std::min)(available, quota > held ? quota - held : 0);
      wave_credits_[at] = {held, additional, cancelled(key)};
      wave_before_[at] = {progress.generated.size(), progress.observed_commits(), progress.turn_index, progress.turn_output_begin};
      wave_reserved_[at] = held;
    }
  }
  template<class Adapter>
  bool advance_wave(Adapter &adapter, RequestPumpResult &result) {
    prepare_wave_credits(adapter);
    const bool observing_failure = wave_observe_only_;
    RequestWaveResult observed;
    executing_ = true;
    try {
      observed = adapter.execute_wave(RequestWave{cohort_, wave_credits_, observing_failure, yield_requested_}, std::span(wave_reserved_));
    } catch (...) {
      executing_ = false;
      // Adapters normally return failure together with actual retirement. A
      // thrown adapter cannot grant release permission: keep its original
      // cohort, and conservatively charge the still-unobserved granted output.
      wave_observe_only_ = true; wave_remaining_ = {false, {}};
      for (std::size_t at = 0; at < cohort_.size(); ++at) {
        const auto &grant = wave_credits_[at];
        const auto produced = adapter.progress(cohort_[at]).generated.size();
        const auto previous = wave_before_[at].produced;
        const auto total = grant.reserved + grant.additional; // Bound by available credit in prepare.
        const auto delta = produced >= previous ? produced - previous : 0;
        slots_[cohort_[at].slot].pending_reserved = delta <= total ? total - delta : 0;
      }
      reported_failure_ = std::current_exception();
      throw;
    }
    executing_ = false;
    const auto previous_status = wave_remaining_.status;
    wave_remaining_.retired = observed.remaining.retired;
    record_retirement_status(wave_remaining_.status, observed.remaining.status);
    result.executed = observed.advanced;
    bool valid = !observing_failure || !observed.advanced;
    for (std::size_t at = 0; at < cohort_.size(); ++at) {
      const auto key = cohort_[at]; const auto &progress = adapter.progress(key);
      const auto &before = wave_before_[at]; const auto &grant = wave_credits_[at];
      const auto total = grant.reserved + grant.additional;
      const bool monotonic = progress.generated.size() >= before.produced && progress.observed_commits() >= before.commits;
      const auto delta = monotonic ? progress.generated.size() - before.produced : 0;
      const auto budget = delta <= total ? total - delta : 0;
      const bool member_valid = monotonic && delta <= total && wave_reserved_[at] <= budget &&
          progress.turn_index == before.turn && progress.turn_output_begin == before.turn_begin &&
          (!observed.remaining.retired || !wave_reserved_[at]);
      valid = valid && member_valid;
      // Store actual valid facts before propagating a later observer failure.
      // Invalid adapter metadata retains the conservative remaining allowance.
      slots_[key.slot].pending_reserved = member_valid ? wave_reserved_[at] : budget;
    }
    auto failure = observed.failure;
    if (!valid) {
      wave_observe_only_ = true;
      if (observed.remaining.retired) release_wave();
      throw std::logic_error("request wave changed its identity, quota, credit or observation-only contract");
    }
    const bool new_failure = failure && failure != reported_failure_;
    if (failure) { wave_observe_only_ = true; reported_failure_ = failure; }
    // Preserve the outcome independently of its release permission. An adapter
    // which has no richer exception still exposes a real native failure.
    const auto failed_key = cohort_.front();
    const bool expected_cancel = observed.remaining.status.IsCancelled() &&
        !observed.remaining.status.has_native_error() &&
        std::all_of(cohort_.begin(), cohort_.end(), [&](auto key) { return cancelled(key); });
    const bool native_failure = !observed.remaining.status.ok() && !expected_cancel && !failure &&
        wave_remaining_.status != previous_status;
    if (native_failure) wave_observe_only_ = true;
    if (observed.remaining.retired) release_wave();
    if (new_failure) std::rethrow_exception(failure);
    if (native_failure) throw RequestLoopFailure(failed_key, wave_remaining_);
    return observed.remaining.retired;
  }
  template<class Adapter>
  void finish_wave(Adapter &adapter, RequestPumpResult &result) {
    // Queued/paused cancellation may have arrived from the original model's
    // post-submit control point. Collect its retirement only after that wave.
    for (std::size_t at = 0; at < retiring_.size();) {
      if (!finish_retirement(adapter, at, result)) ++at;
    }
    finish_turns(adapter, result);
    yield_active();
    recover_turns_ = collect_cancelled_ = false;
  }
  void yield_active() {
    if (!yield_requested_) return;
    // Only called with no in-flight cohort, after normal turn/cancel handling.
    // requeue retains the same key, resident head, pending and output cursor.
    while (!active().empty()) {
      const auto key = active().front(); slots_[key.slot].idle_order = window_.requeue(key);
    }
    yield_requested_ = false;
  }
  void clear_credit_wait(RequestKey key) noexcept {
    auto &slot = slots_[key.slot];
    if (slot.credit_paused) { slot.credit_paused = false; --credit_waiters_; }
  }
  void pause_for_credit(RequestKey key) {
    slots_[key.slot].idle_order = window_.pause(key);
    slots_[key.slot].credit_paused = true; ++credit_waiters_;
  }
  RequestPumpResult waiting_native(RequestPumpResult result) const noexcept {
    result.waiting_retirement = true;
    result.waiting_output = !waiting_output_.empty() || credit_waiters_ != 0;
    return result;
  }
  template<class Adapter> bool output_drained(Adapter &adapter, RequestKey key) const {
    const auto *delivery = slots_[key.slot].delivery;
    if (!delivery) return true;
    const auto produced = adapter.progress(key).generated.size();
    (void)delivery->available(produced); // Preserve the original bounded-credit validation.
    return delivery->acknowledged() == produced;
  }
  void release_terminal(const Terminal &entry, RequestPumpResult &result) {
    window_.release(entry.key); clear_credit_wait(entry.key); slots_[entry.key.slot] = {}; ++result.completed;
    if (entry.reason == RequestRetirement::kCancelled) ++result.cancelled;
  }
  template<class Adapter> std::size_t resident_count(Adapter &adapter) const {
    return static_cast<std::size_t>(std::count_if(slots_.begin(), slots_.end(), [&](const auto &slot) {
      return matches(slot.key) && adapter.resident(slot.key);
    }));
  }
  template<class Adapter>
  bool finish_eviction(Adapter &adapter, RequestKey key) {
    (void)output_drained(adapter, key); // Validate bound delivery before native owner mutation.
    pending_eviction_ = key;
    const auto observed = adapter.retire(key, RequestRetirement::kEvicted);
    if (observed.retired) pending_eviction_.reset();
    if (!observed.status.ok()) throw RequestLoopFailure(key, observed);
    if (observed.retired && adapter.resident(key))
      throw std::logic_error("retired request eviction retained its resident payload");
    return observed.retired;
  }
  template<class Adapter>
  bool finish_retirement(Adapter &adapter, std::size_t at, RequestPumpResult &result) {
    const auto entry = retiring_[at]; const auto key = entry.key;
    const bool drained = output_drained(adapter, key);
    const auto observed = adapter.retire(key, entry.reason);
    if (observed.retired) {
      if (drained) release_terminal(entry, result);
      else waiting_output_.push_back(entry);
      retiring_.erase(retiring_.begin() + static_cast<std::ptrdiff_t>(at));
    }
    const bool expected_cancel = entry.reason == RequestRetirement::kCancelled &&
        observed.status.IsCancelled() && !observed.status.has_native_error();
    if (!observed.status.ok() && !expected_cancel) throw RequestLoopFailure(key, observed);
    return observed.retired;
  }
  template<class Adapter>
  void finish_turns(Adapter &adapter, RequestPumpResult &result) {
    for (std::size_t at = 0; at < active().size();) {
      const auto key = active()[at]; auto &progress = adapter.progress(key);
      if (cancelled(key)) {
        window_.retire_active(key); retiring_.push_back({key, RequestRetirement::kCancelled});
        (void)finish_retirement(adapter, retiring_.size() - 1, result); continue;
      }
      if (!progress.turn_done) {
        if (slots_[key.slot].delivery && !credit(key, progress.generated.size())) pause_for_credit(key);
        else ++at;
        continue;
      }
      adapter.publish_turn(key);
      if (progress.more_turns()) {
        slots_[key.slot].idle_order = window_.release_turn(key, true);
        progress.advance_turn();
      } else if (progress.waiting_for_input()) {
        (void)pause(key);
      } else {
        (void)window_.release_turn(key, false);
        retiring_.push_back({key, RequestRetirement::kCompleted});
        (void)finish_retirement(adapter, retiring_.size() - 1, result);
      }
    }
  }
  RequestWindow window_;
  std::vector<Slot> slots_;
  std::vector<Terminal> retiring_, waiting_output_;
  std::vector<RequestKey> cohort_;
  std::vector<RequestCreditGrant> wave_credits_;
  std::vector<WaveProgress> wave_before_;
  std::vector<std::size_t> wave_reserved_;
  RetirementObservation wave_remaining_{true, {}};
  std::exception_ptr reported_failure_{};
  std::optional<RequestKey> pending_eviction_;
  std::size_t resident_limit_ = 0, credit_waiters_ = 0;
  bool recover_turns_ = false, collect_cancelled_ = false, executing_ = false;
  bool wave_observe_only_ = false, yield_requested_ = false;
};

} // namespace hyle::inference
