#pragma once

#include <algorithm>
#include <cstddef>
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
//   admission_preflight(key), admit(key, active), execute_wave(active),
//   publish_turn(key), retire(key, RequestRetirement) -> RetirementObservation.
// cancel() additionally uses stop(key) -> Status; ordinary pump-only consumers
// need no stop method. A bound OutputDelivery outlives its occupied request key
// and belongs to that request alone. Null means synchronous output consumption.
// admit restores the complete head and begins an unstarted turn using its real
// pending. execute_wave resolves ALL lookups before publishing any prompt; it
// observes committed outputs before returning (or propagating a later error).
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
    std::vector<Slot> prepared(slots);
    std::vector<Terminal> retiring, output; retiring.reserve(slots); output.reserve(slots);
    window_.prepare(slots, resident_limit);
    slots_.swap(prepared); retiring_.swap(retiring); waiting_output_.swap(output); resident_limit_ = resident_limit;
    pending_eviction_.reset(); recover_turns_ = collect_cancelled_ = false; credit_waiters_ = 0;
  }

  RequestKey enqueue(std::size_t slot, OutputDelivery *delivery = nullptr) {
    const auto key = window_.enqueue(slot);
    slots_[slot] = {key, 0, delivery, false, false};
    return key;
  }

  // Caller has already observed retirement of the active episode. Pausing
  // keeps the same request's progress, pending, payload and delivery authority.
  std::size_t pause(RequestKey key) {
    if (executing_) throw std::logic_error("request cannot pause inside its borrowed execution wave");
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
  std::size_t credit(RequestKey key, std::size_t produced) const {
    if (!matches(key)) throw std::out_of_range("request credit query has a stale or foreign key");
    const auto *delivery = slots_[key.slot].delivery;
    return delivery ? delivery->available(produced) : (std::numeric_limits<std::size_t>::max)();
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
      executing_ = true;
      try { adapter.execute_wave(active()); }
      catch (...) { executing_ = false; throw; }
      executing_ = false;
      result.executed = true;
      // Queued/paused cancellation may have arrived from the original model's
      // post-submit control point. Its withdrawal did not alter active; collect
      // its actual retirement only after that original wave has finished.
      for (std::size_t at = 0; at < retiring_.size();) {
        if (!finish_retirement(adapter, at, result)) ++at;
      }
      finish_turns(adapter, result);
      recover_turns_ = collect_cancelled_ = false;
    }
    result.waiting_retirement = !retiring_.empty();
    result.waiting_output = !waiting_output_.empty() || credit_waiters_ != 0;
    return result;
  }

private:
  struct Slot final {
    RequestKey key{}; std::size_t idle_order = 0; OutputDelivery *delivery = nullptr;
    bool cancel_requested = false, credit_paused = false;
  };
  struct Terminal final { RequestKey key{}; RequestRetirement reason = RequestRetirement::kCompleted; };
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
  std::optional<RequestKey> pending_eviction_;
  std::size_t resident_limit_ = 0, credit_waiters_ = 0;
  bool recover_turns_ = false, collect_cancelled_ = false, executing_ = false;
};

} // namespace hyle::inference
