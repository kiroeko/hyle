#pragma once

#include <algorithm>
#include <cstddef>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "request_progress.h"
#include "request_window.h"
#include "retirement_observation.h"

namespace hyle::inference {

enum class RequestRetirement { kCompleted, kEvicted };

struct RequestPumpResult final {
  std::size_t admitted = 0, completed = 0;
  bool executed = false, waiting_retirement = false;
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
// admit restores the complete head and begins an unstarted turn using its real
// pending. execute_wave resolves ALL lookups before publishing any prompt; it
// observes committed outputs before returning (or propagating a later error).
// retire releases its actual payload only when retired is true; an eviction
// does not release this logical key or imply that versions/bytes became free.
// Exceptions propagate without retrying or pretending to roll back model work.
class RequestLoop final {
public:
  RequestLoop() = default;
  RequestLoop(const RequestLoop &) = delete;
  RequestLoop &operator=(const RequestLoop &) = delete;

  void prepare(std::size_t slots, std::size_t resident_limit) {
    std::vector<Slot> prepared(slots);
    std::vector<RequestKey> retiring; retiring.reserve(slots);
    window_.prepare(slots, resident_limit);
    slots_.swap(prepared); retiring_.swap(retiring); resident_limit_ = resident_limit;
    pending_eviction_.reset(); recover_turns_ = false;
  }

  RequestKey enqueue(std::size_t slot) {
    const auto key = window_.enqueue(slot);
    slots_[slot] = {key, 0};
    return key;
  }

  // Caller has already observed retirement of the active episode. Pausing
  // keeps the same request's progress, pending, payload and delivery authority.
  std::size_t pause(RequestKey key) {
    const auto order = window_.pause(key);
    slots_[key.slot].idle_order = order;
    return order;
  }
  void wake(RequestKey key) { window_.wake(key); }
  bool matches(RequestKey key) const noexcept { return window_.matches(key); }
  std::span<const RequestKey> active() const noexcept { return window_.active(); }
  std::optional<RequestKey> waiting_front() const noexcept { return window_.waiting_front(); }
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
    RequestPumpResult result;
    if (pending_eviction_ && !finish_eviction(adapter, *pending_eviction_)) {
      result.waiting_retirement = true; return result;
    }
    // A terminal key is no longer active. Poll its original owner, never its
    // old input, and never free the slot merely because a previous poll failed.
    for (std::size_t at = 0; at < retiring_.size();) {
      if (!finish_retirement(adapter, at, result)) ++at;
    }
    if (!retiring_.empty()) { result.waiting_retirement = true; return result; }
    // execute_wave may have committed/observed a turn before a later observer
    // failed. Finish that boundary before considering another model execution.
    if (recover_turns_) {
      finish_turns(adapter, result); recover_turns_ = false;
      if (!retiring_.empty()) { result.waiting_retirement = true; return result; }
    }
    while (const auto next = window_.waiting_front()) {
      const auto key = *next;
      adapter.admission_preflight(key); // Before any victim or progress mutation.
      auto &progress = adapter.progress(key);
      if (progress.turn_started() && progress.turn_done && !progress.more_turns()) {
        // A paused, already published open turn may be woken just to close
        // input. It needs no head restoration, publication or second output.
        window_.admitted(key); ++result.admitted;
        if (progress.waiting_for_input()) (void)pause(key);
        else {
          (void)window_.release_turn(key, false); retiring_.push_back(key);
          (void)finish_retirement(adapter, retiring_.size() - 1, result);
          if (!retiring_.empty()) { result.waiting_retirement = true; return result; }
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
          if (!finish_eviction(adapter, old)) { result.waiting_retirement = true; return result; }
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
      adapter.execute_wave(active());
      result.executed = true;
      finish_turns(adapter, result);
      recover_turns_ = false;
    }
    result.waiting_retirement = !retiring_.empty();
    return result;
  }

private:
  struct Slot final { RequestKey key{}; std::size_t idle_order = 0; };
  template<class Adapter> std::size_t resident_count(Adapter &adapter) const {
    return static_cast<std::size_t>(std::count_if(slots_.begin(), slots_.end(), [&](const auto &slot) {
      return matches(slot.key) && adapter.resident(slot.key);
    }));
  }
  template<class Adapter>
  bool finish_eviction(Adapter &adapter, RequestKey key) {
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
    const auto key = retiring_[at];
    const auto observed = adapter.retire(key, RequestRetirement::kCompleted);
    if (observed.retired) {
      window_.release(key); slots_[key.slot] = {}; ++result.completed;
      retiring_.erase(retiring_.begin() + static_cast<std::ptrdiff_t>(at));
    }
    if (!observed.status.ok()) throw RequestLoopFailure(key, observed);
    return observed.retired;
  }
  template<class Adapter>
  void finish_turns(Adapter &adapter, RequestPumpResult &result) {
    for (std::size_t at = 0; at < active().size();) {
      const auto key = active()[at]; auto &progress = adapter.progress(key);
      if (!progress.turn_done) { ++at; continue; }
      adapter.publish_turn(key);
      if (progress.more_turns()) {
        slots_[key.slot].idle_order = window_.release_turn(key, true);
        progress.advance_turn();
      } else if (progress.waiting_for_input()) {
        (void)pause(key);
      } else {
        (void)window_.release_turn(key, false);
        retiring_.push_back(key);
        (void)finish_retirement(adapter, retiring_.size() - 1, result);
      }
    }
  }
  RequestWindow window_;
  std::vector<Slot> slots_;
  std::vector<RequestKey> retiring_;
  std::optional<RequestKey> pending_eviction_;
  std::size_t resident_limit_ = 0;
  bool recover_turns_ = false;
};

} // namespace hyle::inference
