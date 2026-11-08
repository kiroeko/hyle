#pragma once

#include <concepts>
#include <exception>
#include <optional>
#include <type_traits>
#include <utility>

#include "request_progress.h"
#include "request_stop_policy.h"
#include "request_window.h"

namespace hyle::inference {

// Borrowed input notification for the original serialized request owner.
// A closed source may still have queued turns: these are returned first. The
// producer must also wake the output consumer's shared wait on input changes;
// polling and waiting may not lose a concurrent append, close or cancellation.
enum class TurnInputState { kWaiting, kTurn, kClosed };
struct TurnInput final { TurnInputState state = TurnInputState::kWaiting; GenerationTurn turn{}; };
struct GenerationAdmission final {
  GenerationSessionSchedule schedule;
  RequestStopPolicy stop;
  std::size_t sampling_index = 0;
};

class GenerationInputRef final {
public:
  template<class T> requires (!std::same_as<std::remove_cvref_t<T>, GenerationInputRef> &&
      requires(T &source, RequestKey key) { { source.poll_turn(key) } noexcept -> std::same_as<TurnInput>; })
  explicit GenerationInputRef(T &source) noexcept : source_(&source), poll_([](void *p, RequestKey key) noexcept {
    return static_cast<T *>(p)->poll_turn(key);
  }), admission_([](void *p, std::size_t slot) noexcept -> std::optional<GenerationAdmission> {
    if constexpr (requires(T &s) { { s.poll_admission(slot) } noexcept -> std::same_as<std::optional<GenerationAdmission>>; })
      return static_cast<T *>(p)->poll_admission(slot);
    else { (void)p; (void)slot; return std::nullopt; }
  }), reject_([](void *p, std::size_t slot, std::exception_ptr error) noexcept {
    if constexpr (requires(T &s) { { s.reject_admission(slot, error) } noexcept -> std::same_as<void>; })
      static_cast<T *>(p)->reject_admission(slot, error);
    else { (void)p; (void)slot; (void)error; }
  }), available_([](void *p, std::size_t slot) noexcept {
    if constexpr (requires(T &s) { { s.can_admit(slot) } noexcept -> std::same_as<bool>; })
      return static_cast<T *>(p)->can_admit(slot);
    else { (void)p; (void)slot; return false; }
  }) {}
  TurnInput poll_turn(RequestKey key) noexcept { return poll_(source_, key); }
  bool can_admit(std::size_t slot) noexcept { return available_(source_, slot); }
  std::optional<GenerationAdmission> poll_admission(std::size_t slot) noexcept { return admission_(source_, slot); }
  void reject_admission(std::size_t slot, std::exception_ptr error) noexcept { reject_(source_, slot, error); }
private:
  void *source_;
  TurnInput (*poll_)(void *, RequestKey) noexcept;
  std::optional<GenerationAdmission> (*admission_)(void *, std::size_t) noexcept;
  void (*reject_)(void *, std::size_t, std::exception_ptr) noexcept;
  bool (*available_)(void *, std::size_t) noexcept;
};

} // namespace hyle::inference
