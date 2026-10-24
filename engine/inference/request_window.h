#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace hyle::inference {

class RequestWindow;
struct RequestKey final {
  std::size_t slot = (std::numeric_limits<std::size_t>::max)();
  std::uint64_t generation = 0;
  friend bool operator==(const RequestKey &, const RequestKey &) = default;
private:
  const RequestWindow *owner_ = nullptr;
  friend class RequestWindow;
};

// Single-owner FIFO/active identities only. The caller owns requests, admission
// proofs and all execution/state lifetimes, including active entries with no
// head yet. Cold storage is bounded by simultaneous slots, not historical turns.
// Valid hot transitions allocate nothing. Keys borrow this immovable window's
// lifetime; a new generation, including after prepare(), never revives an old key.
class RequestWindow final {
public:
  RequestWindow() = default;
  RequestWindow(const RequestWindow &) = delete;
  RequestWindow &operator=(const RequestWindow &) = delete;

  void prepare(std::size_t slot_capacity, std::size_t resident_limit) {
    if (!slot_capacity || !resident_limit)
      throw std::invalid_argument("request window requires positive slot/resident bounds");
    if (std::any_of(slots_.begin(), slots_.end(), [](const auto &slot) { return slot.state != State::kFree; }))
      throw std::logic_error("request window cannot reset occupied or unretired slots");
    // Commit only after every cold allocation succeeds. Generation and idle
    // counters deliberately survive a cold reset of the same window.
    std::vector<Slot> slots(slot_capacity);
    std::vector<RequestKey> waiting(slot_capacity), active;
    active.reserve((std::min)(slot_capacity, resident_limit));
    slots_.swap(slots); waiting_.swap(waiting); active_.swap(active);
    resident_limit_ = resident_limit;
    waiting_head_ = waiting_tail_ = waiting_count_ = 0;
  }

  // The caller has already prepared this free slot's payload/input ownership.
  // Occupied/full/out-of-range/exhausted rejection changes neither FIFO nor slot.
  RequestKey enqueue(std::size_t index) {
    if (index >= slots_.size()) throw std::out_of_range("request slot is outside its prepared window");
    if (slots_[index].state != State::kFree)
      throw std::logic_error("request slot still belongs to an unretired logical request");
    if (waiting_count_ == waiting_.size() || generation_ == (std::numeric_limits<std::uint64_t>::max)())
      throw std::out_of_range("request FIFO or generation capacity exhausted");
    RequestKey key; key.slot = index; key.generation = generation_ + 1; key.owner_ = this;
    push_waiting(key);
    slots_[index] = {key.generation, State::kQueued}; ++generation_;
    return key;
  }

  bool matches(RequestKey key) const noexcept {
    return key.owner_ == this && key.slot < slots_.size() && key.generation != 0 &&
        slots_[key.slot].generation == key.generation && slots_[key.slot].state != State::kFree;
  }

  std::optional<RequestKey> waiting_front() const noexcept {
    if (active_.size() == resident_limit_ || !waiting_count_) return std::nullopt;
    return waiting_[waiting_head_];
  }

  // Call only after the caller has successfully admitted this FIFO head.
  void admitted(RequestKey key) {
    if (!matches(key)) throw std::out_of_range("request admission key is stale or foreign");
    if (slots_[key.slot].state != State::kQueued || active_.size() >= resident_limit_ ||
        !waiting_count_ || waiting_[waiting_head_] != key)
      throw std::logic_error("request admission must take the waiting FIFO head into an available slot");
    active_.push_back(key);
    waiting_[waiting_head_] = {};
    waiting_head_ = next(waiting_head_); --waiting_count_;
    slots_[key.slot].state = State::kActive;
  }

  // Release only an observed complete turn. A later turn joins the FIFO tail;
  // its returned order belongs to the caller's existing idle-request record.
  // A terminal turn becomes Retiring; it does not make its payload slot reusable.
  std::size_t release_turn(RequestKey key, bool more) {
    if (!matches(key)) throw std::out_of_range("turn release key is stale or foreign");
    const auto found = std::find(active_.begin(), active_.end(), key);
    if (slots_[key.slot].state != State::kActive || found == active_.end())
      throw std::logic_error("turn release requires an active request");
    if (more && (waiting_count_ == waiting_.size() || idle_clock_ == (std::numeric_limits<std::size_t>::max)()))
      throw std::out_of_range("turn release exceeds FIFO or idle-order capacity");
    active_.erase(found);
    if (!more) { slots_[key.slot].state = State::kRetiring; return 0; }
    push_waiting(key); slots_[key.slot].state = State::kQueued;
    return ++idle_clock_;
  }

  // Withdraw only a queued request. No device work is cancelled here and no
  // payload is destroyed. An active key returns false unchanged: its owner must
  // abort/retire the actual shared batch before releasing any participant.
  bool cancel_queued(RequestKey key) noexcept {
    if (!matches(key) || slots_[key.slot].state != State::kQueued) return false;
    std::size_t offset = 0;
    while (offset < waiting_count_ && waiting_[waiting_index(offset)] != key) ++offset;
    if (offset == waiting_count_) return false;
    for (auto at = offset; at + 1 < waiting_count_; ++at)
      waiting_[waiting_index(at)] = waiting_[waiting_index(at + 1)];
    waiting_tail_ = waiting_tail_ ? waiting_tail_ - 1 : waiting_.size() - 1;
    waiting_[waiting_tail_] = {}; --waiting_count_;
    slots_[key.slot].state = State::kRetiring;
    return true;
  }

  // The caller has proved actual retirement/withdrawal and released the old
  // request payload's device uses. Retiring alone is not that proof. Active,
  // queued and stale keys reject; native cancellation is never retirement.
  void release(RequestKey key) {
    if (!matches(key)) throw std::out_of_range("request release key is stale or foreign");
    if (slots_[key.slot].state != State::kRetiring)
      throw std::logic_error("request slot can be reused only after terminal retirement");
    slots_[key.slot].state = State::kFree;
  }

  // Synchronous borrow, invalidated by mutations; callers must copy a key before
  // releasing an element and may not carry this span across an event-loop step.
  std::span<const RequestKey> active() const noexcept { return {active_.data(), active_.size()}; }

  // Pure query over actual caller facts. This neither retires a head nor
  // promises that dropping a resident will free any version or byte budget.
  template<class Resident, class IdleOrder>
  std::optional<std::size_t> idle_victim(Resident resident, IdleOrder idle_order) const {
    std::optional<std::size_t> victim;
    for (std::size_t index = 0; index < slots_.size(); ++index)
      if (slots_[index].state == State::kQueued && resident(index) &&
          (!victim || idle_order(index) < idle_order(*victim))) victim = index;
    return victim;
  }

private:
  enum class State : std::uint8_t { kFree, kQueued, kActive, kRetiring };
  struct Slot final { std::uint64_t generation = 0; State state = State::kFree; };
  std::size_t next(std::size_t index) const noexcept { return index + 1 == waiting_.size() ? 0 : index + 1; }
  std::size_t waiting_index(std::size_t offset) const noexcept {
    const auto suffix = waiting_.size() - waiting_head_;
    return offset < suffix ? waiting_head_ + offset : offset - suffix;
  }
  void push_waiting(RequestKey key) noexcept {
    waiting_[waiting_tail_] = key; waiting_tail_ = next(waiting_tail_); ++waiting_count_;
  }
  std::vector<Slot> slots_;
  std::vector<RequestKey> waiting_, active_;
  std::size_t resident_limit_ = 0, waiting_head_ = 0, waiting_tail_ = 0, waiting_count_ = 0, idle_clock_ = 0;
  std::uint64_t generation_ = 0;
};

} // namespace hyle::inference
