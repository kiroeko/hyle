#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace hyle::inference {

// Single-owner FIFO/active indices only. The caller owns requests, admission
// proofs and all execution/state lifetimes, including active entries with no
// head yet. prepare is cold and starts a new window; valid transitions allocate
// no storage. Copies/moves cannot lose its prepared vector capacities.
class RequestWindow final {
public:
  RequestWindow() = default;
  RequestWindow(const RequestWindow &) = delete;
  RequestWindow &operator=(const RequestWindow &) = delete;

  void prepare(std::size_t request_count, std::size_t total_turns, std::size_t resident_limit) {
    if (!request_count || !resident_limit || total_turns < request_count)
      throw std::invalid_argument("request window requires positive request/resident bounds and at least one turn per request");
    waiting_.reserve(total_turns);
    active_.reserve(resident_limit);
    waiting_.clear(); active_.clear();
    for (std::size_t index = 0; index < request_count; ++index) waiting_.push_back(index);
    request_count_ = request_count;
    total_turns_ = total_turns;
    resident_limit_ = resident_limit;
    waiting_cursor_ = 0;
    idle_clock_ = 0;
  }

  std::optional<std::size_t> waiting_front() const noexcept {
    if (active_.size() == resident_limit_ || waiting_cursor_ == waiting_.size()) return std::nullopt;
    return waiting_[waiting_cursor_];
  }

  // Call only after the caller has successfully admitted this FIFO head.
  void admitted(std::size_t index) {
    if (index >= request_count_) throw std::out_of_range("request admission index is outside its prepared window");
    if (active_.size() >= resident_limit_ || waiting_cursor_ == waiting_.size() || waiting_[waiting_cursor_] != index ||
        std::find(active_.begin(), active_.end(), index) != active_.end())
      throw std::logic_error("request admission must take the waiting FIFO head into an available slot");
    active_.push_back(index);
    ++waiting_cursor_;
  }

  // Release only an observed complete turn. A later turn joins the FIFO tail;
  // its returned order belongs to the caller's existing idle-request record.
  std::size_t release_turn(std::size_t index, bool more) {
    const auto found = std::find(active_.begin(), active_.end(), index);
    if (found == active_.end()) throw std::logic_error("turn release requires an active request");
    if (more && (waiting_.size() == total_turns_ || idle_clock_ == (std::numeric_limits<std::size_t>::max)()))
      throw std::out_of_range("turn release exceeds its cold-prepared request window");
    active_.erase(found);
    if (!more) return 0;
    waiting_.push_back(index);
    return ++idle_clock_;
  }

  // Synchronous borrow, invalidated by prepare/admitted/release_turn.
  std::span<const std::size_t> active() const noexcept { return {active_.data(), active_.size()}; }

  // Pure query over actual caller facts. This neither retires a head nor
  // promises that dropping a resident will free any version or byte budget.
  template<class Resident, class IdleOrder>
  std::optional<std::size_t> idle_victim(Resident resident, IdleOrder idle_order) const {
    std::optional<std::size_t> victim;
    for (std::size_t index = 0; index < request_count_; ++index)
      if (resident(index) && std::find(active_.begin(), active_.end(), index) == active_.end() &&
          (!victim || idle_order(index) < idle_order(*victim))) victim = index;
    return victim;
  }

private:
  std::vector<std::size_t> waiting_, active_;
  std::size_t request_count_ = 0, total_turns_ = 0, resident_limit_ = 0;
  std::size_t waiting_cursor_ = 0, idle_clock_ = 0;
};

} // namespace hyle::inference
