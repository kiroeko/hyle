#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace hyle::inference {

// A single request owner's delivery cursor over its existing, bounded output
// record. It owns no tokens and does not observe commits or advance model RNG.
// The owner checks the RequestKey before routing any delayed acknowledgement.
// All calls are serialized, after observing any already committed output.
class OutputDelivery final {
public:
  explicit OutputDelivery(std::size_t maximum_pending) : maximum_pending_(maximum_pending) {
    if (!maximum_pending) throw std::invalid_argument("output delivery requires positive pending capacity");
  }

  std::size_t acknowledged() const noexcept { return acknowledged_; }
  std::size_t maximum_pending() const noexcept { return maximum_pending_; }

  // Check credit before submitting the episode. The bound is the maximum real
  // user prefix it can emit, including a prefill/full-hit result. MTP may choose
  // a narrower supported episode when available credit is smaller than K+1.
  // A lack of credit pauses scheduling; it never means the turn has completed.
  std::size_t available(std::size_t produced) const {
    return maximum_pending_ - pending_count(produced);
  }
  bool can_accept(std::size_t produced, std::size_t maximum_emitted) const {
    return maximum_emitted <= available(produced);
  }

  // Synchronous borrow, invalidated by its owner's output mutations/retirement.
  // A successful model commit alone cannot acknowledge a transport write.
  std::span<const std::int32_t> pending(std::span<const std::int32_t> outputs) const {
    (void)pending_count(outputs.size());
    return outputs.subspan(acknowledged_);
  }

  // Acknowledge the absolute successfully delivered prefix, not another count
  // to add. Duplicate/older acknowledgements are idempotent. Failed/blocked
  // writes leave the cursor unchanged; the owner retains final pending output.
  void acknowledge_prefix(std::size_t produced, std::size_t through) {
    (void)pending_count(produced);
    if (through > produced)
      throw std::out_of_range("output acknowledgement exceeds the undelivered prefix");
    if (through > acknowledged_) acknowledged_ = through;
  }

private:
  std::size_t pending_count(std::size_t produced) const {
    if (produced < acknowledged_ || produced - acknowledged_ > maximum_pending_)
      throw std::out_of_range("output record exceeds its prepared delivery credit");
    return produced - acknowledged_;
  }
  std::size_t maximum_pending_, acknowledged_ = 0;
};

} // namespace hyle::inference
