#pragma once

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "hyle/compute/resource_leases.h"

namespace hyle::inference {

// A bounded cache of original output writers, not another pool or executor.
// The single owner recycles only old successfully committed state, after its
// Graph/native readers have withdrawn. Failed candidates and immutable frozen
// states are never offered. Shared readers make recycle reject unchanged.
// Capacity belongs to the active execution inventory, not every resident
// session; cached blocks continue to consume their original pool/budget.
class ReusableOutputStorage final {
public:
  ReusableOutputStorage() = default;
  ReusableOutputStorage(const ReusableOutputStorage &) = delete;
  ReusableOutputStorage &operator=(const ReusableOutputStorage &) = delete;
  ReusableOutputStorage(ReusableOutputStorage &&other) noexcept
      : slots_(std::move(other.slots_)), head_(std::exchange(other.head_, 0)),
        size_(std::exchange(other.size_, 0)),
        bytes_(std::exchange(other.bytes_, 0)) {}
  ReusableOutputStorage &operator=(ReusableOutputStorage &&other) noexcept {
    if (this != &other) {
      slots_ = std::move(other.slots_);
      head_ = std::exchange(other.head_, 0);
      size_ = std::exchange(other.size_, 0);
      bytes_ = std::exchange(other.bytes_, 0);
    }
    return *this;
  }

  // Cold metadata only. Zero capacity disables reuse. Reconfiguration needs
  // an empty cache; a rejected preparation preserves the original inventory.
  Status prepare(std::size_t capacity, std::size_t bytes) noexcept {
    if (!bytes)
      return Status::InvalidArgument("reusable output storage needs a positive byte extent");
    if (size_)
      return Status::FailedPrecondition("reusable output storage still owns cached writers");
    try {
      std::vector<compute::WriteLease> slots(capacity);
      slots_.swap(slots);
    } catch (...) {
      return Status::ResourceExhausted("reusable output storage metadata");
    }
    head_ = 0;
    bytes_ = bytes;
    return Status::Ok();
  }

  // Successful promotion consumes exactly this original lease, preserving its
  // generation, address and bytes. Failure leaves it untouched. The caller
  // supplies leases from the intended original output pool/resource only.
  bool recycle(compute::ReadLease &ready) noexcept {
    if (!ready.valid() || ready.size() != bytes_ || size_ == slots_.size())
      return false;
    auto writer = ready.try_write();
    if (!writer) return false;
    auto slot = head_ + size_;
    if (slot >= slots_.size()) slot -= slots_.size();
    slots_[slot] = std::move(*writer);
    ++size_;
    return true;
  }

  // No cached block means the caller uses its original LeasePool allocation
  // path. FIFO order preserves the member order of a stable batch instead of
  // reversing its state-address tuple each round. No ready state is published.
  std::optional<compute::WriteLease> take() noexcept {
    if (!size_) return std::nullopt;
    const auto slot = head_;
    if (++head_ == slots_.size()) head_ = 0;
    --size_;
    return std::move(slots_[slot]);
  }

  void clear() noexcept {
    while (size_) {
      slots_[head_] = {};
      if (++head_ == slots_.size()) head_ = 0;
      --size_;
    }
    head_ = 0;
  }
  std::size_t size() const noexcept { return size_; }
  std::size_t capacity() const noexcept { return slots_.size(); }
  std::size_t bytes() const noexcept { return bytes_; }

private:
  std::vector<compute::WriteLease> slots_;
  std::size_t head_ = 0, size_ = 0, bytes_ = 0;
};

} // namespace hyle::inference
