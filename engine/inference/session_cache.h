// Bounded consumer cache metadata. This is not a model owner or scheduler.
// Use only at admission/turn/idle boundaries, never on the evaluate hot path.
// Control is externally serialized. A withdrawn entry stays charged until its
// original owner proves retirement; physical page budgets remain separate.
#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace hyle::inference {

struct SessionCacheLimits {
  std::size_t max_sessions;
  std::size_t max_bytes;
};

enum class CacheAdmissionStatus {
  kAdmitted,
  kAlreadyResident,
  kCapacityExceeded,
  kRetirementUnproven,
  kByteCountMismatch,
};

struct CacheAdmissionResult {
  CacheAdmissionStatus status;
  std::size_t evicted_sessions = 0;
  std::size_t evicted_bytes = 0;
  bool accepted() const noexcept {
    return status == CacheAdmissionStatus::kAdmitted ||
        status == CacheAdmissionStatus::kAlreadyResident;
  }
};

class SessionCache final {
  struct Entry {
    std::size_t owner_index = 0, bytes = 0;
    bool occupied = false, active = false, retiring = false;
  };
  static constexpr std::size_t kNoSlot = (std::numeric_limits<std::size_t>::max)();
  SessionCacheLimits limits_;
  // Entry slots never move after construction. Only idle slot indices reorder.
  std::vector<Entry> entries_;
  std::vector<std::size_t> idle_;
  std::size_t retained_sessions_ = 0, retained_bytes_ = 0;

  std::size_t find(std::size_t owner_index) const noexcept {
    for (std::size_t slot = 0; slot < entries_.size(); ++slot)
      if (entries_[slot].occupied && entries_[slot].owner_index == owner_index) return slot;
    return kNoSlot;
  }
  void release_pin(std::size_t slot) noexcept {
    entries_[slot].active = false;
    // A retiring entry never becomes visible/evictable again when its old pin
    // is returned. The slot cannot be reused before that pin is gone.
    if (!entries_[slot].retiring) idle_.push_back(slot);
  }

public:
  // A pin borrows this directory and prevents its entry's eviction. Its owner
  // index can select the caller's stable Session owner once at a boundary;
  // evaluate then uses that direct Session pointer without consulting us.
  // Destroy/reset all pins before destroying their SessionCache.
  class Pin final {
    friend class SessionCache;
    SessionCache *cache_ = nullptr;
    std::size_t slot_ = kNoSlot, owner_index_ = kNoSlot;
    Pin(SessionCache &cache, std::size_t slot) noexcept
        : cache_(&cache), slot_(slot), owner_index_(cache.entries_[slot].owner_index) {}
  public:
    Pin() noexcept = default;
    Pin(const Pin &) = delete;
    Pin &operator=(const Pin &) = delete;
    Pin(Pin &&other) noexcept
        : cache_(std::exchange(other.cache_, nullptr)), slot_(other.slot_),
          owner_index_(other.owner_index_) {}
    Pin &operator=(Pin &&other) noexcept {
      if (this != &other) {
        reset();
        cache_ = std::exchange(other.cache_, nullptr);
        slot_ = other.slot_; owner_index_ = other.owner_index_;
      }
      return *this;
    }
    ~Pin() { reset(); }
    explicit operator bool() const noexcept { return cache_ != nullptr; }
    std::size_t owner_index() const noexcept { return owner_index_; }
    void reset() noexcept {
      if (cache_) {
        auto *cache = std::exchange(cache_, nullptr);
        cache->release_pin(slot_);
      }
    }
  };

  explicit SessionCache(SessionCacheLimits limits) : limits_(limits) {
    if (limits.max_sessions == 0 || limits.max_sessions > entries_.max_size() ||
        limits.max_sessions > idle_.max_size())
      throw std::runtime_error("session cache requires a finite positive session limit");
    entries_.resize(limits.max_sessions);
    idle_.reserve(limits.max_sessions);
  }
  SessionCache(const SessionCache &) = delete;
  SessionCache &operator=(const SessionCache &) = delete;
  SessionCache(SessionCache &&) = delete;
  SessionCache &operator=(SessionCache &&) = delete;

  bool contains(std::size_t owner_index) const noexcept {
    const auto slot = find(owner_index);
    return slot != kNoSlot && !entries_[slot].retiring;
  }
  bool retiring(std::size_t owner_index) const noexcept {
    const auto slot = find(owner_index);
    return slot != kNoSlot && entries_[slot].retiring;
  }
  std::size_t retained_sessions() const noexcept { return retained_sessions_; }
  std::size_t retained_bytes() const noexcept { return retained_bytes_; }

  // Withdraw discoverability before touching the native owner. No wait or
  // charge return; an existing pin must be returned before collection.
  bool begin_retire(std::size_t owner_index) noexcept {
    const auto slot = find(owner_index);
    if (slot == kNoSlot) return false;
    auto &entry = entries_[slot];
    if (!entry.retiring) {
      entry.retiring = true;
      for (auto it = idle_.begin(); it != idle_.end(); ++it)
        if (*it == slot) { idle_.erase(it); break; }
    }
    return true;
  }
  // collect must nonblockingly prove actual owner retirement/release. A false
  // return or exception leaves the same invisible entry and its charge intact.
  // Never reenter this directory from the callback.
  template<class Collect>
  bool collect_retired(std::size_t owner_index, Collect &&collect) {
    const auto slot = find(owner_index);
    if (slot == kNoSlot || !entries_[slot].retiring || entries_[slot].active) return false;
    if (!collect(owner_index)) return false;
    retained_bytes_ -= entries_[slot].bytes; --retained_sessions_;
    entries_[slot] = {};
    return true;
  }

  // One exclusive pin per entry. No owner load, allocation or native work.
  Pin pin(std::size_t owner_index) noexcept {
    const auto slot = find(owner_index);
    if (slot == kNoSlot || entries_[slot].active || entries_[slot].retiring) return {};
    for (auto it = idle_.begin(); it != idle_.end(); ++it) {
      if (*it != slot) continue;
      idle_.erase(it);
      entries_[slot].active = true;
      return Pin(*this, slot);
    }
    return {};
  }

  // reported_bytes is supplied by the resource owner, not estimated here.
  // It must remain unchanged while resident. Shared-resource deduplication,
  // physical/unknown provider memory and async page retirement need their own
  // resource ledger; this directory neither reserves nor measures that memory.
  //
  // retire(owner_index) may start retirement and return false while work is
  // pending. The victim is withdrawn first and remains charged. A later
  // collect_retired releases it only after the original owner is gone. Existing
  // synchronous callers use this same lifecycle. Callbacks must not reenter the
  // directory; an exception leaves the victim invisible and charged.
  template<class Retire>
  CacheAdmissionResult admit(std::size_t owner_index, std::size_t reported_bytes,
                              Retire &&retire) {
    const auto existing = find(owner_index);
    if (existing != kNoSlot) {
      if (entries_[existing].retiring) return {CacheAdmissionStatus::kRetirementUnproven};
      return {entries_[existing].bytes == reported_bytes ? CacheAdmissionStatus::kAlreadyResident
          : CacheAdmissionStatus::kByteCountMismatch};
    }
    if (reported_bytes > limits_.max_bytes)
      return {CacheAdmissionStatus::kCapacityExceeded};
    std::size_t reclaimable = 0;
    for (const auto slot : idle_) reclaimable += entries_[slot].bytes;
    // reclaimable <= retained_bytes, so the addition is bounded by max_bytes.
    if ((retained_sessions_ == limits_.max_sessions && idle_.empty()) ||
        reported_bytes > limits_.max_bytes - retained_bytes_ + reclaimable) {
      for (const auto &entry : entries_)
        if (entry.occupied && entry.retiring) return {CacheAdmissionStatus::kRetirementUnproven};
      return {CacheAdmissionStatus::kCapacityExceeded};
    }
    CacheAdmissionResult result{CacheAdmissionStatus::kAdmitted};
    while (retained_sessions_ == limits_.max_sessions ||
        reported_bytes > limits_.max_bytes - retained_bytes_) {
      if (idle_.empty()) { result.status = CacheAdmissionStatus::kRetirementUnproven; return result; }
      const auto slot = idle_.front();
      const auto entry = entries_[slot];
      begin_retire(entry.owner_index);
      if (!collect_retired(entry.owner_index, retire)) {
        result.status = CacheAdmissionStatus::kRetirementUnproven;
        return result;
      }
      // Actual retirement/release is proven before any metadata charge returns.
      ++result.evicted_sessions; result.evicted_bytes += entry.bytes;
    }
    for (std::size_t slot = 0; slot < entries_.size(); ++slot) {
      if (entries_[slot].occupied) continue;
      entries_[slot] = {owner_index, reported_bytes, true, false};
      ++retained_sessions_; retained_bytes_ += reported_bytes;
      idle_.push_back(slot);
      return result;
    }
    throw std::logic_error("session cache free-slot invariant violated");
  }
};

} // namespace hyle::inference
