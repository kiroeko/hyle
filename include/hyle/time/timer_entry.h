#pragma once

#include <cstdint>
#include <type_traits>

#include "hyle/container/intrusive_list.h"
#include "hyle/core/compiler.h"

namespace hyle::time {

namespace detail {
struct TimerEntryAccess;
struct TimerEntryListAccess;
}

// Intrusive timer node shared by the concrete timer containers. The caller owns
// the node and must keep its address stable while scheduled. Container metadata
// stays private: users manipulate payload/callback/deadline only through the
// accessors below, so a public data-structure API does not expose its bucket and
// ownership implementation.
class TimerEntry {
 public:
  using FireCallback = void (*)(TimerEntry*) noexcept;

  TimerEntry() noexcept = default;
  explicit TimerEntry(void* payload,
                      FireCallback callback = nullptr) noexcept
      : payload_(payload), fire_callback_(callback) {}
  TimerEntry(const TimerEntry&) = delete;
  TimerEntry& operator=(const TimerEntry&) = delete;
  TimerEntry(TimerEntry&&) = delete;
  TimerEntry& operator=(TimerEntry&&) = delete;

  ~TimerEntry() {
    // An armed node is still referenced by its source, and a fired node must
    // outlive its delivery list. Destroying the list first is safe because the
    // container_intrusive list detaches all hooks during destruction. The safe
    // hook independently diagnoses a linked-node lifetime violation in Debug.
    HYLE_FATAL_ASSERT(owner_ == nullptr);
  }

  [[nodiscard]] bool scheduled() const noexcept { return owner_ != nullptr; }
  [[nodiscard]] std::uint64_t deadline_ns() const noexcept {
    return deadline_ns_;
  }
  [[nodiscard]] void* data() const noexcept { return payload_; }
  void set_data(void* payload) noexcept { payload_ = payload; }
  [[nodiscard]] FireCallback fire_callback() const noexcept {
    return fire_callback_;
  }
  void set_fire_callback(FireCallback callback) noexcept {
    fire_callback_ = callback;
  }

 private:
  friend class TimerQueue;
  friend class CommonTimeoutQueue;
  friend struct detail::TimerEntryAccess;
  friend struct detail::TimerEntryListAccess;

  [[nodiscard]] bool owned_by(const void* source) const noexcept {
    return owner_ == source;
  }
  void set_owner(const void* source) noexcept { owner_ = source; }

  container::SafeIntrusiveListHook hook_;
  std::uint64_t deadline_ns_ = 0;
  void* payload_ = nullptr;
  FireCallback fire_callback_ = nullptr;
  const void* owner_ = nullptr;
};

static_assert(std::is_standard_layout_v<TimerEntry>);

namespace detail {
// Private queue adapter. The public API deliberately does not expose an owner
// token, while queue implementations must distinguish idle/on-this-source from
// owned-by-another-source without duplicating an owner pointer.
struct TimerEntryAccess {
  static bool owned_by(const TimerEntry& entry, const void* source) noexcept {
    return entry.owned_by(source);
  }
};

struct TimerEntryListAccess {
  using Type = container::SafeIntrusiveList<TimerEntry, &TimerEntry::hook_>;
};
}  // namespace detail

// Intrusive delivery list of fired/cancelled entries. A source hands nodes back
// without running callbacks; sched/io consumers decide how and where user work
// executes. TimerQueue preserves insertion order within one radix bucket, but
// callers must not rely on ordering across different deadlines/buckets.
using TimerList = detail::TimerEntryListAccess::Type;

}  // namespace hyle::time
