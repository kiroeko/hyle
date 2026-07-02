#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>

#include "hyle/container/intrusive_list.h"

// Hierarchical timing wheel: the data structure behind "millions of agents each
// arming a timeout". The workload is add/cancel >> expire (a response usually
// arrives and cancels the timeout before it fires), so we optimise cancel to
// O(1) and never pay to keep entries sorted.
//
// SOTA lineage (ported, not invented -- the timing-wheel family is classic):
//   - tokio runtime/time/wheel: 6 levels x 64 slots, 1ms base, computes the
//     exact next wakeup instead of ticking. This is the PRIMARY port -- the
//     level_for/slot_for bit math and the `occupied` bitmap scan are tokio's.
//   - folly HHWheelTimer: 4 levels x 256 slots, also deadline-driven ("does
//     *not* tick constantly"). We took the "no busy tick" idea; chose 64-slot
//     levels (tokio) over 256 (folly) so `occupied` is a single u64 per level
//     and next-slot search is one countr_zero.
//   - seastar timer_set: one bucket per (timestamp ^ last) leading-bit class,
//     defers sorting to expiry. Confirmed the "sort at expiry, not insert"
//     principle; its log(range) buckets are a coarser wheel -- we keep the
//     fixed 6x64 hierarchy for O(1) slotting.
//
// vs a min-heap (Go/libev): heap add/cancel are O(log n) and lazy-cancel bloats
// the heap under add-then-cancel churn -- exactly our workload's worst case.
// The wheel gives O(1) add (amortised), O(1) cancel (unlink an intrusive node,
// NEVER a tombstone), O(fired) expiry.
//
// NOT a scheduler and NOT thread-safe by itself: one wheel is owned by one
// worker (per-runqueue). Cross-core arm/cancel is delivered as a command over
// the scheduler's MPSC queue, not by locking the wheel. Expiry does NOT run
// callbacks -- it hands the fired entries back so the caller reinjects the
// associated fibers into its run-queue (same wake path as io/gpu completions).

namespace hyle::time {

// One armed timer. Embed it in (or point it at) whatever the caller wants woken;
// the wheel treats `data` as opaque and never dereferences it. The intrusive
// hook means arming/cancelling allocates nothing.
struct TimerEntry {
  container::IntrusiveListHook hook;
  // Absolute expiry in wheel ticks (1 tick = 1ms since the wheel epoch). Set by
  // schedule(); read by cancel() to recover the (level, slot) in O(1).
  std::uint64_t deadline_tick = 0;
  // Opaque payload -- typically the Waker/fiber pointer the caller wants woken.
  void* data = nullptr;
  // True iff currently sitting in one of the wheel's slots (not yet fired and
  // not cancelled). Distinguishes "in the wheel" from "handed back on the
  // expired list" so the caller may reuse `hook` on its own list after firing.
  bool scheduled = false;
};

class TimerWheel {
 public:
  using EntryList = container::IntrusiveList<TimerEntry, &TimerEntry::hook>;

  // 6 levels x 64 slots, 1ms base. 6 bits index a level's 64 slots; 6 levels
  // cover 2^(6*6) ticks ~= 2.18 years at 1ms granularity -- past any real fiber
  // timeout. These are the tokio parameters.
  static constexpr std::uint64_t kBitsPerLevel = 6;
  static constexpr std::uint64_t kSlotsPerLevel = 1ull << kBitsPerLevel;  // 64
  static constexpr std::uint64_t kSlotMask = kSlotsPerLevel - 1;          // 63
  static constexpr std::uint64_t kNumLevels = 6;
  static constexpr std::uint64_t kTickNs = 1'000'000;  // 1ms per L0 slot
  // The largest schedulable horizon in ticks. Anything past this is fudged onto
  // the top level's pseudo-ring (see next_expiration wrap handling).
  static constexpr std::uint64_t kMaxDurationTicks =
      1ull << (kBitsPerLevel * kNumLevels);

  // `start_ns` is the monotonic-ns origin (hyle::time::monotonic_ns()). All
  // deadlines are measured from it; the wheel never reads a clock itself so it
  // stays a pure, testable data structure (the caller drives `now`).
  explicit TimerWheel(std::uint64_t start_ns) noexcept : start_ns_(start_ns) {}
  TimerWheel(const TimerWheel&) = delete;
  TimerWheel& operator=(const TimerWheel&) = delete;

  // Arm `e` to fire at absolute monotonic `deadline_ns`. O(1) amortised. If `e`
  // is already scheduled this reschedules it (cancel + re-arm) -- matches folly
  // scheduleTimeout. A deadline at/behind the wheel's current time is pinned to
  // the next tick so it fires on the following advance() (never dropped).
  void schedule(TimerEntry* e, std::uint64_t deadline_ns) noexcept {
    if (e->scheduled) cancel(e);

    std::uint64_t rel_ns =
        deadline_ns > start_ns_ ? deadline_ns - start_ns_ : 0;
    // Round the deadline UP to a whole tick so we never fire EARLY (advance()
    // rounds `now` down; the pair guarantees "at least the requested wait",
    // late by <=1ms -- the documented wheel contract).
    std::uint64_t tick = (rel_ns + kTickNs - 1) / kTickNs;
    if (tick <= elapsed_) {
      tick = elapsed_ + 1;  // due now/past -> nearest future slot
    }
    if (tick > elapsed_ + kMaxDurationTicks) {
      tick = elapsed_ + kMaxDurationTicks;  // clamp to the horizon
    }

    e->deadline_tick = tick;
    add_entry(level_for(elapsed_, tick), e);
    e->scheduled = true;
  }

  // O(1) cancel: recover (level, slot) from the stored deadline and unlink the
  // node. No tombstone, no scan -- this is the whole point of the wheel for the
  // arm-then-cancel-heavy workload. No-op if `e` is not scheduled.
  void cancel(TimerEntry* e) noexcept {
    if (!e->scheduled) return;
    std::uint64_t level = level_for(elapsed_, e->deadline_tick);
    std::uint64_t slot = slot_for(e->deadline_tick, level);
    slots_[level][slot].remove(e);
    if (slots_[level][slot].empty()) {
      occupied_[level] &= ~slot_bit(slot);
    }
    e->scheduled = false;
    --count_;
  }

  // Advance the wheel to absolute monotonic `now_ns`, moving every entry whose
  // deadline has passed onto `out` (FIFO within a slot). Does NOT run anything:
  // the caller drains `out` and reinjects each entry's fiber. Entries in higher
  // levels whose deadline is still in the future cascade down to finer levels.
  void advance(std::uint64_t now_ns, EntryList& out) noexcept {
    std::uint64_t now_tick =
        now_ns > start_ns_ ? (now_ns - start_ns_) / kTickNs : 0;

    // Process each expiring slot in deadline order until the earliest pending
    // slot is still in the future, then jump elapsed_ to now.
    for (;;) {
      Expiration exp;
      if (next_expiration(exp) && exp.deadline <= now_tick) {
        process_expiration(exp, out);
        set_elapsed(exp.deadline);
      } else {
        set_elapsed(now_tick);
        break;
      }
    }
  }

  // Absolute monotonic ns of the next expiry, or UINT64_MAX if the wheel is
  // empty. This is the "no busy tick" core: the caller feeds
  // (next_deadline_ns() - now) as the timeout to park_timeout(), so exactly one
  // worker sleeps to the next deadline instead of spinning every millisecond.
  std::uint64_t next_deadline_ns() const noexcept {
    Expiration exp;
    if (!next_expiration(exp)) return UINT64_MAX;
    return start_ns_ + exp.deadline * kTickNs;
  }

  bool empty() const noexcept { return count_ == 0; }
  std::size_t size() const noexcept { return count_; }

 private:
  // A slot due for processing: which (level, slot) and the tick it represents.
  struct Expiration {
    std::uint64_t level = 0;
    std::uint64_t slot = 0;
    std::uint64_t deadline = 0;  // in ticks
  };

  static constexpr std::uint64_t slot_bit(std::uint64_t slot) noexcept {
    return 1ull << slot;
  }

  // slot = the `level`-th 6-bit digit of `tick` (tokio slot_for).
  static std::uint64_t slot_for(std::uint64_t tick,
                                std::uint64_t level) noexcept {
    return (tick >> (level * kBitsPerLevel)) & kSlotMask;
  }

  // Which level an entry belongs in, given the current `elapsed`: the position
  // of the highest bit in which `elapsed` and `when` differ, divided by the
  // bits-per-level. (tokio level_for.) The `| kSlotMask` floors the search at
  // level 0; the clamp forces anything past the horizon onto the top level.
  static std::uint64_t level_for(std::uint64_t elapsed,
                                 std::uint64_t when) noexcept {
    std::uint64_t masked = (elapsed ^ when) | kSlotMask;
    if (masked >= kMaxDurationTicks) {
      masked = kMaxDurationTicks - 1;
    }
    std::uint64_t significant = 63 - static_cast<std::uint64_t>(
                                         std::countl_zero(masked));
    return significant / kBitsPerLevel;
  }

  void add_entry(std::uint64_t level, TimerEntry* e) noexcept {
    std::uint64_t slot = slot_for(e->deadline_tick, level);
    slots_[level][slot].push_back(e);  // FIFO within a slot
    occupied_[level] |= slot_bit(slot);
    ++count_;
  }

  // Earliest slot needing processing across all levels. The lowest non-empty
  // level always holds the nearest deadline (the hierarchy guarantees no higher
  // level can have anything earlier), so the first hit as we climb wins.
  bool next_expiration(Expiration& out) const noexcept {
    for (std::uint64_t level = 0; level < kNumLevels; ++level) {
      if (occupied_[level] == 0) continue;
      out.level = level;
      out.slot = next_occupied_slot(level);
      out.deadline = slot_deadline(level, out.slot);
      return true;
    }
    return false;
  }

  // Nearest occupied slot at/after the current position in this level, using the
  // occupied bitmap so we never scan empty slots (tokio next_occupied_slot):
  // rotate the bitmap so "now" is bit 0, then countr_zero finds the next set.
  std::uint64_t next_occupied_slot(std::uint64_t level) const noexcept {
    std::uint64_t now_slot = elapsed_ >> (level * kBitsPerLevel);
    std::uint64_t rotated =
        std::rotr(occupied_[level], static_cast<int>(now_slot & kSlotMask));
    std::uint64_t zeros =
        static_cast<std::uint64_t>(std::countr_zero(rotated));
    return (zeros + now_slot) & kSlotMask;
  }

  // The absolute tick a (level, slot) represents relative to elapsed_. If it
  // computes to <= elapsed_ we are on the top level's pseudo-ring (an entry
  // scheduled > MAX_DURATION out wrapped around), so it is really one full level
  // range in the future.
  std::uint64_t slot_deadline(std::uint64_t level,
                              std::uint64_t slot) const noexcept {
    std::uint64_t slot_range = 1ull << (level * kBitsPerLevel);
    std::uint64_t level_range = 1ull << ((level + 1) * kBitsPerLevel);
    std::uint64_t level_start = elapsed_ & ~(level_range - 1);
    std::uint64_t deadline = level_start + slot * slot_range;
    if (deadline <= elapsed_) {
      deadline += level_range;  // top-level wrap
    }
    return deadline;
  }

  // Drain the expiring slot and either fire (level 0, or already-due) or cascade
  // each entry down to a finer level. We take ALL entries out first: on the top
  // level a wrapped entry can re-insert into the same slot, and processing it
  // again in this pass would loop forever (tokio's take-then-process note).
  void process_expiration(const Expiration& exp, EntryList& out) noexcept {
    EntryList draining;
    EntryList& slot = slots_[exp.level][exp.slot];
    while (TimerEntry* e = slot.pop_front()) {
      draining.push_back(e);
    }
    occupied_[exp.level] &= ~slot_bit(exp.slot);

    while (TimerEntry* e = draining.pop_front()) {
      if (exp.level == 0 || e->deadline_tick <= exp.deadline) {
        // Reached 1ms granularity (or already due): fire. Hand ownership back;
        // the caller may now relink `hook` onto its own run-queue list.
        e->scheduled = false;
        --count_;
        out.push_back(e);
      } else {
        // Still in the future: re-slot into a finer level now that elapsed_ has
        // advanced to this slot's tick. count_ is unchanged (still in the wheel).
        add_entry_no_count(level_for(exp.deadline, e->deadline_tick), e);
      }
    }
  }

  // add_entry variant used during cascade: the entry never left the wheel's
  // population, so it must not double-count.
  void add_entry_no_count(std::uint64_t level, TimerEntry* e) noexcept {
    std::uint64_t slot = slot_for(e->deadline_tick, level);
    slots_[level][slot].push_back(e);
    occupied_[level] |= slot_bit(slot);
  }

  void set_elapsed(std::uint64_t when) noexcept {
    if (when > elapsed_) elapsed_ = when;
  }

  std::uint64_t start_ns_;        // monotonic-ns origin
  std::uint64_t elapsed_ = 0;     // ticks processed so far
  std::size_t count_ = 0;         // live entries (for empty()/leak checks)
  std::uint64_t occupied_[kNumLevels] = {};  // per-level non-empty-slot bitmap
  EntryList slots_[kNumLevels][kSlotsPerLevel];
};

}  // namespace hyle::time
