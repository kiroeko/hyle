#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <type_traits>
#include <utility>

#include "hyle/container/hash.h"
#include "hyle/container/probe.h"
#include "hyle/macros.h"
#include "hyle/mem/allocator.h"

namespace hyle::container {

// Open-addressing flat hash map with abseil SwissTable control bytes + SIMD
// group probing (probe.h) + triangular group probing. This is the rewrite clay
// only sketched (its hashmap.h is linear-probing; control bytes were a TODO).
//
// step 2 (this file as written): storage + control-byte mirror + find + insert
// + grow/rehash. step 3 adds erase (tombstones). erase is NOT here yet.
//
// K, V trivially copyable (slots relocate via assignment on rehash), consistent
// with CArray/SmallVector. Non-trivial element support is deferred.
//
// NOT a type-erased core yet -- per-instantiation code. The type-erased
// detail::*_internal(void*, item_size) anti-bloat layer is registered as the
// explicit follow-up container-04b (conventions §7), not silently skipped.
template <class K, class V, class Hash = Hash<K>, class Eq = std::equal_to<K>>
class SwissTable {
  static_assert(std::is_trivially_copyable_v<K> && std::is_trivially_copyable_v<V>,
                "SwissTable: K and V must be trivially copyable (deferred)");

  using ctrl_t = detail::ctrl_t;
  static constexpr std::size_t kGroup = detail::kGroupWidth;
  static constexpr std::size_t kMinCap = 16;  // >= both group widths, pow2

  struct Slot {
    K key;
    V value;
  };

 public:
  explicit SwissTable(mem::Allocator alloc, Hash hash = Hash{}, Eq eq = Eq{})
      : alloc_(alloc), hash_(hash), eq_(eq) {}

  ~SwissTable() { free_storage(); }
  HYLE_DISABLE_COPY(SwissTable);
  SwissTable(SwissTable&& o) noexcept
      : alloc_(o.alloc_), hash_(o.hash_), eq_(o.eq_), ctrl_(o.ctrl_),
        slots_(o.slots_), cap_(o.cap_), size_(o.size_),
        growth_left_(o.growth_left_) {
    o.ctrl_ = nullptr;
    o.slots_ = nullptr;
    o.cap_ = 0;
    o.size_ = 0;
    o.growth_left_ = 0;
  }

  std::size_t size() const noexcept { return size_; }
  std::size_t capacity() const noexcept { return cap_; }
  bool empty() const noexcept { return size_ == 0; }

  // Returns pointer to the value for `key`, or nullptr if absent.
  V* find(const K& key) noexcept {
    if (cap_ == 0) return nullptr;
    std::uint64_t h = hash_(key);
    std::size_t pos = detail::h1(h) & mask_;
    std::size_t stride = 0;
    ctrl_t want = detail::h2(h);
    for (;;) {
      detail::Group g(ctrl_ + pos);
      for (std::uint32_t i : g.match(want)) {
        std::size_t idx = (pos + i) & mask_;
        if (HYLE_LIKELY(eq_(slots_[idx].key, key))) return &slots_[idx].value;
      }
      if (g.match_empty()) return nullptr;  // an empty slot => key not present
      stride += kGroup;
      pos = (pos + stride) & mask_;
    }
  }

  bool contains(const K& key) noexcept { return find(key) != nullptr; }

  // Insert or assign. Returns true if a new key was inserted, false if updated.
  bool insert(const K& key, const V& value) {
    if (V* existing = find(key)) {
      *existing = value;
      return false;
    }
    if (growth_left_ == 0) grow();
    std::uint64_t h = hash_(key);
    std::size_t idx = find_insert_slot(h);
    set_ctrl(idx, detail::h2(h));
    slots_[idx].key = key;
    slots_[idx].value = value;
    ++size_;
    --growth_left_;
    return true;
  }

  // Remove `key`. Returns true if it was present. Marks the slot deleted
  // (tombstone): find skips past it, insert reuses it, grow()/rehash drops it.
  // (abseil's "reclaim to empty when the group has an empty" is deferred to
  // container-04b -- not needed for correctness.)
  bool erase(const K& key) noexcept {
    if (cap_ == 0) return false;
    std::uint64_t h = hash_(key);
    std::size_t pos = detail::h1(h) & mask_;
    std::size_t stride = 0;
    ctrl_t want = detail::h2(h);
    for (;;) {
      detail::Group g(ctrl_ + pos);
      for (std::uint32_t i : g.match(want)) {
        std::size_t idx = (pos + i) & mask_;
        if (HYLE_LIKELY(eq_(slots_[idx].key, key))) {
          set_ctrl(idx, detail::kDeleted);
          --size_;
          return true;
        }
      }
      if (g.match_empty()) return false;  // hit empty => key absent
      stride += kGroup;
      pos = (pos + stride) & mask_;
    }
  }

 private:
  // First empty/deleted slot along the probe sequence for hash h.
  std::size_t find_insert_slot(std::uint64_t h) noexcept {
    std::size_t pos = detail::h1(h) & mask_;
    std::size_t stride = 0;
    for (;;) {
      detail::Group g(ctrl_ + pos);
      if (detail::BitMask m = g.match_empty_or_deleted()) {
        return (pos + m.lowest()) & mask_;
      }
      stride += kGroup;
      pos = (pos + stride) & mask_;
    }
  }

  // ctrl_[i] = c, maintaining the tail mirror: ctrl_[cap_ + j] == ctrl_[j] for
  // j in [0, kGroup), so a group load near the end reads wrap-consistent bytes.
  void set_ctrl(std::size_t i, ctrl_t c) noexcept {
    ctrl_[i] = c;
    if (i < kGroup) ctrl_[cap_ + i] = c;
  }

  void grow() {
    // Double only when genuinely full; if growth_left hit 0 mostly from
    // tombstones (real size sparse), rehash SAME size to reclaim instead.
    std::size_t new_cap = cap_ == 0          ? kMinCap
                          : size_ >= cap_ * 7 / 8 ? cap_ * 2
                                                  : cap_;
    ctrl_t* old_ctrl = ctrl_;
    Slot* old_slots = slots_;
    std::size_t old_cap = cap_;

    ctrl_ = static_cast<ctrl_t*>(
        alloc_.allocate(new_cap + kGroup, alignof(ctrl_t)));
    HYLE_FATAL_ASSERT(ctrl_ != nullptr);
    slots_ = static_cast<Slot*>(
        alloc_.allocate(new_cap * sizeof(Slot), alignof(Slot)));
    HYLE_FATAL_ASSERT(slots_ != nullptr);
    std::memset(ctrl_, static_cast<int>(detail::kEmpty), new_cap + kGroup);
    cap_ = new_cap;
    mask_ = new_cap - 1;
    size_ = 0;
    growth_left_ = new_cap * 7 / 8;

    // Reinsert every full slot from the old table.
    for (std::size_t i = 0; i < old_cap; ++i) {
      if (detail::ctrl_is_full(old_ctrl[i])) {
        std::uint64_t h = hash_(old_slots[i].key);
        std::size_t idx = find_insert_slot(h);
        set_ctrl(idx, detail::h2(h));
        slots_[idx] = old_slots[i];
        ++size_;
        --growth_left_;
      }
    }
    if (old_ctrl != nullptr) {
      alloc_.deallocate(old_ctrl, old_cap + kGroup, alignof(ctrl_t));
      alloc_.deallocate(old_slots, old_cap * sizeof(Slot), alignof(Slot));
    }
  }

  void free_storage() noexcept {
    if (ctrl_ != nullptr) {
      alloc_.deallocate(ctrl_, cap_ + kGroup, alignof(ctrl_t));
      alloc_.deallocate(slots_, cap_ * sizeof(Slot), alignof(Slot));
    }
  }

  mem::Allocator alloc_;
  Hash hash_;
  Eq eq_;
  ctrl_t* ctrl_ = nullptr;   // cap_ + kGroup bytes (tail mirrors first group)
  Slot* slots_ = nullptr;    // cap_ slots
  std::size_t cap_ = 0;      // power of two (or 0 when empty)
  std::size_t mask_ = 0;     // cap_ - 1
  std::size_t size_ = 0;     // full slots
  std::size_t growth_left_ = 0;  // inserts until next grow (7/8 load)
};

}  // namespace hyle::container
