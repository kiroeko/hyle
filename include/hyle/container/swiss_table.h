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

namespace detail {
// Type-erased storage handle for the cold path. The hot find/insert/erase stay
// inlined in the typed shell (indirect hash/eq on every probe would wreck cheap
// keys); only the heavy resize+free machinery is compiled ONCE here, so it does
// not bloat per <K,V> instantiation (conventions §7).
struct SwissStorage {
  ctrl_t* ctrl;
  void* slots;
  std::size_t cap;
  std::size_t mask;
  std::size_t size;
  std::size_t growth_left;
};

// Allocate new_cap slots, memset ctrl to empty, reinsert every full old slot,
// free the old storage. `hash_slot(slot, hash_state)` recomputes a slot's hash
// without knowing K (the key sits at slot offset 0). Updates `s` in place.
HYLE_CONTAINER_API void swiss_resize(SwissStorage& s, std::size_t new_cap,
                                     std::size_t slot_size,
                                     std::size_t slot_align, mem::Allocator alloc,
                                     std::uint64_t (*hash_slot)(const void*,
                                                                const void*),
                                     const void* hash_state);

// Free a table's ctrl+slot storage (no-op when empty).
HYLE_CONTAINER_API void swiss_free(SwissStorage& s, std::size_t slot_size,
                                   std::size_t slot_align, mem::Allocator alloc);
}  // namespace detail

// Open-addressing flat hash map with abseil SwissTable control bytes + SIMD
// group probing (probe.h) + triangular group probing. This is the rewrite clay
// only sketched (its hashmap.h is linear-probing; control bytes were a TODO).
//
// find + insert + grow/rehash + erase (tombstones) + for_each iteration are all
// here. erase marks a tombstone, but RECLAIMS the slot to empty when its group
// still has an empty (abseil's optimization: the probe sequence never ran full
// through it, so no later key depends on it) -- this returns growth and curbs
// tombstone buildup under churn.
//
// K, V trivially copyable (slots relocate via memcpy on rehash), consistent with
// CArray/SmallVector. Non-trivial element support is deferred.
//
// The cold path (resize/free) is type-erased into swiss_table.cc (see detail::
// above); the hot path is inlined per instantiation for zero indirection.
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

  // Remove `key`. Returns true if it was present. The slot becomes empty when
  // its group still has an empty lane (the probe run never filled through it, so
  // no later key needs it as a stepping stone) -- which restores growth; else it
  // becomes a tombstone (find skips it, insert reuses it, grow drops it). This is
  // abseil's reclaim rule and keeps capacity tight under heavy churn.
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
          erase_at(idx);
          return true;
        }
      }
      if (g.match_empty()) return false;  // hit empty => key absent
      stride += kGroup;
      pos = (pos + stride) & mask_;
    }
  }

  // Visit every live (key, value): calls fn(const K&, V&) once per full slot, in
  // unspecified order. A function template (not std::function) so it inlines and
  // adds no allocation. Needed by any owner of heap-pointer values (e.g.
  // StringBuf) to release them on teardown, since there is no iterator yet.
  // fn must not insert/erase/grow the table while iterating.
  template <class Fn>
  void for_each(Fn&& fn) {
    for (std::size_t i = 0; i < cap_; ++i)
      if (detail::ctrl_is_full(ctrl_[i])) fn(slots_[i].key, slots_[i].value);
  }

 private:
  // Recompute a slot's key hash without the concrete type (key is at slot offset
  // 0). Handed to the type-erased core so resize can rehash old slots.
  static std::uint64_t hash_slot_thunk(const void* slot,
                                       const void* hash_state) noexcept {
    return (*static_cast<const Hash*>(hash_state))(
        *static_cast<const K*>(slot));
  }

  // Erase the live slot at `idx`. abseil reclaim: if the group at idx AND the
  // group one width before both have an empty lane, AND the run of non-empties
  // bridging idx is shorter than a group, then nothing probed *through* idx, so
  // it can go back to empty (restoring growth). Otherwise leave a tombstone.
  void erase_at(std::size_t idx) noexcept {
    std::size_t before = (idx - kGroup) & mask_;
    detail::BitMask empty_after = detail::Group(ctrl_ + idx).match_empty();
    detail::BitMask empty_before = detail::Group(ctrl_ + before).match_empty();
    bool never_full =
        empty_before && empty_after &&
        (empty_after.trailing_zeros() + empty_before.leading_zeros(kGroup) <
         kGroup);
    if (never_full) {
      set_ctrl(idx, detail::kEmpty);
      ++growth_left_;  // slot returns to empty -> one more insert before grow
    } else {
      set_ctrl(idx, detail::kDeleted);  // tombstone (reclaimed by next rehash)
    }
    --size_;
  }

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
    // tombstones (real size sparse), rehash SAME size to reclaim instead. The
    // alloc + reinsert loop itself lives in the type-erased core (one DLL copy).
    std::size_t new_cap = cap_ == 0          ? kMinCap
                          : size_ >= cap_ * 7 / 8 ? cap_ * 2
                                                  : cap_;
    detail::SwissStorage s{ctrl_, slots_, cap_, mask_, size_, growth_left_};
    detail::swiss_resize(s, new_cap, sizeof(Slot), alignof(Slot), alloc_,
                         &hash_slot_thunk, &hash_);
    ctrl_ = s.ctrl;
    slots_ = static_cast<Slot*>(s.slots);
    cap_ = s.cap;
    mask_ = s.mask;
    size_ = s.size;
    growth_left_ = s.growth_left;
  }

  void free_storage() noexcept {
    detail::SwissStorage s{ctrl_, slots_, cap_, mask_, size_, growth_left_};
    detail::swiss_free(s, sizeof(Slot), alignof(Slot), alloc_);
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
