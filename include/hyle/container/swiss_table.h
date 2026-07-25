#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <new>
#include <type_traits>
#include <utility>

#include "hyle/container/hash.h"
#include "hyle/container/probe.h"
#include "hyle/core/cache.h"
#include "hyle/macros.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/system_inline.h"

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
using AllocateFn = void* (*)(void*, std::size_t, std::size_t) noexcept;
using DeallocateFn = void (*)(void*, void*, std::size_t,
                              std::size_t) noexcept;
using RelocateFn = void (*)(void*, void*, std::size_t) noexcept;

HYLE_CONTAINER_API bool swiss_resize(
    SwissStorage& s, std::size_t new_cap, std::size_t slot_size,
    std::size_t slot_align, void* alloc_state, AllocateFn allocate,
    DeallocateFn deallocate,
    std::uint64_t (*hash_slot)(const void*, const void*),
    const void* hash_state, RelocateFn relocate_slot);

// Free a table's ctrl+slot storage (no-op when empty).
HYLE_CONTAINER_API void swiss_free(SwissStorage& s, std::size_t slot_size,
                                   std::size_t slot_align, void* alloc_state,
                                   DeallocateFn deallocate);
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
// K and V may be non-trivial. Rehash moves and destroys them through one cold
// type-erased callback; trivially copyable slots retain the memcpy fast path.
// Moves/destruction are required not to throw so a failed allocation leaves the
// original table intact and rehash cannot strand a partially moved table.
//
// The cold path (resize/free) is type-erased into swiss_table.cc (see detail::
// above); the hot path is inlined per instantiation for zero indirection.
template <class K, class V, class Hash = Hash<K>, class Eq = std::equal_to<>,
          class Alloc = mem::SystemAllocator>
class SwissTable {
  static_assert(std::is_nothrow_move_constructible_v<K> &&
                    std::is_nothrow_move_constructible_v<V>,
                "SwissTable requires nothrow move construction");
  static_assert(std::is_nothrow_destructible_v<K> &&
                    std::is_nothrow_destructible_v<V>,
                "SwissTable requires nothrow destruction");
  static_assert(std::is_nothrow_move_assignable_v<V>,
                "SwissTable requires nothrow value move assignment");
  static_assert(std::is_nothrow_invocable_r_v<std::uint64_t, Hash, const K&>,
                "SwissTable requires a noexcept hash function");
  static_assert(std::is_nothrow_invocable_r_v<bool, Eq, const K&, const K&>,
                "SwissTable requires a noexcept equality function");

  using ctrl_t = detail::ctrl_t;
  static constexpr std::size_t kGroup = detail::kGroupWidth;
  static constexpr std::size_t kMinCap = 16;  // >= both group widths, pow2
  static constexpr std::size_t kNoSlot = static_cast<std::size_t>(-1);

 public:
  // The iterator-visible entry keeps keys read-only without storing const K,
  // which would prevent noexcept move relocation during a flat-table rehash.
  // This follows the same invariant as mature flat maps: values are mutable,
  // keys can only change through erase+insert.
  class Entry {
    friend class SwissTable;

   public:
    const K& key() const noexcept { return key_; }
    V& value() noexcept { return value_; }
    const V& value() const noexcept { return value_; }

   private:
    Entry(K key, V value) noexcept
        : key_(std::move(key)), value_(std::move(value)) {}

    K key_;
    HYLE_NO_UNIQUE_ADDRESS V value_;
  };

  using value_type = Entry;

  template <bool IsConst>
  class BasicIterator {
    template <bool>
    friend class BasicIterator;
    friend class SwissTable;

    using Owner =
        std::conditional_t<IsConst, const SwissTable, SwissTable>;

   public:
    using iterator_concept = std::forward_iterator_tag;
    using iterator_category = std::forward_iterator_tag;
    using value_type = Entry;
    using difference_type = std::ptrdiff_t;
    using reference = std::conditional_t<IsConst, const Entry&, Entry&>;
    using pointer = std::conditional_t<IsConst, const Entry*, Entry*>;

    BasicIterator() noexcept = default;

    template <bool Enabled = IsConst>
      requires Enabled
    BasicIterator(const BasicIterator<false>& other) noexcept
        : owner_(other.owner_), index_(other.index_), group_(other.group_),
          remaining_(other.remaining_) {}

    reference operator*() const noexcept { return owner_->slots_[index_]; }
    pointer operator->() const noexcept { return owner_->slots_ + index_; }

    BasicIterator& operator++() noexcept {
      ++remaining_;
      if (remaining_) {
        index_ = group_ + remaining_.lowest();
      } else {
        group_ += kGroup;
        load_next_group();
      }
      return *this;
    }
    BasicIterator operator++(int) noexcept {
      BasicIterator copy = *this;
      ++*this;
      return copy;
    }

    friend bool operator==(const BasicIterator& left,
                           const BasicIterator& right) noexcept {
      return left.owner_ == right.owner_ && left.index_ == right.index_;
    }

   private:
    BasicIterator(Owner* owner, std::size_t index) noexcept
        : owner_(owner), index_(index), group_(index), remaining_(0) {
      if (owner_ != nullptr && index_ < owner_->cap_) load_next_group();
    }

    void load_next_group() noexcept {
      if (owner_ == nullptr) return;
      while (group_ < owner_->cap_) {
        remaining_ = detail::Group(owner_->ctrl_ + group_).match_full();
        if (remaining_) {
          index_ = group_ + remaining_.lowest();
          return;
        }
        group_ += kGroup;
      }
      index_ = owner_->cap_;
    }

    Owner* owner_ = nullptr;
    std::size_t index_ = 0;
    std::size_t group_ = 0;
    detail::BitMask remaining_{0};
  };

  using iterator = BasicIterator<false>;
  using const_iterator = BasicIterator<true>;

 private:
  using Slot = Entry;

 public:
  SwissTable() requires(std::is_default_constructible_v<Alloc> &&
                        std::is_default_constructible_v<Hash> &&
                        std::is_default_constructible_v<Eq>) = default;

  explicit SwissTable(Alloc alloc, Hash hash = Hash{}, Eq eq = Eq{})
      : alloc_(std::move(alloc)), hash_(std::move(hash)), eq_(std::move(eq)) {}

  ~SwissTable() { free_storage(); }
  HYLE_DISABLE_COPY(SwissTable);
  SwissTable(SwissTable&& o) noexcept(
      std::is_nothrow_move_constructible_v<Alloc> &&
      std::is_nothrow_move_constructible_v<Hash> &&
      std::is_nothrow_move_constructible_v<Eq>)
      : alloc_(std::move(o.alloc_)), hash_(std::move(o.hash_)),
        eq_(std::move(o.eq_)), ctrl_(o.ctrl_), slots_(o.slots_), cap_(o.cap_),
        mask_(o.mask_), size_(o.size_),
        growth_left_(o.growth_left_) {
    o.ctrl_ = nullptr;
    o.slots_ = nullptr;
    o.cap_ = 0;
    o.mask_ = 0;
    o.size_ = 0;
    o.growth_left_ = 0;
  }
  SwissTable& operator=(SwissTable&& o) noexcept(
      std::is_nothrow_move_assignable_v<Alloc> &&
      std::is_nothrow_move_assignable_v<Hash> &&
      std::is_nothrow_move_assignable_v<Eq>) {
    if (this == &o) return *this;
    free_storage();
    alloc_ = std::move(o.alloc_);
    hash_ = std::move(o.hash_);
    eq_ = std::move(o.eq_);
    ctrl_ = o.ctrl_;
    slots_ = o.slots_;
    cap_ = o.cap_;
    mask_ = o.mask_;
    size_ = o.size_;
    growth_left_ = o.growth_left_;
    o.ctrl_ = nullptr;
    o.slots_ = nullptr;
    o.cap_ = 0;
    o.mask_ = 0;
    o.size_ = 0;
    o.growth_left_ = 0;
    return *this;
  }

  std::size_t size() const noexcept { return size_; }
  std::size_t capacity() const noexcept { return cap_; }
  bool empty() const noexcept { return size_ == 0; }

  iterator begin() noexcept { return iterator(this, 0); }
  iterator end() noexcept { return iterator(this, cap_); }
  const_iterator begin() const noexcept { return const_iterator(this, 0); }
  const_iterator end() const noexcept { return const_iterator(this, cap_); }
  const_iterator cbegin() const noexcept { return begin(); }
  const_iterator cend() const noexcept { return end(); }

  bool reserve(std::size_t count) {
    if (count <= cap_ - cap_ / 8) return true;
    if (count > static_cast<std::size_t>(-1) / 8) return false;
    const std::size_t required = (count * 8 + 6) / 7;
    std::size_t new_cap = kMinCap;
    while (new_cap < required) {
      if (new_cap > static_cast<std::size_t>(-1) / 2) return false;
      new_cap *= 2;
    }
    detail::SwissStorage s{ctrl_, slots_, cap_, mask_, size_, growth_left_};
    if (!detail::swiss_resize(s, new_cap, sizeof(Slot), alignof(Slot), &alloc_,
                              &allocate_thunk, &deallocate_thunk,
                              &hash_slot_thunk, &hash_, &relocate_slot_thunk))
      return false;
    ctrl_ = s.ctrl;
    slots_ = static_cast<Slot*>(s.slots);
    cap_ = s.cap;
    mask_ = s.mask;
    size_ = s.size;
    growth_left_ = s.growth_left;
    return true;
  }

  // Ensure the next absent-key insert cannot allocate. Unlike reserve(size+1),
  // this also rehashes a table whose growth budget was consumed by tombstones.
  // Callers that promise noexcept/OOM recovery can perform this cold step first,
  // then use insert() while excluding concurrent mutation.
  bool needs_prepare_insert() const noexcept {
    return cap_ == 0 || growth_left_ == 0;
  }

  bool try_prepare_insert() {
    if (!needs_prepare_insert()) return true;
    return grow();
  }

  void clear() noexcept {
    if (ctrl_ == nullptr) return;
    destroy_live_slots();
    std::memset(ctrl_, static_cast<int>(detail::kEmpty), cap_ + kGroup);
    size_ = 0;
    growth_left_ = cap_ - cap_ / 8;
  }

  // Returns pointer to the value for `key`, or nullptr if absent. Compatible
  // transparent Hash/Eq objects permit allocation-free heterogeneous lookup
  // (for example std::string keys queried by string_view).
  template <class Q>
    requires(std::is_nothrow_invocable_r_v<std::uint64_t, const Hash&,
                                           const Q&> &&
             std::is_nothrow_invocable_r_v<bool, const Eq&, const K&,
                                           const Q&>)
  V* find(const Q& key) noexcept {
    Slot* slot = find_slot(key);
    return slot == nullptr ? nullptr : &slot->value_;
  }

  template <class Q>
    requires(std::is_nothrow_invocable_r_v<std::uint64_t, const Hash&,
                                           const Q&> &&
             std::is_nothrow_invocable_r_v<bool, const Eq&, const K&,
                                           const Q&>)
  const V* find(const Q& key) const noexcept {
    return const_cast<SwissTable*>(this)->template find<Q>(key);
  }

  // Key-preserving lookup for set adapters and callers that need the canonical
  // stored key (for example after heterogeneous lookup). The key remains const
  // because mutating it in place would invalidate its probe position.
  template <class Q>
    requires(std::is_nothrow_invocable_r_v<std::uint64_t, const Hash&,
                                           const Q&> &&
             std::is_nothrow_invocable_r_v<bool, const Eq&, const K&,
                                           const Q&>)
  const K* find_key(const Q& key) const noexcept {
    const Slot* slot =
        const_cast<SwissTable*>(this)->template find_slot<Q>(key);
    return slot == nullptr ? nullptr : &slot->key_;
  }

  template <class Q>
    requires(std::is_nothrow_invocable_r_v<std::uint64_t, const Hash&,
                                           const Q&> &&
             std::is_nothrow_invocable_r_v<bool, const Eq&, const K&,
                                           const Q&>)
  bool contains(const Q& key) const noexcept
  {
    return const_cast<SwissTable*>(this)->template find_slot<Q>(key) != nullptr;
  }

  // Insert or assign. Returns true if a new key was inserted, false if updated.
  // ONE hash + ONE probe walk in the common case (clay/abseil's fused
  // find_or_prepare): the walk both checks for an existing key and records the
  // first empty/deleted slot to insert into. A second walk happens only when a
  // grow is actually triggered (amortized rare).
  enum class InsertResult : std::uint8_t {
    kInserted,
    kUpdated,
    kOutOfMemory,
  };

  // By-value arguments deliberately preserve aliases into this table before a
  // possible rehash. Rvalues move once; lvalues pay the copy their insertion
  // already requires. The result distinguishes update from allocation failure.
  InsertResult try_insert_or_assign(K key, V value) {
    if (cap_ == 0) {
      if (!grow()) return InsertResult::kOutOfMemory;
    }
    std::uint64_t h = hash_(key);
    bool found = false;
    std::size_t idx = find_or_prepare(key, h, &found);
    if (found) {
      slots_[idx].value_ = std::move(value);
      return InsertResult::kUpdated;
    }
    // Not present; `idx` is the first empty/deleted slot on the probe path. Only
    // an EMPTY target consumes growth -- reusing a tombstone (deleted) does not
    // (it was already counted against the load factor when erase made it). Grow
    // only when we'd take an empty slot and none is budgeted.
    if (growth_left_ == 0 && ctrl_[idx] == detail::kEmpty) {
      if (!grow()) return InsertResult::kOutOfMemory;
      idx = find_insert_slot(h);  // cold path: rehash invalidated idx
      set_ctrl(idx, detail::h2(h));
      ::new (static_cast<void*>(slots_ + idx))
          Slot(std::move(key), std::move(value));
      ++size_;
      --growth_left_;  // find_insert_slot after a rehash always selects empty
      return InsertResult::kInserted;
    }
    bool was_empty = ctrl_[idx] == detail::kEmpty;
    set_ctrl(idx, detail::h2(h));
    ::new (static_cast<void*>(slots_ + idx))
        Slot(std::move(key), std::move(value));
    ++size_;
    if (was_empty) --growth_left_;
    return InsertResult::kInserted;
  }

  // Familiar convenience shape for callers that treat OOM as fatal. Returns
  // true for a newly inserted key and false when an existing value was updated.
  bool insert(K key, V value) {
    const InsertResult result =
        try_insert_or_assign(std::move(key), std::move(value));
    HYLE_FATAL_ASSERT(result != InsertResult::kOutOfMemory);
    return result == InsertResult::kInserted;
  }

  // Remove `key`. Returns true if it was present. The slot becomes empty when
  // its group still has an empty lane (the probe run never filled through it, so
  // no later key needs it as a stepping stone) -- which restores growth; else it
  // becomes a tombstone (find skips it, insert reuses it, grow drops it). This is
  // abseil's reclaim rule and keeps capacity tight under heavy churn.
  template <class Q>
    requires(std::is_nothrow_invocable_r_v<std::uint64_t, const Hash&,
                                           const Q&> &&
             std::is_nothrow_invocable_r_v<bool, const Eq&, const K&,
                                           const Q&>)
  bool erase(const Q& key) noexcept {
    if (cap_ == 0) return false;
    std::uint64_t h = hash_(key);
    std::size_t pos = detail::h1(h) & mask_;
    std::size_t stride = 0;
    ctrl_t want = detail::h2(h);
    for (;;) {
      detail::Group g(ctrl_ + pos);
      for (std::uint32_t i : g.match(want)) {
        std::size_t idx = (pos + i) & mask_;
        if (HYLE_LIKELY(eq_(slots_[idx].key_, key))) {
          slots_[idx].~Slot();
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
  // adds no allocation. This group-scan form remains the fastest full traversal;
  // begin/end provide the familiar iterator shape.
  // fn must not insert/erase/grow the table while iterating.
  template <class Fn>
  void for_each(Fn&& fn) {
    // Scan a GROUP (kGroup ctrl bytes) at a time: one SIMD match_full gives a
    // bitmask of occupied lanes, then iterate only the set bits -- branchless per
    // slot. The old byte-by-byte scan (load ctrl_[i] + branch per slot) lost absl's
    // iterator ~1.2x on iterate; this matches absl's group-scan shape. cap_ is a
    // pow2 >= kGroup, so groups tile [0,cap_) exactly (no partial tail; the ctrl
    // tail-mirror is never entered).
    if (cap_ == 0) return;
    for (std::size_t g = 0; g < cap_; g += kGroup) {
      for (std::uint32_t lane : detail::Group(ctrl_ + g).match_full()) {
        const std::size_t i = g + lane;
        fn(slots_[i].key_, slots_[i].value_);
      }
    }
  }
  template <class Fn>
  void for_each(Fn&& fn) const {
    if (cap_ == 0) return;
    for (std::size_t g = 0; g < cap_; g += kGroup) {
      for (std::uint32_t lane : detail::Group(ctrl_ + g).match_full()) {
        const std::size_t i = g + lane;
        fn(slots_[i].key_, slots_[i].value_);
      }
    }
  }

 private:
  template <class Q>
    requires(std::is_nothrow_invocable_r_v<std::uint64_t, const Hash&,
                                           const Q&> &&
             std::is_nothrow_invocable_r_v<bool, const Eq&, const K&,
                                           const Q&>)
  Slot* find_slot(const Q& key) noexcept {
    if (cap_ == 0) return nullptr;
    const std::uint64_t h = hash_(key);
    std::size_t pos = detail::h1(h) & mask_;
    std::size_t stride = 0;
    const ctrl_t want = detail::h2(h);
    for (;;) {
      core::prefetch_read(&slots_[pos]);
      detail::Group g(ctrl_ + pos);
      for (std::uint32_t i : g.match(want)) {
        const std::size_t idx = (pos + i) & mask_;
        if (HYLE_LIKELY(eq_(slots_[idx].key_, key))) return slots_ + idx;
      }
      if (g.match_empty()) return nullptr;
      stride += kGroup;
      pos = (pos + stride) & mask_;
    }
  }

  // Recompute a slot's key hash without exposing the concrete type to the cold
  // resize core.
  static std::uint64_t hash_slot_thunk(const void* slot,
                                       const void* hash_state) noexcept {
    return (*static_cast<const Hash*>(hash_state))(
        static_cast<const Slot*>(slot)->key_);
  }

  static void* allocate_thunk(void* state, std::size_t size,
                              std::size_t alignment) noexcept {
    return static_cast<Alloc*>(state)->allocate(size, alignment);
  }

  static void deallocate_thunk(void* state, void* ptr, std::size_t size,
                               std::size_t alignment) noexcept {
    static_cast<Alloc*>(state)->deallocate(ptr, size, alignment);
  }

  static void relocate_slot_thunk(void* destination, void* source,
                                  std::size_t) noexcept {
    auto* from = static_cast<Slot*>(source);
    if constexpr (std::is_trivially_copyable_v<Slot>) {
      std::memcpy(destination, source, sizeof(Slot));
    } else {
      ::new (destination)
          Slot(std::move(from->key_), std::move(from->value_));
      from->~Slot();
    }
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

  // Fused lookup + insert-slot search (one probe walk, one hash). Sets *found and
  // returns the matching slot if `key` is present; otherwise returns the FIRST
  // empty-or-deleted slot on the probe path (where the key would be inserted).
  // Correctness: the probe walk for `key` is deterministic, so the first
  // empty/deleted slot seen here is exactly where a later find() would stop --
  // inserting there keeps every probe chain intact. Requires cap_ > 0.
  std::size_t find_or_prepare(const K& key, std::uint64_t h,
                              bool* found) noexcept {
    std::size_t pos = detail::h1(h) & mask_;
    std::size_t stride = 0;
    ctrl_t want = detail::h2(h);
    std::size_t insert_at = kNoSlot;  // first empty/deleted seen (insert target)
    for (;;) {
      detail::Group g(ctrl_ + pos);
      for (std::uint32_t i : g.match(want)) {
        std::size_t idx = (pos + i) & mask_;
        if (HYLE_LIKELY(eq_(slots_[idx].key_, key))) {
          *found = true;
          return idx;
        }
      }
      // Remember the first reusable (empty/deleted) slot on the path, once.
      if (insert_at == kNoSlot) {
        if (detail::BitMask m = g.match_empty_or_deleted())
          insert_at = (pos + m.lowest()) & mask_;
      }
      if (g.match_empty()) {  // an empty lane ends the probe: key is absent
        *found = false;
        return insert_at;  // guaranteed set (an empty lane is empty-or-deleted)
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

  bool grow() {
    // Double only when genuinely full; if growth_left hit 0 mostly from
    // tombstones (real size sparse), rehash SAME size to reclaim instead. The
    // alloc + reinsert loop itself lives in the type-erased core (one DLL copy).
    HYLE_FATAL_ASSERT(cap_ == 0 || cap_ <= static_cast<std::size_t>(-1) / 2);
    std::size_t new_cap = cap_ == 0 ? kMinCap
                          : size_ >= cap_ - cap_ / 8 ? cap_ * 2
                                                    : cap_;
    detail::SwissStorage s{ctrl_, slots_, cap_, mask_, size_, growth_left_};
    if (!detail::swiss_resize(s, new_cap, sizeof(Slot), alignof(Slot), &alloc_,
                              &allocate_thunk, &deallocate_thunk,
                              &hash_slot_thunk, &hash_, &relocate_slot_thunk))
      return false;
    ctrl_ = s.ctrl;
    slots_ = static_cast<Slot*>(s.slots);
    cap_ = s.cap;
    mask_ = s.mask;
    size_ = s.size;
    growth_left_ = s.growth_left;
    return true;
  }

  void free_storage() noexcept {
    destroy_live_slots();
    detail::SwissStorage s{ctrl_, slots_, cap_, mask_, size_, growth_left_};
    detail::swiss_free(s, sizeof(Slot), alignof(Slot), &alloc_,
                       &deallocate_thunk);
    ctrl_ = nullptr;
    slots_ = nullptr;
    cap_ = 0;
    mask_ = 0;
    size_ = 0;
    growth_left_ = 0;
  }

  void destroy_live_slots() noexcept {
    if constexpr (!std::is_trivially_destructible_v<Slot>) {
      for (std::size_t i = 0; i < cap_; ++i) {
        if (detail::ctrl_is_full(ctrl_[i])) slots_[i].~Slot();
      }
    }
  }

  HYLE_NO_UNIQUE_ADDRESS Alloc alloc_{};
  HYLE_NO_UNIQUE_ADDRESS Hash hash_{};
  HYLE_NO_UNIQUE_ADDRESS Eq eq_{};
  ctrl_t* ctrl_ = nullptr;   // cap_ + kGroup bytes (tail mirrors first group)
  Slot* slots_ = nullptr;    // cap_ slots
  std::size_t cap_ = 0;      // power of two (or 0 when empty)
  std::size_t mask_ = 0;     // cap_ - 1
  std::size_t size_ = 0;     // full slots
  std::size_t growth_left_ = 0;  // inserts until next grow (7/8 load)
};

// Preferred public name. SwissTable remains as the implementation-specific
// spelling for existing Hyle users and documents the probing strategy.
template <class K, class V, class Hash = Hash<K>, class Eq = std::equal_to<>,
          class Alloc = mem::SystemAllocator>
using FlatHashMap = SwissTable<K, V, Hash, Eq, Alloc>;

namespace detail {
struct SwissSetValue {};
}  // namespace detail

// Space-efficient Swiss set sharing FlatHashMap's probing/resize core. The
// empty mapped policy occupies no slot bytes; this is a set API, not a public
// map<Key, Unit> spelling.
template <class K, class Hash = Hash<K>, class Eq = std::equal_to<>,
          class Alloc = mem::SystemAllocator>
class FlatHashSet {
  using Table = SwissTable<K, detail::SwissSetValue, Hash, Eq, Alloc>;
  static_assert(sizeof(typename Table::value_type) == sizeof(K),
                "FlatHashSet must not pay mapped-value storage");

 public:
  enum class InsertResult : std::uint8_t {
    kInserted,
    kAlreadyPresent,
    kOutOfMemory,
  };

  class iterator {
    friend class FlatHashSet;
    using MapIterator = typename Table::const_iterator;

   public:
    using iterator_concept = std::forward_iterator_tag;
    using iterator_category = std::forward_iterator_tag;
    using value_type = K;
    using difference_type = std::ptrdiff_t;
    using reference = const K&;
    using pointer = const K*;

    iterator() noexcept = default;
    reference operator*() const noexcept { return current_->key(); }
    pointer operator->() const noexcept { return &current_->key(); }
    iterator& operator++() noexcept {
      ++current_;
      return *this;
    }
    iterator operator++(int) noexcept {
      iterator copy = *this;
      ++*this;
      return copy;
    }
    friend bool operator==(const iterator& left,
                           const iterator& right) noexcept {
      return left.current_ == right.current_;
    }

   private:
    explicit iterator(MapIterator current) noexcept : current_(current) {}
    MapIterator current_{};
  };

  using const_iterator = iterator;

  FlatHashSet() = default;
  explicit FlatHashSet(Alloc alloc, Hash hash = Hash{}, Eq eq = Eq{})
      : table_(std::move(alloc), std::move(hash), std::move(eq)) {}

  HYLE_DISABLE_COPY(FlatHashSet);
  FlatHashSet(FlatHashSet&&) = default;
  FlatHashSet& operator=(FlatHashSet&&) = default;

  std::size_t size() const noexcept { return table_.size(); }
  std::size_t capacity() const noexcept { return table_.capacity(); }
  bool empty() const noexcept { return table_.empty(); }
  bool reserve(std::size_t count) { return table_.reserve(count); }
  bool try_prepare_insert() { return table_.try_prepare_insert(); }
  void clear() noexcept { table_.clear(); }

  template <class Q>
    requires(std::is_nothrow_invocable_r_v<std::uint64_t, const Hash&,
                                           const Q&> &&
             std::is_nothrow_invocable_r_v<bool, const Eq&, const K&,
                                           const Q&>)
  const K* find(const Q& key) const noexcept {
    return table_.find_key(key);
  }

  template <class Q>
    requires(std::is_nothrow_invocable_r_v<std::uint64_t, const Hash&,
                                           const Q&> &&
             std::is_nothrow_invocable_r_v<bool, const Eq&, const K&,
                                           const Q&>)
  bool contains(const Q& key) const noexcept {
    return table_.contains(key);
  }

  InsertResult try_insert(K key) {
    using MapResult = typename Table::InsertResult;
    const MapResult result = table_.try_insert_or_assign(
        std::move(key), detail::SwissSetValue{});
    if (result == MapResult::kInserted) return InsertResult::kInserted;
    if (result == MapResult::kUpdated) return InsertResult::kAlreadyPresent;
    return InsertResult::kOutOfMemory;
  }

  bool insert(K key) {
    return table_.insert(std::move(key), detail::SwissSetValue{});
  }

  template <class Q>
    requires(std::is_nothrow_invocable_r_v<std::uint64_t, const Hash&,
                                           const Q&> &&
             std::is_nothrow_invocable_r_v<bool, const Eq&, const K&,
                                           const Q&>)
  bool erase(const Q& key) noexcept {
    return table_.erase(key);
  }

  iterator begin() noexcept { return iterator(table_.cbegin()); }
  iterator end() noexcept { return iterator(table_.cend()); }
  const_iterator begin() const noexcept { return iterator(table_.cbegin()); }
  const_iterator end() const noexcept { return iterator(table_.cend()); }
  const_iterator cbegin() const noexcept { return begin(); }
  const_iterator cend() const noexcept { return end(); }

  template <class Fn>
  void for_each(Fn&& fn) const {
    table_.for_each(
        [&](const K& key, const detail::SwissSetValue&) { fn(key); });
  }

 private:
  Table table_;
};

}  // namespace hyle::container
