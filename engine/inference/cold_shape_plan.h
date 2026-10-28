#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace hyle::inference {

// Model-opaque cold usage and bounded extra-shape policy. Graphs, carriers,
// payloads and retirement authority stay with the original caller. A queued
// key records only which successful cold admission may next be evicted.
template<class Shape, class Key, std::size_t Slots>
class ColdShapePlan final {
  static_assert(Slots > 0);
  struct Entry final { Key key; Shape shape; };
  // Once the native owner has been removed, committing its metadata must not
  // throw. Copies and map-node allocation happen before either cold hook.
  static_assert(std::is_nothrow_move_constructible_v<Entry>);
  static_assert(std::is_nothrow_move_assignable_v<Entry>);
  static_assert(noexcept(std::declval<const Shape &>() < std::declval<const Shape &>()));

 public:
  using Uses = std::array<std::size_t, Slots>;
  using Entries = std::map<Shape, Uses>;

  const Entries &entries() const noexcept { return uses_; }
  const Uses &uses(const Shape &shape) const { return uses_.at(shape); }
  bool contains(const Shape &shape) const { return uses_.contains(shape); }
  std::size_t size() const noexcept { return uses_.size(); }
  std::size_t capacity() const noexcept { return capacity_; }
  std::size_t retained() const noexcept { return admitted_.size(); }
  std::size_t admissions() const noexcept { return admissions_; }
  std::size_t evictions() const noexcept { return evictions_; }

  // The original caller still checks that no execution carriers have been
  // prepared. freeze() fixes only extra-shape capacity; it does not invent a
  // second admission gate for cold usage updates allowed by that caller.
  void add_use(const Shape &shape, std::size_t slot, std::size_t count = 1) {
    if (slot >= Slots) throw std::out_of_range("execution use slot exceeds its cold plan");
    const auto found = uses_.find(shape);
    auto next = found == uses_.end() ? Uses{} : found->second;
    next[slot] = add(next[slot], count);
    uses_.insert_or_assign(shape, next);
  }
  void add_uses(const Shape &shape, std::size_t count) {
    const auto found = uses_.find(shape);
    auto next = found == uses_.end() ? Uses{} : found->second;
    for (auto &value : next) value = add(value, count);
    uses_.insert_or_assign(shape, next);
  }

  // Return true only for the original freeze, so its caller can retain the
  // existing log/timing endpoint. Later use counts never resize this bound.
  bool freeze() {
    if (capacity_) return false;
    if (!admitted_.empty())
      throw std::logic_error("freeze the original shape plan before dynamic admission");
    capacity_ = (std::max)(std::size_t{1}, uses_.size());
    return true;
  }

  // retire_erase(key, shape) must prove actual retirement before removing the
  // original native family. If it throws, that family and this FIFO remain
  // retained. build() publishes the caller's complete cold family; rollback()
  // noexcept removes only that new family after a build/publication failure.
  // No hook is stored, and this algorithm is never used on a hot lookup.
  template<class RetireErase, class Build, class Rollback>
  void admit(const Shape &shape, const Key &key, RetireErase &&retire_erase,
      Build &&build, Rollback &&rollback) {
    static_assert(noexcept(rollback()));
    if (!capacity_) throw std::logic_error("freeze the cold shape plan before admission");
    for (const auto &entry : admitted_)
      if (entry.key == key || entry.shape == shape)
        throw std::logic_error("cold shape admission repeats a retained key or shape");

    Entry candidate{key, shape};
    if (admitted_.capacity() < capacity_) admitted_.reserve(capacity_);
    // Allocate an unpublished node even when the same shape has an existing
    // use plan. That existing count stays untouched if build() fails.
    Entries staged;
    Uses initial; initial.fill(1);
    auto node = staged.extract(staged.emplace(candidate.shape, initial).first);
    if (admitted_.size() == capacity_) {
      const auto &victim = admitted_.front();
      retire_erase(victim.key, victim.shape);
      uses_.erase(victim.shape);
      admitted_.erase(admitted_.begin());
      ++evictions_;
    }
    try {
      build();
      auto inserted = uses_.insert(std::move(node));
      if (!inserted.inserted) inserted.position->second = initial;
      admitted_.push_back(std::move(candidate));
      ++admissions_;
    } catch (...) {
      rollback();
      throw;
    }
  }

 private:
  static std::size_t add(std::size_t a, std::size_t b) {
    if (b > (std::numeric_limits<std::size_t>::max)() - a)
      throw std::runtime_error("execution use count overflows");
    return a + b;
  }
  Entries uses_;
  std::vector<Entry> admitted_;
  std::size_t capacity_ = 0, admissions_ = 0, evictions_ = 0;
};

} // namespace hyle::inference
