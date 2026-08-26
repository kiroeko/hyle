#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>

#include "hyle/container/hash.h"  // HYLE_CONTAINER_API
#include "hyle/core/compiler.h"
#include "hyle/mem/allocator.h"

namespace hyle::container {

// Allocator-aware, runtime-sized bit set for graph frontiers, masks and dense
// integer membership. Padding bits are always zero, so block-wise set algebra,
// count() and set-bit iteration share one canonical representation.
class DynamicBitSet {
 public:
  using block_type = std::uint64_t;
  using size_type = std::size_t;
  static constexpr size_type npos = static_cast<size_type>(-1);

  HYLE_CONTAINER_API DynamicBitSet() noexcept;
  HYLE_CONTAINER_API explicit DynamicBitSet(mem::Allocator alloc) noexcept;
  HYLE_CONTAINER_API explicit DynamicBitSet(size_type bit_count,
                                            bool value = false);
  HYLE_CONTAINER_API DynamicBitSet(mem::Allocator alloc, size_type bit_count,
                                   bool value = false);
  HYLE_CONTAINER_API DynamicBitSet(const DynamicBitSet& other);
  HYLE_CONTAINER_API DynamicBitSet& operator=(const DynamicBitSet& other);
  HYLE_CONTAINER_API DynamicBitSet(DynamicBitSet&& other) noexcept;
  HYLE_CONTAINER_API DynamicBitSet& operator=(DynamicBitSet&& other) noexcept;
  HYLE_CONTAINER_API ~DynamicBitSet();

  mem::Allocator allocator() const noexcept { return alloc_; }
  size_type size() const noexcept { return size_bits_; }
  size_type capacity() const noexcept { return capacity_words_ * kBitsPerWord; }
  bool empty() const noexcept { return size_bits_ == 0; }
  size_type block_count() const noexcept { return words_for(size_bits_); }
  const block_type* blocks() const noexcept { return words_; }

  HYLE_CONTAINER_API bool try_reserve(size_type bit_count) noexcept;
  void reserve(size_type bit_count) { HYLE_FATAL_ASSERT(try_reserve(bit_count)); }
  HYLE_CONTAINER_API bool try_resize(size_type bit_count,
                                     bool value = false) noexcept;
  void resize(size_type bit_count, bool value = false) {
    HYLE_FATAL_ASSERT(try_resize(bit_count, value));
  }
  bool try_push_back(bool value) noexcept {
    return size_bits_ != npos && try_resize(size_bits_ + 1, value);
  }
  void push_back(bool value) { HYLE_FATAL_ASSERT(try_push_back(value)); }
  void pop_back() noexcept {
    HYLE_FATAL_ASSERT(size_bits_ != 0);
    resize(size_bits_ - 1);
  }
  void clear() noexcept { size_bits_ = 0; }

  bool operator[](size_type index) const noexcept {
    return (words_[index / kBitsPerWord] & bit_mask(index)) != 0;
  }
  bool test(size_type index) const noexcept {
    HYLE_FATAL_ASSERT(index < size_bits_);
    return (*this)[index];
  }
  HYLE_CONTAINER_API DynamicBitSet& set(size_type index,
                                        bool value = true) noexcept;
  DynamicBitSet& reset(size_type index) noexcept { return set(index, false); }
  HYLE_CONTAINER_API DynamicBitSet& flip(size_type index) noexcept;
  HYLE_CONTAINER_API DynamicBitSet& set() noexcept;
  HYLE_CONTAINER_API DynamicBitSet& reset() noexcept;
  HYLE_CONTAINER_API DynamicBitSet& flip() noexcept;

  HYLE_CONTAINER_API size_type count() const noexcept;
  HYLE_CONTAINER_API bool any() const noexcept;
  bool none() const noexcept { return !any(); }
  HYLE_CONTAINER_API bool all() const noexcept;
  HYLE_CONTAINER_API size_type find_first() const noexcept;
  HYLE_CONTAINER_API size_type find_next(size_type index) const noexcept;

  HYLE_CONTAINER_API DynamicBitSet&
  operator&=(const DynamicBitSet& other) noexcept;
  HYLE_CONTAINER_API DynamicBitSet&
  operator|=(const DynamicBitSet& other) noexcept;
  HYLE_CONTAINER_API DynamicBitSet&
  operator^=(const DynamicBitSet& other) noexcept;
  HYLE_CONTAINER_API DynamicBitSet&
  and_not(const DynamicBitSet& other) noexcept;
  HYLE_CONTAINER_API DynamicBitSet& operator<<=(size_type shift) noexcept;
  HYLE_CONTAINER_API DynamicBitSet& operator>>=(size_type shift) noexcept;
  HYLE_CONTAINER_API bool
  intersects(const DynamicBitSet& other) const noexcept;
  HYLE_CONTAINER_API bool
  is_subset_of(const DynamicBitSet& other) const noexcept;

  HYLE_CONTAINER_API void swap(DynamicBitSet& other) noexcept;

  class SetBitIterator {
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = size_type;
    using difference_type = std::ptrdiff_t;
    using reference = size_type;
    using pointer = void;

    size_type operator*() const noexcept { return position_; }
    SetBitIterator& operator++() noexcept {
      position_ = owner_->find_next(position_);
      return *this;
    }
    SetBitIterator operator++(int) noexcept {
      SetBitIterator copy = *this;
      ++*this;
      return copy;
    }
    friend bool operator==(SetBitIterator left, SetBitIterator right) noexcept {
      return left.owner_ == right.owner_ && left.position_ == right.position_;
    }

   private:
    friend class DynamicBitSet;
    SetBitIterator(const DynamicBitSet* owner, size_type position) noexcept
        : owner_(owner), position_(position) {}
    const DynamicBitSet* owner_ = nullptr;
    size_type position_ = npos;
  };

  class SetBitsView {
   public:
    SetBitIterator begin() const noexcept {
      return SetBitIterator(owner_, owner_->find_first());
    }
    SetBitIterator end() const noexcept { return SetBitIterator(owner_, npos); }

   private:
    friend class DynamicBitSet;
    explicit SetBitsView(const DynamicBitSet* owner) noexcept : owner_(owner) {}
    const DynamicBitSet* owner_;
  };

  SetBitsView set_bits() const noexcept { return SetBitsView(this); }

 private:
  static constexpr size_type kBitsPerWord = 64;
  static constexpr size_type words_for(size_type bits) noexcept {
    return bits / kBitsPerWord + (bits % kBitsPerWord != 0);
  }
  static constexpr block_type bit_mask(size_type index) noexcept {
    return block_type{1} << (index % kBitsPerWord);
  }
  static constexpr block_type low_mask(size_type bits) noexcept {
    return bits == 0 ? ~block_type{0} : (block_type{1} << bits) - 1;
  }
  void sanitize_tail() noexcept;
  void release() noexcept;

  mem::Allocator alloc_;
  block_type* words_ = nullptr;
  size_type size_bits_ = 0;
  size_type capacity_words_ = 0;
};

inline void swap(DynamicBitSet& left, DynamicBitSet& right) noexcept {
  left.swap(right);
}

inline DynamicBitSet operator&(DynamicBitSet left,
                               const DynamicBitSet& right) {
  left &= right;
  return left;
}
inline DynamicBitSet operator|(DynamicBitSet left,
                               const DynamicBitSet& right) {
  left |= right;
  return left;
}
inline DynamicBitSet operator^(DynamicBitSet left,
                               const DynamicBitSet& right) {
  left ^= right;
  return left;
}
inline DynamicBitSet operator<<(DynamicBitSet left,
                                DynamicBitSet::size_type shift) {
  left <<= shift;
  return left;
}
inline DynamicBitSet operator>>(DynamicBitSet left,
                                DynamicBitSet::size_type shift) {
  left >>= shift;
  return left;
}

}  // namespace hyle::container
