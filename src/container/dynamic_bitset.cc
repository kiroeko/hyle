#include "hyle/container/dynamic_bitset.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <utility>

#include "hyle/platform/cpu.h"
#include "src/container/bitset_ops.h"

namespace hyle::container {
namespace {

using BinaryFn = void (*)(std::uint64_t*, const std::uint64_t*,
                          std::size_t) noexcept;
using CountFn = std::size_t (*)(const std::uint64_t*, std::size_t) noexcept;
using UnaryFn = void (*)(std::uint64_t*, std::size_t) noexcept;

struct BitsetOps {
  BinaryFn bit_and;
  BinaryFn bit_or;
  BinaryFn bit_xor;
  BinaryFn bit_and_not;
  UnaryFn bit_not;
  CountFn count;
};

BitsetOps resolve_ops() noexcept {
#if HYLE_ARCH_X64
  const auto isa = static_cast<std::uint32_t>(platform::best_isa());
#if defined(HYLE_CONTAINER_HAS_AVX512)
  if (isa >= static_cast<std::uint32_t>(platform::Isa::kAvx512))
    return {&detail::bitset_and_avx512, &detail::bitset_or_avx512,
            &detail::bitset_xor_avx512, &detail::bitset_and_not_avx512,
            &detail::bitset_not_avx512,
            &detail::bitset_count_avx512};
#endif
  if (isa >= static_cast<std::uint32_t>(platform::Isa::kAvx2))
    return {&detail::bitset_and_avx2, &detail::bitset_or_avx2,
            &detail::bitset_xor_avx2, &detail::bitset_and_not_avx2,
            &detail::bitset_not_avx2,
            &detail::bitset_count_avx2};
  if (isa >= static_cast<std::uint32_t>(platform::Isa::kSse42))
    return {&detail::bitset_and_sse4, &detail::bitset_or_sse4,
            &detail::bitset_xor_sse4, &detail::bitset_and_not_sse4,
            &detail::bitset_not_sse4,
            &detail::bitset_count_sse4};
#elif HYLE_ARCH_ARM64
  if (static_cast<std::uint32_t>(platform::best_isa()) >=
      static_cast<std::uint32_t>(platform::Isa::kNeon))
    return {&detail::bitset_and_neon, &detail::bitset_or_neon,
            &detail::bitset_xor_neon, &detail::bitset_and_not_neon,
            &detail::bitset_not_neon,
            &detail::bitset_count_neon};
#endif
  return {&detail::bitset_and_scalar, &detail::bitset_or_scalar,
          &detail::bitset_xor_scalar, &detail::bitset_and_not_scalar,
          &detail::bitset_not_scalar,
          &detail::bitset_count_scalar};
}

const BitsetOps& ops() noexcept {
  static const BitsetOps selected = resolve_ops();
  return selected;
}

void apply_binary(std::uint64_t* destination, const std::uint64_t* source,
                  std::size_t count, BinaryFn function) noexcept {
  // Avoid indirect-dispatch overhead on tiny masks; large graph frontiers use
  // the cached widest-ISA kernel.
  if (count < 8) {
    if (function == ops().bit_and) {
      for (std::size_t i = 0; i < count; ++i) destination[i] &= source[i];
    } else if (function == ops().bit_or) {
      for (std::size_t i = 0; i < count; ++i) destination[i] |= source[i];
    } else if (function == ops().bit_xor) {
      for (std::size_t i = 0; i < count; ++i) destination[i] ^= source[i];
    } else {
      for (std::size_t i = 0; i < count; ++i) destination[i] &= ~source[i];
    }
    return;
  }
  function(destination, source, count);
}

}  // namespace

DynamicBitSet::DynamicBitSet() noexcept : alloc_(mem::system_allocator()) {}

DynamicBitSet::DynamicBitSet(mem::AllocatorRef alloc) noexcept : alloc_(alloc) {}

DynamicBitSet::DynamicBitSet(size_type bit_count, bool value)
    : DynamicBitSet() {
  resize(bit_count, value);
}

DynamicBitSet::DynamicBitSet(mem::AllocatorRef alloc, size_type bit_count,
                             bool value)
    : DynamicBitSet(alloc) {
  resize(bit_count, value);
}

DynamicBitSet::DynamicBitSet(const DynamicBitSet& other)
    : DynamicBitSet(other.alloc_) {
  HYLE_FATAL_ASSERT(try_resize(other.size_bits_));
  if (block_count() != 0)
    std::memcpy(words_, other.words_, block_count() * sizeof(block_type));
}

DynamicBitSet& DynamicBitSet::operator=(const DynamicBitSet& other) {
  if (this == &other) return *this;
  DynamicBitSet replacement(alloc_);
  HYLE_FATAL_ASSERT(replacement.try_resize(other.size_bits_));
  if (other.block_count() != 0)
    std::memcpy(replacement.words_, other.words_,
                other.block_count() * sizeof(block_type));
  swap(replacement);
  return *this;
}

DynamicBitSet::DynamicBitSet(DynamicBitSet&& other) noexcept
    : alloc_(other.alloc_),
      words_(std::exchange(other.words_, nullptr)),
      size_bits_(std::exchange(other.size_bits_, 0)),
      capacity_words_(std::exchange(other.capacity_words_, 0)) {}

DynamicBitSet& DynamicBitSet::operator=(DynamicBitSet&& other) noexcept {
  if (this == &other) return *this;
  release();
  alloc_ = other.alloc_;
  words_ = std::exchange(other.words_, nullptr);
  size_bits_ = std::exchange(other.size_bits_, 0);
  capacity_words_ = std::exchange(other.capacity_words_, 0);
  return *this;
}

DynamicBitSet::~DynamicBitSet() { release(); }

bool DynamicBitSet::try_reserve(size_type bit_count) noexcept {
  const size_type required = words_for(bit_count);
  if (required <= capacity_words_) return true;
  if (!alloc_.valid() || required > static_cast<size_type>(-1) / sizeof(block_type))
    return false;
  size_type target = required;
  if (capacity_words_ != 0 &&
      capacity_words_ <= static_cast<size_type>(-1) / 2)
    target = std::max(required, capacity_words_ * 2);
  if (target > static_cast<size_type>(-1) / sizeof(block_type)) target = required;

  auto* replacement = alloc_.allocate_n<block_type>(target);
  if (replacement == nullptr) return false;
  std::memset(replacement, 0, target * sizeof(block_type));
  const size_type active = block_count();
  if (active != 0)
    std::memcpy(replacement, words_, active * sizeof(block_type));
  if (words_ != nullptr)
    alloc_.deallocate_n(words_, capacity_words_);
  words_ = replacement;
  capacity_words_ = target;
  return true;
}

bool DynamicBitSet::try_resize(size_type bit_count, bool value) noexcept {
  const size_type old_size = size_bits_;
  if (bit_count <= old_size) {
    size_bits_ = bit_count;
    sanitize_tail();
    return true;
  }
  if (!try_reserve(bit_count)) return false;

  const size_type old_words = words_for(old_size);
  const size_type new_words = words_for(bit_count);
  if (old_size % kBitsPerWord == 0) {
    std::memset(words_ + old_words, value ? 0xff : 0,
                (new_words - old_words) * sizeof(block_type));
  } else {
    const size_type first_word = old_size / kBitsPerWord;
    const block_type new_part = ~low_mask(old_size % kBitsPerWord);
    if (value)
      words_[first_word] |= new_part;
    else
      words_[first_word] &= ~new_part;
    if (new_words > first_word + 1)
      std::memset(words_ + first_word + 1, value ? 0xff : 0,
                  (new_words - first_word - 1) * sizeof(block_type));
  }
  size_bits_ = bit_count;
  sanitize_tail();
  return true;
}

DynamicBitSet& DynamicBitSet::set(size_type index, bool value) noexcept {
  HYLE_FATAL_ASSERT(index < size_bits_);
  if (value)
    words_[index / kBitsPerWord] |= bit_mask(index);
  else
    words_[index / kBitsPerWord] &= ~bit_mask(index);
  return *this;
}

DynamicBitSet& DynamicBitSet::flip(size_type index) noexcept {
  HYLE_FATAL_ASSERT(index < size_bits_);
  words_[index / kBitsPerWord] ^= bit_mask(index);
  return *this;
}

DynamicBitSet& DynamicBitSet::set() noexcept {
  if (block_count() != 0)
    std::memset(words_, 0xff, block_count() * sizeof(block_type));
  sanitize_tail();
  return *this;
}

DynamicBitSet& DynamicBitSet::reset() noexcept {
  if (block_count() != 0)
    std::memset(words_, 0, block_count() * sizeof(block_type));
  return *this;
}

DynamicBitSet& DynamicBitSet::flip() noexcept {
  const size_type count = block_count();
  if (count < 8) {
    for (size_type i = 0; i < count; ++i) words_[i] = ~words_[i];
  } else {
    ops().bit_not(words_, count);
  }
  sanitize_tail();
  return *this;
}

DynamicBitSet::size_type DynamicBitSet::count() const noexcept {
  const size_type words = block_count();
  if (words < 16) {
    size_type result = 0;
    for (size_type i = 0; i < words; ++i) result += std::popcount(words_[i]);
    return result;
  }
  return ops().count(words_, words);
}

bool DynamicBitSet::any() const noexcept {
  for (size_type i = 0; i < block_count(); ++i)
    if (words_[i] != 0) return true;
  return false;
}

bool DynamicBitSet::all() const noexcept {
  if (empty()) return true;
  const size_type words = block_count();
  for (size_type i = 0; i + 1 < words; ++i)
    if (words_[i] != ~block_type{0}) return false;
  return words_[words - 1] == low_mask(size_bits_ % kBitsPerWord);
}

DynamicBitSet::size_type DynamicBitSet::find_first() const noexcept {
  for (size_type i = 0; i < block_count(); ++i)
    if (words_[i] != 0)
      return i * kBitsPerWord + std::countr_zero(words_[i]);
  return npos;
}

DynamicBitSet::size_type DynamicBitSet::find_next(size_type index) const noexcept {
  if (index == npos || index + 1 >= size_bits_) return npos;
  const size_type next = index + 1;
  size_type word_index = next / kBitsPerWord;
  const size_type offset = next % kBitsPerWord;
  block_type word = words_[word_index] & (~block_type{0} << offset);
  if (word != 0)
    return word_index * kBitsPerWord + std::countr_zero(word);
  for (++word_index; word_index < block_count(); ++word_index)
    if (words_[word_index] != 0)
      return word_index * kBitsPerWord + std::countr_zero(words_[word_index]);
  return npos;
}

DynamicBitSet& DynamicBitSet::operator&=(const DynamicBitSet& other) noexcept {
  HYLE_FATAL_ASSERT(size_bits_ == other.size_bits_);
  apply_binary(words_, other.words_, block_count(), ops().bit_and);
  return *this;
}

DynamicBitSet& DynamicBitSet::operator|=(const DynamicBitSet& other) noexcept {
  HYLE_FATAL_ASSERT(size_bits_ == other.size_bits_);
  apply_binary(words_, other.words_, block_count(), ops().bit_or);
  return *this;
}

DynamicBitSet& DynamicBitSet::operator^=(const DynamicBitSet& other) noexcept {
  HYLE_FATAL_ASSERT(size_bits_ == other.size_bits_);
  apply_binary(words_, other.words_, block_count(), ops().bit_xor);
  return *this;
}

DynamicBitSet& DynamicBitSet::and_not(const DynamicBitSet& other) noexcept {
  HYLE_FATAL_ASSERT(size_bits_ == other.size_bits_);
  apply_binary(words_, other.words_, block_count(), ops().bit_and_not);
  return *this;
}

DynamicBitSet& DynamicBitSet::operator<<=(size_type shift) noexcept {
  if (shift == 0 || empty()) return *this;
  if (shift >= size_bits_) return reset();
  const size_type words = block_count();
  const size_type word_shift = shift / kBitsPerWord;
  const size_type bit_shift = shift % kBitsPerWord;
  for (size_type output = words; output-- > 0;) {
    block_type value = 0;
    if (output >= word_shift) {
      value = words_[output - word_shift] << bit_shift;
      if (bit_shift != 0 && output > word_shift)
        value |= words_[output - word_shift - 1] >>
                 (kBitsPerWord - bit_shift);
    }
    words_[output] = value;
  }
  sanitize_tail();
  return *this;
}

DynamicBitSet& DynamicBitSet::operator>>=(size_type shift) noexcept {
  if (shift == 0 || empty()) return *this;
  if (shift >= size_bits_) return reset();
  const size_type words = block_count();
  const size_type word_shift = shift / kBitsPerWord;
  const size_type bit_shift = shift % kBitsPerWord;
  for (size_type output = 0; output < words; ++output) {
    block_type value = 0;
    if (output + word_shift < words) {
      value = words_[output + word_shift] >> bit_shift;
      if (bit_shift != 0 && output + word_shift + 1 < words)
        value |= words_[output + word_shift + 1]
                 << (kBitsPerWord - bit_shift);
    }
    words_[output] = value;
  }
  sanitize_tail();
  return *this;
}

bool DynamicBitSet::intersects(const DynamicBitSet& other) const noexcept {
  HYLE_FATAL_ASSERT(size_bits_ == other.size_bits_);
  for (size_type i = 0; i < block_count(); ++i)
    if ((words_[i] & other.words_[i]) != 0) return true;
  return false;
}

bool DynamicBitSet::is_subset_of(const DynamicBitSet& other) const noexcept {
  HYLE_FATAL_ASSERT(size_bits_ == other.size_bits_);
  for (size_type i = 0; i < block_count(); ++i)
    if ((words_[i] & ~other.words_[i]) != 0) return false;
  return true;
}

void DynamicBitSet::swap(DynamicBitSet& other) noexcept {
  using std::swap;
  swap(alloc_, other.alloc_);
  swap(words_, other.words_);
  swap(size_bits_, other.size_bits_);
  swap(capacity_words_, other.capacity_words_);
}

void DynamicBitSet::sanitize_tail() noexcept {
  const size_type words = block_count();
  if (words != 0) words_[words - 1] &= low_mask(size_bits_ % kBitsPerWord);
}

void DynamicBitSet::release() noexcept {
  if (words_ != nullptr) alloc_.deallocate_n(words_, capacity_words_);
  words_ = nullptr;
  size_bits_ = 0;
  capacity_words_ = 0;
}

}  // namespace hyle::container
