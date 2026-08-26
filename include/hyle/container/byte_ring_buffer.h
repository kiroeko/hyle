#pragma once

#include <bit>
#include <cstddef>
#include <cstring>
#include <span>
#include <utility>

#include "hyle/core/compiler.h"
#include "hyle/mem/allocator.h"

namespace hyle::container {

// Byte-oriented single-thread ring for stream/network staging. readable() and
// writable() expose at most two contiguous spans, so scatter/gather I/O can
// avoid an intermediate copy. Call commit_write()/consume() only with counts
// bounded by the spans returned immediately before them.
class ByteRingBuffer {
public:
  using size_type = std::size_t;
  struct MutableSegments {
    std::span<std::byte> first;
    std::span<std::byte> second;
  };
  struct ConstSegments {
    std::span<const std::byte> first;
    std::span<const std::byte> second;
  };

  ByteRingBuffer() noexcept : alloc_(mem::system_allocator()) {}
  explicit ByteRingBuffer(mem::Allocator alloc) noexcept : alloc_(alloc) {}
  explicit ByteRingBuffer(size_type capacity) : ByteRingBuffer() {
    reserve(capacity);
  }
  ByteRingBuffer(mem::Allocator alloc, size_type capacity)
      : ByteRingBuffer(alloc) {
    reserve(capacity);
  }
  ByteRingBuffer(const ByteRingBuffer &other) : ByteRingBuffer(other.alloc_) {
    HYLE_FATAL_ASSERT(try_reserve(other.size_));
    const ConstSegments source = other.readable();
    write(source.first);
    write(source.second);
  }
  ByteRingBuffer &operator=(const ByteRingBuffer &other) {
    if (this == &other)
      return *this;
    ByteRingBuffer replacement(alloc_);
    HYLE_FATAL_ASSERT(replacement.try_reserve(other.size_));
    const ConstSegments source = other.readable();
    replacement.write(source.first);
    replacement.write(source.second);
    swap(replacement);
    return *this;
  }
  ByteRingBuffer(ByteRingBuffer &&other) noexcept
      : alloc_(other.alloc_), storage_(std::exchange(other.storage_, nullptr)),
        capacity_(std::exchange(other.capacity_, 0)),
        head_(std::exchange(other.head_, 0)),
        size_(std::exchange(other.size_, 0)) {}
  ByteRingBuffer &operator=(ByteRingBuffer &&other) noexcept {
    if (this == &other)
      return *this;
    release();
    alloc_ = other.alloc_;
    storage_ = std::exchange(other.storage_, nullptr);
    capacity_ = std::exchange(other.capacity_, 0);
    head_ = std::exchange(other.head_, 0);
    size_ = std::exchange(other.size_, 0);
    return *this;
  }
  ~ByteRingBuffer() { release(); }

  mem::Allocator allocator() const noexcept { return alloc_; }
  size_type size() const noexcept { return size_; }
  size_type capacity() const noexcept { return capacity_; }
  size_type free_space() const noexcept { return capacity_ - size_; }
  bool empty() const noexcept { return size_ == 0; }
  bool full() const noexcept { return size_ == capacity_; }

  bool try_reserve(size_type requested) noexcept {
    if (requested <= capacity_)
      return true;
    constexpr size_type kMaximum = std::bit_floor(static_cast<size_type>(-1));
    if (!alloc_.valid() || requested > kMaximum)
      return false;
    const size_type target = std::bit_ceil(requested);
    std::byte *replacement = alloc_.allocate_n<std::byte>(target);
    if (replacement == nullptr)
      return false;
    const ConstSegments source = readable();
    if (!source.first.empty())
      std::memcpy(replacement, source.first.data(), source.first.size());
    if (!source.second.empty())
      std::memcpy(replacement + source.first.size(), source.second.data(),
                  source.second.size());
    if (storage_ != nullptr)
      alloc_.deallocate_n(storage_, capacity_);
    storage_ = replacement;
    capacity_ = target;
    head_ = 0;
    return true;
  }
  void reserve(size_type requested) {
    HYLE_FATAL_ASSERT(try_reserve(requested));
  }

  ConstSegments readable() const noexcept {
    if (size_ == 0)
      return {};
    const size_type first_size =
        size_ < capacity_ - head_ ? size_ : capacity_ - head_;
    return {{storage_ + head_, first_size}, {storage_, size_ - first_size}};
  }
  MutableSegments writable() noexcept {
    const size_type available = free_space();
    if (available == 0)
      return {};
    const size_type tail = (head_ + size_) & (capacity_ - 1);
    const size_type first_size =
        available < capacity_ - tail ? available : capacity_ - tail;
    return {{storage_ + tail, first_size}, {storage_, available - first_size}};
  }

  size_type write(std::span<const std::byte> input) noexcept {
    const size_type amount =
        input.size() < free_space() ? input.size() : free_space();
    if (amount == 0)
      return 0;
    const MutableSegments destination = writable();
    const size_type first_size =
        amount < destination.first.size() ? amount : destination.first.size();
    std::memcpy(destination.first.data(), input.data(), first_size);
    if (amount > first_size)
      std::memcpy(destination.second.data(), input.data() + first_size,
                  amount - first_size);
    commit_write(amount);
    return amount;
  }
  size_type read(std::span<std::byte> output) noexcept {
    const size_type amount = output.size() < size_ ? output.size() : size_;
    if (amount == 0)
      return 0;
    const ConstSegments source = readable();
    const size_type first_size =
        amount < source.first.size() ? amount : source.first.size();
    std::memcpy(output.data(), source.first.data(), first_size);
    if (amount > first_size)
      std::memcpy(output.data() + first_size, source.second.data(),
                  amount - first_size);
    consume(amount);
    return amount;
  }
  void commit_write(size_type count) noexcept {
    HYLE_FATAL_ASSERT(count <= free_space());
    size_ += count;
  }
  void consume(size_type count) noexcept {
    HYLE_FATAL_ASSERT(count <= size_);
    if (count != 0)
      head_ = (head_ + count) & (capacity_ - 1);
    size_ -= count;
    if (size_ == 0)
      head_ = 0;
  }
  void clear() noexcept {
    head_ = 0;
    size_ = 0;
  }

  void swap(ByteRingBuffer &other) noexcept {
    using std::swap;
    swap(alloc_, other.alloc_);
    swap(storage_, other.storage_);
    swap(capacity_, other.capacity_);
    swap(head_, other.head_);
    swap(size_, other.size_);
  }

private:
  void release() noexcept {
    if (storage_ != nullptr)
      alloc_.deallocate_n(storage_, capacity_);
    storage_ = nullptr;
    capacity_ = 0;
    head_ = 0;
    size_ = 0;
  }

  mem::Allocator alloc_;
  std::byte *storage_ = nullptr;
  size_type capacity_ = 0;
  size_type head_ = 0;
  size_type size_ = 0;
};

inline void swap(ByteRingBuffer &left, ByteRingBuffer &right) noexcept {
  left.swap(right);
}

} // namespace hyle::container
