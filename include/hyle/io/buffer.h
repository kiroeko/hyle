#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <type_traits>

#include "hyle/core/environment.h"
#include "hyle/error/status.h"

namespace hyle::io {

// Portable buffer-sequence limits. 64 matches the mature Asio native adapter
// and stays below every supported POSIX IOV_MAX; the byte ceiling avoids the
// signed-size and macOS 2 GiB edge while remaining representable by Windows
// completion APIs. Operations reject larger sequences instead of truncating.
inline constexpr std::size_t kMaxBufferSequenceSegments = 64;
inline constexpr std::size_t kMaxBufferSequenceBytes = 0x7fff'ffffu;

class MutableBuffer {
 public:
  constexpr MutableBuffer() noexcept = default;
  constexpr MutableBuffer(void* data, std::size_t size) noexcept
      : data_(data), size_(size) {}

  template <class T, std::size_t Extent>
    requires(!std::is_const_v<T>)
  constexpr MutableBuffer(std::span<T, Extent> bytes) noexcept
      : data_(bytes.data()), size_(bytes.size_bytes()) {}

  constexpr void* data() const noexcept { return data_; }
  constexpr std::size_t size() const noexcept { return size_; }
  constexpr bool empty() const noexcept { return size_ == 0; }

 private:
  void* data_ = nullptr;
  std::size_t size_ = 0;
};

class ConstBuffer {
 public:
  constexpr ConstBuffer() noexcept = default;
  constexpr ConstBuffer(const void* data, std::size_t size) noexcept
      : data_(data), size_(size) {}
  constexpr ConstBuffer(MutableBuffer bytes) noexcept
      : data_(bytes.data()), size_(bytes.size()) {}

  template <class T, std::size_t Extent>
  constexpr ConstBuffer(std::span<T, Extent> bytes) noexcept
      : data_(bytes.data()), size_(bytes.size_bytes()) {}

  constexpr const void* data() const noexcept { return data_; }
  constexpr std::size_t size() const noexcept { return size_; }
  constexpr bool empty() const noexcept { return size_ == 0; }

 private:
  const void* data_ = nullptr;
  std::size_t size_ = 0;
};

template <class T, std::size_t Extent>
  requires(!std::is_const_v<T>)
constexpr MutableBuffer mutable_buffer(std::span<T, Extent> bytes) noexcept {
  return MutableBuffer(bytes);
}

template <class T, std::size_t Extent>
constexpr ConstBuffer const_buffer(std::span<T, Extent> bytes) noexcept {
  return ConstBuffer(bytes);
}

// Position in the caller's logical buffer sequence. Passing a previous result's
// `next` cursor resumes without rebuilding subspans or rescanning transferred
// bytes. Empty segments and an offset exactly at segment end are normalized.
struct BufferSequenceCursor {
  std::size_t segment = 0;
  std::size_t offset = 0;
};

struct BufferSequenceProgress {
  std::size_t bytes = 0;
  BufferSequenceCursor next{};
  bool complete = false;
};

struct [[nodiscard]] BufferSequenceResult {
  BufferSequenceProgress progress{};
  Status status{};
  bool eof = false;

  bool ok() const noexcept { return status.ok(); }
  explicit operator bool() const noexcept { return ok(); }
};

namespace detail {

// Deliberately matches WSABUF on Windows and iovec on POSIX. The backend source
// static-asserts size/alignment/member offsets against the native type before
// reinterpretation. This lets an awaitable own one descriptor array with no
// conversion allocation, while the public MutableBuffer/ConstBuffer shape stays
// ordinary and const-correct.
struct NativeBuffer {
#if HYLE_OS_WINDOWS
  std::uint32_t length = 0;
  void* address = nullptr;
#else
  void* address = nullptr;
  std::size_t length = 0;
#endif

  constexpr NativeBuffer() noexcept = default;
  constexpr NativeBuffer(void* data, std::size_t size) noexcept
#if HYLE_OS_WINDOWS
      : length(static_cast<std::uint32_t>(size)), address(data) {}
#else
      : address(data), length(size) {}
#endif

  constexpr void* data() const noexcept { return address; }
  constexpr std::size_t size() const noexcept {
    return static_cast<std::size_t>(length);
  }
};

template <class Buffer>
class BufferSequenceStorage {
 public:
  BufferSequenceStorage() noexcept = default;

  BufferSequenceStorage(std::span<const Buffer> sequence,
                        BufferSequenceCursor cursor = {}) noexcept {
    prepare(sequence, cursor);
  }

  void prepare(std::span<const Buffer> sequence,
               BufferSequenceCursor cursor = {}) noexcept {
    status_ = Status{};
    logical_count_ = sequence.size();
    native_count_ = 0;
    total_size_ = 0;
    start_ = cursor;

    if (sequence.size() > kMaxBufferSequenceSegments) {
      status_ = Status::OutOfRange(
          "buffer sequence exceeds the portable 64-segment limit");
      return;
    }
    if (cursor.segment > sequence.size() ||
        (cursor.segment == sequence.size() && cursor.offset != 0)) {
      status_ = Status::OutOfRange("buffer sequence cursor is outside the sequence");
      return;
    }

    for (std::size_t i = 0; i < sequence.size(); ++i) {
      logical_[i] = sequence[i];
      logical_sizes_[i] = sequence[i].size();
      if (sequence[i].size() != 0 && sequence[i].data() == nullptr) {
        status_ = Status::InvalidArgument(
            "a non-empty buffer sequence segment has a null address");
        return;
      }
    }
    if (cursor.segment < sequence.size() &&
        cursor.offset > logical_sizes_[cursor.segment]) {
      status_ = Status::OutOfRange(
          "buffer sequence cursor offset exceeds its segment");
      return;
    }

    normalize(start_);
    for (std::size_t i = start_.segment; i < sequence.size(); ++i) {
      const std::size_t skipped = i == start_.segment ? start_.offset : 0;
      const std::size_t length = sequence[i].size() - skipped;
      if (length == 0) continue;
      if (length > kMaxBufferSequenceBytes ||
          total_size_ > kMaxBufferSequenceBytes - length) {
        status_ = Status::OutOfRange(
            "buffer sequence exceeds the portable 2 GiB byte limit");
        native_count_ = 0;
        total_size_ = 0;
        return;
      }
      auto* address = const_cast<void*>(
          static_cast<const void*>(
              static_cast<const std::byte*>(sequence[i].data()) + skipped));
      native_[native_count_++] = NativeBuffer(address, length);
      total_size_ += length;
    }
  }

  bool ok() const noexcept { return status_.ok(); }
  const Status& status() const noexcept { return status_; }
  bool empty() const noexcept { return total_size_ == 0; }
  std::size_t native_count() const noexcept { return native_count_; }
  std::size_t total_size() const noexcept { return total_size_; }
  NativeBuffer* native_data() noexcept { return native_.data(); }
  const NativeBuffer* native_data() const noexcept { return native_.data(); }
  std::span<const NativeBuffer> native_buffers() const noexcept {
    return {native_.data(), native_count_};
  }
  std::span<const Buffer> sequence() const noexcept {
    return {logical_.data(), logical_count_};
  }
  BufferSequenceCursor start() const noexcept { return start_; }

  BufferSequenceProgress progress(std::size_t bytes) const noexcept {
    BufferSequenceProgress out{bytes, start_, false};
    std::size_t remaining = bytes;
    while (out.next.segment < logical_count_) {
      const std::size_t available =
          logical_sizes_[out.next.segment] - out.next.offset;
      if (remaining < available) {
        out.next.offset += remaining;
        remaining = 0;
        break;
      }
      remaining -= available;
      ++out.next.segment;
      out.next.offset = 0;
      normalize(out.next);
      if (remaining == 0) break;
    }
    out.complete = out.next.segment == logical_count_;
    return out;
  }

  void copy_to_contiguous(void* destination) const noexcept {
    auto* output = static_cast<std::byte*>(destination);
    for (const NativeBuffer& buffer : native_buffers()) {
      std::memcpy(output, buffer.data(), buffer.size());
      output += buffer.size();
    }
  }

  void copy_from_contiguous(const void* source, std::size_t bytes) noexcept
    requires(std::is_same_v<Buffer, MutableBuffer>)
  {
    const auto* input = static_cast<const std::byte*>(source);
    std::size_t remaining = bytes;
    for (const NativeBuffer& buffer : native_buffers()) {
      const std::size_t count =
          remaining < buffer.size() ? remaining : buffer.size();
      if (count != 0) std::memcpy(buffer.data(), input, count);
      input += count;
      remaining -= count;
      if (remaining == 0) break;
    }
  }

 private:
  void normalize(BufferSequenceCursor& cursor) const noexcept {
    while (cursor.segment < logical_count_ &&
           cursor.offset == logical_sizes_[cursor.segment]) {
      ++cursor.segment;
      cursor.offset = 0;
    }
  }

  std::array<NativeBuffer, kMaxBufferSequenceSegments> native_{};
  std::array<Buffer, kMaxBufferSequenceSegments> logical_{};
  std::array<std::size_t, kMaxBufferSequenceSegments> logical_sizes_{};
  Status status_{};
  BufferSequenceCursor start_{};
  std::size_t logical_count_ = 0;
  std::size_t native_count_ = 0;
  std::size_t total_size_ = 0;
};

}  // namespace detail
}  // namespace hyle::io
