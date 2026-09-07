#pragma once

#include "hyle/core/environment.h"

#if HYLE_OS_LINUX

#include <coroutine>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

#include "hyle/error/result.h"
#include "hyle/io/capabilities.h"
#include "hyle/io/detail/net_multishot.h"
#include "hyle/io/operation.h"
#include "hyle/mem/allocator.h"

namespace hyle::io {

class TcpStream;

// One kernel-selected provided buffer.  The bytes are borrowed until this
// object is moved from or destroyed; destruction returns the buffer to its
// io_uring ring.  Keeping a buffer indefinitely applies backpressure by
// reducing the ring's available receive capacity.
class IoUringReceiveBuffer {
 public:
  IoUringReceiveBuffer() noexcept = default;
  IoUringReceiveBuffer(IoUringReceiveBuffer&&) noexcept = default;
  IoUringReceiveBuffer& operator=(IoUringReceiveBuffer&&) noexcept = default;
  IoUringReceiveBuffer(const IoUringReceiveBuffer&) = delete;
  IoUringReceiveBuffer& operator=(const IoUringReceiveBuffer&) = delete;

  [[nodiscard]] std::span<const std::byte> bytes() const noexcept {
    return {static_cast<const std::byte*>(buffer_.data()), buffer_.size()};
  }

  // false marks the terminal item (normally EOF).  Do not call next() again.
  [[nodiscard]] bool more() const noexcept { return buffer_.more(); }

 private:
  friend class IoUringReceiveStream;

  explicit IoUringReceiveBuffer(detail::RecvBuf buffer) noexcept
      : buffer_(std::move(buffer)) {}

  detail::RecvBuf buffer_{};
};

// Linux io_uring provided-buffer/multishot receive stream.  Construction is
// available only from a TcpStream whose owner matches the acquired extension.
// Dropping an armed stream submits cancellation and keeps its internal state
// alive until the terminal completion retires.
class IoUringReceiveStream {
 public:
  IoUringReceiveStream() noexcept = default;
  IoUringReceiveStream(IoUringReceiveStream&&) noexcept = default;
  IoUringReceiveStream& operator=(IoUringReceiveStream&&) = delete;
  IoUringReceiveStream(const IoUringReceiveStream&) = delete;
  IoUringReceiveStream& operator=(const IoUringReceiveStream&) = delete;

  class NextAwaitable {
   public:
    bool await_ready() noexcept { return inner_.await_ready(); }
    bool await_suspend(std::coroutine_handle<> handle) noexcept {
      return inner_.await_suspend(handle);
    }
    Result<IoUringReceiveBuffer> await_resume() noexcept {
      auto buffer = inner_.await_resume();
      if (buffer.err() < 0) {
        TransferResult translated = detail::transfer_from_native(
            buffer.err(), detail::kNetworkNativeDomain,
            "io_uring multishot receive", true);
        return make_unexpected(std::move(translated.status));
      }
      return IoUringReceiveBuffer(std::move(buffer));
    }

   private:
    friend class IoUringReceiveStream;
    explicit NextAwaitable(
        detail::MultishotRecvStream::NextAwaitable inner) noexcept
        : inner_(inner) {}

    detail::MultishotRecvStream::NextAwaitable inner_;
  };

  NextAwaitable next() noexcept { return NextAwaitable(inner_.next()); }
  [[nodiscard]] bool ended() const noexcept { return inner_.ended(); }

 private:
  friend class TcpStream;

  IoUringReceiveStream(detail::Proactor* proactor, std::intptr_t handle,
                       int group_id, mem::AllocatorRef allocator) noexcept
      : inner_(proactor, handle, group_id, allocator) {}

  detail::MultishotRecvStream inner_{};
};

}  // namespace hyle::io

#endif
