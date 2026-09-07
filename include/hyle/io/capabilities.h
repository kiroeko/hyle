#pragma once

// The coroutine transport is a product-shape capability, not a property that
// can be inferred from the presence of the public header tree: HYLE installs
// the reviewed headers as one surface even when the target has no sched/
// fcontext implementation.  Linking hyle::io supplies the authoritative 0/1
// definition in both add_subdirectory() and installed-package consumption.
//
// Raw headers + library are also a supported delivery form.  When no CMake
// target supplied the product definition, derive the same public target fact
// as root CMake: x64 on the Microsoft ABI, Linux, or macOS; and arm64 on Linux
// or macOS.  In particular, MinGW x64 and Windows arm64 are not claimed until
// HYLE ships matching fcontext assembly/build integration.
#include <cstdint>
#include <string_view>

#include "hyle/core/environment.h"
#include "hyle/error/status.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_IO_API
#elif defined(HYLE_IO_CORE_BUILD)
#  define HYLE_IO_API HYLE_EXPORT
#else
#  define HYLE_IO_API HYLE_IMPORT
#endif

#if HYLE_ARCH_X64 &&                                           \
    ((HYLE_OS_WINDOWS && defined(_MSC_VER)) ||                 \
     HYLE_OS_LINUX || HYLE_OS_MACOS)
#  define HYLE_IO_DETAIL_DERIVED_COROUTINE_TRANSPORT 1
#elif HYLE_ARCH_ARM64 &&                                      \
    (HYLE_OS_LINUX || HYLE_OS_MACOS)
#  define HYLE_IO_DETAIL_DERIVED_COROUTINE_TRANSPORT 1
#else
#  define HYLE_IO_DETAIL_DERIVED_COROUTINE_TRANSPORT 0
#endif

// A linked source/install target is authoritative, but it must never disagree
// with the public compiler/ABI facts used by a raw headers+library consumer.
// Such a mismatch means the package was generated for a different target, not
// an optional feature that can safely degrade.
#if defined(HYLE_IO_HAS_COROUTINE_TRANSPORT)
#  if HYLE_IO_HAS_COROUTINE_TRANSPORT != HYLE_IO_DETAIL_DERIVED_COROUTINE_TRANSPORT
#    error "HYLE io coroutine-transport target/header capability mismatch; use a library package built for this OS/architecture/compiler ABI."
#  endif
#else
#  define HYLE_IO_HAS_COROUTINE_TRANSPORT \
    HYLE_IO_DETAIL_DERIVED_COROUTINE_TRANSPORT
#endif

namespace hyle::io {

class IoContext;

namespace detail {
struct IoUringAccess;
}

// Runtime backend facts are observable tuning information, not an invitation
// to submit raw completion tokens.  Ordinary transports remain portable; code
// which deliberately opts into an OS-specific accelerator must first acquire
// its typed extension from IoContext.
enum class IoBackend : std::uint8_t {
  kNone = 0,
  kIocp,
  kIoUring,
  kEpoll,
  kKqueue,
};

enum class IoCapability : std::uint64_t {
  kNone = 0,
  kNativeMultishotAccept = 1ull << 0,
  kEmulatedMultishotAccept = 1ull << 1,
  kProvidedBufferRing = 1ull << 2,
  kFixedFiles = 1ull << 3,
  kDirectAccept = 1ull << 4,
  kAsyncFileData = 1ull << 5,
  kAsyncFileMetadata = 1ull << 6,
};

constexpr IoCapability operator|(IoCapability left,
                                 IoCapability right) noexcept {
  return static_cast<IoCapability>(static_cast<std::uint64_t>(left) |
                                   static_cast<std::uint64_t>(right));
}

constexpr IoCapability& operator|=(IoCapability& left,
                                   IoCapability right) noexcept {
  left = left | right;
  return left;
}

constexpr bool has_io_capability(IoCapability capabilities,
                                 IoCapability capability) noexcept {
  return (static_cast<std::uint64_t>(capabilities) &
          static_cast<std::uint64_t>(capability)) != 0;
}

struct IoCapabilities {
  IoBackend backend = IoBackend::kNone;
  IoCapability supported = IoCapability::kNone;

  [[nodiscard]] constexpr bool has(IoCapability capability) const noexcept {
    return has_io_capability(supported, capability);
  }

  [[nodiscard]] constexpr std::string_view backend_name() const noexcept {
    switch (backend) {
      case IoBackend::kIocp: return "iocp";
      case IoBackend::kIoUring: return "io_uring";
      case IoBackend::kEpoll: return "epoll";
      case IoBackend::kKqueue: return "kqueue";
      case IoBackend::kNone: return "none";
    }
    return "none";
  }
};

#if HYLE_OS_LINUX

struct IoUringBufferRingOptions {
  int group_id = 0;
  unsigned entries = 256;
  unsigned buffer_size = 16 * 1024;
};

// Borrowed handle to the io_uring backend owned by one IoContext worker.  The
// type exists only on Linux, and IoContext returns an empty handle when the
// runtime selected epoll.  Configuration is owner-lifetime state: registered
// buffers are reclaimed with the context and every operation using them must be
// retired before context destruction.
class IoUringExtension {
 public:
  IoUringExtension() noexcept = default;

  [[nodiscard]] bool valid() const noexcept { return context_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  [[nodiscard]] unsigned owner_worker() const noexcept { return worker_; }

  HYLE_IO_API Status configure_receive_ring(
      IoUringBufferRingOptions options) const noexcept;

 private:
  friend class IoContext;
  friend struct detail::IoUringAccess;

  explicit IoUringExtension(IoContext* context, unsigned worker) noexcept
      : context_(context), worker_(worker) {}

  IoContext* context_ = nullptr;
  unsigned worker_ = 0;
};

#endif

}  // namespace hyle::io
