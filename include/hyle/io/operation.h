#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

#include "hyle/error/status.h"
#include "hyle/core/environment.h"
#include "hyle/time/clock.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_IO_API
#elif defined(HYLE_IO_CORE_BUILD)
#  define HYLE_IO_API HYLE_EXPORT
#else
#  define HYLE_IO_API HYLE_IMPORT
#endif

namespace hyle::sched {
class CancelToken;
}

namespace hyle::io {

// One budget follows a logical operation through every partial syscall. A
// missing deadline/token means unbounded/cooperatively uncancelled; the default
// single-buffer fast path does not allocate merely because this value exists.
struct OperationOptions {
  // Absolute steady-clock deadline.  `time::deadline_after(...)` is the usual
  // construction helper; wall-clock timestamps are intentionally a different
  // type and cannot be passed accidentally.
  std::optional<time::MonotonicTimePoint> deadline;
  const sched::CancelToken* cancel = nullptr;
};

// Transfer outcome keeps partial progress even when the terminal state is EOF,
// cancellation, timeout, or an OS failure. Result<size_t> alone cannot preserve
// both a useful partial byte count and the error that stopped a composed read or
// write. A successful read with eof=true may have transferred zero or more bytes.
struct [[nodiscard]] TransferResult {
  std::size_t bytes = 0;
  Status status{};
  bool eof = false;

  bool ok() const noexcept { return status.ok(); }
  explicit operator bool() const noexcept { return ok(); }
};

namespace detail {

enum class NativeOperationDomain : std::uint8_t {
  kPosix,
  kWin32,
  kWinSock,
};

inline constexpr NativeOperationDomain kNetworkNativeDomain =
#if HYLE_OS_WINDOWS
    NativeOperationDomain::kWinSock;
#else
    NativeOperationDomain::kPosix;
#endif

inline constexpr NativeOperationDomain kFileNativeDomain =
#if HYLE_OS_WINDOWS
    NativeOperationDomain::kWin32;
#else
    NativeOperationDomain::kPosix;
#endif

// Translate the private proactor's signed result at the ordinary C++ boundary.
// `read_operation` gives zero its EOF meaning; a zero-length caller request is
// handled before submit and therefore never becomes a false EOF here.
HYLE_IO_API TransferResult transfer_from_native(
    std::int64_t result, NativeOperationDomain domain,
    const char* operation, bool read_operation) noexcept;

}  // namespace detail
}  // namespace hyle::io
