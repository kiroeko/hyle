#include "hyle/io/operation.h"

#include <limits>

#include "hyle/io/net.h"

namespace hyle::io::detail {
namespace {

constexpr std::int64_t kCanonicalCancelled = -125;
constexpr std::int64_t kCanonicalTimedOut = -110;

Status status_from_negative(std::int64_t result, NativeOperationDomain domain,
                            const char *operation) noexcept {
  if (result == kSubmitFailed)
    return Status::Unavailable("IO backend rejected the operation submission");
  if (result == kCanonicalCancelled)
    return Status::Cancelled(operation);
  if (result == kCanonicalTimedOut)
    return Status::DeadlineExceeded(operation);
  if (result >= 0 || result == (std::numeric_limits<std::int64_t>::min)())
    return Status::Internal("invalid native IO result");

  const std::uint64_t magnitude = static_cast<std::uint64_t>(-(result + 1)) + 1;
  if (magnitude > static_cast<std::uint64_t>((std::numeric_limits<int>::max)()))
    return Status::Unknown("native IO error is outside the supported range");
  const int code = static_cast<int>(magnitude);
  switch (domain) {
  case NativeOperationDomain::kPosix:
    return status_from_errno(code, operation);
  case NativeOperationDomain::kWin32:
    return status_from_win32(static_cast<std::uint32_t>(code), operation);
  case NativeOperationDomain::kWinSock:
    return status_from_winsock(code, operation);
  }
  return Status::Unknown(operation);
}

} // namespace

TransferResult transfer_from_native(std::int64_t result,
                                    NativeOperationDomain domain,
                                    const char *operation,
                                    bool read_operation) noexcept {
  if (result < 0)
    return TransferResult{0, status_from_negative(result, domain, operation),
                          false};
  return TransferResult{static_cast<std::size_t>(result), Status{},
                        read_operation && result == 0};
}

Status status_from_native_result(std::int64_t result,
                                 NativeOperationDomain domain,
                                 const char *operation) noexcept {
  return result < 0 ? status_from_negative(result, domain, operation)
                    : Status{};
}

} // namespace hyle::io::detail
