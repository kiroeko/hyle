#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_ERROR_API
#elif defined(HYLE_ERROR_BUILD)
#  define HYLE_ERROR_API HYLE_EXPORT
#else
#  define HYLE_ERROR_API HYLE_IMPORT
#endif

namespace hyle {

// ---- status code table (single source of truth) ----------------------------
// X-macro: add a code by adding ONE line here; the enum, the name table, the
// factory functions and the Is<Code>() predicates are all generated from it.
// Code set follows the well-understood abseil/gRPC canonical taxonomy.
//   X(Name, value)
#define HYLE_FOREACH_STATUS_CODE(X)                                            \
  X(Ok, 0)                                                                     \
  X(Cancelled, 1)                                                              \
  X(Unknown, 2)                                                                \
  X(InvalidArgument, 3)                                                        \
  X(DeadlineExceeded, 4)                                                       \
  X(NotFound, 5)                                                               \
  X(AlreadyExists, 6)                                                          \
  X(PermissionDenied, 7)                                                       \
  X(ResourceExhausted, 8)                                                      \
  X(FailedPrecondition, 9)                                                     \
  X(Aborted, 10)                                                               \
  X(OutOfRange, 11)                                                            \
  X(Unimplemented, 12)                                                         \
  X(Internal, 13)                                                              \
  X(Unavailable, 14)                                                           \
  X(DataLoss, 15)

enum class StatusCode : std::uint16_t {
#define HYLE_STATUS_ENUMERATOR(name, value) name = value,
  HYLE_FOREACH_STATUS_CODE(HYLE_STATUS_ENUMERATOR)
#undef HYLE_STATUS_ENUMERATOR
};

// Stable, never-localized code name ("Ok", "NotFound", ...). Defined in the DLL
// so the name table lives in exactly one translation unit.
HYLE_ERROR_API std::string_view status_code_name(StatusCode code) noexcept;

// ---- Status -----------------------------------------------------------------
// Compact error carrier. The OK state and any message-less error allocate
// nothing: the heap string is only created when a non-empty message is attached
// to a non-OK status. Copyable (shared, ref-counted message) so it rides inside
// std::expected<T, Status> without forcing move-only ergonomics.
class [[nodiscard]] Status {
 public:
  // Default = OK. The single most common path, and it is trivially cheap.
  Status() noexcept = default;

  // Build from a code (+ optional message). An OK code ignores the message.
  Status(StatusCode code, std::string_view message) {
    if (code != StatusCode::Ok) {
      code_ = code;
      if (!message.empty()) {
        msg_ = std::make_shared<const std::string>(message);
      }
    }
  }

  explicit Status(StatusCode code) noexcept : code_(code) {}

  bool ok() const noexcept { return code_ == StatusCode::Ok; }
  StatusCode code() const noexcept { return code_; }

  bool has_message() const noexcept { return static_cast<bool>(msg_); }
  std::string_view message() const noexcept {
    return msg_ ? std::string_view(*msg_) : std::string_view();
  }

  // "OK" | "CODE" | "CODE: message"
  HYLE_ERROR_API std::string ToString() const;

  // Per-code predicates: IsNotFound(), IsCancelled(), ... (Ok included).
#define HYLE_STATUS_PREDICATE(name, value)                                     \
  bool Is##name() const noexcept { return code_ == StatusCode::name; }
  HYLE_FOREACH_STATUS_CODE(HYLE_STATUS_PREDICATE)
#undef HYLE_STATUS_PREDICATE

  // Per-code factories: Status::NotFound("msg"), Status::Internal("msg"), ...
#define HYLE_STATUS_FACTORY(name, value)                                       \
  static Status name(std::string_view message = {}) {                          \
    return Status(StatusCode::name, message);                                  \
  }
  HYLE_FOREACH_STATUS_CODE(HYLE_STATUS_FACTORY)
#undef HYLE_STATUS_FACTORY

  friend bool operator==(const Status& a, const Status& b) noexcept {
    return a.code_ == b.code_ && a.message() == b.message();
  }
  friend bool operator!=(const Status& a, const Status& b) noexcept {
    return !(a == b);
  }

 private:
  StatusCode code_ = StatusCode::Ok;
  // null in the common case (OK, or error without a message) -> zero alloc.
  std::shared_ptr<const std::string> msg_;
};

}  // namespace hyle
