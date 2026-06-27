#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

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

namespace detail {
// Heap representation lives entirely in the DLL (status.cc). These are the only
// operations the header needs; the layout of the rep is never exposed.
HYLE_ERROR_API std::uintptr_t status_make_heap(StatusCode code, const char* msg,
                                               std::size_t len);
HYLE_ERROR_API void status_retain(std::uintptr_t rep) noexcept;
HYLE_ERROR_API void status_release(std::uintptr_t rep) noexcept;
HYLE_ERROR_API StatusCode status_heap_code(std::uintptr_t rep) noexcept;
HYLE_ERROR_API std::string_view status_heap_message(std::uintptr_t rep) noexcept;
}  // namespace detail

// ---- Status -----------------------------------------------------------------
// 8-byte error carrier (clay-style tagged pointer). Encoding of rep_:
//   0                       -> OK            (most common; single compare)
//   (code << 2) | 1         -> inline code   (low bit set; NO heap, zero alloc)
//   pointer (low 2 bits 00) -> heap StatusRep (only when a message is attached)
// So OK and message-less errors never touch the heap. Copyable: heap reps are
// ref-counted (retain/release in the DLL). Public surface is a single uintptr_t
// -- no std owning type leaks across the ABI seam (coding-conventions line 22).
class [[nodiscard]] Status {
 public:
  Status() noexcept = default;  // OK

  Status(StatusCode code, std::string_view message) {
    if (code != StatusCode::Ok) {
      if (message.empty()) {
        rep_ = (static_cast<std::uintptr_t>(code) << kCodeShift) | kInlineTag;
      } else {
        rep_ = detail::status_make_heap(code, message.data(), message.size());
      }
    }
  }

  explicit Status(StatusCode code) noexcept
      : rep_(code == StatusCode::Ok
                 ? 0
                 : (static_cast<std::uintptr_t>(code) << kCodeShift) | kInlineTag) {}

  Status(const Status& o) noexcept : rep_(o.rep_) {
    if (is_heap()) detail::status_retain(rep_);
  }
  Status(Status&& o) noexcept : rep_(o.rep_) { o.rep_ = 0; }
  Status& operator=(const Status& o) noexcept {
    if (this != &o) {
      if (is_heap()) detail::status_release(rep_);
      rep_ = o.rep_;
      if (is_heap()) detail::status_retain(rep_);
    }
    return *this;
  }
  Status& operator=(Status&& o) noexcept {
    if (this != &o) {
      if (is_heap()) detail::status_release(rep_);
      rep_ = o.rep_;
      o.rep_ = 0;
    }
    return *this;
  }
  ~Status() {
    if (is_heap()) detail::status_release(rep_);
  }

  bool ok() const noexcept { return rep_ == 0; }

  StatusCode code() const noexcept {
    if (rep_ == 0) return StatusCode::Ok;
    if (rep_ & kInlineTag) return static_cast<StatusCode>(rep_ >> kCodeShift);
    return detail::status_heap_code(rep_);
  }

  bool has_message() const noexcept { return is_heap(); }
  std::string_view message() const noexcept {
    return is_heap() ? detail::status_heap_message(rep_) : std::string_view();
  }

  // Format into a caller buffer (snprintf-style): writes up to cap-1 bytes + a
  // NUL, returns the total length needed (so the caller can detect truncation).
  // No allocation, no std owning type across the seam. "OK" | "CODE" |
  // "CODE: message". Compose from code_name()/message() if you want the pieces.
  HYLE_ERROR_API std::size_t ToString(char* buf, std::size_t cap) const noexcept;

  // Per-code predicates: IsNotFound(), IsCancelled(), ... (Ok included).
#define HYLE_STATUS_PREDICATE(name, value)                                     \
  bool Is##name() const noexcept { return code() == StatusCode::name; }
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
    return a.code() == b.code() && a.message() == b.message();
  }
  friend bool operator!=(const Status& a, const Status& b) noexcept {
    return !(a == b);
  }

 private:
  static constexpr std::uintptr_t kInlineTag = 1;
  static constexpr std::uintptr_t kCodeShift = 2;

  // heap rep: nonzero with low 2 bits clear (malloc is >=16-byte aligned).
  bool is_heap() const noexcept { return rep_ != 0 && (rep_ & 3) == 0; }

  std::uintptr_t rep_ = 0;  // 0 = OK
};

static_assert(sizeof(Status) == sizeof(std::uintptr_t),
              "Status must stay a single tagged word");

}  // namespace hyle
