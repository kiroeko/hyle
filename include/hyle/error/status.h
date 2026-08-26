#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>

#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#define HYLE_ERROR_API
#elif defined(HYLE_ERROR_BUILD)
#define HYLE_ERROR_API HYLE_EXPORT
#else
#define HYLE_ERROR_API HYLE_IMPORT
#endif

namespace hyle {

// ---- status code table (single source of truth) ----------------------------
// X-macro: each row is X(PascalName, SCREAMING_NAME, value). The C++ StatusCode
// enum, name table, factories and Is<Code>() predicates are generated from this
// single source. The code set follows the Abseil/gRPC canonical taxonomy.
//   X(PascalName, SCREAMING_NAME, value)
#define HYLE_FOREACH_ERROR_STATUS_CODE(X)                                      \
  X(Cancelled, CANCELLED, 1)                                                   \
  X(Unknown, UNKNOWN, 2)                                                       \
  X(InvalidArgument, INVALID_ARGUMENT, 3)                                      \
  X(DeadlineExceeded, DEADLINE_EXCEEDED, 4)                                    \
  X(NotFound, NOT_FOUND, 5)                                                    \
  X(AlreadyExists, ALREADY_EXISTS, 6)                                          \
  X(PermissionDenied, PERMISSION_DENIED, 7)                                    \
  X(ResourceExhausted, RESOURCE_EXHAUSTED, 8)                                  \
  X(FailedPrecondition, FAILED_PRECONDITION, 9)                                \
  X(Aborted, ABORTED, 10)                                                      \
  X(OutOfRange, OUT_OF_RANGE, 11)                                              \
  X(Unimplemented, UNIMPLEMENTED, 12)                                          \
  X(Internal, INTERNAL, 13)                                                    \
  X(Unavailable, UNAVAILABLE, 14)                                              \
  X(DataLoss, DATA_LOSS, 15)                                                   \
  X(Unauthenticated, UNAUTHENTICATED, 16)

#define HYLE_FOREACH_STATUS_CODE(X)                                            \
  X(Ok, OK, 0)                                                                 \
  HYLE_FOREACH_ERROR_STATUS_CODE(X)

enum class StatusCode : std::uint16_t {
#define HYLE_STATUS_ENUMERATOR(name, screaming, value) name = value,
  HYLE_FOREACH_STATUS_CODE(HYLE_STATUS_ENUMERATOR)
#undef HYLE_STATUS_ENUMERATOR
};

// Stable categories for one optional backend/OS error number. This deliberately
// stays narrower than std::error_category or an arbitrary payload registry:
// portable recovery uses StatusCode; backend-specific recovery may inspect
// this.
enum class NativeErrorDomain : std::uint16_t {
  None = 0,
  PosixErrno = 1,
  Win32 = 2,
  Vulkan = 3,
  CudaRuntime = 4,
  CudaDriver = 5,
  Nvrtc = 6,
  HResult = 7,
  Unknown = 0xffff,
};

struct NativeError {
  NativeErrorDomain domain = NativeErrorDomain::None;
  std::int64_t code = 0;

  constexpr bool present() const noexcept {
    return domain != NativeErrorDomain::None;
  }
  friend constexpr bool operator==(NativeError, NativeError) noexcept = default;
};

// Stable, never-localized code name ("Ok", "NotFound", ...). Defined in the DLL
// so the name table lives in exactly one translation unit.
HYLE_ERROR_API std::string_view status_code_name(StatusCode code) noexcept;
HYLE_ERROR_API std::string_view
native_error_domain_name(NativeErrorDomain domain) noexcept;

// Map a C/POSIX errno value onto the canonical Status taxonomy. This is a
// pure O(1) classification and returns Ok for error_number == 0. Individual
// OS operations may still override the generic mapping when their contract
// gives an errno a more specific meaning (for example pthread_create(EAGAIN)
// is resource exhaustion rather than a retryable I/O condition).
HYLE_ERROR_API StatusCode status_code_from_errno(int error_number) noexcept;

// Map the stable 32-bit value returned by Win32 GetLastError onto the same
// canonical taxonomy. The mapping is available on every host so tools can
// classify persisted/remote Windows diagnostics without including windows.h.
// Zero is Ok; unknown values remain Unknown rather than being marked retryable.
HYLE_ERROR_API StatusCode
status_code_from_win32(std::uint32_t error_number) noexcept;

// Map a signed 32-bit COM/WinRT HRESULT without requiring windows.h. Any
// non-negative HRESULT is success. HRESULT_FROM_WIN32 values reuse the Win32
// taxonomy; common COM failures retain their precise portable meaning. The
// mapping remains available on every host for persisted/remote diagnostics.
HYLE_ERROR_API StatusCode
status_code_from_hresult(std::int32_t error_number) noexcept;

namespace detail {
class StatusAccess;

constexpr bool status_code_is_valid(StatusCode code) noexcept {
  return static_cast<std::uint16_t>(code) <=
         static_cast<std::uint16_t>(StatusCode::Unauthenticated);
}

constexpr StatusCode normalize_status_code(StatusCode code) noexcept {
  return status_code_is_valid(code) ? code : StatusCode::Unknown;
}

constexpr NativeErrorDomain
normalize_native_error_domain(NativeErrorDomain domain) noexcept {
  switch (domain) {
  case NativeErrorDomain::None:
  case NativeErrorDomain::PosixErrno:
  case NativeErrorDomain::Win32:
  case NativeErrorDomain::Vulkan:
  case NativeErrorDomain::CudaRuntime:
  case NativeErrorDomain::CudaDriver:
  case NativeErrorDomain::Nvrtc:
  case NativeErrorDomain::HResult:
  case NativeErrorDomain::Unknown:
    return domain;
  }
  return NativeErrorDomain::Unknown;
}

// Heap representation for message/detail errors lives in the DLL (status.cc).
// One prefix is deliberately a compiled-code contract: consumer-side Status
// copies inline the relaxed retain below, as mature intrusive carriers do. The
// allocation, flags, message/native layout and final-release path stay opaque.
HYLE_ERROR_API std::uintptr_t status_make_heap(StatusCode code, const char *msg,
                                               std::size_t len) noexcept;
HYLE_ERROR_API std::uintptr_t status_make_native_heap(StatusCode code,
                                                      NativeError native,
                                                      const char *msg,
                                                      std::size_t len) noexcept;
HYLE_ERROR_API std::uintptr_t
status_make_context(StatusCode code, NativeError native, const char *context,
                    std::size_t context_len, const char *message,
                    std::size_t message_len) noexcept;
// Keep the hot retain operation in consumer code, like Abseil/Folly intrusive
// carriers do. Only this stable prefix is visible; allocation, message storage
// and the last-release/free path remain inside the DLL.
struct StatusRepHeader {
  std::atomic<std::uint32_t> refcount;
};
inline void status_retain(std::uintptr_t rep) noexcept {
  reinterpret_cast<StatusRepHeader *>(rep)->refcount.fetch_add(
      1, std::memory_order_relaxed);
}
HYLE_ERROR_API void status_release(std::uintptr_t rep) noexcept;
HYLE_ERROR_API StatusCode status_heap_code(std::uintptr_t rep) noexcept;
HYLE_ERROR_API std::string_view
status_heap_message(std::uintptr_t rep) noexcept;
HYLE_ERROR_API NativeError
status_heap_native_error(std::uintptr_t rep) noexcept;

// TEST-ONLY: while enabled, Status message/detail allocation behaves as if
// malloc returned null. Shipped builds strip this (HYLE_TESTING off).
#if defined(HYLE_TESTING)
HYLE_ERROR_API void status_test_force_oom(bool enable) noexcept;
#endif
} // namespace detail

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
  Status() noexcept = default; // OK

  Status(StatusCode code, std::string_view message) noexcept {
    code = detail::normalize_status_code(code);
    if (code != StatusCode::Ok) {
      std::uintptr_t inlined =
          (static_cast<std::uintptr_t>(code) << kCodeShift) | kInlineTag;
      if (message.empty()) {
        rep_ = inlined;
      } else {
        // status_make_heap returns 0 on OOM -> degrade to the message-less
        // inline encoding (the code survives; only the message is dropped).
        std::uintptr_t heap =
            detail::status_make_heap(code, message.data(), message.size());
        rep_ = heap != 0 ? heap : inlined;
      }
    }
  }

  explicit Status(StatusCode code) noexcept
      : rep_((code = detail::normalize_status_code(code)) == StatusCode::Ok
                 ? 0
                 : (static_cast<std::uintptr_t>(code) << kCodeShift) |
                       kInlineTag) {}

  // Attach one machine-readable native/backend code without growing the
  // one-word Status object. OOM preserves the canonical code and drops only
  // optional message/detail.
  static Status FromNative(StatusCode code, NativeError native,
                           std::string_view message = {}) noexcept {
    code = detail::normalize_status_code(code);
    native.domain = detail::normalize_native_error_domain(native.domain);
    if (code == StatusCode::Ok)
      return Status{};
    if (!native.present())
      return Status(code, message);
    const std::uintptr_t heap = detail::status_make_native_heap(
        code, native, message.data(), message.size());
    if (heap == 0)
      return Status(code);
    Status result;
    result.rep_ = heap;
    return result;
  }

  Status(const Status &o) noexcept : rep_(o.rep_) {
    if (is_heap())
      detail::status_retain(rep_);
  }
  // A moved-from Status remains a valid code-only error rather than becoming
  // OK. This matches mature Status carriers and, critically, prevents moving
  // a Result error arm from creating an error-state Result whose error is OK.
  Status(Status &&o) noexcept : rep_(o.rep_) { o.rep_ = kMovedFromRep; }
  Status &operator=(const Status &o) noexcept {
    // Distinct Status objects commonly share one immutable heap backing.  The
    // value is already identical in that case, so avoid an otherwise needless
    // release/retain pair (and its two atomic operations).
    if (rep_ != o.rep_) {
      if (is_heap())
        detail::status_release(rep_);
      rep_ = o.rep_;
      if (is_heap())
        detail::status_retain(rep_);
    }
    return *this;
  }
  Status &operator=(Status &&o) noexcept {
    if (this != &o) {
      if (is_heap())
        detail::status_release(rep_);
      rep_ = o.rep_;
      o.rep_ = kMovedFromRep;
    }
    return *this;
  }
  ~Status() {
    if (is_heap())
      detail::status_release(rep_);
  }

  bool ok() const noexcept { return rep_ == 0; }

  StatusCode code() const noexcept {
    if (rep_ == 0)
      return StatusCode::Ok;
    if (rep_ & kInlineTag)
      return static_cast<StatusCode>(rep_ >> kCodeShift);
    return detail::status_heap_code(rep_);
  }

  bool has_message() const noexcept { return !message().empty(); }
  std::string_view message() const noexcept {
    return is_heap() ? detail::status_heap_message(rep_) : std::string_view();
  }
  NativeError native_error() const noexcept {
    return is_heap() ? detail::status_heap_native_error(rep_) : NativeError{};
  }
  bool has_native_error() const noexcept { return native_error().present(); }

  // Format into a caller buffer (snprintf-style): writes up to cap-1 bytes + a
  // NUL, returns the total length needed (so the caller can detect truncation).
  // A null buffer is treated as cap=0, allowing a size-only query.
  // No allocation, no std owning type across the seam. "OK" | "CODE" |
  // "CODE: message". Compose from code_name()/message() if you want the pieces.
  HYLE_ERROR_API std::size_t ToString(char *buf,
                                      std::size_t cap) const noexcept;

  // Prefix semantic context on an error: "context: existing message". OK and
  // empty context are no-ops. The first layer is exact-size; a progressive
  // rvalue chain then reuses unique geometrically grown prefix storage
  // (amortized linear total copying). Lvalues/shared errors use copy-on-write.
  // On allocation failure the original status and message are preserved.
  // Keep the common OK/empty path in every consumer and export only the
  // distinctly-named slow path. This avoids a split inline/out-of-line identity
  // for the same public member across the shared-library build boundary.
  Status WithContext(std::string_view context) const & noexcept {
    if (ok() || context.empty())
      return *this;
    return WithContextSlow(context);
  }
  Status WithContext(std::string_view context) && noexcept {
    if (ok() || context.empty())
      return std::move(*this);
    return std::move(*this).WithContextSlow(context);
  }

  // Per-code predicates: IsNotFound(), IsCancelled(), ... (Ok included).
#define HYLE_STATUS_PREDICATE(name, screaming, value)                          \
  bool Is##name() const noexcept { return code() == StatusCode::name; }
  HYLE_FOREACH_STATUS_CODE(HYLE_STATUS_PREDICATE)
#undef HYLE_STATUS_PREDICATE

  // Per-code factories: Status::NotFound("msg"), Status::Internal("msg"), ...
#define HYLE_STATUS_FACTORY(name, screaming, value)                            \
  static Status name(std::string_view message = {}) noexcept {                 \
    return Status(StatusCode::name, message);                                  \
  }
  HYLE_FOREACH_ERROR_STATUS_CODE(HYLE_STATUS_FACTORY)
#undef HYLE_STATUS_FACTORY

  static Status Ok() noexcept { return Status(); }

  friend bool operator==(const Status &a, const Status &b) noexcept {
    // Copies share an immutable representation.  Identity is therefore a
    // complete equality proof and makes equality of shared long diagnostics
    // O(1) instead of re-reading and comparing the full message.
    if (a.rep_ == b.rep_)
      return true;
    return a.code() == b.code() && a.native_error() == b.native_error() &&
           a.message() == b.message();
  }
  friend bool operator!=(const Status &a, const Status &b) noexcept {
    return !(a == b);
  }

  // A value carrier should participate in unqualified swap without paying
  // three move operations (or transiently creating moved-from sentinels).
  friend void swap(Status &a, Status &b) noexcept {
    using std::swap;
    swap(a.rep_, b.rep_);
  }

  // Keep the first failure while combining a sequence of independent
  // operations. A later error never overwrites an earlier one.
  void Update(const Status &status) noexcept {
    if (ok() && !status.ok())
      (void)(*this = status);
  }
  void Update(Status &&status) noexcept {
    if (ok() && !status.ok())
      (void)(*this = std::move(status));
  }

  // Make an intentionally discarded result visible at the call site. This is
  // preferable to a bare `(void)status`, especially at cleanup boundaries.
  void IgnoreError() const noexcept {}

private:
  friend class detail::StatusAccess;

  // Keep the common OK/no-op path header-inline; only a real error crosses the
  // shared-library seam and touches the heap representation.
  HYLE_ERROR_API Status
  WithContextSlow(std::string_view context) const & noexcept;
  HYLE_ERROR_API Status WithContextSlow(std::string_view context) && noexcept;

  static constexpr std::uintptr_t kInlineTag = 1;
  static constexpr std::uintptr_t kCodeShift = 2;
  static constexpr std::uintptr_t kMovedFromRep =
      (static_cast<std::uintptr_t>(StatusCode::Unknown) << kCodeShift) |
      kInlineTag;
  static constexpr std::uintptr_t kMaxInlineRep =
      (static_cast<std::uintptr_t>(StatusCode::Unauthenticated) << kCodeShift) |
      kInlineTag;

  // Every valid inline representation is <= kMaxInlineRep; process heap
  // addresses are above it. This single comparison covers OK (0), code-only
  // errors and heap reps without the two dependent tests needed by
  // `rep_ != 0 && (rep_ & 3) == 0` on AArch64.
  bool is_heap() const noexcept { return rep_ > kMaxInlineRep; }

  std::uintptr_t rep_ = 0; // 0 = OK
};

static_assert(sizeof(Status) == sizeof(std::uintptr_t),
              "Status must stay a single tagged word");

// Construct a Status from errno without making callers repeat strerror and
// canonical-code plumbing. `context` names the failed operation; the final
// message is "context: native description". errno==0 is OK and ignores
// context. The failure path performs one Status allocation; OOM preserves the
// mapped code and drops only the message, like every other Status factory.
HYLE_ERROR_API Status status_from_errno(int error_number,
                                        std::string_view context = {}) noexcept;

// Construct a Status from a Win32 GetLastError value. On Windows the message
// uses FormatMessage; other hosts retain a deterministic numeric description.
// The native Win32 code is always preserved for machine-readable recovery.
HYLE_ERROR_API Status status_from_win32(std::uint32_t error_number,
                                        std::string_view context = {}) noexcept;

// Construct a Status from a COM/WinRT HRESULT. The raw signed 32-bit value is
// preserved as NativeErrorDomain::HResult. Windows uses FormatMessage when it
// knows the value; other hosts retain a deterministic hexadecimal diagnostic.
HYLE_ERROR_API Status status_from_hresult(
    std::int32_t error_number, std::string_view context = {}) noexcept;

} // namespace hyle
