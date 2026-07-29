#include "hyle/error/status.h"

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>

#include "hyle/error/result.h"
#include "hyle/macros.h"
#include "status_access.h"

namespace hyle {

std::string_view status_code_name(StatusCode code) noexcept {
  switch (code) {
#define HYLE_STATUS_NAME_CASE(name, screaming, value) \
  case StatusCode::name:                              \
    return #name;
    HYLE_FOREACH_STATUS_CODE(HYLE_STATUS_NAME_CASE)
#undef HYLE_STATUS_NAME_CASE
  }
  return "Unknown";
}

std::string_view native_error_domain_name(NativeErrorDomain domain) noexcept {
  switch (domain) {
    case NativeErrorDomain::None: return "None";
    case NativeErrorDomain::PosixErrno: return "PosixErrno";
    case NativeErrorDomain::Win32: return "Win32";
    case NativeErrorDomain::Vulkan: return "Vulkan";
    case NativeErrorDomain::CudaRuntime: return "CudaRuntime";
    case NativeErrorDomain::CudaDriver: return "CudaDriver";
    case NativeErrorDomain::Nvrtc: return "Nvrtc";
    case NativeErrorDomain::Unknown: return "Unknown";
  }
  return "Unknown";
}

StatusCode status_code_from_errno(int error_number) noexcept {
  // Keep this mapping aligned with Abseil's mature ErrnoToStatusCode taxonomy.
  // Guard less-portable errno constants so the same TU compiles with the MSVC
  // CRT as well as glibc/libSystem. Do not list alias pairs such as
  // EWOULDBLOCK/EAGAIN twice: many platforms give them the same value.
  switch (error_number) {
    case 0: return StatusCode::Ok;
    case EINVAL:
    case ENAMETOOLONG:
    case E2BIG:
    case EDOM:
    case EFAULT:
    case EILSEQ:
    case ENOTTY:
    case ESPIPE:
#ifdef EDESTADDRREQ
    case EDESTADDRREQ:
#endif
#ifdef ENOPROTOOPT
    case ENOPROTOOPT:
#endif
#ifdef ENOTSOCK
    case ENOTSOCK:
#endif
#ifdef EPROTOTYPE
    case EPROTOTYPE:
#endif
      return StatusCode::InvalidArgument;
#ifdef ETIMEDOUT
    case ETIMEDOUT: return StatusCode::DeadlineExceeded;
#endif
    case ENODEV:
    case ENOENT:
    case ENXIO:
    case ESRCH:
#ifdef ENOMEDIUM
    case ENOMEDIUM:
#endif
      return StatusCode::NotFound;
    case EEXIST:
#ifdef EADDRNOTAVAIL
    case EADDRNOTAVAIL:
#endif
#ifdef EALREADY
    case EALREADY:
#endif
#ifdef ENOTUNIQ
    case ENOTUNIQ:
#endif
      return StatusCode::AlreadyExists;
    case EPERM:
    case EACCES:
    case EROFS:
#ifdef ENOKEY
    case ENOKEY:
#endif
      return StatusCode::PermissionDenied;
    case ENOTEMPTY:
    case EISDIR:
    case ENOTDIR:
    case EBADF:
    case EBUSY:
    case ECHILD:
    case EPIPE:
#ifdef EADDRINUSE
    case EADDRINUSE:
#endif
#ifdef EBADFD
    case EBADFD:
#endif
#ifdef EISCONN
    case EISCONN:
#endif
#ifdef EISNAM
    case EISNAM:
#endif
#ifdef ENOTBLK
    case ENOTBLK:
#endif
#ifdef ENOTCONN
    case ENOTCONN:
#endif
#ifdef ESHUTDOWN
    case ESHUTDOWN:
#endif
    case ETXTBSY:
#ifdef EUNATCH
    case EUNATCH:
#endif
      return StatusCode::FailedPrecondition;
    case ENOSPC:
    case EMFILE:
    case EMLINK:
    case ENFILE:
    case ENOMEM:
#ifdef EDQUOT
    case EDQUOT:
#endif
#ifdef ENOBUFS
    case ENOBUFS:
#endif
#ifdef EUSERS
    case EUSERS:
#endif
      return StatusCode::ResourceExhausted;
    case EFBIG:
    case EOVERFLOW:
    case ERANGE:
#ifdef ECHRNG
    case ECHRNG:
#endif
      return StatusCode::OutOfRange;
    case ENOSYS:
    case EXDEV:
#ifdef ENOTSUP
    case ENOTSUP:
#endif
#ifdef EAFNOSUPPORT
    case EAFNOSUPPORT:
#endif
#ifdef ENOPKG
    case ENOPKG:
#endif
#ifdef EPFNOSUPPORT
    case EPFNOSUPPORT:
#endif
#ifdef EPROTONOSUPPORT
    case EPROTONOSUPPORT:
#endif
#ifdef ESOCKTNOSUPPORT
    case ESOCKTNOSUPPORT:
#endif
      return StatusCode::Unimplemented;
    case EAGAIN:
    case EINTR:
    case ENOLCK:
#ifdef ECOMM
    case ECOMM:
#endif
#ifdef ECONNREFUSED
    case ECONNREFUSED:
#endif
#ifdef ECONNABORTED
    case ECONNABORTED:
#endif
#ifdef ECONNRESET
    case ECONNRESET:
#endif
#ifdef EHOSTDOWN
    case EHOSTDOWN:
#endif
#ifdef EHOSTUNREACH
    case EHOSTUNREACH:
#endif
#ifdef ENETDOWN
    case ENETDOWN:
#endif
#ifdef ENETRESET
    case ENETRESET:
#endif
#ifdef ENETUNREACH
    case ENETUNREACH:
#endif
#ifdef ENOLINK
    case ENOLINK:
#endif
#ifdef ENONET
    case ENONET:
#endif
      return StatusCode::Unavailable;
    case EDEADLK:
#ifdef ESTALE
    case ESTALE:
#endif
      return StatusCode::Aborted;
#ifdef ECANCELED
    case ECANCELED: return StatusCode::Cancelled;
#endif
    default: return StatusCode::Unknown;
  }
}

namespace detail {
namespace {

// Heap representation for a status that carries a message. Allocated with a
// single malloc of sizeof(StatusRep) + len + 1 so the message bytes live inline
// right after the header (header-before-data, same idea as carray). Never
// exposed in a public header -- only reached through the functions below.
struct StatusRep {
  StatusRepHeader header;
  StatusCode code;
  NativeErrorDomain native_domain;
  std::uint32_t len;
  char *msg() noexcept { return reinterpret_cast<char *>(this + 1); }
  const char *msg() const noexcept {
    return reinterpret_cast<const char *>(this + 1);
  }
};
static_assert(sizeof(StatusRep) == 12,
              "native domain must consume the StatusRep's former padding");

static_assert(offsetof(StatusRep, header) == 0,
              "the header-inline retain path requires a stable prefix");

StatusRep *as_rep(std::uintptr_t rep) noexcept {
  return reinterpret_cast<StatusRep *>(rep);
}

#if defined(HYLE_TESTING)
// Test-only latch. Keeping it sticky makes a multi-allocation operation such as
// context composition deterministic until the test explicitly disables it.
std::atomic<bool> g_force_status_oom{false};
#endif

bool force_status_oom() noexcept {
#if defined(HYLE_TESTING)
  return g_force_status_oom.load(std::memory_order_relaxed);
#else
  return false;
#endif
}

std::uintptr_t make_status_rep(StatusCode code, NativeError native,
                               const char *msg, std::size_t len) noexcept {
  if (force_status_oom()) return 0;
  native.domain = normalize_native_error_domain(native.domain);
  const std::size_t detail_size =
      native.present() ? sizeof(native.code) : std::size_t{0};
  if (len > std::numeric_limits<std::uint32_t>::max() ||
      detail_size > std::numeric_limits<std::size_t>::max() -
                        sizeof(StatusRep) - 1 ||
      len > std::numeric_limits<std::size_t>::max() - sizeof(StatusRep) - 1 -
                detail_size) {
    return 0;
  }
  void *mem = std::malloc(sizeof(StatusRep) + len + 1 + detail_size);
  if (mem == nullptr) return 0;
  auto *rep = new (mem) StatusRep{};
  rep->header.refcount.store(1, std::memory_order_relaxed);
  rep->code = normalize_status_code(code);
  rep->native_domain = native.domain;
  rep->len = static_cast<std::uint32_t>(len);
  if (len != 0) std::memcpy(rep->msg(), msg, len);
  rep->msg()[len] = '\0';
  if (native.present()) {
    std::memcpy(rep->msg() + len + 1, &native.code, sizeof(native.code));
  }
  const auto bits = reinterpret_cast<std::uintptr_t>(rep);
  HYLE_FATAL_ASSERT((bits & 3) == 0);
  return bits;
}

}  // namespace

#if defined(HYLE_TESTING)
void status_test_force_oom(bool enable) noexcept {
  g_force_status_oom.store(enable, std::memory_order_relaxed);
}
#endif

std::uintptr_t status_make_heap(StatusCode code, const char *msg,
                                std::size_t len) noexcept {
  return make_status_rep(code, NativeError{}, msg, len);
}

std::uintptr_t status_make_native_heap(StatusCode code, NativeError native,
                                       const char *msg,
                                       std::size_t len) noexcept {
  return make_status_rep(code, native, msg, len);
}

std::uintptr_t status_make_context(StatusCode code, NativeError native,
                                   const char *context,
                                   std::size_t context_len, const char *message,
                                   std::size_t message_len) noexcept {
  if (force_status_oom()) return 0;
  native.domain = normalize_native_error_domain(native.domain);
  constexpr std::size_t kSeparatorLen = 2;
  const std::size_t separator_len = message_len == 0 ? 0 : kSeparatorLen;
  if (context_len > std::numeric_limits<std::uint32_t>::max() ||
      message_len > std::numeric_limits<std::uint32_t>::max() - context_len ||
      separator_len > std::numeric_limits<std::uint32_t>::max() - context_len -
                          message_len) {
    return 0;
  }
  const std::size_t total = context_len + separator_len + message_len;
  const std::size_t detail_size =
      native.present() ? sizeof(native.code) : std::size_t{0};
  if (detail_size > std::numeric_limits<std::size_t>::max() -
                        sizeof(StatusRep) - 1 ||
      total > std::numeric_limits<std::size_t>::max() - sizeof(StatusRep) - 1 -
                  detail_size) {
    return 0;
  }

  void *mem = std::malloc(sizeof(StatusRep) + total + 1 + detail_size);
  if (mem == nullptr) return 0;
  auto *rep = new (mem) StatusRep{};
  rep->header.refcount.store(1, std::memory_order_relaxed);
  rep->code = normalize_status_code(code);
  rep->native_domain = native.domain;
  rep->len = static_cast<std::uint32_t>(total);
  char *out = rep->msg();
  std::memcpy(out, context, context_len);
  if (separator_len != 0) {
    out[context_len] = ':';
    out[context_len + 1] = ' ';
  }
  if (message_len != 0) {
    std::memcpy(out + context_len + separator_len, message, message_len);
  }
  out[total] = '\0';
  if (native.present()) {
    std::memcpy(out + total + 1, &native.code, sizeof(native.code));
  }
  const auto bits = reinterpret_cast<std::uintptr_t>(rep);
  HYLE_FATAL_ASSERT((bits & 3) == 0);
  return bits;
}

void status_release(std::uintptr_t rep) noexcept {
  StatusRep *r = as_rep(rep);
  if (r->header.refcount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    r->~StatusRep();
    std::free(r);
  }
}

StatusCode status_heap_code(std::uintptr_t rep) noexcept {
  return as_rep(rep)->code;
}

std::string_view status_heap_message(std::uintptr_t rep) noexcept {
  const StatusRep *r = as_rep(rep);
  return std::string_view(r->msg(), r->len);
}

NativeError status_heap_native_error(std::uintptr_t rep) noexcept {
  const StatusRep *r = as_rep(rep);
  NativeError native{r->native_domain, 0};
  if (native.present()) {
    std::memcpy(&native.code, r->msg() + r->len + 1, sizeof(native.code));
  }
  return native;
}

}  // namespace detail

namespace {

[[maybe_unused]] const char *errno_text(int result, char *buffer) noexcept {
  return result == 0 ? buffer : nullptr;
}

[[maybe_unused]] const char *errno_text(char *result, char *) noexcept {
  return result;
}

std::string_view describe_errno(int error_number, char *buffer,
                                std::size_t cap) noexcept {
#if defined(_WIN32)
  if (::strerror_s(buffer, cap, error_number) == 0 && buffer[0] != '\0') {
    return buffer;
  }
#else
  const char *text = errno_text(
      ::strerror_r(error_number, buffer, cap), buffer);
  if (text != nullptr && text[0] != '\0') return text;
#endif
  const int written = std::snprintf(buffer, cap, "errno %d", error_number);
  if (written <= 0) return "unknown errno";
  const std::size_t length = static_cast<std::size_t>(written);
  return std::string_view(buffer, length < cap ? length : cap - 1);
}

}  // namespace

Status status_from_errno(int error_number, std::string_view context) noexcept {
  const StatusCode code = status_code_from_errno(error_number);
  if (code == StatusCode::Ok) return Status{};

  // strerror[_r] writes the diagnostic on success. Initializing only the first
  // byte avoids clearing the full scratch buffer on every native error while
  // preserving a deterministic fallback on CRT failure.
  char description[256];
  description[0] = '\0';
  const std::string_view native =
      describe_errno(error_number, description, sizeof(description));
  const NativeError native_error{NativeErrorDomain::PosixErrno,
                                 error_number};
  std::uintptr_t rep = 0;
  if (context.empty()) {
    rep = detail::status_make_native_heap(code, native_error, native.data(),
                                          native.size());
  } else {
    rep = detail::status_make_context(code, native_error, context.data(),
                                      context.size(), native.data(),
                                      native.size());
  }
  return rep != 0 ? detail::StatusAccess::adopt(rep) : Status(code);
}

std::size_t Status::ToString(char *buf, std::size_t cap) const noexcept {
  if (buf == nullptr) cap = 0;
  // Build "OK" | "CODE" | "CODE: message" by appending pieces, tracking the
  // total length needed regardless of cap (snprintf contract). A local appender
  // copies only what fits; `need` keeps counting past the buffer end.
  std::size_t need = 0;
  auto append = [&](std::string_view s) noexcept {
    if (need < cap) {
      // Leave room for the terminator. Block copies matter for long native
      // driver messages and avoid paying a branch for every output byte.
      const std::size_t available = cap - need - 1;
      const std::size_t copied = s.size() < available ? s.size() : available;
      if (copied != 0) std::memcpy(buf + need, s.data(), copied);
    }
    need += s.size();
  };

  if (ok()) {
    append("OK");
  } else {
    append(status_code_name(code()));
    std::string_view m = message();
    if (!m.empty()) {
      append(": ");
      append(m);
    }
  }
  if (cap > 0) buf[need < cap ? need : cap - 1] = '\0';
  return need;
}

#if !defined(HYLE_STATIC)
Status Status::WithContext(std::string_view context) const & noexcept {
  if (ok() || context.empty()) return *this;
  return WithContextSlow(context);
}

Status Status::WithContext(std::string_view context) && noexcept {
  if (ok() || context.empty()) return std::move(*this);
  return std::move(*this).WithContextSlow(context);
}
#endif

Status Status::WithContextSlow(std::string_view context) const & noexcept {
  const std::string_view current = message();
  const std::uintptr_t rep = detail::status_make_context(
      code(), native_error(), context.data(), context.size(), current.data(),
      current.size());
  return rep == 0 ? *this : detail::StatusAccess::adopt(rep);
}

Status Status::WithContextSlow(std::string_view context) && noexcept {
  const std::string_view current = message();
  const std::uintptr_t rep = detail::status_make_context(
      code(), native_error(), context.data(), context.size(), current.data(),
      current.size());
  if (rep == 0) return std::move(*this);
  Status result = detail::StatusAccess::adopt(rep);
  if (is_heap()) detail::status_release(rep_);
  rep_ = 0;
  return result;
}

}  // namespace hyle
