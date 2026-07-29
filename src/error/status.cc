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

// Ordinary message/detail errors keep the compact exact-size representation:
// one allocation of sizeof(StatusRep) + len + 1 + optional native code. A
// status with one semantic context stays exact-size but carries an internal
// history flag. A second rvalue prepend promotes it to ContextStatusRep, which
// reserves geometrically grown prefix space; later unique rvalues prepend in
// place, while shared/lvalue errors use COW.
//
// This distinction is deliberate: the normal Status shape and its 12-byte heap
// header pay no capacity metadata, while progressive error propagation has
// amortized O(final message bytes) copying instead of rebuilding the complete
// message at every stack layer.
struct StatusRep {
  StatusRepHeader header;
  std::uint16_t code_and_flags;
  NativeErrorDomain native_domain;
  std::uint32_t len;
};
static_assert(sizeof(StatusRep) == 12,
              "native domain must consume the StatusRep's former padding");

static_assert(offsetof(StatusRep, header) == 0,
              "the header-inline retain path requires a stable prefix");

struct ContextStatusRep {
  StatusRep base;
  std::uint32_t capacity;
  std::uint32_t offset;
};
static_assert(sizeof(ContextStatusRep) == 20,
              "context metadata should not acquire hidden padding");

constexpr std::uint16_t kContextRepFlag = std::uint16_t{1} << 15;
constexpr std::uint16_t kFlatContextFlag = std::uint16_t{1} << 14;
constexpr std::uint16_t kRepFlags = kContextRepFlag | kFlatContextFlag;
constexpr std::uint16_t kCodeMask =
    static_cast<std::uint16_t>(~kRepFlags);
constexpr std::size_t kSeparatorLen = 2;

bool is_context_rep(const StatusRep *rep) noexcept {
  return (rep->code_and_flags & kContextRepFlag) != 0;
}

StatusCode rep_code(const StatusRep *rep) noexcept {
  return static_cast<StatusCode>(rep->code_and_flags & kCodeMask);
}

std::size_t native_detail_size(const StatusRep *rep) noexcept {
  return rep->native_domain == NativeErrorDomain::None
             ? std::size_t{0}
             : sizeof(std::int64_t);
}

char *ordinary_message(StatusRep *rep) noexcept {
  return reinterpret_cast<char *>(rep + 1);
}

const char *ordinary_message(const StatusRep *rep) noexcept {
  return reinterpret_cast<const char *>(rep + 1);
}

char *context_storage(ContextStatusRep *rep) noexcept {
  return reinterpret_cast<char *>(rep + 1) + native_detail_size(&rep->base);
}

const char *context_storage(const ContextStatusRep *rep) noexcept {
  return reinterpret_cast<const char *>(rep + 1) +
         native_detail_size(&rep->base);
}

char *rep_message(StatusRep *rep) noexcept {
  if (!is_context_rep(rep)) return ordinary_message(rep);
  auto *context_rep = reinterpret_cast<ContextStatusRep *>(rep);
  return context_storage(context_rep) + context_rep->offset;
}

const char *rep_message(const StatusRep *rep) noexcept {
  if (!is_context_rep(rep)) return ordinary_message(rep);
  const auto *context_rep = reinterpret_cast<const ContextStatusRep *>(rep);
  return context_storage(context_rep) + context_rep->offset;
}

char *native_detail(StatusRep *rep) noexcept {
  if (is_context_rep(rep)) {
    return reinterpret_cast<char *>(
        reinterpret_cast<ContextStatusRep *>(rep) + 1);
  }
  return ordinary_message(rep) + rep->len + 1;
}

const char *native_detail(const StatusRep *rep) noexcept {
  if (is_context_rep(rep)) {
    return reinterpret_cast<const char *>(
        reinterpret_cast<const ContextStatusRep *>(rep) + 1);
  }
  return ordinary_message(rep) + rep->len + 1;
}

StatusRep *as_rep(std::uintptr_t rep) noexcept {
  return reinterpret_cast<StatusRep *>(rep);
}

bool has_context_history(std::uintptr_t bits) noexcept {
  return (as_rep(bits)->code_and_flags & kRepFlags) != 0;
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
  rep->code_and_flags =
      static_cast<std::uint16_t>(normalize_status_code(code));
  rep->native_domain = native.domain;
  rep->len = static_cast<std::uint32_t>(len);
  char *out = ordinary_message(rep);
  if (len != 0) std::memcpy(out, msg, len);
  out[len] = '\0';
  if (native.present()) {
    std::memcpy(out + len + 1, &native.code, sizeof(native.code));
  }
  const auto bits = reinterpret_cast<std::uintptr_t>(rep);
  HYLE_FATAL_ASSERT((bits & 3) == 0);
  return bits;
}

std::uint32_t grow_context_capacity(std::size_t required,
                                    std::uint32_t current = 0) noexcept {
  constexpr std::uint64_t kMax =
      std::numeric_limits<std::uint32_t>::max();
  if (required > kMax) return 0;

  std::uint64_t capacity = current == 0 ? required : current;
  while (capacity < required) {
    const std::uint64_t growth = capacity / 2 > 32 ? capacity / 2 : 32;
    if (capacity > kMax - growth) {
      capacity = kMax;
      break;
    }
    capacity += growth;
  }
  return capacity >= required ? static_cast<std::uint32_t>(capacity) : 0;
}

HYLE_FORCE_INLINE std::uintptr_t make_flat_context_rep(
    StatusCode code, NativeError native, const char *context,
    std::size_t context_len, const char *message,
    std::size_t message_len) noexcept {
  if (force_status_oom()) return 0;
  native.domain = normalize_native_error_domain(native.domain);
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
  rep->code_and_flags =
      static_cast<std::uint16_t>(normalize_status_code(code)) |
      kFlatContextFlag;
  rep->native_domain = native.domain;
  rep->len = static_cast<std::uint32_t>(total);
  char *out = ordinary_message(rep);
  if (context_len != 0) std::memcpy(out, context, context_len);
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

std::uintptr_t make_context_rep(StatusCode code, NativeError native,
                                const char *context, std::size_t context_len,
                                const char *message, std::size_t message_len,
                                std::uint32_t current_capacity = 0) noexcept {
  if (force_status_oom()) return 0;
  native.domain = normalize_native_error_domain(native.domain);
  const std::size_t separator_len = message_len == 0 ? 0 : kSeparatorLen;
  if (context_len > std::numeric_limits<std::uint32_t>::max() ||
      message_len > std::numeric_limits<std::uint32_t>::max() - context_len ||
      separator_len > std::numeric_limits<std::uint32_t>::max() - context_len -
                          message_len) {
    return 0;
  }
  const std::size_t total = context_len + separator_len + message_len;
  const std::uint32_t capacity =
      grow_context_capacity(total, current_capacity);
  if (capacity == 0) return 0;

  const std::size_t detail_size =
      native.present() ? sizeof(native.code) : std::size_t{0};
  constexpr std::size_t kTerminatorSize = 1;
  if (detail_size > std::numeric_limits<std::size_t>::max() -
                        sizeof(ContextStatusRep) - kTerminatorSize ||
      capacity > std::numeric_limits<std::size_t>::max() -
                     sizeof(ContextStatusRep) - detail_size -
                     kTerminatorSize) {
    return 0;
  }

  void *mem = std::malloc(sizeof(ContextStatusRep) + detail_size + capacity +
                          kTerminatorSize);
  if (mem == nullptr) return 0;
  auto *rep = new (mem) ContextStatusRep{};
  rep->base.header.refcount.store(1, std::memory_order_relaxed);
  rep->base.code_and_flags =
      static_cast<std::uint16_t>(normalize_status_code(code)) |
      kContextRepFlag;
  rep->base.native_domain = native.domain;
  rep->base.len = static_cast<std::uint32_t>(total);
  rep->capacity = capacity;
  rep->offset = capacity - static_cast<std::uint32_t>(total);
  if (native.present()) {
    std::memcpy(reinterpret_cast<char *>(rep + 1), &native.code,
                sizeof(native.code));
  }

  char *out = context_storage(rep) + rep->offset;
  if (context_len != 0) std::memcpy(out, context, context_len);
  if (separator_len != 0) {
    out[context_len] = ':';
    out[context_len + 1] = ' ';
  }
  if (message_len != 0) {
    std::memcpy(out + context_len + separator_len, message, message_len);
  }
  out[total] = '\0';

  const auto bits = reinterpret_cast<std::uintptr_t>(rep);
  HYLE_FATAL_ASSERT((bits & 3) == 0);
  return bits;
}

struct PrependContextResult {
  std::uintptr_t rep;
  bool succeeded;
};

PrependContextResult prepend_context_owned(std::uintptr_t bits,
                                           const char *context,
                                           std::size_t context_len) noexcept {
  StatusRep *base = as_rep(bits);
  const std::size_t message_len = base->len;
  const std::size_t separator_len = message_len == 0 ? 0 : kSeparatorLen;
  if (context_len > std::numeric_limits<std::uint32_t>::max() ||
      message_len > std::numeric_limits<std::uint32_t>::max() - context_len ||
      separator_len > std::numeric_limits<std::uint32_t>::max() - context_len -
                          message_len) {
    return {bits, false};
  }
  const std::size_t added = context_len + separator_len;

  if (is_context_rep(base) &&
      base->header.refcount.load(std::memory_order_acquire) == 1) {
    auto *rep = reinterpret_cast<ContextStatusRep *>(base);
    if (rep->offset >= added) {
      const std::uint32_t new_offset =
          rep->offset - static_cast<std::uint32_t>(added);
      char *out = context_storage(rep) + new_offset;
      // The context may be a view into the current message. It still cannot
      // overlap this destination: all new bytes end exactly at old offset.
      if (context_len != 0) std::memcpy(out, context, context_len);
      if (separator_len != 0) {
        out[context_len] = ':';
        out[context_len + 1] = ' ';
      }
      rep->offset = new_offset;
      rep->base.len += static_cast<std::uint32_t>(added);
      return {bits, true};
    }
  }

  std::uint32_t current_capacity = 0;
  if (is_context_rep(base)) {
    current_capacity =
        reinterpret_cast<ContextStatusRep *>(base)->capacity;
  }
  NativeError native{base->native_domain, 0};
  if (native.present()) {
    std::memcpy(&native.code, native_detail(base), sizeof(native.code));
  }
  // The caller routes a first semantic layer through the exact-size flat path.
  // Reaching here therefore establishes a progressive chain. Seed growth from
  // the exact first-layer length so this allocation leaves useful prefix space
  // without charging the common one-layer case.
  if (current_capacity == 0) current_capacity = base->len;
  const std::uintptr_t replacement = make_context_rep(
      rep_code(base), native, context, context_len, rep_message(base),
      message_len, current_capacity);
  return {replacement == 0 ? bits : replacement, replacement != 0};
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
  return make_flat_context_rep(code, native, context, context_len, message,
                               message_len);
}

void status_release(std::uintptr_t rep) noexcept {
  StatusRep *r = as_rep(rep);
  if (r->header.refcount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    if (is_context_rep(r)) {
      reinterpret_cast<ContextStatusRep *>(r)->~ContextStatusRep();
    } else {
      r->~StatusRep();
    }
    std::free(r);
  }
}

StatusCode status_heap_code(std::uintptr_t rep) noexcept {
  return rep_code(as_rep(rep));
}

std::string_view status_heap_message(std::uintptr_t rep) noexcept {
  const StatusRep *r = as_rep(rep);
  return std::string_view(rep_message(r), r->len);
}

NativeError status_heap_native_error(std::uintptr_t rep) noexcept {
  const StatusRep *r = as_rep(rep);
  NativeError native{r->native_domain, 0};
  if (native.present()) {
    std::memcpy(&native.code, native_detail(r), sizeof(native.code));
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
  if (is_heap() && detail::has_context_history(rep_)) {
    const std::uintptr_t old_rep = rep_;
    const auto prepended = detail::prepend_context_owned(
        old_rep, context.data(), context.size());
    if (!prepended.succeeded) return std::move(*this);
    if (prepended.rep == old_rep) {
      Status result = detail::StatusAccess::adopt(old_rep);
      rep_ = 0;
      return result;
    }
    Status result = detail::StatusAccess::adopt(prepended.rep);
    detail::status_release(old_rep);
    rep_ = 0;
    return result;
  }

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
