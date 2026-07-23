#include "hyle/error/status.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>

#include "hyle/error/result.h"
#include "hyle/macros.h"

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

namespace detail {
namespace {

// Heap representation for a status that carries a message. Allocated with a
// single malloc of sizeof(StatusRep) + len + 1 so the message bytes live inline
// right after the header (header-before-data, same idea as carray). Never
// exposed in a public header -- only reached through the functions below.
struct StatusRep {
  std::atomic<std::uint32_t> refcount;
  StatusCode code;
  std::uint32_t len;
  char *msg() noexcept { return reinterpret_cast<char *>(this + 1); }
  const char *msg() const noexcept {
    return reinterpret_cast<const char *>(this + 1);
  }
};

StatusRep *as_rep(std::uintptr_t rep) noexcept {
  return reinterpret_cast<StatusRep *>(rep);
}

}  // namespace

#if defined(HYLE_TESTING)
namespace {
// Test-only latch: next status_make_heap behaves as malloc-failure once cleared
// by the call (or sticky until disabled). Lets tests exercise the OOM degrade
// path without mocking the CRT.
std::atomic<bool> g_force_status_oom{false};
}  // namespace

void status_test_force_oom(bool enable) noexcept {
  g_force_status_oom.store(enable, std::memory_order_relaxed);
}
#endif

std::uintptr_t status_make_heap(StatusCode code, const char *msg,
                                std::size_t len) noexcept {
#if defined(HYLE_TESTING)
  if (g_force_status_oom.load(std::memory_order_relaxed)) {
    return 0;
  }
#endif
  if (len > std::numeric_limits<std::uint32_t>::max() ||
      len > std::numeric_limits<std::size_t>::max() - sizeof(StatusRep) - 1) {
    return 0;
  }
  void *mem = std::malloc(sizeof(StatusRep) + len + 1);
  // OOM while building an error message is operational, not a bug: return 0 so
  // the caller degrades to the message-less inline encoding (charter: OOM -> a
  // value, never abort). The error code survives; only the message is dropped.
  if (mem == nullptr) return 0;
  auto *rep = new (mem) StatusRep{};
  rep->refcount.store(1, std::memory_order_relaxed);
  rep->code = code;
  rep->len = static_cast<std::uint32_t>(len);
  if (len != 0) std::memcpy(rep->msg(), msg, len);
  rep->msg()[len] = '\0';
  auto bits = reinterpret_cast<std::uintptr_t>(rep);
  // Tag invariant: malloc must return >=8-byte aligned memory so the low 2 bits
  // stay clear (distinguishes a heap rep from the inline-code encoding). A
  // violation means a broken platform allocator -> fail-fast.
  HYLE_FATAL_ASSERT((bits & 3) == 0);
  return bits;
}

std::uintptr_t status_make_context(StatusCode code, const char *context,
                                   std::size_t context_len, const char *message,
                                   std::size_t message_len) noexcept {
#if defined(HYLE_TESTING)
  if (g_force_status_oom.load(std::memory_order_relaxed)) return 0;
#endif
  constexpr std::size_t kSeparatorLen = 2;
  const std::size_t separator_len = message_len == 0 ? 0 : kSeparatorLen;
  if (context_len > std::numeric_limits<std::uint32_t>::max() ||
      message_len > std::numeric_limits<std::uint32_t>::max() - context_len ||
      separator_len > std::numeric_limits<std::uint32_t>::max() - context_len -
                          message_len) {
    return 0;
  }
  const std::size_t total = context_len + separator_len + message_len;
  if (total > std::numeric_limits<std::size_t>::max() - sizeof(StatusRep) - 1) {
    return 0;
  }

  void *mem = std::malloc(sizeof(StatusRep) + total + 1);
  if (mem == nullptr) return 0;
  auto *rep = new (mem) StatusRep{};
  rep->refcount.store(1, std::memory_order_relaxed);
  rep->code = normalize_status_code(code);
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
  const auto bits = reinterpret_cast<std::uintptr_t>(rep);
  HYLE_FATAL_ASSERT((bits & 3) == 0);
  return bits;
}

void status_retain(std::uintptr_t rep) noexcept {
  as_rep(rep)->refcount.fetch_add(1, std::memory_order_relaxed);
}

void status_release(std::uintptr_t rep) noexcept {
  StatusRep *r = as_rep(rep);
  if (r->refcount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
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

}  // namespace detail

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

Status Status::WithContext(std::string_view context) const & noexcept {
  if (ok() || context.empty()) return *this;
  return WithContextSlow(context);
}

Status Status::WithContext(std::string_view context) && noexcept {
  if (ok() || context.empty()) return std::move(*this);
  return std::move(*this).WithContextSlow(context);
}

Status Status::WithContextSlow(std::string_view context) const & noexcept {
  const std::string_view current = message();
  const std::uintptr_t rep = detail::status_make_context(
      code(), context.data(), context.size(), current.data(), current.size());
  return rep == 0 ? *this : adopt_rep(rep);
}

Status Status::WithContextSlow(std::string_view context) && noexcept {
  const std::string_view current = message();
  const std::uintptr_t rep = detail::status_make_context(
      code(), context.data(), context.size(), current.data(), current.size());
  if (rep == 0) return std::move(*this);
  Status result = adopt_rep(rep);
  if (is_heap()) detail::status_release(rep_);
  rep_ = 0;
  return result;
}

}  // namespace hyle
