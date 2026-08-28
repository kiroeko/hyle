#pragma once

#include <cstddef>
#include <exception>
#include <string>
#include <typeinfo>
#include <utility>

#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_UTIL_API
#elif defined(HYLE_UTIL_BUILD)
#  define HYLE_UTIL_API HYLE_EXPORT
#else
#  define HYLE_UTIL_API HYLE_IMPORT
#endif

// Runtime symbol demangling (clay sys/demangle). Turns a mangled C++ symbol --
// e.g. "_Z1gv" -> "g()" -- into something human-readable, for diagnostics and
// backtraces. This is the RUNTIME complement to util/type_name<T> (compile-time,
// static type, no RTTI): demangle handles names known only at run time -- frames
// pulled from a backtrace, or typeid(x).name() on a polymorphic reference whose
// dynamic type isn't a compile-time T.
//
// Two improvements over clay:
//  * explicit ownership in both natural C++ shapes. Full text returns a
//    caller-owned std::string; bounded text writes into the caller's buffer,
//    snprintf-style. HYLE frees platform temporaries internally. Itanium
//    demangling still performs an internal malloc/free.
//  * MSVC actually demangles (UnDecorateSymbolName). clay's non-Itanium path just
//    echoed the mangled name back.

namespace hyle::util {

namespace detail {

// Compiled implementation seam for the header-inline owning convenience.
// The callback executes consumer code, so std::string storage is allocated by
// the caller's CRT rather than crossing a shared-library heap/layout boundary.
// This is not a supported Native FFI entry point.
using DemangleSink = bool (*)(void* context, const char* text,
                              std::size_t size) noexcept;
HYLE_UTIL_API bool demangle_visit_internal(const char* mangled, void* context,
                                           DemangleSink sink) noexcept;

}  // namespace detail

// Demangle `mangled` into `out` (capacity `cap`), always NUL-terminating when
// cap > 0. Returns the length the demangled name WOULD have (excluding the NUL),
// so a return >= cap signals truncation (snprintf semantics). If demangling fails
// (e.g. "main", or a name not in the platform's mangling scheme) the original
// `mangled` is copied through -- a sane default, never an error. A null `mangled`
// yields an empty string. `out` may be null only when cap == 0; violating that
// buffer contract fails fast. The caller never owns an allocation, but the
// platform demangler may allocate or serialize internally; this is a diagnostic
// cold-path API, not a compute hot-path or async-signal-safe primitive. Input
// and output may overlap.
HYLE_UTIL_API std::size_t demangle(const char* mangled, char* out,
                                   std::size_t cap) noexcept;

// Return the complete human-readable name as caller-owned standard storage.
// Unlike the buffer overload this convenience may throw when std::string
// allocation fails. The platform demangler still runs only once: an internal
// sink transfers its transient view into a string constructed in this TU,
// avoiding both a sizing pass and DLL-owned STL state.
[[nodiscard]] inline std::string demangle(const char* mangled) {
  struct SinkState {
    std::string value;
    std::exception_ptr failure;
  } state;

  const bool delivered = detail::demangle_visit_internal(
      mangled, &state,
      +[](void* context, const char* text, std::size_t size) noexcept {
        auto* sink = static_cast<SinkState*>(context);
        try {
          sink->value.assign(text, size);
          return true;
        } catch (...) {
          sink->failure = std::current_exception();
          return false;
        }
      });
  if (!delivered) {
    if (state.failure) std::rethrow_exception(state.failure);
    std::terminate();  // internal sink contract violation
  }
  return std::move(state.value);
}

[[nodiscard]] inline std::string demangle(const std::type_info& type) {
  return demangle(type.name());
}

// Natural C++ entry points for typeid(x) and fixed caller buffers. These inline
// shells preserve the compiled buffer ABI and snprintf semantics.
inline std::size_t demangle(const std::type_info& type, char* out,
                            std::size_t cap) noexcept {
  return demangle(type.name(), out, cap);
}

template <std::size_t N>
std::size_t demangle(const char* mangled, char (&out)[N]) noexcept {
  return demangle(mangled, out, N);
}

template <std::size_t N>
std::size_t demangle(const std::type_info& type, char (&out)[N]) noexcept {
  return demangle(type.name(), out, N);
}

}  // namespace hyle::util
