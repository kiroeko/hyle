#pragma once

#include <cstddef>
#include <typeinfo>

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
//  * buffer-fill, no allocator ownership on the API surface. clay returned a
//    char* from an Allocator (hidden ownership) and wrapped __cxa_demangle's
//    malloc'd result in a std::function-deleted unique_ptr (a heap closure for a
//    free()). HYLE writes into the caller's buffer, snprintf-style, and frees
//    platform temporaries internally. Itanium demangling still performs an
//    internal malloc/free.
//  * MSVC actually demangles (UnDecorateSymbolName). clay's non-Itanium path just
//    echoed the mangled name back.

namespace hyle::util {

// Demangle `mangled` into `out` (capacity `cap`), always NUL-terminating when
// cap > 0. Returns the length the demangled name WOULD have (excluding the NUL),
// so a return >= cap signals truncation (snprintf semantics). If demangling fails
// (e.g. "main", or a name not in the platform's mangling scheme) the original
// `mangled` is copied through -- a sane default, never an error. A null `mangled`
// yields an empty string. `out` may be null only when cap == 0; violating that
// buffer contract fails fast. The caller never owns an allocation, but the
// platform demangler may allocate or serialize internally; this is a diagnostic
// cold-path API, not a compute hot-path primitive.
HYLE_UTIL_API std::size_t demangle(const char* mangled, char* out,
                                   std::size_t cap) noexcept;

// Natural C++ entry points for typeid(x) and fixed caller buffers. These inline
// shells preserve the single compiled ABI symbol and snprintf semantics.
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
