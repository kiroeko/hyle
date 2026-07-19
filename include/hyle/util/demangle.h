#pragma once

#include <cstddef>

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
//  * buffer-fill, no allocator on the API surface. clay returned a char* from an
//    Allocator (hidden ownership) and wrapped __cxa_demangle's malloc'd result in
//    a std::function-deleted unique_ptr (a heap closure for a free()). HYLE writes
//    into the caller's buffer, snprintf-style, and frees the libc temp internally.
//  * MSVC actually demangles (UnDecorateSymbolName). clay's non-Itanium path just
//    echoed the mangled name back.

namespace hyle::util {

// Demangle `mangled` into `out` (capacity `cap`), always NUL-terminating when
// cap > 0. Returns the length the demangled name WOULD have (excluding the NUL),
// so a return >= cap signals truncation (snprintf semantics). If demangling fails
// (e.g. "main", or a name not in the platform's mangling scheme) the original
// `mangled` is copied through -- a sane default, never an error. A null `mangled`
// yields an empty string. `out` may be null only when cap == 0; violating that
// buffer contract fails fast. No allocation is observable to the caller.
HYLE_UTIL_API std::size_t demangle(const char* mangled, char* out,
                                   std::size_t cap) noexcept;

}  // namespace hyle::util
