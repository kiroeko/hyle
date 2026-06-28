#include "hyle/util/demangle.h"

#include <cstdlib>
#include <cstring>

#if defined(__GNUC__) || defined(__clang__)
#  if __has_include(<cxxabi.h>)
#    include <cxxabi.h>
#    define HYLE_HAS_CXA_DEMANGLE 1
#  endif
#elif defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <dbghelp.h>  // UnDecorateSymbolName
#  pragma comment(lib, "dbghelp.lib")
#  include <mutex>
#endif

namespace hyle::util {

namespace {
// Copy a NUL-terminated source into out[cap], snprintf-style: always terminate
// when cap > 0, return the full source length so the caller can detect overflow.
std::size_t fill(const char* src, char* out, std::size_t cap) noexcept {
  std::size_t n = std::strlen(src);
  if (cap > 0) {
    std::size_t copy = (n < cap - 1) ? n : cap - 1;
    std::memcpy(out, src, copy);
    out[copy] = '\0';
  }
  return n;
}
}  // namespace

std::size_t demangle(const char* mangled, char* out, std::size_t cap) noexcept {
  if (mangled == nullptr) return fill("", out, cap);

#if defined(HYLE_HAS_CXA_DEMANGLE)
  int status = -1;
  // __cxa_demangle mallocs the result; the ABI gives no in-place buffer variant
  // that avoids it, so we let it allocate and free here -- the malloc is not
  // observable on our API surface (no Allocator, no closure deleter).
  char* d = abi::__cxa_demangle(mangled, nullptr, nullptr, &status);
  if (status == 0 && d != nullptr) {
    std::size_t n = fill(d, out, cap);
    std::free(d);
    return n;
  }
  std::free(d);  // null on failure; free(null) is a no-op
  // Demangling fails for names outside the Itanium scheme ("main", "clone"):
  // the mangled name is itself a fine default.
  return fill(mangled, out, cap);

#elif defined(_WIN32)
  // DbgHelp is documented as single-threaded; serialize. Not a hot path.
  static std::mutex m;
  std::lock_guard<std::mutex> lock(m);
  char buf[1024];
  // UNDNAME_COMPLETE: full signature. Note MSVC typeid().name() already yields a
  // readable "class Foo" form (not a mangled "?..." symbol); UnDecorateSymbolName
  // passes those through unchanged, and demangles real decorated linker symbols
  // (e.g. from a backtrace). Returns 0 on failure.
  DWORD n = UnDecorateSymbolName(mangled, buf, sizeof(buf), UNDNAME_COMPLETE);
  if (n != 0) return fill(buf, out, cap);
  return fill(mangled, out, cap);

#else
  return fill(mangled, out, cap);  // unknown toolchain: pass through (safe)
#endif
}

}  // namespace hyle::util
