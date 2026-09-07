#include "hyle/util/demangle.h"

#include <cstdlib>
#include <cstring>
#include <memory>

#include "hyle/core/compiler.h"

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <dbghelp.h>  // UnDecorateSymbolName
#  pragma comment(lib, "dbghelp.lib")
#  include <mutex>
#elif defined(__GNUC__) || defined(__clang__)
#  if __has_include(<cxxabi.h>)
#    include <cxxabi.h>
#    define HYLE_HAS_CXA_DEMANGLE 1
#  endif
#endif

namespace hyle::util {

namespace {
// Copy source into out[cap], snprintf-style: always terminate when cap > 0,
// return the full source length so the caller can detect overflow. memmove
// makes exact and partial input/output overlap a supported cold-path shape.
std::size_t fill(const char* src, std::size_t size, char* out,
                 std::size_t cap) noexcept {
  HYLE_FATAL_ASSERT(out != nullptr || cap == 0);
  if (cap > 0) {
    const std::size_t copy = (size < cap - 1) ? size : cap - 1;
    std::memmove(out, src, copy);
    out[copy] = '\0';
  }
  return size;
}

struct CFree {
  void operator()(char* value) const noexcept { std::free(value); }
};
using MallocString = std::unique_ptr<char, CFree>;

#if defined(_WIN32)
DWORD undecorate_once(const char* mangled, char* out, DWORD capacity) {
  // DbgHelp is documented as single-threaded. Keep only the DbgHelp call under
  // this runtime image's lock; allocation and sink/string copies need not
  // serialize. Other DbgHelp users (including separate static HYLE copies) need
  // host-level coordination covering all calls into the same DbgHelp instance.
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  return UnDecorateSymbolName(mangled, out, capacity, UNDNAME_COMPLETE);
}
#endif

// Present exactly one transient, NUL-terminated final spelling to `sink`.
// Platform failure is not an error: the original input is the final spelling.
// Windows may make several DbgHelp attempts only because that API does not
// report the required output length.
template <class Sink>
auto visit_demangled(const char* mangled, Sink&& sink) {
  if (mangled == nullptr) return sink("", 0);

#if defined(HYLE_HAS_CXA_DEMANGLE)
  int status = -1;
  MallocString demangled(
      abi::__cxa_demangle(mangled, nullptr, nullptr, &status));
  if (status == 0 && demangled != nullptr) {
    return sink(demangled.get(), std::strlen(demangled.get()));
  }
  return sink(mangled, std::strlen(mangled));

#elif defined(_WIN32)
  char buffer[1024];
  DWORD size = undecorate_once(mangled, buffer, sizeof(buffer));
  if (size != 0) return sink(buffer, static_cast<std::size_t>(size));

  // A valid decorated name can exceed the small stack buffer. DbgHelp does not
  // report the required length, so retry only the '?' form used by MSVC linker
  // symbols, growing a temporary buffer to a defensive 1 MiB ceiling.
  if (mangled[0] == '?') {
    for (std::size_t capacity = 2048; capacity <= 1024 * 1024;
         capacity *= 2) {
      MallocString dynamic(static_cast<char*>(std::malloc(capacity)));
      if (dynamic == nullptr) break;
      size = undecorate_once(mangled, dynamic.get(),
                             static_cast<DWORD>(capacity));
      if (size != 0) {
        return sink(dynamic.get(), static_cast<std::size_t>(size));
      }
    }
  }
  return sink(mangled, std::strlen(mangled));

#else
  return sink(mangled, std::strlen(mangled));
#endif
}
}  // namespace

std::size_t demangle(const char* mangled, char* out, std::size_t cap) noexcept {
  return visit_demangled(mangled, [&](const char* text, std::size_t size) {
    return fill(text, size, out, cap);
  });
}

bool detail::demangle_visit_internal(const char* mangled, void* context,
                                     detail::DemangleSink sink) noexcept {
  HYLE_FATAL_ASSERT(sink != nullptr);
  return visit_demangled(mangled, [&](const char* text, std::size_t size) {
    return sink(context, text, size);
  });
}

}  // namespace hyle::util
