/* SPDX: same as HYLE.
 *
 * Pure C ABI for the HYLE core module.
 * Naming: every symbol is `hyle_*` / `HYLE_*`.
 * Docs (1:1 with this header): docs/guide/c-core.md
 */
#ifndef HYLE_C_CORE_H_
#define HYLE_C_CORE_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(HYLE_STATIC)
#  define HYLE_CORE_C_API
#elif defined(_WIN32) || defined(__CYGWIN__)
#  if defined(HYLE_CORE_BUILD)
#    define HYLE_CORE_C_API __declspec(dllexport)
#  else
#    define HYLE_CORE_C_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define HYLE_CORE_C_API __attribute__((visibility("default")))
#else
#  define HYLE_CORE_C_API
#endif

/* Packed version: (major << 16) | (minor << 8) | patch. Same value as the C++
 * hyle::version(). Lets a cross-language caller confirm which library version it
 * bound/dlopen'd -- the FFI counterpart of a diagnostic version query. */
HYLE_CORE_C_API uint32_t hyle_version(void);

/* Human-readable "major.minor.patch" (static storage; never freed). */
HYLE_CORE_C_API const char* hyle_version_string(void);

/* Decompose the packed version (header-inline; no DLL hop). */
static inline uint32_t hyle_version_major(uint32_t packed) {
  return (packed >> 16) & 0xFFu;
}
static inline uint32_t hyle_version_minor(uint32_t packed) {
  return (packed >> 8) & 0xFFu;
}
static inline uint32_t hyle_version_patch(uint32_t packed) {
  return packed & 0xFFu;
}

#ifdef __cplusplus
}
#endif

#endif /* HYLE_C_CORE_H_ */
