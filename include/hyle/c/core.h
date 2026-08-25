/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary
 *
 * Stable Native FFI ABI for the HYLE core module. This header is the schema
 * consumed by binding generators and foreign runtimes; HYLE does not maintain
 * a separate source-level C product API.
 *
 * Naming: every exported symbol is `hyle_*`.
 * Docs (1:1 with this header): docs/guide/c-core.md
 */
#ifndef HYLE_C_CORE_H_
#define HYLE_C_CORE_H_

#include <stdint.h>

#include "hyle/core/version_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(HYLE_STATIC)
#define HYLE_CORE_C_API
#elif defined(_WIN32) || defined(__CYGWIN__)
#if defined(HYLE_CORE_BUILD)
#define HYLE_CORE_C_API __declspec(dllexport)
#else
#define HYLE_CORE_C_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define HYLE_CORE_C_API __attribute__((visibility("default")))
#else
#define HYLE_CORE_C_API
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
#define HYLE_CORE_C_CALL __cdecl
#else
#define HYLE_CORE_C_CALL
#endif

/* Native FFI protocol version of this hyle_core artifact. Bindings compare
 * this with HYLE_FFI_ABI_VERSION before resolving any other core symbol. The
 * high 16 bits are major and the low 16 bits are minor. This is a pre-release
 * protocol: no shipped binding depends on it yet, so it may change before the
 * first FFI product release. After publication, that binding's bootstrap name,
 * C calling convention, and uint32_t(void) signature remain discoverable. */
HYLE_CORE_C_API uint32_t HYLE_CORE_C_CALL hyle_ffi_abi_version(void);

/* Product version of this hyle_core artifact. Compare with HYLE_VERSION_PACKED
 * only to diagnose a binding/native-package mismatch; it is not an ABI gate. */
HYLE_CORE_C_API uint32_t HYLE_CORE_C_CALL hyle_version(void);

/* Human-readable product version (major.minor.patch). The returned UTF-8/ASCII
 * NUL-terminated string is borrowed from hyle_core: never free it, and copy it
 * before unloading the module. */
HYLE_CORE_C_API const char *HYLE_CORE_C_CALL hyle_version_string(void);

#ifdef __cplusplus
}
#endif

#endif /* HYLE_C_CORE_H_ */
