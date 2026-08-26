/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary
 *
 * Pre-release direct Native FFI schema for the HYLE error artifact. It is
 * designed for managed/runtime bindings (Java FFM, .NET P/Invoke, ...), not as
 * a source-level C product API. No released binding consumes this contract
 * yet, so it may change freely before the first FFI product release.
 *
 * Naming: every exported symbol is `hyle_*`.
 * Docs (1:1 with this header): docs/guide/c-error.md
 */
#ifndef HYLE_C_ERROR_H_
#define HYLE_C_ERROR_H_

#include <stdint.h>

// Legacy pre-release generation of this error artifact's provisional FFI.
// It is intentionally error-local: Core has no Native FFI product, and this
// schema will be removed or redesigned when Error receives its own v2 review.
#define HYLE_ERROR_FFI_ABI_VERSION UINT32_C(0x00010000)

#ifdef __cplusplus
extern "C" {
#endif

#if defined(HYLE_STATIC)
#  define HYLE_ERROR_C_API
#elif defined(_WIN32) || defined(__CYGWIN__)
#  if defined(HYLE_ERROR_BUILD)
#    define HYLE_ERROR_C_API __declspec(dllexport)
#  else
#    define HYLE_ERROR_C_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define HYLE_ERROR_C_API __attribute__((visibility("default")))
#else
#  define HYLE_ERROR_C_API
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
#  define HYLE_ERROR_C_CALL __cdecl
#else
#  define HYLE_ERROR_C_CALL
#endif

/* Fixed-width canonical code. The named values are constants rather than a C
 * enum ABI, so -fshort-enums cannot change any function signature. */
typedef uint16_t hyle_status_code;
enum {
  HYLE_STATUS_OK = 0,
  HYLE_STATUS_CANCELLED = 1,
  HYLE_STATUS_UNKNOWN = 2,
  HYLE_STATUS_INVALID_ARGUMENT = 3,
  HYLE_STATUS_DEADLINE_EXCEEDED = 4,
  HYLE_STATUS_NOT_FOUND = 5,
  HYLE_STATUS_ALREADY_EXISTS = 6,
  HYLE_STATUS_PERMISSION_DENIED = 7,
  HYLE_STATUS_RESOURCE_EXHAUSTED = 8,
  HYLE_STATUS_FAILED_PRECONDITION = 9,
  HYLE_STATUS_ABORTED = 10,
  HYLE_STATUS_OUT_OF_RANGE = 11,
  HYLE_STATUS_UNIMPLEMENTED = 12,
  HYLE_STATUS_INTERNAL = 13,
  HYLE_STATUS_UNAVAILABLE = 14,
  HYLE_STATUS_DATA_LOSS = 15,
  HYLE_STATUS_UNAUTHENTICATED = 16
};

typedef uint16_t hyle_native_error_domain;
enum {
  HYLE_NATIVE_ERROR_NONE = 0,
  HYLE_NATIVE_ERROR_POSIX_ERRNO = 1,
  HYLE_NATIVE_ERROR_WIN32 = 2,
  HYLE_NATIVE_ERROR_VULKAN = 3,
  HYLE_NATIVE_ERROR_CUDA_RUNTIME = 4,
  HYLE_NATIVE_ERROR_CUDA_DRIVER = 5,
  HYLE_NATIVE_ERROR_NVRTC = 6,
  HYLE_NATIVE_ERROR_HRESULT = 7,
  HYLE_NATIVE_ERROR_UNKNOWN = UINT16_MAX
};

/* Owning, immutable error handle returned by HYLE module FFIs. NULL is the
 * complete OK representation and needs no call into hyle_error. A non-NULL
 * value is owned exactly once and must eventually be passed to
 * hyle_status_destroy. Never dereference, perform pointer arithmetic on, or
 * duplicate a handle by copying it into a second owner. */
typedef struct hyle_status_handle *hyle_status;

/* ABI generation of this exact hyle_error shared artifact. A binding must call
 * this before resolving any other symbol from the artifact. */
HYLE_ERROR_C_API uint32_t HYLE_ERROR_C_CALL hyle_error_ffi_abi_version(void);

/* Release one owning handle. NULL is accepted. This is the only lifetime
 * operation needed by a zero-invalid .NET SafeHandle or Java AutoCloseable. */
HYLE_ERROR_C_API void HYLE_ERROR_C_CALL hyle_status_destroy(hyle_status status);

/* Borrowing queries. NULL reports OK/no message/no native detail. Immutable
 * queries may run concurrently while the owning handle's lifetime is pinned.
 * Destroying or otherwise consuming that handle concurrently is invalid. */
HYLE_ERROR_C_API hyle_status_code HYLE_ERROR_C_CALL
hyle_status_get_code(hyle_status status);

/* Return 1 when a native detail is present, otherwise 0. Either output may be
 * NULL. Missing detail writes NONE/0 to supplied outputs. */
HYLE_ERROR_C_API int32_t HYLE_ERROR_C_CALL hyle_status_get_native_error(
    hyle_status status, hyle_native_error_domain *out_domain,
    int64_t *out_code);

/* Caller-buffer UTF-8 byte copies. Return the required byte count excluding
 * the trailing NUL. If buffer is non-NULL and capacity is nonzero, write at
 * most capacity-1 bytes followed by NUL. A NULL buffer is a size-only query.
 * Messages may contain embedded NUL bytes; use the returned length rather than
 * a C-string scan. A truncated byte prefix may end inside a UTF-8 sequence;
 * retry with required+1 bytes before decoding text. uint64_t is intentional:
 * HYLE supports 64-bit target ABIs.
 */
HYLE_ERROR_C_API uint64_t HYLE_ERROR_C_CALL hyle_status_message(
    hyle_status status, char *buffer, uint64_t capacity);
HYLE_ERROR_C_API uint64_t HYLE_ERROR_C_CALL hyle_status_to_string(
    hyle_status status, char *buffer, uint64_t capacity);

#ifdef __cplusplus
}
#endif

#endif /* HYLE_C_ERROR_H_ */
