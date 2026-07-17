/* SPDX: same as HYLE.
 *
 * Pure C ABI for the HYLE error module (close-01).
 *
 * Naming: every symbol is `hyle_*` / `HYLE_*` (no clay prefixes).
 * This is the only supported stable surface for FFI consumers of Status and
 * process-wide diagnostic reporting. Do NOT include hyle/error/*.h from C.
 *
 * Ownership (read carefully):
 *   - `hyle_status` is an 8-byte *owning* value (like a refcounted handle).
 *   - Create with hyle_status_ok / hyle_status_new / hyle_status_* factories.
 *   - Copy only with hyle_status_clone (never bare assignment of the struct for
 *     message-carrying statuses — that double-frees).
 *   - Transfer with hyle_status_move.
 *   - Always hyle_status_free when done (idempotent on OK / already-freed).
 *
 * Docs: docs/guide/c-error.md
 */
#ifndef HYLE_C_ERROR_H_
#define HYLE_C_ERROR_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- export ---------------------------------------------------------------- */
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

/* Bump on breaking C surface changes. */
#define HYLE_ERROR_C_ABI_VERSION 1

/* ---- status codes (stable integers; match internal StatusCode) ------------- */
typedef enum hyle_status_code {
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
  HYLE_STATUS_DATA_LOSS = 15
} hyle_status_code;

/* Owning status. Treat `rep` as opaque — only the functions below may touch it. */
typedef struct hyle_status {
  uintptr_t rep;
} hyle_status;

/* ---- construct / destroy --------------------------------------------------- */

/* OK (no resources). */
HYLE_ERROR_C_API hyle_status hyle_status_ok(void);

/* New status. msg may be NULL/empty (message-less, no heap).
 * code == HYLE_STATUS_OK always yields OK (msg ignored).
 * Out-of-range codes become HYLE_STATUS_UNKNOWN. */
HYLE_ERROR_C_API hyle_status hyle_status_new(hyle_status_code code,
                                             const char* msg);

/* Release heap (if any) and set *s to OK. Safe on NULL / already OK. */
HYLE_ERROR_C_API void hyle_status_free(hyle_status* s);

/* Retain: deep logical copy (refcount++). Pair every clone with free. */
HYLE_ERROR_C_API hyle_status hyle_status_clone(hyle_status s);

/* Move: free *dst if needed, take ownership from *src, leave *src OK. */
HYLE_ERROR_C_API void hyle_status_move(hyle_status* dst, hyle_status* src);

/* ---- queries --------------------------------------------------------------- */

HYLE_ERROR_C_API int hyle_status_is_ok(hyle_status s);
HYLE_ERROR_C_API hyle_status_code hyle_status_get_code(hyle_status s);
HYLE_ERROR_C_API int hyle_status_has_message(hyle_status s);

/* 1 if code+message equal. */
HYLE_ERROR_C_API int hyle_status_equals(hyle_status a, hyle_status b);

/* Stable English name: "NotFound", "OK", ... Never NULL. */
HYLE_ERROR_C_API const char* hyle_status_code_name(hyle_status_code code);

/* Message into buf (NUL-terminated if cap>0). snprintf length semantics. */
HYLE_ERROR_C_API size_t hyle_status_message(hyle_status s, char* buf,
                                            size_t cap);

/* "OK" | "CODE" | "CODE: message". snprintf length semantics. */
HYLE_ERROR_C_API size_t hyle_status_to_string(hyle_status s, char* buf,
                                              size_t cap);

/* ---- easy factories / predicates (header-inline, zero extra DLL symbols) --- */

static inline hyle_status hyle_status_cancelled(const char* msg) {
  return hyle_status_new(HYLE_STATUS_CANCELLED, msg);
}
static inline hyle_status hyle_status_unknown(const char* msg) {
  return hyle_status_new(HYLE_STATUS_UNKNOWN, msg);
}
static inline hyle_status hyle_status_invalid_argument(const char* msg) {
  return hyle_status_new(HYLE_STATUS_INVALID_ARGUMENT, msg);
}
static inline hyle_status hyle_status_deadline_exceeded(const char* msg) {
  return hyle_status_new(HYLE_STATUS_DEADLINE_EXCEEDED, msg);
}
static inline hyle_status hyle_status_not_found(const char* msg) {
  return hyle_status_new(HYLE_STATUS_NOT_FOUND, msg);
}
static inline hyle_status hyle_status_already_exists(const char* msg) {
  return hyle_status_new(HYLE_STATUS_ALREADY_EXISTS, msg);
}
static inline hyle_status hyle_status_permission_denied(const char* msg) {
  return hyle_status_new(HYLE_STATUS_PERMISSION_DENIED, msg);
}
static inline hyle_status hyle_status_resource_exhausted(const char* msg) {
  return hyle_status_new(HYLE_STATUS_RESOURCE_EXHAUSTED, msg);
}
static inline hyle_status hyle_status_failed_precondition(const char* msg) {
  return hyle_status_new(HYLE_STATUS_FAILED_PRECONDITION, msg);
}
static inline hyle_status hyle_status_aborted(const char* msg) {
  return hyle_status_new(HYLE_STATUS_ABORTED, msg);
}
static inline hyle_status hyle_status_out_of_range(const char* msg) {
  return hyle_status_new(HYLE_STATUS_OUT_OF_RANGE, msg);
}
static inline hyle_status hyle_status_unimplemented(const char* msg) {
  return hyle_status_new(HYLE_STATUS_UNIMPLEMENTED, msg);
}
static inline hyle_status hyle_status_internal(const char* msg) {
  return hyle_status_new(HYLE_STATUS_INTERNAL, msg);
}
static inline hyle_status hyle_status_unavailable(const char* msg) {
  return hyle_status_new(HYLE_STATUS_UNAVAILABLE, msg);
}
static inline hyle_status hyle_status_data_loss(const char* msg) {
  return hyle_status_new(HYLE_STATUS_DATA_LOSS, msg);
}

static inline int hyle_status_code_is(hyle_status s, hyle_status_code code) {
  return hyle_status_get_code(s) == code;
}
static inline int hyle_status_is_not_found(hyle_status s) {
  return hyle_status_code_is(s, HYLE_STATUS_NOT_FOUND);
}
static inline int hyle_status_is_invalid_argument(hyle_status s) {
  return hyle_status_code_is(s, HYLE_STATUS_INVALID_ARGUMENT);
}
static inline int hyle_status_is_internal(hyle_status s) {
  return hyle_status_code_is(s, HYLE_STATUS_INTERNAL);
}
static inline int hyle_status_is_unimplemented(hyle_status s) {
  return hyle_status_code_is(s, HYLE_STATUS_UNIMPLEMENTED);
}

/* ---- diagnostic sink (process-wide) ---------------------------------------- */

typedef enum hyle_severity {
  HYLE_SEVERITY_WARNING = 0,
  HYLE_SEVERITY_ERROR = 1,
  HYLE_SEVERITY_FATAL = 2
} hyle_severity;

typedef void (*hyle_error_report_fn)(void* user, hyle_severity severity,
                                     const char* file, uint32_t line,
                                     const char* message);

/* Install C callback as default sink. fn=NULL restores built-in stderr. */
HYLE_ERROR_C_API void hyle_error_set_handler(hyle_error_report_fn fn,
                                             void* user);

/* Already-formatted message. FATAL traps after the handler (never returns). */
HYLE_ERROR_C_API void hyle_error_report(hyle_severity severity, const char* file,
                                        uint32_t line, const char* message);

/* ---- optional vtable pack (FFI that prefers one struct of fn-ptrs) --------- */

typedef struct hyle_error_api {
  int abi_version; /* must be HYLE_ERROR_C_ABI_VERSION */

  hyle_status (*status_ok)(void);
  hyle_status (*status_new)(hyle_status_code code, const char* msg);
  void (*status_free)(hyle_status* s);
  hyle_status (*status_clone)(hyle_status s);
  void (*status_move)(hyle_status* dst, hyle_status* src);

  int (*status_is_ok)(hyle_status s);
  hyle_status_code (*status_get_code)(hyle_status s);
  int (*status_has_message)(hyle_status s);
  int (*status_equals)(hyle_status a, hyle_status b);
  const char* (*status_code_name)(hyle_status_code code);
  size_t (*status_message)(hyle_status s, char* buf, size_t cap);
  size_t (*status_to_string)(hyle_status s, char* buf, size_t cap);

  void (*error_set_handler)(hyle_error_report_fn fn, void* user);
  void (*error_report)(hyle_severity severity, const char* file, uint32_t line,
                       const char* message);
} hyle_error_api;

/* Fill *out with the current API table. Returns 0 on success, -1 if out==NULL
 * or abi_version mismatch expectation (pass expected=HYLE_ERROR_C_ABI_VERSION). */
HYLE_ERROR_C_API int hyle_error_api_get(hyle_error_api* out, int expected_abi);

HYLE_ERROR_C_API int hyle_error_c_abi_version(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* HYLE_C_ERROR_H_ */
