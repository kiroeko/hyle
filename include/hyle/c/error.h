/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary
 *
 * Pure C ABI for the HYLE error module (close-01).
 * Naming: every symbol is `hyle_*` / `HYLE_*`.
 * Docs (1:1 with this header): docs/guide/c-error.md
 */
#ifndef HYLE_C_ERROR_H_
#define HYLE_C_ERROR_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(HYLE_STATIC)
#define HYLE_ERROR_C_API
#elif defined(_WIN32) || defined(__CYGWIN__)
#if defined(HYLE_ERROR_BUILD)
#define HYLE_ERROR_C_API __declspec(dllexport)
#else
#define HYLE_ERROR_C_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define HYLE_ERROR_C_API __attribute__((visibility("default")))
#else
#define HYLE_ERROR_C_API
#endif

/* ---- status codes ----------------------------------------------------------
 */
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
  HYLE_STATUS_DATA_LOSS = 15,
  HYLE_STATUS_UNAUTHENTICATED = 16
} hyle_status_code;

typedef struct hyle_status {
  uintptr_t rep; /* opaque owning handle */
} hyle_status;

/* Constant zero/OK initializer. Every hyle_status passed as a destination to
 * free/move/add_context must first be initialized with this macro or returned
 * by a hyle_status_* factory. */
#define HYLE_STATUS_INIT {0}

/* construct / destroy */
HYLE_ERROR_C_API hyle_status hyle_status_ok(void);
HYLE_ERROR_C_API hyle_status hyle_status_new(hyle_status_code code,
                                             const char *msg);
/* Canonical errno adapter. errno==0 produces OK. `context` may be NULL; a
 * non-empty value is prefixed to the native CRT description. */
HYLE_ERROR_C_API hyle_status_code hyle_status_code_from_errno(
    int error_number);
HYLE_ERROR_C_API hyle_status hyle_status_from_errno(int error_number,
                                                    const char *context);
HYLE_ERROR_C_API void hyle_status_free(hyle_status *s);
HYLE_ERROR_C_API hyle_status hyle_status_clone(hyle_status s);
/* dst must already hold an initialized Status. Its old value is freed before
 * src is moved into it; src becomes OK. NULL src means "free dst". */
HYLE_ERROR_C_API void hyle_status_move(hyle_status *dst, hyle_status *src);
/* Prefix context in place. NULL/OK/empty are no-ops; allocation failure keeps
 * the original code and message unchanged. */
HYLE_ERROR_C_API void hyle_status_add_context(hyle_status *s,
                                              const char *context);

/* queries */
HYLE_ERROR_C_API int hyle_status_is_ok(hyle_status s);
HYLE_ERROR_C_API hyle_status_code hyle_status_get_code(hyle_status s);
HYLE_ERROR_C_API int hyle_status_has_message(hyle_status s);
HYLE_ERROR_C_API int hyle_status_equals(hyle_status a, hyle_status b);
HYLE_ERROR_C_API const char *hyle_status_code_name(hyle_status_code code);
HYLE_ERROR_C_API size_t hyle_status_message(hyle_status s, char *buf,
                                            size_t cap);
HYLE_ERROR_C_API size_t hyle_status_to_string(hyle_status s, char *buf,
                                              size_t cap);

/* Borrowed, NUL-terminated pointer to the message bytes, valid until this
 * status is freed/moved/cloned-over. Returns "" when there is no message (OK or
 * a message-less code). Zero-copy, no sizing round-trip -- the FFI-friendly
 * counterpart to the buffer-fill hyle_status_message above. */
HYLE_ERROR_C_API const char *hyle_status_message_cstr(hyle_status s);

/* ---- factories (header-inline) ---------------------------------------------
 */
static inline hyle_status hyle_status_cancelled(const char *msg) {
  return hyle_status_new(HYLE_STATUS_CANCELLED, msg);
}
static inline hyle_status hyle_status_unknown(const char *msg) {
  return hyle_status_new(HYLE_STATUS_UNKNOWN, msg);
}
static inline hyle_status hyle_status_invalid_argument(const char *msg) {
  return hyle_status_new(HYLE_STATUS_INVALID_ARGUMENT, msg);
}
static inline hyle_status hyle_status_deadline_exceeded(const char *msg) {
  return hyle_status_new(HYLE_STATUS_DEADLINE_EXCEEDED, msg);
}
static inline hyle_status hyle_status_not_found(const char *msg) {
  return hyle_status_new(HYLE_STATUS_NOT_FOUND, msg);
}
static inline hyle_status hyle_status_already_exists(const char *msg) {
  return hyle_status_new(HYLE_STATUS_ALREADY_EXISTS, msg);
}
static inline hyle_status hyle_status_permission_denied(const char *msg) {
  return hyle_status_new(HYLE_STATUS_PERMISSION_DENIED, msg);
}
static inline hyle_status hyle_status_resource_exhausted(const char *msg) {
  return hyle_status_new(HYLE_STATUS_RESOURCE_EXHAUSTED, msg);
}
static inline hyle_status hyle_status_failed_precondition(const char *msg) {
  return hyle_status_new(HYLE_STATUS_FAILED_PRECONDITION, msg);
}
static inline hyle_status hyle_status_aborted(const char *msg) {
  return hyle_status_new(HYLE_STATUS_ABORTED, msg);
}
static inline hyle_status hyle_status_out_of_range(const char *msg) {
  return hyle_status_new(HYLE_STATUS_OUT_OF_RANGE, msg);
}
static inline hyle_status hyle_status_unimplemented(const char *msg) {
  return hyle_status_new(HYLE_STATUS_UNIMPLEMENTED, msg);
}
static inline hyle_status hyle_status_internal(const char *msg) {
  return hyle_status_new(HYLE_STATUS_INTERNAL, msg);
}
static inline hyle_status hyle_status_unavailable(const char *msg) {
  return hyle_status_new(HYLE_STATUS_UNAVAILABLE, msg);
}
static inline hyle_status hyle_status_data_loss(const char *msg) {
  return hyle_status_new(HYLE_STATUS_DATA_LOSS, msg);
}
static inline hyle_status hyle_status_unauthenticated(const char *msg) {
  return hyle_status_new(HYLE_STATUS_UNAUTHENTICATED, msg);
}

/* ---- predicates (header-inline; full code set) -----------------------------
 */
static inline int hyle_status_code_is(hyle_status s, hyle_status_code code) {
  return hyle_status_get_code(s) == code;
}
static inline int hyle_status_is_cancelled(hyle_status s) {
  return hyle_status_code_is(s, HYLE_STATUS_CANCELLED);
}
static inline int hyle_status_is_unknown(hyle_status s) {
  return hyle_status_code_is(s, HYLE_STATUS_UNKNOWN);
}
static inline int hyle_status_is_invalid_argument(hyle_status s) {
  return hyle_status_code_is(s, HYLE_STATUS_INVALID_ARGUMENT);
}
static inline int hyle_status_is_deadline_exceeded(hyle_status s) {
  return hyle_status_code_is(s, HYLE_STATUS_DEADLINE_EXCEEDED);
}
static inline int hyle_status_is_not_found(hyle_status s) {
  return hyle_status_code_is(s, HYLE_STATUS_NOT_FOUND);
}
static inline int hyle_status_is_already_exists(hyle_status s) {
  return hyle_status_code_is(s, HYLE_STATUS_ALREADY_EXISTS);
}
static inline int hyle_status_is_permission_denied(hyle_status s) {
  return hyle_status_code_is(s, HYLE_STATUS_PERMISSION_DENIED);
}
static inline int hyle_status_is_resource_exhausted(hyle_status s) {
  return hyle_status_code_is(s, HYLE_STATUS_RESOURCE_EXHAUSTED);
}
static inline int hyle_status_is_failed_precondition(hyle_status s) {
  return hyle_status_code_is(s, HYLE_STATUS_FAILED_PRECONDITION);
}
static inline int hyle_status_is_aborted(hyle_status s) {
  return hyle_status_code_is(s, HYLE_STATUS_ABORTED);
}
static inline int hyle_status_is_out_of_range(hyle_status s) {
  return hyle_status_code_is(s, HYLE_STATUS_OUT_OF_RANGE);
}
static inline int hyle_status_is_unimplemented(hyle_status s) {
  return hyle_status_code_is(s, HYLE_STATUS_UNIMPLEMENTED);
}
static inline int hyle_status_is_internal(hyle_status s) {
  return hyle_status_code_is(s, HYLE_STATUS_INTERNAL);
}
static inline int hyle_status_is_unavailable(hyle_status s) {
  return hyle_status_code_is(s, HYLE_STATUS_UNAVAILABLE);
}
static inline int hyle_status_is_data_loss(hyle_status s) {
  return hyle_status_code_is(s, HYLE_STATUS_DATA_LOSS);
}
static inline int hyle_status_is_unauthenticated(hyle_status s) {
  return hyle_status_code_is(s, HYLE_STATUS_UNAUTHENTICATED);
}

/* Early-return helpers for functions returning hyle_status by value. */
#define HYLE_C_RETURN_IF_ERROR(st)                       \
  do {                                                   \
    hyle_status _hyle_st_ = (st);                        \
    if (!hyle_status_is_ok(_hyle_st_)) return _hyle_st_; \
  } while (0)

#ifdef __cplusplus
}
#endif

#endif /* HYLE_C_ERROR_H_ */
