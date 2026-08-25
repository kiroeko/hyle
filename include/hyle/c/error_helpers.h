/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary
 *
 * Source-only native-C conveniences used by HYLE's compiler/package fixtures.
 * These inline functions are not direct Native FFI exports and are not a C
 * product surface. Managed bindings should map NULL and the exported
 * functions in hyle/c/error.h directly.
 */
#ifndef HYLE_C_ERROR_HELPERS_H_
#define HYLE_C_ERROR_HELPERS_H_

#include <stddef.h>

#include "hyle/c/error.h"

#define HYLE_STATUS_INIT ((hyle_status)0)

static inline hyle_status hyle_status_ok(void) {
  return (hyle_status)0;
}

static inline int hyle_status_is_ok(hyle_status status) {
  return status == (hyle_status)0;
}

static inline void hyle_status_free(hyle_status *status) {
  if (status == NULL)
    return;
  hyle_status_destroy(*status);
  *status = (hyle_status)0;
}

static inline int hyle_status_has_message(hyle_status status) {
  return hyle_status_message(status, NULL, 0) != 0;
}

static inline int hyle_status_code_is(hyle_status status,
                                      hyle_status_code code) {
  return hyle_status_get_code(status) == code;
}

#define HYLE_STATUS_SOURCE_PREDICATE(name, code)                              \
  static inline int hyle_status_is_##name(hyle_status status) {              \
    return hyle_status_code_is(status, code);                                \
  }

HYLE_STATUS_SOURCE_PREDICATE(cancelled, HYLE_STATUS_CANCELLED)
HYLE_STATUS_SOURCE_PREDICATE(unknown, HYLE_STATUS_UNKNOWN)
HYLE_STATUS_SOURCE_PREDICATE(invalid_argument, HYLE_STATUS_INVALID_ARGUMENT)
HYLE_STATUS_SOURCE_PREDICATE(deadline_exceeded,
                             HYLE_STATUS_DEADLINE_EXCEEDED)
HYLE_STATUS_SOURCE_PREDICATE(not_found, HYLE_STATUS_NOT_FOUND)
HYLE_STATUS_SOURCE_PREDICATE(already_exists, HYLE_STATUS_ALREADY_EXISTS)
HYLE_STATUS_SOURCE_PREDICATE(permission_denied,
                             HYLE_STATUS_PERMISSION_DENIED)
HYLE_STATUS_SOURCE_PREDICATE(resource_exhausted,
                             HYLE_STATUS_RESOURCE_EXHAUSTED)
HYLE_STATUS_SOURCE_PREDICATE(failed_precondition,
                             HYLE_STATUS_FAILED_PRECONDITION)
HYLE_STATUS_SOURCE_PREDICATE(aborted, HYLE_STATUS_ABORTED)
HYLE_STATUS_SOURCE_PREDICATE(out_of_range, HYLE_STATUS_OUT_OF_RANGE)
HYLE_STATUS_SOURCE_PREDICATE(unimplemented, HYLE_STATUS_UNIMPLEMENTED)
HYLE_STATUS_SOURCE_PREDICATE(internal, HYLE_STATUS_INTERNAL)
HYLE_STATUS_SOURCE_PREDICATE(unavailable, HYLE_STATUS_UNAVAILABLE)
HYLE_STATUS_SOURCE_PREDICATE(data_loss, HYLE_STATUS_DATA_LOSS)
HYLE_STATUS_SOURCE_PREDICATE(unauthenticated, HYLE_STATUS_UNAUTHENTICATED)

#undef HYLE_STATUS_SOURCE_PREDICATE

#endif /* HYLE_C_ERROR_HELPERS_H_ */
