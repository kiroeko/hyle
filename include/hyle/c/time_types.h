/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary
 *
 * C-syntax time value types shared by current draft FFI schemas for time,
 * platform and io. This header contains no out-of-line functions and has no
 * library dependency; it is not by itself a native-C product or frozen ABI.
 */
#ifndef HYLE_C_TIME_TYPES_H_
#define HYLE_C_TIME_TYPES_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Signed Unix timestamp with the same canonical representation as timespec:
 * seconds may be before 1970 and nanoseconds is always in [0, 1e9).  Keeping
 * the fractional part separate avoids both the 1677--2262 range limit of one
 * int64 nanosecond count and the pre-epoch wrap of an unsigned count. */
typedef struct hyle_unix_timestamp {
  int64_t seconds;
  uint32_t nanoseconds;
  uint32_t reserved;
} hyle_unix_timestamp;

#if defined(__cplusplus)
static_assert(sizeof(hyle_unix_timestamp) == 16,
              "hyle_unix_timestamp ABI size changed");
#elif defined(_MSC_VER)
/* MSVC's C17 front-end does not implement the C11 `_Static_assert` keyword.
 * A negative-size typedef preserves the compile-time ABI guard without
 * requiring the C++ front-end or a vendor extension. */
typedef char hyle_unix_timestamp_abi_size_must_be_16[
    sizeof(hyle_unix_timestamp) == 16 ? 1 : -1];
#else
_Static_assert(sizeof(hyle_unix_timestamp) == 16,
               "hyle_unix_timestamp ABI size changed");
#endif

#define HYLE_UNIX_TIMESTAMP_INIT {INT64_C(0), UINT32_C(0), UINT32_C(0)}

static inline int hyle_unix_timestamp_is_valid(hyle_unix_timestamp value) {
  return value.nanoseconds < UINT32_C(1000000000) && value.reserved == 0;
}

#ifdef __cplusplus
}
#endif

#endif /* HYLE_C_TIME_TYPES_H_ */
