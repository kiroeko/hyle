/* C-compiler/schema/link fixture for the direct Native FFI declarations. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "hyle/c/error.h"

/* The surface audit ignores declarations pulled in by error.h and counts only
 * calls below this marker as exercised by the C-syntax fixture. */
enum { HYLE_C_ERROR_COVERAGE_BODY_MARKER = 0 };

static int fails = 0;

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #condition, __FILE__,     \
              __LINE__);                                                       \
      ++fails;                                                                 \
    }                                                                          \
  } while (0)

int main(void) {
  hyle_status status = (hyle_status)0;
  hyle_native_error_domain domain = 123;
  int64_t native_code = 123;
  char text[8] = {0};

  CHECK(sizeof(hyle_status_code) == sizeof(uint16_t));
  CHECK(sizeof(hyle_native_error_domain) == sizeof(uint16_t));
  CHECK(sizeof(hyle_status) == sizeof(void *));
  CHECK(hyle_error_ffi_abi_version() == HYLE_ERROR_FFI_ABI_VERSION);
  CHECK(hyle_status_get_code(status) == HYLE_STATUS_OK);
  CHECK(hyle_status_get_native_error(status, &domain, &native_code) == 0);
  CHECK(domain == HYLE_NATIVE_ERROR_NONE);
  CHECK(native_code == 0);
  CHECK(hyle_status_message(status, (char *)0, 0) == 0);
  CHECK(hyle_status_to_string(status, text, sizeof(text)) == 2);
  CHECK(strcmp(text, "OK") == 0);
  hyle_status_destroy(status);

  return fails == 0 ? 0 : 1;
}
