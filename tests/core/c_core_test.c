/* Low-level C compiler fixture for the Native FFI ABI schema. This is not a
 * source-level C product API test: every function called below is a real
 * hyle_core dynamic/static symbol that Java/.NET/Rust/Python can bind. */
#include "hyle/c/core.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int g_fails = 0;
#define CHECK(condition)                                                     \
  do {                                                                       \
    if (!(condition)) {                                                      \
      fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #condition, __FILE__,  \
              __LINE__);                                                     \
      ++g_fails;                                                             \
    }                                                                        \
  } while (0)

_Static_assert(HYLE_FFI_ABI_VERSION_MAJOR == 1,
               "unexpected Native FFI ABI major");
_Static_assert(HYLE_FFI_ABI_VERSION_MINOR == 0,
               "unexpected Native FFI ABI minor");
_Static_assert(HYLE_FFI_ABI_VERSION == 0x00010000u,
               "unexpected packed Native FFI ABI version");

int main(void) {
  (void)"HYLE_C_CORE_COVERAGE_BODY_MARKER";
  const uint32_t ffi_abi = hyle_ffi_abi_version();
  const uint32_t product = hyle_version();
  const char *product_string = hyle_version_string();

  CHECK(ffi_abi == HYLE_FFI_ABI_VERSION);
  CHECK((ffi_abi >> 16) == HYLE_FFI_ABI_VERSION_MAJOR);
  CHECK((ffi_abi & 0xFFFFu) == HYLE_FFI_ABI_VERSION_MINOR);
  CHECK(product == HYLE_VERSION_PACKED);
  CHECK(product_string != NULL);
  CHECK(strcmp(product_string, HYLE_VERSION_STRING) == 0);

  if (g_fails != 0) {
    fprintf(stderr, "c_core_test: %d failed\n", g_fails);
    return 1;
  }
  puts("c_core_test: Native FFI ABI checks passed");
  return 0;
}
