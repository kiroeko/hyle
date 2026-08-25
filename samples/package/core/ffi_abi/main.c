/* Installed-package conformance probe for the core Native FFI ABI. This file
 * is compiled as C because the public header is the ABI schema, not because C
 * is a supported source-level product surface. */
#include <hyle/c/core.h>

#include <stdint.h>
#include <string.h>

_Static_assert(HYLE_FFI_ABI_VERSION == 0x00010000u,
               "unexpected core Native FFI ABI");

int main(void) {
  const uint32_t abi = hyle_ffi_abi_version();
  const char *version = hyle_version_string();
  return abi == HYLE_FFI_ABI_VERSION &&
                 hyle_version() == HYLE_VERSION_PACKED && version != 0 &&
                 strcmp(version, HYLE_VERSION_STRING) == 0
             ? 0
             : 1;
}
