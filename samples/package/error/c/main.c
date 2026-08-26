#include <hyle/c/error.h>

#include <stdint.h>
#include <string.h>

#if defined(_MSC_VER)
typedef char installed_status_code_must_be_u16[
    sizeof(hyle_status_code) == sizeof(uint16_t) ? 1 : -1];
typedef char installed_native_domain_must_be_u16[
    sizeof(hyle_native_error_domain) == sizeof(uint16_t) ? 1 : -1];
#else
_Static_assert(sizeof(hyle_status_code) == sizeof(uint16_t),
               "installed status-code ABI width");
_Static_assert(sizeof(hyle_native_error_domain) == sizeof(uint16_t),
               "installed native-domain ABI width");
#endif

int main(void) {
  hyle_status status = (hyle_status)0;
  hyle_native_error_domain domain = HYLE_NATIVE_ERROR_UNKNOWN;
  int64_t native_code = -1;
  char message[1] = {'x'};
  char text[8] = {0};

  const int ok =
      sizeof(hyle_status) == sizeof(void*) &&
      hyle_error_ffi_abi_version() == HYLE_ERROR_FFI_ABI_VERSION &&
      hyle_status_get_code(status) == HYLE_STATUS_OK &&
      hyle_status_get_native_error(status, &domain, &native_code) == 0 &&
      domain == HYLE_NATIVE_ERROR_NONE && native_code == 0 &&
      hyle_status_message(status, message, sizeof(message)) == 0 &&
      message[0] == '\0' &&
      hyle_status_to_string(status, text, sizeof(text)) == 2 &&
      strcmp(text, "OK") == 0;

  hyle_status_destroy(status);
  return ok ? 0 : 1;
}
