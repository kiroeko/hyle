#include <stdio.h>
#include <string.h>

#include "hyle/c/error.h"
#include "hyle/c/mem.h"

int main(void) {
  static const char expected_message[] =
      "hyle_quota_allocator_create: out is NULL";
  static const char expected_text[] =
      "InvalidArgument: hyle_quota_allocator_create: out is NULL";
  hyle_status status = hyle_quota_allocator_create(
      NULL, hyle_system_allocator(), 64);
  hyle_native_error_domain domain = HYLE_NATIVE_ERROR_UNKNOWN;
  int64_t native_code = -1;
  char message[96];
  char text[128];
  const uint64_t message_needed =
      hyle_status_message(status, message, sizeof(message));
  const uint64_t text_needed =
      hyle_status_to_string(status, text, sizeof(text));
  hyle_allocator allocator = hyle_system_allocator();
  void *block = hyle_alloc(&allocator, 64, 0);
  const int ok =
      hyle_error_ffi_abi_version() == HYLE_FFI_ABI_VERSION &&
      status != (hyle_status)0 &&
      hyle_status_get_code(status) == HYLE_STATUS_INVALID_ARGUMENT &&
      hyle_status_get_native_error(status, &domain, &native_code) == 0 &&
      domain == HYLE_NATIVE_ERROR_NONE && native_code == 0 &&
      message_needed == strlen(expected_message) &&
      strcmp(message, expected_message) == 0 &&
      text_needed == strlen(expected_text) && strcmp(text, expected_text) == 0 &&
      hyle_status_get_code((hyle_status)0) == HYLE_STATUS_OK && block != NULL;
  printf("installed receiver-oriented FFI schema fixture: %s\n", text);
  hyle_free(&allocator, block, 64, 0);
  hyle_status_destroy(status);
  hyle_status_destroy((hyle_status)0);
  return ok ? 0 : 1;
}
