#include <hyle/c/error.h>

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
  hyle_status status = hyle_status_not_found("model.bin");
  hyle_status_add_context(&status, "loading model");

  hyle_status copy = hyle_status_clone(status);
  hyle_status moved = HYLE_STATUS_INIT;
  hyle_status_move(&moved, &copy);
  const int32_t access_denied = (int32_t)UINT32_C(0x80070005);
  hyle_status com = hyle_status_from_hresult(access_denied, "naming thread");
  hyle_native_error_domain domain = HYLE_NATIVE_ERROR_NONE;
  int64_t native_code = 0;

  char text[64];
  const size_t needed = hyle_status_to_string(moved, text, sizeof(text));
  const int ok =
      hyle_status_get_code(moved) == HYLE_STATUS_NOT_FOUND &&
      hyle_status_is_ok(copy) &&
      hyle_status_code_from_win32(5) == HYLE_STATUS_PERMISSION_DENIED &&
      hyle_status_is_permission_denied(com) &&
      hyle_status_get_native_error(com, &domain, &native_code) &&
      domain == HYLE_NATIVE_ERROR_HRESULT && native_code == access_denied &&
      needed == strlen("NotFound: loading model: model.bin") &&
      strcmp(text, "NotFound: loading model: model.bin") == 0;

  hyle_status_free(&com);
  hyle_status_free(&moved);
  hyle_status_free(&status);
  return ok ? 0 : 1;
}
