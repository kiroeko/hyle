#include <hyle/c/error.h>

#include <string.h>

int main(void) {
  hyle_status status = hyle_status_not_found("model.bin");
  hyle_status_add_context(&status, "loading model");

  hyle_status copy = hyle_status_clone(status);
  hyle_status moved = HYLE_STATUS_INIT;
  hyle_status_move(&moved, &copy);

  char text[64];
  const size_t needed = hyle_status_to_string(moved, text, sizeof(text));
  const int ok = hyle_status_get_code(moved) == HYLE_STATUS_NOT_FOUND &&
                 hyle_status_is_ok(copy) &&
                 hyle_status_code_from_win32(5) ==
                     HYLE_STATUS_PERMISSION_DENIED &&
                 needed == strlen("NotFound: loading model: model.bin") &&
                 strcmp(text, "NotFound: loading model: model.bin") == 0;

  hyle_status_free(&moved);
  hyle_status_free(&status);
  return ok ? 0 : 1;
}
