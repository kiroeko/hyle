#include <hyle/c/io.h>

#include <string.h>

int main(void) {
  /* The default options call through mem's system allocator even though this
   * C-syntax fixture links only hyle::io. */
  const hyle_io_proactor_options options = hyle_io_proactor_options_default();
  if (options.backing.alloc == NULL || options.backing.free == NULL) return 1;
  if (strcmp(hyle_io_backend_name(HYLE_IO_BACKEND_NONE), "none") != 0) return 2;

  /* This io FFI candidate result is an owning error Status. Force the installed
   * schema fixture to resolve and exercise hyle_status_* rather than leaking the
   * cross-module value or relying on an empty target-link smoke. */
  hyle_io_uring_extension extension = HYLE_IO_URING_EXTENSION_INIT;
  hyle_status status = hyle_io_uring_extension_get(NULL, &extension);
  const int ok = hyle_status_is_invalid_argument(status) &&
                 hyle_status_has_message(status) &&
                 extension.proactor == NULL;
  hyle_status_free(&status);
  return ok ? 0 : 3;
}
