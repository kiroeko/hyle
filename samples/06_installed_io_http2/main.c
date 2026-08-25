#include <hyle/c/io.h>

#include <stdio.h>
#include <stdint.h>

#if defined(_MSC_VER)
typedef char installed_io_backend_must_be_u32[
    sizeof(hyle_io_backend) == sizeof(uint32_t) ? 1 : -1];
typedef char installed_io_http2_role_must_be_u32[
    sizeof(hyle_io_http2_role) == sizeof(uint32_t) ? 1 : -1];
typedef char installed_io_capabilities_candidate_size[
    sizeof(hyle_io_capabilities) == 16 ? 1 : -1];
#else
_Static_assert(sizeof(hyle_io_backend) == sizeof(uint32_t),
               "installed IO backend ABI width");
_Static_assert(sizeof(hyle_io_http2_role) == sizeof(uint32_t),
               "installed HTTP/2 role ABI width");
_Static_assert(sizeof(hyle_io_capabilities) == 16,
               "installed IO capabilities candidate layout");
#endif

int main(void) {
  if (hyle_io_http2_available() != 1) {
    fputs("installed HYLE package has no HTTP/2 protocol core\n", stderr);
    return 1;
  }
  puts("installed HYLE io package: HTTP/2 available");
  return 0;
}
