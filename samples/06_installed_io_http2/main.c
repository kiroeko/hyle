#include <hyle/c/io.h>

#include <stdio.h>

int main(void) {
  if (hyle_io_http2_available() != 1) {
    fputs("installed HYLE package has no HTTP/2 protocol core\n", stderr);
    return 1;
  }
  puts("installed HYLE io package: HTTP/2 available");
  return 0;
}
