#include <stdio.h>
#include <string.h>

#include "hyle/c/error.h"

int main(void) {
  hyle_status status = hyle_status_not_found("model.bin");
  char text[64];
  const size_t needed = hyle_status_to_string(status, text, sizeof(text));
  const int ok = needed == strlen("NotFound: model.bin") &&
                 strcmp(text, "NotFound: model.bin") == 0;
  printf("installed C ABI: %s\n", text);
  hyle_status_free(&status);
  return ok ? 0 : 1;
}
