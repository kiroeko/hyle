#include <stdio.h>
#include <string.h>

#include "hyle/c/error.h"
#include "hyle/c/mem.h"

int main(void) {
  hyle_status status = hyle_status_not_found("model.bin");
  char text[64];
  const size_t needed = hyle_status_to_string(status, text, sizeof(text));
  hyle_allocator allocator = hyle_system_allocator();
  void *block = hyle_alloc(&allocator, 64, 0);
  const int ok = needed == strlen("NotFound: model.bin") &&
                 strcmp(text, "NotFound: model.bin") == 0 && block != NULL;
  printf("installed C ABI: %s\n", text);
  hyle_free(&allocator, block, 64, 0);
  hyle_status_free(&status);
  return ok ? 0 : 1;
}
