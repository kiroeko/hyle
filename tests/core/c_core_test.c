/* Pure C consumer of include/hyle/c/core.h (no C++). */
#include "hyle/c/core.h"

#include <stdio.h>
#include <string.h>

static int g_fails = 0;
#define CHECK(c)                                                              \
  do {                                                                        \
    if (!(c)) {                                                               \
      fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #c, __FILE__, __LINE__); \
      ++g_fails;                                                              \
    }                                                                         \
  } while (0)

int main(void) {
  uint32_t v = hyle_version();
  CHECK(v == ((0u << 16) | (1u << 8) | 0u)); /* 0.1.0 */
  CHECK(hyle_version_major(v) == 0u);
  CHECK(hyle_version_minor(v) == 1u);
  CHECK(hyle_version_patch(v) == 0u);

  const char* s = hyle_version_string();
  CHECK(s != NULL);
  CHECK(strcmp(s, "0.1.0") == 0);

  if (g_fails) {
    fprintf(stderr, "c_core_test: %d failed\n", g_fails);
    return 1;
  }
  printf("c_core_test: all checks passed\n");
  return 0;
}
