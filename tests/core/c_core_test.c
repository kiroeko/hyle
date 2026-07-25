/* Pure C consumer of include/hyle/c/core.h (no C++). */
#include <stdio.h>
#include <string.h>

#include "hyle/c/core.h"

typedef struct test_node {
  int value;
  hyle_intrusive_list_hook hook;
} test_node;

static int g_fails = 0;
#define CHECK(c)                                                             \
  do {                                                                       \
    if (!(c)) {                                                              \
      fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #c, __FILE__, __LINE__); \
      ++g_fails;                                                             \
    }                                                                        \
  } while (0)

int main(void) {
  (void)"HYLE_C_CORE_COVERAGE_BODY_MARKER";
  uint32_t v = hyle_version();
  CHECK(v == HYLE_VERSION_PACKED);
  CHECK(HYLE_STATIC_CACHE_LINE_SIZE == 64 || HYLE_STATIC_CACHE_LINE_SIZE == 128);
  CHECK(hyle_version_major(v) == 0u);
  CHECK(hyle_version_minor(v) == 1u);
  CHECK(hyle_version_patch(v) == 0u);

  const char* s = hyle_version_string();
  CHECK(s != NULL);
  CHECK(strcmp(s, HYLE_VERSION_STRING) == 0);

  hyle_intrusive_list source;
  hyle_intrusive_list destination;
  test_node a = {1, {NULL, NULL}};
  test_node b = {2, {NULL, NULL}};
  test_node c = {3, {NULL, NULL}};
  hyle_intrusive_list_init(&source);
  hyle_intrusive_list_init(&destination);
  CHECK(hyle_intrusive_list_empty(&source));
  CHECK(hyle_intrusive_list_try_push_back(&source, &a.hook));
  hyle_intrusive_list_push_back(&source, &b.hook);
  CHECK(!hyle_intrusive_list_try_push_back(&source, &a.hook));
  CHECK(!hyle_intrusive_list_try_push_front(NULL, &c.hook));
  CHECK(hyle_intrusive_list_try_push_front(&destination, &c.hook));
  CHECK(hyle_intrusive_list_append(&destination, &source));
  CHECK(hyle_intrusive_list_empty(&source));
  CHECK(HYLE_INTRUSIVE_ENTRY(hyle_intrusive_list_pop_front(&destination), test_node, hook) == &c);
  CHECK(HYLE_INTRUSIVE_ENTRY(hyle_intrusive_list_front(&destination), test_node, hook) == &a);
  hyle_intrusive_hook_unlink(&a.hook);
  CHECK(!hyle_intrusive_hook_linked(&a.hook));
  CHECK(HYLE_INTRUSIVE_ENTRY(hyle_intrusive_list_pop_back(&destination), test_node, hook) == &b);
  CHECK(hyle_intrusive_list_empty(&destination));

  if (g_fails) {
    fprintf(stderr, "c_core_test: %d failed\n", g_fails);
    return 1;
  }
  printf("c_core_test: all checks passed\n");
  return 0;
}
