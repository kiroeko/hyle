/* Pure C consumer of core's public C source/ABI headers (no C++). */
#include "hyle/c/compiler.h"

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

int hyle_test_c_compiler_contract_sum(const int* HYLE_C_RESTRICT values,
                                      size_t count) {
  int sum = 0;
  HYLE_C_ASSUME(values != NULL);
  HYLE_C_UNROLL(4)
  for (size_t i = 0; i < count; ++i) sum += values[i];
  return sum;
}

HYLE_C_FLATTEN int hyle_test_c_compiler_contract_default_unroll(
    const int* values, size_t count) {
  int sum = 0;
  HYLE_C_DEFAULT_UNROLL
  for (size_t i = 0; i < count; ++i) sum += values[i];
  return sum;
}

int main(void) {
  (void)"HYLE_C_CORE_COVERAGE_BODY_MARKER";
  uint32_t v = hyle_version();
  const volatile uint32_t cache_line_size = HYLE_STATIC_CACHE_LINE_SIZE;
  CHECK(v == HYLE_VERSION_PACKED);
  CHECK(cache_line_size == 64 || cache_line_size == 128);
  CHECK(hyle_version_major(v) == 0u);
  CHECK(hyle_version_minor(v) == 1u);
  CHECK(hyle_version_patch(v) == 0u);

  {
    const int values[] = {1, 2, 3, 4};
    CHECK(hyle_test_c_compiler_contract_sum(values, 4) == 10);
    CHECK(hyle_test_c_compiler_contract_default_unroll(values, 4) == 10);
  }

  {
    unsigned char cache_line[HYLE_STATIC_CACHE_LINE_SIZE] = {0};
    hyle_prefetch_read(cache_line);
    hyle_prefetch_read_non_temporal(cache_line);
    hyle_prefetch_write(cache_line);
  }

  const char* s = hyle_version_string();
  CHECK(s != NULL);
  CHECK(strcmp(s, HYLE_VERSION_STRING) == 0);

  hyle_intrusive_list source;
  hyle_intrusive_list destination;
  test_node a = {1, {NULL, NULL}};
  test_node b = {2, {NULL, NULL}};
  test_node c = {3, {NULL, NULL}};
  test_node d = {4, {NULL, NULL}};
  hyle_intrusive_list_init(&source);
  hyle_intrusive_list_init(&destination);
  hyle_intrusive_hook_init(&a.hook);
  CHECK(hyle_intrusive_list_empty(NULL));
  CHECK(hyle_intrusive_list_empty(&source));
  CHECK(hyle_intrusive_list_front(&source) == NULL);
  CHECK(hyle_intrusive_list_back(&source) == NULL);
  CHECK(hyle_intrusive_list_pop_front(&source) == NULL);
  CHECK(hyle_intrusive_list_pop_back(&source) == NULL);
  CHECK(hyle_intrusive_list_try_push_back(&source, &a.hook));
  hyle_intrusive_list_push_back(&source, &b.hook);
  CHECK(hyle_intrusive_list_back(&source) == &b.hook);
  CHECK(!hyle_intrusive_list_try_push_back(&source, &a.hook));
  CHECK(!hyle_intrusive_list_try_push_front(NULL, &c.hook));
  CHECK(hyle_intrusive_list_try_push_front(&destination, &c.hook));
  CHECK(!hyle_intrusive_list_append(NULL, &source));
  CHECK(!hyle_intrusive_list_append(&source, NULL));
  CHECK(!hyle_intrusive_list_append(&source, &source));
  CHECK(hyle_intrusive_list_append(&destination, &source));
  CHECK(hyle_intrusive_list_empty(&source));
  CHECK(HYLE_INTRUSIVE_ENTRY(hyle_intrusive_list_pop_front(&destination), test_node, hook) == &c);
  CHECK(HYLE_INTRUSIVE_ENTRY(hyle_intrusive_list_front(&destination), test_node, hook) == &a);
  hyle_intrusive_hook_unlink(&a.hook);
  CHECK(!hyle_intrusive_hook_linked(&a.hook));
  CHECK(HYLE_INTRUSIVE_ENTRY(hyle_intrusive_list_pop_back(&destination), test_node, hook) == &b);
  CHECK(hyle_intrusive_list_empty(&destination));

  hyle_intrusive_list_push_front(&destination, &c.hook);
  hyle_intrusive_list_push_back(&destination, &d.hook);
  hyle_intrusive_list_clear(&destination);
  CHECK(hyle_intrusive_list_empty(&destination));
  CHECK(!hyle_intrusive_hook_linked(&c.hook));
  CHECK(!hyle_intrusive_hook_linked(&d.hook));

  if (g_fails) {
    fprintf(stderr, "c_core_test: %d failed\n", g_fails);
    return 1;
  }
  printf("c_core_test: all checks passed\n");
  return 0;
}
