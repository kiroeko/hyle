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

static HYLE_C_FORCE_INLINE int hyle_test_c_force_inline(int value) {
  return value + 1;
}

HYLE_C_NODISCARD static int hyle_test_c_must_inspect(void) { return 17; }

HYLE_C_NOINLINE int hyle_test_c_branch_contract(int value) {
  if (HYLE_C_LIKELY(value >= 0)) return hyle_test_c_force_inline(value);
  if (HYLE_C_UNLIKELY(value == -1)) return 0;
  HYLE_C_UNREACHABLE();
}

int main(void) {
  (void)"HYLE_C_CORE_COVERAGE_BODY_MARKER";
  uint32_t v = hyle_version();
  const volatile uint32_t constructive_size =
      HYLE_STATIC_CONSTRUCTIVE_INTERFERENCE_SIZE;
  const volatile uint32_t destructive_size =
      HYLE_STATIC_DESTRUCTIVE_INTERFERENCE_SIZE;
  CHECK(v == HYLE_VERSION_PACKED);
  CHECK(constructive_size == 64);
  CHECK(destructive_size == 128);
  CHECK(constructive_size <= destructive_size);
  CHECK(hyle_version_major(v) == 0u);
  CHECK(hyle_version_minor(v) == 1u);
  CHECK(hyle_version_patch(v) == 0u);

  {
    const int values[] = {1, 2, 3, 4};
    CHECK(hyle_test_c_compiler_contract_sum(values, 4) == 10);
    CHECK(hyle_test_c_compiler_contract_default_unroll(values, 4) == 10);
    CHECK(hyle_test_c_branch_contract(4) == 5);
    CHECK(hyle_test_c_must_inspect() == 17);
    HYLE_C_FATAL_ASSERT(values[0] == 1);
  }

  {
    unsigned char cache_line[HYLE_STATIC_CONSTRUCTIVE_INTERFERENCE_SIZE] = {0};
    hyle_prefetch_read(cache_line);
    hyle_prefetch_read_non_temporal(cache_line);
    hyle_prefetch_write(cache_line);
  }

  const char* s = hyle_version_string();
  CHECK(s != NULL);
  CHECK(strcmp(s, HYLE_VERSION_STRING) == 0);

  hyle_intrusive_list source = HYLE_INTRUSIVE_LIST_INIT(source);
  hyle_intrusive_list destination = HYLE_INTRUSIVE_LIST_INIT(destination);
  test_node a = {1, HYLE_INTRUSIVE_HOOK_INIT};
  test_node b = {2, HYLE_INTRUSIVE_HOOK_INIT};
  test_node c = {3, HYLE_INTRUSIVE_HOOK_INIT};
  test_node d = {4, HYLE_INTRUSIVE_HOOK_INIT};
  hyle_intrusive_list_init(&destination);
  hyle_intrusive_hook_init(&a.hook);
  CHECK(hyle_intrusive_list_empty(NULL));
  CHECK(hyle_intrusive_list_empty(&source));
  CHECK(hyle_intrusive_list_front(&source) == NULL);
  CHECK(hyle_intrusive_list_back(&source) == NULL);
  CHECK(hyle_intrusive_list_pop_front(&source) == NULL);
  CHECK(hyle_intrusive_list_pop_back(&source) == NULL);
  CHECK(hyle_intrusive_list_next(NULL, NULL) == NULL);
  CHECK(hyle_intrusive_list_prev_const(NULL, NULL) == NULL);
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
  CHECK(hyle_intrusive_list_next(&destination,
                                 hyle_intrusive_list_front(&destination)) ==
        &a.hook);
  CHECK(hyle_intrusive_list_prev(&destination,
                                 hyle_intrusive_list_back(&destination)) ==
        &a.hook);
  {
    const hyle_intrusive_list* view = &destination;
    const hyle_intrusive_list_hook* cursor =
        hyle_intrusive_list_front_const(view);
    int forward_sum = 0;
    int reverse_sum = 0;
    while (cursor != NULL) {
      forward_sum += HYLE_INTRUSIVE_CONST_ENTRY(cursor, test_node, hook)->value;
      cursor = hyle_intrusive_list_next_const(view, cursor);
    }
    cursor = hyle_intrusive_list_back_const(view);
    while (cursor != NULL) {
      reverse_sum += HYLE_INTRUSIVE_CONST_ENTRY(cursor, test_node, hook)->value;
      cursor = hyle_intrusive_list_prev_const(view, cursor);
    }
    CHECK(forward_sum == 6);
    CHECK(reverse_sum == 6);
  }
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

  /* Arbitrary-position insert and whole-batch splice are natural C tasks;
   * NULL position is the documented end() spelling. */
  hyle_intrusive_list_push_back(&source, &a.hook);
  hyle_intrusive_list_push_back(&source, &b.hook);
  hyle_intrusive_list_push_back(&destination, &c.hook);
  CHECK(!hyle_intrusive_list_try_insert_before(NULL, NULL, &d.hook));
  CHECK(hyle_intrusive_list_splice_before(&destination, &c.hook, &source));
  CHECK(hyle_intrusive_list_empty(&source));
  CHECK(hyle_intrusive_list_try_insert_before(&destination, &b.hook, &d.hook));
  CHECK(!hyle_intrusive_list_try_insert_before(&destination, NULL, &d.hook));
  CHECK(hyle_intrusive_list_front(&destination) == &a.hook);
  CHECK(hyle_intrusive_list_next(&destination, &a.hook) == &d.hook);
  CHECK(hyle_intrusive_list_next(&destination, &d.hook) == &b.hook);
  CHECK(hyle_intrusive_list_back(&destination) == &c.hook);
  CHECK(!hyle_intrusive_list_splice_before(&destination, NULL, &destination));
  hyle_intrusive_list_clear(&destination);

  if (g_fails) {
    fprintf(stderr, "c_core_test: %d failed\n", g_fails);
    return 1;
  }
  printf("c_core_test: all checks passed\n");
  return 0;
}
