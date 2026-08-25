#include <hyle/c/core.h>

#if !defined(__STDC_VERSION__) || __STDC_VERSION__ < 201710L
#error "hyle::core must propagate its public C17 source contract"
#endif

typedef struct node {
  int value;
  hyle_intrusive_list_hook hook;
} node;

static HYLE_C_FORCE_INLINE int twice(int value) { return value * 2; }

HYLE_C_NODISCARD static int
sum_values(const int* HYLE_C_RESTRICT values, size_t count) {
  HYLE_C_ASSUME(values != NULL);
  int total = 0;
  HYLE_C_UNROLL(4)
  for (size_t index = 0; index < count; ++index) total += values[index];
  return total;
}

int main(void) {
  hyle_intrusive_list queue = HYLE_INTRUSIVE_LIST_INIT(queue);
  node item = {21, HYLE_INTRUSIVE_HOOK_INIT};
  const int values[] = {20, 22};
  if (!hyle_intrusive_list_try_push_back(&queue, &item.hook)) return 1;
  hyle_prefetch_read(&item);
  node* result = HYLE_INTRUSIVE_ENTRY(
      hyle_intrusive_list_pop_front(&queue), node, hook);
  return hyle_version() == HYLE_VERSION_PACKED &&
                 result == &item && twice(result->value) == 42 &&
                 sum_values(values, 2) == 42
             ? 0
             : 1;
}
