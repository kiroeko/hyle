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
  node first = {20, HYLE_INTRUSIVE_HOOK_INIT};
  node second = {22, HYLE_INTRUSIVE_HOOK_INIT};
  const int values[] = {20, 22};
  if (!hyle_intrusive_list_try_push_back(&queue, &second.hook)) return 1;
  if (!hyle_intrusive_list_try_insert_before(&queue, &second.hook,
                                             &first.hook))
    return 1;
  hyle_prefetch_read(&first);
  hyle_intrusive_list_hook* first_hook = hyle_intrusive_list_pop_front(&queue);
  hyle_intrusive_list_hook* second_hook = hyle_intrusive_list_pop_front(&queue);
  if (first_hook == NULL || second_hook == NULL) return 1;
  node* first_result = HYLE_INTRUSIVE_ENTRY(first_hook, node, hook);
  node* second_result = HYLE_INTRUSIVE_ENTRY(second_hook, node, hook);
  return hyle_version() == HYLE_VERSION_PACKED &&
                 first_result == &first && second_result == &second &&
                 hyle_intrusive_list_empty(&queue) &&
                 twice(first_result->value) == 40 &&
                 first_result->value + second_result->value == 42 &&
                 sum_values(values, 2) == 42
             ? 0
             : 1;
}
