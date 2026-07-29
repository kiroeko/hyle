#include <hyle/c/core.h>

typedef struct node {
  int value;
  hyle_intrusive_list_hook hook;
} node;

static HYLE_C_FORCE_INLINE int twice(int value) { return value * 2; }

int main(void) {
  hyle_intrusive_list queue;
  node item = {21, {NULL, NULL}};
  hyle_intrusive_list_init(&queue);
  if (!hyle_intrusive_list_try_push_back(&queue, &item.hook)) return 1;
  hyle_prefetch_read(&item);
  node* result = HYLE_INTRUSIVE_ENTRY(
      hyle_intrusive_list_pop_front(&queue), node, hook);
  return hyle_version() == HYLE_VERSION_PACKED &&
                 result == &item && twice(result->value) == 42
             ? 0
             : 1;
}
