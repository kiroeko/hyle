#include <hyle/c/mem.h>

#include <stdint.h>

static int consume(hyle_status status) {
  const int ok = hyle_status_is_ok(status);
  hyle_status_free(&status);
  return ok;
}

int main(void) {
  hyle_allocator system = hyle_system_allocator();
  uint32_t* values = (uint32_t*)hyle_alloc(
      &system, 16 * sizeof(uint32_t), HYLE_ALIGNOF(uint32_t));
  if (values == NULL) return 1;
  values[15] = 42;
  hyle_free(&system, values, 16 * sizeof(uint32_t),
            HYLE_ALIGNOF(uint32_t));

  hyle_quota_allocator* quota = NULL;
  if (!consume(hyle_quota_allocator_create_default(&quota, 128 * 1024))) {
    return 2;
  }
  hyle_region* region = NULL;
  hyle_allocator bounded = hyle_quota_allocator_as_allocator(quota);
  if (!consume(hyle_region_create(&region, bounded, NULL))) {
    hyle_quota_allocator_destroy(quota);
    return 3;
  }
  void* scratch = hyle_region_allocate(region, 4096, 64);
  const int ok = scratch != NULL &&
                 ((uintptr_t)scratch % 64) == 0 &&
                 hyle_quota_allocator_used(quota) != 0;
  hyle_region_destroy(region);
  const int balanced = hyle_quota_allocator_used(quota) == 0;
  hyle_quota_allocator_destroy(quota);
  return ok && balanced ? 0 : 4;
}
