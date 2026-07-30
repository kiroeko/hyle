#include "hyle/c/util.h"

#include <stdint.h>

static int consume(hyle_status status) {
  const int ok = hyle_status_is_ok(status);
  hyle_status_free(&status);
  return ok;
}

static void* create_value(void* context, const void* arguments) {
  const uintptr_t base = *(const uintptr_t*)context;
  const uintptr_t increment = *(const uintptr_t*)arguments;
  return (void*)(base + increment);
}

int main(void) {
  hyle_util_registry* registry = NULL;
  hyle_util_rng rng;
  uint64_t random_values[16];
  uintptr_t base = 100;
  uintptr_t increment = 7;
  void* object = NULL;

  if (!hyle_util_rng_seed(&rng, 42) ||
      !hyle_util_rng_fill_u64(&rng, random_values, 16)) {
    return 1;
  }
  if (!consume(hyle_util_registry_create_system(&registry))) return 2;
  if (!consume(hyle_util_registry_register_cstr(
          registry, "sample", create_value, &base,
          HYLE_UTIL_REGISTRY_DEFAULT, "installed C package probe"))) {
    return 3;
  }
  if (!consume(hyle_util_registry_freeze(registry))) return 4;
  if (!consume(hyle_util_registry_create_object_cstr(
          &object, registry, "sample", &increment))) {
    return 5;
  }
  hyle_util_registry_destroy(registry);
  return (uintptr_t)object == 107 && random_values[0] != random_values[1]
             ? 0
             : 6;
}
