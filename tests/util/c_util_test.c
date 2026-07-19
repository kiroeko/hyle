#include "hyle/c/util.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int g_fails = 0;
#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #condition, __FILE__,    \
              __LINE__);                                                       \
      ++g_fails;                                                               \
    }                                                                          \
  } while (0)

static void swap_ints(void* context, uint64_t left, uint64_t right) {
  int* values = (int*)context;
  int temporary = values[left];
  values[left] = values[right];
  values[right] = temporary;
}

struct ref_destroy_state {
  int calls;
  void* object;
};

static void destroy_ref_object(void* object, void* context) {
  struct ref_destroy_state* state = (struct ref_destroy_state*)context;
  ++state->calls;
  state->object = object;
}

static void increment_int(void* context) {
  ++*(int*)context;
}

struct registry_factory_state {
  uintptr_t base;
};

static void* registry_factory(void* context, const void* arguments) {
  struct registry_factory_state* state =
      (struct registry_factory_state*)context;
  uintptr_t increment =
      arguments != NULL ? *(const uintptr_t*)arguments : 0;
  return (void*)(state->base + increment);
}

struct registry_visit_state {
  size_t count;
  int saw_fast;
  int stop_after_one;
};

static int visit_registry(void* context, const char* key, const char* help,
                          hyle_util_registry_priority priority) {
  struct registry_visit_state* state =
      (struct registry_visit_state*)context;
  ++state->count;
  if (strcmp(key, "engine") == 0 &&
      strcmp(help, "fast implementation") == 0 &&
      priority == HYLE_UTIL_REGISTRY_PREFERRED) {
    state->saw_fast = 1;
  }
  return !state->stop_after_one;
}

static void test_demangle(void) {
  char output[256];
  size_t length = hyle_util_demangle("plain_name", output, sizeof(output));
  CHECK(length == strlen("plain_name"));
  CHECK(strcmp(output, "plain_name") == 0);
  CHECK(hyle_util_last_error()[0] == '\0');

  CHECK(hyle_util_demangle("abcdef", output, 4) == 6);
  CHECK(strcmp(output, "abc") == 0);
  CHECK(hyle_util_demangle("size-only", NULL, 0) == 9);
  CHECK(hyle_util_demangle(NULL, output, sizeof(output)) == 0);
  CHECK(output[0] == '\0');

  CHECK(hyle_util_demangle("bad", NULL, 1) == 0);
  CHECK(hyle_util_last_error()[0] != '\0');
  CHECK(hyle_util_demangle("ok", output, sizeof(output)) == 2);
  CHECK(hyle_util_last_error()[0] == '\0');
}

static void test_rng_scalar_and_state(void) {
  hyle_util_rng invalid = {{0, 0, 0, 0}};
  CHECK(!hyle_util_rng_is_valid(&invalid));
  CHECK(hyle_util_last_error()[0] != '\0');
  CHECK(hyle_util_rng_next_u64(&invalid) == 0);
  CHECK(hyle_util_rng_next_u32(NULL) == 0);
  CHECK(hyle_util_rng_next_bool(NULL) == 0);
  CHECK(hyle_util_rng_next_float(NULL) == 0.0f);
  CHECK(hyle_util_rng_next_double(NULL) == 0.0);

  hyle_util_rng rng;
  CHECK(hyle_util_rng_seed(&rng, UINT64_C(1)));
  CHECK(hyle_util_rng_is_valid(&rng));
  CHECK(hyle_util_rng_next_u64(&rng) == UINT64_C(0xcfc5d07f6f03c29b));
  (void)hyle_util_rng_next_u32(&rng);
  {
    int value = hyle_util_rng_next_bool(&rng);
    CHECK(value == 0 || value == 1);
  }
  {
    float value = hyle_util_rng_next_float(&rng);
    CHECK(value >= 0.0f && value < 1.0f);
  }
  {
    double value = hyle_util_rng_next_double(&rng);
    CHECK(value >= 0.0 && value < 1.0);
  }

  {
    hyle_util_rng copy = hyle_util_rng_clone(&rng);
    CHECK(hyle_util_rng_next_u64(&copy) == hyle_util_rng_next_u64(&rng));
  }
  {
    hyle_util_rng zero = hyle_util_rng_clone(NULL);
    CHECK(!hyle_util_rng_is_valid(&zero));
  }
}

static void test_rng_ranges_batches_and_partitioning(void) {
  hyle_util_rng rng;
  uint64_t integers[64];
  float floats[64];
  double doubles[64];
  size_t i;
  CHECK(hyle_util_rng_seed(&rng, UINT64_C(42)));

  for (i = 0; i < 1000; ++i) {
    uint64_t u64 = hyle_util_rng_range_u64(&rng, 10, 20);
    uint32_t u32 = hyle_util_rng_range_u32(&rng, 30, 40);
    double real = hyle_util_rng_range_double(&rng, -5.0, 7.0);
    CHECK(u64 >= 10 && u64 <= 20);
    CHECK(u32 >= 30 && u32 <= 40);
    CHECK(real >= -5.0 && real < 7.0);
  }
  CHECK(hyle_util_rng_range_u64(&rng, 9, 8) == 0);
  CHECK(hyle_util_last_error()[0] != '\0');
  CHECK(hyle_util_rng_range_u32(&rng, 9, 8) == 0);
  CHECK(hyle_util_rng_range_double(&rng, NAN, 1.0) == 0.0);

  CHECK(hyle_util_rng_fill_u64(&rng, integers, 64));
  CHECK(hyle_util_rng_fill_float(&rng, floats, 64));
  CHECK(hyle_util_rng_fill_double(&rng, doubles, 64));
  CHECK(hyle_util_rng_fill_u64(&rng, NULL, 0));
  CHECK(hyle_util_rng_fill_float(&rng, NULL, 0));
  CHECK(hyle_util_rng_fill_double(&rng, NULL, 0));
  CHECK(!hyle_util_rng_fill_u64(&rng, NULL, 1));
  CHECK(!hyle_util_rng_fill_float(&rng, NULL, 1));
  CHECK(!hyle_util_rng_fill_double(&rng, NULL, 1));
  for (i = 0; i < 64; ++i) {
    CHECK(floats[i] >= 0.0f && floats[i] < 1.0f);
    CHECK(doubles[i] >= 0.0 && doubles[i] < 1.0);
  }

  {
    hyle_util_rng jumped = hyle_util_rng_clone(&rng);
    hyle_util_rng long_jumped = hyle_util_rng_clone(&rng);
    CHECK(hyle_util_rng_jump(&jumped));
    CHECK(hyle_util_rng_long_jump(&long_jumped));
    CHECK(hyle_util_rng_next_u64(&rng) !=
          hyle_util_rng_next_u64(&jumped));
    CHECK(hyle_util_rng_next_u64(&rng) !=
          hyle_util_rng_next_u64(&long_jumped));
  }
}

static void test_rng_shuffle_and_thread_local(void) {
  hyle_util_rng rng;
  int values[64];
  int seen[64] = {0};
  size_t i;
  CHECK(hyle_util_rng_seed(&rng, UINT64_C(99)));
  for (i = 0; i < 64; ++i) values[i] = (int)i;
  CHECK(hyle_util_rng_shuffle(&rng, 64, swap_ints, values));
  for (i = 0; i < 64; ++i) {
    CHECK(values[i] >= 0 && values[i] < 64);
    ++seen[values[i]];
  }
  for (i = 0; i < 64; ++i) CHECK(seen[i] == 1);
  CHECK(hyle_util_rng_shuffle(&rng, 1, NULL, NULL));
  CHECK(!hyle_util_rng_shuffle(&rng, 2, NULL, NULL));

  {
    hyle_util_rng* first = hyle_util_thread_rng();
    hyle_util_rng* second = hyle_util_thread_rng();
    CHECK(first != NULL);
    CHECK(first == second);
    CHECK(hyle_util_rng_is_valid(first));
  }
}

static void test_shared_reference(void) {
  int payload = 42;
  struct ref_destroy_state destroyed = {0, NULL};
  hyle_util_ref* ref =
      hyle_util_ref_create(&payload, destroy_ref_object, &destroyed);
  CHECK(ref != NULL);
  CHECK(hyle_util_ref_get(ref) == &payload);
  CHECK(hyle_util_ref_use_count(ref) == 1);
  CHECK(hyle_util_ref_is_unique(ref));
  CHECK(hyle_util_ref_retain(ref) == ref);
  CHECK(hyle_util_ref_use_count(ref) == 2);
  CHECK(!hyle_util_ref_is_unique(ref));
  hyle_util_ref_release(ref);
  CHECK(hyle_util_ref_use_count(ref) == 1);
  hyle_util_ref_release(ref);
  CHECK(destroyed.calls == 1);
  CHECK(destroyed.object == &payload);

  {
    hyle_util_ref* empty = hyle_util_ref_create(NULL, NULL, NULL);
    CHECK(empty != NULL);
    hyle_util_ref_release(empty);
  }
  CHECK(hyle_util_ref_retain(NULL) == NULL);
  CHECK(hyle_util_ref_get(NULL) == NULL);
  CHECK(hyle_util_ref_use_count(NULL) == 0);
  CHECK(!hyle_util_ref_is_unique(NULL));
  hyle_util_ref_release(NULL);
}

static void test_scope_guard(void) {
  int calls = 0;
  hyle_util_scope_guard* guard =
      hyle_util_scope_guard_create(increment_int, &calls);
  CHECK(guard != NULL);
  CHECK(hyle_util_scope_guard_is_active(guard));
  CHECK(hyle_util_scope_guard_run(guard));
  CHECK(calls == 1);
  CHECK(!hyle_util_scope_guard_is_active(guard));
  CHECK(hyle_util_scope_guard_run(guard));
  CHECK(calls == 1);
  hyle_util_scope_guard_destroy(guard);
  CHECK(calls == 1);

  guard = hyle_util_scope_guard_create(increment_int, &calls);
  CHECK(hyle_util_scope_guard_dismiss(guard));
  hyle_util_scope_guard_destroy(guard);
  CHECK(calls == 1);

  guard = hyle_util_scope_guard_create(increment_int, &calls);
  hyle_util_scope_guard_destroy(guard);
  CHECK(calls == 2);
  hyle_util_scope_guard_destroy(NULL);
  CHECK(hyle_util_scope_guard_create(NULL, NULL) == NULL);
  CHECK(!hyle_util_scope_guard_run(NULL));
  CHECK(!hyle_util_scope_guard_dismiss(NULL));
  CHECK(!hyle_util_scope_guard_is_active(NULL));
}

static void test_runtime_registry(void) {
  hyle_util_registry* registry = hyle_util_registry_create();
  struct registry_factory_state fallback = {100};
  struct registry_factory_state preferred = {1000};
  struct registry_visit_state visited = {0, 0, 0};
  uintptr_t increment = 7;
  char help[8];
  CHECK(registry != NULL);
  CHECK(hyle_util_registry_register(
      registry, "engine", registry_factory, &fallback,
      HYLE_UTIL_REGISTRY_FALLBACK, "fallback"));
  CHECK(hyle_util_registry_has(registry, "engine"));
  CHECK(!hyle_util_registry_has(registry, "missing"));
  CHECK(hyle_util_registry_size(registry) == 1);
  CHECK((uintptr_t)hyle_util_registry_create_object(
            registry, "engine", &increment) == 107);

  CHECK(hyle_util_registry_register(
      registry, "engine", registry_factory, &preferred,
      HYLE_UTIL_REGISTRY_PREFERRED, "fast implementation"));
  CHECK((uintptr_t)hyle_util_registry_create_object(
            registry, "engine", &increment) == 1007);
  CHECK(hyle_util_registry_help(registry, "engine", help, sizeof(help)) ==
        strlen("fast implementation"));
  CHECK(strcmp(help, "fast im") == 0);
  CHECK(hyle_util_registry_help(registry, "engine", NULL, 0) ==
        strlen("fast implementation"));
  CHECK(hyle_util_registry_for_each(registry, visit_registry, &visited));
  CHECK(visited.count == 1);
  CHECK(visited.saw_fast);
  CHECK(hyle_util_registry_register(
      registry, "second", registry_factory, &fallback,
      HYLE_UTIL_REGISTRY_DEFAULT, "second implementation"));
  visited.count = 0;
  visited.stop_after_one = 1;
  CHECK(hyle_util_registry_for_each(registry, visit_registry, &visited));
  CHECK(visited.count == 1);

  CHECK(hyle_util_registry_register(
      registry, "engine", registry_factory, &fallback,
      HYLE_UTIL_REGISTRY_FALLBACK, "ignored"));
  CHECK(!hyle_util_registry_register(
      registry, "engine", registry_factory, &preferred,
      HYLE_UTIL_REGISTRY_PREFERRED, "duplicate"));
  CHECK(hyle_util_last_error()[0] != '\0');
  CHECK(hyle_util_registry_create_object(registry, "missing", NULL) == NULL);
  CHECK(hyle_util_registry_help(registry, "missing", help, sizeof(help)) == 0);
  CHECK(!hyle_util_registry_register(
      registry, NULL, registry_factory, NULL,
      HYLE_UTIL_REGISTRY_DEFAULT, NULL));
  CHECK(!hyle_util_registry_register(
      registry, "bad", NULL, NULL, HYLE_UTIL_REGISTRY_DEFAULT, NULL));
  CHECK(!hyle_util_registry_register(
      registry, "bad", registry_factory, NULL,
      (hyle_util_registry_priority)99, NULL));
  CHECK(!hyle_util_registry_for_each(registry, NULL, NULL));
  CHECK(!hyle_util_registry_has(NULL, "x"));
  CHECK(hyle_util_registry_size(NULL) == 0);

  {
    char owned_key[] = "owned";
    char owned_help[] = "owned help";
    CHECK(hyle_util_registry_register(
        registry, owned_key, registry_factory, &fallback,
        HYLE_UTIL_REGISTRY_DEFAULT, owned_help));
    owned_key[0] = 'X';
    owned_help[0] = 'X';
    CHECK(hyle_util_registry_has(registry, "owned"));
    CHECK(hyle_util_registry_help(
              registry, "owned", help, sizeof(help)) ==
          strlen("owned help"));
    CHECK(strcmp(help, "owned h") == 0);
  }

  hyle_util_registry_destroy(registry);
  hyle_util_registry_destroy(NULL);
}

static void test_spin_lock(void) {
  hyle_util_spin_lock* lock = hyle_util_spin_lock_create();
  CHECK(lock != NULL);
  CHECK(!hyle_util_spin_lock_is_locked(lock));
  CHECK(hyle_util_spin_lock_try_lock(lock));
  CHECK(hyle_util_spin_lock_is_locked(lock));
  CHECK(!hyle_util_spin_lock_try_lock(lock));
  CHECK(!hyle_util_spin_lock_destroy(lock));
  CHECK(hyle_util_spin_lock_unlock(lock));
  CHECK(hyle_util_spin_lock_lock(lock));
  CHECK(hyle_util_spin_lock_unlock(lock));
  CHECK(hyle_util_spin_lock_destroy(lock));
  CHECK(hyle_util_spin_lock_destroy(NULL));
  CHECK(!hyle_util_spin_lock_lock(NULL));
  CHECK(!hyle_util_spin_lock_try_lock(NULL));
  CHECK(!hyle_util_spin_lock_unlock(NULL));
  CHECK(!hyle_util_spin_lock_is_locked(NULL));
}

int main(void) {
  hyle_util_cpu_relax();
  test_demangle();
  test_rng_scalar_and_state();
  test_rng_ranges_batches_and_partitioning();
  test_rng_shuffle_and_thread_local();
  test_shared_reference();
  test_scope_guard();
  test_runtime_registry();
  test_spin_lock();
  if (g_fails == 0) puts("hyle util C ABI: ok");
  return g_fails == 0 ? 0 : 1;
}
