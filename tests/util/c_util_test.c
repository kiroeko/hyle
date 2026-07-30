/* Pure-C util-runtime consumer and C-surface coverage probe. */
#include "hyle/c/util.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HYLE_C_UTIL_COVERAGE_BODY_MARKER 1

static int g_fails = 0;
#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #condition, __FILE__,    \
              __LINE__);                                                       \
      ++g_fails;                                                               \
    }                                                                          \
  } while (0)

static int consume_ok(hyle_status status) {
  int ok = hyle_status_is_ok(status);
  if (!ok) {
    char message[256];
    hyle_status_to_string(status, message, sizeof(message));
    fprintf(stderr, "unexpected status: %s\n", message);
  }
  hyle_status_free(&status);
  return ok;
}

static int consume_code(hyle_status status, hyle_status_code code) {
  int matches = hyle_status_get_code(status) == code;
  hyle_status_free(&status);
  return matches;
}

struct allocator_probe {
  hyle_allocator backing;
  size_t allocations;
  size_t frees;
  int fail;
};

static void* probe_allocate(void* context, size_t size, size_t alignment) {
  struct allocator_probe* probe = (struct allocator_probe*)context;
  if (probe->fail) return NULL;
  ++probe->allocations;
  return probe->backing.alloc(probe->backing.state, size, alignment);
}

static void probe_free(void* context, void* pointer, size_t size,
                       size_t alignment) {
  struct allocator_probe* probe = (struct allocator_probe*)context;
  ++probe->frees;
  probe->backing.free(probe->backing.state, pointer, size, alignment);
}

static void* probe_reallocate(void* context, void* pointer, size_t old_size,
                              size_t new_size, size_t alignment) {
  struct allocator_probe* probe = (struct allocator_probe*)context;
  if (probe->fail) return NULL;
  ++probe->allocations;
  return probe->backing.realloc(probe->backing.state, pointer, old_size,
                                new_size, alignment);
}

static hyle_allocator probe_allocator(struct allocator_probe* probe) {
  hyle_allocator result = {
      probe_allocate, probe_free, probe_reallocate, probe};
  return result;
}

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

struct registry_factory_state {
  uintptr_t base;
};

static void* registry_factory(void* context, const void* arguments) {
  struct registry_factory_state* state =
      (struct registry_factory_state*)context;
  uintptr_t increment = arguments != NULL ? *(const uintptr_t*)arguments : 0;
  return (void*)(state->base + increment);
}

struct registry_visit_state {
  const hyle_util_registry* registry;
  size_t count;
  int saw_fast;
  int stop_after_one;
};

static int view_equals_cstr(hyle_container_string_view view,
                            const char* expected) {
  const size_t size = strlen(expected);
  return view.size == size &&
         (size == 0 || memcmp(view.data, expected, size) == 0);
}

static int visit_registry(void* context, hyle_container_string_view key,
                          hyle_container_string_view help,
                          hyle_util_registry_priority priority) {
  struct registry_visit_state* state =
      (struct registry_visit_state*)context;
  ++state->count;
  CHECK(hyle_util_registry_has(state->registry, key));
  if (view_equals_cstr(key, "engine") &&
      view_equals_cstr(help, "fast implementation") &&
      priority == HYLE_UTIL_REGISTRY_PREFERRED) {
    state->saw_fast = 1;
  }
  return !state->stop_after_one;
}

static void test_demangle(void) {
  char output[256];
  size_t length = 0;
  CHECK(consume_ok(hyle_util_demangle(
      &length, "plain_name", output, sizeof(output))));
  CHECK(length == strlen("plain_name"));
  CHECK(strcmp(output, "plain_name") == 0);

  CHECK(consume_ok(hyle_util_demangle(&length, "abcdef", output, 4)));
  CHECK(length == 6);
  CHECK(strcmp(output, "abc") == 0);
  CHECK(consume_ok(hyle_util_demangle(&length, "size-only", NULL, 0)));
  CHECK(length == 9);
  CHECK(consume_ok(hyle_util_demangle(&length, NULL, output, sizeof(output))));
  CHECK(length == 0 && output[0] == '\0');
  CHECK(consume_code(hyle_util_demangle(&length, "bad", NULL, 1),
                     HYLE_STATUS_INVALID_ARGUMENT));
}

static void test_rng_scalar_and_state(void) {
  hyle_util_rng invalid = {{0, 0, 0, 0}};
  CHECK(!hyle_util_rng_is_valid(&invalid));
  CHECK(hyle_util_rng_next_u64(&invalid) == 0);
  CHECK(hyle_util_rng_next_u32(NULL) == 0);
  CHECK(hyle_util_rng_next_bool(NULL) == 0);
  CHECK(hyle_util_rng_next_float(NULL) == 0.0f);
  CHECK(hyle_util_rng_next_double(NULL) == 0.0);

  {
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
    CHECK(!hyle_util_rng_is_valid(&(hyle_util_rng){{0, 0, 0, 0}}));
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
    CHECK(hyle_util_rng_next_u64(&rng) != hyle_util_rng_next_u64(&jumped));
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
    CHECK(first != NULL && first == second && hyle_util_rng_is_valid(first));
  }
}

static void test_shared_reference(void) {
  int payload = 42;
  struct ref_destroy_state destroyed = {0, NULL};
  struct allocator_probe allocator = {
      hyle_system_allocator(), 0, 0, 0};
  hyle_util_ref* ref = NULL;
  CHECK(consume_ok(hyle_util_ref_create(
      &ref, &payload, destroy_ref_object, &destroyed,
      probe_allocator(&allocator))));
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
  CHECK(destroyed.calls == 1 && destroyed.object == &payload);
  CHECK(allocator.allocations == 1 && allocator.frees == 1);
  allocator.fail = 1;
  CHECK(consume_code(hyle_util_ref_create(
      &ref, &payload, destroy_ref_object, &destroyed,
      probe_allocator(&allocator)), HYLE_STATUS_RESOURCE_EXHAUSTED));
  CHECK(ref == NULL);
  CHECK(hyle_util_ref_retain(NULL) == NULL);
  CHECK(hyle_util_ref_get(NULL) == NULL);
  CHECK(hyle_util_ref_use_count(NULL) == 0);
  CHECK(!hyle_util_ref_is_unique(NULL));
  hyle_util_ref_release(NULL);
}

static void test_runtime_registry(void) {
  hyle_util_registry* registry = NULL;
  struct registry_factory_state fallback = {100};
  struct registry_factory_state preferred = {1000};
  struct registry_visit_state visited = {NULL, 0, 0, 0};
  uintptr_t increment = 7;
  void* object = NULL;
  char help[8];
  size_t help_size = 0;
  int completed = 0;

  CHECK(consume_ok(hyle_util_registry_create(
      &registry, hyle_system_allocator())));
  CHECK(registry != NULL);
  visited.registry = registry;
  CHECK(consume_ok(hyle_util_registry_reserve(registry, 16)));
  CHECK(consume_ok(hyle_util_registry_register(
      registry, hyle_container_cstr("engine"), registry_factory, &fallback,
      HYLE_UTIL_REGISTRY_FALLBACK, hyle_container_cstr("fallback"))));
  CHECK(hyle_util_registry_has_cstr(registry, "engine"));
  CHECK(!hyle_util_registry_has_cstr(registry, "missing"));
  CHECK(hyle_util_registry_size(registry) == 1);
  CHECK(consume_ok(hyle_util_registry_create_object(
      &object, registry, hyle_container_cstr("engine"), &increment)));
  CHECK((uintptr_t)object == 107);
  CHECK((uintptr_t)hyle_util_registry_create_or_null(
            registry, hyle_container_cstr("engine"), &increment) == 107);
  CHECK(hyle_util_registry_create_or_null_cstr(
            registry, "missing", NULL) == NULL);

  CHECK(consume_ok(hyle_util_registry_register_cstr(
      registry, "engine", registry_factory, &preferred,
      HYLE_UTIL_REGISTRY_PREFERRED, "fast implementation")));
  CHECK(consume_ok(hyle_util_registry_create_object_cstr(
      &object, registry, "engine", &increment)));
  CHECK((uintptr_t)object == 1007);
  CHECK(consume_ok(hyle_util_registry_help(
      &help_size, registry, hyle_container_cstr("engine"), help,
      sizeof(help))));
  CHECK(help_size == strlen("fast implementation"));
  CHECK(strcmp(help, "fast im") == 0);
  CHECK(consume_ok(hyle_util_registry_for_each(
      &completed, registry, visit_registry, &visited)));
  CHECK(completed && visited.count == 1 && visited.saw_fast);

  CHECK(consume_ok(hyle_util_registry_register_cstr(
      registry, "second", registry_factory, &fallback,
      HYLE_UTIL_REGISTRY_DEFAULT, "second implementation")));
  visited.count = 0;
  visited.stop_after_one = 1;
  CHECK(consume_ok(hyle_util_registry_for_each(
      &completed, registry, visit_registry, &visited)));
  CHECK(!completed && visited.count == 1);
  visited.stop_after_one = 0;

  CHECK(consume_ok(hyle_util_registry_register_cstr(
      registry, "engine", registry_factory, &fallback,
      HYLE_UTIL_REGISTRY_FALLBACK, "ignored")));
  CHECK(consume_code(hyle_util_registry_register_cstr(
      registry, "engine", registry_factory, &preferred,
      HYLE_UTIL_REGISTRY_PREFERRED, "duplicate"),
      HYLE_STATUS_ALREADY_EXISTS));
  CHECK(consume_code(hyle_util_registry_create_object_cstr(
      &object, registry, "missing", NULL), HYLE_STATUS_NOT_FOUND));

  {
    char owned_key[] = "owned";
    char owned_help[] = "owned help";
    CHECK(consume_ok(hyle_util_registry_register_cstr(
        registry, owned_key, registry_factory, &fallback,
        HYLE_UTIL_REGISTRY_DEFAULT, owned_help)));
    owned_key[0] = 'X';
    owned_help[0] = 'X';
    CHECK(hyle_util_registry_has_cstr(registry, "owned"));
    CHECK(consume_ok(hyle_util_registry_help_cstr(
        &help_size, registry, "owned", help, sizeof(help))));
    CHECK(help_size == strlen("owned help") && strcmp(help, "owned h") == 0);
  }

  {
    const char binary_key[] = {'b', 'i', 'n', '\0', 'x'};
    const hyle_container_string_view key =
        hyle_container_view(binary_key, (uint32_t)sizeof(binary_key));
    CHECK(consume_ok(hyle_util_registry_register(
        registry, key, registry_factory, &fallback,
        HYLE_UTIL_REGISTRY_DEFAULT, hyle_container_view(NULL, 0))));
    CHECK(hyle_util_registry_has(registry, key));
    CHECK((uintptr_t)hyle_util_registry_create_or_null(
              registry, key, &increment) == 107);
  }

  CHECK(consume_code(hyle_util_registry_register(
      registry, hyle_container_view(NULL, 1), registry_factory, &fallback,
      HYLE_UTIL_REGISTRY_DEFAULT, hyle_container_view(NULL, 0)),
      HYLE_STATUS_INVALID_ARGUMENT));
  CHECK(consume_code(hyle_util_registry_register_cstr(
      registry, "", registry_factory, &fallback,
      HYLE_UTIL_REGISTRY_DEFAULT, ""), HYLE_STATUS_INVALID_ARGUMENT));

  CHECK(!hyle_util_registry_is_frozen(registry));
  CHECK(consume_ok(hyle_util_registry_freeze(registry)));
  CHECK(consume_ok(hyle_util_registry_freeze(registry)));
  CHECK(hyle_util_registry_is_frozen(registry));
  CHECK(consume_code(hyle_util_registry_register_cstr(
      registry, "late", registry_factory, &fallback,
      HYLE_UTIL_REGISTRY_DEFAULT, NULL), HYLE_STATUS_FAILED_PRECONDITION));
  CHECK(consume_code(hyle_util_registry_reserve(registry, 32),
                     HYLE_STATUS_FAILED_PRECONDITION));
  visited.count = 0;
  CHECK(consume_ok(hyle_util_registry_for_each(
      &completed, registry, visit_registry, &visited)));
  CHECK(completed && visited.count == hyle_util_registry_size(registry));

  hyle_util_registry_destroy(registry);
  hyle_util_registry_destroy(NULL);

  {
    struct allocator_probe allocator = {
        hyle_system_allocator(), 0, 0, 0};
    size_t before_freeze;
    CHECK(consume_ok(hyle_util_registry_create(
        &registry, probe_allocator(&allocator))));
    CHECK(consume_ok(hyle_util_registry_register_cstr(
        registry, "entry", registry_factory, &fallback,
        HYLE_UTIL_REGISTRY_DEFAULT, "entry help")));
    before_freeze = allocator.allocations;
    allocator.fail = 1;
    CHECK(consume_ok(hyle_util_registry_freeze(registry)));
    CHECK(allocator.allocations == before_freeze);
    CHECK(hyle_util_registry_has_cstr(registry, "entry"));
    hyle_util_registry_destroy(registry);
    registry = NULL;
  }

  {
    struct allocator_probe allocator = {
        hyle_system_allocator(), 0, 0, 1};
    CHECK(consume_code(hyle_util_registry_create(
        &registry, probe_allocator(&allocator)),
        HYLE_STATUS_RESOURCE_EXHAUSTED));
    CHECK(registry == NULL);
  }
}

static void test_spin_lock(void) {
  hyle_util_spin_lock* lock = NULL;
  int acquired = 0;
  size_t storage_size = hyle_util_spin_lock_storage_size();
  size_t storage_alignment = hyle_util_spin_lock_storage_alignment();
  void* allocation = NULL;
  void* storage = NULL;

  CHECK(consume_ok(hyle_util_spin_lock_create(
      &lock, hyle_system_allocator())));
  CHECK(lock != NULL && !hyle_util_spin_lock_is_locked(lock));
  CHECK(consume_ok(hyle_util_spin_lock_try_lock(&acquired, lock)));
  CHECK(acquired && hyle_util_spin_lock_is_locked(lock));
  CHECK(consume_code(hyle_util_spin_lock_destroy(lock),
                     HYLE_STATUS_FAILED_PRECONDITION));
  CHECK(consume_ok(hyle_util_spin_lock_unlock(lock)));
  CHECK(consume_ok(hyle_util_spin_lock_lock(lock)));
  CHECK(consume_ok(hyle_util_spin_lock_try_lock(&acquired, lock)));
  CHECK(!acquired);
  CHECK(consume_ok(hyle_util_spin_lock_unlock(lock)));
  CHECK(consume_ok(hyle_util_spin_lock_destroy(lock)));

  allocation = malloc(storage_size + storage_alignment - 1);
  CHECK(allocation != NULL);
  if (allocation != NULL) {
    uintptr_t address = (uintptr_t)allocation;
    address = (address + storage_alignment - 1) & ~(storage_alignment - 1);
    storage = (void*)address;
    lock = NULL;
    CHECK(consume_ok(hyle_util_spin_lock_init(
        &lock, storage, storage_size)));
    CHECK(consume_ok(hyle_util_spin_lock_lock(lock)));
    CHECK(consume_ok(hyle_util_spin_lock_unlock(lock)));
    CHECK(consume_code(hyle_util_spin_lock_destroy(lock),
                       HYLE_STATUS_FAILED_PRECONDITION));
    CHECK(consume_ok(hyle_util_spin_lock_deinit(lock)));
    free(allocation);
  }
}

int main(void) {
  hyle_util_cpu_relax();
  test_demangle();
  test_rng_scalar_and_state();
  test_rng_ranges_batches_and_partitioning();
  test_rng_shuffle_and_thread_local();
  test_shared_reference();
  test_runtime_registry();
  test_spin_lock();
  if (g_fails == 0) puts("hyle util C ABI: ok");
  return g_fails == 0 ? 0 : 1;
}
