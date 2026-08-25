/* Pure C consumer of include/hyle/c/platform.h */
#if defined(_WIN32)
#define _CRT_SECURE_NO_WARNINGS 1
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#include "hyle/c/platform.h"
#include "hyle/c/core.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(sizeof(hyle_thread_priority) == sizeof(int32_t),
               "thread priority ABI width");
_Static_assert(sizeof(hyle_thread_config_policy) == sizeof(uint32_t),
               "thread config policy ABI width");
_Static_assert(sizeof(hyle_protect) == sizeof(uint32_t), "protect ABI width");
_Static_assert(sizeof(hyle_numa_node) == sizeof(int32_t),
               "NUMA node ABI width");
_Static_assert(sizeof(hyle_numa_policy) == sizeof(uint32_t),
               "NUMA policy ABI width");
_Static_assert(sizeof(hyle_numa_placement) == sizeof(uint32_t),
               "NUMA placement ABI width");
_Static_assert(sizeof(hyle_numa_capability) == sizeof(uint32_t),
               "NUMA capability ABI width");
_Static_assert(sizeof(hyle_file_type) == sizeof(uint32_t),
               "file type ABI width");
_Static_assert(sizeof(hyle_file_perms) == sizeof(uint32_t),
               "file permissions ABI width");
_Static_assert(sizeof(hyle_symlink_kind) == sizeof(uint32_t),
               "symlink kind ABI width");
_Static_assert(sizeof(hyle_file_open_flags) == sizeof(uint32_t),
               "file flags ABI width");
_Static_assert(sizeof(hyle_map_mode) == sizeof(uint32_t), "map mode ABI width");
_Static_assert(sizeof(hyle_map_advice) == sizeof(uint32_t),
               "map advice ABI width");
_Static_assert(sizeof(hyle_isa) == sizeof(uint32_t), "ISA ABI width");
_Static_assert(sizeof(hyle_cpu_kind) == sizeof(uint32_t), "CPU kind ABI width");
_Static_assert(sizeof(hyle_cpu_order) == sizeof(uint32_t),
               "CPU order ABI width");
_Static_assert(offsetof(hyle_thread_opts, reserved) == 4,
               "thread opts reserved ABI offset");
_Static_assert(offsetof(hyle_thread_opts, name) == 8,
               "thread opts first optional field offset");
_Static_assert(offsetof(hyle_thread_opts, has_priority) == 40,
               "thread opts has-priority ABI offset");
_Static_assert(offsetof(hyle_thread_opts, priority) == 44,
               "thread opts priority ABI offset");
_Static_assert(offsetof(hyle_thread_opts, config_policy) == 48,
               "thread opts policy ABI offset");
_Static_assert(offsetof(hyle_thread_opts, affinity_cpus) == 56,
               "thread opts affinity-array ABI offset");
_Static_assert(offsetof(hyle_thread_opts, affinity_cpu_count) == 64,
               "thread opts affinity-count ABI offset");
_Static_assert(HYLE_THREAD_OPTS_REQUIRED_SIZE == 8,
               "thread opts mandatory prefix");
_Static_assert(sizeof(hyle_thread_opts) == 72, "thread opts 64-bit ABI size");
_Static_assert(offsetof(hyle_cpu_features, reserved) == 4,
               "CPU features reserved ABI offset");
_Static_assert(offsetof(hyle_cpu_features, sse2) == 8,
               "CPU features first optional field offset");
_Static_assert(HYLE_CPU_FEATURES_REQUIRED_SIZE == 8,
               "CPU features mandatory prefix");
_Static_assert(sizeof(hyle_cpu_features) == 160,
               "CPU features 64-bit ABI size");

typedef struct cpu_features_short_prefix {
  uint32_t struct_size;
  uint32_t reserved;
  int sse2;
  int ssse3;
} cpu_features_short_prefix;
_Static_assert(offsetof(hyle_cpu_info, kind) == 8, "CPU info kind ABI offset");
_Static_assert(sizeof(hyle_cpu_info) == 12, "CPU info ABI size");

/* Physical layouts produced by a shorter versioned binding. These are
 * deliberately smaller objects, rather than current structs with a reduced
 * struct_size, so sanitizer runs can detect a bridge that reads beyond the
 * declared prefix. */
typedef struct hyle_thread_opts_minimal_prefix {
  uint32_t struct_size;
  uint32_t reserved;
} hyle_thread_opts_minimal_prefix;

typedef struct hyle_thread_opts_historical_prefix {
  uint32_t struct_size;
  uint32_t reserved;
  const char *name;
  int has_stack_size;
  size_t stack_size;
  int has_affinity_cpu;
  unsigned affinity_cpu;
  int has_priority;
  hyle_thread_priority priority;
  hyle_thread_config_policy config_policy;
} hyle_thread_opts_historical_prefix;

_Static_assert(sizeof(hyle_thread_opts_minimal_prefix) == 8,
               "minimal thread opts prefix size");
_Static_assert(sizeof(hyle_thread_opts_historical_prefix) == 56,
               "historical thread opts prefix size");

static int g_fails = 0;
#define CHECK(c)                                                               \
  do {                                                                         \
    if (!(c)) {                                                                \
      fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #c, __FILE__, __LINE__);   \
      ++g_fails;                                                               \
    }                                                                          \
  } while (0)

static int status_ok(hyle_status status) {
  int ok = hyle_status_is_ok(status);
  hyle_status_free(&status);
  return ok;
}

static hyle_status_code status_code(hyle_status status) {
  hyle_status_code code = hyle_status_get_code(status);
  hyle_status_free(&status);
  return code;
}

static void status_discard(hyle_status status) { hyle_status_free(&status); }

static void thr_entry(void *ctx) {
  int *p = (int *)ctx;
  *p = 42;
}

static int visit_count = 0;
static int visit_cb(const char *name, hyle_file_type type, void *ctx) {
  (void)name;
  (void)type;
  (void)ctx;
  ++visit_count;
  return 1;
}

/* Low-level wait/wake worker: block until the shared word turns non-zero, then
 * record that we woke. The value-recheck loop makes the wake race-free (an
 * early store+wake before we park is simply observed by the loop condition) and
 * absorbs spurious wakes. The word is the futex/WaitOnAddress target; C11
 * atomic load/store carry the cross-thread ordering, while wait/wake only block
 * and notify on its address. */
typedef struct {
  volatile uint32_t *word;
  int *woke;
} wait_ctx;
static uint32_t atomic_load_word(volatile uint32_t *word) {
#if defined(_WIN32)
  return (uint32_t)InterlockedCompareExchange((volatile LONG *)word, 0, 0);
#else
  return __atomic_load_n(word, __ATOMIC_ACQUIRE);
#endif
}
static void atomic_store_word(volatile uint32_t *word, uint32_t value) {
#if defined(_WIN32)
  (void)InterlockedExchange((volatile LONG *)word, (LONG)value);
#else
  __atomic_store_n(word, value, __ATOMIC_RELEASE);
#endif
}
static void wait_entry(void *ctx) {
  wait_ctx *x = (wait_ctx *)ctx;
  while (atomic_load_word(x->word) == 0u) {
    hyle_wait_on_address(x->word, 0u);
  }
  *x->woke = 1;
}

int main(void) {
  (void)"HYLE_C_PLATFORM_COVERAGE_BODY_MARKER";
  CHECK(hyle_page_size() >= 4096u);
  CHECK(hyle_page_align_up(SIZE_MAX) == 0u);
  CHECK(hyle_large_page_align_up(SIZE_MAX) == 0u);

  /* Parker */
  hyle_parker_o *p = NULL;
  CHECK(status_code(hyle_parker_create(NULL)) == HYLE_STATUS_INVALID_ARGUMENT);
  CHECK(status_ok(hyle_parker_create(&p)));
  CHECK(p != NULL);
  hyle_parker_unpark(p);
  hyle_parker_park(p);
  CHECK(hyle_parker_park_timeout(p, 0) == 0);
  hyle_parker_destroy(p);

  /* Low-level wait/wake (futex / WaitOnAddress thin forwards). First the
   * non-blocking contracts (no thread needed): a value MISMATCH never blocks
   * and reports "not a timeout" (1); a zero timeout on a matching value is a
   * poll; waking an address with no waiter is a no-op. */
  {
    uint32_t w = 5u;
    CHECK(hyle_wait_on_address_for(&w, 7u, 0) ==
          1); /* 5 != 7 -> immediate, 1 */
    CHECK(hyle_wait_on_address_for(&w, 5u, 0) == 0); /* equal: timed poll */
    hyle_wait_on_address(&w, 7u); /* 5 != 7 -> returns at once */
    hyle_wake_one(&w);            /* no waiter -> no-op */
    hyle_wake_all(&w);            /* no waiter -> no-op */
  }

  /* Real cross-thread wake_one: one waiter blocks on the word, main flips it
   * and wakes exactly one. Proves the addr forward + wake path end to end. */
  {
    volatile uint32_t word = 0u;
    int woke = 0;
    wait_ctx c = {&word, &woke};
    hyle_thread_o *wt = NULL;
    hyle_status ws = hyle_thread_spawn_opts(&wt, wait_entry, &c, "c_wait1", 0,
                                            -1, HYLE_THREAD_PRIORITY_NORMAL);
    CHECK(hyle_status_is_ok(ws));
    hyle_status_free(&ws);
    CHECK(wt != NULL);
    if (wt != NULL) {
      atomic_store_word(&word, 1u);
      hyle_wake_one(&word);
      hyle_thread_join(
          wt); /* hangs only if the wake never lands (broken fwd) */
      CHECK(woke == 1);
    }
  }

  /* Real cross-thread wake_all: two waiters on one word, released together. */
  {
    volatile uint32_t word = 0u;
    int woke_a = 0, woke_b = 0;
    wait_ctx ca = {&word, &woke_a};
    wait_ctx cb = {&word, &woke_b};
    hyle_thread_o *ta = NULL, *tb = NULL;
    hyle_status sa = hyle_thread_spawn_opts(&ta, wait_entry, &ca, "c_waitA", 0,
                                            -1, HYLE_THREAD_PRIORITY_NORMAL);
    hyle_status sb = hyle_thread_spawn_opts(&tb, wait_entry, &cb, "c_waitB", 0,
                                            -1, HYLE_THREAD_PRIORITY_NORMAL);
    CHECK(hyle_status_is_ok(sa));
    CHECK(hyle_status_is_ok(sb));
    hyle_status_free(&sa);
    hyle_status_free(&sb);
    if (ta != NULL && tb != NULL) {
      atomic_store_word(&word, 1u);
      hyle_wake_all(&word);
      hyle_thread_join(ta);
      hyle_thread_join(tb);
      CHECK(woke_a == 1);
      CHECK(woke_b == 1);
    }
  }

  /* Thread + opts sugar */
  int ran = 0;
  hyle_thread_o *t = NULL;
  hyle_thread_opts opts = hyle_thread_opts_default();
  CHECK(opts.struct_size == sizeof(opts));
  hyle_thread_opts_set_name(&opts, "c_plat_worker");
  hyle_thread_opts_set_stack_size(&opts, 2u * 1024u * 1024u);
  CHECK(opts.has_stack_size && opts.stack_size == 2u * 1024u * 1024u);
  opts.has_stack_size =
      0; /* spawn with the OS default after testing the helper */
  hyle_thread_opts_set_affinity(&opts, 0);
  CHECK(opts.has_affinity_cpu && opts.affinity_cpu == 0u);
  {
    const unsigned cpus[] = {2u, 6u};
    hyle_thread_opts_set_affinity_cpus(&opts, cpus, 2u);
    CHECK(!opts.has_affinity_cpu && opts.affinity_cpus == cpus &&
          opts.affinity_cpu_count == 2u);
    hyle_thread_opts_set_affinity(&opts, 0);
    CHECK(opts.has_affinity_cpu && opts.affinity_cpus == NULL &&
          opts.affinity_cpu_count == 0u);
  }
  opts.has_affinity_cpu = 0; /* do not constrain heterogeneous CI hosts */
  hyle_thread_opts_set_priority(&opts, HYLE_THREAD_PRIORITY_NORMAL);
  CHECK(opts.has_priority && opts.priority == HYLE_THREAD_PRIORITY_NORMAL);
  hyle_thread_opts_set_config_policy(&opts, HYLE_THREAD_CONFIG_REQUIRED);
  CHECK(opts.config_policy == HYLE_THREAD_CONFIG_REQUIRED);
  hyle_status st = hyle_thread_spawn(&t, thr_entry, &ran, &opts);
  CHECK(hyle_status_is_ok(st));
  hyle_status_free(&st);
  CHECK(t != NULL);
  CHECK(hyle_thread_joinable(t));
  hyle_thread_join(t);
  CHECK(ran == 42);

  /* Owning outputs are cleared before argument validation, so a caller can
   * use one cleanup path without retaining a stale handle. */
  {
    hyle_thread_o *stale = (hyle_thread_o *)(uintptr_t)1;
    hyle_status invalid_entry = hyle_thread_spawn(&stale, NULL, NULL, NULL);
    CHECK(hyle_status_is_invalid_argument(invalid_entry));
    CHECK(stale == NULL);
    hyle_status_free(&invalid_entry);
  }

  /* A real minimum-size allocation is accepted and uses all defaults. */
  {
    hyle_thread_opts_minimal_prefix minimal = {
        (uint32_t)sizeof(hyle_thread_opts_minimal_prefix), 0};
    int minimal_ran = 0;
    hyle_thread_o *minimal_thread = NULL;
    hyle_status minimal_status = hyle_thread_spawn(
        &minimal_thread, thr_entry, &minimal_ran,
        (const hyle_thread_opts *)(const void *)&minimal);
    CHECK(hyle_status_is_ok(minimal_status));
    hyle_status_free(&minimal_status);
    CHECK(minimal_thread != NULL);
    if (minimal_thread != NULL)
      hyle_thread_join(minimal_thread);
    CHECK(minimal_ran == 42);
  }

  /* Every business field is optional. With only struct_size+reserved readable,
   * even deliberately invalid bytes in later fields must remain invisible and
   * all current defaults must be used. */
  {
    int minimal_ran = 0;
    hyle_thread_o *minimal_thread = NULL;
    hyle_thread_opts minimal = HYLE_THREAD_OPTS_INIT;
    minimal.struct_size = HYLE_THREAD_OPTS_REQUIRED_SIZE;
    minimal.has_stack_size = 1;
    minimal.stack_size = 1;
    minimal.has_affinity_cpu = 1;
    minimal.affinity_cpu = UINT_MAX;
    minimal.has_priority = 1;
    minimal.priority = 99;
    minimal.config_policy = 99;
    minimal.affinity_cpus = NULL;
    minimal.affinity_cpu_count = 1;
    hyle_status minimal_status = hyle_thread_spawn(
        &minimal_thread, thr_entry, &minimal_ran, &minimal);
    CHECK(hyle_status_is_ok(minimal_status));
    hyle_status_free(&minimal_status);
    CHECK(minimal_thread != NULL);
    if (minimal_thread != NULL)
      hyle_thread_join(minimal_thread);
    CHECK(minimal_ran == 42);
  }

  /* Reserved is part of the mandatory prefix and must remain zero. */
  {
    hyle_thread_opts_minimal_prefix invalid_reserved = {
        (uint32_t)sizeof(hyle_thread_opts_minimal_prefix), UINT32_MAX};
    hyle_thread_o *bad = NULL;
    hyle_status invalid_reserved_status = hyle_thread_spawn(
        &bad, thr_entry, &ran,
        (const hyle_thread_opts *)(const void *)&invalid_reserved);
    CHECK(hyle_status_is_invalid_argument(invalid_reserved_status));
    hyle_status_free(&invalid_reserved_status);
    CHECK(bad == NULL);
  }

  /* A real allocation using the complete historical-field prefix and the
   * explicit zero reserved word remains readable; the appended affinity-array
   * fields do not exist in this object. */
  {
    hyle_thread_opts_historical_prefix historical;
    int historical_ran = 0;
    hyle_thread_o *historical_thread = NULL;
    memset(&historical, 0, sizeof(historical));
    historical.struct_size = (uint32_t)sizeof(historical);
    historical.priority = HYLE_THREAD_PRIORITY_NORMAL;
    historical.config_policy = HYLE_THREAD_CONFIG_BEST_EFFORT;
    hyle_status historical_status = hyle_thread_spawn(
        &historical_thread, thr_entry, &historical_ran,
        (const hyle_thread_opts *)(const void *)&historical);
    CHECK(hyle_status_is_ok(historical_status));
    hyle_status_free(&historical_status);
    CHECK(historical_thread != NULL);
    if (historical_thread != NULL)
      hyle_thread_join(historical_thread);
    CHECK(historical_ran == 42);
  }

  /* Versioned C options keep the historical-field prefix through
   * config_policy. Poison the appended fields: a shorter caller did not
   * allocate or declare them, so the library must retain current defaults and
   * never validate/read them. */
  {
    int prefix_ran = 0;
    hyle_thread_o *prefix_thread = NULL;
    hyle_thread_opts prefix = HYLE_THREAD_OPTS_INIT;
    prefix.struct_size =
        (uint32_t)offsetof(hyle_thread_opts, affinity_cpus);
    prefix.affinity_cpus = NULL;
    prefix.affinity_cpu_count = 1;
    hyle_status prefix_status = hyle_thread_spawn(
        &prefix_thread, thr_entry, &prefix_ran, &prefix);
    CHECK(hyle_status_is_ok(prefix_status));
    hyle_status_free(&prefix_status);
    CHECK(prefix_thread != NULL);
    if (prefix_thread != NULL)
      hyle_thread_join(prefix_thread);
    CHECK(prefix_ran == 42);
  }

  /* A flag without its complete value is not an enabled option. Priority and
   * all later fields therefore retain defaults for this intermediate prefix. */
  {
    int partial_priority_ran = 0;
    hyle_thread_o *partial_priority_thread = NULL;
    hyle_thread_opts partial_priority = HYLE_THREAD_OPTS_INIT;
    partial_priority.struct_size =
        (uint32_t)offsetof(hyle_thread_opts, priority);
    partial_priority.has_priority = 1;
    partial_priority.priority = 99;
    partial_priority.config_policy = 99;
    hyle_status partial_priority_status = hyle_thread_spawn(
        &partial_priority_thread, thr_entry, &partial_priority_ran,
        &partial_priority);
    CHECK(hyle_status_is_ok(partial_priority_status));
    hyle_status_free(&partial_priority_status);
    CHECK(partial_priority_thread != NULL);
    if (partial_priority_thread != NULL)
      hyle_thread_join(partial_priority_thread);
    CHECK(partial_priority_ran == 42);
  }

  /* The affinity pointer/count extension is one unit. A prefix that contains
   * the complete pointer but not the count ignores both and keeps defaults. */
  {
    int partial_ran = 0;
    hyle_thread_o *partial_thread = NULL;
    hyle_thread_opts partial = HYLE_THREAD_OPTS_INIT;
    partial.struct_size =
        (uint32_t)offsetof(hyle_thread_opts, affinity_cpu_count);
    partial.affinity_cpus = NULL;
    partial.affinity_cpu_count = 1;
    hyle_status partial_status = hyle_thread_spawn(
        &partial_thread, thr_entry, &partial_ran, &partial);
    CHECK(hyle_status_is_ok(partial_status));
    hyle_status_free(&partial_status);
    CHECK(partial_thread != NULL);
    if (partial_thread != NULL)
      hyle_thread_join(partial_thread);
    CHECK(partial_ran == 42);
  }

  /* A future caller may append fields. The current library consumes only its
   * known 72-byte prefix and ignores the declared unknown tail. */
  {
    struct future_thread_opts {
      hyle_thread_opts known;
      unsigned char tail[16];
    } future;
    int future_ran = 0;
    hyle_thread_o *future_thread = NULL;
    memset(&future, 0xA5, sizeof(future));
    future.known = hyle_thread_opts_default();
    future.known.struct_size = (uint32_t)sizeof(future);
    hyle_status future_status = hyle_thread_spawn(
        &future_thread, thr_entry, &future_ran, &future.known);
    CHECK(hyle_status_is_ok(future_status));
    hyle_status_free(&future_status);
    CHECK(future_thread != NULL);
    if (future_thread != NULL)
      hyle_thread_join(future_thread);
    CHECK(future_ran == 42);
  }

  /* The list-affinity C path accepts real (possibly sparse) available ids and
   * copies the view during spawn. Keep best-effort for macOS, which has no hard
   * affinity API but must still run the entry. */
  {
    unsigned cpus[2] = {0, 0};
    size_t available = 0;
    hyle_status ids_status = hyle_cpu_available_ids(&available, cpus, 2);
    CHECK(hyle_status_is_ok(ids_status));
    hyle_status_free(&ids_status);
    if (available != 0) {
      int list_ran = 0;
      hyle_thread_o *list_thread = NULL;
      hyle_thread_opts list_opts = HYLE_THREAD_OPTS_INIT;
      size_t selected = available < 2 ? available : 2;
      hyle_thread_opts_set_affinity_cpus(&list_opts, cpus, selected);
      hyle_status list_status =
          hyle_thread_spawn(&list_thread, thr_entry, &list_ran, &list_opts);
      CHECK(hyle_status_is_ok(list_status));
      hyle_status_free(&list_status);
      CHECK(list_thread != NULL);
      hyle_thread_join(list_thread);
      CHECK(list_ran == 42);
    }
  }

  {
    hyle_thread_opts too_small = HYLE_THREAD_OPTS_INIT;
    hyle_thread_o *bad = NULL;
    too_small.struct_size = HYLE_THREAD_OPTS_REQUIRED_SIZE - 1;
    hyle_status small = hyle_thread_spawn(&bad, thr_entry, &ran, &too_small);
    CHECK(hyle_status_is_invalid_argument(small));
    CHECK(bad == NULL);
    hyle_status_free(&small);

    too_small.struct_size = 0;
    small = hyle_thread_spawn(&bad, thr_entry, &ran, &too_small);
    CHECK(hyle_status_is_invalid_argument(small));
    CHECK(bad == NULL);
    hyle_status_free(&small);
  }

  /* C list affinity rejects malformed views at the boundary. */
  {
    hyle_thread_o *bad = NULL;
    hyle_thread_opts malformed = HYLE_THREAD_OPTS_INIT;
    malformed.affinity_cpu_count = 2;
    hyle_status malformed_status =
        hyle_thread_spawn(&bad, thr_entry, &ran, &malformed);
    CHECK(hyle_status_is_invalid_argument(malformed_status));
    CHECK(bad == NULL);
    hyle_status_free(&malformed_status);

    malformed = hyle_thread_opts_default();
    hyle_thread_opts_set_affinity(&malformed, 0);
    malformed.affinity_cpus = &malformed.affinity_cpu;
    malformed.affinity_cpu_count = 1;
    malformed_status = hyle_thread_spawn(&bad, thr_entry, &ran, &malformed);
    CHECK(hyle_status_is_invalid_argument(malformed_status));
    CHECK(bad == NULL);
    hyle_status_free(&malformed_status);
  }

  /* Required setup is observable: an impossible CPU rejects spawn and never
   * invokes the user's entry. */
  {
    int must_not_run = 0;
    hyle_thread_o *bad = NULL;
    hyle_thread_opts strict = HYLE_THREAD_OPTS_INIT;
    hyle_thread_opts_set_affinity(&strict, UINT_MAX);
    hyle_thread_opts_set_config_policy(&strict, HYLE_THREAD_CONFIG_REQUIRED);
    hyle_status strict_status =
        hyle_thread_spawn(&bad, thr_entry, &must_not_run, &strict);
    CHECK(!hyle_status_is_ok(strict_status));
    CHECK(bad == NULL);
    CHECK(must_not_run == 0);
    hyle_status_free(&strict_status);
  }

  /* FFI-friendly spawn_opts: no struct/has_ flags, sentinels for defaults
   * (stack_size 0 => default, affinity_cpu -1 => none). */
  {
    int ran2 = 0;
    hyle_thread_o *t2 = NULL;
    hyle_status st2 =
        hyle_thread_spawn_opts(&t2, thr_entry, &ran2, "c_plat_opts", 0, -1,
                               HYLE_THREAD_PRIORITY_NORMAL);
    CHECK(hyle_status_is_ok(st2));
    hyle_status_free(&st2);
    CHECK(t2 != NULL);
    hyle_thread_join(t2);
    CHECK(ran2 == 42);
  }

  /* Invalid enum values are rejected at the C boundary, never silently
   * reinterpreted as a valid scheduling policy. */
  {
    hyle_thread_o *bad = NULL;
    hyle_status bad_st = hyle_thread_spawn_opts(&bad, thr_entry, &ran,
                                                "bad_priority", 0, -1, 99);
    CHECK(hyle_status_is_invalid_argument(bad_st));
    CHECK(bad == NULL);
    hyle_status_free(&bad_st);
  }

  CHECK(hyle_hardware_concurrency() >= 1u);
  CHECK(hyle_this_thread_id() != 0);
  hyle_this_thread_yield();
  CHECK(status_ok(hyle_this_thread_set_name("hyle_c_main")));
  /* Single-CPU entry has the same platform-dependent support contract as the
   * collection form; use a guaranteed-invalid id without changing this thread.
   */
  status_discard(hyle_this_thread_set_affinity(UINT_MAX));
  CHECK(status_code(hyle_this_thread_set_affinity_cpus(NULL, 0)) ==
        HYLE_STATUS_INVALID_ARGUMENT);
  CHECK(status_code(hyle_this_thread_set_priority((hyle_thread_priority)99)) ==
        HYLE_STATUS_INVALID_ARGUMENT);

  /* DynLib */
#if defined(_WIN32)
  const char *sys_lib = "kernel32.dll";
  const char *sys_sym = "GetCurrentProcessId";
  const char *ext = "dll";
#elif defined(__APPLE__)
  /* install-name, not the search alias: hyle_dynlib_get uses RTLD_NOLOAD which
   * keys on the loaded image's install-name (libSystem.B), not
   * "libSystem.dylib". */
  const char *sys_lib = "libSystem.B.dylib";
  const char *sys_sym = "getpid";
  const char *ext = "dylib";
#else
  const char *sys_lib = "libm.so.6";
  const char *sys_sym = "cos";
  const char *ext = "so";
#endif
  hyle_dynlib_o *lib = NULL;
  hyle_status dynlib_status = hyle_dynlib_get(&lib, sys_lib);
  CHECK(hyle_status_is_ok(dynlib_status));
  hyle_status_free(&dynlib_status);
  CHECK(hyle_dynlib_valid(lib));
  {
    void *symbol = NULL;
    dynlib_status = hyle_dynlib_symbol(&symbol, lib, sys_sym);
    CHECK(hyle_status_is_ok(dynlib_status));
    CHECK(symbol != NULL);
    hyle_status_free(&dynlib_status);
  }
  hyle_dynlib_close(lib);
  {
    hyle_dynlib_o *current = NULL;
    dynlib_status = hyle_dynlib_current(&current);
    CHECK(hyle_status_is_ok(dynlib_status));
    hyle_status_free(&dynlib_status);
    CHECK(hyle_dynlib_valid(current));
    hyle_dynlib_close(current);
  }
  CHECK(strcmp(hyle_dynlib_extension(), ext) == 0);

  dynlib_status = hyle_dynlib_open(NULL, sys_lib);
  CHECK(hyle_status_is_invalid_argument(dynlib_status));
  hyle_status_free(&dynlib_status);
  {
    void *cleared = (void *)(uintptr_t)1;
    dynlib_status = hyle_dynlib_symbol(&cleared, NULL, sys_sym);
    CHECK(hyle_status_is_invalid_argument(dynlib_status));
    CHECK(cleared == NULL);
    hyle_status_free(&dynlib_status);
  }

  /* open_symbol on already-loadable system lib path may fail if path needs
   * absolute form; exercise API shape via get+symbol path above. Missing path:
   */
  {
    hyle_dynlib_o *miss = NULL;
    void *miss_sym = NULL;
    dynlib_status = hyle_dynlib_open_symbol(
        &miss, &miss_sym, "hyle_no_such_plugin_xyz.so", "entry");
    CHECK(!hyle_status_is_ok(dynlib_status));
    CHECK(hyle_status_is_not_found(dynlib_status));
    hyle_status_free(&dynlib_status);
    CHECK(miss == NULL);
    CHECK(miss_sym == NULL);
  }

  /* Managed reserve factories clear stale outputs on failure. Their NULL-out
   * checks also happen before attempting any OS reservation. */
  {
    hyle_vm_region_o *stale = (hyle_vm_region_o *)(uintptr_t)1;
    CHECK(status_code(hyle_vm_region_reserve(&stale, SIZE_MAX)) ==
          HYLE_STATUS_INVALID_ARGUMENT);
    CHECK(stale == NULL);

    stale = (hyle_vm_region_o *)(uintptr_t)1;
    CHECK(status_code(hyle_vm_region_reserve_guarded(&stale, SIZE_MAX)) ==
          HYLE_STATUS_OUT_OF_RANGE);
    CHECK(stale == NULL);

    stale = (hyle_vm_region_o *)(uintptr_t)1;
    CHECK(status_code(hyle_vm_region_reserve_numa(
              &stale, hyle_page_size(), (hyle_numa_node)-2)) ==
          HYLE_STATUS_OUT_OF_RANGE);
    CHECK(stale == NULL);

    CHECK(status_code(hyle_vm_region_reserve(NULL, hyle_page_size())) ==
          HYLE_STATUS_INVALID_ARGUMENT);
    CHECK(status_code(
              hyle_vm_region_reserve_guarded(NULL, hyle_page_size())) ==
          HYLE_STATUS_INVALID_ARGUMENT);
    CHECK(status_code(hyle_vm_region_reserve_numa(
              NULL, hyle_page_size(), HYLE_NUMA_ANY)) ==
          HYLE_STATUS_INVALID_ARGUMENT);
  }

  /* VM guarded */
  hyle_vm_region_o *reg = NULL;
  CHECK(status_ok(hyle_vm_region_reserve_guarded(&reg, hyle_page_size())));
  CHECK(hyle_vm_region_valid(reg));
  CHECK(hyle_vm_region_has_guard(reg));
  CHECK(hyle_vm_region_usable_size(reg) == hyle_page_size());
  {
    unsigned char *u = (unsigned char *)hyle_vm_region_usable(reg);
    CHECK(u != NULL);
    u[0] = 0xAB;
    CHECK(u[0] == 0xAB);
  }
  hyle_vm_region_destroy(reg);

  /* v5: region reserve_numa (ANY == plain) + adopt an existing raw reservation
   */
  {
    size_t pg = hyle_page_size();
    hyle_vm_region_o *rn = NULL;
    CHECK(status_ok(hyle_vm_region_reserve_numa(&rn, pg * 2, HYLE_NUMA_ANY)));
    CHECK(hyle_vm_region_valid(rn));
    CHECK(hyle_vm_region_numa_placement(rn) == HYLE_NUMA_PLACEMENT_NOT_NEEDED);
    CHECK(status_ok(hyle_vm_region_commit(rn, 0, pg * 2)));
    ((unsigned char *)hyle_vm_region_data(rn))[0] = 0x5A;
    CHECK(((unsigned char *)hyle_vm_region_data(rn))[0] == 0x5A);
    hyle_vm_region_destroy(rn);

    /* adopt: raw reserve -> hand ownership to a region handle (destroy
     * releases) */
    void *raw = NULL;
    CHECK(status_ok(hyle_vm_reserve(&raw, pg)));
    CHECK(raw != NULL);
    hyle_vm_region_o *ad = NULL;
    CHECK(status_ok(hyle_vm_region_adopt(&ad, raw, pg)));
    CHECK(hyle_vm_region_valid(ad));
    CHECK(hyle_vm_region_data(ad) == raw);
    CHECK(hyle_vm_region_size(ad) == pg);
    hyle_vm_region_destroy(ad); /* releases the adopted reservation */
    /* adopt(NULL) fails without taking ownership. */
    CHECK(status_code(hyle_vm_region_adopt(&ad, NULL, pg)) ==
          HYLE_STATUS_INVALID_ARGUMENT);
  }

  /* free vm + alloc_rw */
  {
    size_t pg = hyle_page_size();
    void *base = NULL;
    CHECK(status_ok(hyle_vm_reserve(&base, pg)));
    CHECK(base != NULL);
    CHECK(status_ok(hyle_vm_commit(base, pg)));
    CHECK(status_ok(hyle_vm_release(base, pg)));

    void *rw = NULL;
    CHECK(status_ok(hyle_vm_alloc_rw(&rw, pg / 2 + 1)));
    CHECK(rw != NULL);
    ((unsigned char *)rw)[0] = 1;
    CHECK(status_ok(hyle_vm_free_rw(rw, pg / 2 + 1)));
  }

  /* free-function vm_protect / vm_decommit thin forwards. Two pages so we can
   * lock/decommit one while keeping the other usable. (The C++ death tests
   * prove a locked page actually faults; here we just cover the C forward +
   * recovery.) */
  {
    size_t pg = hyle_page_size();
    void *base = NULL;
    CHECK(status_ok(hyle_vm_reserve(&base, pg * 2)));
    CHECK(base != NULL);
    CHECK(status_ok(hyle_vm_commit(base, pg * 2)));
    ((unsigned char *)base)[0] = 0x10;

    /* protect the low page read-only: still readable, restore RW and rewrite.
     */
    CHECK(status_ok(hyle_vm_protect(base, pg, HYLE_PROTECT_READ)));
    CHECK(((unsigned char *)base)[0] == 0x10);
    CHECK(status_ok(hyle_vm_protect(base, pg, HYLE_PROTECT_READ_WRITE)));
    CHECK(status_code(hyle_vm_protect(base, pg, (hyle_protect)99)) ==
          HYLE_STATUS_INVALID_ARGUMENT);
    ((unsigned char *)base)[0] = 0x20;
    CHECK(((unsigned char *)base)[0] == 0x20);

    /* decommit the high page, then recommit and use it (contents undefined). */
    CHECK(status_ok(hyle_vm_decommit((unsigned char *)base + pg, pg)));
    CHECK(status_ok(hyle_vm_commit((unsigned char *)base + pg, pg)));
    ((unsigned char *)base)[pg] = 0x30;
    CHECK(((unsigned char *)base)[pg] == 0x30);

    CHECK(status_ok(hyle_vm_release(base, pg * 2)));
  }

  /* managed region: commit_usable / protect / decommit thin forwards. */
  {
    size_t pg = hyle_page_size();
    hyle_vm_region_o *r = NULL;
    CHECK(status_ok(hyle_vm_region_reserve(&r, pg * 2)));
    CHECK(hyle_vm_region_valid(r));
    /* commit_usable: whole reservation RW (no guard on a plain reserve). */
    CHECK(status_ok(hyle_vm_region_commit_usable(r)));
    unsigned char *d = (unsigned char *)hyle_vm_region_data(r);
    CHECK(d != NULL);
    if (d != NULL) {
      d[0] = 0x41;
      CHECK(d[0] == 0x41);
      /* protect page 0 read-only then restore. */
      CHECK(status_ok(hyle_vm_region_protect(r, 0, pg, HYLE_PROTECT_READ)));
      CHECK(d[0] == 0x41); /* read still allowed */
      CHECK(
          status_ok(hyle_vm_region_protect(r, 0, pg, HYLE_PROTECT_READ_WRITE)));
      /* decommit page 1 then recommit via region commit and use it. */
      CHECK(status_ok(hyle_vm_region_decommit(r, pg, pg)));
      CHECK(status_ok(hyle_vm_region_commit(r, pg, pg)));
      d[pg] = 0x42;
      CHECK(d[pg] == 0x42);
    }
    hyle_vm_region_destroy(r);
  }

  /* fs */
  char tmp[512];
  size_t n = 0;
  CHECK(status_code(hyle_fs_temp_directory(NULL, tmp, sizeof(tmp))) ==
        HYLE_STATUS_INVALID_ARGUMENT);
  CHECK(status_ok(hyle_fs_temp_directory(&n, tmp, sizeof(tmp))));
  CHECK(n > 0 && n < sizeof(tmp));
  int path_value = 0;
  CHECK(status_code(hyle_fs_exists(NULL, tmp)) == HYLE_STATUS_INVALID_ARGUMENT);
  CHECK(status_code(hyle_fs_stat(NULL, tmp)) == HYLE_STATUS_INVALID_ARGUMENT);
  CHECK(status_code(hyle_fs_directory_iterate(NULL, tmp, visit_cb, NULL)) ==
        HYLE_STATUS_INVALID_ARGUMENT);
  path_value = 7;
  CHECK(status_code(hyle_fs_exists(&path_value, NULL)) ==
        HYLE_STATUS_INVALID_ARGUMENT);
  CHECK(path_value == 0);
  path_value = 7;
  CHECK(status_code(hyle_fs_directory_iterate(&path_value, tmp, NULL, NULL)) ==
        HYLE_STATUS_INVALID_ARGUMENT);
  CHECK(path_value == 0);
  CHECK(status_ok(hyle_fs_exists(&path_value, tmp)) && path_value);
  CHECK(status_ok(hyle_fs_is_directory(&path_value, tmp)) && path_value);
  {
    char cwd[1024];
    char home[1024];
    size_t cwd_size = 0;
    size_t home_size = 0;
    CHECK(status_ok(hyle_fs_current_directory(&cwd_size, cwd, sizeof(cwd))) &&
          cwd_size > 0u);
    CHECK(status_ok(hyle_fs_home_directory(&home_size, home, sizeof(home))) &&
          home_size > 0u);
  }
  char path[640];
  snprintf(path, sizeof(path), "%s/hyle_c_plat_test_dir", tmp);
  CHECK(status_ok(hyle_fs_make_directory_recursive(path)));
  {
    char single[700];
    snprintf(single, sizeof(single), "%s/single", path);
    CHECK(status_ok(hyle_fs_make_directory(single)));
    CHECK(status_ok(hyle_fs_remove_directory(single)));
  }
  {
    hyle_file_info di = {0};
    CHECK(status_ok(hyle_fs_stat(&di, path)));
    CHECK(di.type == HYLE_FILE_TYPE_DIRECTORY);
    CHECK(hyle_unix_timestamp_is_valid(di.modified_time));
  }
  visit_count = 0;
  int completed = 0;
  CHECK(
      status_ok(hyle_fs_directory_iterate(&completed, path, visit_cb, NULL)) &&
      completed);
  char file[700];
  snprintf(file, sizeof(file), "%s/f.txt", path);
  CHECK(status_ok(hyle_fs_touch(file)));
  CHECK(status_ok(hyle_fs_exists(&path_value, file)) && path_value);
  {
    hyle_file_info fi = {0};
    CHECK(status_ok(hyle_fs_stat(&fi, file)));
    CHECK(hyle_unix_timestamp_is_valid(fi.modified_time));
  }
  {
    uint64_t size = 1;
    CHECK(status_ok(hyle_fs_file_size(&size, file)) && size == 0u);
  }
  {
    hyle_file_perms mode = 0;
    CHECK(status_ok(hyle_fs_get_permissions(&mode, file)));
    CHECK(mode != 0u);
    CHECK(status_ok(hyle_fs_set_permissions(file, mode)));
  }
  {
    char copy[720];
    char renamed[720];
    snprintf(copy, sizeof(copy), "%s/f-copy.txt", path);
    snprintf(renamed, sizeof(renamed), "%s/f-renamed.txt", path);
    CHECK(status_ok(hyle_fs_copy_file(file, copy)));
    CHECK(status_ok(hyle_fs_rename(copy, renamed)));
    CHECK(status_ok(hyle_fs_exists(&path_value, renamed)) && path_value);
  }
  {
    char cwd[1024];
    size_t cwd_size = 0;
    CHECK(status_ok(hyle_fs_current_directory(&cwd_size, cwd, sizeof(cwd))) &&
          cwd_size > 0u);
    CHECK(status_ok(hyle_fs_change_directory(path)));
    CHECK(status_ok(hyle_fs_change_directory(cwd)));
  }
  {
    char link[720];
    snprintf(link, sizeof(link), "%s/f-link", path);
    CHECK(status_code(
              hyle_fs_create_symlink(file, link, (hyle_symlink_kind)99)) ==
          HYLE_STATUS_INVALID_ARGUMENT);
    /* Symlink creation may require a host privilege (notably on Windows). */
    if (status_ok(hyle_fs_create_symlink(file, link, HYLE_SYMLINK_AUTO))) {
      char target[1024];
      size_t need = 0;
      CHECK(status_ok(hyle_fs_read_symlink(&need, link, NULL, 0)));
      CHECK(need > 0u);
      size_t got = 0;
      CHECK(
          status_ok(hyle_fs_read_symlink(&got, link, target, sizeof(target))) &&
          got == need);
      CHECK(strcmp(target, file) == 0);
      CHECK(status_ok(hyle_fs_is_symlink(&path_value, link)) && path_value);
    }
  }
  visit_count = 0;
  CHECK(
      status_ok(hyle_fs_directory_iterate(&completed, path, visit_cb, NULL)) &&
      completed);
  CHECK(visit_count >= 1);
  CHECK(status_ok(hyle_fs_remove_directory_recursive(path)));

  char envbuf[64];
  size_t en = 0;
  CHECK(status_code(hyle_env_get(NULL, "PATH", envbuf, sizeof(envbuf))) ==
        HYLE_STATUS_INVALID_ARGUMENT);
  en = 7u;
  envbuf[0] = 'x';
  CHECK(status_code(hyle_env_get(&en, NULL, envbuf, sizeof(envbuf))) ==
        HYLE_STATUS_INVALID_ARGUMENT);
  CHECK(en == 0u && envbuf[0] == '\0');
  CHECK(status_code(hyle_env_set(NULL, "value")) ==
        HYLE_STATUS_INVALID_ARGUMENT);
  CHECK(status_code(hyle_env_set("NAME", NULL)) ==
        HYLE_STATUS_INVALID_ARGUMENT);
  CHECK(status_code(hyle_env_unset(NULL)) == HYLE_STATUS_INVALID_ARGUMENT);
  CHECK(status_ok(hyle_env_get(&en, "PATH", envbuf, sizeof(envbuf))));
  CHECK(en > 0u);

  hyle_cpu_features cpu = HYLE_CPU_FEATURES_INIT;
  CHECK(status_ok(hyle_cpu_features_get(&cpu)));
  CHECK(cpu.struct_size == sizeof(cpu));
  CHECK(cpu.reserved == 0u);
  (void)cpu; /* architecture-specific assertions below may compile out */
  {
    hyle_cpu_features invalid = HYLE_CPU_FEATURES_INIT;
    invalid.reserved = 1u;
    invalid.sse2 = 7;
    invalid.rvv_zvfhmin = 7;
    CHECK(status_code(hyle_cpu_features_get(&invalid)) ==
          HYLE_STATUS_INVALID_ARGUMENT);
    CHECK(invalid.struct_size == sizeof(invalid));
    CHECK(invalid.reserved == 0u);
    CHECK(invalid.sse2 == 0 && invalid.rvv_zvfhmin == 0);
  }
  {
    struct {
      cpu_features_short_prefix known;
      uint32_t canary;
    } short_out;
    memset(&short_out, 0xA5, sizeof(short_out));
    short_out.known.struct_size = (uint32_t)sizeof(short_out.known);
    short_out.known.reserved = 0u;
    short_out.canary = UINT32_C(0xD15EA5E5);
    CHECK(status_ok(hyle_cpu_features_get(
        (hyle_cpu_features *)(void *)&short_out.known)));
    CHECK(short_out.known.struct_size == sizeof(short_out.known));
    CHECK(short_out.known.reserved == 0u);
    CHECK((short_out.known.sse2 == 0 || short_out.known.sse2 == 1) &&
          (short_out.known.ssse3 == 0 || short_out.known.ssse3 == 1));
    CHECK(short_out.canary == UINT32_C(0xD15EA5E5));
  }
  {
    struct {
      uint32_t struct_size;
      uint32_t reserved;
      unsigned char partial_feature;
      unsigned char padding[3];
      uint32_t canary;
    } odd_prefix;
    memset(&odd_prefix, 0xA5, sizeof(odd_prefix));
    odd_prefix.struct_size = HYLE_CPU_FEATURES_REQUIRED_SIZE + 1u;
    odd_prefix.reserved = 0u;
    odd_prefix.canary = UINT32_C(0xD15EA5E5);
    CHECK(status_ok(hyle_cpu_features_get(
        (hyle_cpu_features *)(void *)&odd_prefix)));
    CHECK(odd_prefix.struct_size == HYLE_CPU_FEATURES_REQUIRED_SIZE + 1u);
    CHECK(odd_prefix.reserved == 0u);
    CHECK(odd_prefix.partial_feature == 0xA5u);
    CHECK(odd_prefix.canary == UINT32_C(0xD15EA5E5));
  }
  {
    struct {
      hyle_cpu_features known;
      uint64_t future_tail;
    } future;
    memset(&future, 0, sizeof(future));
    future.known = (hyle_cpu_features)HYLE_CPU_FEATURES_INIT;
    future.known.struct_size = (uint32_t)sizeof(future);
    future.future_tail = UINT64_C(0xD15EA5E5C0DEC0DE);
    CHECK(status_ok(hyle_cpu_features_get(&future.known)));
    CHECK(future.known.struct_size == sizeof(future));
    CHECK(future.known.reserved == 0u);
    CHECK(future.future_tail == UINT64_C(0xD15EA5E5C0DEC0DE));
  }
  CHECK(status_code(hyle_cpu_features_get(NULL)) ==
        HYLE_STATUS_INVALID_ARGUMENT);
  CHECK(hyle_cpu_isa_name(hyle_cpu_best_isa()) != NULL);
  CHECK(hyle_cpu_supports_isa(HYLE_ISA_SCALAR) == 1);
  {
    unsigned cl = hyle_cpu_cache_line_size();
    CHECK(cl >= 16u && cl <= 256u &&
          (cl & (cl - 1u)) == 0u); /* sane power of two */
    CHECK(cl >= HYLE_STATIC_CONSTRUCTIVE_INTERFERENCE_SIZE);
    CHECK(cl <= HYLE_STATIC_DESTRUCTIVE_INTERFERENCE_SIZE);
  }
  {
    hyle_cache_info cache = hyle_cpu_cache_info();
    hyle_cpu_topology topo = hyle_cpu_topology_get();
    if (cache.l1d && cache.l2)
      CHECK(cache.l1d <= cache.l2);
    if (cache.l1i && cache.l2)
      CHECK(cache.l1i <= cache.l2);
    if (cache.l2 && cache.l3)
      CHECK(cache.l2 <= cache.l3);
    CHECK(topo.system_logical_cpus >= 1u);
    CHECK(topo.performance_cpus + topo.efficiency_cpus ==
          topo.system_logical_cpus);
    CHECK((topo.is_hybrid != 0) ==
          (topo.performance_cpus != 0 && topo.efficiency_cpus != 0));
  }
  {
    size_t available = 0;
    unsigned first = ~0u;
    CHECK(status_code(hyle_cpu_available_count(NULL)) ==
          HYLE_STATUS_INVALID_ARGUMENT);
    CHECK(status_ok(hyle_cpu_available_count(&available)));
    CHECK(available >= 1u);
    CHECK(status_code(hyle_cpu_available_ids(NULL, NULL, 0)) ==
          HYLE_STATUS_INVALID_ARGUMENT);
    CHECK(status_ok(hyle_cpu_available_ids(&available, &first, 1)));
    CHECK(available >= 1u && first != ~0u);
    {
      hyle_cpu_info info = {0, 0, HYLE_CPU_KIND_UNKNOWN};
      size_t info_count = 0;
      CHECK(status_ok(hyle_cpu_available_info(&info_count, &info, 1)));
      CHECK(info_count == available && info.id == first);
      CHECK(info.kind <= HYLE_CPU_KIND_PERFORMANCE);

      unsigned preferred = ~0u;
      size_t ordered_count = 0;
      CHECK(status_ok(hyle_cpu_available_ids_ordered(
          &ordered_count, HYLE_CPU_ORDER_PERFORMANCE_FIRST, &preferred, 1)));
      CHECK(ordered_count == available && preferred != ~0u);
      CHECK(status_code(hyle_cpu_available_ids_ordered(
                &ordered_count, (hyle_cpu_order)99, &preferred, 1)) ==
            HYLE_STATUS_INVALID_ARGUMENT);
    }
  }
  {
    hyle_isa isa = HYLE_ISA_SCALAR; /* named enums usable as types */
    hyle_numa_node nany = HYLE_NUMA_ANY;
    CHECK(hyle_cpu_supports_isa(isa) == 1);
    CHECK((int)nany == -1);
  }
#if defined(_WIN32) || defined(__x86_64__) || defined(_M_X64)
  CHECK(cpu.sse2 == 1);
  CHECK(hyle_cpu_supports_isa(HYLE_ISA_SSE2) == 1);
#endif

  /* v3: NUMA topology + placement */
  {
    unsigned nc = hyle_numa_node_count();
    CHECK(nc >= 1u);
    size_t required_nodes = hyle_numa_nodes(NULL, 0);
    CHECK(required_nodes == nc);
    hyle_numa_node *node_ids =
        (hyle_numa_node *)malloc(required_nodes * sizeof(hyle_numa_node));
    CHECK(node_ids != NULL);
    if (node_ids == NULL)
      return 1;
    CHECK(hyle_numa_nodes(node_ids, required_nodes) == required_nodes);
    int current_found = 0;
    int cpu_found = 0;
    for (size_t i = 0; i < required_nodes; ++i) {
      if (node_ids[i] == hyle_numa_current_node())
        current_found = 1;
      if (node_ids[i] == hyle_numa_node_of_cpu(0))
        cpu_found = 1;
      if (i != 0)
        CHECK(node_ids[i - 1] < node_ids[i]);
    }
    CHECK(current_found);
    CHECK(cpu_found);
    hyle_numa_capability numa_capabilities = hyle_numa_capabilities();
    CHECK((numa_capabilities & ~((1u << 5) - 1u)) == 0u);
    int selected_available =
        nc <= 1u ||
        (numa_capabilities & HYLE_NUMA_CAP_SELECTED_INTERLEAVE) != 0u;

    size_t pg = hyle_page_size();
    /* ANY == plain reserve; region is writable after commit. */
    void *rn = NULL;
    hyle_numa_placement placement = HYLE_NUMA_PLACEMENT_APPLIED;
    CHECK(status_ok(
        hyle_vm_reserve_numa(&rn, &placement, pg * 2, HYLE_NUMA_ANY)));
    CHECK(rn != NULL);
    CHECK(placement == HYLE_NUMA_PLACEMENT_NOT_NEEDED);
    CHECK(status_ok(hyle_vm_commit(rn, pg * 2)));
    ((unsigned char *)rn)[0] = 0x5A;
    CHECK(((unsigned char *)rn)[0] == 0x5A);
    /* Empty range bind is always a no-op success. */
    CHECK(status_ok(
        hyle_vm_bind_numa(&placement, NULL, 0, 0, HYLE_NUMA_PREFERRED)));
    CHECK(placement == HYLE_NUMA_PLACEMENT_NOT_NEEDED);
    placement = HYLE_NUMA_PLACEMENT_APPLIED;
    CHECK(status_code(
              hyle_vm_bind_numa(&placement, NULL, 0, 0,
                                (hyle_numa_policy)99)) ==
          HYLE_STATUS_INVALID_ARGUMENT);
    CHECK(placement == HYLE_NUMA_PLACEMENT_UNAVAILABLE);
    placement = HYLE_NUMA_PLACEMENT_APPLIED;
    CHECK(status_code(hyle_vm_bind_numa_interleave(&placement, NULL, pg)) ==
          HYLE_STATUS_INVALID_ARGUMENT);
    CHECK(placement == HYLE_NUMA_PLACEMENT_UNAVAILABLE);
    /* Preferred to a valid node: soft, always OK. */
    CHECK(status_ok(hyle_vm_bind_numa(&placement, rn, pg * 2, node_ids[0],
                                      HYLE_NUMA_PREFERRED)));
    /* Single-node box: hard bind / interleave degrade to no-op success. */
    if (nc <= 1u) {
      CHECK(status_ok(hyle_vm_bind_numa(&placement, rn, pg * 2, node_ids[0],
                                        HYLE_NUMA_BIND)));
      CHECK(status_ok(hyle_vm_bind_numa_interleave(&placement, rn, pg * 2)));
    }
    CHECK(status_ok(hyle_vm_release(rn, pg * 2)));

    /* reserve preferring the current home node. */
    void *rl = NULL;
    CHECK(status_ok(
        hyle_vm_reserve_numa(&rl, &placement, pg, hyle_numa_current_node())));
    CHECK(rl != NULL);
    CHECK(status_ok(hyle_vm_release(rl, pg)));
    {
      void *stale_reservation = (void *)(uintptr_t)1;
      placement = HYLE_NUMA_PLACEMENT_APPLIED;
      hyle_status bad_reserve = hyle_vm_reserve_numa(
          &stale_reservation, &placement, pg, (hyle_numa_node)-2);
      CHECK(hyle_status_is_out_of_range(bad_reserve));
      CHECK(stale_reservation == NULL);
      CHECK(placement == HYLE_NUMA_PLACEMENT_UNAVAILABLE);
      hyle_status_free(&bad_reserve);
    }

    /* selected-node/chunk committed allocation; ordinary release owns it. */
    void *striped = NULL;
    hyle_status striped_status = hyle_vm_alloc_numa_interleaved(
        &striped, &placement, pg * 8, node_ids, 1, pg * 2);
    if (selected_available) {
      CHECK(hyle_status_is_ok(striped_status));
      CHECK(striped != NULL);
    } else {
      CHECK(hyle_status_is_unimplemented(striped_status));
      CHECK(striped == NULL);
    }
    hyle_status_free(&striped_status);
    if (striped != NULL) {
      ((unsigned char *)striped)[pg * 7] = 0x7C;
      CHECK(((unsigned char *)striped)[pg * 7] == 0x7C);
      CHECK(status_ok(hyle_vm_release(striped, pg * 8)));
    }
    striped = (void *)(uintptr_t)1;
    placement = HYLE_NUMA_PLACEMENT_APPLIED;
    CHECK(status_code(hyle_vm_alloc_numa_interleaved(
              &striped, &placement, pg, NULL, 1, pg)) ==
          HYLE_STATUS_INVALID_ARGUMENT);
    CHECK(striped == NULL);
    CHECK(placement == HYLE_NUMA_PLACEMENT_UNAVAILABLE);
    {
      hyle_numa_node invalid_node = HYLE_NUMA_ANY;
      CHECK(status_code(hyle_vm_alloc_numa_interleaved(&striped, NULL, pg,
                                                       &invalid_node, 1, pg)) ==
            HYLE_STATUS_INVALID_ARGUMENT);
      CHECK(striped == NULL);
    }
    hyle_numa_node duplicate[2] = {node_ids[0], node_ids[0]};
    hyle_status weighted_status =
        hyle_vm_alloc_numa_interleaved(&striped, NULL, pg, duplicate, 2, pg);
    if (selected_available) {
      CHECK(status_ok(weighted_status));
      CHECK(striped != NULL);
      CHECK(status_ok(hyle_vm_release(striped, pg)));
      striped = NULL;
    } else {
      CHECK(hyle_status_is_unimplemented(weighted_status));
      CHECK(striped == NULL);
    }
    hyle_status_free(&weighted_status);

    hyle_vm_region_o *striped_region = NULL;
    hyle_status region_status = hyle_vm_region_alloc_numa_interleaved(
        &striped_region, pg * 4, node_ids, 1, pg);
    if (selected_available) {
      CHECK(hyle_status_is_ok(region_status));
      CHECK(striped_region != NULL);
      CHECK(hyle_vm_region_is_numa_interleaved(striped_region));
      CHECK(hyle_vm_region_numa_placement(striped_region) == placement ||
            hyle_vm_region_numa_placement(striped_region) ==
                HYLE_NUMA_PLACEMENT_NOT_NEEDED);
      CHECK(status_code(hyle_vm_region_decommit(striped_region, 0, pg)) ==
            HYLE_STATUS_FAILED_PRECONDITION);
      hyle_vm_region_destroy(striped_region);
    } else {
      CHECK(hyle_status_is_unimplemented(region_status));
      CHECK(striped_region == NULL);
    }
    hyle_status_free(&region_status);
    {
      hyle_numa_node invalid_node = HYLE_NUMA_ANY;
      striped_region = (hyle_vm_region_o *)(uintptr_t)1;
      region_status = hyle_vm_region_alloc_numa_interleaved(
          &striped_region, pg, &invalid_node, 1, pg);
      CHECK(hyle_status_is_invalid_argument(region_status));
      CHECK(striped_region == NULL);
      hyle_status_free(&region_status);
    }
    free(node_ids);
  }

  /* v3: hugepage / THP hints (soft; size may be 0 on exotic arches) */
  {
    size_t lp = hyle_large_page_size();
    CHECK(hyle_large_page_align_up(0) == 0u);
    if (lp != 0u) {
      CHECK(hyle_large_page_align_up(1) == lp);
    }
    /* advise is best-effort: call on a committed region, don't assert result.
     */
    size_t pg = hyle_page_size();
    void *h = NULL;
    CHECK(status_ok(hyle_vm_alloc_rw(&h, pg)));
    CHECK(h != NULL);
    status_discard(hyle_vm_advise_hugepage(h, pg));
    status_discard(hyle_vm_advise_no_hugepage(h, pg));
    CHECK(status_ok(hyle_vm_free_rw(h, pg)));
  }

  /* v4: fs permissions + env write + temp file + mapped file */
  {
    /* env round trip */
    CHECK(status_ok(hyle_env_set("HYLE_C_PLAT_V4", "yes")));
    char eb[16];
    size_t env_size = 0;
    CHECK(status_ok(hyle_env_get(&env_size, "HYLE_C_PLAT_V4", eb, sizeof(eb))));
    CHECK(env_size == 3u);
    CHECK(strcmp(eb, "yes") == 0);
    CHECK(status_ok(hyle_env_unset("HYLE_C_PLAT_V4")));
    env_size = 7u;
    CHECK(status_code(hyle_env_get(&env_size, "HYLE_C_PLAT_V4", eb,
                                   sizeof(eb))) == HYLE_STATUS_NOT_FOUND);
    CHECK(env_size == 0u);

    /* temp file: open handle + real path */
    char tpath[1024];
    hyle_file_handle fd = HYLE_FILE_HANDLE_INVALID;
    size_t tpath_size = 0;
    CHECK(status_code(hyle_fs_create_temp_file(
              NULL, &tpath_size, NULL, "hyle_c", tpath, sizeof(tpath))) ==
          HYLE_STATUS_INVALID_ARGUMENT);
    fd = 7;
    CHECK(status_code(hyle_fs_create_temp_file(&fd, NULL, NULL, "hyle_c", tpath,
                                               sizeof(tpath))) ==
          HYLE_STATUS_INVALID_ARGUMENT);
    CHECK(fd == HYLE_FILE_HANDLE_INVALID);
    CHECK(status_ok(hyle_fs_create_temp_file(&fd, &tpath_size, NULL, "hyle_c",
                                             tpath, sizeof(tpath))));
    CHECK(fd != HYLE_FILE_HANDLE_INVALID);
    CHECK(tpath_size == strlen(tpath));
    CHECK(status_ok(hyle_fs_exists(&path_value, tpath)) && path_value);
    /* permissions: the temp file exists -> nonzero perms; readonly round trip
     */
    hyle_file_perms temp_mode = 0;
    CHECK(status_ok(hyle_fs_get_permissions(&temp_mode, tpath)) &&
          temp_mode != 0u);
    CHECK(status_ok(hyle_fs_set_readonly(tpath, 1)));
    CHECK(status_ok(hyle_fs_is_readonly(&path_value, tpath)) && path_value);
    CHECK(status_ok(hyle_fs_set_readonly(tpath, 0)));
    CHECK(status_ok(hyle_fs_close(fd)));

    /* File-handle I/O: write bytes, then read them back through the C API. */
    hyle_file_handle wfd = HYLE_FILE_HANDLE_INVALID;
    CHECK(status_code(hyle_fs_open(NULL, tpath, HYLE_OPEN_READ)) ==
          HYLE_STATUS_INVALID_ARGUMENT);
    CHECK(status_ok(hyle_fs_open(
        &wfd, tpath, HYLE_OPEN_WRITE | HYLE_OPEN_CREATE | HYLE_OPEN_TRUNCATE)));
    CHECK(wfd != HYLE_FILE_HANDLE_INVALID);
    const char *payload = "c-abi file io";
    const size_t payload_size = strlen(payload);
    size_t moved = 0;
    CHECK(status_code(hyle_fs_read(NULL, wfd, NULL, 0)) ==
          HYLE_STATUS_INVALID_ARGUMENT);
    CHECK(status_code(hyle_fs_write(NULL, wfd, NULL, 0)) ==
          HYLE_STATUS_INVALID_ARGUMENT);
    CHECK(status_ok(hyle_fs_write(&moved, wfd, payload, payload_size)) &&
          moved == payload_size);
    CHECK(status_ok(hyle_fs_close(wfd)));
    hyle_file_handle rfd = HYLE_FILE_HANDLE_INVALID;
    CHECK(status_ok(hyle_fs_open(&rfd, tpath, HYLE_OPEN_READ)));
    CHECK(rfd != HYLE_FILE_HANDLE_INVALID);
    char rbuf[32] = {0};
    CHECK(status_ok(hyle_fs_read(&moved, rfd, rbuf, sizeof(rbuf))) &&
          moved == payload_size && memcmp(rbuf, payload, payload_size) == 0);
    CHECK(status_ok(hyle_fs_close(rfd)));
    CHECK(status_code(hyle_fs_open(&rfd, "/no/such/hyle/path",
                                   HYLE_OPEN_READ)) == HYLE_STATUS_NOT_FOUND);
    CHECK(rfd == HYLE_FILE_HANDLE_INVALID);
    CHECK(status_code(hyle_fs_open(&rfd, tpath, HYLE_OPEN_CREATE)) ==
          HYLE_STATUS_INVALID_ARGUMENT);
    CHECK(rfd == HYLE_FILE_HANDLE_INVALID);
    size_t temp_size = 0;
    CHECK(status_ok(hyle_fs_temp_directory(&temp_size, NULL, 32)) &&
          temp_size > 0u); /* NULL dst = size query */
    CHECK(status_code(hyle_fs_read(&moved, HYLE_FILE_HANDLE_INVALID, NULL,
                                   1)) == HYLE_STATUS_INVALID_ARGUMENT);
    CHECK(status_ok(hyle_fs_close(HYLE_FILE_HANDLE_INVALID)));
    {
      hyle_file_open_flags typed_flags = HYLE_OPEN_READ;
      hyle_file_perms typed_perms = HYLE_PERM_OWNER_READ;
      CHECK((unsigned)typed_flags == HYLE_OPEN_READ);
      CHECK((unsigned)typed_perms == HYLE_PERM_OWNER_READ);
    }

    /* mapped file: tpath contains the payload written above; exercise the
     * Status-returning open plus invalid-mode failure. */
    hyle_mapped_file_o *m = NULL;
    hyle_status map_status =
        hyle_mapped_file_open(&m, tpath, HYLE_MAP_READ_ONLY, 0, 0);
    CHECK(hyle_status_is_ok(map_status));
    CHECK(m != NULL);
    CHECK(hyle_mapped_file_valid(m));
    CHECK(hyle_mapped_file_data(m) != NULL);
    CHECK(hyle_mapped_file_size(m) == payload_size);
    map_status = hyle_mapped_file_advise(m, HYLE_MAP_ADVICE_NORMAL, 0, 0);
    CHECK(hyle_status_is_ok(map_status));
    hyle_status_free(&map_status);
    CHECK(!hyle_mapped_file_locked(m));
    map_status = hyle_mapped_file_lock(m);
    if (hyle_status_is_ok(map_status)) {
      CHECK(hyle_mapped_file_locked(m));
      hyle_status_free(&map_status);
      map_status = hyle_mapped_file_unlock(m);
      CHECK(hyle_status_is_ok(map_status));
      CHECK(!hyle_mapped_file_locked(m));
    }
    hyle_status_free(&map_status);
    hyle_mapped_file_destroy(m);
    map_status = hyle_mapped_file_open(&m, tpath, (hyle_map_mode)99, 0, 0);
    CHECK(!hyle_status_is_ok(map_status));
    CHECK(hyle_status_is_invalid_argument(map_status));
    CHECK(m == NULL);
    hyle_status_free(&map_status);
    map_status = hyle_mapped_file_open(NULL, tpath, HYLE_MAP_READ_ONLY, 0, 0);
    CHECK(hyle_status_is_invalid_argument(map_status));
    hyle_status_free(&map_status);
    map_status = hyle_mapped_file_flush(NULL);
    CHECK(hyle_status_is_invalid_argument(map_status));
    hyle_status_free(&map_status);
    map_status = hyle_mapped_file_advise(NULL, HYLE_MAP_ADVICE_NORMAL, 0, 0);
    CHECK(hyle_status_is_invalid_argument(map_status));
    hyle_status_free(&map_status);
    map_status = hyle_mapped_file_lock(NULL);
    CHECK(hyle_status_is_invalid_argument(map_status));
    hyle_status_free(&map_status);
    map_status = hyle_mapped_file_unlock(NULL);
    CHECK(hyle_status_is_invalid_argument(map_status));
    hyle_status_free(&map_status);

    /* hard link: a second name for the same file (same volume). */
    char hpath[1040];
    snprintf(hpath, sizeof(hpath), "%s.hln", tpath);
    if (status_ok(hyle_fs_create_hard_link(tpath, hpath))) {
      CHECK(status_ok(hyle_fs_exists(&path_value, hpath)) && path_value);
      CHECK(status_ok(hyle_fs_is_symlink(&path_value, hpath)) && !path_value);
      status_discard(hyle_fs_remove_file(hpath));
    }
    status_discard(hyle_fs_remove_file(tpath));
  }

  /* mapped_file DATA path over the C ABI: valid / data / size / flush on a
   * mapping that actually has content (the empty-file map above only exercised
   * open+destroy). Seed a file, map it read-write, verify size + bytes, dirty a
   * byte + flush, then re-open read-only to confirm the flush hit the file. */
  {
    char mpath[1024];
    hyle_file_handle fd = HYLE_FILE_HANDLE_INVALID;
    size_t mpath_size = 0;
    CHECK(status_ok(hyle_fs_create_temp_file(
        &fd, &mpath_size, NULL, "hyle_c_map", mpath, sizeof(mpath))));
    CHECK(fd != HYLE_FILE_HANDLE_INVALID);
    CHECK(mpath_size == strlen(mpath));
    CHECK(status_ok(hyle_fs_close(fd)));

    const char *seed = "hyle mapped C-ABI payload 0123456789";
    size_t seed_len = strlen(seed);
    FILE *sf = fopen(mpath, "wb");
    CHECK(sf != NULL);
    if (sf != NULL) {
      CHECK(fwrite(seed, 1, seed_len, sf) == seed_len);
      fclose(sf);

      hyle_mapped_file_o *m = NULL;
      hyle_status map_status =
          hyle_mapped_file_open(&m, mpath, HYLE_MAP_READ_WRITE, 0, 0);
      CHECK(hyle_status_is_ok(map_status));
      hyle_status_free(&map_status);
      CHECK(m != NULL);
      CHECK(hyle_mapped_file_valid(m) == 1);
      CHECK(hyle_mapped_file_size(m) == seed_len);
      unsigned char *d = (unsigned char *)hyle_mapped_file_data(m);
      CHECK(d != NULL);
      if (d != NULL) {
        CHECK(memcmp(d, seed, seed_len) == 0);
        d[0] = (unsigned char)'Z'; /* dirty the first byte */
        map_status = hyle_mapped_file_flush(m);
        CHECK(hyle_status_is_ok(map_status)); /* RW flush persists to file */
        hyle_status_free(&map_status);
      }
      hyle_mapped_file_destroy(m);

      /* re-open read-only: the flushed write must be visible on disk. */
      hyle_mapped_file_o *m2 = NULL;
      map_status = hyle_mapped_file_open(&m2, mpath, HYLE_MAP_READ_ONLY, 0, 0);
      CHECK(hyle_status_is_ok(map_status));
      hyle_status_free(&map_status);
      CHECK(m2 != NULL);
      CHECK(hyle_mapped_file_valid(m2) == 1);
      if (hyle_mapped_file_valid(m2)) {
        CHECK(((unsigned char *)hyle_mapped_file_data(m2))[0] ==
              (unsigned char)'Z');
      }
      hyle_mapped_file_destroy(m2);
    }
    status_discard(hyle_fs_remove_file(mpath));
  }

  if (g_fails) {
    fprintf(stderr, "c_platform_test: %d failed\n", g_fails);
    return 1;
  }
  printf("c_platform_test: all checks passed\n");
  return 0;
}
