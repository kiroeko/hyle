/* Pure C consumer of include/hyle/c/platform.h */
#if defined(_WIN32)
#  define _CRT_SECURE_NO_WARNINGS 1
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif
#include "hyle/c/platform.h"

#include <stdio.h>
#include <string.h>

static int g_fails = 0;
#define CHECK(c)                                                               \
  do {                                                                         \
    if (!(c)) {                                                                \
      fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #c, __FILE__, __LINE__);  \
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

static void thr_entry(void* ctx) {
  int* p = (int*)ctx;
  *p = 42;
}

static int visit_count = 0;
static int visit_cb(const char* name, int is_dir, void* ctx) {
  (void)name;
  (void)is_dir;
  (void)ctx;
  ++visit_count;
  return 1;
}

/* Low-level wait/wake worker: block until the shared word turns non-zero, then
 * record that we woke. The value-recheck loop makes the wake race-free (an early
 * store+wake before we park is simply observed by the loop condition) and absorbs
 * spurious wakes. The word is the futex/WaitOnAddress target; C11 atomic
 * load/store carry the cross-thread ordering, while wait/wake only block and
 * notify on its address. */
typedef struct {
  volatile uint32_t* word;
  int* woke;
} wait_ctx;
static uint32_t atomic_load_word(volatile uint32_t* word) {
#if defined(_WIN32)
  return (uint32_t)InterlockedCompareExchange((volatile LONG*)word, 0, 0);
#else
  return __atomic_load_n(word, __ATOMIC_ACQUIRE);
#endif
}
static void atomic_store_word(volatile uint32_t* word, uint32_t value) {
#if defined(_WIN32)
  (void)InterlockedExchange((volatile LONG*)word, (LONG)value);
#else
  __atomic_store_n(word, value, __ATOMIC_RELEASE);
#endif
}
static void wait_entry(void* ctx) {
  wait_ctx* x = (wait_ctx*)ctx;
  while (atomic_load_word(x->word) == 0u) {
    hyle_wait_on_address(x->word, 0u);
  }
  *x->woke = 1;
}

int main(void) {
  CHECK(hyle_page_size() >= 4096u);
  CHECK(hyle_page_align_up(SIZE_MAX) == 0u);
  CHECK(hyle_large_page_align_up(SIZE_MAX) == 0u);
  CHECK(hyle_platform_last_error() != NULL);

  /* Parker */
  hyle_parker_o* p = NULL;
  CHECK(status_code(hyle_parker_create(NULL)) == HYLE_STATUS_INVALID_ARGUMENT);
  CHECK(status_ok(hyle_parker_create(&p)));
  CHECK(p != NULL);
  hyle_parker_unpark(p);
  hyle_parker_park(p);
  CHECK(hyle_parker_park_timeout(p, 0) == 0);
  hyle_parker_destroy(p);

  /* Low-level wait/wake (futex / WaitOnAddress thin forwards). First the
   * non-blocking contracts (no thread needed): a value MISMATCH never blocks and
   * reports "not a timeout" (1); a zero timeout on a matching value is a poll;
   * waking an address with no waiter is a no-op. */
  {
    uint32_t w = 5u;
    CHECK(hyle_wait_on_address_for(&w, 7u, 0) == 1); /* 5 != 7 -> immediate, 1 */
    (void)hyle_wait_on_address_for(&w, 5u, 0);       /* poll: 0 (or spurious 1) */
    hyle_wait_on_address(&w, 7u);                    /* 5 != 7 -> returns at once */
    hyle_wake_one(&w);                               /* no waiter -> no-op */
    hyle_wake_all(&w);                               /* no waiter -> no-op */
  }

  /* Real cross-thread wake_one: one waiter blocks on the word, main flips it and
   * wakes exactly one. Proves the addr forward + wake path end to end. */
  {
    volatile uint32_t word = 0u;
    int woke = 0;
    wait_ctx c = {&word, &woke};
    hyle_thread_o* wt = NULL;
    hyle_status ws = hyle_thread_spawn_opts(&wt, wait_entry, &c, "c_wait1", 0, -1,
                                            HYLE_THREAD_PRIORITY_NORMAL);
    CHECK(hyle_status_is_ok(ws));
    hyle_status_free(&ws);
    CHECK(wt != NULL);
    if (wt != NULL) {
      atomic_store_word(&word, 1u);
      hyle_wake_one(&word);
      hyle_thread_join(wt);  /* hangs only if the wake never lands (broken fwd) */
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
    hyle_status sa = hyle_thread_spawn_opts(&ta, wait_entry, &ca, "c_waitA", 0, -1,
                                            HYLE_THREAD_PRIORITY_NORMAL);
    hyle_status sb = hyle_thread_spawn_opts(&tb, wait_entry, &cb, "c_waitB", 0, -1,
                                            HYLE_THREAD_PRIORITY_NORMAL);
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
  hyle_thread_o* t = NULL;
  hyle_thread_opts opts = hyle_thread_opts_default();
  CHECK(opts.struct_size == sizeof(opts));
  hyle_thread_opts_set_name(&opts, "c_plat_worker");
  hyle_thread_opts_set_stack_size(&opts, 2u * 1024u * 1024u);
  CHECK(opts.has_stack_size && opts.stack_size == 2u * 1024u * 1024u);
  opts.has_stack_size = 0; /* spawn with the OS default after testing the helper */
  hyle_thread_opts_set_affinity(&opts, 0);
  CHECK(opts.has_affinity_cpu && opts.affinity_cpu == 0u);
  opts.has_affinity_cpu = 0; /* do not constrain heterogeneous CI hosts */
  hyle_thread_opts_set_priority(&opts, HYLE_THREAD_PRIORITY_NORMAL);
  CHECK(opts.has_priority && opts.priority == HYLE_THREAD_PRIORITY_NORMAL);
  hyle_status st = hyle_thread_spawn(&t, thr_entry, &ran, &opts);
  CHECK(hyle_status_is_ok(st));
  hyle_status_free(&st);
  CHECK(t != NULL);
  CHECK(hyle_thread_joinable(t));
  hyle_thread_join(t);
  CHECK(ran == 42);

  {
    hyle_thread_opts too_small = HYLE_THREAD_OPTS_INIT;
    hyle_thread_o* bad = NULL;
    too_small.struct_size = 1;
    hyle_status small =
        hyle_thread_spawn(&bad, thr_entry, &ran, &too_small);
    CHECK(hyle_status_is_invalid_argument(small));
    CHECK(bad == NULL);
    hyle_status_free(&small);

    too_small.struct_size = 0;
    small = hyle_thread_spawn(&bad, thr_entry, &ran, &too_small);
    CHECK(hyle_status_is_invalid_argument(small));
    CHECK(bad == NULL);
    hyle_status_free(&small);
  }

  /* FFI-friendly spawn_opts: no struct/has_ flags, sentinels for defaults
   * (stack_size 0 => default, affinity_cpu -1 => none). */
  {
    int ran2 = 0;
    hyle_thread_o* t2 = NULL;
    hyle_status st2 = hyle_thread_spawn_opts(&t2, thr_entry, &ran2, "c_plat_opts",
                                             0, -1, HYLE_THREAD_PRIORITY_NORMAL);
    CHECK(hyle_status_is_ok(st2));
    hyle_status_free(&st2);
    CHECK(t2 != NULL);
    hyle_thread_join(t2);
    CHECK(ran2 == 42);
  }

  /* Invalid enum values are rejected at the C boundary, never silently
   * reinterpreted as a valid scheduling policy. */
  {
    hyle_thread_o* bad = NULL;
    hyle_status bad_st =
        hyle_thread_spawn_opts(&bad, thr_entry, &ran, "bad_priority", 0, -1, 99);
    CHECK(hyle_status_is_invalid_argument(bad_st));
    CHECK(bad == NULL);
    hyle_status_free(&bad_st);
  }

  CHECK(hyle_hardware_concurrency() >= 1u);
  CHECK(hyle_this_thread_id() != 0);
  hyle_this_thread_yield();
  hyle_this_thread_set_name("hyle_c_main");

  /* DynLib */
#if defined(_WIN32)
  const char* sys_lib = "kernel32.dll";
  const char* sys_sym = "GetCurrentProcessId";
  const char* ext = "dll";
#elif defined(__APPLE__)
  /* install-name, not the search alias: hyle_dynlib_get uses RTLD_NOLOAD which
   * keys on the loaded image's install-name (libSystem.B), not "libSystem.dylib". */
  const char* sys_lib = "libSystem.B.dylib";
  const char* sys_sym = "getpid";
  const char* ext = "dylib";
#else
  const char* sys_lib = "libm.so.6";
  const char* sys_sym = "cos";
  const char* ext = "so";
#endif
  hyle_dynlib_o* lib = NULL;
  hyle_status dynlib_status = hyle_dynlib_get(&lib, sys_lib);
  CHECK(hyle_status_is_ok(dynlib_status));
  hyle_status_free(&dynlib_status);
  CHECK(hyle_dynlib_valid(lib));
  {
    void* symbol = NULL;
    dynlib_status = hyle_dynlib_symbol(&symbol, lib, sys_sym);
    CHECK(hyle_status_is_ok(dynlib_status));
    CHECK(symbol != NULL);
    hyle_status_free(&dynlib_status);
  }
  hyle_dynlib_close(lib);
  {
    hyle_dynlib_o* current = NULL;
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
    void* cleared = (void*)(uintptr_t)1;
    dynlib_status = hyle_dynlib_symbol(&cleared, NULL, sys_sym);
    CHECK(hyle_status_is_invalid_argument(dynlib_status));
    CHECK(cleared == NULL);
    hyle_status_free(&dynlib_status);
  }

  /* open_symbol on already-loadable system lib path may fail if path needs
   * absolute form; exercise API shape via get+symbol path above. Missing path: */
  {
    hyle_dynlib_o* miss = NULL;
    void* miss_sym = NULL;
    dynlib_status = hyle_dynlib_open_symbol(
        &miss, &miss_sym, "hyle_no_such_plugin_xyz.so", "entry");
    CHECK(!hyle_status_is_ok(dynlib_status));
    CHECK(hyle_status_is_not_found(dynlib_status));
    hyle_status_free(&dynlib_status);
    CHECK(miss == NULL);
    CHECK(miss_sym == NULL);
  }

  /* VM guarded */
  hyle_vm_region_o* reg = NULL;
  CHECK(status_ok(hyle_vm_region_reserve_guarded(&reg, hyle_page_size())));
  CHECK(hyle_vm_region_valid(reg));
  CHECK(hyle_vm_region_has_guard(reg));
  CHECK(hyle_vm_region_usable_size(reg) == hyle_page_size());
  {
    unsigned char* u = (unsigned char*)hyle_vm_region_usable(reg);
    CHECK(u != NULL);
    u[0] = 0xAB;
    CHECK(u[0] == 0xAB);
  }
  hyle_vm_region_destroy(reg);

  /* v5: region reserve_numa (ANY == plain) + adopt an existing raw reservation */
  {
    size_t pg = hyle_page_size();
    hyle_vm_region_o* rn = NULL;
    CHECK(status_ok(
        hyle_vm_region_reserve_numa(&rn, pg * 2, HYLE_NUMA_ANY)));
    CHECK(hyle_vm_region_valid(rn));
    CHECK(status_ok(hyle_vm_region_commit(rn, 0, pg * 2)));
    ((unsigned char*)hyle_vm_region_data(rn))[0] = 0x5A;
    CHECK(((unsigned char*)hyle_vm_region_data(rn))[0] == 0x5A);
    hyle_vm_region_destroy(rn);

    /* adopt: raw reserve -> hand ownership to a region handle (destroy releases) */
    void* raw = NULL;
    CHECK(status_ok(hyle_vm_reserve(&raw, pg)));
    CHECK(raw != NULL);
    hyle_vm_region_o* ad = NULL;
    CHECK(status_ok(hyle_vm_region_adopt(&ad, raw, pg)));
    CHECK(hyle_vm_region_valid(ad));
    CHECK(hyle_vm_region_data(ad) == raw);
    CHECK(hyle_vm_region_size(ad) == pg);
    hyle_vm_region_destroy(ad);  /* releases the adopted reservation */
    /* adopt(NULL) fails without taking ownership. */
    CHECK(status_code(hyle_vm_region_adopt(&ad, NULL, pg)) ==
          HYLE_STATUS_INVALID_ARGUMENT);
  }

  /* free vm + alloc_rw */
  {
    size_t pg = hyle_page_size();
    void* base = NULL;
    CHECK(status_ok(hyle_vm_reserve(&base, pg)));
    CHECK(base != NULL);
    CHECK(status_ok(hyle_vm_commit(base, pg)));
    CHECK(status_ok(hyle_vm_release(base, pg)));

    void* rw = NULL;
    CHECK(status_ok(hyle_vm_alloc_rw(&rw, pg / 2 + 1)));
    CHECK(rw != NULL);
    ((unsigned char*)rw)[0] = 1;
    CHECK(status_ok(hyle_vm_free_rw(rw, pg / 2 + 1)));
  }

  /* free-function vm_protect / vm_decommit thin forwards. Two pages so we can
   * lock/decommit one while keeping the other usable. (The C++ death tests prove
   * a locked page actually faults; here we just cover the C forward + recovery.) */
  {
    size_t pg = hyle_page_size();
    void* base = NULL;
    CHECK(status_ok(hyle_vm_reserve(&base, pg * 2)));
    CHECK(base != NULL);
    CHECK(status_ok(hyle_vm_commit(base, pg * 2)));
    ((unsigned char*)base)[0] = 0x10;

    /* protect the low page read-only: still readable, restore RW and rewrite. */
    CHECK(status_ok(hyle_vm_protect(base, pg, HYLE_PROTECT_READ)));
    CHECK(((unsigned char*)base)[0] == 0x10);
    CHECK(status_ok(hyle_vm_protect(base, pg, HYLE_PROTECT_READ_WRITE)));
    CHECK(status_code(hyle_vm_protect(base, pg, (hyle_protect)99)) ==
          HYLE_STATUS_INVALID_ARGUMENT);
    ((unsigned char*)base)[0] = 0x20;
    CHECK(((unsigned char*)base)[0] == 0x20);

    /* decommit the high page, then recommit and use it (contents undefined). */
    CHECK(status_ok(hyle_vm_decommit((unsigned char*)base + pg, pg)));
    CHECK(status_ok(hyle_vm_commit((unsigned char*)base + pg, pg)));
    ((unsigned char*)base)[pg] = 0x30;
    CHECK(((unsigned char*)base)[pg] == 0x30);

    CHECK(status_ok(hyle_vm_release(base, pg * 2)));
  }

  /* managed region: commit_usable / protect / decommit thin forwards. */
  {
    size_t pg = hyle_page_size();
    hyle_vm_region_o* r = NULL;
    CHECK(status_ok(hyle_vm_region_reserve(&r, pg * 2)));
    CHECK(hyle_vm_region_valid(r));
    /* commit_usable: whole reservation RW (no guard on a plain reserve). */
    CHECK(status_ok(hyle_vm_region_commit_usable(r)));
    unsigned char* d = (unsigned char*)hyle_vm_region_data(r);
    CHECK(d != NULL);
    if (d != NULL) {
      d[0] = 0x41;
      CHECK(d[0] == 0x41);
      /* protect page 0 read-only then restore. */
      CHECK(status_ok(
          hyle_vm_region_protect(r, 0, pg, HYLE_PROTECT_READ)));
      CHECK(d[0] == 0x41); /* read still allowed */
      CHECK(status_ok(
          hyle_vm_region_protect(r, 0, pg, HYLE_PROTECT_READ_WRITE)));
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
  size_t n = hyle_fs_temp_directory(tmp, sizeof(tmp));
  CHECK(n > 0 && n < sizeof(tmp));
  int path_value = 0;
  CHECK(status_code(hyle_fs_exists(NULL, tmp)) ==
        HYLE_STATUS_INVALID_ARGUMENT);
  CHECK(status_code(hyle_fs_stat(NULL, tmp)) ==
        HYLE_STATUS_INVALID_ARGUMENT);
  CHECK(status_code(hyle_fs_directory_iterate(
            NULL, tmp, visit_cb, NULL)) == HYLE_STATUS_INVALID_ARGUMENT);
  path_value = 7;
  CHECK(status_code(hyle_fs_exists(&path_value, NULL)) ==
        HYLE_STATUS_INVALID_ARGUMENT);
  CHECK(path_value == 0);
  path_value = 7;
  CHECK(status_code(hyle_fs_directory_iterate(
            &path_value, tmp, NULL, NULL)) == HYLE_STATUS_INVALID_ARGUMENT);
  CHECK(path_value == 0);
  CHECK(status_ok(hyle_fs_exists(&path_value, tmp)) && path_value);
  CHECK(status_ok(hyle_fs_is_directory(&path_value, tmp)) && path_value);
  {
    char cwd[1024];
    char home[1024];
    CHECK(hyle_fs_current_directory(cwd, sizeof(cwd)) > 0u);
    CHECK(hyle_fs_home_directory(home, sizeof(home)) > 0u);
  }
  char path[640];
  snprintf(path, sizeof(path), "%s/hyle_c_plat_test_dir", tmp);
  CHECK(status_ok(hyle_fs_make_directory_recursive(path)));
  {
    hyle_file_info di = {0};
    CHECK(status_ok(hyle_fs_stat(&di, path)));
    CHECK(di.exists && di.is_directory);
  }
  visit_count = 0;
  int completed = 0;
  CHECK(status_ok(
            hyle_fs_directory_iterate(&completed, path, visit_cb, NULL)) &&
        completed);
  char file[700];
  snprintf(file, sizeof(file), "%s/f.txt", path);
  CHECK(status_ok(hyle_fs_touch(file)));
  CHECK(status_ok(hyle_fs_exists(&path_value, file)) && path_value);
  {
    uint64_t size = 1;
    CHECK(status_ok(hyle_fs_file_size(&size, file)) && size == 0u);
  }
  {
    uint32_t mode = 0;
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
    CHECK(hyle_fs_current_directory(cwd, sizeof(cwd)) > 0u);
    CHECK(hyle_fs_change_directory(path));
    CHECK(hyle_fs_change_directory(cwd));
  }
  {
    char link[720];
    snprintf(link, sizeof(link), "%s/f-link", path);
    /* Symlink creation may require a host privilege (notably on Windows). */
    if (status_ok(hyle_fs_create_symlink(file, link))) {
      char target[1024];
      size_t need = 0;
      CHECK(status_ok(hyle_fs_read_symlink(&need, link, NULL, 0)));
      CHECK(need > 0u);
      size_t got = 0;
      CHECK(status_ok(
                hyle_fs_read_symlink(&got, link, target, sizeof(target))) &&
            got == need);
      CHECK(strcmp(target, file) == 0);
      CHECK(status_ok(hyle_fs_is_symlink(&path_value, link)) && path_value);
    }
  }
  visit_count = 0;
  CHECK(status_ok(
            hyle_fs_directory_iterate(&completed, path, visit_cb, NULL)) &&
        completed);
  CHECK(visit_count >= 1);
  CHECK(status_ok(hyle_fs_remove_directory_recursive(path)));

  char envbuf[64];
  size_t en = hyle_env_get("PATH", envbuf, sizeof(envbuf));
  CHECK(en != HYLE_ENV_NOT_SET);

  hyle_cpu_features cpu = hyle_cpu_features_get();
  CHECK(cpu.logical_cpus >= 1u);
  CHECK(hyle_cpu_isa_name(cpu.best_isa) != NULL);
  CHECK(hyle_cpu_supports_isa(HYLE_ISA_SCALAR) == 1);
  {
    unsigned cl = hyle_cpu_cache_line_size();
    CHECK(cl >= 16u && cl <= 256u && (cl & (cl - 1u)) == 0u); /* sane power of two */
  }
  {
    hyle_cache_info cache = hyle_cpu_cache_info();
    hyle_cpu_topology topo = hyle_cpu_topology_get();
    if (cache.l1d && cache.l2) CHECK(cache.l1d <= cache.l2);
    if (cache.l1i && cache.l2) CHECK(cache.l1i <= cache.l2);
    if (cache.l2 && cache.l3) CHECK(cache.l2 <= cache.l3);
    CHECK(topo.logical_cpus >= 1u);
    CHECK(topo.performance_cpus + topo.efficiency_cpus ==
          topo.logical_cpus);
    CHECK((topo.is_hybrid != 0) ==
          (topo.performance_cpus != 0 && topo.efficiency_cpus != 0));
  }
  {
    hyle_isa isa = HYLE_ISA_SCALAR;      /* named enums usable as types */
    hyle_numa_node nany = HYLE_NUMA_ANY;
    CHECK(hyle_cpu_supports_isa((uint32_t)isa) == 1);
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
    CHECK(hyle_numa_current_node() < nc);
    CHECK(hyle_numa_node_of_cpu(0) < nc);         /* CPU 0 -> a valid node */
    CHECK(hyle_numa_node_of_cpu(1u << 20) < nc);  /* out-of-range CPU: 0 fallback */

    size_t pg = hyle_page_size();
    /* ANY == plain reserve; region is writable after commit. */
    void* rn = NULL;
    hyle_numa_placement placement = HYLE_NUMA_PLACEMENT_APPLIED;
    CHECK(status_ok(hyle_vm_reserve_numa(
        &rn, &placement, pg * 2, HYLE_NUMA_ANY)));
    CHECK(rn != NULL);
    CHECK(placement == HYLE_NUMA_PLACEMENT_NOT_NEEDED);
    CHECK(status_ok(hyle_vm_commit(rn, pg * 2)));
    ((unsigned char*)rn)[0] = 0x5A;
    CHECK(((unsigned char*)rn)[0] == 0x5A);
    /* Empty range bind is always a no-op success. */
    CHECK(status_ok(hyle_vm_bind_numa(
        &placement, NULL, 0, 0, HYLE_NUMA_PREFERRED)));
    CHECK(placement == HYLE_NUMA_PLACEMENT_NOT_NEEDED);
    CHECK(status_code(hyle_vm_bind_numa(
              NULL, NULL, 0, 0, (hyle_numa_policy)99)) ==
          HYLE_STATUS_INVALID_ARGUMENT);
    /* Preferred to a valid node: soft, always OK. */
    CHECK(status_ok(hyle_vm_bind_numa(
        &placement, rn, pg * 2, 0, HYLE_NUMA_PREFERRED)));
    /* Single-node box: hard bind / interleave degrade to no-op success. */
    if (nc <= 1u) {
      CHECK(status_ok(hyle_vm_bind_numa(
          &placement, rn, pg * 2, 0, HYLE_NUMA_BIND)));
      CHECK(status_ok(hyle_vm_bind_numa_interleave(
          &placement, rn, pg * 2)));
    }
    CHECK(status_ok(hyle_vm_release(rn, pg * 2)));

    /* reserve preferring the current home node. */
    void* rl = NULL;
    CHECK(status_ok(hyle_vm_reserve_numa(
        &rl, &placement, pg, (int)hyle_numa_current_node())));
    CHECK(rl != NULL);
    CHECK(status_ok(hyle_vm_release(rl, pg)));
  }

  /* v3: hugepage / THP hints (soft; size may be 0 on exotic arches) */
  {
    size_t lp = hyle_large_page_size();
    CHECK(hyle_large_page_align_up(0) == 0u);
    if (lp != 0u) {
      CHECK(hyle_large_page_align_up(1) == lp);
    }
    /* advise is best-effort: call on a committed region, don't assert result. */
    size_t pg = hyle_page_size();
    void* h = NULL;
    CHECK(status_ok(hyle_vm_alloc_rw(&h, pg)));
    CHECK(h != NULL);
    status_discard(hyle_vm_advise_hugepage(h, pg));
    status_discard(hyle_vm_advise_no_hugepage(h, pg));
    CHECK(status_ok(hyle_vm_free_rw(h, pg)));
  }

  /* v4: fs permissions + env write + temp file + mapped file */
  {
    /* env round trip */
    CHECK(hyle_fs_set_env_var("HYLE_C_PLAT_V4", "yes"));
    char eb[16];
    CHECK(hyle_env_get("HYLE_C_PLAT_V4", eb, sizeof(eb)) != HYLE_ENV_NOT_SET);
    CHECK(strcmp(eb, "yes") == 0);
    CHECK(hyle_fs_unset_env_var("HYLE_C_PLAT_V4"));
    CHECK(hyle_env_get("HYLE_C_PLAT_V4", eb, sizeof(eb)) == HYLE_ENV_NOT_SET);

    /* temp file: open handle + real path */
    char tpath[1024];
    hyle_file_handle fd = HYLE_FILE_HANDLE_INVALID;
    CHECK(status_code(hyle_fs_create_temp_file(
              NULL, NULL, "hyle_c", tpath, sizeof(tpath))) ==
          HYLE_STATUS_INVALID_ARGUMENT);
    CHECK(status_ok(hyle_fs_create_temp_file(
        &fd, NULL, "hyle_c", tpath, sizeof(tpath))));
    CHECK(fd != HYLE_FILE_HANDLE_INVALID);
    CHECK(status_ok(hyle_fs_exists(&path_value, tpath)) && path_value);
    /* permissions: the temp file exists -> nonzero perms; readonly round trip */
    uint32_t temp_mode = 0;
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
        &wfd, tpath,
        HYLE_OPEN_WRITE | HYLE_OPEN_CREATE | HYLE_OPEN_TRUNCATE)));
    CHECK(wfd != HYLE_FILE_HANDLE_INVALID);
    const char* payload = "c-abi file io";
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
    CHECK(status_code(hyle_fs_open(
              &rfd, "/no/such/hyle/path", HYLE_OPEN_READ)) ==
          HYLE_STATUS_NOT_FOUND);
    CHECK(rfd == HYLE_FILE_HANDLE_INVALID);
    CHECK(status_code(hyle_fs_open(&rfd, tpath, HYLE_OPEN_CREATE)) ==
          HYLE_STATUS_INVALID_ARGUMENT);
    CHECK(rfd == HYLE_FILE_HANDLE_INVALID);
    CHECK(hyle_fs_temp_directory(NULL, 32) > 0u); /* NULL = size query */
    CHECK(hyle_platform_last_error()[0] == '\0');
    CHECK(status_code(hyle_fs_read(
              &moved, HYLE_FILE_HANDLE_INVALID, NULL, 1)) ==
          HYLE_STATUS_INVALID_ARGUMENT);
    CHECK(status_ok(hyle_fs_close(HYLE_FILE_HANDLE_INVALID)));
    {
      hyle_file_open_flags typed_flags = HYLE_OPEN_READ;
      hyle_file_perms typed_perms = HYLE_PERM_OWNER_READ;
      CHECK((unsigned)typed_flags == HYLE_OPEN_READ);
      CHECK((unsigned)typed_perms == HYLE_PERM_OWNER_READ);
    }

    /* mapped file: tpath contains the payload written above; exercise the
     * Status-returning open plus invalid-mode failure. */
    hyle_mapped_file_o* m = NULL;
    hyle_status map_status = hyle_mapped_file_open(
        &m, tpath, HYLE_MAP_READ_ONLY, 0, 0);
    CHECK(hyle_status_is_ok(map_status));
    CHECK(m != NULL);
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

    /* hard link: a second name for the same file (same volume). */
    char hpath[1040];
    snprintf(hpath, sizeof(hpath), "%s.hln", tpath);
    if (status_ok(hyle_fs_create_hard_link(tpath, hpath))) {
      CHECK(status_ok(hyle_fs_exists(&path_value, hpath)) && path_value);
      CHECK(status_ok(hyle_fs_is_symlink(&path_value, hpath)) &&
            !path_value);
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
    CHECK(status_ok(hyle_fs_create_temp_file(
        &fd, NULL, "hyle_c_map", mpath, sizeof(mpath))));
    CHECK(fd != HYLE_FILE_HANDLE_INVALID);
    CHECK(status_ok(hyle_fs_close(fd)));

    const char* seed = "hyle mapped C-ABI payload 0123456789";
    size_t seed_len = strlen(seed);
    FILE* sf = fopen(mpath, "wb");
    CHECK(sf != NULL);
    if (sf != NULL) {
      CHECK(fwrite(seed, 1, seed_len, sf) == seed_len);
      fclose(sf);

      hyle_mapped_file_o* m = NULL;
      hyle_status map_status = hyle_mapped_file_open(
          &m, mpath, HYLE_MAP_READ_WRITE, 0, 0);
      CHECK(hyle_status_is_ok(map_status));
      hyle_status_free(&map_status);
      CHECK(m != NULL);
      CHECK(hyle_mapped_file_valid(m) == 1);
      CHECK(hyle_mapped_file_size(m) == seed_len);
      unsigned char* d = (unsigned char*)hyle_mapped_file_data(m);
      CHECK(d != NULL);
      if (d != NULL) {
        CHECK(memcmp(d, seed, seed_len) == 0);
        d[0] = (unsigned char)'Z';             /* dirty the first byte */
        map_status = hyle_mapped_file_flush(m);
        CHECK(hyle_status_is_ok(map_status)); /* RW flush persists to file */
        hyle_status_free(&map_status);
      }
      hyle_mapped_file_destroy(m);

      /* re-open read-only: the flushed write must be visible on disk. */
      hyle_mapped_file_o* m2 = NULL;
      map_status = hyle_mapped_file_open(
          &m2, mpath, HYLE_MAP_READ_ONLY, 0, 0);
      CHECK(hyle_status_is_ok(map_status));
      hyle_status_free(&map_status);
      CHECK(m2 != NULL);
      CHECK(hyle_mapped_file_valid(m2) == 1);
      if (hyle_mapped_file_valid(m2)) {
        CHECK(((unsigned char*)hyle_mapped_file_data(m2))[0] == (unsigned char)'Z');
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
