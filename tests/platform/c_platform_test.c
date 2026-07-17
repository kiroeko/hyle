/* Pure C consumer of include/hyle/c/platform.h */
#if defined(_WIN32)
#  define _CRT_SECURE_NO_WARNINGS 1
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

int main(void) {
  CHECK(hyle_platform_c_abi_version() == HYLE_PLATFORM_C_ABI_VERSION);
  CHECK(HYLE_PLATFORM_C_ABI_VERSION == 3);

  hyle_platform_api api;
  CHECK(hyle_platform_api_get(&api, HYLE_PLATFORM_C_ABI_VERSION) == 0);
  CHECK(api.abi_version == HYLE_PLATFORM_C_ABI_VERSION);
  CHECK(api.page_size() >= 4096u);
  CHECK(api.thread_joinable != NULL);
  CHECK(api.dynlib_open_symbol != NULL);
  CHECK(api.vm_alloc_rw != NULL);
  CHECK(api.cpu_features_get != NULL);
  CHECK(api.c_abi_version != NULL);
  CHECK(api.c_abi_version() == HYLE_PLATFORM_C_ABI_VERSION);
  /* v3 appended fields present */
  CHECK(api.vm_set_last_error != NULL);
  CHECK(api.large_page_size != NULL);
  CHECK(api.vm_advise_hugepage != NULL);
  CHECK(api.vm_advise_no_hugepage != NULL);
  CHECK(api.numa_node_count != NULL);
  CHECK(api.numa_current_node != NULL);
  CHECK(api.vm_bind_numa != NULL);
  CHECK(api.vm_bind_numa_interleave != NULL);
  CHECK(api.vm_reserve_numa != NULL);
  CHECK(hyle_platform_api_get(NULL, HYLE_PLATFORM_C_ABI_VERSION) == -1);
  CHECK(hyle_platform_api_get(&api, 999) == -1);
  CHECK(hyle_platform_api_get(&api, 2) == -1); /* old ABI rejected */

  /* Parker */
  hyle_parker_o* p = hyle_parker_create();
  CHECK(p != NULL);
  hyle_parker_unpark(p);
  hyle_parker_park(p);
  CHECK(hyle_parker_park_timeout(p, 0) == 0);
  hyle_parker_destroy(p);

  /* Thread + opts sugar */
  int ran = 0;
  hyle_thread_o* t = NULL;
  hyle_thread_opts opts = HYLE_THREAD_OPTS_INIT;
  hyle_thread_opts_set_name(&opts, "c_plat_worker");
  hyle_status st = hyle_thread_spawn(&t, thr_entry, &ran, &opts);
  CHECK(hyle_status_is_ok(st));
  hyle_status_free(&st);
  CHECK(t != NULL);
  CHECK(hyle_thread_joinable(t));
  hyle_thread_join(t);
  CHECK(ran == 42);
  CHECK(hyle_hardware_concurrency() >= 1u);
  CHECK(hyle_this_thread_id() != 0);

  /* DynLib */
#if defined(_WIN32)
  const char* sys_lib = "kernel32.dll";
  const char* sys_sym = "GetCurrentProcessId";
  const char* ext = "dll";
#elif defined(__APPLE__)
  const char* sys_lib = "libSystem.dylib";
  const char* sys_sym = "getpid";
  const char* ext = "dylib";
#else
  const char* sys_lib = "libm.so.6";
  const char* sys_sym = "cos";
  const char* ext = "so";
#endif
  hyle_dynlib_o* lib = hyle_dynlib_get(sys_lib);
  CHECK(hyle_dynlib_valid(lib));
  CHECK(hyle_dynlib_symbol(lib, sys_sym) != NULL);
  hyle_dynlib_close(lib);
  CHECK(strcmp(hyle_dynlib_extension(), ext) == 0);

  /* open_symbol on already-loadable system lib path may fail if path needs
   * absolute form; exercise API shape via get+symbol path above. Missing path: */
  {
    hyle_dynlib_o* miss = NULL;
    void* miss_sym = NULL;
    CHECK(hyle_dynlib_open_symbol("hyle_no_such_plugin_xyz.so", "entry", &miss,
                                 &miss_sym) == 0);
    CHECK(miss == NULL);
    CHECK(miss_sym == NULL);
  }

  /* VM guarded */
  hyle_vm_region_o* reg = hyle_vm_region_reserve_guarded(hyle_page_size());
  CHECK(hyle_vm_region_valid(reg));
  CHECK(hyle_vm_region_has_guard(reg));
  {
    unsigned char* u = (unsigned char*)hyle_vm_region_usable(reg);
    CHECK(u != NULL);
    u[0] = 0xAB;
    CHECK(u[0] == 0xAB);
  }
  hyle_vm_region_destroy(reg);

  /* free vm + alloc_rw */
  {
    size_t pg = hyle_page_size();
    void* base = hyle_vm_reserve(pg);
    CHECK(base != NULL);
    CHECK(hyle_vm_commit(base, pg));
    CHECK(hyle_vm_release(base, pg));

    void* rw = hyle_vm_alloc_rw(pg / 2 + 1);
    CHECK(rw != NULL);
    ((unsigned char*)rw)[0] = 1;
    CHECK(hyle_vm_free_rw(rw, pg / 2 + 1));
  }

  /* fs */
  char tmp[512];
  size_t n = hyle_fs_temp_directory(tmp, sizeof(tmp));
  CHECK(n > 0 && n < sizeof(tmp));
  CHECK(hyle_fs_exists(tmp));
  char path[640];
  snprintf(path, sizeof(path), "%s/hyle_c_plat_test_dir", tmp);
  CHECK(hyle_fs_make_directory_recursive(path));
  visit_count = 0;
  CHECK(hyle_fs_directory_iterate(path, visit_cb, NULL));
  char file[700];
  snprintf(file, sizeof(file), "%s/f.txt", path);
  CHECK(hyle_fs_touch(file));
  CHECK(hyle_fs_exists(file));
  visit_count = 0;
  CHECK(hyle_fs_directory_iterate(path, visit_cb, NULL));
  CHECK(visit_count >= 1);
  CHECK(hyle_fs_remove_directory_recursive(path));

  char envbuf[64];
  size_t en = hyle_env_get("PATH", envbuf, sizeof(envbuf));
  CHECK(en != HYLE_ENV_NOT_SET);

  hyle_cpu_features cpu = hyle_cpu_features_get();
  CHECK(cpu.logical_cpus >= 1u);
  CHECK(hyle_cpu_isa_name(cpu.best_isa) != NULL);
  CHECK(hyle_cpu_supports_isa(HYLE_ISA_SCALAR) == 1);
#if defined(_WIN32) || defined(__x86_64__) || defined(_M_X64)
  CHECK(cpu.sse2 == 1);
  CHECK(hyle_cpu_supports_isa(HYLE_ISA_SSE2) == 1);
#endif

  /* v3: NUMA topology + placement */
  {
    unsigned nc = hyle_numa_node_count();
    CHECK(nc >= 1u);
    CHECK(hyle_numa_current_node() < nc);

    /* set/clear last-error round trip */
    hyle_vm_set_last_error("c_plat_probe");
    CHECK(strcmp(hyle_vm_last_error(), "c_plat_probe") == 0);
    hyle_vm_set_last_error(NULL);
    CHECK(hyle_vm_last_error()[0] == '\0');

    size_t pg = hyle_page_size();
    /* ANY == plain reserve; region is writable after commit. */
    void* rn = hyle_vm_reserve_numa(pg * 2, HYLE_NUMA_ANY);
    CHECK(rn != NULL);
    CHECK(hyle_vm_commit(rn, pg * 2));
    ((unsigned char*)rn)[0] = 0x5A;
    CHECK(((unsigned char*)rn)[0] == 0x5A);
    /* Empty range bind is always a no-op success. */
    CHECK(hyle_vm_bind_numa(NULL, 0, 0, HYLE_NUMA_PREFERRED) == 1);
    /* Preferred to a valid node: soft, always OK. */
    CHECK(hyle_vm_bind_numa(rn, pg * 2, 0, HYLE_NUMA_PREFERRED) == 1);
    /* Single-node box: hard bind / interleave degrade to no-op success. */
    if (nc <= 1u) {
      CHECK(hyle_vm_bind_numa(rn, pg * 2, 0, HYLE_NUMA_BIND) == 1);
      CHECK(hyle_vm_bind_numa_interleave(rn, pg * 2) == 1);
    }
    CHECK(hyle_vm_release(rn, pg * 2));

    /* reserve preferring the current home node. */
    void* rl = hyle_vm_reserve_numa(pg, (int)hyle_numa_current_node());
    CHECK(rl != NULL);
    CHECK(hyle_vm_release(rl, pg));
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
    void* h = hyle_vm_alloc_rw(pg);
    CHECK(h != NULL);
    (void)hyle_vm_advise_hugepage(h, pg);
    (void)hyle_vm_advise_no_hugepage(h, pg);
    CHECK(hyle_vm_free_rw(h, pg));
  }

  if (g_fails) {
    fprintf(stderr, "c_platform_test: %d failed\n", g_fails);
    return 1;
  }
  printf("c_platform_test: all checks passed\n");
  return 0;
}
