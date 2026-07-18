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
  CHECK(hyle_page_size() >= 4096u);

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

  CHECK(hyle_hardware_concurrency() >= 1u);
  CHECK(hyle_this_thread_id() != 0);

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

  /* v5: region reserve_numa (ANY == plain) + adopt an existing raw reservation */
  {
    size_t pg = hyle_page_size();
    hyle_vm_region_o* rn = hyle_vm_region_reserve_numa(pg * 2, HYLE_NUMA_ANY);
    CHECK(hyle_vm_region_valid(rn));
    CHECK(hyle_vm_region_commit(rn, 0, pg * 2));
    ((unsigned char*)hyle_vm_region_data(rn))[0] = 0x5A;
    CHECK(((unsigned char*)hyle_vm_region_data(rn))[0] == 0x5A);
    hyle_vm_region_destroy(rn);

    /* adopt: raw reserve -> hand ownership to a region handle (destroy releases) */
    void* raw = hyle_vm_reserve(pg);
    CHECK(raw != NULL);
    hyle_vm_region_o* ad = hyle_vm_region_adopt(raw, pg);
    CHECK(hyle_vm_region_valid(ad));
    CHECK(hyle_vm_region_data(ad) == raw);
    CHECK(hyle_vm_region_size(ad) == pg);
    hyle_vm_region_destroy(ad);  /* releases the adopted reservation */
    /* adopt(NULL) -> NULL (nothing adopted) */
    CHECK(hyle_vm_region_adopt(NULL, pg) == NULL);
  }

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
    intptr_t fd = hyle_fs_create_temp_file(NULL, "hyle_c", tpath, sizeof(tpath));
    CHECK(fd != -1);
    CHECK(hyle_fs_exists(tpath));
    /* permissions: the temp file exists -> nonzero perms; readonly round trip */
    CHECK(hyle_fs_get_permissions(tpath) != 0u);
    CHECK(hyle_fs_set_readonly(tpath, 1) == 1);
    CHECK(hyle_fs_is_readonly(tpath) == 1);
    CHECK(hyle_fs_set_readonly(tpath, 0) == 1);
    hyle_fs_close_fd(fd);  /* void; closes the open temp handle */

    /* mapped file: an empty temp file yields a NULL (zero-length) mapping -- fine;
     * exercise open + destroy-null here. A non-empty map is covered by the C++ test. */
    hyle_mapped_file_o* m = hyle_mapped_file_open(tpath, HYLE_MAP_READ_ONLY, 0, 0);
    hyle_mapped_file_destroy(m);

    /* hard link: a second name for the same file (same volume). */
    char hpath[1040];
    snprintf(hpath, sizeof(hpath), "%s.hln", tpath);
    if (hyle_fs_create_hard_link(tpath, hpath)) {
      CHECK(hyle_fs_exists(hpath));
      CHECK(hyle_fs_is_symlink(hpath) == 0);  /* hard link, not a symlink */
      hyle_fs_remove_file(hpath);
    }
    hyle_fs_remove_file(tpath);
  }

  if (g_fails) {
    fprintf(stderr, "c_platform_test: %d failed\n", g_fails);
    return 1;
  }
  printf("c_platform_test: all checks passed\n");
  return 0;
}
