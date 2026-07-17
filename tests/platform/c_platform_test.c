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

  hyle_platform_api api;
  CHECK(hyle_platform_api_get(&api, HYLE_PLATFORM_C_ABI_VERSION) == 0);
  CHECK(api.page_size() >= 4096u);

  /* Parker */
  hyle_parker_o* p = hyle_parker_create();
  CHECK(p != NULL);
  hyle_parker_unpark(p);
  hyle_parker_park(p);
  CHECK(hyle_parker_park_timeout(p, 0) == 0);
  hyle_parker_destroy(p);

  /* Thread */
  int ran = 0;
  hyle_thread_o* t = NULL;
  hyle_status st = hyle_thread_spawn(&t, thr_entry, &ran, NULL);
  CHECK(hyle_status_is_ok(st));
  hyle_status_free(&st);
  CHECK(t != NULL);
  CHECK(hyle_thread_joinable(t));
  hyle_thread_join(t);
  CHECK(ran == 42);
  CHECK(hyle_hardware_concurrency() >= 1u);

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

  /* free vm */
  {
    size_t pg = hyle_page_size();
    void* base = hyle_vm_reserve(pg);
    CHECK(base != NULL);
    CHECK(hyle_vm_commit(base, pg));
    CHECK(hyle_vm_release(base, pg));
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
#if defined(_WIN32) || defined(__x86_64__) || defined(_M_X64)
  CHECK(cpu.sse2 == 1);
#endif

  if (g_fails) {
    fprintf(stderr, "c_platform_test: %d failed\n", g_fails);
    return 1;
  }
  printf("c_platform_test: all checks passed\n");
  return 0;
}
