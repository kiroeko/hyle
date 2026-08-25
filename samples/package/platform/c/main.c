#include <hyle/c/error_helpers.h>
#include <hyle/c/platform.h>

#include <stddef.h>

#if defined(_MSC_VER)
#define HYLE_PACKAGE_ASSERT_JOIN_INNER(left, right) left##right
#define HYLE_PACKAGE_ASSERT_JOIN(left, right)                              \
  HYLE_PACKAGE_ASSERT_JOIN_INNER(left, right)
#define HYLE_PACKAGE_STATIC_ASSERT(condition, message)                     \
  typedef char HYLE_PACKAGE_ASSERT_JOIN(hyle_package_static_assert_,       \
                                        __LINE__)[(condition) ? 1 : -1]
#else
#define HYLE_PACKAGE_STATIC_ASSERT(condition, message)                     \
  _Static_assert(condition, message)
#endif

HYLE_PACKAGE_STATIC_ASSERT(sizeof(hyle_thread_priority) == sizeof(int32_t),
               "installed thread-priority ABI width");
HYLE_PACKAGE_STATIC_ASSERT(sizeof(hyle_thread_config_policy) == sizeof(uint32_t),
               "installed thread-policy ABI width");
HYLE_PACKAGE_STATIC_ASSERT(sizeof(hyle_protect) == sizeof(uint32_t),
               "installed protect ABI width");
HYLE_PACKAGE_STATIC_ASSERT(sizeof(hyle_numa_node) == sizeof(int32_t),
               "installed NUMA-node ABI width");
HYLE_PACKAGE_STATIC_ASSERT(sizeof(hyle_numa_policy) == sizeof(uint32_t),
               "installed NUMA-policy ABI width");
HYLE_PACKAGE_STATIC_ASSERT(sizeof(hyle_numa_placement) == sizeof(uint32_t),
               "installed NUMA-placement ABI width");
HYLE_PACKAGE_STATIC_ASSERT(sizeof(hyle_numa_capability) == sizeof(uint32_t),
               "installed NUMA-capability ABI width");
HYLE_PACKAGE_STATIC_ASSERT(sizeof(hyle_file_type) == sizeof(uint32_t),
               "installed file-type ABI width");
HYLE_PACKAGE_STATIC_ASSERT(sizeof(hyle_file_perms) == sizeof(uint32_t),
               "installed file-permissions ABI width");
HYLE_PACKAGE_STATIC_ASSERT(sizeof(hyle_symlink_kind) == sizeof(uint32_t),
               "installed symlink-kind ABI width");
HYLE_PACKAGE_STATIC_ASSERT(sizeof(hyle_file_open_flags) == sizeof(uint32_t),
               "installed file-open-flags ABI width");
HYLE_PACKAGE_STATIC_ASSERT(sizeof(hyle_map_mode) == sizeof(uint32_t),
               "installed map-mode ABI width");
HYLE_PACKAGE_STATIC_ASSERT(sizeof(hyle_map_advice) == sizeof(uint32_t),
               "installed map-advice ABI width");
HYLE_PACKAGE_STATIC_ASSERT(sizeof(hyle_isa) == sizeof(uint32_t),
                           "installed ISA ABI width");
HYLE_PACKAGE_STATIC_ASSERT(sizeof(hyle_cpu_kind) == sizeof(uint32_t),
               "installed CPU-kind ABI width");
HYLE_PACKAGE_STATIC_ASSERT(sizeof(hyle_cpu_order) == sizeof(uint32_t),
               "installed CPU-order ABI width");
HYLE_PACKAGE_STATIC_ASSERT(offsetof(hyle_thread_opts, reserved) == 4,
               "installed thread opts reserved ABI offset");
HYLE_PACKAGE_STATIC_ASSERT(offsetof(hyle_thread_opts, name) == 8,
               "installed thread opts first optional field offset");
HYLE_PACKAGE_STATIC_ASSERT(offsetof(hyle_thread_opts, has_priority) == 40,
               "installed thread opts has-priority ABI offset");
HYLE_PACKAGE_STATIC_ASSERT(offsetof(hyle_thread_opts, priority) == 44,
               "installed thread opts priority ABI offset");
HYLE_PACKAGE_STATIC_ASSERT(offsetof(hyle_thread_opts, config_policy) == 48,
               "installed thread opts policy ABI offset");
HYLE_PACKAGE_STATIC_ASSERT(offsetof(hyle_thread_opts, affinity_cpus) == 56,
               "installed thread opts affinity-array ABI offset");
HYLE_PACKAGE_STATIC_ASSERT(offsetof(hyle_thread_opts, affinity_cpu_count) == 64,
               "installed thread opts affinity-count ABI offset");
HYLE_PACKAGE_STATIC_ASSERT(HYLE_THREAD_OPTS_REQUIRED_SIZE == 8,
               "installed thread opts mandatory prefix");
HYLE_PACKAGE_STATIC_ASSERT(sizeof(hyle_thread_opts) == 72,
               "installed thread opts 64-bit ABI size");
HYLE_PACKAGE_STATIC_ASSERT(offsetof(hyle_cpu_info, kind) == 8,
               "installed CPU info kind ABI offset");
HYLE_PACKAGE_STATIC_ASSERT(sizeof(hyle_cpu_info) == 12,
                           "installed CPU info ABI size");

#undef HYLE_PACKAGE_STATIC_ASSERT
#if defined(_MSC_VER)
#undef HYLE_PACKAGE_ASSERT_JOIN
#undef HYLE_PACKAGE_ASSERT_JOIN_INNER
#endif

static int status_ok(hyle_status status) {
  const int ok = hyle_status_is_ok(status);
  hyle_status_free(&status);
  return ok;
}

static void thread_entry(void *context) {
  int *value = (int *)context;
  *value = 42;
}

int main(void) {
  const size_t page = hyle_page_size();
  if (page == 0 || hyle_page_align_up(page + 1) != page * 2)
    return 1;

  size_t cpu_count = 0;
  if (!status_ok(hyle_cpu_available_count(&cpu_count)) || cpu_count == 0) {
    return 2;
  }

  hyle_parker_o *parker = NULL;
  if (!status_ok(hyle_parker_create(&parker)) || parker == NULL)
    return 3;
  hyle_parker_unpark(parker);
  if (!hyle_parker_park_timeout(parker, 0)) {
    hyle_parker_destroy(parker);
    return 4;
  }
  hyle_parker_destroy(parker);

  void *memory = NULL;
  if (!status_ok(hyle_vm_alloc_rw(&memory, page)) || memory == NULL)
    return 5;
  ((unsigned char *)memory)[0] = 0x5a;
  if (!status_ok(hyle_vm_free_rw(memory, page)))
    return 6;

  int ran = 0;
  hyle_thread_o *thread = NULL;
  if (!status_ok(hyle_thread_spawn(&thread, thread_entry, &ran, NULL)) ||
      thread == NULL) {
    return 7;
  }
  hyle_thread_join(thread);
  if (ran != 42)
    return 8;

  hyle_dynlib_o *current = NULL;
  if (!status_ok(hyle_dynlib_current(&current)) || current == NULL ||
      !hyle_dynlib_valid(current)) {
    hyle_dynlib_close(current);
    return 9;
  }
  hyle_dynlib_close(current);

  size_t cwd_size = 0;
  if (!status_ok(hyle_fs_current_directory(&cwd_size, NULL, 0)) ||
      cwd_size == 0) {
    return 10;
  }
  return 0;
}
