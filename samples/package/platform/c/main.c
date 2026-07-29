#include <hyle/c/platform.h>

#include <stddef.h>

static int status_ok(hyle_status status) {
  const int ok = hyle_status_is_ok(status);
  hyle_status_free(&status);
  return ok;
}

static void thread_entry(void* context) {
  int* value = (int*)context;
  *value = 42;
}

int main(void) {
  const size_t page = hyle_page_size();
  if (page == 0 || hyle_page_align_up(page + 1) != page * 2) return 1;

  size_t cpu_count = 0;
  if (!status_ok(hyle_cpu_available_count(&cpu_count)) || cpu_count == 0) {
    return 2;
  }

  hyle_parker_o* parker = NULL;
  if (!status_ok(hyle_parker_create(&parker)) || parker == NULL) return 3;
  hyle_parker_unpark(parker);
  if (!hyle_parker_park_timeout(parker, 0)) {
    hyle_parker_destroy(parker);
    return 4;
  }
  hyle_parker_destroy(parker);

  void* memory = NULL;
  if (!status_ok(hyle_vm_alloc_rw(&memory, page)) || memory == NULL) return 5;
  ((unsigned char*)memory)[0] = 0x5a;
  if (!status_ok(hyle_vm_free_rw(memory, page))) return 6;

  int ran = 0;
  hyle_thread_o* thread = NULL;
  if (!status_ok(hyle_thread_spawn(&thread, thread_entry, &ran, NULL)) ||
      thread == NULL) {
    return 7;
  }
  hyle_thread_join(thread);
  if (ran != 42) return 8;

  hyle_dynlib_o* current = NULL;
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
