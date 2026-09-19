// Failure injection is local to this executable's exact patched context TU.
#define _CRT_SECURE_NO_DEPRECATE
#define _USE_MATH_DEFINES
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
static int fail_context, fail_backing, live_contexts, context_calls;
static void *fault_malloc(size_t size) {
  ++context_calls;
  if (fail_context)
    return NULL;
  void *p = malloc(size);
  if (p)
    ++live_contexts;
  return p;
}
static void fault_free(void *p) {
  if (p)
    --live_contexts;
  free(p);
}
#if defined(_WIN32)
#include <malloc.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
static void *fault_backing(size_t size, size_t alignment) {
  return fail_backing ? NULL : _aligned_malloc(size, alignment);
}
#define _aligned_malloc fault_backing
#elif defined(__APPLE__)
#include <mach/mach.h>
static kern_return_t fault_backing(vm_map_t task, vm_address_t *address,
                                   vm_size_t size, int flags) {
  return fail_backing ? KERN_NO_SPACE : vm_allocate(task, address, size, flags);
}
#define vm_allocate fault_backing
#else
static int fault_backing(void **p, size_t alignment, size_t size) {
  return fail_backing ? ENOMEM : posix_memalign(p, alignment, size);
}
#define posix_memalign fault_backing
#endif
#define malloc fault_malloc
#define free fault_free
#include "ggml.c"
#undef malloc
#undef free

#define CHECK(c)                                                               \
  do {                                                                         \
    if (!(c)) {                                                                \
      fprintf(stderr, "context check failed at %d: %s\n", __LINE__, #c);       \
      return 1;                                                                \
    }                                                                          \
  } while (0)
int main(void) {
#if defined(_WIN32)
  SetErrorMode(0x8003);
#endif
  void *buffer = ggml_aligned_malloc(4096);
  CHECK(buffer != NULL);
  struct ggml_init_params external = {4096, buffer, true};
  fail_context = 1;
  CHECK(ggml_init(external) == NULL && live_contexts == 0);
  fail_context = 0;
  fail_backing = 1;
  CHECK(ggml_init((struct ggml_init_params){4096, NULL, false}) == NULL);
  CHECK(live_contexts == 0);
  // Borrowed backing remains usable even when owned backing is exhausted.
  struct ggml_context *ctx = ggml_init(external);
  CHECK(ctx != NULL && live_contexts == 1);
  ggml_free(ctx);
  CHECK(live_contexts == 0);
  fail_backing = 0;
  ctx = ggml_init((struct ggml_init_params){4096, NULL, false});
  CHECK(ctx != NULL && live_contexts == 1);
  ggml_free(ctx);
  CHECK(live_contexts == 0);
  ggml_cpu_init();
  const int cold_calls = context_calls;
  for (int i = 0; i < 10; ++i)
    ggml_cpu_init();
  CHECK(context_calls == cold_calls);
  ggml_aligned_free(buffer, 4096);
  puts(
      "ggml context allocation failures and one-time CPU initialization: PASS");
  return 0;
}
