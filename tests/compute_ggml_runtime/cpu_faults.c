// Compile the exact patched CPU TU with local resource failures. All successful
// starts use real OS threads; production objects contain none of these hooks.
#define _CRT_SECURE_NO_DEPRECATE
#define _USE_MATH_DEFINES
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

static int allocation_call, fail_allocation, live_allocations, bad_free;
static void *allocations[2];
static size_t allocation_sizes[2];
static int starts, joins, fail_start;
static void *fault_aligned_malloc(size_t size) {
  if (++allocation_call == fail_allocation)
    return NULL;
  void *p = ggml_aligned_malloc(size);
  if (p != NULL) {
    int slot = allocations[0] == NULL ? 0 : 1;
    allocations[slot] = p;
    allocation_sizes[slot] = size;
    ++live_allocations;
  }
  return p;
}
static void fault_aligned_free(void *p, size_t size) {
  if (p != NULL) {
    int slot = allocations[0] == p ? 0 : 1;
    if (allocations[slot] != p || allocation_sizes[slot] != size)
      ++bad_free;
    allocations[slot] = NULL;
    --live_allocations;
  }
  ggml_aligned_free(p, size);
}

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
static HANDLE fault_create_thread(LPSECURITY_ATTRIBUTES attributes,
                                  SIZE_T stack, LPTHREAD_START_ROUTINE entry,
                                  LPVOID argument, DWORD flags, LPDWORD id) {
  if (++starts == fail_start) {
    SetLastError(ERROR_NOT_ENOUGH_MEMORY);
    return NULL;
  }
  return CreateThread(attributes, stack, entry, argument, flags, id);
}
static DWORD fault_wait(HANDLE thread, DWORD timeout) {
  ++joins;
  return WaitForSingleObject(thread, timeout);
}
#define CreateThread fault_create_thread
#define WaitForSingleObject fault_wait
#define EXPECTED_THREAD_ERROR ERROR_NOT_ENOUGH_MEMORY
#else
#include <pthread.h>
static int fail_mutex, fail_cond, live_mutex, live_cond;
static int fault_create(pthread_t *thread, const pthread_attr_t *attributes,
                        void *(*entry)(void *), void *argument) {
  if (++starts == fail_start)
    return EAGAIN;
  return pthread_create(thread, attributes, entry, argument);
}
static int fault_join(pthread_t thread, void **result) {
  ++joins;
  return pthread_join(thread, result);
}
static int fault_mutex_init(pthread_mutex_t *mutex,
                            const pthread_mutexattr_t *attributes) {
  if (fail_mutex)
    return ENOMEM;
  int rc = pthread_mutex_init(mutex, attributes);
  if (!rc)
    ++live_mutex;
  return rc;
}
static int fault_mutex_destroy(pthread_mutex_t *mutex) {
  --live_mutex;
  return pthread_mutex_destroy(mutex);
}
static int fault_cond_init(pthread_cond_t *cond,
                           const pthread_condattr_t *attributes) {
  if (fail_cond)
    return EAGAIN;
  int rc = pthread_cond_init(cond, attributes);
  if (!rc)
    ++live_cond;
  return rc;
}
static int fault_cond_destroy(pthread_cond_t *cond) {
  --live_cond;
  return pthread_cond_destroy(cond);
}
#define pthread_create fault_create
#define pthread_join fault_join
#define pthread_mutex_init fault_mutex_init
#define pthread_mutex_destroy fault_mutex_destroy
#define pthread_cond_init fault_cond_init
#define pthread_cond_destroy fault_cond_destroy
#define EXPECTED_THREAD_ERROR EAGAIN
#endif
#define ggml_aligned_malloc fault_aligned_malloc
#define ggml_aligned_free fault_aligned_free
#include "ggml-cpu.c"
#undef ggml_aligned_malloc
#undef ggml_aligned_free

#define CHECK(c)                                                               \
  do {                                                                         \
    if (!(c)) {                                                                \
      fprintf(stderr, "resource check failed at %d: %s\n", __LINE__, #c);      \
      return 1;                                                                \
    }                                                                          \
  } while (0)
int main(void) {
#if defined(_WIN32)
  SetErrorMode(0x8003);
#endif
  ggml_cpu_init();
  struct ggml_threadpool_params p = ggml_threadpool_params_default(4);
  p.poll = 0;
  p.prio = GGML_SCHED_PRIO_NORMAL;
  int error = 99;
  CHECK(ggml_hyle_threadpool_new(NULL, &error) == NULL && error != 0);
  p.n_threads = 0;
  CHECK(ggml_hyle_threadpool_new(&p, &error) == NULL && error != 0);
  p.n_threads = GGML_MAX_N_THREADS + 1;
  CHECK(ggml_hyle_threadpool_new(&p, &error) == NULL && error != 0);
  p.n_threads = 4;
  for (int failure = 1; failure <= 2; ++failure) {
    allocation_call = 0;
    fail_allocation = failure;
    CHECK(ggml_hyle_threadpool_new(&p, &error) == NULL);
    CHECK(error == 0 && live_allocations == 0 && bad_free == 0);
  }
  fail_allocation = 0;
#if !defined(_WIN32)
  fail_mutex = 1;
  CHECK(ggml_hyle_threadpool_new(&p, &error) == NULL && error == ENOMEM);
  CHECK(live_allocations == 0 && live_mutex == 0 && live_cond == 0);
  fail_mutex = 0;
  fail_cond = 1;
  CHECK(ggml_hyle_threadpool_new(&p, &error) == NULL && error == EAGAIN);
  CHECK(live_allocations == 0 && live_mutex == 0 && live_cond == 0);
  fail_cond = 0;
#endif
  for (int paused = 0; paused < 2; ++paused) {
    p.paused = paused != 0;
    for (int failure = 1; failure <= 3; ++failure) {
      starts = 0;
      joins = 0;
      fail_start = failure;
      CHECK(ggml_hyle_threadpool_new(&p, &error) == NULL);
      CHECK(error == EXPECTED_THREAD_ERROR && starts == failure &&
            joins == failure - 1);
      CHECK(live_allocations == 0 && bad_free == 0);
    }
  }
  fail_start = 0;
  starts = 0;
  joins = 0;
  p.paused = false;
  struct ggml_threadpool *pool = ggml_hyle_threadpool_new(&p, &error);
  CHECK(pool != NULL && error == 0 && live_allocations == 2);

  struct ggml_context *ctx =
      ggml_init((struct ggml_init_params){16384, NULL, false});
  CHECK(ctx != NULL);
  struct ggml_tensor *x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
  for (int i = 0; i < 8; ++i)
    ((float *)x->data)[i] = 1.0f;
  struct ggml_tensor *y = ggml_rms_norm(ctx, x, 1.0e-5f);
  struct ggml_cgraph *g = ggml_new_graph_custom(ctx, 8, false);
  ggml_build_forward_expand(g, y);
  struct ggml_cplan plan = ggml_graph_plan(g, 4, pool);
  plan.work_data = malloc(plan.work_size);
  CHECK(plan.work_size == 0 || plan.work_data != NULL);
  int cold_allocations = allocation_call;
  for (int i = 0; i < 10; ++i)
    CHECK(ggml_graph_compute(g, &plan) == GGML_STATUS_SUCCESS);
  CHECK(allocation_call == cold_allocations && starts == 3);
  CHECK(((float *)y->data)[0] > 0.999f);
  free(plan.work_data);
  // The native convenience path must also return, not abort, on pool OOM.
  plan = ggml_graph_plan(g, 1, NULL);
  plan.work_data = malloc(plan.work_size);
  CHECK(plan.work_size == 0 || plan.work_data != NULL);
  fail_allocation = allocation_call + 1;
  CHECK(ggml_graph_compute(g, &plan) == GGML_STATUS_ALLOC_FAILED);
  free(plan.work_data);
  ggml_free(ctx);
  ggml_threadpool_free(pool);
  CHECK(live_allocations == 0 && bad_free == 0 && joins == 3);
#if !defined(_WIN32)
  CHECK(live_mutex == 0 && live_cond == 0);
#endif
  puts("ggml CPU resource failures, partial starts, reuse and joins: PASS");
  return 0;
}
