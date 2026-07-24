/* Pure C consumer of include/hyle/c/mem.h */
#if defined(_WIN32)
#define _CRT_SECURE_NO_WARNINGS 1
#endif
#include "hyle/c/mem.h"

#include <stdio.h>
#include <string.h>

static int g_fails = 0;
#define CHECK(c)                                                               \
  do {                                                                         \
    if (!(c)) {                                                                \
      fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #c, __FILE__, __LINE__);   \
      ++g_fails;                                                               \
    }                                                                          \
  } while (0)

#define CHECK_OK(expr)                                                         \
  do {                                                                         \
    hyle_status _status = (expr);                                              \
    CHECK(hyle_status_is_ok(_status));                                         \
    hyle_status_free(&_status);                                                \
  } while (0)

/* A backing allocator whose alloc/realloc always fail -- simulates slab OOM so
 * the graceful Status-returning create paths can be exercised. */
static void *oom_alloc(void *st, size_t sz, size_t al) {
  (void)st;
  (void)sz;
  (void)al;
  return NULL;
}
static void oom_free(void *st, void *p, size_t sz, size_t al) {
  (void)st;
  (void)p;
  (void)sz;
  (void)al;
}
static void *oom_realloc(void *st, void *p, size_t os, size_t ns, size_t al) {
  (void)st;
  (void)p;
  (void)os;
  (void)ns;
  (void)al;
  return NULL;
}

typedef struct counting_backing_state {
  hyle_allocator upstream;
  size_t live_allocations;
} counting_backing_state;

typedef struct failing_metadata_state {
  hyle_allocator upstream;
  size_t calls;
  size_t fail_at;
  size_t live_allocations;
} failing_metadata_state;

static void *failing_metadata_alloc(void *st, size_t size, size_t alignment) {
  failing_metadata_state *state = (failing_metadata_state *)st;
  size_t call = state->calls++;
  if (call == state->fail_at)
    return NULL;
  void *p = state->upstream.alloc(state->upstream.state, size, alignment);
  if (p != NULL)
    ++state->live_allocations;
  return p;
}

static void failing_metadata_free(void *st, void *p, size_t size,
                                  size_t alignment) {
  failing_metadata_state *state = (failing_metadata_state *)st;
  if (p != NULL)
    --state->live_allocations;
  state->upstream.free(state->upstream.state, p, size, alignment);
}

static void *failing_metadata_realloc(void *st, void *p, size_t old_size,
                                      size_t new_size, size_t alignment) {
  failing_metadata_state *state = (failing_metadata_state *)st;
  size_t call = state->calls++;
  if (call == state->fail_at)
    return NULL;
  void *out = state->upstream.realloc(state->upstream.state, p, old_size,
                                      new_size, alignment);
  if (p == NULL && out != NULL)
    ++state->live_allocations;
  else if (p != NULL && new_size == 0)
    --state->live_allocations;
  return out;
}

static void *counting_alloc(void *st, size_t size, size_t alignment) {
  counting_backing_state *state = (counting_backing_state *)st;
  void *p = state->upstream.alloc(state->upstream.state, size, alignment);
  if (p != NULL)
    ++state->live_allocations;
  return p;
}

static void counting_free(void *st, void *p, size_t size, size_t alignment) {
  counting_backing_state *state = (counting_backing_state *)st;
  if (p != NULL)
    --state->live_allocations;
  state->upstream.free(state->upstream.state, p, size, alignment);
}

static void *counting_realloc(void *st, void *p, size_t old_size,
                              size_t new_size, size_t alignment) {
  counting_backing_state *state = (counting_backing_state *)st;
  void *out = state->upstream.realloc(state->upstream.state, p, old_size,
                                      new_size, alignment);
  if (p == NULL && out != NULL)
    ++state->live_allocations;
  else if (p != NULL && new_size == 0)
    --state->live_allocations;
  return out;
}

int main(void) {
  /* The mechanical ABI audit counts only references below this marker, excluding
     declarations pulled from the header. Keep it as code so comment stripping
     cannot erase it. */
  (void)"HYLE_C_MEM_COVERAGE_BODY_MARKER";
  CHECK(hyle_default_alignment() >= sizeof(void *));

  /* ---- system allocator: alloc / write / realloc / free ---- */
  hyle_allocator sys = hyle_system_allocator();
  CHECK(hyle_allocator_is_valid(sys));
  CHECK(sys.alloc != NULL && sys.free != NULL && sys.realloc != NULL);
  {
    counting_backing_state state = {sys, 0};
    hyle_allocator counting = {counting_alloc, counting_free,
                               counting_realloc, &state};
    hyle_shared_pool *pool = NULL;
    CHECK_OK(hyle_shared_pool_create(&pool, counting, 64, 8, 0));
    CHECK(pool != NULL);
    CHECK(state.live_allocations >= 4u); /* control + slab + links + heads */
    hyle_shared_pool_destroy(pool);
    CHECK(state.live_allocations == 0u);
  }
  {
    hyle_allocator bad = {0};
    hyle_shared_pool *bad_pool = (hyle_shared_pool *)(size_t)1;
    hyle_status status = hyle_shared_pool_create(&bad_pool, bad, 16, 1, 0);
    CHECK(hyle_status_is_invalid_argument(status));
    CHECK(hyle_status_has_message(status));
    CHECK(bad_pool == NULL); /* every factory clears out before work */
    hyle_status_free(&status);
    hyle_shared_pool *good = NULL;
    CHECK_OK(hyle_shared_pool_create(&good, sys, 16, 1, 0));
    CHECK(good != NULL);
    hyle_shared_pool_destroy(good);

    /* C inputs are data, not C++ contracts: invalid layouts fail gracefully. */
    status = hyle_shared_pool_create(&bad_pool, sys, 0, 1, 0);
    CHECK(hyle_status_is_invalid_argument(status));
    hyle_status_free(&status);
    status = hyle_shared_pool_create(&bad_pool, sys, 16, 1, 3);
    CHECK(hyle_status_is_invalid_argument(status));
    hyle_status_free(&status);
    hyle_shared_pool_cache *bad_cache = NULL;
    status = hyle_shared_pool_cache_create(&bad_cache, NULL, sys, 8, 2);
    CHECK(hyle_status_is_invalid_argument(status));
    hyle_status_free(&status);
    hyle_stack_allocator *bad_stack = NULL;
    status = hyle_stack_allocator_create(&bad_stack, sys, 0);
    CHECK(hyle_status_is_invalid_argument(status));
    hyle_status_free(&status);
  }
  {
    /* direct vtable call */
    unsigned char *p = (unsigned char *)sys.alloc(sys.state, 128, 64);
    CHECK(p != NULL);
    CHECK(((size_t)p % 64u) == 0u); /* honored explicit alignment */
    memset(p, 0xAB, 128);
    CHECK(p[0] == 0xAB && p[127] == 0xAB);
    sys.free(sys.state, p, 128, 64);

    /* convenience wrappers (alignment 0 => default) */
    int *q = (int *)hyle_alloc(&sys, sizeof(int) * 8, 0);
    CHECK(q != NULL);
    for (int i = 0; i < 8; ++i)
      q[i] = i * 3;
    int *q2 =
        (int *)hyle_realloc(&sys, q, sizeof(int) * 8, sizeof(int) * 16, 0);
    CHECK(q2 != NULL);
    for (int i = 0; i < 8; ++i)
      CHECK(q2[i] == i * 3); /* data preserved */
    hyle_free(&sys, q2, sizeof(int) * 16, 0);

    /* NULL-handle guards on the inline helpers */
    CHECK(hyle_alloc(NULL, 16, 0) == NULL);
    hyle_free(NULL, NULL, 0, 0); /* must not crash */
  }

  /* ---- system usable / good size ---- */
  {
    size_t g = hyle_system_good_size(100);
    CHECK(g >= 100u);
    void *p = hyle_alloc(&sys, g, 0);
    CHECK(p != NULL);
    CHECK(hyle_system_usable_size(p) >= g);
    hyle_free(&sys, p, g, 0);
    CHECK(hyle_system_usable_size(NULL) == 0u);
    CHECK(hyle_system_good_size(0) == 0u);
  }

  /* ---- strict byte quota allocator adapter ---- */
  {
    hyle_quota_allocator *quota = NULL;
    CHECK_OK(hyle_quota_allocator_create(&quota, sys, 128));
    CHECK(quota != NULL);
    hyle_allocator a = hyle_quota_allocator_as_allocator(quota);
    void *p = hyle_alloc(&a, 96, 16);
    CHECK(p != NULL);
    CHECK(hyle_quota_allocator_used(quota) == 96u);
    CHECK(hyle_quota_allocator_remaining(quota) == 32u);
    CHECK(hyle_alloc(&a, 33, 16) == NULL);
    CHECK(hyle_quota_allocator_remaining(quota) == 32u);
    CHECK(hyle_quota_allocator_limit(quota) == 128u);
    hyle_free(&a, p, 96, 16);
    CHECK(hyle_quota_allocator_used(quota) == 0u);
    CHECK(hyle_quota_allocator_remaining(quota) == 128u);
    CHECK(hyle_quota_allocator_peak(quota) == 96u);
    hyle_quota_allocator_destroy(quota);
  }

  /* ---- opt-in stats allocator + optional leak inventory ---- */
  {
    hyle_stats_allocator *stats = NULL;
    CHECK_OK(hyle_stats_allocator_create(&stats, sys, sys, 1));
    CHECK(stats != NULL);
    hyle_allocator a = hyle_stats_allocator_as_allocator(stats);
    void *p = hyle_alloc(&a, 96, 16);
    void *q = hyle_alloc(&a, 32, 8);
    hyle_alloc_stats snapshot;
    hyle_stats_allocator_snapshot(stats, &snapshot);
    CHECK(snapshot.live_count == 2u);
    CHECK(snapshot.live_bytes == 128u);
    CHECK(snapshot.total_allocs == 2u);
    CHECK(snapshot.peak_live_bytes == 128u);
    CHECK(!hyle_stats_allocator_balanced(stats));
    hyle_mem_leak leaks[1];
    size_t leak_count = 0;
    CHECK_OK(hyle_stats_allocator_copy_leaks(&leak_count, stats, leaks, 1));
    CHECK(leak_count == 2u);
    CHECK(leaks[0].ptr != NULL && leaks[0].size > 0u);
    hyle_free(&a, p, 96, 16);
    hyle_free(&a, q, 32, 8);
    CHECK(hyle_stats_allocator_balanced(stats));
    CHECK_OK(
        hyle_stats_allocator_copy_leaks(&leak_count, stats, NULL, 0));
    CHECK(leak_count == 0u);
    hyle_stats_allocator_reset(stats);
    hyle_stats_allocator_snapshot(stats, &snapshot);
    CHECK(snapshot.total_allocs == 0u && snapshot.peak_live_bytes == 0u);
    hyle_stats_allocator_destroy(stats);
  }
  {
    /* Opaque control, LeakTracker state and StatsState are three independent
     * metadata allocations. Every failure point must preserve Status and
     * unwind earlier allocations through the same injected domain. */
    size_t fail_at;
    for (fail_at = 0; fail_at != 3; ++fail_at) {
      failing_metadata_state state = {sys, 0, fail_at, 0};
      hyle_allocator metadata = {failing_metadata_alloc,
                                 failing_metadata_free,
                                 failing_metadata_realloc, &state};
      hyle_stats_allocator *failed = (hyle_stats_allocator *)(size_t)1;
      hyle_status status =
          hyle_stats_allocator_create(&failed, sys, metadata, 1);
      CHECK(hyle_status_is_resource_exhausted(status));
      CHECK(failed == NULL);
      CHECK(state.live_allocations == 0u);
      hyle_status_free(&status);
    }
  }
  {
    /* Measuring a constrained/failing backing must not force control state
     * through it. The convenience helper keeps metadata on the system heap. */
    hyle_allocator oom = {oom_alloc, oom_free, oom_realloc, NULL};
    hyle_stats_allocator *stats = NULL;
    CHECK_OK(hyle_stats_allocator_create_with_default_metadata(&stats, oom,
                                                               0));
    CHECK(stats != NULL);
    hyle_allocator measured = hyle_stats_allocator_as_allocator(stats);
    CHECK(hyle_alloc(&measured, 64, 16) == NULL);
    CHECK(hyle_stats_allocator_balanced(stats));
    hyle_stats_allocator_destroy(stats);
  }

  /* ---- compile-checked recipe: quota + counters-only stats ---- */
  {
    hyle_stats_allocator *stats = NULL;
    CHECK_OK(hyle_stats_allocator_create_default(&stats, 0));
    CHECK(stats != NULL);
    hyle_allocator measured = hyle_stats_allocator_as_allocator(stats);
    hyle_quota_allocator *quota = NULL;
    CHECK_OK(hyle_quota_allocator_create(&quota, measured, 256));
    CHECK(quota != NULL);
    hyle_allocator bounded = hyle_quota_allocator_as_allocator(quota);
    void *p = hyle_alloc(&bounded, 96, 16);
    CHECK(p != NULL);
    CHECK(hyle_quota_allocator_remaining(quota) == 160u);
    hyle_free(&bounded, p, 96, 16);
    /* The C quota handle itself was allocated through measured. */
    hyle_quota_allocator_destroy(quota);
    CHECK(hyle_stats_allocator_balanced(stats));
    hyle_stats_allocator_destroy(stats);
  }

  /* ---- compile-checked recipe: owned request Region ---- */
  {
    hyle_region_options options = HYLE_REGION_OPTIONS_INIT;
    CHECK(options.max_block_size == 4u * 1024u * 1024u);
    options.initial_block_size = 128;
    options.max_block_size = 256;
    options.large_allocation_threshold = 192;
    options.size_limit = 2048;
    hyle_region *r = NULL;
    CHECK_OK(hyle_region_create_default(&r, &options));
    CHECK(r != NULL);
    void *p = hyle_region_allocate(r, 80, 64);
    CHECK(p != NULL && ((size_t)p % 64u) == 0u);
    CHECK(hyle_region_used(r) == 80u);
    CHECK(hyle_region_reserved(r) > 0u);
    CHECK(hyle_region_peak_used(r) >= 80u);
    CHECK(hyle_region_size_limit(r) == options.size_limit);
    hyle_allocator a = hyle_region_as_allocator(r);
    CHECK(hyle_alloc(&a, 32, 8) != NULL);
    hyle_region_clear(r);
    CHECK(hyle_region_used(r) == 0u);
    hyle_region_release(r);
    CHECK(hyle_region_reserved(r) == 0u);
    hyle_region_destroy(r);

    options.struct_size = 0;
    hyle_status status = hyle_region_create(&r, sys, &options);
    CHECK(hyle_status_is_invalid_argument(status));
    CHECK(r == NULL);
    hyle_status_free(&status);
  }

  /* ---- external fixed Region + variable-size FixedPool ---- */
  {
    unsigned char region_storage[512];
    hyle_fixed_region region;
    CHECK_OK(hyle_fixed_region_init(&region, region_storage,
                                    sizeof region_storage));
    CHECK(hyle_fixed_region_capacity(&region) == sizeof region_storage);
    CHECK(hyle_fixed_region_remaining(&region) == sizeof region_storage);
    void *p = hyle_fixed_region_allocate(&region, 64, 32);
    CHECK(p != NULL && ((size_t)p % 32u) == 0u);
    CHECK(hyle_fixed_region_used(&region) >= 64u);
    CHECK(hyle_fixed_region_remaining(&region) ==
          hyle_fixed_region_capacity(&region) -
              hyle_fixed_region_used(&region));
    hyle_allocator region_alloc = hyle_fixed_region_as_allocator(&region);
    CHECK(hyle_alloc(&region_alloc, 32, 8) != NULL);
    hyle_fixed_region_clear(&region);
    CHECK(hyle_fixed_region_used(&region) == 0u);
    CHECK(hyle_fixed_region_remaining(&region) == sizeof region_storage);
    hyle_status status =
        hyle_fixed_region_init(NULL, region_storage, sizeof region_storage);
    CHECK(hyle_status_is_invalid_argument(status));
    hyle_status_free(&status);
    {
      hyle_fixed_region invalid = {{0, 0, 0}};
      CHECK(hyle_fixed_region_allocate(&invalid, 8, 8) == NULL);
      CHECK(!hyle_allocator_is_valid(
          hyle_fixed_region_as_allocator(&invalid)));
    }

    unsigned char pool_storage[4096];
    hyle_fixed_pool_storage pool_control;
    hyle_fixed_pool *pool = NULL;
    CHECK_OK(hyle_fixed_pool_init(&pool, &pool_control, pool_storage,
                                  sizeof pool_storage));
    CHECK(pool != NULL);
    CHECK(hyle_fixed_pool_capacity(pool) <= sizeof pool_storage);
    void *a = hyle_fixed_pool_allocate(pool, 100, 64);
    void *b = hyle_fixed_pool_allocate(pool, 200, 32);
    CHECK(a != NULL && b != NULL);
    CHECK(hyle_fixed_pool_used(pool) == 300u);
    CHECK(hyle_fixed_pool_peak(pool) == 300u);
    a = hyle_fixed_pool_reallocate(pool, a, 100, 128, 64);
    CHECK(a != NULL);
    hyle_allocator pool_alloc = hyle_fixed_pool_as_allocator(pool);
    void *c = hyle_alloc(&pool_alloc, 64, 16);
    CHECK(c != NULL);
    CHECK(hyle_fixed_pool_peak(pool) >= hyle_fixed_pool_used(pool));
    hyle_free(&pool_alloc, c, 64, 16);
    hyle_fixed_pool_deallocate(pool, a);
    hyle_fixed_pool_deallocate(pool, b);
    CHECK(hyle_fixed_pool_used(pool) == 0u);
    CHECK(hyle_fixed_pool_peak(pool) >= 300u);
    CHECK(hyle_fixed_pool_allocate(pool, 2048, 64) != NULL);
    hyle_fixed_pool_reset(pool); /* caller has discarded every allocation */
    hyle_fixed_pool_destroy(pool);

    CHECK_OK(hyle_fixed_pool_create(&pool, sys, pool_storage,
                                    sizeof pool_storage));
    CHECK(pool != NULL);
    hyle_fixed_pool_destroy(pool);
  }

  /* ---- SharedPool: allocate to exhaustion / deallocate / reset ---- */
  {
    hyle_shared_pool *pool = NULL;
    CHECK_OK(hyle_shared_pool_create_default(&pool, 256, 4, 0));
    CHECK(pool != NULL);
    CHECK(hyle_shared_pool_block_size(pool) == 256u);
    CHECK(hyle_shared_pool_capacity(pool) == 4u);
    void *b[4];
    for (int i = 0; i < 4; ++i) {
      b[i] = hyle_shared_pool_allocate(pool);
      CHECK(b[i] != NULL);
      memset(b[i], i, 256); /* fully writable */
    }
    CHECK(hyle_shared_pool_allocate(pool) == NULL); /* exhausted */
    hyle_shared_pool_deallocate(pool, b[0]);
    void *reuse = hyle_shared_pool_allocate(pool); /* freed slot returns */
    CHECK(reuse != NULL);
    for (int i = 1; i < 4; ++i)
      hyle_shared_pool_deallocate(pool, b[i]);
    hyle_shared_pool_deallocate(pool, reuse);
    hyle_shared_pool_reset(pool); /* all blocks free again */
    void *after_reset = hyle_shared_pool_allocate(pool);
    CHECK(after_reset != NULL);
    hyle_shared_pool_deallocate(pool, after_reset);

    /* as_allocator view: complete fixed-block realloc contract */
    hyle_allocator pv = hyle_shared_pool_as_allocator(pool);
    CHECK(pv.alloc != NULL && pv.free != NULL && pv.realloc != NULL);
    void *v = pv.realloc(pv.state, NULL, 0, 200, HYLE_ALIGNOF(double));
    CHECK(v != NULL);
    ((unsigned char *)v)[0] = 0x5A;
    CHECK(pv.realloc(pv.state, v, 200, 240, HYLE_ALIGNOF(double)) == v);
    CHECK(((unsigned char *)v)[0] == 0x5A);
    CHECK(pv.realloc(pv.state, v, 240, 512, HYLE_ALIGNOF(double)) == NULL);
    CHECK(((unsigned char *)v)[0] == 0x5A); /* failed grow preserves old */
    CHECK(pv.realloc(pv.state, v, 240, 0, HYLE_ALIGNOF(double)) == NULL);
    v = pv.alloc(pv.state, 200, HYLE_ALIGNOF(double));
    CHECK(v != NULL); /* zero-size realloc restored the slot */
    pv.free(pv.state, v, 200, HYLE_ALIGNOF(double));

    hyle_shared_pool_destroy(pool);

    /* invalid backing -> NULL (system-boundary validation) */
    hyle_allocator bad;
    memset(&bad, 0, sizeof bad);
    hyle_status status = hyle_shared_pool_create(&pool, bad, 64, 4, 0);
    CHECK(hyle_status_is_invalid_argument(status));
    CHECK(pool == NULL);
    hyle_status_free(&status);
  }

  /* ---- compile-checked recipe: typed SharedPoolCache + publish ----------- */
  {
    typedef struct {
      unsigned char bytes[128];
    } CachedNode;
    hyle_shared_pool *pool = NULL;
    CHECK_OK(HYLE_OBJECT_POOL_CREATE_DEFAULT(&pool, CachedNode, 8));
    CHECK(pool != NULL);
    hyle_shared_pool_cache *cache = NULL;
    CHECK_OK(HYLE_OBJECT_POOL_CACHE_CREATE_DEFAULT(&cache, pool, 4, 2));
    CHECK(cache != NULL);
    CHECK(hyle_shared_pool_cache_block_size(cache) == sizeof(CachedNode));
    CHECK(hyle_shared_pool_cache_capacity(cache) == 4u);
    CHECK(hyle_shared_pool_cache_transfer_batch(cache) == 2u);
    CachedNode *a =
        (CachedNode *)hyle_shared_pool_cache_allocate(cache);
    CHECK(a != NULL);
    a->bytes[0] = 0x5A;
    CHECK(a->bytes[0] == 0x5A);
    hyle_shared_pool_cache_deallocate(cache, a);
    CachedNode *macro_node =
        HYLE_OBJECT_POOL_CACHE_ACQUIRE(cache, CachedNode);
    CHECK(macro_node != NULL);
    HYLE_OBJECT_POOL_CACHE_RELEASE(cache, macro_node);
    CHECK(hyle_shared_pool_cache_cached(cache) > 0u);
    CHECK(hyle_shared_pool_cache_cached(cache) <= 2u);
    HYLE_OBJECT_POOL_CACHE_PUBLISH(cache);
    CHECK(hyle_shared_pool_cache_cached(cache) == 0u);

    hyle_allocator pv = hyle_shared_pool_cache_as_allocator(cache);
    CHECK(pv.alloc != NULL && pv.free != NULL && pv.realloc != NULL);
    void *b = pv.realloc(pv.state, NULL, 0, 64, HYLE_ALIGNOF(CachedNode));
    CHECK(b != NULL);
    CHECK(pv.realloc(pv.state, b, 64, 96, HYLE_ALIGNOF(CachedNode)) == b);
    CHECK(pv.realloc(pv.state, b, 96, 256, HYLE_ALIGNOF(CachedNode)) == NULL);
    CHECK(pv.realloc(pv.state, b, 96, 0, HYLE_ALIGNOF(CachedNode)) == NULL);
    b = pv.alloc(pv.state, 64, HYLE_ALIGNOF(CachedNode));
    CHECK(b != NULL);
    pv.free(pv.state, b, 64, HYLE_ALIGNOF(CachedNode));
    hyle_shared_pool_cache_publish(cache);
    hyle_shared_pool_cache_destroy(cache);
    hyle_shared_pool_reset(pool);
    hyle_shared_pool_destroy(pool);
  }

  /* ---- typed object-pool convenience macros (raw slot, user-init) ---- */
  {
    typedef struct {
      int a;
      double b;
    } Node;
    hyle_shared_pool *pool = NULL;
    CHECK_OK(HYLE_OBJECT_POOL_CREATE_DEFAULT(&pool, Node, 8));
    CHECK(pool != NULL);
    CHECK(hyle_shared_pool_block_size(pool) == sizeof(Node));
    CHECK(hyle_shared_pool_capacity(pool) == 8u);
    Node *n =
        HYLE_OBJECT_POOL_ACQUIRE(pool, Node); /* typed, no cast at call site */
    CHECK(n != NULL);
    n->a = 7;
    n->b = 1.5; /* user-initialized slot */
    CHECK(n->a == 7 && n->b == 1.5);
    HYLE_OBJECT_POOL_RELEASE(pool, n);
    hyle_shared_pool_destroy(pool);
  }

  /* ---- StackAllocator: bump / mark / rewind / reset ---- */
  {
    hyle_stack_allocator *sa = NULL;
    CHECK_OK(hyle_stack_allocator_create_default(&sa, 4096));
    CHECK(sa != NULL);
    CHECK(hyle_stack_allocator_capacity(sa) == 4096u);
    CHECK(hyle_stack_allocator_remaining(sa) == 4096u);
    void *a = hyle_stack_allocator_allocate(sa, 16, 16);
    void *bb = hyle_stack_allocator_allocate(sa, 16, 16);
    CHECK(a != NULL && bb != NULL);
    CHECK((char *)a < (char *)bb); /* bumps upward */
    CHECK(((size_t)a % 16u) == 0u);
    CHECK(hyle_stack_allocator_used(sa) > 0u);

    void *m = hyle_stack_allocator_mark(sa);
    void *x = hyle_stack_allocator_allocate(sa, 200, 1);
    CHECK(x != NULL);
    hyle_stack_allocator_rewind(sa, m);
    void *y = hyle_stack_allocator_allocate(sa, 200, 1);
    CHECK(x == y); /* region reused after rewind */

    hyle_stack_allocator_reset(sa);
    CHECK(hyle_stack_allocator_used(sa) == 0u);

    /* as_allocator view */
    hyle_allocator av = hyle_stack_allocator_as_allocator(sa);
    void *z = av.alloc(av.state, 32, 8);
    CHECK(z != NULL);
    av.free(av.state, z, 32, 8); /* no-op, must not crash */
    hyle_stack_allocator_destroy(sa);
  }

  /* ---- PooledRegion: bump over one pooled block ---- */
  {
    hyle_shared_pool *pool = NULL;
    CHECK_OK(hyle_shared_pool_create_default(&pool, 4096, 2, 0));
    CHECK(pool != NULL);
    hyle_pooled_region *region = NULL;
    CHECK_OK(hyle_pooled_region_create(&region, pool));
    CHECK(region != NULL);
    CHECK(hyle_pooled_region_block_size(region) == 4096u);
    CHECK(hyle_pooled_region_remaining(region) == 4096u);
    CHECK(hyle_pooled_region_allocate(region, 0, 16) == NULL);
    CHECK(hyle_pooled_region_used(region) == 0u);
    void *p = hyle_pooled_region_allocate(region, 100, 16);
    CHECK(p != NULL);
    CHECK(((size_t)p % 16u) == 0u);
    CHECK(hyle_pooled_region_used(region) >= 100u);
    {
      hyle_allocator av = hyle_pooled_region_as_allocator(region);
      void *q = hyle_alloc(&av, 32, 8);
      CHECK(q != NULL);
      hyle_free(&av, q, 32, 8); /* bump allocator free is a no-op */
    }
    hyle_pooled_region_clear(region);
    CHECK(hyle_pooled_region_used(region) == 0u);
    hyle_pooled_region_destroy(region); /* returns the block to the pool */
    hyle_shared_pool_destroy(pool);
  }

  /* ---- PooledRegion: pool exhaustion is structured and graceful -------- */
  {
    hyle_shared_pool *pool = NULL;
    CHECK_OK(hyle_shared_pool_create_default(&pool, 4096, 1, 0));
    CHECK(pool != NULL);
    hyle_pooled_region *a1 = NULL;
    CHECK_OK(hyle_pooled_region_create(&a1, pool));
    CHECK(a1 != NULL);
    hyle_pooled_region *a2 = (hyle_pooled_region *)(size_t)1;
    hyle_status status = hyle_pooled_region_create(&a2, pool);
    CHECK(hyle_status_is_resource_exhausted(status));
    CHECK(hyle_status_has_message(status));
    CHECK(a2 == NULL);
    hyle_status_free(&status);
    hyle_pooled_region_destroy(a1); /* returns the only block */
    hyle_pooled_region *a3 = NULL;
    CHECK_OK(hyle_pooled_region_create(&a3, pool));
    CHECK(a3 != NULL);
    hyle_pooled_region_destroy(a3);
    hyle_shared_pool_destroy(pool);
  }

  /* ---- GrowablePool: growable raw-slot pool (grow / reuse / reset / macros)
   * ---- */
  {
    /* tiny initial count forces geometric growth across several blocks */
    hyle_growable_pool *gp = NULL;
    CHECK_OK(hyle_growable_pool_create_default(&gp, sizeof(long), 0, 2));
    CHECK(gp != NULL);
    CHECK(hyle_growable_pool_live(gp) == 0u);
    CHECK(hyle_growable_pool_slot_stride(gp) >= sizeof(long));
    long *slots[100];
    for (int i = 0; i < 100; ++i) {
      slots[i] = (long *)hyle_growable_pool_acquire(gp);
      CHECK(slots[i] != NULL);
      *slots[i] = i; /* raw slot, user-init */
    }
    CHECK(hyle_growable_pool_live(gp) == 100u);
    CHECK(hyle_growable_pool_capacity(gp) >= 100u); /* grew to hold them */
    for (int i = 0; i < 100; ++i)
      CHECK(*slots[i] == i); /* growth moved nothing */

    /* a freed slot is recycled by the next acquire (free-list) */
    hyle_growable_pool_release(gp, slots[50]);
    CHECK(hyle_growable_pool_live(gp) == 99u);
    long *reuse = (long *)hyle_growable_pool_acquire(gp);
    CHECK(reuse == slots[50]);
    slots[50] = reuse;

    /* release all, reset -> blocks handed back; still usable afterwards */
    for (int i = 0; i < 100; ++i)
      hyle_growable_pool_release(gp, slots[i]);
    CHECK(hyle_growable_pool_live(gp) == 0u);
    hyle_growable_pool_reset(gp);
    CHECK(hyle_growable_pool_capacity(gp) == 0u); /* trimmed to nothing */
    void *again = hyle_growable_pool_acquire(gp); /* regrows from initial */
    CHECK(again != NULL);
    hyle_growable_pool_release(gp, again);
    hyle_growable_pool_destroy(gp);

    /* Invalid backing -> structured error and cleared output. */
    hyle_allocator bad;
    memset(&bad, 0, sizeof bad);
    hyle_status status = hyle_growable_pool_create(&gp, bad, 8, 0, 4);
    CHECK(hyle_status_is_invalid_argument(status));
    CHECK(hyle_status_has_message(status));
    CHECK(gp == NULL);
    hyle_status_free(&status);

    /* Runtime C layout is untrusted: a stride that would wrap must fail at
       create, not under-allocate and hand out a one-past-block pointer. */
    status = hyle_growable_pool_create(&gp, sys, SIZE_MAX, 16, 1);
    CHECK(hyle_status_is_invalid_argument(status));
    CHECK(gp == NULL);
    hyle_status_free(&status);
    /* typed convenience macros (raw slot, user-init) */
    typedef struct {
      int x;
      int y;
    } Pt;
    hyle_growable_pool *tp = NULL;
    CHECK_OK(HYLE_GROWABLE_POOL_CREATE_DEFAULT(&tp, Pt, 8));
    CHECK(tp != NULL);
    Pt *pt =
        HYLE_GROWABLE_POOL_ACQUIRE(tp, Pt); /* typed, no cast at call site */
    CHECK(pt != NULL);
    pt->x = 3;
    pt->y = 4;
    CHECK(pt->x == 3 && pt->y == 4);
    HYLE_GROWABLE_POOL_RELEASE(tp, pt);
    hyle_growable_pool_destroy(tp);
  }

  /* ---- mock OOM backing: create is graceful Status, not abort ---------- */
  {
    hyle_allocator oom;
    oom.alloc = oom_alloc;
    oom.free = oom_free;
    oom.realloc = oom_realloc;
    oom.state = NULL;

    /* SharedPool slab OOM -> RESOURCE_EXHAUSTED (was a fatal abort). */
    hyle_shared_pool *failed_pool = (hyle_shared_pool *)(size_t)1;
    hyle_status status =
        hyle_shared_pool_create(&failed_pool, oom, 64, 4, 0);
    CHECK(hyle_status_is_resource_exhausted(status));
    CHECK(hyle_status_has_message(status));
    CHECK(failed_pool == NULL);
    hyle_status_free(&status);
    hyle_shared_pool *cache_core = NULL;
    CHECK_OK(hyle_shared_pool_create_default(&cache_core, 64, 4, 0));
    CHECK(cache_core != NULL);
    hyle_shared_pool_cache *failed_cache = NULL;
    status = hyle_shared_pool_cache_create(&failed_cache, cache_core, oom, 4,
                                           2);
    CHECK(hyle_status_is_resource_exhausted(status));
    CHECK(failed_cache == NULL);
    hyle_status_free(&status);
    hyle_shared_pool_destroy(cache_core);

    /* StackAllocator slab OOM -> RESOURCE_EXHAUSTED. */
    hyle_stack_allocator *failed_stack = NULL;
    status = hyle_stack_allocator_create(&failed_stack, oom, 4096);
    CHECK(hyle_status_is_resource_exhausted(status));
    CHECK(failed_stack == NULL);
    hyle_status_free(&status);

    hyle_stats_allocator *failed_stats = NULL;
    status = hyle_stats_allocator_create(&failed_stats, oom, oom, 0);
    CHECK(hyle_status_is_resource_exhausted(status));
    CHECK(failed_stats == NULL);
    hyle_status_free(&status);

    /* The injected backing owns the opaque control object too. An always-OOM
       backing therefore fails create without falling through to global new. */
    hyle_growable_pool *gp = NULL;
    status = hyle_growable_pool_create(&gp, oom, 32, 0, 4);
    CHECK(hyle_status_is_resource_exhausted(status));
    CHECK(gp == NULL);
    hyle_status_free(&status);
  }

  /* ---- HugeRegion: open / write / destroy (THP hint) ---- */
  {
    /* 4 MiB span; THP hint is soft, so open must still succeed everywhere. */
    size_t bytes = 4u * 1024u * 1024u;
    hyle_huge_region *hr = NULL;
    CHECK_OK(hyle_huge_region_open(&hr, bytes, HYLE_HUGE_THP_HINT,
                                   HYLE_MEM_NUMA_ANY));
    CHECK(hr != NULL);
    if (hr != NULL) {
      CHECK(hyle_huge_region_size(hr) >= bytes);
      CHECK((hyle_huge_region_requested_flags(hr) & HYLE_HUGE_THP_HINT) != 0);
      CHECK((hyle_huge_region_accepted_flags(hr) & ~HYLE_HUGE_THP_HINT) == 0);
      CHECK(hyle_huge_region_requested_numa_node(hr) == HYLE_MEM_NUMA_ANY);
      CHECK(hyle_huge_region_numa_placement_applied(hr) == 0);
      unsigned char *d = (unsigned char *)hyle_huge_region_data(hr);
      CHECK(d != NULL);
      d[0] = 0x5A;
      d[bytes - 1] = 0xA5; /* fully committed + writable */
      CHECK(d[0] == 0x5A && d[bytes - 1] == 0xA5);
      hyle_huge_region_destroy(hr);
    }

    hyle_huge_region *hl = NULL;
    CHECK_OK(hyle_huge_region_open_local(&hl, bytes, HYLE_HUGE_THP_HINT));
    CHECK(hl != NULL);
    hyle_huge_region_destroy(hl);

    hyle_huge_region *hi = NULL;
    CHECK_OK(hyle_huge_region_open_interleaved(&hi, bytes));
    CHECK(hi != NULL);
    hyle_huge_region_destroy(hi);

    hyle_status status = hyle_huge_region_open(
        &hr, bytes, HYLE_HUGE_NUMA_BIND | HYLE_HUGE_NUMA_INTERLEAVE,
        HYLE_MEM_NUMA_ANY);
    CHECK(hyle_status_is_invalid_argument(status));
    CHECK(hr == NULL);
    hyle_status_free(&status);
    status = hyle_huge_region_open(&hr, bytes, 1u << 31,
                                   HYLE_MEM_NUMA_ANY);
    CHECK(hyle_status_is_invalid_argument(status));
    CHECK(hr == NULL);
    hyle_status_free(&status);

    hyle_huge_region_destroy(NULL); /* NULL-safe */
  }

  /* ---- structured Status + named-enum types (FFI discoverability) ---- */
  {
    hyle_huge_flags one_flag = HYLE_HUGE_THP_HINT; /* named enum as a type */
    hyle_mem_numa_node any = HYLE_MEM_NUMA_ANY;
    uint32_t flags = (uint32_t)one_flag | (uint32_t)HYLE_HUGE_PREFAULT;
    hyle_huge_region *hr = NULL;
    CHECK_OK(hyle_huge_region_open(&hr, 2u * 1024u * 1024u, flags,
                                   (int)any));
    CHECK(hr != NULL);
    hyle_huge_region_destroy(hr);
  }

  /* ---- process-wide policy: callable, no crash (effect is global/soft) ----
   */
  hyle_enable_allocator_hugepages(0, 0);
  hyle_enable_allocator_numa(0);
  hyle_system_collect(0);
  CHECK_OK(hyle_reserve_huge_1g_pages(0, HYLE_MEM_NUMA_ANY, 0));
  {
    hyle_status status = hyle_reserve_huge_1g_pages(1, -2, 1);
    CHECK(hyle_status_is_out_of_range(status));
    CHECK(hyle_status_has_message(status));
    hyle_status_free(&status);
  }

  if (g_fails) {
    fprintf(stderr, "c_mem_test: %d failed\n", g_fails);
    return 1;
  }
  printf("c_mem_test: all checks passed\n");
  return 0;
}
