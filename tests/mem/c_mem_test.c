/* Pure C consumer of include/hyle/c/mem.h */
#if defined(_WIN32)
#  define _CRT_SECURE_NO_WARNINGS 1
#endif
#include "hyle/c/mem.h"

#include <stdio.h>
#include <string.h>

static int g_fails = 0;
#define CHECK(c)                                                              \
  do {                                                                        \
    if (!(c)) {                                                               \
      fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #c, __FILE__, __LINE__); \
      ++g_fails;                                                              \
    }                                                                         \
  } while (0)

int main(void) {
  CHECK(hyle_default_alignment() >= sizeof(void*));

  /* ---- system allocator: alloc / write / realloc / free ---- */
  hyle_allocator sys = hyle_system_allocator();
  CHECK(sys.alloc != NULL && sys.free != NULL && sys.realloc != NULL);
  {
    /* direct vtable call */
    unsigned char* p = (unsigned char*)sys.alloc(sys.state, 128, 64);
    CHECK(p != NULL);
    CHECK(((size_t)p % 64u) == 0u);  /* honored explicit alignment */
    memset(p, 0xAB, 128);
    CHECK(p[0] == 0xAB && p[127] == 0xAB);
    sys.free(sys.state, p, 128, 64);

    /* convenience wrappers (alignment 0 => default) */
    int* q = (int*)hyle_alloc(&sys, sizeof(int) * 8, 0);
    CHECK(q != NULL);
    for (int i = 0; i < 8; ++i) q[i] = i * 3;
    int* q2 = (int*)hyle_realloc(&sys, q, sizeof(int) * 8, sizeof(int) * 16, 0);
    CHECK(q2 != NULL);
    for (int i = 0; i < 8; ++i) CHECK(q2[i] == i * 3);  /* data preserved */
    hyle_free(&sys, q2, sizeof(int) * 16, 0);

    /* NULL-handle guards on the inline helpers */
    CHECK(hyle_alloc(NULL, 16, 0) == NULL);
    hyle_free(NULL, NULL, 0, 0);  /* must not crash */
  }

  /* ---- system usable / good size ---- */
  {
    size_t g = hyle_system_good_size(100);
    CHECK(g >= 100u);
    void* p = hyle_alloc(&sys, 100, 0);
    CHECK(p != NULL);
    CHECK(hyle_system_usable_size(p) >= 100u);
    CHECK(hyle_system_usable_size(p) == g);  /* preview == reality */
    hyle_free(&sys, p, 100, 0);
    CHECK(hyle_system_usable_size(NULL) == 0u);
    CHECK(hyle_system_good_size(0) == 0u);
  }

  /* ---- MemoryArena: allocate to exhaustion / deallocate / reset ---- */
  {
    hyle_memory_arena_o* pool =
        hyle_memory_arena_create(sys, 256, 4, 0);
    CHECK(pool != NULL);
    CHECK(hyle_memory_arena_block_size(pool) == 256u);
    CHECK(hyle_memory_arena_capacity(pool) == 4u);
    void* b[4];
    for (int i = 0; i < 4; ++i) {
      b[i] = hyle_memory_arena_allocate(pool);
      CHECK(b[i] != NULL);
      memset(b[i], i, 256);  /* fully writable */
    }
    CHECK(hyle_memory_arena_allocate(pool) == NULL);  /* exhausted */
    hyle_memory_arena_deallocate(pool, b[0]);
    void* reuse = hyle_memory_arena_allocate(pool);   /* freed slot returns */
    CHECK(reuse != NULL);
    for (int i = 1; i < 4; ++i) hyle_memory_arena_deallocate(pool, b[i]);
    hyle_memory_arena_deallocate(pool, reuse);
    hyle_memory_arena_reset(pool);                    /* all blocks free again */
    CHECK(hyle_memory_arena_allocate(pool) != NULL);

    /* as_allocator view: pop a fixed block through the vtable */
    hyle_allocator pv = hyle_memory_arena_as_allocator(pool);
    CHECK(pv.alloc != NULL);
    void* v = pv.alloc(pv.state, 200, 0);  /* must fit block_size */
    CHECK(v != NULL);
    pv.free(pv.state, v, 200, 0);

    hyle_memory_arena_destroy(pool);

    /* invalid backing -> NULL (system-boundary validation) */
    hyle_allocator bad;
    memset(&bad, 0, sizeof bad);
    CHECK(hyle_memory_arena_create(bad, 64, 4, 0) == NULL);
  }

  /* ---- typed object-pool convenience macros (raw slot, user-init) ---- */
  {
    typedef struct { int a; double b; } Node;
    hyle_memory_arena_o* pool = HYLE_OBJECT_POOL_CREATE(sys, Node, 8);
    CHECK(pool != NULL);
    CHECK(hyle_memory_arena_block_size(pool) == sizeof(Node));
    CHECK(hyle_memory_arena_capacity(pool) == 8u);
    Node* n = HYLE_OBJECT_POOL_ACQUIRE(pool, Node);  /* typed, no cast at call site */
    CHECK(n != NULL);
    n->a = 7;
    n->b = 1.5;                                       /* user-initialized slot */
    CHECK(n->a == 7 && n->b == 1.5);
    HYLE_OBJECT_POOL_RELEASE(pool, n);
    hyle_memory_arena_destroy(pool);
  }

  /* ---- StackAllocator: bump / mark / rewind / reset ---- */
  {
    hyle_stack_allocator_o* sa = hyle_stack_allocator_create(sys, 4096);
    CHECK(sa != NULL);
    CHECK(hyle_stack_allocator_capacity(sa) == 4096u);
    CHECK(hyle_stack_allocator_remaining(sa) == 4096u);
    void* a = hyle_stack_allocator_allocate(sa, 16, 16);
    void* bb = hyle_stack_allocator_allocate(sa, 16, 16);
    CHECK(a != NULL && bb != NULL);
    CHECK((char*)a < (char*)bb);              /* bumps upward */
    CHECK(((size_t)a % 16u) == 0u);
    CHECK(hyle_stack_allocator_used(sa) > 0u);

    void* m = hyle_stack_allocator_mark(sa);
    void* x = hyle_stack_allocator_allocate(sa, 200, 1);
    CHECK(x != NULL);
    hyle_stack_allocator_rewind(sa, m);
    void* y = hyle_stack_allocator_allocate(sa, 200, 1);
    CHECK(x == y);                            /* region reused after rewind */

    hyle_stack_allocator_reset(sa);
    CHECK(hyle_stack_allocator_used(sa) == 0u);

    /* as_allocator view */
    hyle_allocator av = hyle_stack_allocator_as_allocator(sa);
    void* z = av.alloc(av.state, 32, 8);
    CHECK(z != NULL);
    av.free(av.state, z, 32, 8);              /* no-op, must not crash */
    hyle_stack_allocator_destroy(sa);
  }

  /* ---- Arena: bump over one pooled block ---- */
  {
    hyle_memory_arena_o* pool = hyle_memory_arena_create(sys, 4096, 2, 0);
    CHECK(pool != NULL);
    hyle_arena_o* ar = hyle_arena_create(pool);
    CHECK(ar != NULL);
    CHECK(hyle_arena_block_size(ar) == 4096u);
    CHECK(hyle_arena_remaining(ar) == 4096u);
    void* p = hyle_arena_allocate(ar, 100, 16);
    CHECK(p != NULL);
    CHECK(((size_t)p % 16u) == 0u);
    CHECK(hyle_arena_used(ar) >= 100u);
    hyle_arena_reset(ar);
    CHECK(hyle_arena_used(ar) == 0u);
    hyle_arena_destroy(ar);                   /* returns the block to the pool */
    hyle_memory_arena_destroy(pool);
  }

  /* ---- Arena: pool exhaustion is graceful -- NULL + last-error, not abort
   * (followup #2 arch A: hyle_arena_create no longer aborts when the backing
   * pool has no free block; it returns NULL and sets hyle_mem_last_error) ---- */
  {
    hyle_memory_arena_o* pool = hyle_memory_arena_create(sys, 4096, 1, 0);
    CHECK(pool != NULL);
    hyle_arena_o* a1 = hyle_arena_create(pool);  /* borrows the only block */
    CHECK(a1 != NULL);
    hyle_arena_o* a2 = hyle_arena_create(pool);  /* pool exhausted */
    CHECK(a2 == NULL);                           /* graceful NULL, not abort */
    CHECK(hyle_mem_last_error()[0] != '\0');     /* a non-empty reason string */
    hyle_arena_destroy(a1);                      /* returns the only block */
    hyle_arena_o* a3 = hyle_arena_create(pool);  /* borrowable again */
    CHECK(a3 != NULL);
    hyle_arena_destroy(a3);
    hyle_memory_arena_destroy(pool);
  }

  /* ---- GrowablePool: growable raw-slot pool (grow / reuse / reset / macros) ---- */
  {
    /* tiny initial count forces geometric growth across several blocks */
    hyle_growable_pool_o* gp = hyle_growable_pool_create(sys, sizeof(long), 0, 2);
    CHECK(gp != NULL);
    CHECK(hyle_growable_pool_live(gp) == 0u);
    long* slots[100];
    for (int i = 0; i < 100; ++i) {
      slots[i] = (long*)hyle_growable_pool_acquire(gp);
      CHECK(slots[i] != NULL);
      *slots[i] = i;                                  /* raw slot, user-init */
    }
    CHECK(hyle_growable_pool_live(gp) == 100u);
    CHECK(hyle_growable_pool_capacity(gp) >= 100u);   /* grew to hold them */
    for (int i = 0; i < 100; ++i) CHECK(*slots[i] == i);  /* growth moved nothing */

    /* a freed slot is recycled by the next acquire (free-list) */
    hyle_growable_pool_release(gp, slots[50]);
    CHECK(hyle_growable_pool_live(gp) == 99u);
    long* reuse = (long*)hyle_growable_pool_acquire(gp);
    CHECK(reuse == slots[50]);
    slots[50] = reuse;

    /* release all, reset -> blocks handed back; still usable afterwards */
    for (int i = 0; i < 100; ++i) hyle_growable_pool_release(gp, slots[i]);
    CHECK(hyle_growable_pool_live(gp) == 0u);
    hyle_growable_pool_reset(gp);
    CHECK(hyle_growable_pool_capacity(gp) == 0u);     /* trimmed to nothing */
    void* again = hyle_growable_pool_acquire(gp);     /* regrows from initial */
    CHECK(again != NULL);
    hyle_growable_pool_release(gp, again);
    hyle_growable_pool_destroy(gp);

    /* invalid backing -> NULL + non-empty last-error */
    hyle_allocator bad;
    memset(&bad, 0, sizeof bad);
    CHECK(hyle_growable_pool_create(bad, 8, 0, 4) == NULL);
    CHECK(hyle_mem_last_error()[0] != '\0');

    /* typed convenience macros (raw slot, user-init) */
    typedef struct { int x; int y; } Pt;
    hyle_growable_pool_o* tp = HYLE_GROWABLE_POOL_CREATE(sys, Pt, 8);
    CHECK(tp != NULL);
    Pt* pt = HYLE_GROWABLE_POOL_ACQUIRE(tp, Pt);      /* typed, no cast at call site */
    CHECK(pt != NULL);
    pt->x = 3;
    pt->y = 4;
    CHECK(pt->x == 3 && pt->y == 4);
    HYLE_GROWABLE_POOL_RELEASE(tp, pt);
    hyle_growable_pool_destroy(tp);
  }

  /* ---- HugeRegion: open / write / destroy (THP hint) ---- */
  {
    /* 4 MiB span; THP hint is soft, so open must still succeed everywhere. */
    size_t bytes = 4u * 1024u * 1024u;
    hyle_huge_region_o* hr =
        hyle_huge_region_open(bytes, HYLE_HUGE_THP_HINT, HYLE_MEM_NUMA_ANY);
    CHECK(hr != NULL);
    if (hr != NULL) {
      CHECK(hyle_huge_region_valid(hr));
      CHECK(hyle_huge_region_size(hr) >= bytes);
      unsigned char* d = (unsigned char*)hyle_huge_region_data(hr);
      CHECK(d != NULL);
      d[0] = 0x5A;
      d[bytes - 1] = 0xA5;  /* fully committed + writable */
      CHECK(d[0] == 0x5A && d[bytes - 1] == 0xA5);
      hyle_huge_region_destroy(hr);
    }

    hyle_huge_region_o* hl =
        hyle_huge_region_open_local(bytes, HYLE_HUGE_THP_HINT);
    CHECK(hl != NULL);
    hyle_huge_region_destroy(hl);

    hyle_huge_region_o* hi = hyle_huge_region_open_interleaved(bytes);
    CHECK(hi != NULL);
    hyle_huge_region_destroy(hi);

    hyle_huge_region_destroy(NULL);  /* NULL-safe */
  }

  /* ---- diagnostics + named-enum types (FFI discoverability) ---- */
  {
    hyle_huge_flags one_flag = HYLE_HUGE_THP_HINT;   /* named enum as a type */
    hyle_mem_numa_node any = HYLE_MEM_NUMA_ANY;
    uint32_t flags = (uint32_t)one_flag | (uint32_t)HYLE_HUGE_PREFAULT;
    hyle_huge_region_o* hr =
        hyle_huge_region_open(2u * 1024u * 1024u, flags, (int)any);
    CHECK(hr != NULL);
    hyle_huge_region_destroy(hr);
    CHECK(hyle_mem_last_error() != NULL);  /* always a valid C string ("") */
  }

  /* ---- process-wide policy: callable, no crash (effect is global/soft) ---- */
  hyle_enable_allocator_numa(0);
  CHECK(hyle_reserve_huge_1g_pages(0, HYLE_MEM_NUMA_ANY, 0) == 1);  /* pages==0 ok */

  if (g_fails) {
    fprintf(stderr, "c_mem_test: %d failed\n", g_fails);
    return 1;
  }
  printf("c_mem_test: all checks passed\n");
  return 0;
}
