/* SPDX: same as HYLE.
 *
 * Pre-release pure-C surface for the HYLE mem module. This header is reviewed
 * by C user tasks rather than mechanically mirroring the native C++ classes;
 * names and signatures are not an ABI compatibility promise before 1.0.
 *
 * Naming: every symbol is `hyle_*` / `HYLE_*`.
 * Do NOT include hyle/mem/ headers from C or cross-compiler FFI.
 *
 * Surface: the injectable Allocator vtable; quota/stats decorators; owned and
 * external regions; variable-size, fixed-slot, cached and growable pools;
 * HugeRegion; and process-wide mimalloc maintenance/policy. Opaque public
 * aliases hide owning implementation objects; common constructors have
 * header-inline system-backing forms. Docs (1:1):
 * docs/guide/c-mem.md
 */
#ifndef HYLE_C_MEM_H_
#define HYLE_C_MEM_H_

/* Intentionally the C header: this file is the language-neutral ABI boundary.
 * Native C++ users include the hyle/mem headers, which use <cstddef> and
 * std::size_t. */
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- export (same module as C++ mem: HYLE_MEM_BUILD) -----------------------
 */
#if defined(HYLE_STATIC)
#define HYLE_MEM_C_API
#elif defined(_WIN32) || defined(__CYGWIN__)
#if defined(HYLE_MEM_BUILD)
#define HYLE_MEM_C_API __declspec(dllexport)
#else
#define HYLE_MEM_C_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define HYLE_MEM_C_API __attribute__((visibility("default")))
#else
#define HYLE_MEM_C_API
#endif

/* alignof spelling for the typed-pool macros below: C++ / C23 keyword, C11
 * `_Alignof`, else pass 0 (hyle_shared_pool_create normalizes it to the default
 * alignment). */
#if defined(__cplusplus)
#define HYLE_ALIGNOF(T) alignof(T)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define HYLE_ALIGNOF(T) _Alignof(T)
#else
#define HYLE_ALIGNOF(T) 0
#endif

/* ========================================================================== */
/* Allocator vtable — the injectable {alloc, free, realloc, state} handle. */
/*                                                                            */
/* This is a NATURAL C ABI shape: three C function pointers + a state void*, */
/* trivially copyable and cross-DLL stable (exactly what jemalloc / C++17 pmr */
/* / cudaMalloc expose). It is layout-compatible with the C++ hyle::mem:: */
/* Allocator by design, but the bridge NEVER reinterpret-casts: it constructs */
/* a C++ Allocator explicitly from these four fields (and vice versa). */
/*                                                                            */
/* Contract (SIZED-FREE model, like Rust GlobalAlloc / jemalloc sdallocx): */
/*   alloc  (state, size, alignment)                -> ptr (NULL on
 * OOM/size==0)*/
/*   free   (state, ptr, size, alignment)           -> void (ptr==NULL no-op) */
/*   realloc(state, ptr, old_size, new_size, align) -> ptr (the COLD path) */
/*     ptr==NULL allocates; new_size==0 applies the backing's deallocation */
/*     semantics and returns NULL (bulk allocators may reclaim on reset); a */
/*     failed non-zero resize leaves the original allocation live unchanged. */
/* deallocate carries the size — do not pass a size that differs from the */
/* allocating request (arenas/pools/stacks do not store per-pointer size). */
/* ========================================================================== */

typedef struct hyle_allocator {
  void *(*alloc)(void *state, size_t size, size_t alignment);
  void (*free)(void *state, void *ptr, size_t size, size_t alignment);
  void *(*realloc)(void *state, void *ptr, size_t old_size, size_t new_size,
                   size_t alignment);
  void *state;
} hyle_allocator;

/* A zero-initialized allocator is the null handle. This validator is useful at
 * FFI/configuration boundaries; hot code should normally dispatch directly. */
static inline int hyle_allocator_is_valid(hyle_allocator a) {
  return a.alloc != NULL && a.free != NULL && a.realloc != NULL;
}

/* Default alignment when a caller passes 0 (matches alignof(max_align_t)).
 * Header-inline so the common hyle_alloc(..., 0) path folds this to a constant
 * instead of paying a DLL call before the allocator dispatch. */
static inline size_t hyle_default_alignment(void) {
#if defined(_MSC_VER)
  /* MSVC's C17 <stddef.h> does not expose max_align_t. HYLE's Windows
   * allocator ABI and mimalloc backing both guarantee 16-byte fundamental
   * alignment on the supported MSVC/clang-cl targets. */
  return 16u;
#else
  return HYLE_ALIGNOF(max_align_t);
#endif
}

/* Process-wide mimalloc-backed system allocator (the same handle C++ uses). */
HYLE_MEM_C_API hyle_allocator hyle_system_allocator(void);

/* Convenience: call through a handle without hand-dispatching the fn pointers.
 * alignment==0 => hyle_default_alignment(). Header-inline (no DLL hop). */
static inline void *hyle_alloc(const hyle_allocator *a, size_t size,
                               size_t alignment) {
  if (a == NULL || a->alloc == NULL)
    return NULL;
  return a->alloc(a->state, size,
                  alignment != 0 ? alignment : hyle_default_alignment());
}
static inline void hyle_free(const hyle_allocator *a, void *ptr, size_t size,
                             size_t alignment) {
  if (a == NULL || a->free == NULL)
    return;
  a->free(a->state, ptr, size,
          alignment != 0 ? alignment : hyle_default_alignment());
}
static inline void *hyle_realloc(const hyle_allocator *a, void *ptr,
                                 size_t old_size, size_t new_size,
                                 size_t alignment) {
  if (a == NULL || a->realloc == NULL)
    return NULL;
  return a->realloc(a->state, ptr, old_size, new_size,
                    alignment != 0 ? alignment : hyle_default_alignment());
}

/* ---- system usable/good size (SystemAllocator only) ------------------------
 */
/* HYLE's Allocator is a sized-free model, so there is no vtable usable_size().
 * These two are SPECIFIC to hyle_system_allocator(), where the mimalloc backing
 * knows the true size-class. */
/* Actual usable bytes of a live system_allocator block (>= requested); NULL->0.
 */
HYLE_MEM_C_API size_t hyle_system_usable_size(const void *ptr);
/* Recommended request size for at least `size` bytes (mi_good_size/nallocx
 * shape).
 * >= size; not guaranteed equal to usable_size(alloc(size)); 0->0. */
HYLE_MEM_C_API size_t hyle_system_good_size(size_t size);
/* Ask the process heap to return unused memory. force==0 is the ordinary
 * maintenance path; non-zero requests an aggressive, potentially millisecond
 * cold-path collection. */
HYLE_MEM_C_API void hyle_system_collect(int force);

/* ========================================================================== */
/* Allocator adapter — strict requested-byte quota */
/* ========================================================================== */

typedef struct hyle_quota_allocator_o hyle_quota_allocator;

/* Thread-safe strict byte budget over `backing`. The opaque control and the
 * returned allocator borrow `backing`; all must be quiesced and all allocations
 * freed before destroy. */
HYLE_MEM_C_API hyle_quota_allocator *
hyle_quota_allocator_create(hyle_allocator backing, size_t limit);
HYLE_MEM_C_API void hyle_quota_allocator_destroy(hyle_quota_allocator *q);
HYLE_MEM_C_API hyle_allocator
hyle_quota_allocator_as_allocator(hyle_quota_allocator *q);
HYLE_MEM_C_API size_t
hyle_quota_allocator_used(const hyle_quota_allocator *q);
HYLE_MEM_C_API size_t
hyle_quota_allocator_peak(const hyle_quota_allocator *q);
HYLE_MEM_C_API size_t
hyle_quota_allocator_limit(const hyle_quota_allocator *q);
HYLE_MEM_C_API size_t
hyle_quota_allocator_remaining(const hyle_quota_allocator *q);

static inline hyle_quota_allocator *
hyle_quota_allocator_create_default(size_t limit) {
  return hyle_quota_allocator_create(hyle_system_allocator(), limit);
}

/* ========================================================================== */
/* Allocator adapter — opt-in counters and optional leak inventory */
/* ========================================================================== */

typedef struct hyle_alloc_stats {
  uint64_t live_count;
  uint64_t live_bytes;
  uint64_t total_allocs;
  uint64_t total_frees;
  uint64_t total_bytes_allocated;
  uint64_t peak_live_bytes;
  uint64_t reallocs;
} hyle_alloc_stats;

typedef struct hyle_mem_leak {
  void *ptr;
  size_t size;
} hyle_mem_leak;

typedef struct hyle_stats_allocator_o hyle_stats_allocator;

/* Thread-safe opt-in instrumentation over `backing`. Non-zero track_leaks adds
 * a locked pointer map and is intended for diagnostics, not production hot
 * paths. The returned allocator and all its allocations must be quiesced
 * before destroy. */
HYLE_MEM_C_API hyle_stats_allocator *
hyle_stats_allocator_create(hyle_allocator backing, int track_leaks);
HYLE_MEM_C_API void hyle_stats_allocator_destroy(hyle_stats_allocator *s);
HYLE_MEM_C_API hyle_allocator
hyle_stats_allocator_as_allocator(hyle_stats_allocator *s);
/* Writes a point-in-time POD snapshot. A NULL s writes zeros; a NULL out is a
 * no-op. Using an out parameter avoids platform-specific large-struct return
 * conventions in FFI generators. */
HYLE_MEM_C_API void
hyle_stats_allocator_snapshot(const hyle_stats_allocator *s,
                              hyle_alloc_stats *out);
/* Reset requires no live allocation through this wrapper; it also clears the
 * optional leak inventory. */
HYLE_MEM_C_API void hyle_stats_allocator_reset(hyle_stats_allocator *s);
HYLE_MEM_C_API int
hyle_stats_allocator_balanced(const hyle_stats_allocator *s);
/* Copies at most capacity entries and returns the total outstanding count.
 * The diagnostic copy may obtain a temporary array from the original backing;
 * when that allocation fails, out is untouched and last-error reports OOM.
 * When leak tracking is disabled (or s==NULL), returns 0. */
HYLE_MEM_C_API size_t
hyle_stats_allocator_copy_leaks(const hyle_stats_allocator *s,
                                hyle_mem_leak *out, size_t capacity);

static inline hyle_stats_allocator *
hyle_stats_allocator_create_default(int track_leaks) {
  return hyle_stats_allocator_create(hyle_system_allocator(), track_leaks);
}

/* ========================================================================== */
/* Region — owned growable bump allocation + bulk reset */
/* ========================================================================== */

typedef struct hyle_region_options {
  size_t struct_size;
  /* Ordinary-block total backing-byte targets, including internal metadata
   * and alignment padding. One user allocation may force a larger block. */
  size_t initial_block_size;
  size_t max_block_size;
  size_t large_allocation_threshold;
  size_t size_limit;
} hyle_region_options;

#define HYLE_REGION_OPTIONS_INIT                                               \
  {sizeof(hyle_region_options), 64u * 1024u, 4u * 1024u * 1024u,              \
   256u * 1024u,                                                              \
   SIZE_MAX}

typedef struct hyle_region_o hyle_region;

/* `options==NULL` selects HYLE_REGION_OPTIONS_INIT. Unknown trailing fields are
 * ignored; struct_size must cover every field through size_limit for this ABI
 * version. */
HYLE_MEM_C_API hyle_region *
hyle_region_create(hyle_allocator backing, const hyle_region_options *options);
HYLE_MEM_C_API void hyle_region_destroy(hyle_region *r);
HYLE_MEM_C_API void *hyle_region_allocate(hyle_region *r, size_t bytes,
                                          size_t alignment);
/* clear retains ordinary blocks; release returns every block to backing. */
HYLE_MEM_C_API void hyle_region_clear(hyle_region *r);
HYLE_MEM_C_API void hyle_region_release(hyle_region *r);
HYLE_MEM_C_API size_t hyle_region_used(const hyle_region *r);
HYLE_MEM_C_API size_t hyle_region_reserved(const hyle_region *r);
HYLE_MEM_C_API size_t hyle_region_peak_used(const hyle_region *r);
HYLE_MEM_C_API size_t hyle_region_size_limit(const hyle_region *r);
HYLE_MEM_C_API hyle_allocator hyle_region_as_allocator(hyle_region *r);

static inline hyle_region *
hyle_region_create_default(const hyle_region_options *options) {
  return hyle_region_create(hyle_system_allocator(), options);
}

/* ========================================================================== */
/* FixedRegion — non-owning bump allocation over caller storage */
/* ========================================================================== */

/* A FixedRegion is exactly a three-pointer bump cursor, so C owns it by value:
 * no opaque 128-byte control buffer, hidden allocation, or destroy step. Treat
 * `state` as private and initialize it with hyle_fixed_region_init(). */
typedef struct hyle_fixed_region {
  uintptr_t state[3];
} hyle_fixed_region;

/* Initializes r over caller-owned storage. Returns 1 on success, 0 with
 * last-error on invalid/null/overflowing storage. */
HYLE_MEM_C_API int hyle_fixed_region_init(hyle_fixed_region *r, void *storage,
                                          size_t size);
HYLE_MEM_C_API void *hyle_fixed_region_allocate(hyle_fixed_region *r,
                                                size_t bytes, size_t alignment);
HYLE_MEM_C_API void hyle_fixed_region_clear(hyle_fixed_region *r);
HYLE_MEM_C_API size_t hyle_fixed_region_used(const hyle_fixed_region *r);
HYLE_MEM_C_API size_t hyle_fixed_region_capacity(const hyle_fixed_region *r);
HYLE_MEM_C_API size_t hyle_fixed_region_remaining(const hyle_fixed_region *r);
HYLE_MEM_C_API hyle_allocator
hyle_fixed_region_as_allocator(hyle_fixed_region *r);

/* ========================================================================== */
/* FixedPool — variable-size segregated-fit allocator over caller storage */
/* ========================================================================== */

typedef struct hyle_fixed_pool_o hyle_fixed_pool;

#define HYLE_FIXED_POOL_CONTROL_BYTES 1024u
typedef union hyle_fixed_pool_storage {
  void *alignment;
  unsigned char bytes[HYLE_FIXED_POOL_CONTROL_BYTES];
} hyle_fixed_pool_storage;

/* Strict no-heap construction: control is caller-owned and allocation metadata
 * lives inside payload storage. destroy() frees neither buffer. */
HYLE_MEM_C_API hyle_fixed_pool *
hyle_fixed_pool_init(hyle_fixed_pool_storage *control, void *storage,
                     size_t size);
/* Convenience factory: no payload fallback; create() allocates the opaque
 * control from the supplied allocator. Individual allocations can be freed. */
HYLE_MEM_C_API hyle_fixed_pool *
hyle_fixed_pool_create(hyle_allocator control_allocator, void *storage,
                       size_t size);
HYLE_MEM_C_API void hyle_fixed_pool_destroy(hyle_fixed_pool *p);
HYLE_MEM_C_API void *hyle_fixed_pool_allocate(hyle_fixed_pool *p,
                                              size_t bytes, size_t alignment);
HYLE_MEM_C_API void hyle_fixed_pool_deallocate(hyle_fixed_pool *p, void *ptr);
HYLE_MEM_C_API void *hyle_fixed_pool_reallocate(hyle_fixed_pool *p, void *ptr,
                                                size_t old_size,
                                                size_t new_size,
                                                size_t alignment);
HYLE_MEM_C_API void hyle_fixed_pool_reset(hyle_fixed_pool *p);
HYLE_MEM_C_API size_t hyle_fixed_pool_used(const hyle_fixed_pool *p);
HYLE_MEM_C_API size_t hyle_fixed_pool_peak(const hyle_fixed_pool *p);
HYLE_MEM_C_API size_t hyle_fixed_pool_capacity(const hyle_fixed_pool *p);
HYLE_MEM_C_API hyle_allocator
hyle_fixed_pool_as_allocator(hyle_fixed_pool *p);

static inline hyle_fixed_pool *hyle_fixed_pool_create_default(void *storage,
                                                              size_t size) {
  return hyle_fixed_pool_create(hyle_system_allocator(), storage, size);
}

/* ========================================================================== */
/* SharedPool — sharded fixed-size block pool (HARD capacity) */
/* ========================================================================== */
/* In normal tagged mode, the successful home-shard path and each deallocate use
 * one CAS; allocate scans other shards when needed. A saturated tag safely
 * falls back to a per-shard serialized path rather than wrapping. No block is
 * parked in TLS, so a completed free is immediately visible (no
 * cache-induced false exhaustion).
 * The safe default for a bounded cross-thread quota. For a same-thread-churn
 * hot spot with an over-provisioned pool, attach an explicit
 * hyle_shared_pool_cache below (soft visibility, faster). */

typedef struct hyle_shared_pool_o hyle_shared_pool;

/* `block_count` blocks of `block_size` bytes each, aligned to `alignment`
 * (0 => default). The opaque control, slab and link arrays all come from
 * `backing`. Invalid/overflowing layout or a resource failure returns NULL and
 * sets hyle_mem_last_error; C input validation does not expose C++ constructor
 * assertions. */
HYLE_MEM_C_API hyle_shared_pool *
hyle_shared_pool_create(hyle_allocator backing, size_t block_size,
                        uint32_t block_count, size_t alignment);
/* Contract: no live block and all concurrent pool access is quiesced. */
HYLE_MEM_C_API void hyle_shared_pool_destroy(hyle_shared_pool *p);
/* Pop a block; NULL after this call observes every shard empty (none hide in
 * TLS). */
HYLE_MEM_C_API void *hyle_shared_pool_allocate(hyle_shared_pool *p);
/* Return a block from allocate(); NULL is a no-op. Immediately visible to all.
 */
HYLE_MEM_C_API void hyle_shared_pool_deallocate(hyle_shared_pool *p,
                                                void *ptr);
/* Return EVERY block to the free pool. Contract: NO block may be live. */
HYLE_MEM_C_API void hyle_shared_pool_reset(hyle_shared_pool *p);
HYLE_MEM_C_API size_t hyle_shared_pool_block_size(const hyle_shared_pool *p);
HYLE_MEM_C_API uint32_t hyle_shared_pool_capacity(const hyle_shared_pool *p);
/* A vtable view of the pool. realloc resizes in place while new_size fits the
 * fixed block, frees on new_size==0, and otherwise fails without freeing.
 * The handle stays owned by the pool; valid until destroy. */
HYLE_MEM_C_API hyle_allocator
hyle_shared_pool_as_allocator(hyle_shared_pool *p);

static inline hyle_shared_pool *
hyle_shared_pool_create_default(size_t block_size, uint32_t block_count,
                                size_t alignment) {
  return hyle_shared_pool_create(hyle_system_allocator(), block_size,
                                 block_count, alignment);
}

/* ========================================================================== */
/* SharedPoolCache — explicit owner-local cache over SharedPool (SOFT view) */
/* ========================================================================== */
/* The hard SharedPool owns all blocks. Each worker may attach one cache with
 * an explicit lifetime, local capacity and transfer batch. Cached free blocks
 * are invisible through the shared pool until publish/destroy, so use the hard
 * pool directly for strict capacity admission. `pool` must outlive the cache
 * and all blocks checked out through it. One cache is owner-thread-only;
 * distinct caches over the same pool may run concurrently. */

typedef struct hyle_shared_pool_cache_o hyle_shared_pool_cache;

/* `metadata` owns the opaque handle and uint32_t local-index array.
 * transfer_batch==0 selects the default (32, clamped to capacity). */
HYLE_MEM_C_API hyle_shared_pool_cache *
hyle_shared_pool_cache_create(hyle_shared_pool *pool, hyle_allocator metadata,
                              uint32_t capacity, uint32_t transfer_batch);
/* Publishes locally free blocks, then releases metadata. No user-held block. */
HYLE_MEM_C_API void hyle_shared_pool_cache_destroy(hyle_shared_pool_cache *p);
HYLE_MEM_C_API void *
hyle_shared_pool_cache_allocate(hyle_shared_pool_cache *p);
HYLE_MEM_C_API void
hyle_shared_pool_cache_deallocate(hyle_shared_pool_cache *p, void *ptr);
HYLE_MEM_C_API void hyle_shared_pool_cache_publish(hyle_shared_pool_cache *p);
HYLE_MEM_C_API uint32_t
hyle_shared_pool_cache_capacity(const hyle_shared_pool_cache *p);
HYLE_MEM_C_API uint32_t
hyle_shared_pool_cache_cached(const hyle_shared_pool_cache *p);
HYLE_MEM_C_API uint32_t
hyle_shared_pool_cache_transfer_batch(const hyle_shared_pool_cache *p);
HYLE_MEM_C_API size_t
hyle_shared_pool_cache_block_size(const hyle_shared_pool_cache *p);
HYLE_MEM_C_API hyle_allocator
hyle_shared_pool_cache_as_allocator(hyle_shared_pool_cache *p);

static inline hyle_shared_pool_cache *
hyle_shared_pool_cache_create_default(hyle_shared_pool *pool,
                                      uint32_t capacity,
                                      uint32_t transfer_batch) {
  return hyle_shared_pool_cache_create(pool, hyle_system_allocator(), capacity,
                                       transfer_batch);
}

/* ---- typed object-pool convenience (header-inline; zero new ABI) ------------
 * A SharedPool with block_size == sizeof(T) IS a fixed-capacity object pool of
 * T -- these macros just fill sizeof/alignof and cast, so C code stops
 * hand-writing them. This mirrors ggml's model (and HYLE's own C++
 * ObjectPool<T>): the allocator hands out RAW O(1) slots; there is no
 * constructor callback on the allocate hot path (a cross-DLL indirect init()
 * call would tax every pop). C has no ctor/dtor, so initialize the returned
 * slot yourself and clean it up before release. For real construct/destroy
 * semantics use the C++ hyle::mem::ObjectPool<T>. The unqualified macros bind
 * to the HARD SharedPool (safe default); an explicit cache can be layered on
 * the returned pool with the *_CACHE_* macros below. */
#define HYLE_OBJECT_POOL_CREATE(backing, T, count)                             \
  hyle_shared_pool_create((backing), sizeof(T), (count), HYLE_ALIGNOF(T))
/* Acquire a raw T* slot (NULL when the pool is exhausted); release returns it.
 */
#define HYLE_OBJECT_POOL_ACQUIRE(pool, T) ((T *)hyle_shared_pool_allocate(pool))
#define HYLE_OBJECT_POOL_RELEASE(pool, p)                                      \
  hyle_shared_pool_deallocate((pool), (p))
#define HYLE_OBJECT_POOL_CREATE_DEFAULT(T, count)                              \
  hyle_shared_pool_create_default(sizeof(T), (count), HYLE_ALIGNOF(T))

/* Explicit cached typed raw-slot counterpart over an existing hard pool. */
#define HYLE_OBJECT_POOL_CACHE_CREATE(pool, metadata, capacity, batch)         \
  hyle_shared_pool_cache_create((pool), (metadata), (capacity), (batch))
#define HYLE_OBJECT_POOL_CACHE_ACQUIRE(cache, T)                               \
  ((T *)hyle_shared_pool_cache_allocate(cache))
#define HYLE_OBJECT_POOL_CACHE_RELEASE(cache, p)                               \
  hyle_shared_pool_cache_deallocate((cache), (p))
#define HYLE_OBJECT_POOL_CACHE_PUBLISH(cache)                                  \
  hyle_shared_pool_cache_publish((cache))
#define HYLE_OBJECT_POOL_CACHE_CREATE_DEFAULT(pool, capacity, batch)           \
  hyle_shared_pool_cache_create_default((pool), (capacity), (batch))

/* ========================================================================== */
/* StackAllocator — LIFO bump over one owned slab */
/* ========================================================================== */

typedef struct hyle_stack_allocator_o hyle_stack_allocator;

/* Owns a `size`-byte slab from `backing`; the opaque control uses the same
 * backing. Invalid backing, zero size, control allocation failure, or slab OOM
 * returns NULL and sets hyle_mem_last_error.
 */
HYLE_MEM_C_API hyle_stack_allocator *
hyle_stack_allocator_create(hyle_allocator backing, size_t size);
HYLE_MEM_C_API void hyle_stack_allocator_destroy(hyle_stack_allocator *s);
/* Bump `bytes` aligned to `alignment` (0 => default). NULL if it won't fit. */
HYLE_MEM_C_API void *hyle_stack_allocator_allocate(hyle_stack_allocator *s,
                                                   size_t bytes,
                                                   size_t alignment);
HYLE_MEM_C_API void hyle_stack_allocator_reset(hyle_stack_allocator *s);
/* Save the cursor; rewind() returns the slab to exactly this point (LIFO). */
HYLE_MEM_C_API void *hyle_stack_allocator_mark(const hyle_stack_allocator *s);
HYLE_MEM_C_API void hyle_stack_allocator_rewind(hyle_stack_allocator *s,
                                                void *marker);
HYLE_MEM_C_API size_t
hyle_stack_allocator_used(const hyle_stack_allocator *s);
HYLE_MEM_C_API size_t
hyle_stack_allocator_capacity(const hyle_stack_allocator *s);
HYLE_MEM_C_API size_t
hyle_stack_allocator_remaining(const hyle_stack_allocator *s);
/* Vtable view (alloc=bump, free=no-op, realloc=bump+copy). Owned by s. */
HYLE_MEM_C_API hyle_allocator
hyle_stack_allocator_as_allocator(hyle_stack_allocator *s);

static inline hyle_stack_allocator *
hyle_stack_allocator_create_default(size_t size) {
  return hyle_stack_allocator_create(hyle_system_allocator(), size);
}

/* ========================================================================== */
/* PooledRegion — bump over ONE borrowed SharedPool block */
/* ========================================================================== */

typedef struct hyle_pooled_region_o hyle_pooled_region;

/* Borrows one block from `pool` (a SharedPool, which must outlive the region).
 * Returns NULL if `pool` is NULL, the control allocation fails, or the pool has
 * no free block (exhausted) -- the exhausted case sets
 * hyle_mem_last_error("...pool exhausted..."). Pool exhaustion is a graceful
 * resource failure (NULL), not a contract violation. */
HYLE_MEM_C_API hyle_pooled_region *
hyle_pooled_region_create(hyle_shared_pool *pool);
HYLE_MEM_C_API void
hyle_pooled_region_destroy(hyle_pooled_region *r); /* returns block to pool */
HYLE_MEM_C_API void *hyle_pooled_region_allocate(hyle_pooled_region *r,
                                                 size_t bytes,
                                                 size_t alignment);
HYLE_MEM_C_API void hyle_pooled_region_clear(hyle_pooled_region *r);
HYLE_MEM_C_API size_t hyle_pooled_region_used(const hyle_pooled_region *r);
HYLE_MEM_C_API size_t
hyle_pooled_region_remaining(const hyle_pooled_region *r);
HYLE_MEM_C_API size_t
hyle_pooled_region_block_size(const hyle_pooled_region *r);
/* Vtable view (alloc=bump, free=no-op, realloc=bump+copy). Owned by r. */
HYLE_MEM_C_API hyle_allocator
hyle_pooled_region_as_allocator(hyle_pooled_region *r);

/* ========================================================================== */
/* GrowablePool — single-threaded, unbounded object pool of raw slots */
/* ========================================================================== */

typedef struct hyle_growable_pool_o hyle_growable_pool;

/* A pool of raw `slot_size`-byte slots (each aligned to `slot_align`, 0 => the
 * default alignment) that GROWS on demand -- unlike hyle_shared_pool_* it is
 * never "full" until the backing itself OOMs. Single-threaded (no atomics, so
 * faster when uncontended; do NOT share across threads). `initial_slots` is the
 * first block's slot count (0 => 1); blocks then double geometrically. Blocks
 * come from `backing`. Invalid/overflowing layout, invalid backing, or control
 * allocation failure returns NULL and sets hyle_mem_last_error. A grow that
 * later hits backing OOM is likewise graceful -- acquire() returns NULL.
 *
 * Like the fixed object-pool macro (HYLE_OBJECT_POOL_CREATE), acquire() hands
 * out RAW uninitialized storage and release() runs NO destructor: the C caller
 * inits the slot and cleans it up before releasing. (The ctor/dtor-running C++
 * hyle::mem::GrowablePool<T> is a C++-only convenience, not a gap here.) */
HYLE_MEM_C_API hyle_growable_pool *
hyle_growable_pool_create(hyle_allocator backing, size_t slot_size,
                          size_t slot_align, uint32_t initial_slots);
HYLE_MEM_C_API void hyle_growable_pool_destroy(hyle_growable_pool *p);
/* Acquire a raw uninitialized slot (a freed one is reused, else the pool
 * grows). */
HYLE_MEM_C_API void *hyle_growable_pool_acquire(hyle_growable_pool *p);
/* Return a slot from acquire() to the free-list; runs no destructor. NULL
 * no-op. */
HYLE_MEM_C_API void hyle_growable_pool_release(hyle_growable_pool *p,
                                               void *slot);
/* Hand every block back to the backing (trim). Contract: NO slot may be live.
 */
HYLE_MEM_C_API void hyle_growable_pool_reset(hyle_growable_pool *p);
/* Total slots across all blocks currently held (grows geometrically). */
HYLE_MEM_C_API size_t
hyle_growable_pool_capacity(const hyle_growable_pool *p);
/* Live (acquired, not yet released) slot count. */
HYLE_MEM_C_API size_t hyle_growable_pool_live(const hyle_growable_pool *p);
/* Actual aligned byte stride between raw slots; may exceed slot_size. */
HYLE_MEM_C_API size_t
hyle_growable_pool_slot_stride(const hyle_growable_pool *p);

static inline hyle_growable_pool *
hyle_growable_pool_create_default(size_t slot_size, size_t slot_align,
                                  uint32_t initial_slots) {
  return hyle_growable_pool_create(hyle_system_allocator(), slot_size,
                                   slot_align, initial_slots);
}

/* ---- typed convenience (header-inline; zero new ABI) ------------------------
 * A GrowablePool with slot_size == sizeof(T) is a growable raw-slot pool of T.
 * Same raw-slot model as HYLE_OBJECT_POOL_*: no ctor on acquire, no dtor on
 * release -- initialize the returned slot and clean it up yourself. For real
 * construct/destroy semantics, use the C++ hyle::mem::GrowablePool<T>. */
#define HYLE_GROWABLE_POOL_CREATE(backing, T, initial_slots)                   \
  hyle_growable_pool_create((backing), sizeof(T), HYLE_ALIGNOF(T),             \
                            (initial_slots))
/* Acquire a raw T* slot (uninitialized); release returns it. */
#define HYLE_GROWABLE_POOL_ACQUIRE(pool, T)                                    \
  ((T *)hyle_growable_pool_acquire(pool))
#define HYLE_GROWABLE_POOL_RELEASE(pool, p)                                    \
  hyle_growable_pool_release((pool), (p))
#define HYLE_GROWABLE_POOL_CREATE_DEFAULT(T, initial_slots)                    \
  hyle_growable_pool_create_default(sizeof(T), HYLE_ALIGNOF(T),                \
                                    (initial_slots))

/* ========================================================================== */
/* HugeRegion — huge-page / NUMA-placed span */
/* ========================================================================== */

/* HugeFlags bits (soft hints; correctness never depends on them). Named enum so
 * cross-language FFI tools surface the flags as typed constants. */
typedef enum hyle_huge_flags {
  HYLE_HUGE_NONE = 0,
  HYLE_HUGE_THP_HINT = 1u << 0,  /* MADV_HUGEPAGE hint (Linux) */
  HYLE_HUGE_PREFAULT = 1u << 1,  /* touch every page up front */
  HYLE_HUGE_NUMA_BIND = 1u << 2, /* hard-bind to node (else preferred) */
  HYLE_HUGE_NUMA_INTERLEAVE = 1u
                              << 3 /* stripe across all nodes (ignores node) */
} hyle_huge_flags;

/* No NUMA constraint (matches HYLE_NUMA_ANY in c/platform.h). Named enum for
 * the same FFI discoverability reason. */
typedef enum hyle_mem_numa_node { HYLE_MEM_NUMA_ANY = -1 } hyle_mem_numa_node;

typedef struct hyle_huge_region_o hyle_huge_region;

/* Reserve+commit `bytes` (large-page aligned), placed per flags/node.
 * NULL on failure (see hyle_mem_last_error via platform's hyle_vm_last_error).
 */
HYLE_MEM_C_API hyle_huge_region *
hyle_huge_region_open(size_t bytes, uint32_t flags, int numa_node);
/* Prefer the calling thread's home node. */
HYLE_MEM_C_API hyle_huge_region *hyle_huge_region_open_local(size_t bytes,
                                                             uint32_t flags);
/* Interleave across all nodes + THP (shared multi-MB arena). */
HYLE_MEM_C_API hyle_huge_region *
hyle_huge_region_open_interleaved(size_t bytes);
HYLE_MEM_C_API void hyle_huge_region_destroy(hyle_huge_region *r);
HYLE_MEM_C_API void *hyle_huge_region_data(hyle_huge_region *r);
HYLE_MEM_C_API size_t hyle_huge_region_size(const hyle_huge_region *r);
/* Requested versus OS-accepted policy. THP accepted means the advice call
 * succeeded; physical page promotion remains kernel-controlled and may happen
 * later. */
HYLE_MEM_C_API uint32_t
hyle_huge_region_requested_flags(const hyle_huge_region *r);
HYLE_MEM_C_API uint32_t
hyle_huge_region_accepted_flags(const hyle_huge_region *r);
HYLE_MEM_C_API int
hyle_huge_region_requested_numa_node(const hyle_huge_region *r);
HYLE_MEM_C_API int
hyle_huge_region_numa_placement_applied(const hyle_huge_region *r);

/* ========================================================================== */
/* Diagnostics */
/* ========================================================================== */

/* Machine-readable reason from the last factory/policy operation that updated
 * the mem diagnostic channel on this thread. Pointer-returning construction
 * APIs keep their direct NULL path; this errno-style code distinguishes
 * expected pool-borrow exhaustion from OOM and bad input without parsing a
 * string. Hot allocate/free/pop paths deliberately do not write TLS merely to
 * clear this code; their documented NULL capacity/OOM result remains direct. */
typedef enum hyle_mem_error {
  HYLE_MEM_ERROR_NONE = 0,
  HYLE_MEM_ERROR_INVALID_ARGUMENT = 1,
  HYLE_MEM_ERROR_OUT_OF_MEMORY = 2,
  HYLE_MEM_ERROR_EXHAUSTED = 3,
  HYLE_MEM_ERROR_PLATFORM = 4
} hyle_mem_error;

HYLE_MEM_C_API hyle_mem_error hyle_mem_last_error_code(void);

/* Last error from a mem C operation on THIS thread, or "" if none. This is an
 * independent mem channel: mem factories do not overwrite hyle_vm_last_error;
 * HugeRegion temporarily captures and copies its VM cause here, then restores
 * the platform channel. Valid until the next mem operation that updates this
 * thread's channel. */
HYLE_MEM_C_API const char *hyle_mem_last_error(void);

/* ========================================================================== */
/* Process-wide mimalloc policy (free functions) */
/* ========================================================================== */

/* THP hint + large OS pages for the system allocator's arenas (call at start).
 */
HYLE_MEM_C_API void hyle_enable_allocator_hugepages(int allow_thp,
                                                    int allow_large_os_pages);
/* max_nodes==0 => all NUMA nodes (mimalloc default). */
HYLE_MEM_C_API void hyle_enable_allocator_numa(unsigned max_nodes);
/* Reserve `pages` contiguous 1GiB huge OS pages into mimalloc's arenas.
 * numa_node==HYLE_MEM_NUMA_ANY => interleave; timeout_ms==0 => mimalloc
 * default. Returns 1 on success (or pages==0); soft-fails to 0 (no privilege /
 * no RAM). */
HYLE_MEM_C_API int hyle_reserve_huge_1g_pages(size_t pages, int numa_node,
                                              unsigned timeout_ms);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* HYLE_C_MEM_H_ */
