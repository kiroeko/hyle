/* SPDX: same as HYLE.
 *
 * Pure C ABI for the HYLE mem module (close-01).
 *
 * Naming: every symbol is `hyle_*` / `HYLE_*`.
 * Do NOT include hyle/mem/ headers from C or cross-compiler FFI.
 *
 * Surface: the injectable Allocator vtable (system_allocator + usable/good size),
 * plus opaque handles for MemoryArena (sharded lock-free block pool), StackAllocator
 * (LIFO bump), Arena (bump over one pooled block), and HugeRegion (huge-page / NUMA
 * span), and the process-wide mimalloc hugepage/NUMA policy knobs.
 * Docs (1:1 with this header): docs/guide/c-mem.md
 */
#ifndef HYLE_C_MEM_H_
#define HYLE_C_MEM_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- export (same module as C++ mem: HYLE_MEM_BUILD) ----------------------- */
#if defined(HYLE_STATIC)
#  define HYLE_MEM_C_API
#elif defined(_WIN32) || defined(__CYGWIN__)
#  if defined(HYLE_MEM_BUILD)
#    define HYLE_MEM_C_API __declspec(dllexport)
#  else
#    define HYLE_MEM_C_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define HYLE_MEM_C_API __attribute__((visibility("default")))
#else
#  define HYLE_MEM_C_API
#endif

/* alignof spelling for the typed-pool macros below: C++ / C23 keyword, C11
 * `_Alignof`, else pass 0 (hyle_memory_arena_create normalizes 0 to the default
 * alignment -- always safe, never under-aligns). */
#if defined(__cplusplus)
#  define HYLE_ALIGNOF(T) alignof(T)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#  define HYLE_ALIGNOF(T) _Alignof(T)
#else
#  define HYLE_ALIGNOF(T) 0
#endif

/* ========================================================================== */
/* Allocator vtable — the injectable {alloc, free, realloc, state} handle.     */
/*                                                                            */
/* This is a NATURAL C ABI shape: three C function pointers + a state void*,   */
/* trivially copyable and cross-DLL stable (exactly what jemalloc / C++17 pmr  */
/* / cudaMalloc expose). It is layout-compatible with the C++ hyle::mem::      */
/* Allocator by design, but the bridge NEVER reinterpret-casts: it constructs  */
/* a C++ Allocator explicitly from these four fields (and vice versa).         */
/*                                                                            */
/* Contract (SIZED-FREE model, like Rust GlobalAlloc / jemalloc sdallocx):     */
/*   alloc  (state, size, alignment)                -> ptr (NULL on OOM/size==0)*/
/*   free   (state, ptr, size, alignment)           -> void (ptr==NULL no-op)  */
/*   realloc(state, ptr, old_size, new_size, align) -> ptr (the COLD path)     */
/* deallocate carries the size — do not pass a size that differs from the      */
/* allocating request (arenas/pools/stacks do not store per-pointer size).     */
/* ========================================================================== */

typedef struct hyle_allocator {
  void* (*alloc)(void* state, size_t size, size_t alignment);
  void (*free)(void* state, void* ptr, size_t size, size_t alignment);
  void* (*realloc)(void* state, void* ptr, size_t old_size, size_t new_size,
                   size_t alignment);
  void* state;
} hyle_allocator;

/* Default alignment when a caller passes 0 (matches alignof(max_align_t)); the
 * backing never returns less than this. */
HYLE_MEM_C_API size_t hyle_default_alignment(void);

/* Process-wide mimalloc-backed system allocator (the same handle C++ uses). */
HYLE_MEM_C_API hyle_allocator hyle_system_allocator(void);

/* Convenience: call through a handle without hand-dispatching the fn pointers.
 * alignment==0 => hyle_default_alignment(). Header-inline (no DLL hop). */
static inline void* hyle_alloc(const hyle_allocator* a, size_t size,
                               size_t alignment) {
  if (a == NULL || a->alloc == NULL) return NULL;
  return a->alloc(a->state, size,
                  alignment != 0 ? alignment : hyle_default_alignment());
}
static inline void hyle_free(const hyle_allocator* a, void* ptr, size_t size,
                             size_t alignment) {
  if (a == NULL || a->free == NULL) return;
  a->free(a->state, ptr, size,
          alignment != 0 ? alignment : hyle_default_alignment());
}
static inline void* hyle_realloc(const hyle_allocator* a, void* ptr,
                                 size_t old_size, size_t new_size,
                                 size_t alignment) {
  if (a == NULL || a->realloc == NULL) return NULL;
  return a->realloc(a->state, ptr, old_size, new_size,
                    alignment != 0 ? alignment : hyle_default_alignment());
}

/* ---- system usable/good size (SystemAllocator only) ------------------------ */
/* HYLE's Allocator is a sized-free model, so there is no vtable usable_size().
 * These two are SPECIFIC to hyle_system_allocator(), where the mimalloc backing
 * knows the true size-class. */
/* Actual usable bytes of a live system_allocator block (>= requested); NULL->0. */
HYLE_MEM_C_API size_t hyle_system_usable_size(const void* ptr);
/* Usable size a request of `size` WOULD get, without allocating (nallocx); 0->0. */
HYLE_MEM_C_API size_t hyle_system_good_size(size_t size);

/* ========================================================================== */
/* MemoryArena — sharded lock-free fixed-size block pool                       */
/* ========================================================================== */

typedef struct hyle_memory_arena_o hyle_memory_arena_o;

/* `block_count` blocks of `block_size` bytes each, aligned to `alignment`
 * (0 => default). Slab + link arrays come from `backing`. Returns NULL (and sets
 * hyle_mem_last_error) on a resource failure: invalid backing, wrapper-alloc
 * failure, or slab/link/heads OOM in the ctor (followup #2 arch B -- no longer
 * aborts). Contract violations (block_size/block_count == 0, or
 * block_size*block_count overflowing size_t) still fail-fast (abort) in the C++
 * ctor, same as a C++ caller. */
HYLE_MEM_C_API hyle_memory_arena_o* hyle_memory_arena_create(
    hyle_allocator backing, size_t block_size, uint32_t block_count,
    size_t alignment);
HYLE_MEM_C_API void hyle_memory_arena_destroy(hyle_memory_arena_o* a);
/* Pop a free block; NULL when the whole pool is exhausted. */
HYLE_MEM_C_API void* hyle_memory_arena_allocate(hyle_memory_arena_o* a);
/* Return a block from allocate(); NULL is a no-op. */
HYLE_MEM_C_API void hyle_memory_arena_deallocate(hyle_memory_arena_o* a, void* p);
/* Return EVERY block to the free pool. Contract: NO block may be live. */
HYLE_MEM_C_API void hyle_memory_arena_reset(hyle_memory_arena_o* a);
HYLE_MEM_C_API size_t hyle_memory_arena_block_size(const hyle_memory_arena_o* a);
HYLE_MEM_C_API uint32_t hyle_memory_arena_capacity(const hyle_memory_arena_o* a);
/* A vtable view of the pool (alloc=pop fixed block, free=push, realloc=NULL).
 * The handle stays owned by the arena; valid until destroy. */
HYLE_MEM_C_API hyle_allocator hyle_memory_arena_as_allocator(
    hyle_memory_arena_o* a);

/* ---- typed object-pool convenience (header-inline; zero new ABI) ------------
 * A MemoryArena with block_size == sizeof(T) IS a fixed-capacity object pool of
 * T -- these macros just fill sizeof/alignof and cast, so C code stops hand-
 * writing them. This mirrors ggml's model (and HYLE's own C++ ObjectPool<T>):
 * the allocator hands out RAW O(1) slots; there is no constructor callback on
 * the allocate hot path (a cross-DLL indirect init() call would tax every pop
 * -- exactly the dispatch cost the split alloc/free/realloc vtable avoids). C
 * has no ctor/dtor, so initialize the returned slot yourself and clean it up
 * before release. If you need real construct/destroy semantics (ctor runs on
 * create, ~T on destroy), use the C++ hyle::mem::ObjectPool<T> instead -- that
 * is a C++-only convenience, not a gap in this C ABI. */
#define HYLE_OBJECT_POOL_CREATE(backing, T, count) \
  hyle_memory_arena_create((backing), sizeof(T), (count), HYLE_ALIGNOF(T))
/* Acquire a raw T* slot (NULL when the pool is exhausted); release returns it. */
#define HYLE_OBJECT_POOL_ACQUIRE(pool, T) \
  ((T*)hyle_memory_arena_allocate(pool))
#define HYLE_OBJECT_POOL_RELEASE(pool, p) \
  hyle_memory_arena_deallocate((pool), (p))

/* ========================================================================== */
/* StackAllocator — LIFO bump over one owned slab                              */
/* ========================================================================== */

typedef struct hyle_stack_allocator_o hyle_stack_allocator_o;

/* Owns a `size`-byte slab from `backing`. Returns NULL (and sets
 * hyle_mem_last_error) on invalid backing, wrapper-alloc failure, or slab OOM
 * (followup #2 arch B -- no longer aborts); size==0 still fail-fasts (abort). */
HYLE_MEM_C_API hyle_stack_allocator_o* hyle_stack_allocator_create(
    hyle_allocator backing, size_t size);
HYLE_MEM_C_API void hyle_stack_allocator_destroy(hyle_stack_allocator_o* s);
/* Bump `bytes` aligned to `alignment` (0 => default). NULL if it won't fit. */
HYLE_MEM_C_API void* hyle_stack_allocator_allocate(hyle_stack_allocator_o* s,
                                                   size_t bytes,
                                                   size_t alignment);
HYLE_MEM_C_API void hyle_stack_allocator_reset(hyle_stack_allocator_o* s);
/* Save the cursor; rewind() returns the slab to exactly this point (LIFO). */
HYLE_MEM_C_API void* hyle_stack_allocator_mark(const hyle_stack_allocator_o* s);
HYLE_MEM_C_API void hyle_stack_allocator_rewind(hyle_stack_allocator_o* s,
                                                void* marker);
HYLE_MEM_C_API size_t hyle_stack_allocator_used(const hyle_stack_allocator_o* s);
HYLE_MEM_C_API size_t
hyle_stack_allocator_capacity(const hyle_stack_allocator_o* s);
HYLE_MEM_C_API size_t
hyle_stack_allocator_remaining(const hyle_stack_allocator_o* s);
/* Vtable view (alloc=bump, free=no-op, realloc=bump+copy). Owned by s. */
HYLE_MEM_C_API hyle_allocator
hyle_stack_allocator_as_allocator(hyle_stack_allocator_o* s);

/* ========================================================================== */
/* Arena — bump over ONE borrowed MemoryArena block                            */
/* ========================================================================== */

typedef struct hyle_arena_o hyle_arena_o;

/* Borrows one block from `pool` (which must outlive the arena). Returns NULL if
 * `pool` is NULL, the wrapper allocation fails, or the pool has no free block
 * (exhausted) -- the exhausted case sets hyle_mem_last_error("...pool
 * exhausted..."). Pool exhaustion is a graceful resource failure (NULL), not a
 * contract violation: the C++ Arena ctor does NOT abort on it. */
HYLE_MEM_C_API hyle_arena_o* hyle_arena_create(hyle_memory_arena_o* pool);
HYLE_MEM_C_API void hyle_arena_destroy(hyle_arena_o* a); /* returns block to pool */
HYLE_MEM_C_API void* hyle_arena_allocate(hyle_arena_o* a, size_t bytes,
                                         size_t alignment);
HYLE_MEM_C_API void hyle_arena_reset(hyle_arena_o* a);
HYLE_MEM_C_API size_t hyle_arena_used(const hyle_arena_o* a);
HYLE_MEM_C_API size_t hyle_arena_remaining(const hyle_arena_o* a);
HYLE_MEM_C_API size_t hyle_arena_block_size(const hyle_arena_o* a);
/* Vtable view (alloc=bump, free=no-op, realloc=bump+copy). Owned by a. */
HYLE_MEM_C_API hyle_allocator hyle_arena_as_allocator(hyle_arena_o* a);

/* ========================================================================== */
/* GrowablePool — single-threaded, unbounded object pool of raw slots          */
/* ========================================================================== */

typedef struct hyle_growable_pool_o hyle_growable_pool_o;

/* A pool of raw `slot_size`-byte slots (each aligned to `slot_align`, 0 => the
 * default alignment) that GROWS on demand -- unlike hyle_memory_arena_* it is
 * never "full" until the backing itself OOMs. Single-threaded (no atomics, so
 * faster when uncontended; do NOT share across threads). `initial_slots` is the
 * first block's slot count (0 => 1); blocks then double geometrically. Blocks
 * come from `backing`. Returns NULL (and sets hyle_mem_last_error) on invalid
 * backing or wrapper-alloc failure; slot_size==0 fail-fasts (abort) in the C++
 * core, and a grow that hits backing OOM aborts too (same as the other pools).
 *
 * Like the arena object pool (HYLE_OBJECT_POOL_CREATE), acquire() hands out RAW
 * uninitialized storage and release() runs NO destructor: the C caller inits the
 * slot and cleans it up before releasing. (The ctor/dtor-running C++
 * hyle::mem::GrowablePool<T> is a C++-only convenience, not a gap here.) */
HYLE_MEM_C_API hyle_growable_pool_o* hyle_growable_pool_create(
    hyle_allocator backing, size_t slot_size, size_t slot_align,
    uint32_t initial_slots);
HYLE_MEM_C_API void hyle_growable_pool_destroy(hyle_growable_pool_o* p);
/* Acquire a raw uninitialized slot (a freed one is reused, else the pool grows). */
HYLE_MEM_C_API void* hyle_growable_pool_acquire(hyle_growable_pool_o* p);
/* Return a slot from acquire() to the free-list; runs no destructor. NULL no-op. */
HYLE_MEM_C_API void hyle_growable_pool_release(hyle_growable_pool_o* p, void* slot);
/* Hand every block back to the backing (trim). Contract: NO slot may be live. */
HYLE_MEM_C_API void hyle_growable_pool_reset(hyle_growable_pool_o* p);
/* Total slots across all blocks currently held (grows geometrically). */
HYLE_MEM_C_API size_t hyle_growable_pool_capacity(const hyle_growable_pool_o* p);
/* Live (acquired, not yet released) slot count. */
HYLE_MEM_C_API size_t hyle_growable_pool_live(const hyle_growable_pool_o* p);

/* ---- typed convenience (header-inline; zero new ABI) ------------------------
 * A GrowablePool with slot_size == sizeof(T) is a growable raw-slot pool of T.
 * Same raw-slot model as HYLE_OBJECT_POOL_*: no ctor on acquire, no dtor on
 * release -- initialize the returned slot and clean it up yourself. For real
 * construct/destroy semantics, use the C++ hyle::mem::GrowablePool<T>. */
#define HYLE_GROWABLE_POOL_CREATE(backing, T, initial_slots) \
  hyle_growable_pool_create((backing), sizeof(T), HYLE_ALIGNOF(T), (initial_slots))
/* Acquire a raw T* slot (uninitialized); release returns it. */
#define HYLE_GROWABLE_POOL_ACQUIRE(pool, T) \
  ((T*)hyle_growable_pool_acquire(pool))
#define HYLE_GROWABLE_POOL_RELEASE(pool, p) \
  hyle_growable_pool_release((pool), (p))

/* ========================================================================== */
/* HugeRegion — huge-page / NUMA-placed span                                   */
/* ========================================================================== */

/* HugeFlags bits (soft hints; correctness never depends on them). Named enum so
 * cross-language FFI tools surface the flags as typed constants. */
typedef enum hyle_huge_flags {
  HYLE_HUGE_NONE = 0,
  HYLE_HUGE_THP_HINT = 1u << 0,       /* MADV_HUGEPAGE hint (Linux) */
  HYLE_HUGE_PREFAULT = 1u << 1,       /* touch every page up front */
  HYLE_HUGE_NUMA_BIND = 1u << 2,      /* hard-bind to node (else preferred) */
  HYLE_HUGE_NUMA_INTERLEAVE = 1u << 3 /* stripe across all nodes (ignores node) */
} hyle_huge_flags;

/* No NUMA constraint (matches HYLE_NUMA_ANY in c/platform.h). Named enum for the
 * same FFI discoverability reason. */
typedef enum hyle_mem_numa_node {
  HYLE_MEM_NUMA_ANY = -1
} hyle_mem_numa_node;

typedef struct hyle_huge_region_o hyle_huge_region_o;

/* Reserve+commit `bytes` (large-page aligned), placed per flags/node.
 * NULL on failure (see hyle_mem_last_error via platform's hyle_vm_last_error). */
HYLE_MEM_C_API hyle_huge_region_o* hyle_huge_region_open(size_t bytes,
                                                         uint32_t flags,
                                                         int numa_node);
/* Prefer the calling thread's home node. */
HYLE_MEM_C_API hyle_huge_region_o* hyle_huge_region_open_local(size_t bytes,
                                                               uint32_t flags);
/* Interleave across all nodes + THP (shared multi-MB arena). */
HYLE_MEM_C_API hyle_huge_region_o* hyle_huge_region_open_interleaved(
    size_t bytes);
HYLE_MEM_C_API void hyle_huge_region_destroy(hyle_huge_region_o* r);
HYLE_MEM_C_API int hyle_huge_region_valid(const hyle_huge_region_o* r);
HYLE_MEM_C_API void* hyle_huge_region_data(hyle_huge_region_o* r);
HYLE_MEM_C_API size_t hyle_huge_region_size(const hyle_huge_region_o* r);

/* ========================================================================== */
/* Diagnostics                                                                 */
/* ========================================================================== */

/* Last error from a mem VM-backed op (HugeRegion open / huge-page reservation)
 * on THIS thread, or "" if none. Forwards platform's thread-local vm_last_error
 * so a caller binding only hyle/c/mem.h can read why a hyle_huge_region_open*
 * returned NULL without also binding hyle/c/platform.h. Valid until the next
 * failing VM call on the same thread. */
HYLE_MEM_C_API const char* hyle_mem_last_error(void);

/* ========================================================================== */
/* Process-wide mimalloc policy (free functions)                               */
/* ========================================================================== */

/* THP hint + large OS pages for the system allocator's arenas (call at start). */
HYLE_MEM_C_API void hyle_enable_allocator_hugepages(int allow_thp,
                                                    int allow_large_os_pages);
/* max_nodes==0 => all NUMA nodes (mimalloc default). */
HYLE_MEM_C_API void hyle_enable_allocator_numa(unsigned max_nodes);
/* Reserve `pages` contiguous 1GiB huge OS pages into mimalloc's arenas.
 * numa_node==HYLE_MEM_NUMA_ANY => interleave; timeout_ms==0 => mimalloc default.
 * Returns 1 on success (or pages==0); soft-fails to 0 (no privilege / no RAM). */
HYLE_MEM_C_API int hyle_reserve_huge_1g_pages(size_t pages, int numa_node,
                                              unsigned timeout_ms);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* HYLE_C_MEM_H_ */
