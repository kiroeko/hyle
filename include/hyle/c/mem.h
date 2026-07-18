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

/* Breaking change to hyle_mem_api layout bumps this. New fields append at the END
 * of hyle_mem_api so an older caller's prefix reads stay valid; only the version
 * gate rejects older numbers. */
#define HYLE_MEM_C_ABI_VERSION 1

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
 * (0 => default). Slab + link arrays come from `backing`. Returns NULL if the
 * backing is missing a fn pointer or the wrapper allocation fails. Contract
 * violations (block_size/block_count == 0) and slab OOM fail-fast (abort) in the
 * C++ ctor, same as a C++ caller -- not masked to a NULL here. */
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

/* ========================================================================== */
/* StackAllocator — LIFO bump over one owned slab                              */
/* ========================================================================== */

typedef struct hyle_stack_allocator_o hyle_stack_allocator_o;

/* Owns a `size`-byte slab from `backing`. NULL on failure. */
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

/* Borrows one block from `pool` (which must outlive the arena). NULL if pool is
 * NULL or the wrapper allocation fails. A pool with no free block fail-fasts
 * (abort) in the C++ Arena ctor -- borrowing is a contract, same as C++. */
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
/* HugeRegion — huge-page / NUMA-placed span                                   */
/* ========================================================================== */

/* HugeFlags bits (soft hints; correctness never depends on them). */
enum {
  HYLE_HUGE_NONE = 0,
  HYLE_HUGE_THP_HINT = 1u << 0,       /* MADV_HUGEPAGE hint (Linux) */
  HYLE_HUGE_PREFAULT = 1u << 1,       /* touch every page up front */
  HYLE_HUGE_NUMA_BIND = 1u << 2,      /* hard-bind to node (else preferred) */
  HYLE_HUGE_NUMA_INTERLEAVE = 1u << 3 /* stripe across all nodes (ignores node) */
};

/* No NUMA constraint (matches HYLE_NUMA_ANY in c/platform.h). */
#define HYLE_MEM_NUMA_ANY (-1)

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

/* ========================================================================== */
/* API table — every DLL-exported function (header-only inlines omitted)      */
/* ========================================================================== */

typedef struct hyle_mem_api {
  int abi_version;

  /* allocator */
  size_t (*default_alignment)(void);
  hyle_allocator (*system_allocator)(void);
  size_t (*system_usable_size)(const void*);
  size_t (*system_good_size)(size_t);

  /* memory arena */
  hyle_memory_arena_o* (*memory_arena_create)(hyle_allocator, size_t, uint32_t,
                                              size_t);
  void (*memory_arena_destroy)(hyle_memory_arena_o*);
  void* (*memory_arena_allocate)(hyle_memory_arena_o*);
  void (*memory_arena_deallocate)(hyle_memory_arena_o*, void*);
  void (*memory_arena_reset)(hyle_memory_arena_o*);
  size_t (*memory_arena_block_size)(const hyle_memory_arena_o*);
  uint32_t (*memory_arena_capacity)(const hyle_memory_arena_o*);
  hyle_allocator (*memory_arena_as_allocator)(hyle_memory_arena_o*);

  /* stack allocator */
  hyle_stack_allocator_o* (*stack_allocator_create)(hyle_allocator, size_t);
  void (*stack_allocator_destroy)(hyle_stack_allocator_o*);
  void* (*stack_allocator_allocate)(hyle_stack_allocator_o*, size_t, size_t);
  void (*stack_allocator_reset)(hyle_stack_allocator_o*);
  void* (*stack_allocator_mark)(const hyle_stack_allocator_o*);
  void (*stack_allocator_rewind)(hyle_stack_allocator_o*, void*);
  size_t (*stack_allocator_used)(const hyle_stack_allocator_o*);
  size_t (*stack_allocator_capacity)(const hyle_stack_allocator_o*);
  size_t (*stack_allocator_remaining)(const hyle_stack_allocator_o*);
  hyle_allocator (*stack_allocator_as_allocator)(hyle_stack_allocator_o*);

  /* arena */
  hyle_arena_o* (*arena_create)(hyle_memory_arena_o*);
  void (*arena_destroy)(hyle_arena_o*);
  void* (*arena_allocate)(hyle_arena_o*, size_t, size_t);
  void (*arena_reset)(hyle_arena_o*);
  size_t (*arena_used)(const hyle_arena_o*);
  size_t (*arena_remaining)(const hyle_arena_o*);
  size_t (*arena_block_size)(const hyle_arena_o*);
  hyle_allocator (*arena_as_allocator)(hyle_arena_o*);

  /* huge region */
  hyle_huge_region_o* (*huge_region_open)(size_t, uint32_t, int);
  hyle_huge_region_o* (*huge_region_open_local)(size_t, uint32_t);
  hyle_huge_region_o* (*huge_region_open_interleaved)(size_t);
  void (*huge_region_destroy)(hyle_huge_region_o*);
  int (*huge_region_valid)(const hyle_huge_region_o*);
  void* (*huge_region_data)(hyle_huge_region_o*);
  size_t (*huge_region_size)(const hyle_huge_region_o*);

  /* process-wide policy */
  void (*enable_allocator_hugepages)(int, int);
  void (*enable_allocator_numa)(unsigned);
  int (*reserve_huge_1g_pages)(size_t, int, unsigned);

  int (*c_abi_version)(void);
} hyle_mem_api;

HYLE_MEM_C_API int hyle_mem_api_get(hyle_mem_api* out, int expected_abi);
HYLE_MEM_C_API int hyle_mem_c_abi_version(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* HYLE_C_MEM_H_ */
