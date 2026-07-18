/* SPDX: same as HYLE.
 *
 * Pure C ABI for the HYLE platform module (close-01).
 *
 * Naming: every symbol is `hyle_*` / `HYLE_*`.
 * Do NOT include hyle/platform/ headers from C or cross-compiler FFI.
 *
 * Subsystems: wait/Parker, Thread, DynLib, virtual memory, fs/env, cpu features.
 * Docs: docs/guide/c-platform.md
 */
#ifndef HYLE_C_PLATFORM_H_
#define HYLE_C_PLATFORM_H_

#include <stddef.h>
#include <stdint.h>

#include "hyle/c/error.h" /* hyle_status for Thread spawn failures */

#ifdef __cplusplus
extern "C" {
#endif

/* ---- export (same module as C++ platform: HYLE_PLATFORM_BUILD) ------------- */
#if defined(HYLE_STATIC)
#  define HYLE_PLATFORM_C_API
#elif defined(_WIN32) || defined(__CYGWIN__)
#  if defined(HYLE_PLATFORM_BUILD)
#    define HYLE_PLATFORM_C_API __declspec(dllexport)
#  else
#    define HYLE_PLATFORM_C_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define HYLE_PLATFORM_C_API __attribute__((visibility("default")))
#else
#  define HYLE_PLATFORM_C_API
#endif

/* ========================================================================== */
/* wait / Parker                                                              */
/* ========================================================================== */

/* Low-level address wait (futex / WaitOnAddress / os_sync). Spurious wakes OK. */
HYLE_PLATFORM_C_API void hyle_wait_on_address(volatile uint32_t* addr,
                                              uint32_t expected);
/* Returns 0 on timeout, 1 otherwise (wake / value changed / spurious). */
HYLE_PLATFORM_C_API int hyle_wait_on_address_for(volatile uint32_t* addr,
                                                 uint32_t expected,
                                                 uint64_t timeout_ns);
HYLE_PLATFORM_C_API void hyle_wake_one(volatile uint32_t* addr);
HYLE_PLATFORM_C_API void hyle_wake_all(volatile uint32_t* addr);

/* Token-safe parker (Rust std model). Opaque; create/destroy pair. */
typedef struct hyle_parker_o hyle_parker_o;
HYLE_PLATFORM_C_API hyle_parker_o* hyle_parker_create(void);
HYLE_PLATFORM_C_API void hyle_parker_destroy(hyle_parker_o* p);
HYLE_PLATFORM_C_API void hyle_parker_park(hyle_parker_o* p);
HYLE_PLATFORM_C_API void hyle_parker_unpark(hyle_parker_o* p);
/* 1 = token consumed, 0 = timeout/spurious without token. */
HYLE_PLATFORM_C_API int hyle_parker_park_timeout(hyle_parker_o* p,
                                                 uint64_t timeout_ns);

/* ========================================================================== */
/* Thread                                                                     */
/* ========================================================================== */

typedef struct hyle_thread_o hyle_thread_o;
typedef void (*hyle_thread_entry_fn)(void* ctx);

/* Portable 5-level scheduling priority (v4). Maps per-OS; best-effort (raising
 * priority may need privilege on POSIX -> soft no-op). Matches C++ ThreadPriority. */
typedef enum hyle_thread_priority {
  HYLE_THREAD_PRIORITY_LOWEST = -2,
  HYLE_THREAD_PRIORITY_LOW = -1,
  HYLE_THREAD_PRIORITY_NORMAL = 0,
  HYLE_THREAD_PRIORITY_HIGH = 1,
  HYLE_THREAD_PRIORITY_HIGHEST = 2
} hyle_thread_priority;

typedef struct hyle_thread_opts {
  const char* name;          /* optional; NULL = none; truncated to OS limit */
  int has_stack_size;        /* 0/1 */
  size_t stack_size;         /* bytes if has_stack_size */
  int has_affinity_cpu;      /* 0/1 */
  unsigned affinity_cpu;     /* logical CPU if has_affinity_cpu */
  int has_priority;          /* 0/1 (v4) */
  int priority;              /* hyle_thread_priority if has_priority */
} hyle_thread_opts;

/* Zero-init helper: hyle_thread_opts o = HYLE_THREAD_OPTS_INIT; */
#define HYLE_THREAD_OPTS_INIT \
  { NULL, 0, 0, 0, 0, 0, 0 }

static inline hyle_thread_opts hyle_thread_opts_default(void) {
  hyle_thread_opts o = HYLE_THREAD_OPTS_INIT;
  return o;
}
static inline void hyle_thread_opts_set_name(hyle_thread_opts* o,
                                             const char* name) {
  if (o) o->name = name;
}
static inline void hyle_thread_opts_set_stack_size(hyle_thread_opts* o,
                                                   size_t bytes) {
  if (!o) return;
  o->has_stack_size = 1;
  o->stack_size = bytes;
}
static inline void hyle_thread_opts_set_affinity(hyle_thread_opts* o,
                                                 unsigned cpu) {
  if (!o) return;
  o->has_affinity_cpu = 1;
  o->affinity_cpu = cpu;
}
static inline void hyle_thread_opts_set_priority(hyle_thread_opts* o,
                                                 hyle_thread_priority p) {
  if (!o) return;
  o->has_priority = 1;
  o->priority = (int)p;
}

/* *out set on success. Returns hyle_status (free on error; OK has no heap). */
HYLE_PLATFORM_C_API hyle_status hyle_thread_spawn(hyle_thread_o** out,
                                                  hyle_thread_entry_fn entry,
                                                  void* ctx,
                                                  const hyle_thread_opts* opts);
/* Convenience spawn without filling hyle_thread_opts + its has_* flags: options
 * pass directly, sentinels instead of flags -- name==NULL => none; stack_size==0
 * => OS default; affinity_cpu<0 => no affinity; priority is a hyle_thread_priority
 * (HYLE_THREAD_PRIORITY_NORMAL==0 is the default). FFI-friendly: nothing to
 * zero-init, no has_* flag to remember. */
HYLE_PLATFORM_C_API hyle_status hyle_thread_spawn_opts(
    hyle_thread_o** out, hyle_thread_entry_fn entry, void* ctx, const char* name,
    size_t stack_size, int affinity_cpu, int priority);

/* Join and destroy the handle (joinable only once). */
HYLE_PLATFORM_C_API void hyle_thread_join(hyle_thread_o* t);
HYLE_PLATFORM_C_API int hyle_thread_joinable(const hyle_thread_o* t);

HYLE_PLATFORM_C_API unsigned hyle_hardware_concurrency(void);
HYLE_PLATFORM_C_API uint64_t hyle_this_thread_id(void);
HYLE_PLATFORM_C_API void hyle_this_thread_yield(void);
HYLE_PLATFORM_C_API void hyle_this_thread_set_name(const char* name);

/* ========================================================================== */
/* DynLib                                                                     */
/* ========================================================================== */

typedef struct hyle_dynlib_o hyle_dynlib_o;

HYLE_PLATFORM_C_API hyle_dynlib_o* hyle_dynlib_open(const char* path);
/* Already-loaded only; does not load if missing. */
HYLE_PLATFORM_C_API hyle_dynlib_o* hyle_dynlib_get(const char* path);
HYLE_PLATFORM_C_API hyle_dynlib_o* hyle_dynlib_current(void);
HYLE_PLATFORM_C_API void hyle_dynlib_close(hyle_dynlib_o* lib);
HYLE_PLATFORM_C_API int hyle_dynlib_valid(const hyle_dynlib_o* lib);
HYLE_PLATFORM_C_API void* hyle_dynlib_symbol(hyle_dynlib_o* lib, const char* name);
HYLE_PLATFORM_C_API const char* hyle_dynlib_last_error(void);
HYLE_PLATFORM_C_API const char* hyle_dynlib_extension(void); /* "dll"/"so"/"dylib" */

/* Open path and resolve symbol. On success *out_lib owns the library (caller
 * must hyle_dynlib_close) and *out_sym is the address. Returns 1 on success. */
HYLE_PLATFORM_C_API int hyle_dynlib_open_symbol(const char* path,
                                                const char* name,
                                                hyle_dynlib_o** out_lib,
                                                void** out_sym);

/* ========================================================================== */
/* Virtual memory                                                             */
/* ========================================================================== */

typedef enum hyle_protect {
  HYLE_PROTECT_NONE = 0,
  HYLE_PROTECT_READ = 1,
  HYLE_PROTECT_READ_WRITE = 2
} hyle_protect;

HYLE_PLATFORM_C_API size_t hyle_page_size(void);
/* Round up to page; 0 -> 0. */
static inline size_t hyle_page_align_up(size_t n) {
  size_t p = hyle_page_size();
  if (p == 0 || n == 0) return 0;
  return (n + (p - 1)) & ~(p - 1);
}

HYLE_PLATFORM_C_API void* hyle_vm_reserve(size_t size);
HYLE_PLATFORM_C_API int hyle_vm_commit(void* ptr, size_t size);   /* 1 ok */
HYLE_PLATFORM_C_API int hyle_vm_decommit(void* ptr, size_t size);
HYLE_PLATFORM_C_API int hyle_vm_release(void* base, size_t size);
HYLE_PLATFORM_C_API int hyle_vm_protect(void* ptr, size_t size, hyle_protect prot);
HYLE_PLATFORM_C_API const char* hyle_vm_last_error(void);

/* Reserve+commit RW; size rounded up to page. Pair with hyle_vm_free_rw,
 * passing the original request size. */
HYLE_PLATFORM_C_API void* hyle_vm_alloc_rw(size_t size);
/* Release a block from hyle_vm_alloc_rw (size is the original request; rounded
 * up to page internally). Exported (not inline) so cross-language FFI callers
 * release without reconstructing the page rounding themselves. Returns 1 ok. */
HYLE_PLATFORM_C_API int hyle_vm_free_rw(void* p, size_t size);

/* Set the thread-local vm error string (NULL clears). Mirrors the C++
 * vm_set_last_error; lets an FFI soft-fail path leave a breadcrumb. */
HYLE_PLATFORM_C_API void hyle_vm_set_last_error(const char* msg);

/* ========================================================================== */
/* Hugepage / THP hints (v3) — soft advice, never required for correctness.   */
/* ========================================================================== */

/* Large/huge page size in bytes (0 if unknown/unsupported). */
HYLE_PLATFORM_C_API size_t hyle_large_page_size(void);
/* Round up to large_page_size() (falls back to page on 0). n==0 -> 0. */
static inline size_t hyle_large_page_align_up(size_t n) {
  if (n == 0) return 0;
  size_t lp = hyle_large_page_size();
  size_t p = lp != 0 ? lp : hyle_page_size();
  if (p == 0) return 0;
  return (n + (p - 1)) & ~(p - 1);
}
/* MADV_HUGEPAGE / MADV_NOHUGEPAGE on Linux; false + vm_last_error elsewhere.
 * ptr must be page-aligned; size rounded up. Empty range -> 1 (no-op). */
HYLE_PLATFORM_C_API int hyle_vm_advise_hugepage(void* ptr, size_t size);
HYLE_PLATFORM_C_API int hyle_vm_advise_no_hugepage(void* ptr, size_t size);

/* ========================================================================== */
/* NUMA topology + placement (v3)                                             */
/* ========================================================================== */

/* Sentinel node value: "no NUMA constraint" for reserve/bind. A named enum (not
 * a bare #define) so cross-language FFI tools surface it as a real constant. */
typedef enum hyle_numa_node {
  HYLE_NUMA_ANY = -1
} hyle_numa_node;

typedef enum hyle_numa_policy {
  HYLE_NUMA_PREFERRED = 0, /* soft home node (never fails the process) */
  HYLE_NUMA_BIND = 1,      /* hard node (Linux MPOL_BIND; may fail) */
  HYLE_NUMA_INTERLEAVE = 2 /* stripe across all nodes (Linux) */
} hyle_numa_policy;

/* Node count (>=1) and the calling thread's home node (< count). */
HYLE_PLATFORM_C_API unsigned hyle_numa_node_count(void);
HYLE_PLATFORM_C_API unsigned hyle_numa_current_node(void);

/* Bind an existing page-aligned mapping. Empty range -> 1. Honest contract:
 * single-node box -> 1 (no-op); multi-node hard bind / interleave without an OS
 * API -> 0 (+ vm_last_error). See hyle::platform::vm_bind_numa. */
HYLE_PLATFORM_C_API int hyle_vm_bind_numa(void* ptr, size_t size, int node,
                                          hyle_numa_policy policy);
/* Stripe [ptr,ptr+size) across all nodes (alias for INTERLEAVE). */
HYLE_PLATFORM_C_API int hyle_vm_bind_numa_interleave(void* ptr, size_t size);
/* Reserve `size` (page-rounded) preferring `node` (HYLE_NUMA_ANY == plain
 * reserve). NULL on failure. Commit/release with the usual hyle_vm_* APIs. */
HYLE_PLATFORM_C_API void* hyle_vm_reserve_numa(size_t size, int node);

/* RAII region (opaque). destroy releases the reservation. */
typedef struct hyle_vm_region_o hyle_vm_region_o;
HYLE_PLATFORM_C_API hyle_vm_region_o* hyle_vm_region_reserve(size_t size);
HYLE_PLATFORM_C_API hyle_vm_region_o* hyle_vm_region_reserve_guarded(
    size_t usable);
/* Reserve preferring NUMA `node` (HYLE_NUMA_ANY == plain reserve), as a managed
 * region handle. NULL on failure. (v5) */
HYLE_PLATFORM_C_API hyle_vm_region_o* hyle_vm_region_reserve_numa(size_t size,
                                                                 int node);
/* Adopt an existing page-aligned reservation from hyle_vm_reserve* into a region
 * handle (destroy releases it). `size` must be the rounded size that pairs with
 * release. NULL on failure (e.g. base==NULL). Ownership transfers to the region.
 * (v5) */
HYLE_PLATFORM_C_API hyle_vm_region_o* hyle_vm_region_adopt(void* base,
                                                          size_t size);
HYLE_PLATFORM_C_API void hyle_vm_region_destroy(hyle_vm_region_o* r);
HYLE_PLATFORM_C_API int hyle_vm_region_valid(const hyle_vm_region_o* r);
HYLE_PLATFORM_C_API void* hyle_vm_region_data(hyle_vm_region_o* r);
HYLE_PLATFORM_C_API size_t hyle_vm_region_size(const hyle_vm_region_o* r);
HYLE_PLATFORM_C_API void* hyle_vm_region_usable(hyle_vm_region_o* r);
HYLE_PLATFORM_C_API size_t hyle_vm_region_usable_size(const hyle_vm_region_o* r);
HYLE_PLATFORM_C_API int hyle_vm_region_has_guard(const hyle_vm_region_o* r);
HYLE_PLATFORM_C_API int hyle_vm_region_commit(hyle_vm_region_o* r, size_t off,
                                              size_t len);
HYLE_PLATFORM_C_API int hyle_vm_region_commit_usable(hyle_vm_region_o* r);
HYLE_PLATFORM_C_API int hyle_vm_region_decommit(hyle_vm_region_o* r, size_t off,
                                                size_t len);
HYLE_PLATFORM_C_API int hyle_vm_region_protect(hyle_vm_region_o* r, size_t off,
                                               size_t len, hyle_protect prot);

/* ========================================================================== */
/* Filesystem + environment (buffer-fill / visitor; no hidden alloc)          */
/* ========================================================================== */

typedef struct hyle_file_info {
  int exists; /* 0/1 */
  int is_directory;
  uint64_t size;
  int64_t modified_time; /* seconds since epoch */
  int is_symlink;        /* v4: path itself is a symbolic link (lstat semantics) */
} hyle_file_info;

HYLE_PLATFORM_C_API hyle_file_info hyle_fs_stat(const char* path);
HYLE_PLATFORM_C_API int hyle_fs_exists(const char* path);
HYLE_PLATFORM_C_API int hyle_fs_is_directory(const char* path);
HYLE_PLATFORM_C_API uint64_t hyle_fs_file_size(const char* path);

HYLE_PLATFORM_C_API int hyle_fs_make_directory(const char* path);
HYLE_PLATFORM_C_API int hyle_fs_make_directory_recursive(const char* path);
HYLE_PLATFORM_C_API int hyle_fs_remove_directory(const char* path);
HYLE_PLATFORM_C_API int hyle_fs_remove_directory_recursive(const char* path);
HYLE_PLATFORM_C_API int hyle_fs_remove_file(const char* path);
HYLE_PLATFORM_C_API int hyle_fs_rename(const char* from, const char* to);
HYLE_PLATFORM_C_API int hyle_fs_copy_file(const char* from, const char* to);
HYLE_PLATFORM_C_API int hyle_fs_touch(const char* path);

/* snprintf-style buffer fill: returns full length; >=cap means truncated. */
HYLE_PLATFORM_C_API size_t hyle_fs_current_directory(char* dst, size_t cap);
HYLE_PLATFORM_C_API int hyle_fs_change_directory(const char* path);
HYLE_PLATFORM_C_API size_t hyle_fs_temp_directory(char* dst, size_t cap);
HYLE_PLATFORM_C_API size_t hyle_fs_home_directory(char* dst, size_t cap);

/* Env sentinel: hyle_env_get returns this (== (size_t)-1 / SIZE_MAX) when the
 * variable is unset. Stays a #define rather than an enum: a size_t-valued
 * sentinel does not fit C's int-typed enumerators before C23. */
#define HYLE_ENV_NOT_SET ((size_t)-1)
HYLE_PLATFORM_C_API size_t hyle_env_get(const char* name, char* dst, size_t cap);

/* Directory iterate: visit(name, is_dir, ctx) return 0 to stop. Returns 0 if
 * path cannot be opened. */
typedef int (*hyle_fs_visit_fn)(const char* name, int is_dir, void* ctx);
HYLE_PLATFORM_C_API int hyle_fs_directory_iterate(const char* path,
                                                  hyle_fs_visit_fn visit,
                                                  void* ctx);

/* ---- fs permissions / symlink / tempfile / env-write (v4) ------------------ */
/* Permission bits (low 9 of POSIX mode); Windows synthesizes from READONLY. */
HYLE_PLATFORM_C_API uint32_t hyle_fs_get_permissions(const char* path);
HYLE_PLATFORM_C_API int hyle_fs_set_permissions(const char* path, uint32_t mode);
HYLE_PLATFORM_C_API int hyle_fs_is_readonly(const char* path);
HYLE_PLATFORM_C_API int hyle_fs_set_readonly(const char* path, int readonly);
/* Symlinks. create_symlink needs privilege on Windows (soft-fails -> 0). */
HYLE_PLATFORM_C_API int hyle_fs_create_symlink(const char* target, const char* link);
HYLE_PLATFORM_C_API int hyle_fs_create_hard_link(const char* target, const char* link);
HYLE_PLATFORM_C_API int hyle_fs_is_symlink(const char* path);
HYLE_PLATFORM_C_API size_t hyle_fs_read_symlink(const char* path, char* dst,
                                                size_t cap);
/* Atomically create a temp file; returns an OPEN read-write handle as intptr_t
 * (-1 on failure, TOCTOU-safe idiom), path written to dst[cap]. Close it with
 * hyle_fs_close_fd. dir==NULL => system temp dir. */
HYLE_PLATFORM_C_API intptr_t hyle_fs_create_temp_file(const char* dir,
                                                      const char* prefix,
                                                      char* dst, size_t cap);
/* Close a raw fd/HANDLE (from create_temp_file). Safe on -1. */
HYLE_PLATFORM_C_API void hyle_fs_close_fd(intptr_t fd);
/* Environment write / unset (process-wide; not thread-safe vs concurrent reads). */
HYLE_PLATFORM_C_API int hyle_fs_set_env_var(const char* name, const char* value);
HYLE_PLATFORM_C_API int hyle_fs_unset_env_var(const char* name);

/* ========================================================================== */
/* Memory-mapped file (v4)                                                    */
/* ========================================================================== */

typedef enum hyle_map_mode {
  HYLE_MAP_READ_ONLY = 0,
  HYLE_MAP_READ_WRITE = 1,
  HYLE_MAP_COPY_ON_WRITE = 2
} hyle_map_mode;

/* Opaque mapping handle; destroy unmaps + closes. */
typedef struct hyle_mapped_file_o hyle_mapped_file_o;
/* Map [offset,offset+length) of `path` (length==0 => to EOF; offset page-aligned).
 * NULL on failure (see hyle_vm_last_error). */
HYLE_PLATFORM_C_API hyle_mapped_file_o* hyle_mapped_file_open(const char* path,
                                                             hyle_map_mode mode,
                                                             size_t offset,
                                                             size_t length);
HYLE_PLATFORM_C_API void hyle_mapped_file_destroy(hyle_mapped_file_o* m);
HYLE_PLATFORM_C_API int hyle_mapped_file_valid(const hyle_mapped_file_o* m);
HYLE_PLATFORM_C_API void* hyle_mapped_file_data(hyle_mapped_file_o* m);
HYLE_PLATFORM_C_API size_t hyle_mapped_file_size(const hyle_mapped_file_o* m);
HYLE_PLATFORM_C_API int hyle_mapped_file_flush(hyle_mapped_file_o* m);

/* ========================================================================== */
/* CPU features (1:1 with hyle::platform::CpuFeatures bools, as int 0/1)      */
/* ========================================================================== */

typedef struct hyle_cpu_features {
  /* x86 (0 on non-x86) */
  int sse2, ssse3, sse41, sse42;
  int avx, avx2, fma, f16c, lzcnt, bmi1, bmi2;
  int avx512f, avx512bw, avx512vl, avx512dq, avx512cd;
  int avx512vnni, avx_vnni, avx512bf16, avx512fp16, avx10_2;
  /* arm (0 on non-ARM) */
  int neon, sve, sve2, sve_aes, sve_pmull, sve_i8mm, sve_f32mm, sve_f64mm,
      sve_bf16, arm_i8mm, arm_bf16;
  /* RISC-V (0 on non-RISC-V) */
  int rvv, rvv_zvbb, rvv_zvbc, rvv_zvkned, rvv_zvfh, rvv_zvfhmin;
  /* extras for C convenience (not fields of C++ CpuFeatures) */
  unsigned logical_cpus; /* >=1 */
  uint32_t best_isa;     /* hyle::platform::Isa numeric value */
} hyle_cpu_features;

/* Coarse ISA tier values (match hyle::platform::Isa). Named enum (not an
 * anonymous one) so cross-language FFI tools surface it as a typed constant. */
typedef enum hyle_isa {
  HYLE_ISA_SCALAR = 0,
  HYLE_ISA_SSE2 = 10,
  HYLE_ISA_SSSE3 = 11,
  HYLE_ISA_SSE42 = 12,
  HYLE_ISA_AVX = 15,
  HYLE_ISA_AVX2 = 20,
  HYLE_ISA_AVX512 = 30,
  HYLE_ISA_NEON = 110,
  HYLE_ISA_SVE = 120,
  HYLE_ISA_RVV = 210
} hyle_isa;

HYLE_PLATFORM_C_API hyle_cpu_features hyle_cpu_features_get(void);
/* 1 if this process can run the given HYLE_ISA_* tier (same-arch compare). */
HYLE_PLATFORM_C_API int hyle_cpu_supports_isa(uint32_t isa);
/* "scalar", "avx2", ... — stable English name for logging. */
HYLE_PLATFORM_C_API const char* hyle_cpu_isa_name(uint32_t isa);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* HYLE_C_PLATFORM_H_ */
