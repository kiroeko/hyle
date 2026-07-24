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

#include "hyle/c/error.h" /* hyle_status for recoverable resource failures */

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

/* Low-level address wait (futex / WaitOnAddress / os_sync). Spurious wakes OK.
 * `addr` must be naturally aligned and every concurrent read/write of that word
 * must use atomic operations (C11 _Atomic, Interlocked, or an FFI equivalent);
 * volatile alone does not make a shared word race-free. */
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
/* Clears *out first; success transfers an opaque Parker to the caller. */
HYLE_PLATFORM_C_API hyle_status hyle_parker_create(hyle_parker_o** out);
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
  /* Set by HYLE_THREAD_OPTS_INIT/default(). Values smaller than the current
   * readable prefix (including zero) are rejected, preventing a newer library
   * from reading past an older FFI struct. */
  uint32_t struct_size;
  const char* name;          /* optional; NULL = none; truncated to OS limit */
  int has_stack_size;        /* 0/1 */
  size_t stack_size;         /* bytes if has_stack_size */
  int has_affinity_cpu;      /* 0/1 */
  unsigned affinity_cpu;     /* logical CPU if has_affinity_cpu */
  int has_priority;          /* 0/1 (v4) */
  int priority;              /* hyle_thread_priority if has_priority */
} hyle_thread_opts;

/* Default-init helper: hyle_thread_opts o = HYLE_THREAD_OPTS_INIT; */
#define HYLE_THREAD_OPTS_INIT \
  { (uint32_t)sizeof(hyle_thread_opts), NULL, 0, 0, 0, 0, 0, 0 }

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

/* Status-returning creators clear *out before work. On success the caller owns
 * *out and must hyle_dynlib_close it; status is always caller-freed. */
HYLE_PLATFORM_C_API hyle_status hyle_dynlib_open(hyle_dynlib_o** out,
                                                 const char* path);
/* Already-loaded only; does not load if missing. */
HYLE_PLATFORM_C_API hyle_status hyle_dynlib_get(hyle_dynlib_o** out,
                                                const char* path);
HYLE_PLATFORM_C_API hyle_status hyle_dynlib_current(hyle_dynlib_o** out);
HYLE_PLATFORM_C_API void hyle_dynlib_close(hyle_dynlib_o* lib);
HYLE_PLATFORM_C_API int hyle_dynlib_valid(const hyle_dynlib_o* lib);
/* Clears *out before lookup; a successful NULL symbol is distinguished from
 * failure by the returned Status. */
HYLE_PLATFORM_C_API hyle_status hyle_dynlib_symbol(void** out,
                                                   hyle_dynlib_o* lib,
                                                   const char* name);
HYLE_PLATFORM_C_API const char* hyle_dynlib_extension(void); /* "dll"/"so"/"dylib" */

/* Open path and resolve symbol. Both out values are cleared before work. On
 * success *out_lib owns the library (caller must hyle_dynlib_close) and
 * *out_sym is the address. */
HYLE_PLATFORM_C_API hyle_status hyle_dynlib_open_symbol(
    hyle_dynlib_o** out_lib, void** out_sym, const char* path,
    const char* name);

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
  if (n > SIZE_MAX - (p - 1)) return 0;
  return (n + (p - 1)) & ~(p - 1);
}

HYLE_PLATFORM_C_API hyle_status hyle_vm_reserve(void** out, size_t size);
HYLE_PLATFORM_C_API hyle_status hyle_vm_commit(void* ptr, size_t size);
HYLE_PLATFORM_C_API hyle_status hyle_vm_decommit(void* ptr, size_t size);
HYLE_PLATFORM_C_API hyle_status hyle_vm_release(void* base, size_t size);
HYLE_PLATFORM_C_API hyle_status hyle_vm_protect(void* ptr, size_t size,
                                                hyle_protect prot);

/* Reserve+commit RW; size rounded up to page. Pair with hyle_vm_free_rw,
 * passing the original request size. */
HYLE_PLATFORM_C_API hyle_status hyle_vm_alloc_rw(void** out, size_t size);
/* Release a block from hyle_vm_alloc_rw (size is the original request; rounded
 * up to page internally). Exported (not inline) so cross-language FFI callers
 * release without reconstructing the page rounding themselves. */
HYLE_PLATFORM_C_API hyle_status hyle_vm_free_rw(void* p, size_t size);

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
  if (n > SIZE_MAX - (p - 1)) return 0;
  return (n + (p - 1)) & ~(p - 1);
}
/* MADV_HUGEPAGE / MADV_NOHUGEPAGE on Linux. Unsupported platforms return
 * HYLE_STATUS_UNIMPLEMENTED. Empty ranges are successful no-ops. */
HYLE_PLATFORM_C_API hyle_status hyle_vm_advise_hugepage(void* ptr, size_t size);
HYLE_PLATFORM_C_API hyle_status hyle_vm_advise_no_hugepage(void* ptr,
                                                           size_t size);

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

typedef enum hyle_numa_placement {
  HYLE_NUMA_PLACEMENT_APPLIED = 0,
  HYLE_NUMA_PLACEMENT_NOT_NEEDED = 1,
  HYLE_NUMA_PLACEMENT_UNAVAILABLE = 2
} hyle_numa_placement;

/* Node count (>=1) and the calling thread's home node (< count). */
HYLE_PLATFORM_C_API unsigned hyle_numa_node_count(void);
HYLE_PLATFORM_C_API unsigned hyle_numa_current_node(void);

/* NUMA node owning logical CPU `cpu` (0 on failure / single-node / out-of-range). */
HYLE_PLATFORM_C_API unsigned hyle_numa_node_of_cpu(unsigned cpu);

/* Bind an existing page-aligned mapping. `out_placement` may be NULL when the
 * caller does not need to distinguish applied/no-op/soft fallback. */
HYLE_PLATFORM_C_API hyle_status hyle_vm_bind_numa(
    hyle_numa_placement* out_placement, void* ptr, size_t size, int node,
    hyle_numa_policy policy);
/* Stripe [ptr,ptr+size) across all nodes (alias for INTERLEAVE). */
HYLE_PLATFORM_C_API hyle_status hyle_vm_bind_numa_interleave(
    hyle_numa_placement* out_placement, void* ptr, size_t size);
/* Reserve `size` (page-rounded) preferring `node` (HYLE_NUMA_ANY == plain
 * reserve). `out_placement` may be NULL. */
HYLE_PLATFORM_C_API hyle_status hyle_vm_reserve_numa(
    void** out, hyle_numa_placement* out_placement, size_t size, int node);

/* RAII region (opaque). destroy releases the reservation. */
typedef struct hyle_vm_region_o hyle_vm_region_o;
HYLE_PLATFORM_C_API hyle_status hyle_vm_region_reserve(
    hyle_vm_region_o** out, size_t size);
HYLE_PLATFORM_C_API hyle_status hyle_vm_region_reserve_guarded(
    hyle_vm_region_o** out, size_t usable);
/* Reserve preferring NUMA `node` (HYLE_NUMA_ANY == plain reserve), as a managed
 * region handle. Returns Status and writes `*out` on success. */
HYLE_PLATFORM_C_API hyle_status hyle_vm_region_reserve_numa(
    hyle_vm_region_o** out, size_t size, int node);
/* Adopt an existing page-aligned reservation from hyle_vm_reserve* into a region
 * handle (destroy releases it). `size` must be the rounded size that pairs with
 * release. Ownership transfers only after Status success. */
HYLE_PLATFORM_C_API hyle_status hyle_vm_region_adopt(
    hyle_vm_region_o** out, void* base, size_t size);
HYLE_PLATFORM_C_API void hyle_vm_region_destroy(hyle_vm_region_o* r);
HYLE_PLATFORM_C_API int hyle_vm_region_valid(const hyle_vm_region_o* r);
HYLE_PLATFORM_C_API void* hyle_vm_region_data(hyle_vm_region_o* r);
HYLE_PLATFORM_C_API size_t hyle_vm_region_size(const hyle_vm_region_o* r);
HYLE_PLATFORM_C_API void* hyle_vm_region_usable(hyle_vm_region_o* r);
HYLE_PLATFORM_C_API size_t hyle_vm_region_usable_size(const hyle_vm_region_o* r);
HYLE_PLATFORM_C_API int hyle_vm_region_has_guard(const hyle_vm_region_o* r);
HYLE_PLATFORM_C_API hyle_numa_placement hyle_vm_region_numa_placement(
    const hyle_vm_region_o* r);
HYLE_PLATFORM_C_API hyle_status hyle_vm_region_commit(
    hyle_vm_region_o* r, size_t off, size_t len);
HYLE_PLATFORM_C_API hyle_status hyle_vm_region_commit_usable(
    hyle_vm_region_o* r);
HYLE_PLATFORM_C_API hyle_status hyle_vm_region_decommit(
    hyle_vm_region_o* r, size_t off, size_t len);
HYLE_PLATFORM_C_API hyle_status hyle_vm_region_protect(
    hyle_vm_region_o* r, size_t off, size_t len, hyle_protect prot);

/* ========================================================================== */
/* Filesystem + environment (buffer-fill / visitor; no hidden alloc)          */
/* ========================================================================== */

typedef enum hyle_file_type {
  HYLE_FILE_TYPE_UNKNOWN = 0,
  HYLE_FILE_TYPE_REGULAR = 1,
  HYLE_FILE_TYPE_DIRECTORY = 2,
  HYLE_FILE_TYPE_SYMLINK = 3,
  HYLE_FILE_TYPE_BLOCK_DEVICE = 4,
  HYLE_FILE_TYPE_CHARACTER_DEVICE = 5,
  HYLE_FILE_TYPE_FIFO = 6,
  HYLE_FILE_TYPE_SOCKET = 7
} hyle_file_type;

typedef struct hyle_file_info {
  hyle_file_type type; /* target kind; stat follows symlinks */
  uint64_t size;
  int64_t modified_time; /* seconds since epoch */
  int is_symlink;        /* path itself is a symbolic link (lstat semantics) */
} hyle_file_info;

HYLE_PLATFORM_C_API hyle_status hyle_fs_stat(hyle_file_info* out,
                                             const char* path);
HYLE_PLATFORM_C_API hyle_status hyle_fs_exists(int* out, const char* path);
HYLE_PLATFORM_C_API hyle_status hyle_fs_is_directory(int* out,
                                                     const char* path);
HYLE_PLATFORM_C_API hyle_status hyle_fs_file_size(uint64_t* out,
                                                  const char* path);

HYLE_PLATFORM_C_API hyle_status hyle_fs_make_directory(const char* path);
HYLE_PLATFORM_C_API hyle_status hyle_fs_make_directory_recursive(
    const char* path);
HYLE_PLATFORM_C_API hyle_status hyle_fs_remove_directory(const char* path);
HYLE_PLATFORM_C_API hyle_status hyle_fs_remove_directory_recursive(
    const char* path);
HYLE_PLATFORM_C_API hyle_status hyle_fs_remove_file(const char* path);
HYLE_PLATFORM_C_API hyle_status hyle_fs_rename(const char* from,
                                               const char* to);
HYLE_PLATFORM_C_API hyle_status hyle_fs_copy_file(const char* from,
                                                  const char* to);
HYLE_PLATFORM_C_API hyle_status hyle_fs_touch(const char* path);

/* snprintf-style buffer fills: *out_size is the full length, so >=cap means
 * truncated. It is zeroed on entry/failure. dst==NULL is a size query. */
HYLE_PLATFORM_C_API hyle_status hyle_fs_current_directory(
    size_t* out_size, char* dst, size_t cap);
HYLE_PLATFORM_C_API hyle_status hyle_fs_change_directory(const char* path);
HYLE_PLATFORM_C_API hyle_status hyle_fs_temp_directory(
    size_t* out_size, char* dst, size_t cap);
HYLE_PLATFORM_C_API hyle_status hyle_fs_home_directory(
    size_t* out_size, char* dst, size_t cap);

/* Environment read: unset is HYLE_STATUS_NOT_FOUND; a present empty value is
 * success with *out_size==0. */
HYLE_PLATFORM_C_API hyle_status hyle_env_get(
    size_t* out_size, const char* name, char* dst, size_t cap);

/* Directory iterate: visit(name,type,ctx) returns 0 to stop normally. `type`
 * describes the entry itself (symlinks are not followed).
 * out_completed is 1 when every entry was visited, 0 on early stop/failure. */
typedef int (*hyle_fs_visit_fn)(const char* name, hyle_file_type type,
                                void* ctx);
HYLE_PLATFORM_C_API hyle_status hyle_fs_directory_iterate(
    int* out_completed, const char* path, hyle_fs_visit_fn visit, void* ctx);

/* ---- fs permissions / symlink / tempfile / env-write (v4) ------------------ */
/* Permission bits (low 9 of POSIX mode); Windows synthesizes from READONLY. */
typedef enum hyle_file_perms {
  HYLE_PERM_OWNER_READ = 0400,
  HYLE_PERM_OWNER_WRITE = 0200,
  HYLE_PERM_OWNER_EXEC = 0100,
  HYLE_PERM_GROUP_READ = 0040,
  HYLE_PERM_GROUP_WRITE = 0020,
  HYLE_PERM_GROUP_EXEC = 0010,
  HYLE_PERM_OTHER_READ = 0004,
  HYLE_PERM_OTHER_WRITE = 0002,
  HYLE_PERM_OTHER_EXEC = 0001,
  HYLE_PERM_ALL = 0777
} hyle_file_perms;
HYLE_PLATFORM_C_API hyle_status hyle_fs_get_permissions(uint32_t* out,
                                                        const char* path);
HYLE_PLATFORM_C_API hyle_status hyle_fs_set_permissions(const char* path,
                                                        uint32_t mode);
HYLE_PLATFORM_C_API hyle_status hyle_fs_is_readonly(int* out,
                                                    const char* path);
HYLE_PLATFORM_C_API hyle_status hyle_fs_set_readonly(const char* path,
                                                     int readonly);
typedef enum hyle_symlink_kind {
  HYLE_SYMLINK_AUTO = 0,
  HYLE_SYMLINK_FILE = 1,
  HYLE_SYMLINK_DIRECTORY = 2
} hyle_symlink_kind;
/* Symlinks. Pass an explicit kind for a portable dangling link. AUTO infers an
 * existing target on Windows. Creation may return PERMISSION_DENIED there
 * without Developer Mode/admin privilege. */
HYLE_PLATFORM_C_API hyle_status hyle_fs_create_symlink(const char* target,
                                                        const char* link,
                                                        hyle_symlink_kind kind);
HYLE_PLATFORM_C_API hyle_status hyle_fs_create_hard_link(const char* target,
                                                         const char* link);
HYLE_PLATFORM_C_API hyle_status hyle_fs_is_symlink(int* out,
                                                   const char* path);
HYLE_PLATFORM_C_API hyle_status hyle_fs_read_symlink(size_t* out_size,
                                                     const char* path,
                                                     char* dst, size_t cap);
/* A native file resource stored losslessly as one integer word: POSIX fd or
 * Windows HANDLE. The type name prevents cross-platform callers from assuming
 * POSIX fd semantics. */
typedef intptr_t hyle_file_handle;
#define HYLE_FILE_HANDLE_INVALID ((hyle_file_handle)-1)
/* Atomically create a temp file; writes an OPEN read-write handle to *out and
 * the path to dst[cap]. *out_path_size receives the full path length even when
 * dst is truncated. Close it with hyle_fs_close. dir==NULL => system temp dir.
 * Outputs are INVALID/0 on entry and failure. */
HYLE_PLATFORM_C_API hyle_status hyle_fs_create_temp_file(
    hyle_file_handle* out, size_t* out_path_size, const char* dir,
    const char* prefix, char* dst, size_t cap);
/* Close a handle from create_temp_file/open. Safe on INVALID. */
HYLE_PLATFORM_C_API hyle_status hyle_fs_close(hyle_file_handle file);
/* open_file flags (OR them): READ/WRITE pick access; CREATE makes a missing file;
 * TRUNCATE empties it; APPEND puts writes at EOF. */
typedef enum hyle_file_open_flags {
  HYLE_OPEN_READ = 1u << 0,
  HYLE_OPEN_WRITE = 1u << 1,
  HYLE_OPEN_CREATE = 1u << 2,
  HYLE_OPEN_TRUNCATE = 1u << 3,
  HYLE_OPEN_APPEND = 1u << 4
} hyle_file_open_flags;
/* Open `path` with `flags` (HYLE_OPEN_* OR'd); READ and/or WRITE is required.
 * Writes a native resource to *out, INVALID on entry/failure. */
HYLE_PLATFORM_C_API hyle_status hyle_fs_open(hyle_file_handle* out,
                                             const char* path, unsigned flags);
/* Read/write on a handle from hyle_fs_open/create_temp_file. On success
 * *out_moved is the byte count (read 0 == EOF; a short count is normal). */
HYLE_PLATFORM_C_API hyle_status hyle_fs_read(size_t* out_moved,
                                             hyle_file_handle file, void* buf,
                                             size_t n);
HYLE_PLATFORM_C_API hyle_status hyle_fs_write(size_t* out_moved,
                                              hyle_file_handle file,
                                              const void* buf, size_t n);
/* Environment write / unset (process-wide; not thread-safe vs concurrent reads). */
HYLE_PLATFORM_C_API hyle_status hyle_env_set(const char* name,
                                             const char* value);
HYLE_PLATFORM_C_API hyle_status hyle_env_unset(const char* name);

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
/* Map [offset,offset+length) of `path` (length==0 => to EOF). Unaligned offsets
 * are supported; the implementation aligns the OS view internally. *out is
 * cleared before work; on success caller owns it and must destroy it. */
HYLE_PLATFORM_C_API hyle_status hyle_mapped_file_open(
    hyle_mapped_file_o** out, const char* path, hyle_map_mode mode,
    size_t offset, size_t length);
HYLE_PLATFORM_C_API void hyle_mapped_file_destroy(hyle_mapped_file_o* m);
HYLE_PLATFORM_C_API int hyle_mapped_file_valid(const hyle_mapped_file_o* m);
HYLE_PLATFORM_C_API void* hyle_mapped_file_data(hyle_mapped_file_o* m);
HYLE_PLATFORM_C_API size_t hyle_mapped_file_size(const hyle_mapped_file_o* m);
HYLE_PLATFORM_C_API hyle_status hyle_mapped_file_flush(hyle_mapped_file_o* m);

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
/* L1 data cache line size in bytes (64 on mainstream CPUs; for alignment/padding). */
HYLE_PLATFORM_C_API unsigned hyle_cpu_cache_line_size(void);

typedef struct hyle_cache_info {
  uint64_t l1d, l1i, l2, l3; /* bytes; 0 means unavailable */
} hyle_cache_info;

typedef struct hyle_cpu_topology {
  unsigned logical_cpus;
  unsigned performance_cpus;
  unsigned efficiency_cpus;
  int is_hybrid;
} hyle_cpu_topology;

HYLE_PLATFORM_C_API hyle_cache_info hyle_cpu_cache_info(void);
HYLE_PLATFORM_C_API hyle_cpu_topology hyle_cpu_topology_get(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* HYLE_C_PLATFORM_H_ */
