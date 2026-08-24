/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary
 *
 * Pure C ABI for language-independent HYLE util-runtime capabilities.
 * C++-only lexical helpers remain in hyle::util_core; this surface covers
 * diagnostics, reproducible random streams, shared ownership, named factories,
 * and an opaque FFI synchronization handle.
 * Docs (1:1 with this header): docs/guide/c-util.md
 */
#ifndef HYLE_C_UTIL_H_
#define HYLE_C_UTIL_H_

#include <stddef.h>
#include <stdint.h>

#include "hyle/c/error.h"
#include "hyle/c/container.h"
#include "hyle/c/mem.h"

#if defined(_MSC_VER)
#  include <intrin.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if defined(HYLE_STATIC)
#  define HYLE_UTIL_C_API
#elif defined(_WIN32) || defined(__CYGWIN__)
#  if defined(HYLE_UTIL_BUILD)
#    define HYLE_UTIL_C_API __declspec(dllexport)
#  else
#    define HYLE_UTIL_C_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define HYLE_UTIL_C_API __attribute__((visibility("default")))
#else
#  define HYLE_UTIL_C_API
#endif

/* snprintf-style runtime symbol demangling. On success, out_size receives the
 * full required length excluding NUL. A non-mangled input is copied through;
 * NULL mangled means the empty string. cap>0 requires non-NULL out. */
HYLE_UTIL_C_API hyle_status hyle_util_demangle(
    size_t* out_size, const char* mangled, char* out, size_t cap);

/* One architecture spin hint. Header-inline because an ABI call would cost
 * more than the instruction. This is not a wait or synchronization operation. */
static inline void hyle_util_cpu_relax(void) {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
  _mm_pause();
#elif defined(_MSC_VER) && (defined(_M_ARM64) || defined(_M_ARM))
  __yield();
#elif defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
  __asm__ __volatile__("isb" ::: "memory");
#endif
}

/* xoshiro256++ POD state. Copying clones a reproducible stream. Seed before
 * use; the all-zero state is invalid. Invalid pointers/state/bounds return the
 * documented zero/false value without changing state. These checks are kept
 * value-shaped because random generation is a hot primitive; use is_valid at
 * configuration boundaries. This PRNG is not cryptographically secure. */
typedef struct hyle_util_rng {
  uint64_t state[4];
} hyle_util_rng;

typedef void (*hyle_util_rng_swap_fn)(void* ctx, uint64_t left,
                                      uint64_t right);

HYLE_UTIL_C_API int hyle_util_rng_seed(hyle_util_rng* rng, uint64_t seed);
HYLE_UTIL_C_API int hyle_util_rng_is_valid(const hyle_util_rng* rng);
HYLE_UTIL_C_API uint64_t hyle_util_rng_next_u64(hyle_util_rng* rng);
HYLE_UTIL_C_API uint32_t hyle_util_rng_next_u32(hyle_util_rng* rng);
HYLE_UTIL_C_API int hyle_util_rng_next_bool(hyle_util_rng* rng);
HYLE_UTIL_C_API float hyle_util_rng_next_float(hyle_util_rng* rng);
HYLE_UTIL_C_API double hyle_util_rng_next_double(hyle_util_rng* rng);
HYLE_UTIL_C_API uint64_t hyle_util_rng_range_u64(
    hyle_util_rng* rng, uint64_t lo, uint64_t hi);
HYLE_UTIL_C_API uint32_t hyle_util_rng_range_u32(
    hyle_util_rng* rng, uint32_t lo, uint32_t hi);
HYLE_UTIL_C_API double hyle_util_rng_range_double(
    hyle_util_rng* rng, double lo, double hi);
HYLE_UTIL_C_API int hyle_util_rng_fill_u64(
    hyle_util_rng* rng, uint64_t* out, size_t count);
HYLE_UTIL_C_API int hyle_util_rng_fill_float(
    hyle_util_rng* rng, float* out, size_t count);
HYLE_UTIL_C_API int hyle_util_rng_fill_double(
    hyle_util_rng* rng, double* out, size_t count);
/* Advance exactly count ordinary outputs. This is O(count), matching standard
 * C++ engines; use jump/long_jump for fixed-size parallel-stream partitioning. */
HYLE_UTIL_C_API int hyle_util_rng_discard(
    hyle_util_rng* rng, uint64_t count);
HYLE_UTIL_C_API int hyle_util_rng_jump(hyle_util_rng* rng);
HYLE_UTIL_C_API int hyle_util_rng_long_jump(hyle_util_rng* rng);
HYLE_UTIL_C_API int hyle_util_rng_shuffle(
    hyle_util_rng* rng, uint64_t count, hyle_util_rng_swap_fn swap, void* ctx);

/* Borrowed thread-local state, valid until current-thread exit. Do not destroy,
 * retain, or share its address across threads. Explicit state is preferable
 * whenever reproducibility matters. */
HYLE_UTIL_C_API hyle_util_rng* hyle_util_thread_rng(void);

static inline hyle_util_rng hyle_util_rng_clone(const hyle_util_rng* rng) {
  hyle_util_rng copy = {{0, 0, 0, 0}};
  if (rng != NULL) copy = *rng;
  return copy;
}

/* Type-erased shared ownership. create starts at +1 and stores a copy of the
 * backing allocator in the opaque control block. Final release synchronously
 * invokes destroy_object once, then frees the block through that allocator.
 * retain(NULL) returns NULL; refcount exhaustion also returns NULL. */
typedef struct hyle_util_ref_o hyle_util_ref;
typedef void (*hyle_util_ref_destroy_fn)(void* object, void* ctx);

HYLE_UTIL_C_API hyle_status hyle_util_ref_create(
    hyle_util_ref** out, void* object,
    hyle_util_ref_destroy_fn destroy_object, void* destroy_ctx,
    hyle_allocator backing);
static inline hyle_status hyle_util_ref_create_system(
    hyle_util_ref** out, void* object,
    hyle_util_ref_destroy_fn destroy_object, void* destroy_ctx) {
  return hyle_util_ref_create(out, object, destroy_object, destroy_ctx,
                              hyle_system_allocator());
}
HYLE_UTIL_C_API hyle_util_ref* hyle_util_ref_retain(hyle_util_ref* ref);
HYLE_UTIL_C_API void hyle_util_ref_release(hyle_util_ref* ref);
HYLE_UTIL_C_API void* hyle_util_ref_get(const hyle_util_ref* ref);
HYLE_UTIL_C_API uint32_t hyle_util_ref_use_count(const hyle_util_ref* ref);
HYLE_UTIL_C_API int hyle_util_ref_is_unique(const hyle_util_ref* ref);

/* Type-erased named factory registry. It owns key/help copies. Registration,
 * lookup and enumeration are thread-safe; creators and visitors run outside
 * its lock and may re-enter. freeze is a one-way/idempotent transition that
 * rejects later mutation and makes reads lock-free expected O(1). Quiesce all
 * users before destroy. Callback-returned objects remain caller-owned. Every
 * creator context, including one replaced by a higher-priority registration,
 * must outlive its in-flight calls. */
typedef struct hyle_util_registry_o hyle_util_registry;

/* Fixed-width at the C ABI boundary; -fshort-enums must not alter callback or
 * exported-function calling conventions. */
typedef uint32_t hyle_util_registry_priority;
enum {
  HYLE_UTIL_REGISTRY_FALLBACK = 1,
  HYLE_UTIL_REGISTRY_DEFAULT = 2,
  HYLE_UTIL_REGISTRY_PREFERRED = 3
};

/* A creator returns its caller-owned object directly. `out_status` is either
 * NULL for a diagnostic-free nullable call, or points to an initialized OK
 * Status for a checked call. On failure return NULL and, when out_status is
 * non-NULL, replace it with one owning error Status. On success leave it OK;
 * a successful object may itself be NULL. Callbacks must not unwind C++
 * exceptions through C. */
typedef void* (*hyle_util_registry_create_fn)(
    hyle_status* out_status, void* creator_ctx, const void* create_args);
/* Return non-zero to continue; zero stops successfully. key/help remain valid
 * until registry destruction but are always borrowed. */
typedef int (*hyle_util_registry_visit_fn)(
    void* visit_ctx, hyle_container_string_view key,
    hyle_container_string_view help,
    hyle_util_registry_priority priority);

HYLE_UTIL_C_API hyle_status hyle_util_registry_create(
    hyle_util_registry** out, hyle_allocator backing);
static inline hyle_status hyle_util_registry_create_system(
    hyle_util_registry** out) {
  return hyle_util_registry_create(out, hyle_system_allocator());
}
HYLE_UTIL_C_API void hyle_util_registry_destroy(
    hyle_util_registry* registry);
HYLE_UTIL_C_API hyle_status hyle_util_registry_reserve(
    hyle_util_registry* registry, size_t count);
HYLE_UTIL_C_API hyle_status hyle_util_registry_register(
    hyle_util_registry* registry, hyle_container_string_view key,
    hyle_util_registry_create_fn creator, void* creator_ctx,
    hyle_util_registry_priority priority, hyle_container_string_view help);
static inline hyle_status hyle_util_registry_register_cstr(
    hyle_util_registry* registry, const char* key,
    hyle_util_registry_create_fn creator, void* creator_ctx,
    hyle_util_registry_priority priority, const char* help) {
  return hyle_util_registry_register(
      registry, hyle_container_cstr(key), creator, creator_ctx, priority,
      hyle_container_cstr(help != NULL ? help : ""));
}
HYLE_UTIL_C_API hyle_status hyle_util_registry_freeze(
    hyle_util_registry* registry);
HYLE_UTIL_C_API int hyle_util_registry_is_frozen(
    const hyle_util_registry* registry);
HYLE_UTIL_C_API int hyle_util_registry_has(
    const hyle_util_registry* registry, hyle_container_string_view key);
static inline int hyle_util_registry_has_cstr(
    const hyle_util_registry* registry, const char* key) {
  return hyle_util_registry_has(registry, hyle_container_cstr(key));
}
HYLE_UTIL_C_API hyle_status hyle_util_registry_create_object(
    void** out, const hyle_util_registry* registry,
    hyle_container_string_view key,
    const void* create_args);
/* Hot/value-shaped counterpart of create_object: returns NULL for an invalid or
 * absent key, a creator failure, or a successful NULL result. The creator is
 * called with out_status == NULL so it can skip diagnostics entirely. Use the
 * checked form when those outcomes must be distinguished. */
HYLE_UTIL_C_API void* hyle_util_registry_create_or_null(
    const hyle_util_registry* registry, hyle_container_string_view key,
    const void* create_args);
static inline hyle_status hyle_util_registry_create_object_cstr(
    void** out, const hyle_util_registry* registry, const char* key,
    const void* create_args) {
  return hyle_util_registry_create_object(
      out, registry, hyle_container_cstr(key), create_args);
}
static inline void* hyle_util_registry_create_or_null_cstr(
    const hyle_util_registry* registry, const char* key,
    const void* create_args) {
  return hyle_util_registry_create_or_null(
      registry, hyle_container_cstr(key), create_args);
}
HYLE_UTIL_C_API hyle_status hyle_util_registry_help(
    size_t* out_size, const hyle_util_registry* registry,
    hyle_container_string_view key, char* out, size_t cap);
static inline hyle_status hyle_util_registry_help_cstr(
    size_t* out_size, const hyle_util_registry* registry, const char* key,
    char* out, size_t cap) {
  return hyle_util_registry_help(
      out_size, registry, hyle_container_cstr(key), out, cap);
}
HYLE_UTIL_C_API size_t hyle_util_registry_size(
    const hyle_util_registry* registry);
/* out_completed receives non-zero if every snapshot entry was visited and zero
 * if the visitor stopped early. It may be NULL when the distinction is unused. */
HYLE_UTIL_C_API hyle_status hyle_util_registry_for_each(
    int* out_completed, const hyle_util_registry* registry,
    hyle_util_registry_visit_fn visit, void* visit_ctx);

/* Opaque FFI SpinLock. Native C++ should use header-inline util::SpinLock for
 * hot critical sections. Heap creation accepts an allocator; init/deinit lets
 * arenas or foreign runtimes own the opaque storage. Do not mix the two
 * lifecycle pairs. Destroy/deinit reject a locked instance. */
typedef struct hyle_util_spin_lock_o hyle_util_spin_lock;
HYLE_UTIL_C_API hyle_status hyle_util_spin_lock_create(
    hyle_util_spin_lock** out, hyle_allocator backing);
static inline hyle_status hyle_util_spin_lock_create_system(
    hyle_util_spin_lock** out) {
  return hyle_util_spin_lock_create(out, hyle_system_allocator());
}
HYLE_UTIL_C_API size_t hyle_util_spin_lock_storage_size(void);
HYLE_UTIL_C_API size_t hyle_util_spin_lock_storage_alignment(void);
HYLE_UTIL_C_API hyle_status hyle_util_spin_lock_init(
    hyle_util_spin_lock** out, void* storage, size_t storage_size);
HYLE_UTIL_C_API hyle_status hyle_util_spin_lock_destroy(
    hyle_util_spin_lock* lock);
HYLE_UTIL_C_API hyle_status hyle_util_spin_lock_deinit(
    hyle_util_spin_lock* lock);
HYLE_UTIL_C_API hyle_status hyle_util_spin_lock_lock(
    hyle_util_spin_lock* lock);
HYLE_UTIL_C_API hyle_status hyle_util_spin_lock_try_lock(
    int* out_acquired, hyle_util_spin_lock* lock);
HYLE_UTIL_C_API hyle_status hyle_util_spin_lock_unlock(
    hyle_util_spin_lock* lock);
HYLE_UTIL_C_API int hyle_util_spin_lock_is_locked(
    const hyle_util_spin_lock* lock);

#ifdef __cplusplus
}
#endif

#endif /* HYLE_C_UTIL_H_ */
