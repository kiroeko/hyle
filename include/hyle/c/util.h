/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary
 *
 * Pure C ABI for reusable HYLE util runtime facilities (close-01).
 * C++ template types remain source-level; their capabilities are erased below
 * into POD state, opaque handles, callbacks, and pointer/length operations.
 * Docs (1:1 with this header): docs/guide/c-util.md
 */
#ifndef HYLE_C_UTIL_H_
#define HYLE_C_UTIL_H_

#include <stddef.h>
#include <stdint.h>

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

/* Thread-local diagnostic for recoverable C-boundary failures. Successful
 * fallible calls clear stale text. Never free the returned pointer. */
HYLE_UTIL_C_API const char* hyle_util_last_error(void);

/* snprintf-style runtime symbol demangling. Returns the full required length,
 * excluding NUL; cap > 0 requires non-NULL out. Invalid/non-mangled input is
 * copied through, and NULL mangled input means an empty string. */
HYLE_UTIL_C_API size_t hyle_util_demangle(const char* mangled, char* out,
                                          size_t cap);

/* One architecture spin hint, kept header-inline because an ABI call would cost
 * more than the instruction. It is only a power/pipeline hint, not a wait or a
 * synchronization operation. */
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

/* xoshiro256++ state. Copying this POD clones a reproducible stream. Initialize
 * with hyle_util_rng_seed before use; the all-zero state is invalid. This PRNG
 * is NOT cryptographically secure. */
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

/* Batch loops stay inside the DLL so FFI callers do not pay one ABI call per
 * value. A zero count accepts NULL out. */
HYLE_UTIL_C_API int hyle_util_rng_fill_u64(
    hyle_util_rng* rng, uint64_t* out, size_t count);
HYLE_UTIL_C_API int hyle_util_rng_fill_float(
    hyle_util_rng* rng, float* out, size_t count);
HYLE_UTIL_C_API int hyle_util_rng_fill_double(
    hyle_util_rng* rng, double* out, size_t count);

/* jump partitions one seed into 2^128 substreams. long_jump advances 2^192 and
 * can partition groups before per-worker jump calls. */
HYLE_UTIL_C_API int hyle_util_rng_jump(hyle_util_rng* rng);
HYLE_UTIL_C_API int hyle_util_rng_long_jump(hyle_util_rng* rng);
HYLE_UTIL_C_API int hyle_util_rng_shuffle(
    hyle_util_rng* rng, uint64_t count, hyle_util_rng_swap_fn swap, void* ctx);

/* Borrowed thread-local state, valid until the current thread exits. Do not
 * destroy or share its address across threads. Explicitly seeded state is the
 * right choice when reproducibility matters. */
HYLE_UTIL_C_API hyle_util_rng* hyle_util_thread_rng(void);

static inline hyle_util_rng hyle_util_rng_clone(const hyle_util_rng* rng) {
  hyle_util_rng copy = {{0, 0, 0, 0}};
  if (rng != NULL) copy = *rng;
  return copy;
}

/* Type-erased shared ownership. create starts at +1. retain returns the same
 * handle with another strong reference; each successful retain requires one
 * release. The final release synchronously calls a non-NULL
 * destroy_object(object, ctx) once, then frees the control handle. A NULL
 * callback is allowed for a borrowed/no-op payload. Callbacks must not throw
 * through this C boundary. No thread may use a handle after its final release. */
typedef struct hyle_util_ref_o hyle_util_ref;
typedef void (*hyle_util_ref_destroy_fn)(void* object, void* ctx);

HYLE_UTIL_C_API hyle_util_ref* hyle_util_ref_create(
    void* object, hyle_util_ref_destroy_fn destroy_object, void* destroy_ctx);
HYLE_UTIL_C_API hyle_util_ref* hyle_util_ref_retain(hyle_util_ref* ref);
HYLE_UTIL_C_API void hyle_util_ref_release(hyle_util_ref* ref);
HYLE_UTIL_C_API void* hyle_util_ref_get(const hyle_util_ref* ref);
HYLE_UTIL_C_API uint32_t hyle_util_ref_use_count(const hyle_util_ref* ref);
HYLE_UTIL_C_API int hyle_util_ref_is_unique(const hyle_util_ref* ref);

/* Type-erased scope cleanup. create owns callback+ctx until destroy. destroy
 * runs an active callback exactly once; run invokes it early and dismisses it;
 * dismiss cancels it. The handle is not thread-safe. The callback must not
 * throw through the C boundary or destroy its own guard. */
typedef struct hyle_util_scope_guard_o hyle_util_scope_guard;
typedef void (*hyle_util_scope_guard_fn)(void* ctx);

HYLE_UTIL_C_API hyle_util_scope_guard* hyle_util_scope_guard_create(
    hyle_util_scope_guard_fn callback, void* ctx);
HYLE_UTIL_C_API int hyle_util_scope_guard_run(
    hyle_util_scope_guard* guard);
HYLE_UTIL_C_API int hyle_util_scope_guard_dismiss(
    hyle_util_scope_guard* guard);
HYLE_UTIL_C_API int hyle_util_scope_guard_is_active(
    const hyle_util_scope_guard* guard);
HYLE_UTIL_C_API void hyle_util_scope_guard_destroy(
    hyle_util_scope_guard* guard);

/* Type-erased runtime registry. It owns copies of key/help strings; callback
 * results remain caller-owned. Register/create/queries are thread-safe.
 * for_each invokes the visitor under the registry lock: do not re-enter the
 * same registry from that visitor. Quiesce all users before destroy. Callbacks
 * must not throw through this C boundary. */
typedef struct hyle_util_registry_o hyle_util_registry;

typedef enum hyle_util_registry_priority {
  HYLE_UTIL_REGISTRY_FALLBACK = 1,
  HYLE_UTIL_REGISTRY_DEFAULT = 2,
  HYLE_UTIL_REGISTRY_PREFERRED = 3
} hyle_util_registry_priority;

typedef void* (*hyle_util_registry_create_fn)(
    void* creator_ctx, const void* create_args);
/* Return non-zero to continue, zero to stop callbacks successfully. */
typedef int (*hyle_util_registry_visit_fn)(
    void* visit_ctx, const char* key, const char* help,
    hyle_util_registry_priority priority);

HYLE_UTIL_C_API hyle_util_registry* hyle_util_registry_create(void);
HYLE_UTIL_C_API void hyle_util_registry_destroy(
    hyle_util_registry* registry);
HYLE_UTIL_C_API int hyle_util_registry_register(
    hyle_util_registry* registry, const char* key,
    hyle_util_registry_create_fn creator, void* creator_ctx,
    hyle_util_registry_priority priority, const char* help);
HYLE_UTIL_C_API int hyle_util_registry_has(
    const hyle_util_registry* registry, const char* key);
HYLE_UTIL_C_API void* hyle_util_registry_create_object(
    hyle_util_registry* registry, const char* key, const void* create_args);
/* snprintf semantics: returns full help length, excluding NUL. Missing keys
 * return 0 plus last_error; a present empty help returns 0 with no error. */
HYLE_UTIL_C_API size_t hyle_util_registry_help(
    const hyle_util_registry* registry, const char* key, char* out, size_t cap);
HYLE_UTIL_C_API size_t hyle_util_registry_size(
    const hyle_util_registry* registry);
HYLE_UTIL_C_API int hyle_util_registry_for_each(
    hyle_util_registry* registry, hyle_util_registry_visit_fn visit,
    void* visit_ctx);

/* Opaque SpinLock capability. The ABI seam makes this suitable for FFI control
 * paths; native C++ callers should use header-inline hyle::util::SpinLock in hot
 * critical sections. destroy is NULL-safe and rejects a locked instance. */
typedef struct hyle_util_spin_lock_o hyle_util_spin_lock;
HYLE_UTIL_C_API hyle_util_spin_lock* hyle_util_spin_lock_create(void);
HYLE_UTIL_C_API int hyle_util_spin_lock_destroy(hyle_util_spin_lock* lock);
HYLE_UTIL_C_API int hyle_util_spin_lock_lock(hyle_util_spin_lock* lock);
HYLE_UTIL_C_API int hyle_util_spin_lock_try_lock(hyle_util_spin_lock* lock);
HYLE_UTIL_C_API int hyle_util_spin_lock_unlock(hyle_util_spin_lock* lock);
HYLE_UTIL_C_API int hyle_util_spin_lock_is_locked(
    const hyle_util_spin_lock* lock);

#ifdef __cplusplus
}
#endif

#endif /* HYLE_C_UTIL_H_ */
