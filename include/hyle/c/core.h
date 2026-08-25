/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary
 *
 * Pure C ABI for the HYLE core module.
 * Naming: every symbol is `hyle_*` / `HYLE_*`.
 * Docs (1:1 with this header): docs/guide/c-core.md
 */
#ifndef HYLE_C_CORE_H_
#define HYLE_C_CORE_H_

#include "hyle/c/compiler.h"

#include <stddef.h>
#include <stdint.h>

#include "hyle/core/cache_defs.h"
#include "hyle/core/version_defs.h"

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if defined(HYLE_STATIC)
#define HYLE_CORE_C_API
#elif defined(_WIN32) || defined(__CYGWIN__)
#if defined(HYLE_CORE_BUILD)
#define HYLE_CORE_C_API __declspec(dllexport)
#else
#define HYLE_CORE_C_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define HYLE_CORE_C_API __attribute__((visibility("default")))
#else
#define HYLE_CORE_C_API
#endif

#define HYLE_CORE_C_INLINE static HYLE_C_FORCE_INLINE

/* Runtime version of the linked hyle_core. Compare with HYLE_VERSION_PACKED to
 * diagnose a header/runtime mismatch after dlopen or across an FFI boundary. */
HYLE_CORE_C_API uint32_t hyle_version(void);

/* Human-readable "major.minor.patch" (static storage; never freed). */
HYLE_CORE_C_API const char *hyle_version_string(void);

/* Decompose the packed version (header-inline; no DLL hop). */
static inline uint32_t hyle_version_major(uint32_t packed) {
  return (packed >> 16) & 0xFFu;
}
static inline uint32_t hyle_version_minor(uint32_t packed) {
  return (packed >> 8) & 0xFFu;
}
static inline uint32_t hyle_version_patch(uint32_t packed) {
  return packed & 0xFFu;
}

/* Cache prefetch hints. They never establish validity, ownership or memory
 * ordering; use only after profiling a representative workload. */
HYLE_CORE_C_INLINE void hyle_prefetch_read(const void *address) {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(address, 0, 3);
#elif defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
  _mm_prefetch((const char *)address, _MM_HINT_T0);
#elif defined(_MSC_VER) && defined(_M_ARM64)
  __prefetch2(address, 0x0); /* PLD L1 KEEP */
#else
  (void)address;
#endif
}

HYLE_CORE_C_INLINE void hyle_prefetch_read_non_temporal(const void *address) {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(address, 0, 0);
#elif defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
  _mm_prefetch((const char *)address, _MM_HINT_NTA);
#elif defined(_MSC_VER) && defined(_M_ARM64)
  __prefetch2(address, 0x1); /* PLD L1 STRM */
#else
  (void)address;
#endif
}

HYLE_CORE_C_INLINE void hyle_prefetch_write(const void *address) {
#if defined(__GNUC__) || defined(__clang__)
  /* Preserve the TU's ISA baseline: the compiler may select a weaker hint
   * when PREFETCHW is not enabled instead of injecting an unsupported opcode.
   */
  __builtin_prefetch(address, 1, 3);
#elif defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
#if defined(_MM_HINT_ET0)
  _mm_prefetch((const char *)address, _MM_HINT_ET0);
#else
  _mm_prefetch((const char *)address, _MM_HINT_T0);
#endif
#elif defined(_MSC_VER) && defined(_M_ARM64)
  __prefetch2(address, 0x10); /* PST L1 KEEP */
#else
  (void)address;
#endif
}

/* Allocation-free intrusive doubly-linked list. The caller owns every node;
 * each hook may belong to at most one list and must be initialized before use.
 * Unlike C++ auto-unlink, C owners must unlink before their storage expires. */
typedef struct hyle_intrusive_list_hook {
  struct hyle_intrusive_list_hook *prev;
  struct hyle_intrusive_list_hook *next;
} hyle_intrusive_list_hook;

typedef struct hyle_intrusive_list {
  hyle_intrusive_list_hook sentinel;
} hyle_intrusive_list;

#define HYLE_INTRUSIVE_HOOK_INIT                                             \
  { NULL, NULL }
#define HYLE_INTRUSIVE_LIST_INIT(name)                                       \
  { { &(name).sentinel, &(name).sentinel } }

#define HYLE_INTRUSIVE_ENTRY(hook, type, member)                               \
  ((type *)((char *)(hook) - offsetof(type, member)))
#define HYLE_INTRUSIVE_CONST_ENTRY(hook, type, member)                         \
  ((const type *)((const char *)(hook) - offsetof(type, member)))

static inline void hyle_intrusive_hook_init(hyle_intrusive_list_hook *hook) {
  if (hook != NULL) {
    hook->prev = NULL;
    hook->next = NULL;
  }
}

static inline int
hyle_intrusive_hook_linked(const hyle_intrusive_list_hook *hook) {
  return hook != NULL && hook->prev != NULL;
}

static inline void hyle_intrusive_hook_unlink(hyle_intrusive_list_hook *hook) {
  if (!hyle_intrusive_hook_linked(hook))
    return;
  hook->prev->next = hook->next;
  hook->next->prev = hook->prev;
  hook->prev = NULL;
  hook->next = NULL;
}

static inline void hyle_intrusive_list_init(hyle_intrusive_list *list) {
  if (list != NULL) {
    list->sentinel.prev = &list->sentinel;
    list->sentinel.next = &list->sentinel;
  }
}

static inline int hyle_intrusive_list_empty(const hyle_intrusive_list *list) {
  return list == NULL || list->sentinel.next == &list->sentinel;
}

/* Insert before position; NULL position means end(). Preconditions for the
 * unchecked form: list/hook are non-NULL, position is NULL or belongs to list,
 * and hook is not linked. */
static inline void
hyle_intrusive_list_insert_before(hyle_intrusive_list *list,
                                  hyle_intrusive_list_hook *position,
                                  hyle_intrusive_list_hook *hook) {
  if (position == NULL)
    position = &list->sentinel;
  hook->prev = position->prev;
  hook->next = position;
  position->prev->next = hook;
  position->prev = hook;
}

HYLE_C_NODISCARD static inline int
hyle_intrusive_list_try_insert_before(hyle_intrusive_list *list,
                                      hyle_intrusive_list_hook *position,
                                      hyle_intrusive_list_hook *hook) {
  if (list == NULL || hook == NULL || hyle_intrusive_hook_linked(hook))
    return 0;
  hyle_intrusive_list_insert_before(list, position, hook);
  return 1;
}

static inline void
hyle_intrusive_list_push_back(hyle_intrusive_list *list,
                              hyle_intrusive_list_hook *hook) {
  hyle_intrusive_list_insert_before(list, NULL, hook);
}

static inline void
hyle_intrusive_list_push_front(hyle_intrusive_list *list,
                               hyle_intrusive_list_hook *hook) {
  hyle_intrusive_list_insert_before(list, list->sentinel.next, hook);
}

HYLE_C_NODISCARD static inline int
hyle_intrusive_list_try_push_back(hyle_intrusive_list *list,
                                  hyle_intrusive_list_hook *hook) {
  if (list == NULL || hook == NULL || hyle_intrusive_hook_linked(hook))
    return 0;
  hyle_intrusive_list_push_back(list, hook);
  return 1;
}

HYLE_C_NODISCARD static inline int
hyle_intrusive_list_try_push_front(hyle_intrusive_list *list,
                                   hyle_intrusive_list_hook *hook) {
  if (list == NULL || hook == NULL || hyle_intrusive_hook_linked(hook))
    return 0;
  hyle_intrusive_list_push_front(list, hook);
  return 1;
}

static inline hyle_intrusive_list_hook *
hyle_intrusive_list_front(hyle_intrusive_list *list) {
  return hyle_intrusive_list_empty(list) ? NULL : list->sentinel.next;
}

static inline hyle_intrusive_list_hook *
hyle_intrusive_list_back(hyle_intrusive_list *list) {
  return hyle_intrusive_list_empty(list) ? NULL : list->sentinel.prev;
}

static inline const hyle_intrusive_list_hook *
hyle_intrusive_list_front_const(const hyle_intrusive_list *list) {
  return hyle_intrusive_list_empty(list) ? NULL : list->sentinel.next;
}

static inline const hyle_intrusive_list_hook *
hyle_intrusive_list_back_const(const hyle_intrusive_list *list) {
  return hyle_intrusive_list_empty(list) ? NULL : list->sentinel.prev;
}

/* Traversal helpers return NULL at the list boundary. `hook` must be NULL or
 * belong to `list`; capture next before unlinking the current hook. */
static inline hyle_intrusive_list_hook *
hyle_intrusive_list_next(hyle_intrusive_list *list,
                         hyle_intrusive_list_hook *hook) {
  if (list == NULL || hook == NULL || hook->next == &list->sentinel)
    return NULL;
  return hook->next;
}

static inline hyle_intrusive_list_hook *
hyle_intrusive_list_prev(hyle_intrusive_list *list,
                         hyle_intrusive_list_hook *hook) {
  if (list == NULL || hook == NULL || hook->prev == &list->sentinel)
    return NULL;
  return hook->prev;
}

static inline const hyle_intrusive_list_hook *
hyle_intrusive_list_next_const(const hyle_intrusive_list *list,
                               const hyle_intrusive_list_hook *hook) {
  if (list == NULL || hook == NULL || hook->next == &list->sentinel)
    return NULL;
  return hook->next;
}

static inline const hyle_intrusive_list_hook *
hyle_intrusive_list_prev_const(const hyle_intrusive_list *list,
                               const hyle_intrusive_list_hook *hook) {
  if (list == NULL || hook == NULL || hook->prev == &list->sentinel)
    return NULL;
  return hook->prev;
}

HYLE_C_NODISCARD static inline hyle_intrusive_list_hook *
hyle_intrusive_list_pop_front(hyle_intrusive_list *list) {
  hyle_intrusive_list_hook *hook;
  hyle_intrusive_list_hook *next;
  if (hyle_intrusive_list_empty(list))
    return NULL;
  hook = list->sentinel.next;
  next = hook->next;
  list->sentinel.next = next;
  next->prev = &list->sentinel;
  hook->prev = NULL;
  hook->next = NULL;
  return hook;
}

HYLE_C_NODISCARD static inline hyle_intrusive_list_hook *
hyle_intrusive_list_pop_back(hyle_intrusive_list *list) {
  hyle_intrusive_list_hook *hook;
  hyle_intrusive_list_hook *prev;
  if (hyle_intrusive_list_empty(list))
    return NULL;
  hook = list->sentinel.prev;
  prev = hook->prev;
  list->sentinel.prev = prev;
  prev->next = &list->sentinel;
  hook->prev = NULL;
  hook->next = NULL;
  return hook;
}

static inline void hyle_intrusive_list_clear(hyle_intrusive_list *list) {
  hyle_intrusive_list_hook *current;
  if (list == NULL)
    return;
  current = list->sentinel.next;
  while (current != &list->sentinel) {
    hyle_intrusive_list_hook *next = current->next;
    current->prev = NULL;
    current->next = NULL;
    current = next;
  }
  list->sentinel.prev = &list->sentinel;
  list->sentinel.next = &list->sentinel;
}

/* Splice all of source before position in O(1); NULL position means end().
 * Position ownership is an unchecked precondition: it must be NULL or belong
 * to destination. Returns zero for a NULL list/source or self input, and one
 * for a successful no-op or transfer. */
HYLE_C_NODISCARD static inline int
hyle_intrusive_list_splice_before(hyle_intrusive_list *destination,
                                  hyle_intrusive_list_hook *position,
                                  hyle_intrusive_list *source) {
  hyle_intrusive_list_hook *first;
  hyle_intrusive_list_hook *last;
  hyle_intrusive_list_hook *before;
  if (destination == NULL || source == NULL || destination == source)
    return 0;
  if (hyle_intrusive_list_empty(source))
    return 1;
  if (position == NULL)
    position = &destination->sentinel;

  first = source->sentinel.next;
  last = source->sentinel.prev;
  source->sentinel.prev = &source->sentinel;
  source->sentinel.next = &source->sentinel;

  before = position->prev;
  before->next = first;
  first->prev = before;
  last->next = position;
  position->prev = last;
  return 1;
}

/* Append is the common tail-batch spelling of splice_before(..., NULL, ...). */
HYLE_C_NODISCARD static inline int
hyle_intrusive_list_append(hyle_intrusive_list *destination,
                           hyle_intrusive_list *source) {
  return hyle_intrusive_list_splice_before(destination, NULL, source);
}

#ifdef __cplusplus
}
#endif

#undef HYLE_CORE_C_INLINE

#endif /* HYLE_C_CORE_H_ */
