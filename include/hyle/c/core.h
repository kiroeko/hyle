/* SPDX: same as HYLE.
 *
 * Pure C ABI for the HYLE core module.
 * Naming: every symbol is `hyle_*` / `HYLE_*`.
 * Docs (1:1 with this header): docs/guide/c-core.md
 */
#ifndef HYLE_C_CORE_H_
#define HYLE_C_CORE_H_

#include <stddef.h>
#include <stdint.h>

#include "hyle/core/cache_defs.h"
#include "hyle/core/version_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(HYLE_STATIC)
#  define HYLE_CORE_C_API
#elif defined(_WIN32) || defined(__CYGWIN__)
#  if defined(HYLE_CORE_BUILD)
#    define HYLE_CORE_C_API __declspec(dllexport)
#  else
#    define HYLE_CORE_C_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define HYLE_CORE_C_API __attribute__((visibility("default")))
#else
#  define HYLE_CORE_C_API
#endif

/* Runtime version of the linked hyle_core. Compare with HYLE_VERSION_PACKED to
 * diagnose a header/runtime mismatch after dlopen or across an FFI boundary. */
HYLE_CORE_C_API uint32_t hyle_version(void);

/* Human-readable "major.minor.patch" (static storage; never freed). */
HYLE_CORE_C_API const char* hyle_version_string(void);

/* Decompose the packed version (header-inline; no DLL hop). */
static inline uint32_t hyle_version_major(uint32_t packed) { return (packed >> 16) & 0xFFu; }
static inline uint32_t hyle_version_minor(uint32_t packed) { return (packed >> 8) & 0xFFu; }
static inline uint32_t hyle_version_patch(uint32_t packed) { return packed & 0xFFu; }

/* Allocation-free intrusive doubly-linked list. The caller owns every node;
 * each hook may belong to at most one list and must be initialized before use.
 * Unlike C++ auto-unlink, C owners must unlink before their storage expires. */
typedef struct hyle_intrusive_list_hook {
  struct hyle_intrusive_list_hook* prev;
  struct hyle_intrusive_list_hook* next;
} hyle_intrusive_list_hook;

typedef struct hyle_intrusive_list {
  hyle_intrusive_list_hook sentinel;
} hyle_intrusive_list;

#define HYLE_INTRUSIVE_ENTRY(hook, type, member) ((type*)((char*)(hook) - offsetof(type, member)))

static inline void hyle_intrusive_hook_init(hyle_intrusive_list_hook* hook) {
  if (hook != NULL) {
    hook->prev = NULL;
    hook->next = NULL;
  }
}

static inline int hyle_intrusive_hook_linked(const hyle_intrusive_list_hook* hook) {
  return hook != NULL && hook->prev != NULL;
}

static inline void hyle_intrusive_hook_unlink(hyle_intrusive_list_hook* hook) {
  if (!hyle_intrusive_hook_linked(hook)) return;
  hook->prev->next = hook->next;
  hook->next->prev = hook->prev;
  hook->prev = NULL;
  hook->next = NULL;
}

static inline void hyle_intrusive_list_init(hyle_intrusive_list* list) {
  if (list != NULL) {
    list->sentinel.prev = &list->sentinel;
    list->sentinel.next = &list->sentinel;
  }
}

static inline int hyle_intrusive_list_empty(const hyle_intrusive_list* list) {
  return list == NULL || list->sentinel.next == &list->sentinel;
}

static inline int hyle_intrusive_list_push_back(hyle_intrusive_list* list,
                                                hyle_intrusive_list_hook* hook) {
  if (list == NULL || hook == NULL || hyle_intrusive_hook_linked(hook)) return 0;
  hook->prev = list->sentinel.prev;
  hook->next = &list->sentinel;
  list->sentinel.prev->next = hook;
  list->sentinel.prev = hook;
  return 1;
}

static inline int hyle_intrusive_list_push_front(hyle_intrusive_list* list,
                                                 hyle_intrusive_list_hook* hook) {
  if (list == NULL || hook == NULL || hyle_intrusive_hook_linked(hook)) return 0;
  hook->prev = &list->sentinel;
  hook->next = list->sentinel.next;
  list->sentinel.next->prev = hook;
  list->sentinel.next = hook;
  return 1;
}

static inline hyle_intrusive_list_hook* hyle_intrusive_list_front(hyle_intrusive_list* list) {
  return hyle_intrusive_list_empty(list) ? NULL : list->sentinel.next;
}

static inline hyle_intrusive_list_hook* hyle_intrusive_list_back(hyle_intrusive_list* list) {
  return hyle_intrusive_list_empty(list) ? NULL : list->sentinel.prev;
}

static inline hyle_intrusive_list_hook* hyle_intrusive_list_pop_front(hyle_intrusive_list* list) {
  hyle_intrusive_list_hook* hook = hyle_intrusive_list_front(list);
  hyle_intrusive_hook_unlink(hook);
  return hook;
}

static inline hyle_intrusive_list_hook* hyle_intrusive_list_pop_back(hyle_intrusive_list* list) {
  hyle_intrusive_list_hook* hook = hyle_intrusive_list_back(list);
  hyle_intrusive_hook_unlink(hook);
  return hook;
}

static inline void hyle_intrusive_list_clear(hyle_intrusive_list* list) {
  while (hyle_intrusive_list_pop_front(list) != NULL) {
  }
}

/* Splice source onto destination in O(1), preserving order. Returns zero for
 * invalid/self input and one for a successful no-op or transfer. */
static inline int hyle_intrusive_list_append(hyle_intrusive_list* destination,
                                             hyle_intrusive_list* source) {
  hyle_intrusive_list_hook* first;
  hyle_intrusive_list_hook* last;
  hyle_intrusive_list_hook* destination_last;
  if (destination == NULL || source == NULL || destination == source) return 0;
  if (hyle_intrusive_list_empty(source)) return 1;

  first = source->sentinel.next;
  last = source->sentinel.prev;
  source->sentinel.prev = &source->sentinel;
  source->sentinel.next = &source->sentinel;

  destination_last = destination->sentinel.prev;
  destination_last->next = first;
  first->prev = destination_last;
  last->next = &destination->sentinel;
  destination->sentinel.prev = last;
  return 1;
}

#ifdef __cplusplus
}
#endif

#endif /* HYLE_C_CORE_H_ */
