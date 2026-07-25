/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary
 * Pure C ABI for HYLE container capabilities (pre-release draft).
 * Docs: docs/guide/c-container.md */
#ifndef HYLE_C_CONTAINER_H_
#define HYLE_C_CONTAINER_H_

#include <stddef.h>
#include <stdint.h>

#include "hyle/c/error.h"
#include "hyle/c/mem.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(HYLE_STATIC)
#  define HYLE_CONTAINER_C_API
#elif defined(_WIN32) || defined(__CYGWIN__)
#  if defined(HYLE_CONTAINER_BUILD)
#    define HYLE_CONTAINER_C_API __declspec(dllexport)
#  else
#    define HYLE_CONTAINER_C_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define HYLE_CONTAINER_C_API __attribute__((visibility("default")))
#else
#  define HYLE_CONTAINER_C_API
#endif

#define HYLE_CONTAINER_NPOS UINT32_MAX
#if defined(__cplusplus)
#  define HYLE_CONTAINER_ALIGNOF(type) alignof(type)
#elif defined(_MSC_VER)
#  define HYLE_CONTAINER_ALIGNOF(type) __alignof(type)
#else
#  define HYLE_CONTAINER_ALIGNOF(type) _Alignof(type)
#endif

typedef struct hyle_container_string_view {
  const char* data;
  uint32_t size;
  uint32_t null_terminated;
} hyle_container_string_view;

static inline hyle_container_string_view hyle_container_view(
    const char* data, uint32_t size) {
  hyle_container_string_view value = {data, size, 0};
  return value;
}

/* Header-only byte helpers: ASCII semantics, independent of process locale. */
static inline int hyle_container_ascii_is_space(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' ||
         c == '\f';
}
static inline int hyle_container_ascii_is_digit(char c) {
  return c >= '0' && c <= '9';
}
static inline int hyle_container_ascii_is_print(char c) {
  const unsigned char value = (unsigned char)c;
  return value > 0x1fu && value < 0x7fu;
}
static inline int hyle_container_ascii_is_hex_digit(char c) {
  return hyle_container_ascii_is_digit(c) ||
         (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}
static inline int hyle_container_ascii_hex_value(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}
static inline char hyle_container_ascii_lower(char c) {
  return (c >= 'A' && c <= 'Z') ? (char)(c + ('a' - 'A')) : c;
}
static inline char hyle_container_ascii_upper(char c) {
  return (c >= 'a' && c <= 'z') ? (char)(c - ('a' - 'A')) : c;
}
static inline int hyle_container_path_is_separator(char c) {
  return c == '/' || c == '\\';
}

/* Error contract:
 * - create/reserve/insert/intern operations return owning hyle_status values
 *   and initialize their leading out parameter before doing work;
 * - queue full/empty, map miss, and stale IDs are normal value results;
 * - borrowed views require size == 0 || data != NULL.
 * There is deliberately no module-private TLS last-error channel. */

HYLE_CONTAINER_C_API uint64_t hyle_container_hash_bytes(
    const void* data, size_t size, uint64_t seed);

HYLE_CONTAINER_C_API int hyle_container_string_compare(
    hyle_container_string_view left, hyle_container_string_view right);
HYLE_CONTAINER_C_API int hyle_container_string_equal_ignore_case(
    hyle_container_string_view left, hyle_container_string_view right);
HYLE_CONTAINER_C_API uint32_t hyle_container_string_find(
    hyle_container_string_view text, hyle_container_string_view needle,
    uint32_t position);
HYLE_CONTAINER_C_API uint32_t hyle_container_string_rfind(
    hyle_container_string_view text, hyle_container_string_view needle,
    uint32_t position);
HYLE_CONTAINER_C_API uint32_t hyle_container_string_find_char(
    hyle_container_string_view text, char needle, uint32_t position);
HYLE_CONTAINER_C_API uint32_t hyle_container_string_rfind_char(
    hyle_container_string_view text, char needle, uint32_t position);
HYLE_CONTAINER_C_API uint32_t hyle_container_string_count_char(
    hyle_container_string_view text, char needle);
HYLE_CONTAINER_C_API uint32_t hyle_container_string_find_first_of(
    hyle_container_string_view text, hyle_container_string_view set,
    uint32_t position);
HYLE_CONTAINER_C_API uint32_t hyle_container_string_find_first_not_of(
    hyle_container_string_view text, hyle_container_string_view set,
    uint32_t position);
HYLE_CONTAINER_C_API uint32_t hyle_container_string_find_last_of(
    hyle_container_string_view text, hyle_container_string_view set,
    uint32_t position);
HYLE_CONTAINER_C_API uint32_t hyle_container_string_find_last_not_of(
    hyle_container_string_view text, hyle_container_string_view set,
    uint32_t position);
HYLE_CONTAINER_C_API int hyle_container_string_starts_with(
    hyle_container_string_view text, hyle_container_string_view prefix);
HYLE_CONTAINER_C_API int hyle_container_string_ends_with(
    hyle_container_string_view text, hyle_container_string_view suffix);
HYLE_CONTAINER_C_API int hyle_container_string_contains(
    hyle_container_string_view text, hyle_container_string_view needle);
HYLE_CONTAINER_C_API hyle_container_string_view hyle_container_string_slice(
    hyle_container_string_view text, uint32_t start, uint32_t length);
HYLE_CONTAINER_C_API hyle_container_string_view hyle_container_string_split(
    hyle_container_string_view* text, char delimiter);
HYLE_CONTAINER_C_API hyle_container_string_view hyle_container_string_rsplit(
    hyle_container_string_view* text, char delimiter);
HYLE_CONTAINER_C_API hyle_container_string_view hyle_container_string_trim(
    hyle_container_string_view text, hyle_container_string_view cut);
HYLE_CONTAINER_C_API size_t hyle_container_string_copy(
    hyle_container_string_view text, char* out, size_t capacity);
HYLE_CONTAINER_C_API size_t hyle_container_string_concat(
    hyle_container_string_view left, hyle_container_string_view right,
    char* out, size_t capacity);
HYLE_CONTAINER_C_API size_t hyle_container_string_lower(
    hyle_container_string_view text, char* out, size_t capacity);
HYLE_CONTAINER_C_API size_t hyle_container_string_upper(
    hyle_container_string_view text, char* out, size_t capacity);
HYLE_CONTAINER_C_API size_t hyle_container_hex_encode(
    const void* data, size_t size, char* out, size_t capacity);
HYLE_CONTAINER_C_API size_t hyle_container_hex_decode(
    hyle_container_string_view hex, void* out, size_t capacity);
HYLE_CONTAINER_C_API size_t hyle_container_string_escape(
    hyle_container_string_view text, char* out, size_t capacity);
HYLE_CONTAINER_C_API void hyle_container_string_capitalize(char* text);
HYLE_CONTAINER_C_API int64_t hyle_container_utf8_length(
    hyle_container_string_view text);
HYLE_CONTAINER_C_API int hyle_container_utf8_icmp(
    hyle_container_string_view left, hyle_container_string_view right,
    int codepoint_limit);

HYLE_CONTAINER_C_API hyle_container_string_view hyle_container_path_extension(
    hyle_container_string_view path);
HYLE_CONTAINER_C_API hyle_container_string_view
hyle_container_path_strip_extension(hyle_container_string_view path);
HYLE_CONTAINER_C_API hyle_container_string_view hyle_container_path_base(
    hyle_container_string_view path);
HYLE_CONTAINER_C_API hyle_container_string_view hyle_container_path_directory(
    hyle_container_string_view path);
HYLE_CONTAINER_C_API int hyle_container_path_is_absolute(
    hyle_container_string_view path);
HYLE_CONTAINER_C_API int hyle_container_path_is_unc(
    hyle_container_string_view path);
HYLE_CONTAINER_C_API hyle_container_string_view hyle_container_path_pop_first(
    hyle_container_string_view* path);
HYLE_CONTAINER_C_API hyle_container_string_view hyle_container_path_pop_last(
    hyle_container_string_view* path);
HYLE_CONTAINER_C_API size_t hyle_container_path_join(
    hyle_container_string_view left, hyle_container_string_view right,
    char* out, size_t capacity);
HYLE_CONTAINER_C_API size_t hyle_container_path_join_array(
    const hyle_container_string_view* parts, size_t count,
    char* out, size_t capacity);
/* Returns the full component count; writes at most output_capacity views. */
HYLE_CONTAINER_C_API size_t hyle_container_path_split_all(
    hyle_container_string_view path, hyle_container_string_view* output,
    size_t output_capacity);
HYLE_CONTAINER_C_API size_t hyle_container_path_replace_extension(
    hyle_container_string_view path, hyle_container_string_view extension,
    char* out, size_t capacity);
HYLE_CONTAINER_C_API uint32_t hyle_container_path_normalize(char* path);

typedef struct hyle_container_string_buf_o hyle_container_string_buf;
typedef uint64_t hyle_container_intern_id;
HYLE_CONTAINER_C_API hyle_status hyle_container_string_buf_create(
    hyle_container_string_buf** out, hyle_allocator backing);
static inline hyle_status hyle_container_string_buf_create_system(
    hyle_container_string_buf** out) {
  return hyle_container_string_buf_create(out, hyle_system_allocator());
}
HYLE_CONTAINER_C_API void hyle_container_string_buf_destroy(
    hyle_container_string_buf* buffer);
HYLE_CONTAINER_C_API hyle_status hyle_container_string_buf_add(
    hyle_container_intern_id* out, hyle_container_string_buf* buffer,
    hyle_container_string_view text);
HYLE_CONTAINER_C_API hyle_status hyle_container_string_buf_add_many(
    size_t* out_completed,
    hyle_container_string_buf* buffer,
    const hyle_container_string_view* input,
    hyle_container_intern_id* output, size_t count);
HYLE_CONTAINER_C_API int hyle_container_string_buf_retain(
    hyle_container_string_buf* buffer, hyle_container_intern_id id);
HYLE_CONTAINER_C_API size_t hyle_container_string_buf_retain_many(
    hyle_container_string_buf* buffer,
    const hyle_container_intern_id* ids, size_t count);
HYLE_CONTAINER_C_API int hyle_container_string_buf_remove(
    hyle_container_string_buf* buffer, hyle_container_intern_id id);
HYLE_CONTAINER_C_API size_t hyle_container_string_buf_remove_many(
    hyle_container_string_buf* buffer,
    const hyle_container_intern_id* ids, size_t count);
HYLE_CONTAINER_C_API hyle_container_string_view
hyle_container_string_buf_lookup(hyle_container_string_buf* buffer,
                                 hyle_container_intern_id id);
HYLE_CONTAINER_C_API hyle_status hyle_container_string_buf_intern(
    hyle_container_string_view* out, hyle_container_string_buf* buffer,
    hyle_container_string_view text);
HYLE_CONTAINER_C_API hyle_status hyle_container_string_buf_intern_many(
    size_t* out_completed,
    hyle_container_string_buf* buffer,
    const hyle_container_string_view* input,
    hyle_container_string_view* output, size_t count);
HYLE_CONTAINER_C_API uint32_t hyle_container_string_buf_size(
    const hyle_container_string_buf* buffer);

/* Runtime-erased trivially-copyable vector (the CArray capability). */
typedef struct hyle_container_vector_o hyle_container_vector;
typedef struct hyle_container_vector_options {
  uint32_t struct_size;
  uint32_t reserved;
  size_t item_size;
  size_t item_alignment;
  size_t initial_capacity;
  hyle_allocator backing;
} hyle_container_vector_options;
#define HYLE_CONTAINER_VECTOR_OPTIONS_INIT                                    \
  {sizeof(hyle_container_vector_options), 0, 0, 0, 0, hyle_system_allocator()}
#define HYLE_CONTAINER_VECTOR_OPTIONS_FOR(type, capacity)                      \
  {sizeof(hyle_container_vector_options), 0, sizeof(type),                    \
   HYLE_CONTAINER_ALIGNOF(type), (capacity), hyle_system_allocator()}
HYLE_CONTAINER_C_API hyle_status hyle_container_vector_create(
    hyle_container_vector** out,
    const hyle_container_vector_options* options);
HYLE_CONTAINER_C_API void hyle_container_vector_destroy(
    hyle_container_vector* vector);
HYLE_CONTAINER_C_API size_t hyle_container_vector_size(
    const hyle_container_vector* vector);
HYLE_CONTAINER_C_API size_t hyle_container_vector_capacity(
    const hyle_container_vector* vector);
HYLE_CONTAINER_C_API void* hyle_container_vector_data(
    hyle_container_vector* vector);
HYLE_CONTAINER_C_API const void* hyle_container_vector_data_const(
    const hyle_container_vector* vector);
HYLE_CONTAINER_C_API void* hyle_container_vector_at(
    hyle_container_vector* vector, size_t index);
HYLE_CONTAINER_C_API hyle_status hyle_container_vector_reserve(
    hyle_container_vector* vector, size_t capacity);
HYLE_CONTAINER_C_API hyle_status hyle_container_vector_push(
    hyle_container_vector* vector, const void* item);
HYLE_CONTAINER_C_API int hyle_container_vector_pop(
    hyle_container_vector* vector, void* out_item);
HYLE_CONTAINER_C_API void hyle_container_vector_clear(
    hyle_container_vector* vector);

/* Runtime-erased POD hash map. hash/equal may be NULL for hash_bytes/memcmp. */
typedef uint64_t (*hyle_container_hash_fn)(const void* key, size_t key_size,
                                           void* context);
typedef int (*hyle_container_equal_fn)(const void* left, const void* right,
                                       size_t key_size, void* context);
typedef int (*hyle_container_hash_map_visit_fn)(
    const void* key, void* value, void* context);
typedef struct hyle_container_hash_map_o hyle_container_hash_map;
typedef struct hyle_container_hash_map_options {
  uint32_t struct_size;
  uint32_t reserved;
  size_t key_size;
  size_t key_alignment;
  size_t value_size;
  size_t value_alignment;
  hyle_container_hash_fn hash;
  hyle_container_equal_fn equal;
  void* context;
  hyle_allocator backing;
} hyle_container_hash_map_options;
#define HYLE_CONTAINER_HASH_MAP_OPTIONS_INIT                                  \
  {sizeof(hyle_container_hash_map_options), 0, 0, 0, 0, 0, NULL, NULL, NULL,  \
   hyle_system_allocator()}
#define HYLE_CONTAINER_HASH_MAP_OPTIONS_FOR(key_type, value_type)              \
  {sizeof(hyle_container_hash_map_options), 0, sizeof(key_type),              \
   HYLE_CONTAINER_ALIGNOF(key_type), sizeof(value_type),                      \
   HYLE_CONTAINER_ALIGNOF(value_type), NULL, NULL, NULL,                       \
   hyle_system_allocator()}
HYLE_CONTAINER_C_API hyle_status hyle_container_hash_map_create(
    hyle_container_hash_map** out,
    const hyle_container_hash_map_options* options);
HYLE_CONTAINER_C_API void hyle_container_hash_map_destroy(
    hyle_container_hash_map* map);
HYLE_CONTAINER_C_API size_t hyle_container_hash_map_size(
    const hyle_container_hash_map* map);
HYLE_CONTAINER_C_API size_t hyle_container_hash_map_capacity(
    const hyle_container_hash_map* map);
HYLE_CONTAINER_C_API hyle_status hyle_container_hash_map_reserve(
    hyle_container_hash_map* map, size_t count);
HYLE_CONTAINER_C_API void hyle_container_hash_map_clear(
    hyle_container_hash_map* map);
typedef enum hyle_container_insert_result {
  HYLE_CONTAINER_INSERTED = 0,
  HYLE_CONTAINER_ALREADY_PRESENT = 1,
  HYLE_CONTAINER_ASSIGNED = 2
} hyle_container_insert_result;
HYLE_CONTAINER_C_API hyle_status hyle_container_hash_map_insert(
    hyle_container_insert_result* out_result,
    hyle_container_hash_map* map, const void* key, const void* value);
HYLE_CONTAINER_C_API hyle_status hyle_container_hash_map_insert_or_assign(
    hyle_container_insert_result* out_result,
    hyle_container_hash_map* map, const void* key, const void* value);
HYLE_CONTAINER_C_API void* hyle_container_hash_map_find(
    hyle_container_hash_map* map, const void* key);
HYLE_CONTAINER_C_API const void* hyle_container_hash_map_find_const(
    const hyle_container_hash_map* map, const void* key);
HYLE_CONTAINER_C_API int hyle_container_hash_map_contains(
    const hyle_container_hash_map* map, const void* key);
HYLE_CONTAINER_C_API int hyle_container_hash_map_erase(
    hyle_container_hash_map* map, const void* key);
HYLE_CONTAINER_C_API int hyle_container_hash_map_for_each(
    hyle_container_hash_map* map, hyle_container_hash_map_visit_fn visitor,
    void* context);

/* Pointer payload queues: HYLE never owns or frees payload pointers. All
 * try_* operations are non-blocking; 0 means full/empty/contended as documented,
 * and an output item is written only on success. */
typedef struct hyle_container_bounded_queue_o hyle_container_bounded_queue;
HYLE_CONTAINER_C_API hyle_status hyle_container_bounded_queue_create(
    hyle_container_bounded_queue** out, hyle_allocator backing,
    size_t capacity);
static inline hyle_status hyle_container_bounded_queue_create_system(
    hyle_container_bounded_queue** out, size_t capacity) {
  return hyle_container_bounded_queue_create(out, hyle_system_allocator(),
                                              capacity);
}
HYLE_CONTAINER_C_API void hyle_container_bounded_queue_destroy(
    hyle_container_bounded_queue* queue);
HYLE_CONTAINER_C_API int hyle_container_bounded_queue_try_push(
    hyle_container_bounded_queue* queue, void* item);
HYLE_CONTAINER_C_API int hyle_container_bounded_queue_try_pop(
    hyle_container_bounded_queue* queue, void** out_item);
HYLE_CONTAINER_C_API size_t hyle_container_bounded_queue_capacity(
    const hyle_container_bounded_queue* queue);

typedef struct hyle_container_sharded_queue_o hyle_container_sharded_queue;
HYLE_CONTAINER_C_API hyle_status hyle_container_sharded_queue_create(
    hyle_container_sharded_queue** out, hyle_allocator backing,
    size_t shard_count, size_t capacity_per_shard);
static inline hyle_status hyle_container_sharded_queue_create_system(
    hyle_container_sharded_queue** out, size_t shard_count,
    size_t capacity_per_shard) {
  return hyle_container_sharded_queue_create(
      out, hyle_system_allocator(), shard_count, capacity_per_shard);
}
HYLE_CONTAINER_C_API void hyle_container_sharded_queue_destroy(
    hyle_container_sharded_queue* queue);
HYLE_CONTAINER_C_API int hyle_container_sharded_queue_try_push(
    hyle_container_sharded_queue* queue, size_t producer_hint, void* item);
HYLE_CONTAINER_C_API int hyle_container_sharded_queue_try_pop(
    hyle_container_sharded_queue* queue, size_t* cursor, void** out_item);
HYLE_CONTAINER_C_API size_t hyle_container_sharded_queue_capacity(
    const hyle_container_sharded_queue* queue);
HYLE_CONTAINER_C_API size_t hyle_container_sharded_queue_shard_count(
    const hyle_container_sharded_queue* queue);

typedef struct hyle_container_work_stealing_deque_o
    hyle_container_work_stealing_deque;
typedef enum hyle_container_steal_status {
  HYLE_CONTAINER_STEAL_OK = 0,
  HYLE_CONTAINER_STEAL_EMPTY = 1,
  HYLE_CONTAINER_STEAL_ABORT = 2
} hyle_container_steal_status;
HYLE_CONTAINER_C_API hyle_status hyle_container_work_stealing_deque_create(
    hyle_container_work_stealing_deque** out, hyle_allocator backing,
    size_t capacity);
static inline hyle_status hyle_container_work_stealing_deque_create_system(
    hyle_container_work_stealing_deque** out, size_t capacity) {
  return hyle_container_work_stealing_deque_create(
      out, hyle_system_allocator(), capacity);
}
HYLE_CONTAINER_C_API void hyle_container_work_stealing_deque_destroy(
    hyle_container_work_stealing_deque* deque);
/* push/pop bottom are owner-only. Any thread may try_steal_top. */
HYLE_CONTAINER_C_API int hyle_container_work_stealing_deque_try_push_bottom(
    hyle_container_work_stealing_deque* deque, void* item);
HYLE_CONTAINER_C_API int hyle_container_work_stealing_deque_try_pop_bottom(
    hyle_container_work_stealing_deque* deque, void** out_item);
HYLE_CONTAINER_C_API hyle_container_steal_status
hyle_container_work_stealing_deque_try_steal_top(
    hyle_container_work_stealing_deque* deque, void** out_item);
HYLE_CONTAINER_C_API size_t hyle_container_work_stealing_deque_capacity(
    const hyle_container_work_stealing_deque* deque);

typedef struct hyle_container_ws_fifo_queue_o hyle_container_ws_fifo_queue;
HYLE_CONTAINER_C_API hyle_status hyle_container_ws_fifo_queue_create(
    hyle_container_ws_fifo_queue** out, hyle_allocator backing,
    size_t capacity);
static inline hyle_status hyle_container_ws_fifo_queue_create_system(
    hyle_container_ws_fifo_queue** out, size_t capacity) {
  return hyle_container_ws_fifo_queue_create(out, hyle_system_allocator(),
                                              capacity);
}
HYLE_CONTAINER_C_API void hyle_container_ws_fifo_queue_destroy(
    hyle_container_ws_fifo_queue* queue);
/* The owner pushes at the back and pops at the front. A thief transfers a
 * batch from source into its own equal-capacity target and receives one item. */
HYLE_CONTAINER_C_API int hyle_container_ws_fifo_queue_try_push_back(
    hyle_container_ws_fifo_queue* queue, void* item);
HYLE_CONTAINER_C_API int hyle_container_ws_fifo_queue_try_pop_front(
    hyle_container_ws_fifo_queue* queue, void** out_item);
HYLE_CONTAINER_C_API int hyle_container_ws_fifo_queue_try_steal_into(
    hyle_container_ws_fifo_queue* source, hyle_container_ws_fifo_queue* target,
    void** out_item);
HYLE_CONTAINER_C_API size_t hyle_container_ws_fifo_queue_capacity(
    const hyle_container_ws_fifo_queue* queue);

typedef struct hyle_container_concurrent_vector_o
    hyle_container_concurrent_vector;
/* Multi-writer append-only pointer vector. size() counts reserved indices and
 * is not a completion fence for in-flight pushes. Read a newly appended index
 * only after its push returns, or quiesce writers before traversing [0,size()). */
HYLE_CONTAINER_C_API hyle_status hyle_container_concurrent_vector_create(
    hyle_container_concurrent_vector** out, hyle_allocator backing,
    size_t initial_capacity);
static inline hyle_status hyle_container_concurrent_vector_create_system(
    hyle_container_concurrent_vector** out, size_t initial_capacity) {
  return hyle_container_concurrent_vector_create(
      out, hyle_system_allocator(), initial_capacity);
}
HYLE_CONTAINER_C_API void hyle_container_concurrent_vector_destroy(
    hyle_container_concurrent_vector* vector);
HYLE_CONTAINER_C_API hyle_status hyle_container_concurrent_vector_push(
    size_t* out_index, hyle_container_concurrent_vector* vector, void* item);
HYLE_CONTAINER_C_API size_t hyle_container_concurrent_vector_size(
    const hyle_container_concurrent_vector* vector);
HYLE_CONTAINER_C_API size_t hyle_container_concurrent_vector_capacity(
    const hyle_container_concurrent_vector* vector);
HYLE_CONTAINER_C_API void* hyle_container_concurrent_vector_at(
    const hyle_container_concurrent_vector* vector, size_t index);

#ifdef __cplusplus
}
#endif
#endif /* HYLE_C_CONTAINER_H_ */
