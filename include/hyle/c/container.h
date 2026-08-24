/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary
 * Pure C ABI for HYLE container capabilities (pre-release draft).
 * Docs: docs/guide/c-container.md */
#ifndef HYLE_C_CONTAINER_H_
#define HYLE_C_CONTAINER_H_

#include <stddef.h>
#include <stdint.h>
#include <string.h>

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

/* Convenience for configuration paths that already have a C string. Length-
 * aware hot paths should keep and pass hyle_container_string_view directly. A
 * NULL input or a length that cannot fit the ABI view produces an invalid view
 * which consuming functions reject. */
static inline hyle_container_string_view hyle_container_cstr(
    const char* data) {
  hyle_container_string_view value = {data, HYLE_CONTAINER_NPOS, 0};
  if (data != NULL) {
    const size_t size = strlen(data);
    if (size < HYLE_CONTAINER_NPOS) {
      value.size = (uint32_t)size;
      value.null_terminated = 1;
    }
  }
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
 *   and initialize every non-NULL leading out parameter before validation or
 *   work: create -> NULL, insert -> ALREADY_PRESENT, interner acquire -> 0,
 *   interner intern -> empty view, batch completed -> 0, concurrent pointer
 *   append index -> SIZE_MAX;
 * - queue full/empty, map miss, and stale IDs are normal value results;
 * - borrowed views require size == 0 || data != NULL.
 * There is deliberately no module-private TLS last-error channel. */

/* Process-local non-cryptographic table hash. Repeatable inside this process,
 * but the value may change across HYLE versions, builds, or CPU ISA. Do not use
 * it as a persistent fingerprint, wire value, or adversarial-input MAC. */
HYLE_CONTAINER_C_API uint64_t hyle_container_hash_bytes(
    const void* data, size_t size, uint64_t seed);
/* SipHash-2-4 for attacker-controlled keys. key0/key1 are a caller-owned
 * unpredictable 128-bit secret; this stateless entry never stores them. */
HYLE_CONTAINER_C_API uint64_t hyle_container_siphash24(
    const void* data, size_t size, uint64_t key0, uint64_t key1);

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
HYLE_CONTAINER_C_API size_t hyle_container_string_c_escape(
    hyle_container_string_view text, char* out, size_t capacity);
/* One-pass validate-only path. Returns 1 for valid UTF-8 (including empty), 0
 * for malformed input or an invalid {NULL, nonzero} view. */
HYLE_CONTAINER_C_API int hyle_container_utf8_validate(
    hyle_container_string_view text);
/* Validate + count; valid input may require two dispatched passes. */
HYLE_CONTAINER_C_API int64_t hyle_container_utf8_length(
    hyle_container_string_view text);

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

typedef struct hyle_container_string_interner_o hyle_container_string_interner;
typedef uint64_t hyle_container_string_id;
HYLE_CONTAINER_C_API hyle_status hyle_container_string_interner_create(
    hyle_container_string_interner** out, hyle_allocator backing);
static inline hyle_status hyle_container_string_interner_create_system(
    hyle_container_string_interner** out) {
  return hyle_container_string_interner_create(out, hyle_system_allocator());
}
HYLE_CONTAINER_C_API void hyle_container_string_interner_destroy(
    hyle_container_string_interner* interner);
HYLE_CONTAINER_C_API hyle_status hyle_container_string_interner_acquire(
    hyle_container_string_id* out, hyle_container_string_interner* interner,
    hyle_container_string_view text);
HYLE_CONTAINER_C_API hyle_status hyle_container_string_interner_acquire_many(
    size_t* out_completed,
    hyle_container_string_interner* interner,
    const hyle_container_string_view* input,
    hyle_container_string_id* output, size_t count);
HYLE_CONTAINER_C_API int hyle_container_string_interner_retain(
    hyle_container_string_interner* interner, hyle_container_string_id id);
HYLE_CONTAINER_C_API size_t hyle_container_string_interner_retain_many(
    hyle_container_string_interner* interner,
    const hyle_container_string_id* ids, size_t count);
HYLE_CONTAINER_C_API int hyle_container_string_interner_release(
    hyle_container_string_interner* interner, hyle_container_string_id id);
HYLE_CONTAINER_C_API size_t hyle_container_string_interner_release_many(
    hyle_container_string_interner* interner,
    const hyle_container_string_id* ids, size_t count);
HYLE_CONTAINER_C_API hyle_container_string_view
hyle_container_string_interner_lookup(hyle_container_string_interner* interner,
                                      hyle_container_string_id id);
HYLE_CONTAINER_C_API hyle_status hyle_container_string_interner_intern(
    hyle_container_string_view* out, hyle_container_string_interner* interner,
    hyle_container_string_view text);
HYLE_CONTAINER_C_API hyle_status hyle_container_string_interner_intern_many(
    size_t* out_completed,
    hyle_container_string_interner* interner,
    const hyle_container_string_view* input,
    hyle_container_string_view* output, size_t count);
HYLE_CONTAINER_C_API uint32_t hyle_container_string_interner_size(
    const hyle_container_string_interner* interner);

/* Runtime-sized dense bit set. Bulk algebra is SIMD-dispatched for large
 * masks; find_first/find_next support allocation-free set-bit iteration.
 * Index mutators return zero for NULL/out-of-range input. */
#define HYLE_CONTAINER_BITSET_NPOS SIZE_MAX
typedef struct hyle_container_dynamic_bitset_o hyle_container_dynamic_bitset;
HYLE_CONTAINER_C_API hyle_status hyle_container_dynamic_bitset_create(
    hyle_container_dynamic_bitset** out, size_t bit_count, int initial_value,
    hyle_allocator backing);
static inline hyle_status hyle_container_dynamic_bitset_create_system(
    hyle_container_dynamic_bitset** out, size_t bit_count, int initial_value) {
  return hyle_container_dynamic_bitset_create(
      out, bit_count, initial_value, hyle_system_allocator());
}
HYLE_CONTAINER_C_API void hyle_container_dynamic_bitset_destroy(
    hyle_container_dynamic_bitset* set);
HYLE_CONTAINER_C_API hyle_status hyle_container_dynamic_bitset_reserve(
    hyle_container_dynamic_bitset* set, size_t bit_count);
HYLE_CONTAINER_C_API hyle_status hyle_container_dynamic_bitset_resize(
    hyle_container_dynamic_bitset* set, size_t bit_count, int value);
HYLE_CONTAINER_C_API size_t hyle_container_dynamic_bitset_size(
    const hyle_container_dynamic_bitset* set);
HYLE_CONTAINER_C_API size_t hyle_container_dynamic_bitset_capacity(
    const hyle_container_dynamic_bitset* set);
HYLE_CONTAINER_C_API int hyle_container_dynamic_bitset_test(
    const hyle_container_dynamic_bitset* set, size_t index);
HYLE_CONTAINER_C_API int hyle_container_dynamic_bitset_set(
    hyle_container_dynamic_bitset* set, size_t index, int value);
HYLE_CONTAINER_C_API int hyle_container_dynamic_bitset_flip(
    hyle_container_dynamic_bitset* set, size_t index);
HYLE_CONTAINER_C_API void hyle_container_dynamic_bitset_set_all(
    hyle_container_dynamic_bitset* set);
HYLE_CONTAINER_C_API void hyle_container_dynamic_bitset_reset_all(
    hyle_container_dynamic_bitset* set);
HYLE_CONTAINER_C_API void hyle_container_dynamic_bitset_flip_all(
    hyle_container_dynamic_bitset* set);
HYLE_CONTAINER_C_API void hyle_container_dynamic_bitset_shift_left(
    hyle_container_dynamic_bitset* set, size_t shift);
HYLE_CONTAINER_C_API void hyle_container_dynamic_bitset_shift_right(
    hyle_container_dynamic_bitset* set, size_t shift);
HYLE_CONTAINER_C_API size_t hyle_container_dynamic_bitset_count(
    const hyle_container_dynamic_bitset* set);
HYLE_CONTAINER_C_API int hyle_container_dynamic_bitset_any(
    const hyle_container_dynamic_bitset* set);
HYLE_CONTAINER_C_API int hyle_container_dynamic_bitset_all(
    const hyle_container_dynamic_bitset* set);
HYLE_CONTAINER_C_API size_t hyle_container_dynamic_bitset_find_first(
    const hyle_container_dynamic_bitset* set);
HYLE_CONTAINER_C_API size_t hyle_container_dynamic_bitset_find_next(
    const hyle_container_dynamic_bitset* set, size_t index);
HYLE_CONTAINER_C_API hyle_status hyle_container_dynamic_bitset_and(
    hyle_container_dynamic_bitset* destination,
    const hyle_container_dynamic_bitset* source);
HYLE_CONTAINER_C_API hyle_status hyle_container_dynamic_bitset_or(
    hyle_container_dynamic_bitset* destination,
    const hyle_container_dynamic_bitset* source);
HYLE_CONTAINER_C_API hyle_status hyle_container_dynamic_bitset_xor(
    hyle_container_dynamic_bitset* destination,
    const hyle_container_dynamic_bitset* source);
HYLE_CONTAINER_C_API hyle_status hyle_container_dynamic_bitset_and_not(
    hyle_container_dynamic_bitset* destination,
    const hyle_container_dynamic_bitset* source);
HYLE_CONTAINER_C_API int hyle_container_dynamic_bitset_intersects(
    const hyle_container_dynamic_bitset* left,
    const hyle_container_dynamic_bitset* right);
HYLE_CONTAINER_C_API int hyle_container_dynamic_bitset_is_subset(
    const hyle_container_dynamic_bitset* subset,
    const hyle_container_dynamic_bitset* superset);

/* uint32 dense/sparse ID set. The universe is fixed between reset calls;
 * insert/erase/contains/clear are O(1), insert does not allocate, and erase
 * uses swap-remove (iteration order is not stable). */
typedef struct hyle_container_sparse_u32_set_o hyle_container_sparse_u32_set;
HYLE_CONTAINER_C_API hyle_status hyle_container_sparse_u32_set_create(
    hyle_container_sparse_u32_set** out, size_t universe_size,
    hyle_allocator backing);
static inline hyle_status hyle_container_sparse_u32_set_create_system(
    hyle_container_sparse_u32_set** out, size_t universe_size) {
  return hyle_container_sparse_u32_set_create(
      out, universe_size, hyle_system_allocator());
}
HYLE_CONTAINER_C_API void hyle_container_sparse_u32_set_destroy(
    hyle_container_sparse_u32_set* set);
HYLE_CONTAINER_C_API hyle_status hyle_container_sparse_u32_set_reset_universe(
    hyle_container_sparse_u32_set* set, size_t universe_size);
HYLE_CONTAINER_C_API size_t hyle_container_sparse_u32_set_universe_size(
    const hyle_container_sparse_u32_set* set);
HYLE_CONTAINER_C_API size_t hyle_container_sparse_u32_set_size(
    const hyle_container_sparse_u32_set* set);
HYLE_CONTAINER_C_API const uint32_t* hyle_container_sparse_u32_set_data(
    const hyle_container_sparse_u32_set* set);
HYLE_CONTAINER_C_API int hyle_container_sparse_u32_set_contains(
    const hyle_container_sparse_u32_set* set, uint32_t id);
/* Returns 1 only when inserted; returns 0 for duplicate, NULL, or out-of-range. */
HYLE_CONTAINER_C_API int hyle_container_sparse_u32_set_insert(
    hyle_container_sparse_u32_set* set, uint32_t id);
HYLE_CONTAINER_C_API int hyle_container_sparse_u32_set_erase(
    hyle_container_sparse_u32_set* set, uint32_t id);
HYLE_CONTAINER_C_API void hyle_container_sparse_u32_set_clear(
    hyle_container_sparse_u32_set* set);

/* Single-thread byte ring for stream/network staging. Capacity is rounded up
 * to a power of two. readable/writable return at most two spans; their pointers
 * are invalidated by reserve or destroy, and their logical availability changes
 * after any write/read/commit/consume/clear operation. */
typedef struct hyle_container_byte_ring_o hyle_container_byte_ring;
typedef struct hyle_container_const_byte_span {
  const void* data;
  size_t size;
} hyle_container_const_byte_span;
typedef struct hyle_container_mutable_byte_span {
  void* data;
  size_t size;
} hyle_container_mutable_byte_span;
HYLE_CONTAINER_C_API hyle_status hyle_container_byte_ring_create(
    hyle_container_byte_ring** out, size_t capacity, hyle_allocator backing);
static inline hyle_status
hyle_container_byte_ring_create_system(hyle_container_byte_ring** out,
                                       size_t capacity) {
  return hyle_container_byte_ring_create(out, capacity,
                                         hyle_system_allocator());
}
HYLE_CONTAINER_C_API void
hyle_container_byte_ring_destroy(hyle_container_byte_ring* ring);
HYLE_CONTAINER_C_API hyle_status hyle_container_byte_ring_reserve(
    hyle_container_byte_ring* ring, size_t capacity);
HYLE_CONTAINER_C_API size_t
hyle_container_byte_ring_size(const hyle_container_byte_ring* ring);
HYLE_CONTAINER_C_API size_t
hyle_container_byte_ring_capacity(const hyle_container_byte_ring* ring);
HYLE_CONTAINER_C_API size_t
hyle_container_byte_ring_free_space(const hyle_container_byte_ring* ring);
/* Copy APIs may complete partially and return the byte count transferred. */
HYLE_CONTAINER_C_API size_t hyle_container_byte_ring_write(
    hyle_container_byte_ring* ring, const void* data, size_t size);
HYLE_CONTAINER_C_API size_t hyle_container_byte_ring_read(
    hyle_container_byte_ring* ring, void* output, size_t capacity);
/* out_spans must point to two entries. Returns the number of non-empty spans. */
HYLE_CONTAINER_C_API size_t
hyle_container_byte_ring_readable(const hyle_container_byte_ring* ring,
                                  hyle_container_const_byte_span out_spans[2]);
HYLE_CONTAINER_C_API size_t hyle_container_byte_ring_writable(
    hyle_container_byte_ring* ring,
    hyle_container_mutable_byte_span out_spans[2]);
/* Return zero without mutation when count exceeds current readable/writable
 * availability. */
HYLE_CONTAINER_C_API int
hyle_container_byte_ring_commit_write(hyle_container_byte_ring* ring,
                                      size_t count);
HYLE_CONTAINER_C_API int
hyle_container_byte_ring_consume(hyle_container_byte_ring* ring, size_t count);
HYLE_CONTAINER_C_API void
hyle_container_byte_ring_clear(hyle_container_byte_ring* ring);

/* Runtime-erased trivially-copyable vector (the PodVector capability).
 * Versioned options rules used by every options struct below:
 * - struct_size and reserved are in the required prefix; reserved must be zero.
 * - struct_size must cover the family-specific REQUIRED_SIZE.
 * - each fully present trailing field is read; a missing/partial field keeps
 *   the documented default, and bytes beyond the current struct are ignored.
 * - callers must make struct_size bytes readable at the supplied address. */
typedef struct hyle_container_vector_o hyle_container_vector;
typedef struct hyle_container_vector_options {
  uint32_t struct_size;
  uint32_t reserved;
  size_t item_size;
  size_t item_alignment;
  size_t initial_capacity;
  hyle_allocator backing;
} hyle_container_vector_options;
#define HYLE_CONTAINER_VECTOR_OPTIONS_REQUIRED_SIZE                           \
  ((uint32_t)offsetof(hyle_container_vector_options, initial_capacity))
#define HYLE_CONTAINER_VECTOR_OPTIONS_INIT                                    \
  {(uint32_t)sizeof(hyle_container_vector_options), 0, 0, 0, 0,               \
   hyle_system_allocator()}
#define HYLE_CONTAINER_VECTOR_OPTIONS_FOR(type, capacity)                      \
  {(uint32_t)sizeof(hyle_container_vector_options), 0, sizeof(type),          \
   HYLE_CONTAINER_ALIGNOF(type), (capacity), hyle_system_allocator()}
/* initial_capacity defaults to zero and backing defaults to the system
 * allocator when their complete fields are outside struct_size. */
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
/* Tail mutation. try_push_back reports allocation failure through Status;
 * try_pop_back returns 0 when empty and writes out_item only on success. */
HYLE_CONTAINER_C_API hyle_status hyle_container_vector_try_push_back(
    hyle_container_vector* vector, const void* item);
/* Batch mutation uses item-count, not byte-count. A non-zero count requires a
 * pointer to count contiguous items. Input may alias the vector's live data. */
HYLE_CONTAINER_C_API hyle_status hyle_container_vector_try_append(
    hyle_container_vector* vector, const void* items, size_t count);
/* fill_item is ignored when shrinking and required when growing. */
HYLE_CONTAINER_C_API hyle_status hyle_container_vector_try_resize(
    hyle_container_vector* vector, size_t size, const void* fill_item);
HYLE_CONTAINER_C_API hyle_status hyle_container_vector_try_insert(
    hyle_container_vector* vector, size_t index, const void* items,
    size_t count);
HYLE_CONTAINER_C_API int hyle_container_vector_try_pop_back(
    hyle_container_vector* vector, void* out_item);
HYLE_CONTAINER_C_API hyle_status hyle_container_vector_erase(
    hyle_container_vector* vector, size_t index, size_t count);
HYLE_CONTAINER_C_API hyle_status hyle_container_vector_try_shrink_to_fit(
    hyle_container_vector* vector);
HYLE_CONTAINER_C_API void hyle_container_vector_clear(
    hyle_container_vector* vector);

/* Runtime-erased POD hash map. NULL hash + NULL equal selects the
 * process-seeded byte hash + memcmp defaults. A custom hash may retain memcmp
 * by leaving equal NULL. A custom equal REQUIRES a compatible custom hash;
 * create rejects equal != NULL && hash == NULL because every equivalent pair
 * must produce the same hash. */
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
#define HYLE_CONTAINER_HASH_MAP_OPTIONS_REQUIRED_SIZE                         \
  ((uint32_t)offsetof(hyle_container_hash_map_options, hash))
#define HYLE_CONTAINER_HASH_MAP_OPTIONS_INIT                                  \
  {(uint32_t)sizeof(hyle_container_hash_map_options), 0, 0, 0, 0, 0, NULL,    \
   NULL, NULL, hyle_system_allocator()}
#define HYLE_CONTAINER_HASH_MAP_OPTIONS_FOR(key_type, value_type)              \
  {(uint32_t)sizeof(hyle_container_hash_map_options), 0, sizeof(key_type),    \
   HYLE_CONTAINER_ALIGNOF(key_type), sizeof(value_type),                      \
   HYLE_CONTAINER_ALIGNOF(value_type), NULL, NULL, NULL,                       \
   hyle_system_allocator()}
/* hash/equal/context default to NULL and backing defaults to the system
 * allocator when their complete fields are outside struct_size. */
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
/* Fixed-width across compilers and -fshort-enums. */
typedef uint32_t hyle_container_insert_result;
enum {
  HYLE_CONTAINER_INSERTED = 0,
  HYLE_CONTAINER_ALREADY_PRESENT = 1,
  HYLE_CONTAINER_ASSIGNED = 2
};
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
/* Visits occupied entries until visitor returns 0. Returns 1 only when the
 * complete table was visited; returns 0 for early stop or invalid arguments. */
HYLE_CONTAINER_C_API int hyle_container_hash_map_visit(
    hyle_container_hash_map* map, hyle_container_hash_map_visit_fn visitor,
    void* context);

/* Sharded concurrent POD hash map. Unlike the single-thread map, lookup never
 * returns an unguarded pointer: find_copy copies while holding a read lock and
 * update invokes a callback while holding the key's write lock. hash/equal and
 * callbacks can run concurrently on different shards and must be thread-safe.
 * Destroy requires all calls and callbacks to be quiescent. */
typedef struct hyle_container_concurrent_hash_map_o
    hyle_container_concurrent_hash_map;
typedef int (*hyle_container_concurrent_hash_map_visit_fn)(
    const void* key, const void* value, void* context);
typedef void (*hyle_container_concurrent_hash_map_update_fn)(
    void* value, void* context);
HYLE_CONTAINER_C_API hyle_status hyle_container_concurrent_hash_map_create(
    hyle_container_concurrent_hash_map** out,
    const hyle_container_hash_map_options* options, size_t shard_count);
HYLE_CONTAINER_C_API void hyle_container_concurrent_hash_map_destroy(
    hyle_container_concurrent_hash_map* map);
HYLE_CONTAINER_C_API size_t hyle_container_concurrent_hash_map_shard_count(
    const hyle_container_concurrent_hash_map* map);
HYLE_CONTAINER_C_API size_t hyle_container_concurrent_hash_map_size(
    const hyle_container_concurrent_hash_map* map);
HYLE_CONTAINER_C_API size_t hyle_container_concurrent_hash_map_capacity(
    const hyle_container_concurrent_hash_map* map);
HYLE_CONTAINER_C_API hyle_status hyle_container_concurrent_hash_map_reserve(
    hyle_container_concurrent_hash_map* map, size_t count);
HYLE_CONTAINER_C_API void hyle_container_concurrent_hash_map_clear(
    hyle_container_concurrent_hash_map* map);
HYLE_CONTAINER_C_API hyle_status hyle_container_concurrent_hash_map_insert(
    hyle_container_insert_result* out_result,
    hyle_container_concurrent_hash_map* map, const void* key,
    const void* value);
HYLE_CONTAINER_C_API hyle_status
hyle_container_concurrent_hash_map_insert_or_assign(
    hyle_container_insert_result* out_result,
    hyle_container_concurrent_hash_map* map, const void* key,
    const void* value);
/* Returns 1 and writes exactly options.value_size bytes when present; otherwise
 * returns 0 and leaves out_value unchanged. */
HYLE_CONTAINER_C_API int hyle_container_concurrent_hash_map_find_copy(
    void* out_value, const hyle_container_concurrent_hash_map* map,
    const void* key);
/* Snapshot only: another thread may erase immediately after the return. */
HYLE_CONTAINER_C_API int hyle_container_concurrent_hash_map_contains(
    const hyle_container_concurrent_hash_map* map, const void* key);
/* Runs updater under exclusive ownership of this key's shard. Returns 0 when
 * absent/invalid. updater must not re-enter the same map. */
HYLE_CONTAINER_C_API int hyle_container_concurrent_hash_map_update(
    hyle_container_concurrent_hash_map* map, const void* key,
    hyle_container_concurrent_hash_map_update_fn updater, void* context);
HYLE_CONTAINER_C_API int hyle_container_concurrent_hash_map_erase(
    hyle_container_concurrent_hash_map* map, const void* key);
/* Weakly consistent across shards; each visited key/value pair is stable during
 * its callback. Returns 1 only after a complete visit. No same-map re-entry. */
HYLE_CONTAINER_C_API int hyle_container_concurrent_hash_map_visit(
    const hyle_container_concurrent_hash_map* map,
    hyle_container_concurrent_hash_map_visit_fn visitor, void* context);

/* Runtime-erased POD hash set. Stores one canonical key per member and does
 * not allocate a dummy mapped-value field in each flat slot. The map callback
 * rule also applies: a custom equal requires a compatible custom hash. */
typedef int (*hyle_container_hash_set_visit_fn)(const void* key,
                                                 void* context);
typedef struct hyle_container_hash_set_o hyle_container_hash_set;
typedef struct hyle_container_hash_set_options {
  uint32_t struct_size;
  uint32_t reserved;
  size_t key_size;
  size_t key_alignment;
  hyle_container_hash_fn hash;
  hyle_container_equal_fn equal;
  void* context;
  hyle_allocator backing;
} hyle_container_hash_set_options;
#define HYLE_CONTAINER_HASH_SET_OPTIONS_REQUIRED_SIZE                         \
  ((uint32_t)offsetof(hyle_container_hash_set_options, hash))
#define HYLE_CONTAINER_HASH_SET_OPTIONS_INIT                                  \
  {(uint32_t)sizeof(hyle_container_hash_set_options), 0, 0, 0, NULL, NULL,    \
   NULL, hyle_system_allocator()}
#define HYLE_CONTAINER_HASH_SET_OPTIONS_FOR(key_type)                         \
  {(uint32_t)sizeof(hyle_container_hash_set_options), 0, sizeof(key_type),    \
   HYLE_CONTAINER_ALIGNOF(key_type), NULL, NULL, NULL,                        \
   hyle_system_allocator()}
/* hash/equal/context default to NULL and backing defaults to the system
 * allocator when their complete fields are outside struct_size. */
HYLE_CONTAINER_C_API hyle_status hyle_container_hash_set_create(
    hyle_container_hash_set** out,
    const hyle_container_hash_set_options* options);
HYLE_CONTAINER_C_API void hyle_container_hash_set_destroy(
    hyle_container_hash_set* set);
HYLE_CONTAINER_C_API size_t hyle_container_hash_set_size(
    const hyle_container_hash_set* set);
HYLE_CONTAINER_C_API size_t hyle_container_hash_set_capacity(
    const hyle_container_hash_set* set);
HYLE_CONTAINER_C_API hyle_status hyle_container_hash_set_reserve(
    hyle_container_hash_set* set, size_t count);
HYLE_CONTAINER_C_API void hyle_container_hash_set_clear(
    hyle_container_hash_set* set);
HYLE_CONTAINER_C_API hyle_status hyle_container_hash_set_insert(
    hyle_container_insert_result* out_result,
    hyle_container_hash_set* set, const void* key);
HYLE_CONTAINER_C_API const void* hyle_container_hash_set_find(
    const hyle_container_hash_set* set, const void* key);
HYLE_CONTAINER_C_API int hyle_container_hash_set_contains(
    const hyle_container_hash_set* set, const void* key);
HYLE_CONTAINER_C_API int hyle_container_hash_set_erase(
    hyle_container_hash_set* set, const void* key);
/* Visits occupied keys until visitor returns 0. Returns 1 only when the
 * complete set was visited; returns 0 for early stop or invalid arguments. */
HYLE_CONTAINER_C_API int hyle_container_hash_set_visit(
    const hyle_container_hash_set* set,
    hyle_container_hash_set_visit_fn visitor, void* context);

/* Mutable ordered POD indexes. The comparator must define a strict weak order
 * and return <0 / 0 / >0. HYLE stores key/value bytes by value; pointers
 * returned from find/cursor access and every cursor are invalidated by any
 * successful mutation. The numeric helpers avoid alignment/aliasing hazards
 * and require key_size to match their suffix exactly (i32/u32: 4 bytes,
 * i64/u64: 8 bytes); B-tree create rejects a mismatched options layout. */
typedef int (*hyle_container_compare_fn)(const void *left, const void *right,
                                         size_t key_size, void *context);
HYLE_CONTAINER_C_API int hyle_container_compare_i32(const void *left,
                                                    const void *right,
                                                    size_t key_size,
                                                    void *context);
HYLE_CONTAINER_C_API int hyle_container_compare_u32(const void *left,
                                                    const void *right,
                                                    size_t key_size,
                                                    void *context);
HYLE_CONTAINER_C_API int hyle_container_compare_i64(const void *left,
                                                    const void *right,
                                                    size_t key_size,
                                                    void *context);
HYLE_CONTAINER_C_API int hyle_container_compare_u64(const void *left,
                                                    const void *right,
                                                    size_t key_size,
                                                    void *context);
HYLE_CONTAINER_C_API int hyle_container_compare_bytes(const void *left,
                                                      const void *right,
                                                      size_t key_size,
                                                      void *context);

typedef struct hyle_container_btree_options {
  uint32_t struct_size;
  uint32_t reserved;
  size_t key_size;
  size_t key_alignment;
  size_t value_size;
  size_t value_alignment;
  hyle_container_compare_fn compare;
  void *context;
  hyle_allocator backing;
} hyle_container_btree_options;
#define HYLE_CONTAINER_BTREE_OPTIONS_REQUIRED_SIZE                            \
  ((uint32_t)offsetof(hyle_container_btree_options, context))
#define HYLE_CONTAINER_BTREE_OPTIONS_INIT                                      \
  {(uint32_t)sizeof(hyle_container_btree_options),                             \
   0,                                                                          \
   0,                                                                          \
   0,                                                                          \
   0,                                                                          \
   0,                                                                          \
   NULL,                                                                       \
   NULL,                                                                       \
   hyle_system_allocator()}
#define HYLE_CONTAINER_BTREE_MAP_OPTIONS_FOR(key_type, value_type, compare_fn) \
  {(uint32_t)sizeof(hyle_container_btree_options),                             \
   0,                                                                          \
   sizeof(key_type),                                                           \
   HYLE_CONTAINER_ALIGNOF(key_type),                                           \
   sizeof(value_type),                                                         \
   HYLE_CONTAINER_ALIGNOF(value_type),                                         \
   compare_fn,                                                                 \
   NULL,                                                                       \
   hyle_system_allocator()}
#define HYLE_CONTAINER_BTREE_SET_OPTIONS_FOR(key_type, compare_fn)             \
  {(uint32_t)sizeof(hyle_container_btree_options),                             \
   0,                                                                          \
   sizeof(key_type),                                                           \
   HYLE_CONTAINER_ALIGNOF(key_type),                                           \
   0,                                                                          \
   1,                                                                          \
   compare_fn,                                                                 \
   NULL,                                                                       \
   hyle_system_allocator()}
/* compare is part of the required prefix and must be non-NULL. context
 * defaults to NULL and backing defaults to the system allocator when absent. */

/* Trivial cursor representation; fields are opaque. end is represented by a
 * valid owner and a NULL node. previous(end) reaches the last element. */
typedef struct hyle_container_btree_cursor {
  const void *_owner;
  const void *_node;
  size_t _index;
} hyle_container_btree_cursor;

typedef struct hyle_container_btree_map_o hyle_container_btree_map;
typedef int (*hyle_container_btree_map_visit_fn)(const void *key, void *value,
                                                 void *context);
HYLE_CONTAINER_C_API hyle_status
hyle_container_btree_map_create(hyle_container_btree_map **out,
                                const hyle_container_btree_options *options);
HYLE_CONTAINER_C_API void
hyle_container_btree_map_destroy(hyle_container_btree_map *map);
HYLE_CONTAINER_C_API size_t
hyle_container_btree_map_size(const hyle_container_btree_map *map);
HYLE_CONTAINER_C_API void
hyle_container_btree_map_clear(hyle_container_btree_map *map);
/* Both map insert forms initialize a non-NULL out_result to ALREADY_PRESENT
 * before validation or growth, so invalid/OOM returns never expose
 * caller-stale result bytes. */
HYLE_CONTAINER_C_API hyle_status hyle_container_btree_map_insert(
    hyle_container_insert_result *out_result, hyle_container_btree_map *map,
    const void *key, const void *value);
HYLE_CONTAINER_C_API hyle_status hyle_container_btree_map_insert_or_assign(
    hyle_container_insert_result *out_result, hyle_container_btree_map *map,
    const void *key, const void *value);
HYLE_CONTAINER_C_API void *
hyle_container_btree_map_find(hyle_container_btree_map *map, const void *key);
HYLE_CONTAINER_C_API const void *
hyle_container_btree_map_find_const(const hyle_container_btree_map *map,
                                    const void *key);
HYLE_CONTAINER_C_API int
hyle_container_btree_map_contains(const hyle_container_btree_map *map,
                                  const void *key);
HYLE_CONTAINER_C_API int
hyle_container_btree_map_erase(hyle_container_btree_map *map, const void *key);
/* Erases [lower, upper) and returns the number removed. A NULL bound is
 * unbounded. If anything is removed, all pointers and cursors are invalid. */
HYLE_CONTAINER_C_API size_t hyle_container_btree_map_erase_range(
    hyle_container_btree_map *map, const void *lower, const void *upper);
HYLE_CONTAINER_C_API int
hyle_container_btree_map_begin(const hyle_container_btree_map *map,
                               hyle_container_btree_cursor *out_cursor);
HYLE_CONTAINER_C_API int
hyle_container_btree_map_lower_bound(const hyle_container_btree_map *map,
                                     const void *key,
                                     hyle_container_btree_cursor *out_cursor);
HYLE_CONTAINER_C_API int
hyle_container_btree_map_upper_bound(const hyle_container_btree_map *map,
                                     const void *key,
                                     hyle_container_btree_cursor *out_cursor);
HYLE_CONTAINER_C_API int hyle_container_btree_map_cursor_valid(
    const hyle_container_btree_map *map,
    const hyle_container_btree_cursor *cursor);
HYLE_CONTAINER_C_API const void *
hyle_container_btree_map_cursor_key(const hyle_container_btree_map *map,
                                    const hyle_container_btree_cursor *cursor);
HYLE_CONTAINER_C_API void *hyle_container_btree_map_cursor_value(
    hyle_container_btree_map *map, const hyle_container_btree_cursor *cursor);
HYLE_CONTAINER_C_API int
hyle_container_btree_map_cursor_next(const hyle_container_btree_map *map,
                                     hyle_container_btree_cursor *cursor);
HYLE_CONTAINER_C_API int
hyle_container_btree_map_cursor_previous(const hyle_container_btree_map *map,
                                         hyle_container_btree_cursor *cursor);
/* NULL lower/upper means unbounded. Bounds are [lower, upper). */
HYLE_CONTAINER_C_API int hyle_container_btree_map_visit_range(
    hyle_container_btree_map *map, const void *lower, const void *upper,
    hyle_container_btree_map_visit_fn visitor, void *context);

typedef struct hyle_container_btree_set_o hyle_container_btree_set;
typedef int (*hyle_container_btree_set_visit_fn)(const void *key,
                                                 void *context);
HYLE_CONTAINER_C_API hyle_status
hyle_container_btree_set_create(hyle_container_btree_set **out,
                                const hyle_container_btree_options *options);
HYLE_CONTAINER_C_API void
hyle_container_btree_set_destroy(hyle_container_btree_set *set);
HYLE_CONTAINER_C_API size_t
hyle_container_btree_set_size(const hyle_container_btree_set *set);
HYLE_CONTAINER_C_API void
hyle_container_btree_set_clear(hyle_container_btree_set *set);
/* Uses the same stable failure initialization as B-tree map insert. */
HYLE_CONTAINER_C_API hyle_status
hyle_container_btree_set_insert(hyle_container_insert_result *out_result,
                                hyle_container_btree_set *set, const void *key);
HYLE_CONTAINER_C_API const void *
hyle_container_btree_set_find(const hyle_container_btree_set *set,
                              const void *key);
HYLE_CONTAINER_C_API int
hyle_container_btree_set_contains(const hyle_container_btree_set *set,
                                  const void *key);
HYLE_CONTAINER_C_API int
hyle_container_btree_set_erase(hyle_container_btree_set *set, const void *key);
/* Erases [lower, upper) and returns the number removed. A NULL bound is
 * unbounded. If anything is removed, all pointers and cursors are invalid. */
HYLE_CONTAINER_C_API size_t hyle_container_btree_set_erase_range(
    hyle_container_btree_set *set, const void *lower, const void *upper);
HYLE_CONTAINER_C_API int
hyle_container_btree_set_begin(const hyle_container_btree_set *set,
                               hyle_container_btree_cursor *out_cursor);
HYLE_CONTAINER_C_API int
hyle_container_btree_set_lower_bound(const hyle_container_btree_set *set,
                                     const void *key,
                                     hyle_container_btree_cursor *out_cursor);
HYLE_CONTAINER_C_API int
hyle_container_btree_set_upper_bound(const hyle_container_btree_set *set,
                                     const void *key,
                                     hyle_container_btree_cursor *out_cursor);
HYLE_CONTAINER_C_API int hyle_container_btree_set_cursor_valid(
    const hyle_container_btree_set *set,
    const hyle_container_btree_cursor *cursor);
HYLE_CONTAINER_C_API const void *
hyle_container_btree_set_cursor_key(const hyle_container_btree_set *set,
                                    const hyle_container_btree_cursor *cursor);
HYLE_CONTAINER_C_API int
hyle_container_btree_set_cursor_next(const hyle_container_btree_set *set,
                                     hyle_container_btree_cursor *cursor);
HYLE_CONTAINER_C_API int
hyle_container_btree_set_cursor_previous(const hyle_container_btree_set *set,
                                         hyle_container_btree_cursor *cursor);
HYLE_CONTAINER_C_API int hyle_container_btree_set_visit_range(
    const hyle_container_btree_set *set, const void *lower, const void *upper,
    hyle_container_btree_set_visit_fn visitor, void *context);

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

/* Fixed-capacity single-producer/single-consumer queue. The producer is the
 * only thread allowed to call try_push; the consumer is the only thread
 * allowed to call try_pop. It performs no allocation after create. */
typedef struct hyle_container_spsc_queue_o hyle_container_spsc_queue;
HYLE_CONTAINER_C_API hyle_status hyle_container_spsc_queue_create(
    hyle_container_spsc_queue** out, hyle_allocator backing,
    size_t capacity);
static inline hyle_status hyle_container_spsc_queue_create_system(
    hyle_container_spsc_queue** out, size_t capacity) {
  return hyle_container_spsc_queue_create(out, hyle_system_allocator(),
                                           capacity);
}
HYLE_CONTAINER_C_API void hyle_container_spsc_queue_destroy(
    hyle_container_spsc_queue* queue);
HYLE_CONTAINER_C_API int hyle_container_spsc_queue_try_push(
    hyle_container_spsc_queue* queue, void* item);
HYLE_CONTAINER_C_API int hyle_container_spsc_queue_try_pop(
    hyle_container_spsc_queue* queue, void** out_item);
/* Snapshot only: concurrent progress may make it stale before return. */
HYLE_CONTAINER_C_API size_t hyle_container_spsc_queue_size_approx(
    const hyle_container_spsc_queue* queue);
HYLE_CONTAINER_C_API size_t hyle_container_spsc_queue_capacity(
    const hyle_container_spsc_queue* queue);

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
/* Fixed-width across compilers and -fshort-enums. */
typedef uint32_t hyle_container_steal_status;
enum {
  HYLE_CONTAINER_STEAL_OK = 0,
  HYLE_CONTAINER_STEAL_EMPTY = 1,
  /* Another owner/thief won this attempt; retry according to caller policy. */
  HYLE_CONTAINER_STEAL_RETRY = 2
};
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

typedef struct hyle_container_concurrent_ptr_vector_o
    hyle_container_concurrent_ptr_vector;
/* Multi-writer append-only pointer vector. size() counts reserved indices and
 * is not a completion fence for in-flight pushes. Read a newly appended index
 * only after its push returns, or quiesce writers before traversing [0,size()). */
HYLE_CONTAINER_C_API hyle_status hyle_container_concurrent_ptr_vector_create(
    hyle_container_concurrent_ptr_vector** out, hyle_allocator backing,
    size_t first_segment_capacity);
static inline hyle_status hyle_container_concurrent_ptr_vector_create_system(
    hyle_container_concurrent_ptr_vector** out,
    size_t first_segment_capacity) {
  return hyle_container_concurrent_ptr_vector_create(
      out, hyle_system_allocator(), first_segment_capacity);
}
HYLE_CONTAINER_C_API void hyle_container_concurrent_ptr_vector_destroy(
    hyle_container_concurrent_ptr_vector* vector);
/* Preallocates stable segments for at least count pointers. */
HYLE_CONTAINER_C_API hyle_status hyle_container_concurrent_ptr_vector_reserve(
    hyle_container_concurrent_ptr_vector* vector, size_t count);
/* Appends one caller-owned pointer and returns its stable index. */
HYLE_CONTAINER_C_API hyle_status
hyle_container_concurrent_ptr_vector_try_push_back(
    size_t* out_index, hyle_container_concurrent_ptr_vector* vector,
    void* item);
HYLE_CONTAINER_C_API size_t hyle_container_concurrent_ptr_vector_size(
    const hyle_container_concurrent_ptr_vector* vector);
HYLE_CONTAINER_C_API size_t hyle_container_concurrent_ptr_vector_capacity(
    const hyle_container_concurrent_ptr_vector* vector);
/* Returns 1 and writes the payload (which may itself be NULL) when index is
 * present. Returns 0 and leaves out_item unchanged when out of range/invalid. */
HYLE_CONTAINER_C_API int hyle_container_concurrent_ptr_vector_try_get(
    const hyle_container_concurrent_ptr_vector* vector, size_t index,
    void** out_item);

#ifdef __cplusplus
}
#endif
#endif /* HYLE_C_CONTAINER_H_ */
