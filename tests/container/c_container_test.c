#define _CRT_SECURE_NO_WARNINGS 1
#include "hyle/c/container.h"

/* Everything below this marker is part of the reviewed pure-C surface. */
enum { HYLE_C_CONTAINER_COVERAGE_BODY_MARKER = 0 };

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int g_failures;
#define CHECK(x)                                                               \
  do {                                                                         \
    if (!(x)) {                                                                \
      fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #x, __FILE__, __LINE__); \
      ++g_failures;                                                            \
    }                                                                          \
  } while (0)

static int status_ok(hyle_status status) {
  const int ok = hyle_status_is_ok(status);
  hyle_status_free(&status);
  return ok;
}

static hyle_status_code status_code(hyle_status status) {
  const hyle_status_code code = hyle_status_get_code(status);
  hyle_status_free(&status);
  return code;
}

static int map_inserted(hyle_container_hash_map* map, const void* key,
                        const void* value) {
  hyle_container_insert_result result = HYLE_CONTAINER_ALREADY_PRESENT;
  return status_ok(hyle_container_hash_map_insert(&result, map, key, value)) &&
         result == HYLE_CONTAINER_INSERTED;
}

static int set_inserted(hyle_container_hash_set* set, const void* key) {
  hyle_container_insert_result result = HYLE_CONTAINER_ALREADY_PRESENT;
  return status_ok(hyle_container_hash_set_insert(&result, set, key)) &&
         result == HYLE_CONTAINER_INSERTED;
}

static hyle_container_string_view sv(const char* text) {
  hyle_container_string_view value =
      hyle_container_view(text, (uint32_t)strlen(text));
  value.null_terminated = 1;
  return value;
}

static uint64_t constant_hash(const void* key, size_t size, void* context) {
  (void)key;
  (void)size;
  (void)context;
  return 7;
}
static int visit_map(const void* key, void* value, void* context) {
  (void)key;
  (void)value;
  ++*(int*)context;
  return 1;
}
static int stop_map_visit(const void* key, void* value, void* context) {
  (void)key;
  (void)value;
  ++*(int*)context;
  return 0;
}
static int visit_aligned_key(const void* key, void* value, void* context) {
  (void)value;
  CHECK(((uintptr_t)key % HYLE_CONTAINER_ALIGNOF(uint64_t)) == 0);
  ++*(int*)context;
  return 1;
}
static int visit_aligned_set_key(const void* key, void* context) {
  CHECK(((uintptr_t)key % HYLE_CONTAINER_ALIGNOF(uint64_t)) == 0);
  ++*(int*)context;
  return 1;
}
static int stop_set_visit(const void* key, void* context) {
  (void)key;
  ++*(int*)context;
  return 0;
}

typedef struct tracking_allocator {
  hyle_allocator inner;
  size_t allocations;
  size_t frees;
} tracking_allocator;
static void* tracking_alloc(void* state, size_t size, size_t alignment) {
  tracking_allocator* tracking = (tracking_allocator*)state;
  void* result =
      tracking->inner.alloc(tracking->inner.state, size, alignment);
  if (result != NULL) ++tracking->allocations;
  return result;
}
static void tracking_free(void* state, void* ptr, size_t size,
                          size_t alignment) {
  tracking_allocator* tracking = (tracking_allocator*)state;
  if (ptr != NULL) ++tracking->frees;
  tracking->inner.free(tracking->inner.state, ptr, size, alignment);
}
static void* tracking_realloc(void* state, void* ptr, size_t old_size,
                              size_t new_size, size_t alignment) {
  tracking_allocator* tracking = (tracking_allocator*)state;
  return tracking->inner.realloc(tracking->inner.state, ptr, old_size,
                                 new_size, alignment);
}
static void* fail_alloc(void* state, size_t size, size_t alignment) {
  (void)state;
  (void)size;
  (void)alignment;
  return NULL;
}
static void fail_free(void* state, void* ptr, size_t size, size_t alignment) {
  (void)state;
  (void)ptr;
  (void)size;
  (void)alignment;
}
static void* fail_realloc(void* state, void* ptr, size_t old_size,
                          size_t new_size, size_t alignment) {
  (void)state;
  (void)ptr;
  (void)old_size;
  (void)new_size;
  (void)alignment;
  return NULL;
}

typedef struct fail_after_allocator {
  hyle_allocator inner;
  size_t successful_allocations_left;
} fail_after_allocator;
static void* fail_after_alloc(void* state, size_t size, size_t alignment) {
  fail_after_allocator* failing = (fail_after_allocator*)state;
  if (failing->successful_allocations_left == 0) return NULL;
  --failing->successful_allocations_left;
  return failing->inner.alloc(failing->inner.state, size, alignment);
}
static void fail_after_free(void* state, void* ptr, size_t size,
                            size_t alignment) {
  fail_after_allocator* failing = (fail_after_allocator*)state;
  failing->inner.free(failing->inner.state, ptr, size, alignment);
}
static void* fail_after_realloc(void* state, void* ptr, size_t old_size,
                                size_t new_size, size_t alignment) {
  fail_after_allocator* failing = (fail_after_allocator*)state;
  if (failing->successful_allocations_left == 0) return NULL;
  --failing->successful_allocations_left;
  return failing->inner.realloc(failing->inner.state, ptr, old_size, new_size,
                                alignment);
}

static void test_strings_and_paths(void) {
  char output[128];
  hyle_container_string_view candidates[] = {
      sv("src/config.h"), sv("cfg.ini"), sv("none.txt")};
  hyle_container_string_view value;

  CHECK(hyle_container_hash_bytes("abc", 3, 0) != 0);
  CHECK(hyle_container_string_compare(sv("abc"), sv("abd")) < 0);
  CHECK(hyle_container_string_equal_ignore_case(sv("AbC"), sv("aBc")));
  CHECK(hyle_container_string_find(sv("hello"), sv("ll"), 0) == 2);
  CHECK(hyle_container_string_rfind(sv("ababa"), sv("ba"),
                                    HYLE_CONTAINER_NPOS) == 3);
  CHECK(hyle_container_string_find_char(sv("abc"), 'b', 0) == 1);
  CHECK(hyle_container_string_rfind_char(sv("abca"), 'a',
                                         HYLE_CONTAINER_NPOS) == 3);
  CHECK(hyle_container_string_count_char(sv("abca"), 'a') == 2);
  CHECK(hyle_container_string_find_first_of(sv("abc"), sv("cx"), 0) == 2);
  CHECK(hyle_container_string_find_first_not_of(sv("aaab"), sv("a"), 0) == 3);
  CHECK(hyle_container_string_find_last_of(sv("abca"), sv("a"),
                                           HYLE_CONTAINER_NPOS) == 3);
  CHECK(hyle_container_string_find_last_not_of(sv("abbb"), sv("b"),
                                               HYLE_CONTAINER_NPOS) == 0);
  CHECK(hyle_container_string_starts_with(sv("abcdef"), sv("abc")));
  CHECK(hyle_container_string_ends_with(sv("abcdef"), sv("def")));
  CHECK(hyle_container_string_contains(sv("abcdef"), sv("cd")));
  value = hyle_container_string_slice(sv("abcdef"), 2, 3);
  CHECK(value.size == 3 && memcmp(value.data, "cde", 3) == 0);
  {
    hyle_container_string_view rest = sv("a:b:c");
    value = hyle_container_string_split(&rest, ':');
    CHECK(value.size == 1 && value.data[0] == 'a' && rest.size == 3);
    value = hyle_container_string_rsplit(&rest, ':');
    CHECK(value.size == 1 && value.data[0] == 'c' && rest.size == 1);
  }
  value = hyle_container_string_trim(sv("  abc \t"), sv(" \t"));
  CHECK(value.size == 3 && memcmp(value.data, "abc", 3) == 0);
  CHECK(hyle_container_string_copy(sv("copy"), output, sizeof(output)) == 4);
  CHECK(strcmp(output, "copy") == 0);
  CHECK(hyle_container_string_concat(sv("ab"), sv("cd"), output,
                                     sizeof(output)) == 4);
  CHECK(strcmp(output, "abcd") == 0);
  CHECK(hyle_container_ascii_is_space('\t'));
  CHECK(hyle_container_ascii_is_digit('7'));
  CHECK(hyle_container_ascii_is_print('x'));
  CHECK(hyle_container_ascii_is_hex_digit('F'));
  CHECK(hyle_container_ascii_hex_value('b') == 11);
  CHECK(hyle_container_ascii_lower('Q') == 'q');
  CHECK(hyle_container_ascii_upper('q') == 'Q');
  CHECK(hyle_container_path_is_separator('\\'));
  CHECK(hyle_container_string_lower(sv("AbC"), output, sizeof(output)) == 3);
  CHECK(strcmp(output, "abc") == 0);
  CHECK(hyle_container_string_upper(sv("AbC"), output, sizeof(output)) == 3);
  CHECK(strcmp(output, "ABC") == 0);
  CHECK(hyle_container_hex_encode("\x01\xaf", 2, output, sizeof(output)) == 4);
  CHECK(strcmp(output, "01af") == 0);
  {
    unsigned char decoded[2] = {0, 0};
    CHECK(hyle_container_hex_decode(sv("01af"), decoded, sizeof(decoded)) == 2);
    CHECK(decoded[0] == 1 && decoded[1] == 0xaf);
  }
  CHECK(hyle_container_string_c_escape(sv("a\n"), output,
                                       sizeof(output)) == 3);
  CHECK(strcmp(output, "a\\n") == 0);
  {
    const unsigned char bytes[] = {'A', 0, 0x1f, 0xff};
    CHECK(hyle_container_string_c_escape(
              hyle_container_view((const char*)bytes, 4), output,
              sizeof(output)) == 13);
    CHECK(strcmp(output, "A\\000\\037\\377") == 0);
  }
  CHECK(hyle_container_utf8_length(sv("hello")) == 5);
  {
    const unsigned char malformed[] = {0xc0u, 0x80u};
    CHECK(hyle_container_utf8_length(
              hyle_container_view((const char*)malformed, 2)) == -1);
  }

  value = hyle_container_path_extension(sv("a/b.txt"));
  CHECK(value.size == 4 && memcmp(value.data, ".txt", 4) == 0);
  value = hyle_container_path_strip_extension(sv("a/b.txt"));
  CHECK(value.size == 3 && memcmp(value.data, "a/b", 3) == 0);
  value = hyle_container_path_base(sv("a/b.txt"));
  CHECK(value.size == 5 && memcmp(value.data, "b.txt", 5) == 0);
  value = hyle_container_path_directory(sv("a/b.txt"));
  CHECK(value.size == 1 && value.data[0] == 'a');
  CHECK(hyle_container_path_is_absolute(sv("/a")));
  CHECK(hyle_container_path_is_unc(sv("\\\\server\\share")));
  {
    hyle_container_string_view path = sv("a/b/c");
    value = hyle_container_path_pop_first(&path);
    CHECK(value.size == 1 && value.data[0] == 'a');
    value = hyle_container_path_pop_last(&path);
    CHECK(value.size == 1 && value.data[0] == 'c');
  }
  CHECK(hyle_container_path_join(sv("a"), sv("b"), output,
                                 sizeof(output)) == 3);
  CHECK(strcmp(output, "a/b") == 0);
  CHECK(hyle_container_path_join_array(candidates, 2, output,
                                       sizeof(output)) > 0);
  {
    hyle_container_string_view parts[4];
    CHECK(hyle_container_path_split_all(sv("/usr/local/bin"), parts, 4) == 4);
    CHECK(parts[0].size == 1 && parts[0].data[0] == '/');
    CHECK(parts[3].size == 3 && memcmp(parts[3].data, "bin", 3) == 0);
  }
  CHECK(hyle_container_path_replace_extension(
            sv("a.txt"), sv(".bin"), output, sizeof(output)) == 5);
  CHECK(strcmp(output, "a.bin") == 0);
  strcpy(output, "a//b/../c");
  CHECK(hyle_container_path_normalize(output) == 3);
  CHECK(strcmp(output, "a/c") == 0);
}

static void test_string_interner(void) {
  hyle_container_string_interner* buffer = NULL;
  hyle_container_string_id id;
  hyle_container_string_view value;
  size_t completed;
  CHECK(status_ok(hyle_container_string_interner_create_system(&buffer)));
  CHECK(buffer != NULL);
  CHECK(status_ok(hyle_container_string_interner_acquire(&id, buffer, sv("alpha"))));
  CHECK(id != 0);
  CHECK(hyle_container_string_interner_size(buffer) == 1);
  value = hyle_container_string_interner_lookup(buffer, id);
  CHECK(value.size == 5 && value.null_terminated);
  CHECK(hyle_container_string_interner_retain(buffer, id));
  CHECK(hyle_container_string_interner_release(buffer, id));
  CHECK(status_ok(
      hyle_container_string_interner_intern(&value, buffer, sv("alpha"))));
  CHECK(value.size == 5);
  CHECK(hyle_container_string_interner_release(buffer, id));
  CHECK(!hyle_container_string_interner_release(buffer, id));
  {
    hyle_container_string_view input[] = {sv("one"), sv("two"), sv("one")};
    hyle_container_string_view output[3];
    CHECK(status_ok(hyle_container_string_interner_intern_many(
        &completed, buffer, input, output, 3)) && completed == 3);
    CHECK(output[0].data == output[2].data);
    CHECK(status_ok(hyle_container_string_interner_intern_many(
        &completed, buffer, input, NULL, 3)) && completed == 3);
  }
  {
    hyle_container_string_view input[] = {sv("batch-a"), sv("batch-b")};
    hyle_container_string_id ids[2];
    CHECK(status_ok(hyle_container_string_interner_acquire_many(
        &completed, buffer, input, ids, 2)) && completed == 2);
    CHECK(ids[0] != 0 && ids[1] != 0);
    CHECK(hyle_container_string_interner_retain_many(buffer, ids, 2) == 2);
    CHECK(hyle_container_string_interner_release_many(buffer, ids, 2) == 2);
    CHECK(hyle_container_string_interner_release_many(buffer, ids, 2) == 2);
  }
  hyle_container_string_interner_destroy(buffer);
}

static void test_vector_and_map(void) {
  hyle_container_vector_options vector_options =
      HYLE_CONTAINER_VECTOR_OPTIONS_INIT;
  hyle_container_hash_map_options map_options =
      HYLE_CONTAINER_HASH_MAP_OPTIONS_INIT;
  hyle_container_vector* vector;
  hyle_container_hash_map* map;
  int batch[2] = {22, 33};
  int input;
  int output;
  int visits = 0;
  tracking_allocator tracking = {hyle_system_allocator(), 0, 0};
  hyle_allocator backing = {tracking_alloc, tracking_free, tracking_realloc,
                            &tracking};

  vector_options.item_size = sizeof(int);
  vector_options.item_alignment = HYLE_CONTAINER_ALIGNOF(int);
  vector_options.initial_capacity = 1;
  vector_options.backing = backing;
  CHECK(status_ok(hyle_container_vector_create(&vector, &vector_options)));
  CHECK(vector != NULL);
  input = 11;
  CHECK(status_ok(hyle_container_vector_try_push_back(vector, &input)));
  CHECK(status_ok(hyle_container_vector_try_append(vector, batch, 2)));
  CHECK(hyle_container_vector_size(vector) == 3);
  CHECK(status_ok(hyle_container_vector_try_append(
      vector, (const int*)hyle_container_vector_data_const(vector) + 1, 2)));
  CHECK(hyle_container_vector_size(vector) == 5);
  CHECK(status_ok(hyle_container_vector_try_resize(
      vector, 7, hyle_container_vector_at(vector, 0))));
  CHECK(hyle_container_vector_size(vector) == 7);
  CHECK(status_ok(hyle_container_vector_try_insert(
      vector, 1, (const int*)hyle_container_vector_data_const(vector) + 2,
      2)));
  CHECK(hyle_container_vector_size(vector) == 9);
  CHECK(((int*)hyle_container_vector_data(vector))[0] == 11);
  CHECK(((int*)hyle_container_vector_data(vector))[1] == 33);
  CHECK(((int*)hyle_container_vector_data(vector))[2] == 22);
  CHECK(status_ok(hyle_container_vector_erase(vector, 2, 3)));
  CHECK(hyle_container_vector_size(vector) == 6);
  CHECK(status_code(hyle_container_vector_try_insert(
            vector, 7, batch, 2)) == HYLE_STATUS_OUT_OF_RANGE);
  CHECK(status_code(hyle_container_vector_erase(vector, 5, 2)) ==
        HYLE_STATUS_OUT_OF_RANGE);
  CHECK(status_code(hyle_container_vector_try_resize(vector, 8, NULL)) ==
        HYLE_STATUS_INVALID_ARGUMENT);
  CHECK(status_ok(hyle_container_vector_try_append(vector, NULL, 0)));
  CHECK(status_code(hyle_container_vector_try_append(vector, NULL, 1)) ==
        HYLE_STATUS_INVALID_ARGUMENT);
  CHECK(*(int*)hyle_container_vector_at(vector, 0) == 11);
  CHECK(hyle_container_vector_data(vector) != NULL);
  CHECK(hyle_container_vector_data_const(vector) != NULL);
  CHECK(status_ok(hyle_container_vector_reserve(vector, 32)));
  CHECK(status_ok(hyle_container_vector_try_shrink_to_fit(vector)));
  CHECK(hyle_container_vector_capacity(vector) == 6);
  CHECK(hyle_container_vector_try_pop_back(vector, &output) && output == 11);
  CHECK(status_ok(hyle_container_vector_try_resize(vector, 2, NULL)));
  hyle_container_vector_clear(vector);
  CHECK(hyle_container_vector_size(vector) == 0);
  CHECK(status_ok(hyle_container_vector_try_shrink_to_fit(vector)));
  CHECK(hyle_container_vector_data(vector) == NULL);
  CHECK(hyle_container_vector_capacity(vector) == 0);
  hyle_container_vector_destroy(vector);

  map_options.key_size = sizeof(int);
  map_options.key_alignment = HYLE_CONTAINER_ALIGNOF(int);
  map_options.value_size = sizeof(int);
  map_options.value_alignment = HYLE_CONTAINER_ALIGNOF(int);
  map_options.hash = constant_hash; /* force collision-chain coverage */
  map_options.backing = backing;
  {
    hyle_container_hash_map_options invalid = map_options;
    hyle_container_hash_map* invalid_map = (hyle_container_hash_map*)1;
    invalid.key_size = SIZE_MAX - 9;
    invalid.key_alignment = 1;
    CHECK(status_code(hyle_container_hash_map_create(&invalid_map, &invalid)) ==
          HYLE_STATUS_INVALID_ARGUMENT);
    CHECK(invalid_map == NULL);
  }
  CHECK(status_ok(hyle_container_hash_map_create(&map, &map_options)));
  CHECK(map != NULL);
  input = 1;
  output = 10;
  CHECK(map_inserted(map, &input, &output));
  input = 2;
  output = 20;
  CHECK(map_inserted(map, &input, &output));
  CHECK(hyle_container_hash_map_size(map) == 2);
  CHECK(status_ok(hyle_container_hash_map_reserve(map, 64)));
  CHECK(hyle_container_hash_map_capacity(map) >= 64);
  input = 1;
  CHECK(*(int*)hyle_container_hash_map_find(map, &input) == 10);
  CHECK(*(const int*)hyle_container_hash_map_find_const(map, &input) == 10);
  CHECK(hyle_container_hash_map_contains(map, &input));
  {
    hyle_container_insert_result result;
    output = 99;
    CHECK(status_ok(
        hyle_container_hash_map_insert(&result, map, &input, &output)));
    CHECK(result == HYLE_CONTAINER_ALREADY_PRESENT);
    CHECK(*(int*)hyle_container_hash_map_find(map, &input) == 10);
    CHECK(status_ok(hyle_container_hash_map_insert_or_assign(
        &result, map, &input, &output)));
    CHECK(result == HYLE_CONTAINER_ASSIGNED);
    CHECK(*(int*)hyle_container_hash_map_find(map, &input) == 99);
    output = 10;
    CHECK(status_ok(hyle_container_hash_map_insert_or_assign(
        &result, map, &input, &output)));
  }
  CHECK(hyle_container_hash_map_visit(map, visit_map, &visits));
  CHECK(visits == 2);
  {
    int stopped_visits = 0;
    CHECK(!hyle_container_hash_map_visit(map, stop_map_visit,
                                         &stopped_visits));
    CHECK(stopped_visits == 1);
  }
  {
    int i;
    for (i = 3; i < 200; ++i) {
      int mapped = i * 10;
      CHECK(map_inserted(map, &i, &mapped));
    }
    for (i = 1; i < 200; ++i) {
      const int* mapped =
          (const int*)hyle_container_hash_map_find_const(map, &i);
      CHECK(mapped != NULL && *mapped == i * 10);
    }
    /* Preserve a value returned by find across an insertion-time flat-table
       relocation. */
    {
      int source_key = 7;
      int fill_key = 10000;
      while (hyle_container_hash_map_size(map) <
             hyle_container_hash_map_capacity(map) -
                 hyle_container_hash_map_capacity(map) / 8) {
        int mapped = fill_key * 10;
        CHECK(map_inserted(map, &fill_key, &mapped));
        ++fill_key;
      }
      const int* aliased =
          (const int*)hyle_container_hash_map_find_const(map, &source_key);
      CHECK(aliased != NULL);
      CHECK(map_inserted(map, &fill_key, aliased));
      CHECK(*(const int*)hyle_container_hash_map_find_const(map, &fill_key) ==
            70);
    }
    for (i = 3; i < 200; i += 2)
      CHECK(hyle_container_hash_map_erase(map, &i));
    for (i = 3; i < 200; ++i)
      CHECK(hyle_container_hash_map_contains(map, &i) == ((i & 1) == 0));
  }
  CHECK(hyle_container_hash_map_erase(map, &input));
  CHECK(!hyle_container_hash_map_contains(map, &input));
  hyle_container_hash_map_clear(map);
  CHECK(hyle_container_hash_map_size(map) == 0);
  hyle_container_hash_map_destroy(map);

  {
    /* value size 1 makes an unrounded raw slot stride misalign key #2+. */
    hyle_container_hash_map_options aligned_options =
        HYLE_CONTAINER_HASH_MAP_OPTIONS_FOR(uint64_t, char);
    hyle_container_hash_map* aligned = NULL;
    CHECK(status_ok(
        hyle_container_hash_map_create(&aligned, &aligned_options)));
    int visited = 0;
    for (uint64_t i = 0; i < 40; ++i) {
      const char byte = (char)i;
      CHECK(map_inserted(aligned, &i, &byte));
    }
    CHECK(hyle_container_hash_map_visit(aligned, visit_aligned_key,
                                        &visited));
    CHECK(visited == 40);
    hyle_container_hash_map_destroy(aligned);
  }

  CHECK(tracking.allocations > 0);
  CHECK(tracking.allocations == tracking.frees);
}

static void test_hash_set(void) {
  hyle_container_hash_set_options options =
      HYLE_CONTAINER_HASH_SET_OPTIONS_FOR(uint64_t);
  hyle_container_hash_set* set = NULL;
  tracking_allocator tracking = {hyle_system_allocator(), 0, 0};
  hyle_allocator backing = {tracking_alloc, tracking_free, tracking_realloc,
                            &tracking};
  uint64_t i;
  int visits = 0;

  options.hash = constant_hash; /* force collision-chain coverage */
  options.backing = backing;
  {
    hyle_container_hash_set_options invalid = options;
    hyle_container_hash_set* invalid_set = (hyle_container_hash_set*)1;
    invalid.key_size = SIZE_MAX - 9;
    invalid.key_alignment = 1;
    CHECK(status_code(hyle_container_hash_set_create(&invalid_set, &invalid)) ==
          HYLE_STATUS_INVALID_ARGUMENT);
    CHECK(invalid_set == NULL);
  }
  CHECK(status_ok(hyle_container_hash_set_create(&set, &options)));
  CHECK(set != NULL);
  CHECK(status_ok(hyle_container_hash_set_reserve(set, 64)));
  CHECK(hyle_container_hash_set_capacity(set) >= 64);
  for (i = 0; i < 200; ++i) CHECK(set_inserted(set, &i));
  CHECK(hyle_container_hash_set_size(set) == 200);
  for (i = 0; i < 200; ++i) {
    const uint64_t* canonical =
        (const uint64_t*)hyle_container_hash_set_find(set, &i);
    CHECK(canonical != NULL && *canonical == i);
    CHECK(hyle_container_hash_set_contains(set, &i));
  }
  {
    hyle_container_insert_result result = HYLE_CONTAINER_INSERTED;
    i = 42;
    CHECK(status_ok(hyle_container_hash_set_insert(&result, set, &i)));
    CHECK(result == HYLE_CONTAINER_ALREADY_PRESENT);
    CHECK(hyle_container_hash_set_size(set) == 200);
  }
  CHECK(hyle_container_hash_set_visit(set, visit_aligned_set_key, &visits));
  CHECK(visits == 200);
  {
    int stopped_visits = 0;
    CHECK(!hyle_container_hash_set_visit(set, stop_set_visit,
                                         &stopped_visits));
    CHECK(stopped_visits == 1);
  }
  for (i = 1; i < 200; i += 2)
    CHECK(hyle_container_hash_set_erase(set, &i));
  for (i = 0; i < 200; ++i)
    CHECK(hyle_container_hash_set_contains(set, &i) == ((i & 1) == 0));
  for (i = 1000; i < 1100; ++i) CHECK(set_inserted(set, &i));
  for (i = 1000; i < 1100; ++i)
    CHECK(hyle_container_hash_set_contains(set, &i));
  hyle_container_hash_set_clear(set);
  CHECK(hyle_container_hash_set_size(set) == 0);
  hyle_container_hash_set_destroy(set);
  CHECK(tracking.allocations > 0);
  CHECK(tracking.allocations == tracking.frees);
}

static void test_queues_and_concurrent_vector(void) {
  int a = 1;
  int b = 2;
  void* output = NULL;
  size_t cursor = 0;
  size_t index;
  hyle_container_bounded_queue* bounded = NULL;
  hyle_container_sharded_queue* sharded = NULL;
  hyle_container_work_stealing_deque* deque = NULL;
  hyle_container_ws_fifo_queue* source = NULL;
  hyle_container_ws_fifo_queue* target = NULL;
  hyle_container_concurrent_ptr_vector* vector = NULL;
  {
    hyle_container_concurrent_ptr_vector* invalid =
        (hyle_container_concurrent_ptr_vector*)1;
    CHECK(status_code(hyle_container_concurrent_ptr_vector_create_system(
              &invalid, SIZE_MAX)) == HYLE_STATUS_OUT_OF_RANGE);
    CHECK(invalid == NULL);
  }
  CHECK(status_ok(
      hyle_container_bounded_queue_create_system(&bounded, 8)));
  CHECK(status_ok(
      hyle_container_sharded_queue_create_system(&sharded, 2, 4)));
  CHECK(status_ok(
      hyle_container_work_stealing_deque_create_system(&deque, 8)));
  CHECK(status_ok(
      hyle_container_ws_fifo_queue_create_system(&source, 8)));
  CHECK(status_ok(
      hyle_container_ws_fifo_queue_create_system(&target, 8)));
  CHECK(status_ok(
      hyle_container_concurrent_ptr_vector_create_system(&vector, 2)));

  CHECK(bounded != NULL && hyle_container_bounded_queue_capacity(bounded) == 8);
  CHECK(hyle_container_bounded_queue_try_push(bounded, &a));
  CHECK(hyle_container_bounded_queue_try_pop(bounded, &output) && output == &a);
  hyle_container_bounded_queue_destroy(bounded);

  CHECK(sharded != NULL);
  CHECK(hyle_container_sharded_queue_shard_count(sharded) == 2);
  CHECK(hyle_container_sharded_queue_capacity(sharded) == 8);
  CHECK(hyle_container_sharded_queue_try_push(sharded, 1, &b));
  CHECK(hyle_container_sharded_queue_try_pop(sharded, &cursor, &output) &&
        output == &b);
  hyle_container_sharded_queue_destroy(sharded);

  CHECK(deque != NULL);
  CHECK(hyle_container_work_stealing_deque_capacity(deque) == 8);
  CHECK(hyle_container_work_stealing_deque_try_push_bottom(deque, &a));
  CHECK(hyle_container_work_stealing_deque_try_steal_top(deque, &output) ==
        HYLE_CONTAINER_STEAL_OK);
  CHECK(output == &a);
  CHECK(!hyle_container_work_stealing_deque_try_pop_bottom(deque, &output));
  hyle_container_work_stealing_deque_destroy(deque);

  CHECK(source != NULL && target != NULL);
  CHECK(hyle_container_ws_fifo_queue_capacity(source) == 8);
  CHECK(hyle_container_ws_fifo_queue_try_push_back(source, &a));
  CHECK(hyle_container_ws_fifo_queue_try_push_back(source, &b));
  CHECK(hyle_container_ws_fifo_queue_try_steal_into(source, target, &output));
  CHECK(hyle_container_ws_fifo_queue_try_pop_front(source, &output) ||
        hyle_container_ws_fifo_queue_try_pop_front(target, &output));
  hyle_container_ws_fifo_queue_destroy(source);
  hyle_container_ws_fifo_queue_destroy(target);

  CHECK(vector != NULL);
  CHECK(status_code(hyle_container_concurrent_ptr_vector_reserve(
            vector, SIZE_MAX)) == HYLE_STATUS_OUT_OF_RANGE);
  CHECK(status_ok(hyle_container_concurrent_ptr_vector_reserve(vector, 8)));
  CHECK(hyle_container_concurrent_ptr_vector_capacity(vector) >= 8);
  CHECK(status_ok(hyle_container_concurrent_ptr_vector_try_push_back(
                      &index, vector, &a)) &&
        index == 0);
  CHECK(status_ok(hyle_container_concurrent_ptr_vector_try_push_back(
                      &index, vector, NULL)) &&
        index == 1);
  CHECK(status_ok(hyle_container_concurrent_ptr_vector_try_push_back(
                      &index, vector, &b)) &&
        index == 2);
  CHECK(hyle_container_concurrent_ptr_vector_size(vector) == 3);
  output = &a;
  CHECK(hyle_container_concurrent_ptr_vector_try_get(vector, 1, &output));
  CHECK(output == NULL);
  output = &a;
  CHECK(!hyle_container_concurrent_ptr_vector_try_get(vector, 3, &output));
  CHECK(output == &a);
  CHECK(hyle_container_concurrent_ptr_vector_try_get(vector, 2, &output));
  CHECK(output == &b);
  hyle_container_concurrent_ptr_vector_destroy(vector);
}

int main(void) {
  hyle_allocator invalid_backing = {0};
  hyle_allocator failing_backing = {fail_alloc, fail_free, fail_realloc, NULL};
  hyle_container_bounded_queue* bounded_queue = NULL;
  hyle_container_string_interner* string_interner = NULL;
  test_strings_and_paths();
  test_string_interner();
  test_vector_and_map();
  test_hash_set();
  test_queues_and_concurrent_vector();
  CHECK(status_code(hyle_container_bounded_queue_create(
            &bounded_queue, invalid_backing, 8)) ==
        HYLE_STATUS_INVALID_ARGUMENT);
  CHECK(bounded_queue == NULL);
  CHECK(status_code(hyle_container_bounded_queue_create(
            &bounded_queue, failing_backing, 8)) ==
        HYLE_STATUS_RESOURCE_EXHAUSTED);
  CHECK(status_code(hyle_container_string_interner_create(
            &string_interner, failing_backing)) == HYLE_STATUS_RESOURCE_EXHAUSTED);
  {
    /* C handle, C++ implementation, then runtime-sized shard storage. */
    fail_after_allocator failing = {hyle_system_allocator(), 3};
    hyle_allocator fail_storage = {fail_after_alloc, fail_after_free,
                                   fail_after_realloc, &failing};
    hyle_container_string_interner* strings = NULL;
    hyle_container_string_view interned;
    CHECK(status_ok(
        hyle_container_string_interner_create(&strings, fail_storage)));
    CHECK(strings != NULL);
    CHECK(status_code(hyle_container_string_interner_intern(
              &interned, strings, sv("allocation"))) ==
          HYLE_STATUS_RESOURCE_EXHAUSTED);
    CHECK(interned.data == NULL);
    hyle_container_string_interner_destroy(strings);
  }
  {
    fail_after_allocator bounded = {hyle_system_allocator(), 1};
    fail_after_allocator sharded = {hyle_system_allocator(), 1};
    fail_after_allocator stealing = {hyle_system_allocator(), 1};
    fail_after_allocator sharing = {hyle_system_allocator(), 1};
    hyle_allocator a = {fail_after_alloc, fail_after_free, fail_after_realloc,
                        &bounded};
    hyle_container_sharded_queue* sharded_out = NULL;
    hyle_container_work_stealing_deque* stealing_out = NULL;
    hyle_container_ws_fifo_queue* sharing_out = NULL;
    CHECK(status_code(hyle_container_bounded_queue_create(
              &bounded_queue, a, 8)) == HYLE_STATUS_RESOURCE_EXHAUSTED);
    a.state = &sharded;
    CHECK(status_code(hyle_container_sharded_queue_create(
              &sharded_out, a, 2, 8)) == HYLE_STATUS_RESOURCE_EXHAUSTED);
    a.state = &stealing;
    CHECK(status_code(hyle_container_work_stealing_deque_create(
              &stealing_out, a, 8)) == HYLE_STATUS_RESOURCE_EXHAUSTED);
    a.state = &sharing;
    CHECK(status_code(hyle_container_ws_fifo_queue_create(
              &sharing_out, a, 8)) == HYLE_STATUS_RESOURCE_EXHAUSTED);
  }
  {
    hyle_container_vector_options options =
        HYLE_CONTAINER_VECTOR_OPTIONS_FOR(int, 1);
    hyle_container_vector* vector = NULL;
    options.backing = failing_backing;
    CHECK(status_code(hyle_container_vector_create(&vector, &options)) ==
          HYLE_STATUS_RESOURCE_EXHAUSTED);
  }
  {
    hyle_container_vector_options options =
        HYLE_CONTAINER_VECTOR_OPTIONS_FOR(int, 4);
    hyle_container_vector* vector = NULL;
    int values[3] = {1, 2, 3};
    fail_after_allocator failing = {hyle_system_allocator(), 2};
    hyle_allocator fail_storage = {fail_after_alloc, fail_after_free,
                                   fail_after_realloc, &failing};
    options.backing = fail_storage;
    CHECK(status_ok(hyle_container_vector_create(&vector, &options)));
    CHECK(status_ok(hyle_container_vector_try_append(vector, values, 2)));
    CHECK(status_code(hyle_container_vector_try_insert(
              vector, 1, hyle_container_vector_data_const(vector), 2)) ==
          HYLE_STATUS_RESOURCE_EXHAUSTED);
    CHECK(hyle_container_vector_size(vector) == 2);
    CHECK(status_code(hyle_container_vector_try_append(vector, values, 3)) ==
          HYLE_STATUS_RESOURCE_EXHAUSTED);
    CHECK(hyle_container_vector_size(vector) == 2);
    CHECK(status_code(hyle_container_vector_try_shrink_to_fit(vector)) ==
          HYLE_STATUS_RESOURCE_EXHAUSTED);
    CHECK(hyle_container_vector_capacity(vector) == 4);
    hyle_container_vector_destroy(vector);
  }
  {
    hyle_container_hash_map_options options =
        HYLE_CONTAINER_HASH_MAP_OPTIONS_FOR(int, int);
    hyle_container_hash_map* map;
    int key = 1;
    int value = 2;
    fail_after_allocator failing = {hyle_system_allocator(), 1};
    hyle_allocator fail_storage = {fail_after_alloc, fail_after_free,
                                   fail_after_realloc, &failing};
    options.backing = fail_storage;
    CHECK(status_ok(hyle_container_hash_map_create(&map, &options)));
    CHECK(map != NULL);
    {
      hyle_container_insert_result result;
      CHECK(status_code(hyle_container_hash_map_insert(
                &result, map, &key, &value)) == HYLE_STATUS_RESOURCE_EXHAUSTED);
    }
    hyle_container_hash_map_destroy(map);
  }
  {
    hyle_container_hash_map_options options =
        HYLE_CONTAINER_HASH_MAP_OPTIONS_FOR(int, int);
    hyle_container_hash_map* map = NULL;
    fail_after_allocator failing = {hyle_system_allocator(), 3};
    hyle_allocator fail_storage = {fail_after_alloc, fail_after_free,
                                   fail_after_realloc, &failing};
    int key;
    int value;
    options.hash = constant_hash;
    options.backing = fail_storage;
    CHECK(status_ok(hyle_container_hash_map_create(&map, &options)));
    CHECK(status_ok(hyle_container_hash_map_reserve(map, 28)));
    CHECK(failing.successful_allocations_left == 0);
    for (key = 0; key < 28; ++key) {
      value = key * 2;
      CHECK(map_inserted(map, &key, &value));
    }
    key = 3;
    CHECK(hyle_container_hash_map_erase(map, &key));
    failing.successful_allocations_left = 2;
    CHECK(status_ok(hyle_container_hash_map_reserve(map, 28)));
    CHECK(failing.successful_allocations_left == 0);
    CHECK(hyle_container_hash_map_capacity(map) == 32);
    key = 100;
    value = 200;
    CHECK(map_inserted(map, &key, &value));
    hyle_container_hash_map_destroy(map);
  }
  {
    hyle_container_hash_set_options options =
        HYLE_CONTAINER_HASH_SET_OPTIONS_FOR(int);
    hyle_container_hash_set* set = NULL;
    int key = 1;
    fail_after_allocator failing = {hyle_system_allocator(), 1};
    hyle_allocator fail_storage = {fail_after_alloc, fail_after_free,
                                   fail_after_realloc, &failing};
    options.backing = fail_storage;
    CHECK(status_ok(hyle_container_hash_set_create(&set, &options)));
    CHECK(set != NULL);
    {
      hyle_container_insert_result result;
      CHECK(status_code(hyle_container_hash_set_insert(&result, set, &key)) ==
            HYLE_STATUS_RESOURCE_EXHAUSTED);
      CHECK(result == HYLE_CONTAINER_ALREADY_PRESENT);
    }
    hyle_container_hash_set_destroy(set);
  }
  {
    fail_after_allocator failing = {hyle_system_allocator(), 1};
    hyle_allocator fail_storage = {fail_after_alloc, fail_after_free,
                                   fail_after_realloc, &failing};
    hyle_container_concurrent_ptr_vector* vector = NULL;
    size_t index;
    CHECK(status_ok(hyle_container_concurrent_ptr_vector_create(
        &vector, fail_storage, 1)));
    CHECK(vector != NULL);
    CHECK(status_code(hyle_container_concurrent_ptr_vector_reserve(
              vector, 1)) == HYLE_STATUS_RESOURCE_EXHAUSTED);
    CHECK(status_code(hyle_container_concurrent_ptr_vector_try_push_back(
              &index, vector, NULL)) == HYLE_STATUS_RESOURCE_EXHAUSTED);
    CHECK(index == SIZE_MAX);
    hyle_container_concurrent_ptr_vector_destroy(vector);
  }
  CHECK(hyle_container_hash_bytes(NULL, 1, 0) == 0);
  (void)hyle_container_hash_bytes(NULL, 0, 0);
  printf("container_c_test: %s\n",
         g_failures == 0 ? "all checks passed" : "FAILED");
  return g_failures == 0 ? 0 : 1;
}
