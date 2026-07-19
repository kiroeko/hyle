#define _CRT_SECURE_NO_WARNINGS 1
#include "hyle/c/container.h"

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
static int visit_aligned_key(const void* key, void* value, void* context) {
  (void)value;
  CHECK(((uintptr_t)key % HYLE_CONTAINER_ALIGNOF(uint64_t)) == 0);
  ++*(int*)context;
  return 1;
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
  uint32_t indices[3];
  int scores[3];
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
  CHECK(hyle_container_string_escape(sv("a\n"), output, sizeof(output)) == 3);
  CHECK(strcmp(output, "a\\n") == 0);
  strcpy(output, "ui_value");
  hyle_container_string_capitalize(output);
  CHECK(strcmp(output, "UI Value") == 0);
  CHECK(hyle_container_utf8_length(sv("hello")) == 5);
  {
    const unsigned char malformed[] = {0xc0u, 0x80u};
    CHECK(hyle_container_utf8_length(
              hyle_container_view((const char*)malformed, 2)) == -1);
    CHECK(hyle_container_last_error()[0] != '\0');
    CHECK(hyle_container_utf8_icmp(
              hyle_container_view((const char*)malformed, 2), sv("ok"), 0) ==
          0);
    CHECK(hyle_container_last_error()[0] != '\0');
  }
  CHECK(hyle_container_utf8_icmp(sv("Hello"), sv("hello"), 0) == 0);
  CHECK(hyle_container_fuzzy_score(sv("config.h"), sv("cfg"), 1) > 0);
  CHECK(hyle_container_find_path_matches(candidates, 3, sv("cfg"), indices,
                                         scores, 3, 1) == 2);
  CHECK(scores[0] >= scores[1]);

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
  CHECK(hyle_container_last_error()[0] == '\0');
}

static void test_string_buf(void) {
  hyle_container_string_buf* buffer =
      hyle_container_string_buf_create_system();
  hyle_container_intern_id id;
  hyle_container_string_view value;
  CHECK(buffer != NULL);
  id = hyle_container_string_buf_add(buffer, sv("alpha"));
  CHECK(id != 0);
  CHECK(hyle_container_string_buf_size(buffer) == 1);
  value = hyle_container_string_buf_lookup(buffer, id);
  CHECK(value.size == 5 && value.null_terminated);
  CHECK(hyle_container_string_buf_retain(buffer, id));
  CHECK(hyle_container_string_buf_remove(buffer, id));
  value = hyle_container_string_buf_intern(buffer, sv("alpha"));
  CHECK(value.size == 5);
  CHECK(hyle_container_string_buf_remove(buffer, id));
  CHECK(!hyle_container_string_buf_remove(buffer, id));
  {
    hyle_container_string_view input[] = {sv("one"), sv("two"), sv("one")};
    hyle_container_string_view output[3];
    CHECK(hyle_container_string_buf_intern_many(buffer, input, output, 3) == 3);
    CHECK(output[0].data == output[2].data);
    CHECK(hyle_container_string_buf_intern_many(buffer, input, NULL, 3) == 3);
  }
  hyle_container_string_buf_destroy(buffer);
}

static void test_vector_and_map(void) {
  hyle_container_vector_options vector_options =
      HYLE_CONTAINER_VECTOR_OPTIONS_INIT;
  hyle_container_hash_map_options map_options =
      HYLE_CONTAINER_HASH_MAP_OPTIONS_INIT;
  hyle_container_vector* vector;
  hyle_container_hash_map* map;
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
  vector = hyle_container_vector_create(&vector_options);
  CHECK(vector != NULL);
  input = 11;
  CHECK(hyle_container_vector_push(vector, &input));
  input = 22;
  CHECK(hyle_container_vector_push(vector, &input));
  CHECK(hyle_container_vector_size(vector) == 2);
  CHECK(hyle_container_vector_capacity(vector) >= 2);
  CHECK(*(int*)hyle_container_vector_at(vector, 0) == 11);
  CHECK(hyle_container_vector_data(vector) != NULL);
  CHECK(hyle_container_vector_data_const(vector) != NULL);
  CHECK(hyle_container_vector_reserve(vector, 32));
  CHECK(hyle_container_vector_pop(vector, &output) && output == 22);
  hyle_container_vector_clear(vector);
  CHECK(hyle_container_vector_size(vector) == 0);
  hyle_container_vector_destroy(vector);

  map_options.key_size = sizeof(int);
  map_options.key_alignment = HYLE_CONTAINER_ALIGNOF(int);
  map_options.value_size = sizeof(int);
  map_options.value_alignment = HYLE_CONTAINER_ALIGNOF(int);
  map_options.hash = constant_hash; /* force collision-chain coverage */
  map_options.backing = backing;
  map = hyle_container_hash_map_create(&map_options);
  CHECK(map != NULL);
  input = 1;
  output = 10;
  CHECK(hyle_container_hash_map_insert(map, &input, &output) == 1);
  input = 2;
  output = 20;
  CHECK(hyle_container_hash_map_insert(map, &input, &output) == 1);
  CHECK(hyle_container_hash_map_size(map) == 2);
  CHECK(hyle_container_hash_map_reserve(map, 64));
  CHECK(hyle_container_hash_map_capacity(map) >= 64);
  input = 1;
  CHECK(*(int*)hyle_container_hash_map_find(map, &input) == 10);
  CHECK(*(const int*)hyle_container_hash_map_find_const(map, &input) == 10);
  CHECK(hyle_container_hash_map_contains(map, &input));
  CHECK(hyle_container_hash_map_for_each(map, visit_map, &visits));
  CHECK(visits == 2);
  {
    int i;
    for (i = 3; i < 200; ++i) {
      int mapped = i * 10;
      CHECK(hyle_container_hash_map_insert(map, &i, &mapped) == 1);
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
        CHECK(hyle_container_hash_map_insert(map, &fill_key, &mapped) == 1);
        ++fill_key;
      }
      const int* aliased =
          (const int*)hyle_container_hash_map_find_const(map, &source_key);
      CHECK(aliased != NULL);
      CHECK(hyle_container_hash_map_insert(map, &fill_key, aliased) == 1);
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
    hyle_container_hash_map* aligned =
        hyle_container_hash_map_create(&aligned_options);
    int visited = 0;
    for (uint64_t i = 0; i < 40; ++i) {
      const char byte = (char)i;
      CHECK(hyle_container_hash_map_insert(aligned, &i, &byte) == 1);
    }
    CHECK(hyle_container_hash_map_for_each(aligned, visit_aligned_key,
                                           &visited));
    CHECK(visited == 40);
    hyle_container_hash_map_destroy(aligned);
  }

  CHECK(tracking.allocations > 0);
  CHECK(tracking.allocations == tracking.frees);
}

static void test_queues_and_concurrent_vector(void) {
  int a = 1;
  int b = 2;
  void* output = NULL;
  size_t cursor = 0;
  hyle_container_bounded_queue* bounded =
      hyle_container_bounded_queue_create_system(8);
  hyle_container_sharded_queue* sharded =
      hyle_container_sharded_queue_create_system(2, 4);
  hyle_container_work_stealing_deque* deque =
      hyle_container_work_stealing_deque_create_system(8);
  hyle_container_ws_fifo_queue* source =
      hyle_container_ws_fifo_queue_create_system(8);
  hyle_container_ws_fifo_queue* target =
      hyle_container_ws_fifo_queue_create_system(8);
  hyle_container_concurrent_vector* vector =
      hyle_container_concurrent_vector_create_system(2);

  CHECK(bounded != NULL && hyle_container_bounded_queue_capacity(bounded) == 8);
  CHECK(hyle_container_bounded_queue_push(bounded, &a));
  CHECK(hyle_container_bounded_queue_pop(bounded, &output) && output == &a);
  hyle_container_bounded_queue_destroy(bounded);

  CHECK(sharded != NULL);
  CHECK(hyle_container_sharded_queue_shard_count(sharded) == 2);
  CHECK(hyle_container_sharded_queue_capacity(sharded) == 8);
  CHECK(hyle_container_sharded_queue_push(sharded, 1, &b));
  CHECK(hyle_container_sharded_queue_pop(sharded, &cursor, &output) &&
        output == &b);
  hyle_container_sharded_queue_destroy(sharded);

  CHECK(deque != NULL);
  CHECK(hyle_container_work_stealing_deque_capacity(deque) == 8);
  CHECK(hyle_container_work_stealing_deque_push(deque, &a));
  CHECK(hyle_container_work_stealing_deque_steal(deque, &output) ==
        HYLE_CONTAINER_STEAL_OK);
  CHECK(output == &a);
  CHECK(!hyle_container_work_stealing_deque_pop(deque, &output));
  hyle_container_work_stealing_deque_destroy(deque);

  CHECK(source != NULL && target != NULL);
  CHECK(hyle_container_ws_fifo_queue_capacity(source) == 8);
  CHECK(hyle_container_ws_fifo_queue_push(source, &a));
  CHECK(hyle_container_ws_fifo_queue_push(source, &b));
  CHECK(hyle_container_ws_fifo_queue_steal_into(source, target, &output));
  CHECK(hyle_container_ws_fifo_queue_pop(source, &output) ||
        hyle_container_ws_fifo_queue_pop(target, &output));
  hyle_container_ws_fifo_queue_destroy(source);
  hyle_container_ws_fifo_queue_destroy(target);

  CHECK(vector != NULL);
  CHECK(hyle_container_concurrent_vector_push(vector, &a) == 0);
  CHECK(hyle_container_concurrent_vector_push(vector, &b) == 1);
  CHECK(hyle_container_concurrent_vector_size(vector) == 2);
  CHECK(hyle_container_concurrent_vector_capacity(vector) >= 2);
  CHECK(hyle_container_concurrent_vector_at(vector, 1) == &b);
  hyle_container_concurrent_vector_destroy(vector);
}

int main(void) {
  hyle_allocator invalid_backing = {0};
  hyle_allocator failing_backing = {fail_alloc, fail_free, fail_realloc, NULL};
  test_strings_and_paths();
  test_string_buf();
  test_vector_and_map();
  test_queues_and_concurrent_vector();
  CHECK(hyle_container_bounded_queue_create(invalid_backing, 8) == NULL);
  CHECK(hyle_container_last_error()[0] != '\0');
  CHECK(hyle_container_bounded_queue_create(failing_backing, 8) == NULL);
  CHECK(hyle_container_string_buf_create(failing_backing) == NULL);
  {
    /* One allocation for the C handle and one for StringBuf::Impl. */
    fail_after_allocator failing = {hyle_system_allocator(), 2};
    hyle_allocator fail_storage = {fail_after_alloc, fail_after_free,
                                   fail_after_realloc, &failing};
    hyle_container_string_buf* strings =
        hyle_container_string_buf_create(fail_storage);
    CHECK(strings != NULL);
    CHECK(hyle_container_string_buf_intern(strings, sv("allocation")).data ==
          NULL);
    CHECK(hyle_container_last_error()[0] != '\0');
    hyle_container_string_buf_destroy(strings);
  }
  {
    fail_after_allocator bounded = {hyle_system_allocator(), 1};
    fail_after_allocator sharded = {hyle_system_allocator(), 1};
    fail_after_allocator stealing = {hyle_system_allocator(), 1};
    fail_after_allocator sharing = {hyle_system_allocator(), 1};
    hyle_allocator a = {fail_after_alloc, fail_after_free, fail_after_realloc,
                        &bounded};
    CHECK(hyle_container_bounded_queue_create(a, 8) == NULL);
    a.state = &sharded;
    CHECK(hyle_container_sharded_queue_create(a, 2, 8) == NULL);
    a.state = &stealing;
    CHECK(hyle_container_work_stealing_deque_create(a, 8) == NULL);
    a.state = &sharing;
    CHECK(hyle_container_ws_fifo_queue_create(a, 8) == NULL);
  }
  {
    hyle_container_vector_options options =
        HYLE_CONTAINER_VECTOR_OPTIONS_FOR(int, 1);
    options.backing = failing_backing;
    CHECK(hyle_container_vector_create(&options) == NULL);
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
    map = hyle_container_hash_map_create(&options);
    CHECK(map != NULL);
    CHECK(hyle_container_hash_map_insert(map, &key, &value) == -1);
    CHECK(hyle_container_last_error()[0] != '\0');
    hyle_container_hash_map_destroy(map);
  }
  {
    fail_after_allocator failing = {hyle_system_allocator(), 1};
    hyle_allocator fail_storage = {fail_after_alloc, fail_after_free,
                                   fail_after_realloc, &failing};
    hyle_container_concurrent_vector* vector =
        hyle_container_concurrent_vector_create(fail_storage, 1);
    CHECK(vector != NULL);
    CHECK(hyle_container_concurrent_vector_push(vector, NULL) == SIZE_MAX);
    CHECK(hyle_container_last_error()[0] != '\0');
    hyle_container_concurrent_vector_destroy(vector);
  }
  CHECK(hyle_container_hash_bytes(NULL, 1, 0) == 0);
  CHECK(hyle_container_last_error()[0] != '\0');
  (void)hyle_container_hash_bytes(NULL, 0, 0);
  CHECK(hyle_container_last_error()[0] == '\0');
  printf("container_c_test: %s\n",
         g_failures == 0 ? "all checks passed" : "FAILED");
  return g_failures == 0 ? 0 : 1;
}
