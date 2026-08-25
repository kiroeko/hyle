#include <hyle/c/container.h>
#include <hyle/c/error_helpers.h>

#include <stdint.h>

#if defined(_MSC_VER)
typedef char installed_insert_result_must_be_u32[
    sizeof(hyle_container_insert_result) == sizeof(uint32_t) ? 1 : -1];
typedef char installed_steal_status_must_be_u32[
    sizeof(hyle_container_steal_status) == sizeof(uint32_t) ? 1 : -1];
#else
_Static_assert(sizeof(hyle_container_insert_result) == sizeof(uint32_t),
               "installed insert-result ABI must stay fixed-width");
_Static_assert(sizeof(hyle_container_steal_status) == sizeof(uint32_t),
               "installed steal-status ABI must stay fixed-width");
#endif

static int status_ok_and_free(hyle_status status) {
  const int ok = hyle_status_is_ok(status);
  hyle_status_free(&status);
  return ok;
}

int main(void) {
  const unsigned char message[] = {0, 1, 2};
  if (!hyle_container_utf8_validate(hyle_container_cstr("installed-✓"))) {
    return 1;
  }
  if (hyle_container_siphash24(
          message, sizeof(message), UINT64_C(0x0706050403020100),
          UINT64_C(0x0f0e0d0c0b0a0908)) !=
      UINT64_C(0x85676696d7fb7e2d)) {
    return 2;
  }

  hyle_container_hash_map_options options =
      HYLE_CONTAINER_HASH_MAP_OPTIONS_FOR(uint64_t, uint64_t);
  hyle_container_hash_map* map = NULL;
  if (!status_ok_and_free(hyle_container_hash_map_create(&map, &options)))
    return 3;
  const uint64_t key = 7;
  const uint64_t value = 11;
  hyle_container_insert_result inserted = HYLE_CONTAINER_ALREADY_PRESENT;
  if (!status_ok_and_free(
          hyle_container_hash_map_insert(&inserted, map, &key, &value)) ||
      inserted != HYLE_CONTAINER_INSERTED) {
    hyle_container_hash_map_destroy(map);
    return 4;
  }
  hyle_container_hash_map_destroy(map);

  hyle_container_work_stealing_deque* deque = NULL;
  if (!status_ok_and_free(
          hyle_container_work_stealing_deque_create_system(&deque, 8)))
    return 5;
  void* item = NULL;
  const hyle_container_steal_status steal =
      hyle_container_work_stealing_deque_try_steal_top(deque, &item);
  hyle_container_work_stealing_deque_destroy(deque);
  return steal == HYLE_CONTAINER_STEAL_EMPTY && item == NULL ? 0 : 6;
}
