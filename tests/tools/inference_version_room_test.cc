// Finite directory/version metadata contracts; no native retirement is simulated.
#include "../../engine/inference/version_admission.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <new>

namespace {
bool track_allocations = false;
std::size_t allocation_count = 0;
#if defined(__GNUC__)
[[gnu::noinline]]
#endif
void free_test_allocation(void *memory) noexcept { std::free(memory); }
}
void *operator new(std::size_t bytes) {
  if (track_allocations) ++allocation_count;
  if (auto *memory = std::malloc(bytes ? bytes : 1)) return memory;
  throw std::bad_alloc{};
}
void *operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void *memory) noexcept { free_test_allocation(memory); }
void operator delete[](void *memory) noexcept { free_test_allocation(memory); }
void operator delete(void *memory, std::size_t) noexcept { free_test_allocation(memory); }
void operator delete[](void *memory, std::size_t) noexcept { free_test_allocation(memory); }

namespace {
namespace i = hyle::inference;
struct Version {};
struct Session {};
using Holder = i::VersionHolder<Session, Version>;
void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
template<class Operation>
void rejects_inventory(Operation operation) {
  try { operation(); }
  catch (const std::logic_error &) { return; }
  throw std::runtime_error("invalid or growing version inventory was accepted");
}

void aliases_require_fresh_complete_admission_facts() {
  Version version; Session session;
  const std::array owners{Holder{&session, &version}};
  std::size_t aliases = 2, observations = 0, evictions = 0;
  const auto observe = [&] {
    ++observations;
    const i::VersionCapacity capacity{1, 1};
    return i::VersionRoomObservation{capacity, aliases,
        i::version_admission(capacity, owners, owners, [&](const Version *) { return aliases != 0; }).fits};
  };
  allocation_count = 0; track_allocations = true;
  const auto result = i::make_version_room(observe, [&] { --aliases; ++evictions; return true; });
  track_allocations = false;
  require(result == i::VersionRoomResult::kReady && observations == 3 && evictions == 2 && !aliases,
      "one alias removal was mistaken for free capacity or complete admission was not reread");
  require(!allocation_count, "valid version room traversal allocated host storage");
}

void exhausted_directory_does_not_invent_released_versions() {
  const std::array fresh{Holder{}};
  const std::array<Holder, 0> residents{};
  std::size_t entries = 2, evictions = 0, observations = 0;
  const auto result = i::make_version_room([&] {
    ++observations;
    const i::VersionCapacity capacity{2, 2}; // External readers keep both versions alive.
    return i::VersionRoomObservation{capacity, entries,
        i::version_admission(capacity, fresh, residents, [](const Version *) { return false; }).fits};
  }, [&] { --entries; ++evictions; return true; });
  require(result == i::VersionRoomResult::kRetained && !entries && evictions == 2 && observations == 3,
      "removed entries were credited as versions, bytes or an unbounded extra eviction");
}

void ready_empty_and_unavailable_candidates_are_distinct() {
  std::size_t evictions = 0;
  volatile bool candidate_available = false;
  const auto evict = [&] { ++evictions; return candidate_available; };
  const auto ready = i::make_version_room([] { return i::VersionRoomObservation{{1, 2}, 2, true}; }, evict);
  require(ready == i::VersionRoomResult::kReady && !evictions, "ready admission evicted an unnecessary entry");
  const auto empty = i::make_version_room([] { return i::VersionRoomObservation{{2, 2}, 0, false}; }, evict);
  require(empty == i::VersionRoomResult::kRetained && !evictions, "empty candidate inventory invoked eviction");
  const auto pinned = i::make_version_room([] { return i::VersionRoomObservation{{2, 2}, 2, false}; }, evict);
  require(pinned == i::VersionRoomResult::kRetained && evictions == 1,
      "unavailable candidate was retried or treated as admitted");
}

void illegal_progress_is_rejected_even_when_callback_claims_fit() {
  const std::array invalid_after{
      i::VersionRoomObservation{{2, 2}, 1, true}, // Live inventory grew.
      i::VersionRoomObservation{{1, 3}, 1, true}, // Limit grew implicitly.
      i::VersionRoomObservation{{1, 1}, 1, true}, // Limit shrank implicitly.
      i::VersionRoomObservation{{1, 2}, 2, true}, // No candidate consumed.
      i::VersionRoomObservation{{1, 2}, 3, true}}; // Candidate inventory grew.
  for (const auto after : invalid_after) {
    std::size_t observations = 0, evictions = 0;
    rejects_inventory([&] {
      (void)i::make_version_room([&] {
        return observations++ ? after : i::VersionRoomObservation{{1, 2}, 2, false};
      }, [&] { ++evictions; return true; });
    });
    require(observations == 2 && evictions == 1, "invalid inventory was retried or accepted as a fit");
  }
  for (const auto capacity : {i::VersionCapacity{0, 0}, i::VersionCapacity{3, 2}}) {
    std::size_t evictions = 0;
    rejects_inventory([&] {
      (void)i::make_version_room([&] { return i::VersionRoomObservation{capacity, 1, true}; },
          [&] { ++evictions; return true; });
    });
    require(!evictions, "invalid initial capacity mutated the directory");
  }
}

void callback_errors_propagate_without_retry_or_rollback() {
  struct OwnerError { unsigned identity; };
  std::size_t evictions = 0, observations = 0;
  volatile bool inject_error = true;
  bool original_error = false;
  try {
    (void)i::make_version_room([] { return i::VersionRoomObservation{{1, 1}, 2, false}; },
        [&]() -> bool { ++evictions; if (inject_error) throw OwnerError{17}; return false; });
  } catch (const OwnerError &error) { original_error = error.identity == 17; }
  require(original_error && evictions == 1, "owner error was replaced, swallowed or retried");
  original_error = false;
  try {
    (void)i::make_version_room([&]() -> i::VersionRoomObservation {
      if (observations++) throw OwnerError{23};
      return {{1, 1}, 2, false};
    }, [&] { ++evictions; return true; });
  } catch (const OwnerError &error) { original_error = error.identity == 23; }
  require(original_error && observations == 2 && evictions == 2,
      "post-eviction observation error was retried or caller mutation was represented as rollback");
}
}

int main() {
  try {
    aliases_require_fresh_complete_admission_facts();
    exhausted_directory_does_not_invent_released_versions();
    ready_empty_and_unavailable_candidates_are_distinct();
    illegal_progress_is_rejected_even_when_callback_claims_fit();
    callback_errors_propagate_without_retry_or_rollback();
    std::puts("inference_version_room: PASS (5 host contracts)");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "inference_version_room: %s\n", error.what());
    return 1;
  }
}
