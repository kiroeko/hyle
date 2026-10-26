// Finite host dispatch contracts; no model, native route or test framework.
#include "../../engine/inference/batch_dispatch.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <stdexcept>

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
using hyle::inference::BatchIndices;
using hyle::inference::dispatch_batches;
void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
template<class Error, class Operation>
void rejects(Operation operation) {
  try { operation(); }
  catch (const Error &) { return; }
  throw std::runtime_error("invalid batch dispatch contract was accepted");
}

void partial_admission_keeps_original_candidate_order() {
  BatchIndices indices; indices.prepare(7, 3);
  const std::array tags{'A', 'B', 'A', 'A', 'B', 'A', 'C'};
  const std::array eligible{true, true, true, false, true, true, true};
  std::array<std::size_t, 6> published{};
  std::array<unsigned, 7> visits{};
  std::size_t writes = 0, batches = 0;
  dispatch_batches(indices, tags.size(),
      [&](std::size_t index) { return eligible[index]; },
      [](std::size_t) { return std::size_t{3}; },
      [&](std::size_t first, std::size_t index) { return tags[first] == tags[index]; },
      [&](std::span<const std::size_t> selected) {
        // A's first resource admission accepts only candidate 0. Candidate 1
        // must run before the unadmitted A tail, and the ineligible A is absent.
        if (!batches) require(selected.size() == 3 && selected[0] == 0 &&
            selected[1] == 2 && selected[2] == 5, "selection included an ineligible peer or changed A order");
        const auto admitted = batches++ ? selected.size() : std::size_t{1};
        for (auto index : selected.first(admitted)) { published.at(writes++) = index; ++visits[index]; }
        return admitted;
      });
  require(published == std::array<std::size_t, 6>{0, 1, 4, 2, 5, 6} && writes == 6 && batches == 4,
      "partial admission lost a tail, duplicated a member or reordered distinct batch classes");
  for (std::size_t index = 0; index < visits.size(); ++index)
    require(visits[index] == (eligible[index] ? 1u : 0u) && indices.completed(index) == eligible[index],
        "dispatch completion did not describe exactly the eligible published candidates");
}

void singleton_limit_and_empty_waves_remain_explicit() {
  BatchIndices indices; indices.prepare(3, 3);
  std::array<std::size_t, 2> widths{};
  std::size_t batches = 0;
  const auto execute = [&](std::span<const std::size_t> selected) {
    widths.at(batches++) = selected.size(); return selected.size();
  };
  dispatch_batches(indices, 3, [](std::size_t) { return true; },
      [](std::size_t first) { return first ? std::size_t{2} : std::size_t{1}; },
      [](std::size_t, std::size_t) { return true; }, execute);
  require(widths == std::array<std::size_t, 2>{1, 2}, "singleton cache-hit limit was ignored");
  batches = 0;
  dispatch_batches(indices, 0, [](std::size_t) { return true; },
      [](std::size_t) { return std::size_t{3}; }, [](std::size_t, std::size_t) { return true; }, execute);
  dispatch_batches(indices, 3, [](std::size_t) { return false; },
      [](std::size_t) { return std::size_t{3}; }, [](std::size_t, std::size_t) { return true; }, execute);
  require(!batches, "empty or wholly ineligible wave executed work");
}

void observer_failure_is_propagated_without_retry_or_claimed_rollback() {
  struct ObserverFailure { unsigned identity; };
  BatchIndices indices; indices.prepare(2, 2);
  std::size_t calls = 0, caller_commits = 0;
  bool observed = false;
  try {
    dispatch_batches(indices, 2, [](std::size_t) { return true; },
        [](std::size_t) { return std::size_t{2}; }, [](std::size_t, std::size_t) { return true; },
        [&](std::span<const std::size_t>) -> std::size_t {
          ++calls; ++caller_commits; throw ObserverFailure{17};
        });
  } catch (const ObserverFailure &error) { observed = error.identity == 17; }
  require(observed && calls == 1 && caller_commits == 1 && !indices.completed(0) && !indices.completed(1),
      "observer failure was swallowed, retried or represented as rollback/completed dispatch");
}

void malformed_completion_cannot_skip_the_first_or_repeat_forever() {
  BatchIndices indices; indices.prepare(2, 2);
  const auto eligible = [](std::size_t) { return true; };
  const auto limit = [](std::size_t) { return std::size_t{2}; };
  const auto compatible = [](std::size_t, std::size_t) { return true; };
  std::size_t calls = 0;
  rejects<std::logic_error>([&] {
    dispatch_batches(indices, 2, eligible, limit, compatible,
        [&](std::span<const std::size_t>) { ++calls; return std::size_t{0}; });
  });
  require(calls == 1 && !indices.completed(0), "zero completion advanced or retried its candidate");
  rejects<std::out_of_range>([&] {
    dispatch_batches(indices, 2, eligible, limit, compatible,
        [&](std::span<const std::size_t>) { ++calls; return std::size_t{3}; });
  });
  require(calls == 2 && !indices.completed(0) && !indices.completed(1), "oversized completion marked a partial prefix");
  rejects<std::logic_error>([&] {
    dispatch_batches(indices, 2, eligible, limit,
        [](std::size_t first, std::size_t index) { return first != index; },
        [&](std::span<const std::size_t> selected) { ++calls; return selected.size(); });
  });
  require(calls == 2, "compatibility that excluded the first was executed");
}

void repeated_waves_reuse_cold_selection_storage() {
  BatchIndices indices; indices.prepare(8, 8); // Both index arrays exceed inline storage.
  const std::size_t *selection = nullptr;
  std::size_t calls = 0;
  bool stable = true;
  const auto execute = [&](std::span<const std::size_t> selected) {
    if (!selection) selection = selected.data();
    stable = stable && selected.data() == selection && selected.size() == 8;
    ++calls; return selected.size();
  };
  allocation_count = 0; track_allocations = true;
  for (unsigned wave = 0; wave < 128; ++wave)
    dispatch_batches(indices, 8, [](std::size_t) { return true; },
        [](std::size_t) { return std::size_t{8}; }, [](std::size_t, std::size_t) { return true; }, execute);
  track_allocations = false;
  require(stable && calls == 128 && allocation_count == 0,
      "hot dispatch changed cold selection storage or introduced C++ heap allocation");
}
}

int main() {
  try {
    partial_admission_keeps_original_candidate_order();
    singleton_limit_and_empty_waves_remain_explicit();
    observer_failure_is_propagated_without_retry_or_claimed_rollback();
    malformed_completion_cannot_skip_the_first_or_repeat_forever();
    repeated_waves_reuse_cold_selection_storage();
    std::puts("inference_batch_dispatch: PASS (5 host contracts)");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "inference_batch_dispatch: %s\n", error.what());
    return 1;
  }
}
