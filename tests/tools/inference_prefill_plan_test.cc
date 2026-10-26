// Shared host input partition facts only. No model, device or test framework.
#include "../../engine/inference/prefill_plan.h"

#include <array>
#include <bit>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace {
using hyle::inference::PrefillPart;
using hyle::inference::PrefillPolicy;
constexpr auto max_size = (std::numeric_limits<std::size_t>::max)();
static_assert(std::is_trivially_copyable_v<PrefillPart>);
static_assert(std::is_trivially_copyable_v<PrefillPolicy>);
static_assert(PrefillPolicy(3).next(8).rows == 3);
static_assert(PrefillPolicy(3).parts(8) == 3);
static_assert(PrefillPolicy().next(max_size).complete);

void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}

void require_part(PrefillPart actual, PrefillPart expected) {
  require(actual.begin == expected.begin && actual.rows == expected.rows &&
      actual.complete == expected.complete, "prefill slice differs from its input boundary");
}

template<class Error, class Operation> void rejects(Operation operation) {
  try { operation(); }
  catch (const Error &) { return; }
  throw std::runtime_error("invalid prefill extent or quantum was accepted");
}

void invalid_and_finished_inputs() {
  rejects<std::invalid_argument>([] { (void)PrefillPolicy(0); });
  const PrefillPolicy policy(4);
  rejects<std::out_of_range>([&] { (void)policy.next(3, 4); });
  rejects<std::out_of_range>([&] { (void)policy.parts(3, 4); });
  require_part(policy.next(0), {0, 0, true});
  require_part(policy.next(9, 9), {9, 0, true});
  require(policy.parts(0) == 0 && policy.parts(9, 9) == 0,
      "empty or fully reused input gained a Graph episode");
}

void equal_chunks_and_remainder() {
  const PrefillPolicy policy(3);
  const std::array expected{PrefillPart{0, 3, false}, PrefillPart{3, 3, false}, PrefillPart{6, 2, true}};
  std::size_t consumed = 0;
  require(policy.parts(8) == expected.size(), "cold remainder episode count differs");
  for (const auto part : expected) {
    const auto actual = policy.next(8, consumed);
    require_part(actual, part);
    consumed += actual.rows; // The caller commits the input, not the policy.
  }
  require(consumed == 8 && policy.parts(8, consumed) == 0,
      "prefill partition lost or repeated an input row");
  require_part(policy.next(6, 3), {3, 3, true});
  require(policy.parts(6) == 2, "exact division gained an empty tail");
}

void consumed_offsets_are_borrowed_facts() {
  const PrefillPolicy policy(3);
  const std::size_t consumed = 2;
  const auto first = policy.next(9, consumed);
  require_part(first, {2, 3, false});
  require_part(policy.next(9, consumed), first);
  require(consumed == 2 && policy.parts(9, consumed) == 3,
      "planning mutated the caller cursor or ignored a reused prefix");
  require_part(policy.next(9, 8), {8, 1, true});
  require(policy.parts(9, 8) == 1, "remaining suffix was not planned independently");
}

void maximum_extents_do_not_overflow() {
  const PrefillPolicy unbounded;
  require_part(unbounded.next(max_size), {0, max_size, true});
  require_part(unbounded.next(max_size, max_size - 1), {max_size - 1, 1, true});
  require_part(unbounded.next(max_size, max_size), {max_size, 0, true});
  require(unbounded.parts(max_size) == 1 && unbounded.parts(max_size, max_size) == 0,
      "unbounded quantum overflowed its episode count");
  require(PrefillPolicy(1).parts(max_size) == max_size,
      "unit quantum overflowed its exact maximum count");
  const PrefillPolicy pair(2);
  require(pair.parts(max_size) == max_size / 2 + max_size % 2,
      "rounded maximum episode count overflowed");
  require_part(pair.next(max_size, max_size - 2), {max_size - 2, 2, true});
  require(PrefillPolicy(max_size - 1).parts_bound(max_size) >=
      PrefillPolicy(max_size - 1).parts(max_size - 2), "finite maximum suffix bound overflowed");
}

void finite_tail_blocks_keep_every_real_row() {
  const PrefillPolicy policy(8);
  const std::array expected{PrefillPart{0, 8, false}, PrefillPart{8, 8, false},
      PrefillPart{16, 4, false}, PrefillPart{20, 2, false}, PrefillPart{22, 1, true}};
  std::size_t consumed = 0;
  for (const auto part : expected) {
    require_part(policy.next(23, consumed), part);
    consumed += part.rows;
  }
  require(consumed == 23 && policy.parts(23) == expected.size(),
      "bounded tail planning lost input or disagreed with actual episodes");
  require_part(policy.next(8, 1), {1, 4, false});
  require(policy.parts(8) == 1 && policy.parts(8, 1) == 3 && policy.parts_bound(8) >= 3,
      "unknown prefix reservation assumed a monotonic exact episode count");
  require_part(PrefillPolicy().next(23), {0, 23, true});
}

void possible_keys_match_real_suffixes() {
  for (const auto quantum : std::array<std::size_t, 6>{1, 2, 3, 5, 8, 16}) {
    const PrefillPolicy policy(quantum);
    for (std::size_t maximum = 0; maximum <= 40; ++maximum) {
      std::array<std::array<bool, 2>, 41> declared{}, witnessed{};
      std::size_t keys = 0;
      policy.for_each_shape(maximum, [&](std::size_t rows, bool complete) {
        require(rows && rows <= quantum && rows <= maximum && !declared[rows][complete],
            "cold suffix keys contained empty/padded/duplicate extents");
        declared[rows][complete] = true; ++keys;
      });
      for (std::size_t remaining = 1; remaining <= maximum; ++remaining) {
        const auto part = policy.next(remaining);
        witnessed[part.rows][part.complete] = true;
        require(policy.parts(remaining) <= policy.parts_bound(maximum),
            "cold maximum did not cover a shorter unknown suffix");
      }
      require(declared == witnessed && keys <= 2 * (std::size_t{1} + static_cast<std::size_t>(std::bit_width(quantum - 1))),
          "cold keys differed from possible real next blocks or grew with context length");
    }
  }
}
} // namespace

int main() {
  try {
    invalid_and_finished_inputs();
    equal_chunks_and_remainder();
    consumed_offsets_are_borrowed_facts();
    maximum_extents_do_not_overflow();
    finite_tail_blocks_keep_every_real_row();
    possible_keys_match_real_suffixes();
    std::puts("prefill plan: 6 host contract groups passed; no model/device loaded");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "prefill plan: %s\n", error.what());
    return 1;
  }
}
