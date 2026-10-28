// Shared host input partition facts only. No model, device or test framework.
#include "../../engine/inference/prefill_plan.h"
#include "../../engine/inference/packed_rows.h"
#include "../../engine/inference/cold_shape_plan.h"

#include <array>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace {
using hyle::inference::PrefillPart;
using hyle::inference::PrefillPolicy;
using hyle::inference::PackedRowInput;
using hyle::inference::pack_sequence_rows;
constexpr auto max_size = (std::numeric_limits<std::size_t>::max)();
static_assert(std::is_trivially_copyable_v<PrefillPart>);
static_assert(std::is_trivially_copyable_v<PrefillPolicy>);
static_assert(PrefillPolicy(3).next(8).rows == 3);
static_assert(PrefillPolicy(3).parts(8) == 3);
static_assert(PrefillPolicy(64).parts(154) == 3);
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
  require(PrefillPolicy(max_size - 1).parts(max_size) >=
      PrefillPolicy(max_size - 1).parts(max_size - 2), "finite maximum suffix bound overflowed");
}

void finite_tail_blocks_keep_every_real_row() {
  const PrefillPolicy policy(8);
  const std::array expected{PrefillPart{0, 8, false}, PrefillPart{8, 8, false},
      PrefillPart{16, 7, true}};
  std::size_t consumed = 0;
  for (const auto part : expected) {
    require_part(policy.next(23, consumed), part);
    consumed += part.rows;
  }
  require(consumed == 23 && policy.parts(23) == expected.size(),
      "bounded tail planning lost input or disagreed with actual episodes");
  require_part(policy.next(8, 1), {1, 7, true});
  require(policy.parts(8) == 1 && policy.parts(8, 1) == 1,
      "short reused suffix gained artificial episodes");
  const PrefillPolicy realistic(64);
  require_part(realistic.next(154,128), {128,26,true});
  require(realistic.parts(154) == 3 && realistic.parts(154,1) == 3,
      "P154/Q64 did not retain a single real tail");
  require_part(PrefillPolicy().next(23), {0, 23, true});
}

void cold_capacities_cover_real_suffixes() {
  for (const auto quantum : std::array<std::size_t, 6>{1, 2, 3, 5, 8, 16}) {
    const PrefillPolicy policy(quantum);
    for (std::size_t maximum = 0; maximum <= 40; ++maximum) {
      std::array<std::array<bool, 2>, 41> declared{}, used{};
      std::size_t keys = 0;
      policy.for_each_capacity(maximum, [&](std::size_t rows, bool complete) {
        require(rows && rows <= quantum && rows <= maximum && !declared[rows][complete],
            "cold capacities contained empty/oversized/duplicate extents");
        declared[rows][complete] = true; ++keys;
      });
      for (std::size_t remaining = 1; remaining <= maximum; ++remaining) {
        const auto part = policy.next(remaining);
        bool covered = false;
        for (std::size_t capacity = part.rows; capacity <= maximum; ++capacity)
          if (declared[capacity][part.complete]) { covered = true; used[capacity][part.complete] = true; break; }
        require(covered,"real suffix did not fit its declared cold capacity/completion");
        require(policy.parts(remaining) <= policy.parts(maximum),
            "cold maximum did not cover a shorter unknown suffix");
      }
      require(declared == used && keys <= 2,
          "cold capacities included unused keys or grew with every exact tail length");
    }
  }
}

void packed_rows_use_actual_member_boundaries() {
  // Capacity need not be rectangular or divisible by B. Both producers use
  // the same caller buffers; reordering changes geometry, not a Session cursor.
  std::array<std::int32_t, 3> offsets{};
  std::array<std::int32_t, 2> last{};
  std::array<std::int32_t, 7> positions{};
  const std::array target{PackedRowInput{10, 1}, PackedRowInput{20, 3}};
  const auto facts = pack_sequence_rows(target, positions.size(), offsets, last, positions);
  require(facts.active_rows == 4 && facts.max_member_rows == 3 &&
      offsets == std::array<std::int32_t, 3>{0, 1, 4} &&
      last == std::array<std::int32_t, 2>{0, 3} &&
      positions == std::array<std::int32_t, 7>{10, 20, 21, 22, 0, 0, 0},
      "packed target geometry consumed padding or crossed a member boundary");
  const std::array reordered{target[1], target[0]};
  (void)pack_sequence_rows(reordered, positions.size(), offsets, last, positions);
  require(offsets == std::array<std::int32_t, 3>{0, 3, 4} &&
      last == std::array<std::int32_t, 2>{2, 3} &&
      positions == std::array<std::int32_t, 7>{20, 21, 22, 10, 0, 0, 0},
      "packed metadata reused an old member ordering");

  // The model supplies its actual lag-adjusted rows/frontiers. The shared
  // function has no MTP role, hidden-state or permission policy of its own.
  const std::array draft{PackedRowInput{9, 2}, PackedRowInput{19, 4}};
  const auto draft_facts = pack_sequence_rows(draft, positions.size(), offsets, last, positions);
  require(draft_facts.active_rows == 6 && draft_facts.max_member_rows == 4 &&
      offsets == std::array<std::int32_t, 3>{0, 2, 6} &&
      last == std::array<std::int32_t, 2>{1, 5} &&
      positions == std::array<std::int32_t, 7>{9, 10, 19, 20, 21, 22, 0},
      "packed draft geometry borrowed target offsets or advanced padded rows");
}

void packed_rows_reject_before_writing_any_buffer() {
  constexpr std::int32_t sentinel = -77;
  std::array<std::int32_t, 3> offsets; offsets.fill(sentinel);
  std::array<std::int32_t, 2> last; last.fill(sentinel);
  std::array<std::int32_t, 5> positions; positions.fill(sentinel);
  const auto unchanged = [&] {
    for (auto value : offsets) require(value == sentinel, "failed pack changed offsets");
    for (auto value : last) require(value == sentinel, "failed pack changed last rows");
    for (auto value : positions) require(value == sentinel, "failed pack changed positions");
  };
  const auto late_error = [&](PackedRowInput bad) {
    const std::array inputs{PackedRowInput{10, 2}, bad};
    rejects<std::out_of_range>([&] {
      (void)pack_sequence_rows(inputs, positions.size(), offsets, last, positions);
    });
    unchanged();
  };
  late_error({0, 4}); // The sum exceeds capacity although each member fits.
  late_error({max_size, 1});
  late_error({0, max_size});
  const auto i32_max = static_cast<std::size_t>((std::numeric_limits<std::int32_t>::max)());
  late_error({i32_max, 2});
  const std::array empty_member{PackedRowInput{0, 2}, PackedRowInput{0, 0}};
  rejects<std::invalid_argument>([&] {
    (void)pack_sequence_rows(empty_member, positions.size(), offsets, last, positions);
  });
  unchanged();
  const std::array good{PackedRowInput{0, 2}, PackedRowInput{10, 2}};
  rejects<std::invalid_argument>([&] {
    (void)pack_sequence_rows(good, positions.size(), std::span(offsets).first(2), last, positions);
  });
  unchanged();
  rejects<std::invalid_argument>([&] {
    (void)pack_sequence_rows(good, positions.size(), offsets,
        std::span(offsets).first(2), positions);
  });
  unchanged();
  rejects<std::invalid_argument>([&] {
    (void)pack_sequence_rows({}, positions.size(), offsets, last, positions);
  });
  unchanged();
}

void packed_rows_preserve_exact_i32_position_limit() {
  const auto limit = (std::numeric_limits<std::int32_t>::max)();
  const std::array input{PackedRowInput{static_cast<std::size_t>(limit), 1}};
  std::array<std::int32_t, 2> offsets{};
  std::array<std::int32_t, 1> last{}, positions{};
  const auto facts = pack_sequence_rows(input, 1, offsets, last, positions);
  require(facts.active_rows == 1 && facts.max_member_rows == 1 &&
      offsets == std::array<std::int32_t, 2>{0, 1} && last[0] == 0 && positions[0] == limit,
      "packed rows rejected or narrowed the final valid I32 position");
  const auto previous_offsets = offsets; const auto previous_last = last;
  const auto previous_positions = positions;
  rejects<std::out_of_range>([&] {
    (void)pack_sequence_rows(input, static_cast<std::size_t>(limit) + 1, offsets, last, positions);
  });
  rejects<std::invalid_argument>([&] {
    (void)pack_sequence_rows(input, 0, offsets, last, positions);
  });
  require(offsets == previous_offsets && last == previous_last && positions == previous_positions,
      "invalid physical extent changed the previous metadata");
}

using ShapePlan = hyle::inference::ColdShapePlan<int, int, 3>;
struct BuildFailure {};

void cold_shape_counts_and_freeze() {
  ShapePlan plan;
  plan.add_uses(10, 2); plan.add_use(10, 1, 3); plan.add_uses(20, 1);
  require(plan.uses(10) == ShapePlan::Uses{2, 5, 2} && plan.size() == 2,
      "cold use plan lost a physical slot or shape");
  require(plan.freeze() && !plan.freeze() && plan.capacity() == 2 && !plan.retained(),
      "frozen capacity changed or included an unadmitted shape");
  ShapePlan empty;
  int hooks = 0;
  rejects<std::logic_error>([&] {
    empty.admit(1, 10, [&](int, int) { ++hooks; }, [&] { ++hooks; }, []() noexcept {});
  });
  require(!hooks && !empty.retained(), "unfrozen admission reached a resource hook");
  require(empty.freeze() && empty.capacity() == 1, "empty plan lost its bounded suffix slot");
}

void cold_shape_rejects_overflow_without_partial_counts() {
  ShapePlan plan;
  plan.add_use(10, 1, max_size);
  const auto before = plan.uses(10);
  rejects<std::runtime_error>([&] { plan.add_uses(10, 1); });
  require(plan.uses(10) == before, "overflow partially changed a physical slot's use count");
  rejects<std::out_of_range>([&] { plan.add_use(20, 3, 1); });
  require(!plan.contains(20), "invalid slot published a new shape");
}

void cold_shape_requires_retirement_before_eviction() {
  ShapePlan plan; plan.freeze();
  int resident = 0, builds = 0, retire_calls = 0;
  bool retired = false;
  const auto retire = [&](int key, int shape) {
    ++retire_calls;
    require(key == resident && shape == key / 10, "eviction named a foreign program");
    if (!retired) throw BuildFailure{};
    resident = 0;
  };
  plan.admit(1, 10, retire, [&] { resident = 10; ++builds; }, [&]() noexcept { resident = 0; });
  rejects<BuildFailure>([&] {
    plan.admit(2, 20, retire, [&] { resident = 20; ++builds; }, [&]() noexcept { resident = 0; });
  });
  require(resident == 10 && builds == 1 && plan.contains(1) && !plan.contains(2) &&
      plan.retained() == 1 && plan.admissions() == 1 && plan.evictions() == 0,
      "unretired victim was removed or a replacement was built");
  retired = true;
  plan.admit(2, 20, retire, [&] {
    require(!resident && !plan.contains(1), "replacement preceded actual victim release");
    resident = 20; ++builds;
  }, [&]() noexcept { resident = 0; });
  require(resident == 20 && builds == 2 && retire_calls == 2 && !plan.contains(1) &&
      plan.uses(2) == ShapePlan::Uses{1, 1, 1} && plan.admissions() == 2 && plan.evictions() == 1,
      "retired replacement lost its bounded admission facts");
}

void cold_shape_failed_build_preserves_fifo_and_existing_plan() {
  ShapePlan plan;
  plan.add_uses(90, 7); plan.add_uses(91, 9); plan.freeze();
  int victim = 0, rollback = 0;
  const auto retire = [&](int key, int) { victim = key; };
  const auto clean = [&]() noexcept { ++rollback; };
  plan.admit(1, 10, retire, [] {}, clean);
  rejects<BuildFailure>([&] { plan.admit(2, 20, retire, [] { throw BuildFailure{}; }, clean); });
  require(!victim && rollback == 1 && !plan.contains(2) && plan.retained() == 1 &&
      plan.admissions() == 1 && plan.uses(90) == ShapePlan::Uses{7, 7, 7},
      "failed build published use/FIFO metadata or changed the original plan");
  plan.admit(2, 20, retire, [] {}, clean);
  plan.admit(3, 30, retire, [] {}, clean);
  require(victim == 10 && !plan.contains(1) && plan.contains(2) && plan.contains(3) &&
      plan.retained() == 2 && plan.admissions() == 3 && plan.evictions() == 1,
      "failed build changed the next successful eviction order");
}

void cold_shape_failed_replacement_leaves_no_unpublished_usage() {
  ShapePlan plan; plan.freeze();
  int resident = 0, rollback = 0;
  const auto retire = [&](int, int) { resident = 0; };
  const auto clean = [&]() noexcept { resident = 0; ++rollback; };
  plan.admit(1, 10, retire, [&] { resident = 10; }, clean);
  rejects<BuildFailure>([&] {
    plan.admit(2, 20, retire, [&] { resident = 20; throw BuildFailure{}; }, clean);
  });
  require(!resident && rollback == 1 && !plan.retained() && !plan.contains(1) && !plan.contains(2) &&
      plan.admissions() == 1 && plan.evictions() == 1, "failed replacement retained an unpublished program");
  plan.admit(2, 20, retire, [&] { resident = 20; }, clean);
  require(resident == 20 && plan.admissions() == 2 && plan.evictions() == 1,
      "retry counted a second eviction for an already released victim");
}

void cold_shape_existing_counts_and_duplicate_admissions() {
  ShapePlan plan; plan.add_uses(1, 6); plan.freeze();
  int calls = 0;
  const auto retire = [&](int, int) { ++calls; };
  const auto clean = []() noexcept {};
  rejects<BuildFailure>([&] { plan.admit(1, 10, retire, [] { throw BuildFailure{}; }, clean); });
  require(plan.uses(1) == ShapePlan::Uses{6, 6, 6} && !plan.retained(),
      "failed build overwrote the existing planned shape");
  plan.admit(1, 10, retire, [&] { ++calls; }, clean);
  const auto before = calls;
  rejects<std::logic_error>([&] { plan.admit(1, 10, retire, [&] { ++calls; }, clean); });
  require(calls == before && plan.admissions() == 1 && !plan.evictions() && plan.retained() == 1,
      "duplicate admission executed native hooks or changed capacity accounting");
}
} // namespace

int main() {
  try {
    invalid_and_finished_inputs();
    equal_chunks_and_remainder();
    consumed_offsets_are_borrowed_facts();
    maximum_extents_do_not_overflow();
    finite_tail_blocks_keep_every_real_row();
    cold_capacities_cover_real_suffixes();
    packed_rows_use_actual_member_boundaries();
    packed_rows_reject_before_writing_any_buffer();
    packed_rows_preserve_exact_i32_position_limit();
    std::puts("prefill plan: 9 host contract groups passed; no model/device loaded");
    cold_shape_counts_and_freeze();
    cold_shape_rejects_overflow_without_partial_counts();
    cold_shape_requires_retirement_before_eviction();
    cold_shape_failed_build_preserves_fifo_and_existing_plan();
    cold_shape_failed_replacement_leaves_no_unpublished_usage();
    cold_shape_existing_counts_and_duplicate_admissions();
    std::puts("cold shape plan: 6 host contract groups passed; no model/device loaded");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "prefill plan: %s\n", error.what());
    return 1;
  }
}
