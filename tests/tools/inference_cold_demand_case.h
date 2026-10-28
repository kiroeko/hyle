#pragma once

#include <array>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../../engine/inference/cold_demand.h"

namespace inference_cold_demand_case {
namespace inference = hyle::inference;

inline void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

// Pure host metadata contract. No Model, fixture, Graph or native submission.
inline int run() {
  using Key = std::pair<int, int>;
  using Inventory = inference::ColdDemandInventory<Key>;
  using Kind = inference::ColdDemandKind;
  const Key a{1, 0}, b{1, 1};
  const std::array<std::size_t, 4> sampling{0, 1, 0, 1};
  const auto policy0 = [&](std::size_t member) { return sampling.at(member) == 0; };
  const auto policy1 = [&](std::size_t member) { return sampling.at(member) == 1; };
  Inventory demand;
  demand.inventory(a, 1);
  const auto *initial = demand.find(a);
  require(initial && initial->size() == 1 && initial->at(1).uses == 0 &&
      initial->at(1).members.empty() && initial->at(1).eligible_count(policy0) == 0 &&
      Inventory::eligible_members(*initial, policy0) == 0,
      "inventory-only shape acquired a hinted use or request member");

  // A Possible note followed by a later Counted note retains the original
  // ordered conservative hint; it is not max(total-counted, possible).
  demand.note(a, 3, 0, Kind::kPossible);
  demand.note(a, 3, 1, Kind::kCounted);
  demand.note(a, 3, 1, Kind::kPossible);
  demand.note(a, 3, 0, Kind::kCounted);
  demand.inventory(a, 3);
  const auto &three = demand.find(a)->at(3);
  require(three.uses == 3 && three.members == std::vector<std::size_t>{0, 1} &&
      three.eligible_count(policy0) == 1 && three.eligible_count(policy1) == 1,
      "ordered Counted/Possible notes or unique member support changed");
  demand.note(a, 5, 0, Kind::kCounted);
  demand.note(a, 5, 0, Kind::kPossible);
  require(demand.uses(a, 5) == 1,
      "Possible incorrectly incremented an existing Counted hint");
  demand.note(a, 8, 2, Kind::kPossible);
  demand.note(b, 8, 3, Kind::kCounted);

  std::vector<std::pair<std::size_t, std::size_t>> rows0, rows1;
  const auto count0 = Inventory::visit_support(*demand.find(a), policy0,
      [&](std::size_t rows, const Inventory::Entry &, std::size_t eligible) {
        rows0.emplace_back(rows, eligible);
      });
  const auto count1 = Inventory::visit_support(*demand.find(a), policy1,
      [&](std::size_t rows, const Inventory::Entry &, std::size_t eligible) {
        rows1.emplace_back(rows, eligible);
      });
  const std::vector<std::pair<std::size_t, std::size_t>> expected0{{1, 0}, {3, 1}, {5, 1}, {8, 2}};
  const std::vector<std::pair<std::size_t, std::size_t>> expected1{{1, 0}, {3, 1}, {5, 1}, {8, 1}};
  require(rows0 == expected0 && rows1 == expected1 && count0 == 2 && count1 == 1 &&
      Inventory::eligible_members(*demand.find(a), policy0) == count0 &&
      Inventory::eligible_members(*demand.find(b), policy0) == 0 &&
      Inventory::eligible_members(*demand.find(b), policy1) == 1 &&
      !demand.find(a)->at(8).supports(policy1),
      "cumulative row support duplicated members, mixed policies or crossed keys");

  // Test the same Entry mutation used by the inventory without adding an
  // unrestricted setter or a synthetic bulk-count API to the container.
  Inventory::Entry full{(std::numeric_limits<std::size_t>::max)(), {0}};
  bool overflow = false;
  try { full.note(1, Kind::kCounted); }
  catch (const std::overflow_error &) { overflow = true; }
  require(overflow && full.uses == (std::numeric_limits<std::size_t>::max)() &&
      full.members == std::vector<std::size_t>{0},
      "overflow was accepted or published partial member support");
  full.note(1, Kind::kPossible);
  require(full.uses == (std::numeric_limits<std::size_t>::max)() &&
      full.members == std::vector<std::size_t>{0, 1},
      "Possible incorrectly overflowed the existing maximum hint");
  require(demand.find(Key{9, 9}) == nullptr && demand.uses(Key{9, 9}, 1) == 0 &&
      demand.find(a)->at(1).uses == 0 && demand.find(a)->at(1).members.empty(),
      "read-only queries changed missing or zero-use inventory facts");

  std::cout << "{\"status\":\"PASS\",\"case\":\"cold-demand\",\"models\":0,\"native_submissions\":0,"
      "\"inventory_zero_without_support\":true,\"ordered_counted_possible\":true,\"unique_members\":true,"
      "\"cumulative_rows_policy_and_key_isolation\":true,\"overflow_without_partial_note\":true,"
      "\"use_hints_not_measured_native_calls\":true,\"performance_claim\":false}\n";
  return 0;
}
} // namespace inference_cold_demand_case
