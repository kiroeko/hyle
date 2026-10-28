#pragma once

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>

#include "../../engine/inference/packed_rows.h"

namespace inference_packed_prefix_rows_case {

inline void run() {
  const auto require = [](bool condition) {
    if (!condition) throw std::runtime_error("packed prefix row contract failed");
  };
  std::array<std::int32_t, 6> hidden{};
  std::array<std::int32_t, 12> history{};
  const std::array<std::int32_t, 3> short_first{0, 1, 4}, long_first{0, 3, 4};
  auto facts = hyle::inference::pack_sequence_prefixes(short_first, 3, 2, hidden, history);
  require(facts.active_rows == 4 && facts.max_member_rows == 3);
  require(hidden == std::array<std::int32_t, 6>{0, 0, 0, 1, 2, 3});
  require(history == std::array<std::int32_t, 12>{1, 4, 1, 4, 1, 4, 3, 5, 5, 6, 6, 7});
  facts = hyle::inference::pack_sequence_prefixes(long_first, 3, 2, hidden, history);
  require(facts.active_rows == 4 && facts.max_member_rows == 3);
  require(hidden == std::array<std::int32_t, 6>{0, 1, 2, 3, 3, 3});
  require(history == std::array<std::int32_t, 12>{1, 4, 4, 5, 5, 6, 3, 7, 3, 7, 3, 7});
  (void)hyle::inference::pack_sequence_prefixes(short_first, 3, 0, hidden, {});
  require(hidden == std::array<std::int32_t, 6>{0, 0, 0, 1, 2, 3});

  const auto old_hidden = hidden;
  const auto old_history = history;
  for (const auto bad : {std::array<std::int32_t, 3>{0, 1, 5},
                         std::array<std::int32_t, 3>{0, 1, 1},
                         std::array<std::int32_t, 3>{0, 1, -1}}) {
    bool rejected = false;
    try { (void)hyle::inference::pack_sequence_prefixes(bad, 3, 2, hidden, history); }
    catch (const std::invalid_argument &) { rejected = true; }
    require(rejected && hidden == old_hidden && history == old_history);
  }
  bool rejected = false;
  try {
    (void)hyle::inference::pack_sequence_prefixes(short_first, 3, 2,
        std::span(history).first(6), history);
  } catch (const std::invalid_argument &) { rejected = true; }
  require(rejected && hidden == old_hidden && history == old_history);
  std::array<std::int32_t, 3> aliased_offsets{0, 1, 2};
  const auto old_offsets = aliased_offsets;
  rejected = false;
  try {
    (void)hyle::inference::pack_sequence_prefixes(aliased_offsets, 1, 0,
        std::span(aliased_offsets).first(2), {});
  } catch (const std::invalid_argument &) { rejected = true; }
  require(rejected && aliased_offsets == old_offsets);
  rejected = false;
  try {
    (void)hyle::inference::pack_sequence_prefixes(short_first,
        static_cast<std::size_t>((std::numeric_limits<std::int32_t>::max)()), 0, hidden, {});
  } catch (const std::out_of_range &) { rejected = true; }
  require(rejected && hidden == old_hidden && history == old_history);
  std::cout << "{\"status\":\"PASS\",\"case\":\"packed-prefix-rows\","
      "\"allocation_free_selector\":true,\"manual_orders\":2,"
      "\"zero_history\":true,\"late_failure_preserves_outputs\":true,"
      "\"overlap_preserves_outputs\":true,\"models\":0}" << '\n';
}

} // namespace inference_packed_prefix_rows_case
