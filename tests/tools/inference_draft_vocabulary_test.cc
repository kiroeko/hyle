// Cold host metadata only; no model, Graph, device or tokenizer is executed.
#include "../../engine/inference/draft_vocabulary.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {
namespace i = hyle::inference;
void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }

static_assert(noexcept(i::DraftVocabulary::create(8, std::span<const std::int32_t>{})));
static_assert(noexcept(std::declval<const i::DraftVocabulary &>().full_size()));
static_assert(noexcept(std::declval<const i::DraftVocabulary &>().size()));
static_assert(noexcept(std::declval<const i::DraftVocabulary &>().ids()));
static_assert(std::is_same_v<decltype(std::declval<const i::DraftVocabulary &>().ids()),
    std::span<const std::int32_t>>);

void subset_owns_canonical_target_ids() {
  auto selected = [] {
    std::array<std::int32_t, 4> ranked{7, 2, 5, 0};
    constexpr std::array<std::int32_t, 4> original{7, 2, 5, 0};
    auto result = i::DraftVocabulary::create(10, ranked);
    require(result && ranked == original,
        "creating a draft vocabulary modified the caller's frequency-ranked IDs");
    ranked.fill(-1);
    return result;
  }();
  constexpr std::array<std::int32_t, 4> expected{0, 2, 5, 7};
  require(selected && selected->full_size() == 10 && selected->size() == expected.size() &&
      std::ranges::equal(selected->ids(), expected),
      "draft IDs lost their original full vocabulary, owned storage or ascending target order");
  const auto copy = *selected;
  require(copy.ids().data() != selected->ids().data() && std::ranges::equal(copy.ids(), expected),
      "copying a cold vocabulary borrowed its peer's ID storage");
}

void full_permutation_and_i32_boundaries_are_accepted() {
  constexpr std::array<std::int32_t, 8> permutation{7, 3, 6, 0, 5, 1, 4, 2};
  constexpr std::array<std::int32_t, 8> expected{0, 1, 2, 3, 4, 5, 6, 7};
  const auto full = i::DraftVocabulary::create(8, permutation);
  require(full && full->size() == 8 && std::ranges::equal(full->ids(), expected),
      "a complete target-vocabulary permutation was rejected or kept permuted row order");
  // Equal maxima must encounter target ID 1 before target ID 4, regardless of
  // the input permutation. This is the row-order contract, not a device test.
  constexpr std::array<float, 8> logits{0, 9, -4, 2, 9, 1, 3, -8};
  const auto first = std::max_element(full->ids().begin(), full->ids().end(),
      [&](auto a, auto b) { return logits[a] < logits[b]; });
  require(first != full->ids().end() && *first == 1,
      "canonical selected-row order changed the original target-ID tie winner");
  constexpr auto maximum = (std::numeric_limits<std::int32_t>::max)();
  constexpr std::array<std::int32_t, 3> boundary{maximum - 1, 17, 0};
  constexpr std::array<std::int32_t, 3> ordered{0, 17, maximum - 1};
  const auto sparse = i::DraftVocabulary::create(maximum, boundary);
  const auto singleton = i::DraftVocabulary::create(1, std::array<std::int32_t, 1>{0});
  require(sparse && sparse->full_size() == maximum && std::ranges::equal(sparse->ids(), ordered) &&
      singleton && singleton->size() == 1 && singleton->ids()[0] == 0,
      "valid sparse I32-boundary or singleton vocabulary required full-size allocation");
}

void invalid_vocabulary_and_token_ids_keep_precise_errors() {
  constexpr std::array<std::int32_t, 1> zero{0};
  constexpr auto maximum = (std::numeric_limits<std::int32_t>::max)();
  for (const std::int64_t full : {std::int64_t{0}, std::int64_t{-1},
      static_cast<std::int64_t>(maximum) + 1, (std::numeric_limits<std::int64_t>::max)()}) {
    const auto result = i::DraftVocabulary::create(full, zero);
    require(!result && result.error().IsInvalidArgument(),
        "invalid full vocabulary was accepted or lost InvalidArgument");
  }
  const auto empty = i::DraftVocabulary::create(8, {});
  require(!empty && empty.error().IsInvalidArgument(),
      "an empty draft vocabulary was accepted or lost InvalidArgument");
  for (const auto &ids : {std::array<std::int32_t, 3>{0, -1, 2},
      std::array<std::int32_t, 3>{0, 8, 2},
      std::array<std::int32_t, 3>{0, (std::numeric_limits<std::int32_t>::min)(), 2},
      std::array<std::int32_t, 3>{0, maximum, 2}}) {
    const auto result = i::DraftVocabulary::create(8, ids);
    require(!result && result.error().IsOutOfRange(),
        "out-of-range target token was accepted or lost OutOfRange");
  }
  constexpr std::array<std::int32_t, 4> repeated{4, 2, 1, 4};
  constexpr std::array<std::int32_t, 2> repeated_zero{0, 0};
  const auto duplicate = i::DraftVocabulary::create(8, repeated);
  const auto singleton_duplicate = i::DraftVocabulary::create(1, repeated_zero);
  require(!duplicate && duplicate.error().IsInvalidArgument() &&
      !singleton_duplicate && singleton_duplicate.error().IsInvalidArgument() &&
      repeated == std::array<std::int32_t, 4>{4, 2, 1, 4},
      "repeated target IDs were silently deduplicated or changed the caller's input");
  const auto retry = i::DraftVocabulary::create(8, std::array<std::int32_t, 2>{4, 1});
  require(retry && std::ranges::equal(retry->ids(), std::array<std::int32_t, 2>{1, 4}),
      "an earlier invalid cold mapping prevented a valid independent construction");
}
} // namespace

int main() {
  try {
    subset_owns_canonical_target_ids();
    full_permutation_and_i32_boundaries_are_accepted();
    invalid_vocabulary_and_token_ids_keep_precise_errors();
    std::puts("{\"status\":\"PASS\",\"groups\":3,\"device_execution\":false}");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "inference draft vocabulary: %s\n", error.what());
    return 1;
  } catch (...) {
    std::fputs("inference draft vocabulary: unexpected exception\n", stderr);
    return 1;
  }
}
