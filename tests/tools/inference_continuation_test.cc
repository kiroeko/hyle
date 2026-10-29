// Pure host continuation control. These tests do not execute or retire a device.
#include "../../engine/inference/continuation.h"

#include <array>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
namespace i = hyle::inference;
void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }

void continuation_cohort_preserves_pending_identity_and_drain_intent() {
  using Cohort = i::ContinuationCohort;
  i::RequestWindow window, foreign; window.prepare(2, 2); foreign.prepare(2, 2);
  const auto a = window.enqueue(0), b = window.enqueue(1);
  const auto foreign_a = foreign.enqueue(0);
  const std::array original{i::ContinuationMember{a, 3}, i::ContinuationMember{b, 7}};
  const auto identity = [](const i::ContinuationMember &member) { return member; };
  Cohort cohort; cohort.prepare(2);
  require(cohort.select(original, identity, false) == Cohort::Selection::kChanged &&
      cohort.select(original, identity, true) == Cohort::Selection::kSame,
      "original continuation cohort could not be selected and resumed");
  const std::array reordered{original[1], original[0]};
  auto other_owner = original; other_owner[0].key = foreign_a;
  auto next_turn = original; ++next_turn[1].turn;
  require(foreign_a.slot == a.slot && foreign_a.generation == a.generation && foreign_a != a,
      "foreign continuation case did not isolate RequestWindow ownership");
  require(window.cancel_waiting(a), "old continuation request could not leave its actual waiting window");
  window.release(a); const auto next_a = window.enqueue(0);
  auto next_generation = original; next_generation[0].key = next_a;
  require(next_a.slot == a.slot && next_a.generation != a.generation,
      "continuation generation case did not reuse the original logical slot");
  for (const auto &incoming : {reordered, other_owner, next_generation, next_turn}) {
    require(!cohort.matches(incoming, identity) &&
        cohort.select(incoming, identity, true) == Cohort::Selection::kPendingConflict &&
        cohort.matches(original, identity) && !cohort.drain_requested(),
        "pending conflict changed the original cohort, accepted a stale identity or invented drain intent");
  }
  const std::array<i::RequestCreditGrant, 2> grants{{{1, 1, false}, {1, 2, false}}};
  auto cancelled = grants; cancelled[1].cancelled = true;
  require(!cohort.draining(false, grants) && cohort.draining(true, grants) &&
      cohort.draining(false, cancelled) &&
      cohort.admission(original[0], true) == Cohort::Release::kKeep &&
      cohort.release(foreign_a, true) == Cohort::Release::kKeep,
      "unrelated controls drained an unchanged continuation or explicit drain/cancel was ignored");
  require(cohort.admission(next_turn[1], true) == Cohort::Release::kDrain &&
      cohort.drain_requested() && cohort.matches(original, identity),
      "admitting another turn erased a still-pending cohort instead of requesting drain");
  cohort.reset_ready();
  require(cohort.members().empty() && cohort.drain_requested() && cohort.draining(false, grants),
      "releasing ready hints erased drain intent for caller-owned pending work");
  // Acknowledging this control intent is not native retirement. Pending calls,
  // Head/Source and their actual retirement remain with the original owner.
  cohort.clear_drain();
  require(!cohort.drain_requested() &&
      cohort.select(original, identity, false) == Cohort::Selection::kChanged &&
      cohort.release(a, true) == Cohort::Release::kDrain && cohort.matches(original, identity),
      "member release failed to preserve its pending cohort and request drain");
  cohort.clear_drain();
  require(!cohort.drain_requested() && cohort.matches(original, identity) &&
      cohort.admission(next_turn[1], false) == Cohort::Release::kReady &&
      cohort.release(a, false) == Cohort::Release::kReady,
      "idle controls could not release ready hints or acknowledging intent changed member identity");
  require(cohort.select(next_generation, identity, false) == Cohort::Selection::kChanged &&
      cohort.matches(next_generation, identity) && !cohort.matches(original, identity),
      "an idle continuation reused the preceding generation's cohort");
}

void continuation_quota_bounds_original_grants_without_overflow() {
  const i::RequestCreditGrant ordinary{1, 3, false}, k1{2, 4, false};
  static_assert(noexcept(i::continuation_quota(ordinary, 1, 7)));
  const auto ordinary_quota = i::continuation_quota(ordinary, 1, 7);
  const auto k1_quota = i::continuation_quota(k1, 2, 8);
  const auto held_only = i::continuation_quota({1, 0, false}, 1, 1);
  require(ordinary_quota && *ordinary_quota == 4 && k1_quota && *k1_quota == 6 &&
      held_only && *held_only == 1 && ordinary.reserved == 1 && ordinary.additional == 3 &&
      k1.reserved == 2 && k1.additional == 4,
      "pending renewal changed its original grant or confused ordinary and K1 output widths");
  const auto maximum = (std::numeric_limits<std::size_t>::max)();
  const auto largest_ordinary = i::continuation_quota({1, maximum - 1, false}, 1, maximum);
  const auto largest_k1 = i::continuation_quota({2, maximum - 2, false}, 2, maximum);
  require(largest_ordinary && *largest_ordinary == maximum && largest_k1 && *largest_k1 == maximum,
      "bounded pending grants rejected their exact SIZE_MAX sum");
  require(!i::continuation_quota({1, maximum, false}, 1, maximum) &&
      !i::continuation_quota({2, maximum - 1, false}, 2, maximum) &&
      !i::continuation_quota(ordinary, 2, 7) && !i::continuation_quota(k1, 2, 1) &&
      !i::continuation_quota(k1, 2, 5) && !i::continuation_quota({0, 0, false}, 0, 8) &&
      !i::continuation_quota({1, 3, true}, 1, 7),
      "overflow, unbacked output width, exhausted turn or cancelled grant produced a pending quota");
}

void continuation_retention_keeps_or_retires_the_whole_actual_successor() {
  using Retention = i::ContinuationRetention;
  const auto progress = [](const i::ContinuationProgress &member) { return member; };
  const std::array published{i::ContinuationProgress{true, false, false, false, 5},
      i::ContinuationProgress{true, false, false, false, 3}};
  require(i::continuation_retention(published, progress, true, false, false, false) == Retention::kKeep &&
      i::continuation_retention(published, progress, true, false, false, true) == Retention::kKeep,
      "pure drain cancelled an already accepted successor with fully published predecessors");
  auto stopped = published; stopped[0].stopping = true;
  auto unpublished = published; unpublished[1].published = false;
  auto completed = published; completed[0].completed = true; completed[0].remaining_outputs = 0;
  auto failed = published; failed[1].failed = true;
  for (const auto &partial : {stopped, unpublished, completed, failed}) {
    require(i::continuation_retention(partial, progress, true, false, false, false) == Retention::kRetirePending &&
        i::continuation_retention(partial, progress, false, false, false, false) == Retention::kResetReady,
        "partial predecessor outcome kept part of its accepted successor or reusable ready cohort");
  }
  auto no_outputs = published; no_outputs[1].remaining_outputs = 0;
  require(i::continuation_retention(no_outputs, progress, true, false, false, false) == Retention::kRetirePending &&
      i::continuation_retention(published, progress, true, true, false, false) == Retention::kRetirePending &&
      i::continuation_retention(published, progress, true, false, true, false) == Retention::kRetirePending,
      "exhausted outputs, native cancellation or unavailable continuation retained accepted B");
  require(i::continuation_retention(published, progress, false, false, false, false) == Retention::kKeep &&
      i::continuation_retention(published, progress, false, false, false, true) == Retention::kReleaseReady &&
      published[0].published && published[1].published && published[0].remaining_outputs == 5 &&
      published[1].remaining_outputs == 3,
      "idle drain retained ready hints or retention decisions mutated committed predecessor facts");

  // Project real observed host progress. A projection can fail independently
  // of the native owner; shared policy must preserve that original exception.
  std::array requests{i::RequestProgress{6}, i::RequestProgress{4}};
  const std::array<std::int32_t, 1> emitted{17};
  for (auto &request : requests) {
    (void)request.begin_turn();
    require(request.observe_commit(0, emitted, true, true, false),
        "progress projection case did not establish its original observed commit");
  }
  struct ProjectionFailure final {const i::RequestProgress *owner; unsigned marker;};
  std::size_t projected = 0; bool caught = false;
  try {
    (void)i::continuation_retention(requests, [&](const i::RequestProgress &request) {
      ++projected;
      const auto remaining = request.remaining_outputs();
      if (&request == &requests[1]) throw ProjectionFailure{&request, 0x71ab};
      return i::ContinuationProgress{true, request.turn_done, false, false, remaining};
    }, true, false, false, false);
  } catch (const ProjectionFailure &failure) {
    caught = failure.owner == &requests[1] && failure.marker == 0x71ab;
  }
  require(caught && projected == 2 && requests[0].observed_commits() == 1 &&
      requests[1].observed_commits() == 1 && requests[0].generated == std::vector<std::int32_t>{17} &&
      requests[1].generated == std::vector<std::int32_t>{17},
      "continuation policy replaced a progress projection exception or changed prior committed outputs");
}
} // namespace

int main() {
  try {
    continuation_cohort_preserves_pending_identity_and_drain_intent();
    continuation_quota_bounds_original_grants_without_overflow();
    continuation_retention_keeps_or_retires_the_whole_actual_successor();
    std::puts("{\"status\":\"PASS\",\"groups\":3,\"device_execution\":false}");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "inference continuation: %s\n", error.what());
    return 1;
  } catch (...) {
    std::fputs("inference continuation: unexpected exception\n", stderr);
    return 1;
  }
}
