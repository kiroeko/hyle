// Host-only output credit and actual-prefix acknowledgement, no model/device.
#include "../../engine/inference/output_delivery.h"
#include "../../engine/inference/request_progress.h"

#include <array>
#include <cstdio>
#include <stdexcept>

namespace {
using namespace hyle::inference;
void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejects(F action) {
  bool rejected = false;
  try { action(); } catch (const std::exception &) { rejected = true; }
  require(rejected, "invalid output delivery operation was accepted");
}

void partial_delivery_does_not_recompute_or_complete_a_turn() {
  RequestProgress progress(5);
  OutputDelivery delivery(2);
  const std::array turns{GenerationTurn{{1}, 3}, GenerationTurn{{}, 2}};
  progress.bind_turns(turns);
  std::array<std::int32_t, 1> input{1}; progress.begin_turn(input);
  const std::array<std::int32_t, 2> first{7, 8};
  require(delivery.can_accept(progress.generated.size(), 2), "initial MTP prefix lacks reserved credit");
  progress.observe_commit(0, first, true, true, false);
  require(!delivery.can_accept(progress.generated.size(), 1) && !progress.turn_done,
      "backpressure completed the turn or allowed another episode");
  const auto blocked = delivery.pending(progress.generated);
  require(blocked.size() == 2 && blocked[0] == 7 && blocked[1] == 8,
      "delivery does not borrow the committed prefix");
  require(!progress.observe_commit(0, first, true, true, false), "late observation duplicated the model commit");
  require(delivery.acknowledged() == 0 && progress.observed_commits() == 1,
      "model observation incorrectly acknowledged transport output");
  delivery.acknowledge_prefix(progress.generated.size(), 1);
  delivery.acknowledge_prefix(progress.generated.size(), 1); // Same actual write observed twice.
  delivery.acknowledge_prefix(progress.generated.size(), 0); // Delayed older acknowledgement.
  require(delivery.available(progress.generated.size()) == 1 &&
      !delivery.can_accept(progress.generated.size(), 2) && delivery.can_accept(progress.generated.size(), 1),
      "partial acknowledgement failed to expose exactly one output credit");
  const std::array<std::int32_t, 1> third{9};
  progress.observe_commit(1, third, true, true, false);
  require(progress.turn_done && delivery.pending(progress.generated)[0] == 8,
      "new model output replaced the old undelivered prefix");
  progress.advance_turn();
  input[0] = -1; progress.begin_turn(input, 9);
  require(input[0] == 9 && !delivery.can_accept(progress.generated.size(), 1),
      "continuation reset the delivery cursor or consumed a different pending");
  delivery.acknowledge_prefix(progress.generated.size(), 3);
  const std::array<std::int32_t, 2> last{10, 11};
  require(delivery.can_accept(progress.generated.size(), 2), "draining did not wake credit");
  progress.observe_commit(2, last, true, true, false);
  require(progress.request_done() && delivery.pending(progress.generated).size() == 2 &&
      delivery.pending(progress.generated)[0] == 10,
      "model completion discarded final undelivered output");
  delivery.acknowledge_prefix(progress.generated.size(), 5);
  require(delivery.pending(progress.generated).empty() && delivery.acknowledged() == 5 &&
      progress.generated.size() == 5 && progress.observed_commits() == 3,
      "delivery mutated the authoritative output record or commit count");
}

void rejected_acknowledgements_preserve_credit() {
  rejects([] { OutputDelivery invalid(0); });
  OutputDelivery delivery(1);
  const std::array<std::int32_t, 1> output{7};
  rejects([&] { delivery.acknowledge_prefix(1, 2); });
  require(delivery.acknowledged() == 0 && !delivery.can_accept(1, 1), "over-acknowledgement changed credit");
  delivery.acknowledge_prefix(1, 0);
  require(delivery.pending(output)[0] == 7, "blocked write lost its retry prefix");
  delivery.acknowledge_prefix(1, 1);
  delivery.acknowledge_prefix(1, 1);
  rejects([&] { delivery.acknowledge_prefix(1, 2); });
  rejects([&] { (void)delivery.available(0); });
  rejects([&] { (void)delivery.available(3); });
  require(delivery.acknowledged() == 1 && delivery.available(1) == 1,
      "rejected stale extent or excess unreserved output changed the cursor");
  require(!delivery.can_accept(1, 2), "one-token delivery capacity accepted a two-token episode");
}
} // namespace

int main() {
  try {
    partial_delivery_does_not_recompute_or_complete_a_turn();
    std::puts("PASS partial delivery, one-token fallback credit, duplicate commit and pending-once continuation");
    rejected_acknowledgements_preserve_credit();
    std::puts("PASS bounded credit, idempotent prefix acknowledgement and atomic invalid-extent rejection");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what()); return 1;
  }
}
