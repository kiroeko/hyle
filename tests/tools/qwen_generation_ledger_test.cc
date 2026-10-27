// Host ledger transitions only: no model, SDK, native work or fake device ID.
#include "../../engine/models/qwen35/generation_ledger.h"

#include <array>
#include <cstdio>
#include <stdexcept>
#include <type_traits>

namespace {
namespace q = qwen35;

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
template<class Operation> void rejects(Operation operation) {
  try { operation(); }
  catch (const std::runtime_error &) { return; }
  throw std::runtime_error("invalid ledger transition was accepted");
}
using Capacities = std::array<std::size_t, 3>;
Capacities capacities(const q::GenerationCommitRecord &record) {
  return {record.emitted.capacity(), record.accepted_inputs.capacity(), record.verification_inputs.capacity()};
}

void prepare(q::GenerationCommitRecord &record, std::size_t quota = 3, std::size_t turn = 2,
             std::size_t requests = 2, std::size_t member = 1) {
  record.role = q::GraphRole::kText; record.generation = true;
  record.pending_input = 17; record.quota = quota; record.turn = turn;
  record.requests = requests; record.member_index = member;
  record.before = {.length = 9, .next_id = 17, .sampling = {123, 7, 20}};
}

// Stand in only for the original owner's already checked, noexcept head/ledger
// publication. No RequestProgress, model state, RNG or output is advanced here.
q::GenerationBoundary commit(q::GenerationLedger &ledger, std::int32_t token) {
  auto &record = ledger.next();
  record.accepted = 1; record.verified_rows = 1;
  record.accepted_inputs.resize(2, -1);
  record.accepted_inputs[0] = record.pending_input; record.accepted_inputs[1] = -1;
  record.after = record.before; ++record.after.length;
  record.after.next_id = token; record.after.sampling = record.before.sampling.advanced(1);
  record.target_first = token; record.emitted.resize(2, -1);
  record.emitted[0] = token; record.emitted[1] = -1; record.emitted_count = 1;
  record.committed = true; ++ledger.committed_count;
  return record.after;
}

void reserve_then_bind_only_actual_observed_input() {
  q::GenerationLedger ledger(4, 8);
  prepare(ledger.next());
  const auto a_before = ledger.records[0];
  const auto *storage = ledger.records.data();
  const auto b_capacity = capacities(ledger.records[1]);
  auto &b = ledger.reserve_successor(0, 2, 2, 2, 1);
  require(ledger.successor_ordinal() == 1 && !ledger.committed_count &&
      &ledger.next() == &ledger.records[0] && b.input_deferred && b.ids().empty() &&
      b.committed_ids().empty() && b.pending_input == -1 && b.before == q::GenerationBoundary{} &&
      b.quota == 2 && b.turn == 2 && b.requests == 2 && b.member_index == 1 &&
      ledger.records[0].same_result(a_before) && capacities(b) == b_capacity,
      "reservation fabricated an input/base, advanced the prefix or changed cold storage");
  rejects([&] { (void)ledger.reserve_successor(0, 2, 2, 2, 1); });
  rejects([&] { (void)ledger.bind_successor({}, 0, 2, 23); });
  require(b.input_deferred && b.ids().empty() && !ledger.committed_count,
      "early bind changed the still unknown successor input");

  const auto a_after = commit(ledger, 23);
  const auto committed_a = ledger.records[0];
  rejects([&] { (void)ledger.next(); });
  rejects([&] { (void)ledger.bind_successor(a_after, 0, 2, 23); });
  rejects([&] { (void)ledger.bind_successor(a_after, 2, 2, 23); });
  rejects([&] { (void)ledger.bind_successor(a_after, 1, 3, 23); });
  rejects([&] { (void)ledger.bind_successor(a_after, 1, 2, 24); });
  auto wrong = a_after; ++wrong.sampling.draw;
  rejects([&] { (void)ledger.bind_successor(wrong, 1, 2, 23); });
  ledger.records[0].stopped = true;
  rejects([&] { (void)ledger.bind_successor(a_after, 1, 2, 23); });
  ledger.records[0].stopped = false;
  ledger.records[0].member_index = 0;
  rejects([&] { (void)ledger.bind_successor(a_after, 1, 2, 23); });
  ledger.records[0].member_index = 1;
  require(ledger.successor_ordinal() == 1 && b.input_deferred && b.ids().empty() &&
      b.before == q::GenerationBoundary{} && ledger.committed_count == 1 &&
      ledger.records[0].same_result(committed_a),
      "rejected binding changed the committed predecessor or its deferred input");
  require(&ledger.bind_successor(a_after, 1, 2, 23) == &b && !b.input_deferred &&
      b.ids().size() == 1 && b.ids()[0] == 23 && b.before == a_after &&
      !ledger.successor_ordinal() && ledger.committed_count == 1 && &ledger.next() == &b,
      "binding did not preserve the prefix while installing its exact observed input");

  // After A's observation, B is the current known-input record. Reserve C
  // while B remains pending; do not require B's future output on the host.
  auto &c = ledger.reserve_successor(1, 1, 2, 2, 1);
  require(ledger.successor_ordinal() == 2 && c.input_deferred && c.ids().empty() &&
      ledger.committed_count == 1 && b.ids()[0] == 23,
      "continued reservation duplicated or prematurely observed B");
  const auto b_after = commit(ledger, 31);
  require(&ledger.bind_successor(b_after, 2, 2, 31) == &c &&
      c.before == b_after && c.ids()[0] == 31 && ledger.committed_count == 2 &&
      ledger.records[0].same_result(committed_a) && ledger.records.data() == storage &&
      capacities(ledger.records[1]) == b_capacity,
      "continuous reservation changed an old commit or reallocated the ledger");
}

void cancellation_preserves_capacity_and_committed_prefix() {
  q::GenerationLedger ledger(3, 16);
  prepare(ledger.next());
  const auto a_before = ledger.records[0];
  const auto a_capacity = capacities(ledger.records[0]);
  const auto b_capacity = capacities(ledger.records[1]);
  (void)ledger.reserve_successor(0, 2, 2, 2, 1);
  ledger.discard_successor(); ledger.discard_successor();
  require(!ledger.successor_ordinal() && !ledger.committed_count &&
      ledger.records[0].same_result(a_before) && !ledger.records[1].input_deferred &&
      ledger.records[1].emitted.empty() && capacities(ledger.records[1]) == b_capacity,
      "discarding the successor changed A or released its cold record capacity");
  (void)ledger.reserve_successor(0, 2, 2, 2, 1);
  ledger.discard_uncommitted();
  require(!ledger.successor_ordinal() && !ledger.committed_count &&
      ledger.records[0].emitted.empty() && ledger.records[1].emitted.empty() &&
      capacities(ledger.records[0]) == a_capacity && capacities(ledger.records[1]) == b_capacity,
      "cancelling A retained its unbound successor or discarded prepared capacity");

  prepare(ledger.next());
  (void)ledger.reserve_successor(0, 2, 2, 2, 1);
  (void)commit(ledger, 23);
  const auto committed_a = ledger.records[0];
  ledger.discard_uncommitted(); // Current B is still unknown; next() rejects it.
  require(ledger.committed_count == 1 && ledger.records[0].committed &&
      ledger.records[0].same_result(committed_a) && !ledger.successor_ordinal() &&
      !ledger.records[1].input_deferred && &ledger.next() == &ledger.records[1] &&
      capacities(ledger.records[1]) == b_capacity,
      "cancelling unbound B erased A's commit or made its ordinary record unusable");
}

void reservation_rejects_without_mutation() {
  const auto invalid = [](auto mutate, auto reserve) {
    q::GenerationLedger ledger(3, 8); prepare(ledger.next()); mutate(ledger);
    const auto records = ledger.records;
    const std::array saved_capacities{capacities(ledger.records[0]),
        capacities(ledger.records[1]), capacities(ledger.records[2])};
    const auto committed = ledger.committed_count;
    rejects([&] { reserve(ledger); });
    require(!ledger.successor_ordinal() && ledger.committed_count == committed &&
        ledger.records.size() == records.size(), "rejected reservation changed ledger counters");
    for (std::size_t i = 0; i < records.size(); ++i)
      require(ledger.records[i].same_result(records[i]) &&
          ledger.records[i].committed == records[i].committed &&
          ledger.records[i].pending_input == records[i].pending_input &&
          capacities(ledger.records[i]) == saved_capacities[i],
          "rejected reservation partially changed a prepared record");
  };
  const auto reserve = [](auto &ledger) { (void)ledger.reserve_successor(0, 2, 2, 2, 1); };
  invalid([](auto &ledger) { ledger.records[0].role = q::GraphRole::kMtpRound; }, reserve);
  invalid([](auto &ledger) { ledger.records[0].generation = false; }, reserve);
  invalid([](auto &ledger) { ledger.records[0].completion = q::TextCompletion::kStateOnly; }, reserve);
  invalid([](auto &ledger) { ledger.records[0].pending_input = -1; }, reserve);
  invalid([](auto &ledger) { ledger.records[1].committed = true; }, reserve);
  invalid([](auto &) {}, [](auto &ledger) { (void)ledger.reserve_successor(1, 2, 2, 2, 1); });
  invalid([](auto &) {}, [](auto &ledger) { (void)ledger.reserve_successor(0, 0, 2, 2, 1); });
  invalid([](auto &) {}, [](auto &ledger) { (void)ledger.reserve_successor(0, 3, 2, 2, 1); });
  invalid([](auto &) {}, [](auto &ledger) { (void)ledger.reserve_successor(0, 2, 3, 2, 1); });
  invalid([](auto &) {}, [](auto &ledger) { (void)ledger.reserve_successor(0, 2, 2, 0, 0); });
  invalid([](auto &) {}, [](auto &ledger) { (void)ledger.reserve_successor(0, 2, 2, 2, 2); });
  for (const auto size : {std::size_t{0}, std::size_t{1}}) {
    q::GenerationLedger ledger(size, 8);
    if (size) prepare(ledger.next());
    rejects([&] { (void)ledger.reserve_successor(0, 2, 2); });
    require(!ledger.committed_count && !ledger.successor_ordinal(), "capacity rejection advanced the prefix");
  }
}

void ordinary_and_mtp_records_keep_their_existing_contract() {
  static_assert(std::is_same_v<decltype(q::GenerationCommitRecord{}.endpoint),
      std::chrono::steady_clock::time_point>);
  q::GenerationLedger ledger(1, 16);
  auto &record = ledger.next();
  record.generation = true; record.pending_input = 7; record.accepted = 2;
  record.accepted_inputs.assign({7, 8}); record.verification_inputs.assign({7, 8, 9});
  const auto before = record;
  require(!record.input_deferred && record.ids().size() == 1 && record.ids()[0] == 7 &&
      record.committed_ids().size() == 2 && record.committed_ids()[1] == 8 && record.same_result(before),
      "ordinary/MTP input or accepted-prefix behavior changed");
  const auto capacity = capacities(record);
  ledger.discard_uncommitted();
  require(!ledger.committed_count && !ledger.successor_ordinal() && capacities(record) == capacity &&
      record.emitted.empty() && record.accepted_inputs.empty() && record.verification_inputs.empty(),
      "ordinary/MTP discard no longer preserves its original prepared storage");
}
} // namespace

int main() {
  try {
    reserve_then_bind_only_actual_observed_input();
    cancellation_preserves_capacity_and_committed_prefix();
    reservation_rejects_without_mutation();
    ordinary_and_mtp_records_keep_their_existing_contract();
    std::puts("{\"status\":\"PASS\",\"groups\":4,\"device_execution\":false}");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what()); return 1;
  }
}
