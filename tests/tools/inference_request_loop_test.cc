// Host owner contracts only. Retirement observations below are explicit test
// inputs, not evidence that any device completed or freed physical memory.
#include "../../engine/inference/request_loop.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <optional>
#include <type_traits>

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
void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
template<class Error, class Operation> void rejects(Operation operation) {
  try { operation(); } catch (const Error &) { return; }
  throw std::runtime_error("invalid request-loop transition was accepted");
}
static_assert(!std::is_copy_constructible_v<i::RequestLoop> && !std::is_move_constructible_v<i::RequestLoop>);

struct Adapter final {
  std::array<std::optional<i::RequestProgress>, 4> progress_values;
  std::array<bool, 4> residents{};
  std::array<bool, 4> stopped{};
  std::array<std::array<std::int32_t, 4>, 4> inputs{};
  std::array<std::size_t, 4> executions{}, publications{}, retirement_polls{}, eviction_polls{};
  std::array<std::size_t, 4> stops{};
  std::array<i::RequestKey, 32> admission_order{};
  std::array<std::size_t, 32> active_at_admission{};
  std::size_t admissions = 0, waves = 0, resident_queries = 0;
  std::size_t deferred_terminal = 0, deferred_eviction = 0, physical_bytes = 4096;
  std::optional<std::size_t> reject_preflight, throw_after_commit, retirement_error;
  std::optional<std::size_t> cancel_during_wave;
  i::RequestLoop *cancel_owner = nullptr;
  hyle::Status stop_result{};
  bool error_is_retired = false;
  hyle::Status error = hyle::Status::FromNative(hyle::StatusCode::Internal,
      {hyle::NativeErrorDomain::CudaDriver, 999}, "test native retirement failure");

  explicit Adapter(std::initializer_list<std::size_t> quotas) {
    std::size_t index = 0;
    for (auto quota : quotas) progress_values[index++].emplace(quota);
  }
  i::RequestProgress &progress(i::RequestKey key) { return *progress_values[key.slot]; }
  bool resident(i::RequestKey key) { ++resident_queries; return residents[key.slot]; }
  void admission_preflight(i::RequestKey key) {
    if (reject_preflight == key.slot) throw std::invalid_argument("test admission preflight");
  }
  void admit(i::RequestKey key, std::span<const i::RequestKey> active) {
    admission_order[admissions] = key; active_at_admission[admissions++] = active.size();
    auto &p = progress(key);
    if (!p.turn_started()) {
      if (p.turns().empty()) (void)p.begin_turn();
      else {
        auto &buffer = inputs[key.slot]; const auto &turn = p.turns()[p.turn_index];
        const auto pending = p.turn_index ? 1u : 0u;
        std::copy(turn.prompt.begin(), turn.prompt.end(), buffer.begin() + pending);
        (void)p.begin_turn(std::span(buffer).first(turn.prompt.size() + pending),
            pending ? std::optional{p.generated.back()} : std::nullopt);
      }
    }
    // Deliberately no head yet: the owner must charge this active reservation
    // when admitting another member before execute_wave creates real payloads.
  }
  void execute_wave(std::span<const i::RequestKey> active) {
    ++waves;
    if (cancel_during_wave) {
      const auto found = std::find_if(active.begin(), active.end(), [&](auto key) { return key.slot == *cancel_during_wave; });
      require(found != active.end() && cancel_owner, "missing original active cancellation target");
      const auto width = active.size();
      require(cancel_owner->cancel(*found, *this).ok() && cancel_owner->active().size() == width,
          "active cancellation invalidated its borrowed original wave");
      cancel_during_wave.reset();
    }
    for (auto key : active) {
      if (stopped[key.slot]) continue; // Explicit original model cancelled-member outcome.
      auto &p = progress(key); residents[key.slot] = true; ++executions[key.slot];
      const std::array output{static_cast<std::int32_t>(key.slot * 10 + p.generated.size() + 1)};
      require(p.observe_commit(p.observed_commits(), output, true, true, false), "host output was replayed");
      if (throw_after_commit == key.slot) throw std::runtime_error("test observer failed after commit");
    }
  }
  hyle::Status stop(i::RequestKey key) {
    ++stops[key.slot]; stopped[key.slot] = true; return stop_result;
  }
  void publish_turn(i::RequestKey key) {
    require(progress(key).turn_done, "owner published an unfinished turn");
    ++publications[key.slot];
  }
  i::RetirementObservation retire(i::RequestKey key, i::RequestRetirement reason) {
    auto &polls = reason == i::RequestRetirement::kEvicted ? eviction_polls[key.slot] : retirement_polls[key.slot];
    ++polls;
    if (retirement_error == key.slot) {
      if (error_is_retired) residents[key.slot] = false;
      return {error_is_retired, error};
    }
    auto &remaining = reason == i::RequestRetirement::kEvicted ? deferred_eviction : deferred_terminal;
    if (remaining) { --remaining; return {false, {}}; }
    residents[key.slot] = false;
    return {true, {}}; // Pool backing/externally retained versions remain owned elsewhere.
  }
};

void fifo_and_headless_reservations() {
  i::RequestLoop owner; owner.prepare(3, 2); Adapter adapter{2, 2, 1};
  const std::array keys{owner.enqueue(0), owner.enqueue(1), owner.enqueue(2)};
  auto result = owner.pump_once(adapter);
  require(result.admitted == 2 && result.executed && !result.completed && owner.active().size() == 2 &&
      adapter.active_at_admission[0] == 0 && adapter.active_at_admission[1] == 1 &&
      adapter.admission_order[0] == keys[0] && adapter.admission_order[1] == keys[1],
      "headless active reservations exceeded N or changed FIFO");
  const auto queries = adapter.resident_queries;
  result = owner.pump_once(adapter);
  require(result.completed == 2 && !result.admitted && adapter.resident_queries == queries &&
      owner.waiting_front() == keys[2], "steady decode scanned resident slots or skipped the waiting owner");
  result = owner.pump_once(adapter);
  require(result.admitted == 1 && result.completed == 1 && adapter.admission_order[2] == keys[2] &&
      owner.active().empty() && !owner.waiting_front(), "finite FIFO did not drain");

  i::RequestLoop turns; turns.prepare(2, 1); Adapter two{3, 3};
  const std::array schedule{i::GenerationTurn{{1}, 1}, i::GenerationTurn{{}, 1}, i::GenerationTurn{{}, 1}};
  for (auto &p : two.progress_values) if (p) p->bind_turns(schedule);
  const std::array order{turns.enqueue(0), turns.enqueue(1)};
  for (std::size_t step = 0; step < 6; ++step) (void)turns.pump_once(two);
  for (std::size_t step = 0; step < 6; ++step)
    require(two.admission_order[step] == order[step % 2], "completed turn did not rejoin the FIFO tail");
  require(two.publications[0] == 3 && two.publications[1] == 3 && two.executions[0] == 3 && two.executions[1] == 3,
      "known successor was skipped, advanced twice or republished");
}

void preflight_and_actual_eviction_facts() {
  i::RequestLoop owner; owner.prepare(2, 1); Adapter adapter{1, 1};
  const std::array first{i::GenerationTurn{{1}, 1}};
  adapter.progress_values[0]->bind_open_turns(first, 1, 1);
  const auto a = owner.enqueue(0); (void)owner.pump_once(adapter);
  require(owner.matches(a) && adapter.residents[0] && owner.active().empty(), "open owner was not paused");
  const auto b = owner.enqueue(1); adapter.reject_preflight = 1;
  rejects<std::invalid_argument>([&] { (void)owner.pump_once(adapter); });
  require(!adapter.eviction_polls[0] && !adapter.progress(b).turn_started() && adapter.residents[0],
      "invalid new request evicted an existing resident before preflight");
  adapter.reject_preflight.reset(); adapter.deferred_eviction = 1;
  const auto waiting = owner.pump_once(adapter);
  require(waiting.waiting_retirement && !waiting.executed && !waiting.admitted && adapter.residents[0] &&
      owner.waiting_front() == b && owner.matches(a), "unproven eviction granted admission or released identity");
  const auto done = owner.pump_once(adapter);
  require(done.completed == 1 && done.admitted == 1 && adapter.eviction_polls[0] == 2 &&
      owner.matches(a) && !adapter.residents[0] && adapter.physical_bytes == 4096,
      "eviction inferred physical release or destroyed the paused logical request");

  i::RequestLoop transfer; transfer.prepare(2, 1); Adapter cancelled{1, 1};
  cancelled.progress_values[0]->bind_open_turns(first, 1, 1);
  const auto victim = transfer.enqueue(0); (void)transfer.pump_once(cancelled);
  (void)transfer.enqueue(1); cancelled.deferred_eviction = 1;
  require(transfer.pump_once(cancelled).waiting_retirement, "fixture did not retain the original eviction");
  require(transfer.cancel(victim, cancelled).ok(), "pending eviction cancellation was not acknowledged");
  const auto transferred = transfer.pump_once(cancelled);
  require(transferred.completed == 2 && transferred.cancelled == 1 && cancelled.eviction_polls[0] == 1 &&
      cancelled.retirement_polls[0] == 1 && !transfer.matches(victim),
      "pending eviction was polled/reset twice or its cancelled terminal was lost");
}

void deferred_terminal_keeps_original_identity_without_execution() {
  i::RequestLoop owner; owner.prepare(2, 1); Adapter adapter{1, 1};
  const auto old = owner.enqueue(0); adapter.deferred_terminal = 2;
  const auto first = owner.pump_once(adapter);
  require(first.executed && first.waiting_retirement && !first.completed && owner.matches(old),
      "terminal owner was freed before actual retirement");
  const auto peer = owner.enqueue(1);
  const auto second = owner.pump_once(adapter);
  require(second.waiting_retirement && !second.executed && !second.admitted && adapter.executions[0] == 1 &&
      !adapter.executions[1] && adapter.publications[0] == 1 && owner.waiting_front() == peer,
      "pending native retirement admitted a peer or replayed a completed input");
  const auto third = owner.pump_once(adapter);
  require(third.completed == 2 && third.executed && !owner.matches(old) && !owner.matches(peer) &&
      adapter.executions[0] == 1, "terminal retry lost original retirement or peer execution");
  adapter.progress_values[0].emplace(1);
  const auto replacement = owner.enqueue(0);
  require(replacement.generation > old.generation && !owner.matches(old), "slot reuse revived an old key");
  rejects<std::out_of_range>([&] { owner.wake(old); });
  i::RequestLoop foreign; foreign.prepare(1, 1); const auto other = foreign.enqueue(0);
  require(!owner.matches(other) && !owner.acknowledge(other, 0, adapter) &&
      owner.cancel(other, adapter).IsNotFound() && !adapter.stops[0],
      "foreign ACK/cancel reached an owner with the same slot and generation");
}

void errors_preserve_proof_and_committed_boundaries() {
  i::RequestLoop owner; owner.prepare(1, 1); Adapter adapter{1};
  const auto key = owner.enqueue(0); adapter.retirement_error = 0;
  try { (void)owner.pump_once(adapter); throw std::logic_error("missing retirement error"); }
  catch (const i::RequestLoopFailure &failure) {
    require(failure.key() == key && failure.status() == adapter.error && !failure.retirement().retired &&
        failure.status().has_native_error() && owner.matches(key), "retirement error lost key/native status/proof");
  }
  adapter.retirement_error.reset();
  const auto result = owner.pump_once(adapter);
  require(result.completed == 1 && !result.executed && adapter.executions[0] == 1 && adapter.publications[0] == 1,
      "error recovery replayed a terminal model input or publication");
  adapter.progress_values[0].emplace(1); const auto retired_error = owner.enqueue(0);
  adapter.retirement_error = 0; adapter.error_is_retired = true;
  try { (void)owner.pump_once(adapter); throw std::logic_error("missing retired error"); }
  catch (const i::RequestLoopFailure &failure) {
    require(failure.key() == retired_error && failure.retirement().retired && !owner.matches(retired_error),
        "real retired error retained a dead slot or erased its error");
  }

  i::RequestLoop partial; partial.prepare(2, 2); Adapter observed{1, 1};
  const auto a = partial.enqueue(0), b = partial.enqueue(1); observed.throw_after_commit = 0;
  rejects<std::runtime_error>([&] { (void)partial.pump_once(observed); });
  require(observed.progress(a).turn_done && !observed.progress(b).turn_done,
      "host observer fixture did not preserve its actual partial commit");
  observed.throw_after_commit.reset(); const auto resumed = partial.pump_once(observed);
  require(resumed.completed == 2 && observed.executions[0] == 1 && observed.executions[1] == 1 &&
      observed.publications[0] == 1 && !partial.matches(a) && !partial.matches(b),
      "post-commit error replayed the completed turn or rolled back its survivor");

  i::RequestLoop queued; queued.prepare(2, 1); Adapter withdrawn{1, 1};
  const auto withdrawn_key = queued.enqueue(0); (void)queued.enqueue(1);
  require(queued.cancel(withdrawn_key, withdrawn).ok() && queued.cancelled(withdrawn_key),
      "queued cancellation did not retain its typed request intent");
  require(queued.cancel(withdrawn_key, withdrawn).ok(), "duplicate cancel was not idempotent");
  const auto withdrawn_result = queued.pump_once(withdrawn);
  const auto stop_count = withdrawn.stops[0];
  require(withdrawn_result.completed == 2 && withdrawn_result.cancelled == 1 &&
      !withdrawn.progress_values[0]->turn_started() && !withdrawn.executions[0] && !withdrawn.publications[0] &&
      queued.cancel(withdrawn_key, withdrawn).IsNotFound() && withdrawn.stops[0] == stop_count,
      "queued/stale cancellation executed work, published a turn or reached a replacement owner");

  i::RequestLoop shared; shared.prepare(2, 2); Adapter survivor{1, 1};
  const auto cancelled_key = shared.enqueue(0), survivor_key = shared.enqueue(1);
  survivor.cancel_owner = &shared; survivor.cancel_during_wave = 0;
  const auto shared_result = shared.pump_once(survivor);
  require(shared_result.completed == 2 && shared_result.cancelled == 1 && !survivor.executions[0] &&
      !survivor.publications[0] && survivor.executions[1] == 1 && survivor.publications[1] == 1 &&
      !shared.matches(cancelled_key) && !shared.matches(survivor_key),
      "active cancellation changed its surviving peer or published a cancelled turn");

  i::RequestLoop outside; outside.prepare(1, 1); Adapter idle_active{3};
  const auto idle_key = outside.enqueue(0); (void)outside.pump_once(idle_active);
  require(outside.cancel(idle_key, idle_active).ok(), "between-wave active cancellation failed");
  const auto idle_stopped = outside.pump_once(idle_active);
  require(idle_stopped.completed == 1 && idle_stopped.cancelled == 1 && !idle_stopped.executed &&
      idle_active.executions[0] == 1 && idle_active.progress_values[0]->generated.size() == 1 &&
      !idle_active.publications[0], "stopping idle-active request was prepared/executed again");

  i::RequestLoop expected; expected.prepare(1, 1); Adapter cancelled_status{1};
  const auto expected_key = expected.enqueue(0);
  cancelled_status.stop_result = hyle::Status(hyle::StatusCode::Cancelled);
  require(expected.cancel(expected_key, cancelled_status).IsCancelled(), "cancel did not return the original stop status");
  cancelled_status.retirement_error = 0; cancelled_status.error_is_retired = true;
  cancelled_status.error = hyle::Status(hyle::StatusCode::Cancelled);
  require(expected.pump_once(cancelled_status).cancelled == 1, "expected retired cancellation became a native failure");

  i::RequestLoop native; native.prepare(1, 1); Adapter native_status{1};
  const auto native_key = native.enqueue(0); require(native.cancel(native_key, native_status).ok(), "native case cancel failed");
  native_status.retirement_error = 0; native_status.error_is_retired = true;
  native_status.error = hyle::Status::FromNative(hyle::StatusCode::Cancelled,
      {hyle::NativeErrorDomain::CudaDriver, 999}, "test native cancelled error");
  try { (void)native.pump_once(native_status); throw std::logic_error("native cancellation error was hidden"); }
  catch (const i::RequestLoopFailure &failure) {
    require(failure.key() == native_key && failure.status() == native_status.error &&
        failure.retirement().retired && !native.matches(native_key), "native cancelled error lost its proof or identity");
  }
}

void open_input_and_midturn_pause_keep_progress() {
  i::RequestLoop owner; owner.prepare(1, 1); Adapter adapter{3};
  const std::array turns{i::GenerationTurn{{1}, 2}, i::GenerationTurn{{}, 1}};
  adapter.progress_values[0]->bind_open_turns(std::span(turns).first(1), 2, 1);
  i::OutputDelivery open_delivery(3);
  const auto key = owner.enqueue(0, &open_delivery); (void)owner.pump_once(adapter);
  const auto generated = adapter.progress(key).generated;
  const auto commits = adapter.progress(key).observed_commits();
  (void)owner.pause(key);
  require(owner.acknowledge(key, 1, adapter), "actual manual-pause output was not acknowledged");
  require(owner.pump_once(adapter).executed == false && adapter.progress(key).generated == generated &&
      adapter.progress(key).observed_commits() == commits && adapter.progress(key).turn_index == 0,
      "midturn pause changed observed outputs or turn position");
  owner.wake(key); (void)owner.pump_once(adapter);
  require(adapter.progress(key).waiting_for_input() && owner.matches(key) && owner.active().empty() &&
      adapter.publications[0] == 1, "open input was completed or its boundary was not paused");
  require(owner.acknowledge(key, 2, adapter) && !owner.waiting_front(), "ACK woke an input waiter without new input");
  adapter.progress(key).extend_turns(turns); owner.wake(key);
  require(adapter.progress(key).turn_index == 0, "wake advanced model-facing progress before admission");
  (void)owner.pump_once(adapter);
  require(adapter.progress(key).turn_index == 1 && adapter.progress(key).generated.size() == 3 &&
      adapter.publications[0] == 2 && owner.matches(key), "actual later input did not resume the same request");
  require(owner.acknowledge(key, 3, adapter), "actual final open output was not acknowledged");
  adapter.progress(key).close_input(); owner.wake(key);
  const auto closed = owner.pump_once(adapter);
  require(closed.completed == 1 && !closed.executed && !owner.matches(key) && adapter.executions[0] == 3 &&
      adapter.publications[0] == 2, "closing paused input restored/reexecuted or republished an old turn");

  i::RequestLoop credit; credit.prepare(1, 1); Adapter blocked{3}; i::OutputDelivery delivery(1);
  const auto blocked_key = credit.enqueue(0, &delivery);
  const auto blocked_first = credit.pump_once(blocked);
  require(blocked_first.waiting_output && !blocked_first.waiting_retirement && credit.active().empty() &&
      !blocked.progress(blocked_key).turn_done && credit.credit(blocked_key, 1) == 0,
      "zero credit completed a turn or was labelled native retirement");
  require(credit.acknowledge(blocked_key, 0, blocked) && !credit.waiting_front(), "zero ACK granted output credit");
  rejects<std::out_of_range>([&] { (void)credit.acknowledge(blocked_key, 2, blocked); });
  require(delivery.acknowledged() == 0 && credit.acknowledge(blocked_key, 1, blocked) &&
      credit.acknowledge(blocked_key, 1, blocked) && credit.waiting_front() == blocked_key,
      "invalid/duplicate ACK changed credit or same-key FIFO wake");
  (void)credit.pump_once(blocked);
  require(blocked.progress(blocked_key).generated.size() == 2 && !blocked.progress(blocked_key).turn_done,
      "credit wake consumed a pending input twice or completed the wrong quota");
  require(credit.cancel(blocked_key, blocked).ok(), "credit-paused cancellation failed");
  const auto cancelled_pending = credit.pump_once(blocked);
  require(cancelled_pending.waiting_output && !cancelled_pending.waiting_retirement && !cancelled_pending.completed &&
      credit.matches(blocked_key) && !blocked.residents[0] && blocked.retirement_polls[0] == 1,
      "native retirement discarded undelivered cancelled output");
  (void)credit.pump_once(blocked);
  require(blocked.retirement_polls[0] == 1 && blocked.executions[0] == 2 && !blocked.publications[0],
      "delivery wait repolled native retirement, replayed input or published a cancelled turn");
  require(credit.acknowledge(blocked_key, 2, blocked), "cancelled output could not drain through its original key");
  const auto drained_cancel = credit.pump_once(blocked);
  require(drained_cancel.completed == 1 && drained_cancel.cancelled == 1 && !credit.matches(blocked_key) &&
      blocked.progress_values[0]->generated.size() == 2, "final cancel ACK cleared output payload or failed logical release");

  i::RequestLoop final; final.prepare(2, 1); Adapter retained{1, 1}; i::OutputDelivery last_delivery(1);
  const auto final_key = final.enqueue(0, &last_delivery); (void)final.enqueue(1);
  const auto last_output = final.pump_once(retained);
  require(last_output.waiting_output && !last_output.waiting_retirement && !last_output.completed && final.matches(final_key),
      "unacknowledged final output lost its terminal key");
  const auto peer_done = final.pump_once(retained);
  require(peer_done.completed == 1 && peer_done.executed && peer_done.waiting_output && retained.retirement_polls[0] == 1,
      "native-retired output waiter blocked a ready peer or repolled native work");
  require(final.acknowledge(final_key, 1, retained), "final output ACK was rejected");
  const auto last_release = final.pump_once(retained);
  require(last_release.completed == 1 && !last_release.executed && !last_release.waiting_output &&
      retained.executions[0] == 1 && retained.publications[0] == 1 && retained.progress_values[0]->generated.size() == 1,
      "final ACK replayed/published old work or destroyed authoritative output");
  retained.progress_values[0].emplace(1); i::OutputDelivery replacement_delivery(1);
  const auto replacement = final.enqueue(0, &replacement_delivery);
  require(!final.acknowledge(final_key, 1, retained) && replacement_delivery.acknowledged() == 0 &&
      final.matches(replacement), "delayed ACK advanced a replacement request's cursor");
  rejects<std::out_of_range>([&] { (void)final.credit(final_key, 0); });

  i::RequestLoop malformed; malformed.prepare(1, 1); Adapter oversized{2}; i::OutputDelivery bounded(1);
  const auto bad_key = malformed.enqueue(0, &bounded); (void)malformed.pump_once(oversized);
  // Deliberate owner misuse: commit another user output without delivery credit.
  const std::array illegal_output{std::int32_t{99}};
  require(oversized.progress(bad_key).observe_commit(1, illegal_output, true, true, false), "invalid-credit fixture failed");
  require(malformed.cancel(bad_key, oversized).ok(), "invalid-credit fixture could not request stop");
  rejects<std::out_of_range>([&] { (void)malformed.pump_once(oversized); });
  require(!oversized.retirement_polls[0] && malformed.matches(bad_key) && !bounded.acknowledged(),
      "invalid terminal delivery mutated native retirement or acknowledged fabricated credit");
}

void prepared_hot_owner_allocates_nothing() {
  i::RequestLoop owner; owner.prepare(2, 2); Adapter adapter{3, 3};
  (void)owner.enqueue(0); (void)owner.enqueue(1);
  allocation_count = 0; track_allocations = true;
  for (std::size_t step = 0; step < 3; ++step) (void)owner.pump_once(adapter);
  track_allocations = false;
  require(!allocation_count && adapter.executions[0] == 3 && adapter.executions[1] == 3 &&
      owner.active().empty() && !owner.waiting_front(), "valid prepared owner loop allocated host storage");

  i::RequestLoop controls; controls.prepare(2, 1); Adapter controlled{2, 1}; i::OutputDelivery delivery(1);
  const auto key = controls.enqueue(0, &delivery);
  allocation_count = 0; track_allocations = true;
  (void)controls.pump_once(controlled);
  const bool acknowledged = controls.acknowledge(key, 1, controlled);
  const auto cancelled = controls.cancel(key, controlled);
  (void)controls.pump_once(controlled);
  (void)controls.enqueue(1); (void)controls.pump_once(controlled);
  track_allocations = false;
  require(acknowledged && cancelled.ok() && !allocation_count && controlled.executions[0] == 1 &&
      controlled.executions[1] == 1, "prepared delivery/cancel control allocated or replayed host work");
}
}

int main() {
  try {
    fifo_and_headless_reservations();
    preflight_and_actual_eviction_facts();
    deferred_terminal_keeps_original_identity_without_execution();
    errors_preserve_proof_and_committed_boundaries();
    open_input_and_midturn_pause_keep_progress();
    prepared_hot_owner_allocates_nothing();
    std::puts("request loop: 6 host contract groups passed; no model/device loaded");
    return 0;
  } catch (const std::exception &error) {
    track_allocations = false;
    std::fprintf(stderr, "request loop: %s\n", error.what()); return 1;
  }
}
