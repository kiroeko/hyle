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

struct Adapter {
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
  i::RequestWaveResult execute_wave(i::RequestWave wave, std::span<std::size_t> remaining_reserved) {
    require(!wave.observe_only && !wave.drain_requested && wave.keys.size() == remaining_reserved.size() &&
        wave.keys.size() == wave.credits.size(), "invalid synchronous wave contract");
    const auto active = wave.keys;
    std::fill(remaining_reserved.begin(), remaining_reserved.end(), 0);
    ++waves;
    try {
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
    } catch (...) { return {true, {true, {}}, std::current_exception()}; }
    return {true, {true, {}}, {}};
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

// Explicit host-scripted submission/observation facts. No completion object or
// device executor is duplicated here: each result is an input to the owner.
struct WaveStage final {
  std::array<std::size_t, 4> reserved{}, additional{}, remaining{}, outputs{};
  std::array<bool, 4> cancelled{}, eos{};
  bool advanced = false, retired = false, observe_only = false, drain_requested = false, request_yield = false;
  std::exception_ptr failure{};
};
struct AsyncAdapter final : Adapter {
  using Adapter::Adapter;
  std::array<WaveStage, 8> stages{};
  std::array<i::RequestKey, 4> original_keys{};
  std::array<int, 16> events{};
  const i::RequestKey *original_data = nullptr;
  std::size_t stage_count = 0, stage_at = 0, original_width = 0, event_count = 0;
  bool previous_retired = true;
  i::RequestWaveResult execute_wave(i::RequestWave wave, std::span<std::size_t> remaining_reserved) {
    require(stage_at < stage_count && wave.keys.size() == wave.credits.size() &&
        wave.keys.size() == remaining_reserved.size(), "missing or malformed scripted wave");
    const auto &stage = stages[stage_at++]; ++waves;
    if (previous_retired) {
      original_data = wave.keys.data(); original_width = wave.keys.size();
      std::copy(wave.keys.begin(), wave.keys.end(), original_keys.begin());
    }
    require(wave.observe_only == stage.observe_only && wave.drain_requested == stage.drain_requested &&
        wave.keys.data() == original_data &&
        wave.keys.size() == original_width && std::equal(wave.keys.begin(), wave.keys.end(), original_keys.begin()),
        "in-flight wave changed its original identities, order or borrow");
    if (stage.advanced) events[event_count++] = static_cast<int>(100 + stage_at);
    if (stage.request_yield) {
      require(cancel_owner, "missing scripted yield owner"); const auto width = wave.keys.size();
      cancel_owner->request_yield();
      require(cancel_owner->active().size() == width, "yield intent invalidated its executing wave");
    }
    for (std::size_t at = 0; at < wave.keys.size(); ++at) {
      const auto key = wave.keys[at]; const auto slot = key.slot; const auto &grant = wave.credits[at];
      require(grant.reserved == stage.reserved[slot] && grant.additional == stage.additional[slot] &&
          grant.cancelled == stage.cancelled[slot], "scripted wave received incorrect reserved/additional/cancel facts");
      residents[slot] = true;
      if (stage.outputs[slot]) {
        events[event_count++] = static_cast<int>(200 + stage_at);
        auto &p = progress(key); std::array<std::int32_t, 4> output{};
        for (std::size_t n = 0; n < stage.outputs[slot]; ++n)
          output[n] = static_cast<std::int32_t>(slot * 10 + p.generated.size() + n + 1);
        require(p.observe_commit(p.observed_commits(), std::span(output).first(stage.outputs[slot]),
            true, true, stage.eos[slot]), "scripted original output was observed twice");
        ++executions[slot];
      }
      remaining_reserved[at] = stage.remaining[slot];
    }
    previous_retired = stage.retired;
    return {stage.advanced, {stage.retired, {}}, stage.failure};
  }
};

struct WaveFailure final : std::runtime_error {
  static constexpr unsigned expected_payload = 0x41b2;
  unsigned payload = expected_payload;
  WaveFailure() : std::runtime_error("scripted failure after committed A") {}
};

void in_flight_waves_keep_exact_credit_and_allow_continuation() {
  i::RequestLoop owner; owner.prepare(1, 1); AsyncAdapter adapter{3}; i::OutputDelivery delivery(2);
  const auto key = owner.enqueue(0, &delivery); adapter.stage_count = 4;
  auto &first = adapter.stages[0]; first.additional[0] = 2; first.remaining[0] = 1; first.advanced = true;
  for (std::size_t at = 1; at < 3; ++at) {
    auto &stage = adapter.stages[at]; stage.reserved[0] = stage.additional[0] = stage.remaining[0] = stage.outputs[0] = 1;
    stage.advanced = true;
  }
  auto &last = adapter.stages[3]; last.reserved[0] = last.outputs[0] = 1; last.advanced = last.retired = true;
  allocation_count = 0; track_allocations = true;
  const auto submitted = owner.pump_once(adapter);
  const bool first_held = submitted.executed && submitted.waiting_retirement && owner.credit(key, 0) == 1;
  const auto advanced = owner.pump_once(adapter);
  const bool both_charged = advanced.executed && advanced.waiting_retirement && owner.credit(key, 1) == 0;
  const bool ack_one = owner.acknowledge(key, 1, adapter) && owner.credit(key, 1) == 1;
  (void)owner.pump_once(adapter);
  const bool next_charged = owner.credit(key, 2) == 0 && owner.acknowledge(key, 2, adapter) && owner.credit(key, 2) == 1;
  const auto finished = owner.pump_once(adapter);
  track_allocations = false;
  require(first_held && both_charged && ack_one && next_charged && !allocation_count &&
      finished.executed && !finished.waiting_retirement && finished.waiting_output && owner.matches(key) &&
      adapter.progress(key).generated.size() == 3 && adapter.publications[0] == 1 &&
      adapter.events[1] == 102 && adapter.events[2] == 202 &&
      adapter.events[3] == 103 && adapter.events[4] == 203,
      "continuous successor-before-observation lost credit, serialized pumps, allocated or republished output");
  require(owner.acknowledge(key, 3, adapter) && owner.pump_once(adapter).completed == 1 && !owner.matches(key),
      "last output did not release its original delivery owner");
}

void pending_cohort_defers_cancel_eos_and_admission() {
  i::RequestLoop owner; owner.prepare(3, 2); AsyncAdapter adapter{5, 5, 1};
  i::OutputDelivery a_delivery(2), b_delivery(2);
  const auto a = owner.enqueue(0, &a_delivery), b = owner.enqueue(1, &b_delivery), queued = owner.enqueue(2);
  adapter.stage_count = 4;
  auto &first = adapter.stages[0]; first.advanced = true;
  first.additional[0] = first.additional[1] = 2; first.remaining[0] = first.remaining[1] = 1;
  auto &second = adapter.stages[1]; second.advanced = true;
  for (std::size_t slot = 0; slot < 2; ++slot)
    second.reserved[slot] = second.additional[slot] = second.remaining[slot] = second.outputs[slot] = 1;
  second.eos[0] = true;
  for (std::size_t at = 2; at < 4; ++at) {
    auto &stage = adapter.stages[at]; stage.reserved[0] = stage.reserved[1] = 1;
    stage.additional[1] = 1; stage.cancelled[0] = true;
    if (at == 2) stage.remaining[0] = stage.remaining[1] = 1;
    else stage.retired = true;
  }
  (void)owner.pump_once(adapter);
  rejects<std::logic_error>([&] { (void)owner.pause(a); });
  rejects<std::logic_error>([&] { owner.prepare(3, 2); });
  rejects<std::logic_error>([&] { (void)owner.enqueue(0); });
  (void)owner.pump_once(adapter);
  require(adapter.progress(a).turn_done && adapter.progress(a).generated.size() == 1 && !adapter.publications[0] &&
      owner.active().size() == 2 && owner.matches(a) && adapter.admissions == 2,
      "early EOS published/reordered a cohort before its already submitted successor retired");
  require(owner.acknowledge(a, 1, adapter) && owner.acknowledge(b, 1, adapter) && owner.cancel(a, adapter).ok(),
      "in-flight ingress did not preserve original ACK/cancel authority");
  allocation_count = 0; track_allocations = true;
  const auto waiting = owner.pump_once(adapter);
  const bool held = waiting.waiting_retirement && owner.active().size() == 2 && owner.matches(a) &&
      !adapter.retirement_polls[0] && !adapter.eviction_polls[0] && adapter.admissions == 2;
  const auto retired = owner.pump_once(adapter);
  track_allocations = false;
  require(held && !allocation_count && retired.cancelled == 1 && retired.completed == 1 && !retired.waiting_retirement &&
      !owner.matches(a) && owner.matches(b) && owner.waiting_front() == queued && adapter.admissions == 2 &&
      owner.active().size() == 1 && owner.active()[0] == b && !adapter.publications[0] &&
      adapter.progress_values[0]->generated.size() == 1 && adapter.progress(b).generated.size() == 1,
      "actual discard lost committed A, published a cancelled turn or automatically executed a new cohort");
}

void failure_observation_is_once_and_cleanup_cannot_resubmit() {
  i::RequestLoop owner; owner.prepare(1, 1); AsyncAdapter adapter{3}; i::OutputDelivery delivery(2);
  const auto key = owner.enqueue(0, &delivery); adapter.stage_count = 4;
  const auto error = std::make_exception_ptr(WaveFailure{});
  auto &first = adapter.stages[0]; first.additional[0] = 2; first.remaining[0] = 1; first.advanced = true;
  auto &second = adapter.stages[1]; second.reserved[0] = second.additional[0] = second.remaining[0] = second.outputs[0] = 1;
  second.advanced = true; second.failure = error;
  for (std::size_t at = 2; at < 4; ++at) {
    auto &stage = adapter.stages[at]; stage.observe_only = true; stage.failure = error; stage.reserved[0] = 1;
    if (at == 2) stage.remaining[0] = 1;
    else stage.retired = true;
  }
  (void)owner.pump_once(adapter);
  try { (void)owner.pump_once(adapter); throw std::logic_error("missing wave failure"); }
  catch (const WaveFailure &failure) {
    // A runtime may copy an exception during rethrow/current_exception. Check
    // the original dynamic type and payload, not cross-catch pointer equality.
    require(failure.payload == WaveFailure::expected_payload &&
        std::string_view(failure.what()) == "scripted failure after committed A",
        "wave replaced its original exception type or payload");
  }
  require(owner.matches(key) && adapter.progress(key).generated.size() == 1 && !adapter.publications[0] &&
      !adapter.retirement_polls[0] && owner.credit(key, 1) == 0 && owner.acknowledge(key, 1, adapter),
      "late failure rolled back committed A, released pending B or lost its charged output");
  const auto pending = owner.pump_once(adapter);
  require(!pending.executed && pending.waiting_retirement && owner.credit(key, 1) == 1,
      "cleanup repeated the old exception or used fresh ACK credit for another submission");
  const auto retired = owner.pump_once(adapter);
  require(!retired.executed && !retired.waiting_retirement && !retired.completed && owner.matches(key) &&
      adapter.stage_at == 4 && adapter.progress(key).generated.size() == 1 && adapter.executions[0] == 1 &&
      !adapter.publications[0] && !adapter.retirement_polls[0] && owner.credit(key, 1) == 2,
      "failure retirement automatically retried old work, erased A or repeated an observation");

  i::RequestLoop state_only; state_only.prepare(1, 1); AsyncAdapter state{2};
  const std::array turns{i::GenerationTurn{{1}, 1}, i::GenerationTurn{{}, 1}};
  state.progress_values[0]->bind_open_turns(std::span(turns).first(1), 2, 1);
  const auto state_key = state_only.enqueue(0); state.stage_count = 3;
  state.stages[0].advanced = true; state.stages[0].additional[0] = 1;
  state.stages[1].advanced = state.stages[1].retired = true;
  state.stages[1].additional[0] = state.stages[1].outputs[0] = 1;
  require(state_only.pump_once(state).waiting_retirement && state_only.matches(state_key),
      "zero-output in-flight state lost its actual episode ownership");
  rejects<std::logic_error>([&] { (void)state_only.pause(state_key); });
  state.progress(state_key).extend_turns(turns); state.progress(state_key).close_input();
  require(state.progress(state_key).turn_index == 0 && state.admissions == 1,
      "in-flight input intent advanced/replaced the original turn");
  const auto first_turn = state_only.pump_once(state);
  require(!first_turn.completed && state.progress(state_key).turn_index == 1 &&
      !state.progress(state_key).turn_started() && state_only.waiting_front() == state_key && state.publications[0] == 1,
      "retired state-only cohort did not hand off its later input at the real turn boundary");
  state.stages[2].advanced = state.stages[2].retired = true;
  state.stages[2].additional[0] = state.stages[2].outputs[0] = 1;
  require(state_only.pump_once(state).completed == 1 && state.publications[0] == 2,
      "later input did not finish through the same wave callback");
}

void invalid_wave_reservation_retains_real_retirement_gate() {
  i::RequestLoop owner; owner.prepare(1, 1); AsyncAdapter adapter{2};
  i::OutputDelivery delivery((std::numeric_limits<std::size_t>::max)());
  const auto key = owner.enqueue(0, &delivery); adapter.stage_count = 2;
  auto &invalid = adapter.stages[0]; invalid.advanced = true; invalid.additional[0] = 2;
  invalid.remaining[0] = (std::numeric_limits<std::size_t>::max)();
  auto &cleanup = adapter.stages[1]; cleanup.observe_only = cleanup.retired = true; cleanup.reserved[0] = 2;
  rejects<std::logic_error>([&] { (void)owner.pump_once(adapter); });
  require(owner.matches(key) && owner.credit(key, 0) == (std::numeric_limits<std::size_t>::max)() - 2 &&
      !adapter.retirement_polls[0], "overflowing candidate reservation escaped the original bounded retirement gate");
  rejects<std::logic_error>([&] { (void)owner.pause(key); });
  const auto retired = owner.pump_once(adapter);
  require(!retired.executed && !retired.waiting_retirement && !retired.completed && owner.matches(key) &&
      !adapter.progress(key).observed_commits(), "invalid candidate bookkeeping granted release or replayed work");
}

void explicit_yield_drains_then_preserves_original_fifo_and_turn() {
  i::RequestLoop owner; owner.prepare(3, 2); AsyncAdapter adapter{3, 3, 1};
  const auto a = owner.enqueue(0), b = owner.enqueue(1), c = owner.enqueue(2);
  adapter.cancel_owner = &owner; adapter.stage_count = 5;
  auto &first = adapter.stages[0]; first.advanced = first.request_yield = true;
  first.additional[0] = first.additional[1] = 3; first.remaining[0] = first.remaining[1] = 1;
  auto &draining = adapter.stages[1]; draining.advanced = draining.drain_requested = true;
  draining.reserved[0] = draining.reserved[1] = 1; draining.cancelled[0] = true; draining.outputs[1] = 1;
  auto &retire = adapter.stages[2]; retire.drain_requested = retire.retired = true; retire.cancelled[0] = true;
  auto &next = adapter.stages[3]; next.advanced = next.retired = true;
  next.additional[1] = 2; next.additional[2] = next.outputs[1] = next.outputs[2] = 1;
  auto &last = adapter.stages[4]; last.advanced = last.retired = true; last.additional[1] = last.outputs[1] = 1;
  require(owner.pump_once(adapter).waiting_retirement && owner.active().size() == 2,
      "yield intent ended its original submitted wave");
  owner.request_yield(); // Duplicate ingress does not create another scheduling authority.
  require(owner.cancel(a, adapter).ok(), "yielding original member could not be cancelled");
  allocation_count = 0; track_allocations = true;
  const auto pending = owner.pump_once(adapter);
  const bool retained = pending.waiting_retirement && owner.active().size() == 2 && owner.matches(a) &&
      adapter.progress(b).generated.size() == 1 && adapter.admissions == 2;
  const auto retired = owner.pump_once(adapter);
  const bool yielded = retired.completed == 1 && retired.cancelled == 1 && !retired.waiting_retirement &&
      !owner.matches(a) && owner.matches(b) && owner.active().empty() && owner.waiting_front() == c &&
      !adapter.progress(b).turn_done && adapter.progress(b).turn_index == 0 && adapter.progress(b).turn_started();
  const auto resumed = owner.pump_once(adapter);
  const auto complete = owner.pump_once(adapter);
  track_allocations = false;
  require(retained && yielded && resumed.admitted == 2 && resumed.completed == 1 && complete.completed == 1 &&
      !allocation_count && adapter.admission_order[2] == c && adapter.admission_order[3] == b &&
      !adapter.publications[0] && adapter.publications[1] == 1 && adapter.publications[2] == 1 &&
      adapter.progress_values[1]->generated.size() == 3 && adapter.progress_values[1]->observed_commits() == 3 &&
      adapter.executions[1] == 3 && adapter.executions[2] == 1 && !owner.matches(b) && !owner.matches(c),
      "bounded yield skipped FIFO, advanced a turn, revived cancellation, allocated or repeated an original commit");

  // The same explicit intent also applies between already retired sync waves;
  // with no intent, all earlier FIFO tests keep their original policy.
  i::RequestLoop synchronous; synchronous.prepare(3, 2); Adapter sync{3, 3, 1};
  const auto sa = synchronous.enqueue(0), sb = synchronous.enqueue(1), sc = synchronous.enqueue(2);
  (void)synchronous.pump_once(sync); synchronous.request_yield();
  const auto switched = synchronous.pump_once(sync);
  require(switched.admitted == 2 && sync.admission_order[2] == sc && sync.admission_order[3] == sa &&
      sync.progress(sb).generated.size() == 1 && sync.progress(sa).turn_index == 0 &&
      sync.progress(sa).generated.size() == 2, "between-wave yield changed the default turn or skipped its FIFO peer");
}

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
    in_flight_waves_keep_exact_credit_and_allow_continuation();
    pending_cohort_defers_cancel_eos_and_admission();
    failure_observation_is_once_and_cleanup_cannot_resubmit();
    invalid_wave_reservation_retains_real_retirement_gate();
    explicit_yield_drains_then_preserves_original_fifo_and_turn();
    std::puts("request loop: 11 host contract groups passed; no model/device loaded");
    return 0;
  } catch (const std::exception &error) {
    track_allocations = false;
    std::fprintf(stderr, "request loop: %s\n", error.what()); return 1;
  }
}
