// Host owner contracts only. Retirement observations below are explicit test
// inputs, not evidence that any device completed or freed physical memory.
#include "../../engine/inference/request_loop.h"
#include "../../engine/inference/request_admission.h"
#include "../../engine/inference/token_delivery.h"
#include "../../engine/inference/queued_token_consumer.h"
#include "../../engine/inference/model_executor.h"
#include "../../engine/inference/model_provider.h"
#include "../../engine/inference/generation_dispatcher.h"
#include "../../engine/inference/token_worker.h"
#include "../../engine/inference/request_wave_execution.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <deque>
#include <limits>
#include <cstdlib>
#include <new>
#include <optional>
#include <sstream>
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
static_assert(!std::is_copy_constructible_v<i::TokenDelivery> && !std::is_move_constructible_v<i::TokenDelivery>);

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
  bool error_is_retired = false, stop_first_turn = false;
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
      const bool is_stopped = stop_first_turn && !p.turn_index && p.generated.empty();
      require(p.observe_commit(p.observed_commits(), output, true, true, is_stopped), "host output was replayed");
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

void arrival_during_native_work_waits_for_original_retirement() {
  i::RequestLoop owner; owner.prepare(2, 1); AsyncAdapter adapter{1, 1};
  i::OutputDelivery first_output(1), later_output(1);
  const auto first = owner.enqueue(0, &first_output); adapter.stage_count = 3;
  adapter.stages[0].additional[0] = 1; adapter.stages[0].remaining[0] = 1;
  adapter.stages[0].advanced = true;
  adapter.stages[1].reserved[0] = 1; adapter.stages[1].outputs[0] = 1;
  adapter.stages[1].advanced = adapter.stages[1].retired = true;
  adapter.stages[2].additional[1] = 1; adapter.stages[2].outputs[1] = 1;
  adapter.stages[2].advanced = adapter.stages[2].retired = true;
  const auto submitted = owner.pump_once(adapter);
  require(submitted.waiting_retirement && owner.free_slot() == 1 && adapter.admissions == 1,
      "native submission lost its in-flight owner or free ingress slot");
  const auto later = owner.enqueue(*owner.free_slot(), &later_output);
  require(!owner.free_slot() && !owner.waiting_front() && adapter.admissions == 1,
      "new arrival modified or entered the in-flight cohort");
  const auto retired = owner.pump_once(adapter);
  require(!retired.waiting_retirement && adapter.admissions == 1 && owner.waiting_front() == later &&
      owner.matches(first) && owner.matches(later), "new arrival entered before native retirement");
  require(owner.acknowledge(first, 1, adapter), "retired first output was not delivered");
  (void)owner.pump_once(adapter);
  require(adapter.admissions == 2 && adapter.progress(later).generated == std::vector<std::int32_t>{11} &&
      owner.acknowledge(later, 1, adapter), "new arrival did not enter after original retirement");
  require(owner.pump_once(adapter).completed == 1 && owner.free_slot() == 0 && !owner.matches(later),
      "new arrival did not retire and release its own slot");
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

void paused_session_allows_reuse_of_actually_retired_peer_slot() {
  i::RequestLoop owner; owner.prepare(2, 2); Adapter adapter{2, 1};
  const std::array turns{i::GenerationTurn{{1}, 1}, i::GenerationTurn{{2}, 1}};
  adapter.progress_values[0]->bind_open_turns(std::span(turns).first(1), 2, 2);
  adapter.progress_values[0]->initial_sampling = {42, 7, 0};
  i::OutputDelivery live_output(1), first_output(1), second_output(1);
  require(owner.free_slot() == 0, "fresh owner did not expose its first free slot");
  const auto live = owner.enqueue(0, &live_output);
  (void)owner.pump_once(adapter);
  require(adapter.progress(live).waiting_for_input() && adapter.residents[0] && owner.free_slot() == 1 &&
      owner.acknowledge(live, 1, adapter), "paused live session lost its key, head or remaining slot");

  const auto first = owner.enqueue(*owner.free_slot(), &first_output);
  (void)owner.pump_once(adapter);
  require(owner.matches(first) && !owner.free_slot() && adapter.retirement_polls[1] == 1 &&
      adapter.progress(live).generated == std::vector<std::int32_t>{1},
      "undelivered peer output or paused session exposed an occupied slot");
  require(owner.acknowledge(first, 1, adapter), "first peer output did not acknowledge");
  require(owner.pump_once(adapter).completed == 1 && owner.free_slot() == 1 && !owner.matches(first),
      "actually retired and drained peer did not release its slot");

  adapter.progress_values[1].emplace(1);
  const auto second = owner.enqueue(*owner.free_slot(), &second_output);
  require(second != first && !owner.acknowledge(first, 1, adapter) && owner.matches(live),
      "stale peer identity affected a replacement or the paused session");
  (void)owner.pump_once(adapter);
  require(owner.acknowledge(second, 1, adapter) && owner.pump_once(adapter).completed == 1 &&
      owner.free_slot() == 1, "replacement peer failed to retire its own slot");

  auto &progress = adapter.progress(live);
  progress.extend_turns(turns); owner.wake(live); (void)owner.pump_once(adapter);
  require(progress.generated == std::vector<std::int32_t>({1, 2}) && progress.observed_commits() == 2 &&
      progress.initial_sampling.seed == 42 && progress.initial_sampling.stream == 7 && adapter.residents[0] &&
      owner.acknowledge(live, 2, adapter), "new peers changed the original live progress or sampling identity");
  progress.close_input(); owner.wake(live);
  require(owner.pump_once(adapter).completed == 1 && !owner.matches(live) && owner.free_slot() == 0,
      "closed live session did not retire and release its original slot");
}

struct DeliveryConsumer final {
  std::array<std::array<std::int32_t, 4>, 2> ids{};
  std::array<std::size_t, 2> count{}, notifications{};
  std::array<bool, 2> cancel{}, finished_cancelled{};
  bool block_first = false, cancel_after_write = false;
  std::size_t max_pending() const noexcept { return 1; }
  bool try_write(i::RequestKey key, std::int32_t id) noexcept {
    if (block_first && key.slot == 0) return false;
    ids[key.slot][count[key.slot]++] = id;
    if (cancel_after_write && key.slot == 0) cancel[0] = true;
    return true;
  }
  bool cancel_requested(i::RequestKey key) const noexcept { return cancel[key.slot]; }
  void finished(i::RequestKey key, bool stopped) noexcept {
    ++notifications[key.slot]; finished_cancelled[key.slot] = stopped;
  }
  void wait() noexcept {}
};

void generic_token_delivery_preserves_peer_and_retirement() {
  i::RequestLoop owner; owner.prepare(2, 2); Adapter adapter{3, 2};
  i::TokenDelivery a(1), b(1); DeliveryConsumer consumer; consumer.block_first = true;
  const auto ka = owner.enqueue(0, &a.window()), kb = owner.enqueue(1, &b.window());
  a.bind(owner, ka); b.bind(owner, kb);
  rejects<std::logic_error>([&] { a.bind(owner, kb); });
  i::RequestLoop foreign; foreign.prepare(1, 1);
  rejects<std::logic_error>([&] { (void)a.service(foreign, adapter, consumer); });
  (void)owner.pump_once(adapter);
  require(!a.service(owner, adapter, consumer) && b.service(owner, adapter, consumer) &&
      !a.window().acknowledged() && b.window().acknowledged() == 1,
      "blocked delivery lost credit or prevented the peer's actual write");
  (void)owner.pump_once(adapter);
  (void)b.service(owner, adapter, consumer);
  consumer.cancel[0] = true; adapter.deferred_terminal = 2;
  require(a.service(owner, adapter, consumer) && a.cancelled() && a.window().acknowledged() == 1 &&
      adapter.stops[0] == 1 && !consumer.notifications[0],
      "cancellation failed to abandon only the undelivered prefix");
  for (std::size_t n = 0; n < 12 && (owner.matches(ka) || owner.matches(kb)); ++n) {
    (void)owner.pump_once(adapter);
    (void)a.service(owner, adapter, consumer);
    (void)b.service(owner, adapter, consumer);
    require(!owner.matches(ka) || !consumer.notifications[0],
        "consumer was finished before the original request retired");
  }
  require(!owner.matches(ka) && !owner.matches(kb) && consumer.notifications[0] == 1 &&
      consumer.notifications[1] == 1 && consumer.finished_cancelled[0] && !consumer.finished_cancelled[1] &&
      !consumer.count[0] && consumer.count[1] == 2 && consumer.ids[1][0] == 11 && consumer.ids[1][1] == 12,
      "peer delivery, cancellation or terminal notification was lost");
  require(!a.service(owner, adapter, consumer) && !b.service(owner, adapter, consumer) &&
      consumer.notifications[0] == 1 && consumer.notifications[1] == 1,
      "a repeated terminal observation duplicated notification");
}

void generic_token_delivery_catches_cancel_during_write() {
  i::RequestLoop owner; owner.prepare(1, 1); Adapter adapter{3};
  i::TokenDelivery delivery(1); DeliveryConsumer consumer; consumer.cancel_after_write = true;
  const auto key = owner.enqueue(0, &delivery.window());
  delivery.bind(owner, key);
  (void)owner.pump_once(adapter);
  allocation_count = 0; track_allocations = true;
  const auto advanced = delivery.service(owner, adapter, consumer);
  track_allocations = false;
  require(advanced && !allocation_count && consumer.count[0] == 1 && delivery.cancelled() &&
      delivery.window().acknowledged() == 1 && adapter.stops[0] == 1,
      "write-time cancellation was missed, duplicated or allocated");
  (void)owner.pump_once(adapter);
  (void)delivery.service(owner, adapter, consumer);
  require(!owner.matches(key) && consumer.notifications[0] == 1 && consumer.finished_cancelled[0] &&
      adapter.executions[0] == 1, "write-time cancellation continued execution or lost its terminal");
}

struct TurnConsumer final {
  std::array<int, 8> events{};
  std::size_t count = 0, finishes = 0;
  bool blocked = true, cancel = false, cancelled = false;
  std::size_t max_pending() const noexcept { return 1; }
  bool try_write(i::RequestKey, std::int32_t id) noexcept { events[count++] = id; return true; }
  bool try_turn_end(i::RequestKey, std::size_t turn) noexcept {
    if (blocked) return false;
    events[count++] = -1 - static_cast<int>(turn); return true;
  }
  bool cancel_requested(i::RequestKey) const noexcept { return cancel; }
  void finished(i::RequestKey, bool stopped) noexcept { ++finishes; cancelled = stopped; }
  void wait() noexcept {}
};

void delivery_orders_turn_boundaries_and_retirement() {
  i::RequestLoop owner; owner.prepare(1, 1); Adapter adapter{3};
  const std::array turns{i::GenerationTurn{{1}, 2}, i::GenerationTurn{{2}, 1}};
  adapter.progress_values[0]->bind_turns(turns);
  i::TokenDelivery delivery(1, turns.size()); TurnConsumer consumer;
  const auto key = owner.enqueue(0, &delivery.window()); delivery.bind(owner, key);
  (void)owner.pump_once(adapter); (void)delivery.service(owner, adapter, consumer);
  (void)owner.pump_once(adapter); (void)delivery.service(owner, adapter, consumer);
  require(consumer.count == 2 && consumer.events[0] == 1 && consumer.events[1] == 2 &&
      delivery.window().next_turn_end()->through == 2 && !consumer.finishes,
      "blocked boundary lost its preceding output or finished the request");
  (void)owner.pump_once(adapter); (void)delivery.service(owner, adapter, consumer);
  require(consumer.count == 2 && owner.matches(key) && !consumer.finishes,
      "a later token overtook a blocked turn boundary");
  consumer.blocked = false;
  allocation_count = 0; track_allocations = true;
  (void)delivery.service(owner, adapter, consumer);
  track_allocations = false;
  require(!allocation_count && consumer.count == 5 && consumer.events[2] == -1 &&
      consumer.events[3] == 3 && consumer.events[4] == -2 && !consumer.finishes,
      "turn events were reordered, duplicated or allocated during delivery");
  (void)owner.pump_once(adapter); (void)delivery.service(owner, adapter, consumer);
  require(!owner.matches(key) && consumer.finishes == 1 && !consumer.cancelled,
      "turn delivery bypassed the original terminal retirement");

  i::RequestLoop stopped; stopped.prepare(1, 1); Adapter cancel_adapter{1};
  const std::array one{i::GenerationTurn{{1}, 1}}; cancel_adapter.progress_values[0]->bind_turns(one);
  i::TokenDelivery cancelled(1, 1); TurnConsumer cancel_consumer;
  const auto stopped_key = stopped.enqueue(0, &cancelled.window()); cancelled.bind(stopped, stopped_key);
  (void)stopped.pump_once(cancel_adapter); (void)cancelled.service(stopped, cancel_adapter, cancel_consumer);
  require(cancel_consumer.count == 1 && stopped.matches(stopped_key), "blocked terminal boundary lost its key");
  cancel_consumer.cancel = true; (void)cancelled.service(stopped, cancel_adapter, cancel_consumer);
  (void)stopped.pump_once(cancel_adapter); (void)cancelled.service(stopped, cancel_adapter, cancel_consumer);
  require(!stopped.matches(stopped_key) && cancel_consumer.count == 1 &&
      cancel_consumer.finishes == 1 && cancel_consumer.cancelled,
      "cancelled boundary was emitted or terminal notification was lost");

  i::RequestLoop early; early.prepare(1, 1); Adapter early_adapter{4}; early_adapter.stop_first_turn = true;
  const std::array early_turns{i::GenerationTurn{{1}, 3}, i::GenerationTurn{{2}, 1}};
  early_adapter.progress_values[0]->bind_turns(early_turns);
  i::TokenDelivery early_delivery(1, 2); TurnConsumer early_consumer; early_consumer.blocked = false;
  const auto early_key = early.enqueue(0, &early_delivery.window()); early_delivery.bind(early, early_key);
  for (std::size_t n = 0; n < 8 && early.matches(early_key); ++n) {
    (void)early.pump_once(early_adapter); (void)early_delivery.service(early, early_adapter, early_consumer);
  }
  require(!early.matches(early_key) && early_consumer.count == 4 && early_consumer.events[0] == 1 &&
      early_consumer.events[1] == -1 && early_consumer.events[2] == 2 && early_consumer.events[3] == -2 &&
      early_consumer.finishes == 1, "early stop did not publish its actual turn boundary");
}

void turn_boundary_queue_is_bounded_and_keyed() {
  i::RequestWindow owner; owner.prepare(1, 1);
  const auto key = owner.enqueue(0);
  i::QueuedTokenConsumer queue(1, 1); i::TokenConsumerRef ref(queue);
  DeliveryConsumer legacy; i::TokenConsumerRef legacy_ref(legacy);
  require(ref.supports_turn_ends() && !legacy_ref.supports_turn_ends() && ref.start(key) &&
      ref.try_write(key, 7) && !ref.try_turn_end(key, 0), "boundary bypassed queue credit or capability");
  require(queue.read(key).token == 7 && ref.try_turn_end(key, 0) && !ref.try_write(key, 8),
      "boundary did not occupy the existing bounded queue");
  const auto boundary = queue.read(key);
  require(boundary.state == i::QueuedTokenConsumer::ReadState::kTurnEnd && boundary.turn == 0 &&
      ref.try_write(key, 8) && queue.read(key).token == 8, "queue reordered a turn boundary");
  queue.finished(key, false);
  require(queue.read(key).state == i::QueuedTokenConsumer::ReadState::kCompleted && queue.release(key) &&
      !ref.try_turn_end(key, 1), "retired key accepted a stale boundary");

  i::OutputDelivery delivery(1, 2);
  delivery.record_turn_end(0, 1); delivery.record_turn_end(0, 1);
  rejects<std::logic_error>([&] { delivery.record_turn_end(0, 2); });
  rejects<std::out_of_range>([&] { delivery.record_turn_end(2, 2); });
  rejects<std::logic_error>([&] { delivery.accept_turn_end(); });
  delivery.acknowledge_prefix(1, 1); delivery.accept_turn_end();
  require(delivery.drained(1), "accepted boundary did not drain its original prefix");
}

void queued_delivery_preserves_bounds_and_terminal_identity() {
  i::RequestWindow window, foreign; window.prepare(2, 2); foreign.prepare(2, 2);
  const auto a = window.enqueue(0), b = window.enqueue(1), other = foreign.enqueue(0);
  i::QueuedTokenConsumer queue(2, 1);
  i::TokenConsumerRef ref(queue);
  require(ref.start(a) && ref.start(b) && !ref.start(a) && !ref.start(other),
      "queue replaced an active request identity");
  require(queue.wait_for_start(0) == a && ref.try_write(a, 7) && ref.try_write(b, 11) &&
      !ref.try_write(a, 8), "queue did not apply per-request bounds");
  const auto first = queue.read(a);
  require(first.state == i::QueuedTokenConsumer::ReadState::kToken && first.token == 7,
      "queue changed the accepted token");
  ref.wait(); // Reader signalled before the wait; the wake must not be lost.
  require(queue.try_write(a, 8) && queue.request_cancel(a) && queue.cancel_requested(a) &&
      !queue.try_write(a, 9) && !queue.release(a), "cancel bypassed retirement or accepted more output");
  queue.finished(a, true);
  require(!queue.release(a), "terminal release discarded a queued token");
  const auto last = queue.read(a), terminal = queue.read(a);
  require(last.state == i::QueuedTokenConsumer::ReadState::kToken && last.token == 8 &&
      terminal.state == i::QueuedTokenConsumer::ReadState::kCancelled && queue.release(a) &&
      queue.read(a).state == i::QueuedTokenConsumer::ReadState::kStale,
      "cancelled terminal lost an accepted token or retained an old key");
  const auto peer = queue.read(b); queue.finished(b, false);
  require(peer.state == i::QueuedTokenConsumer::ReadState::kToken && peer.token == 11 &&
      queue.read(b).state == i::QueuedTokenConsumer::ReadState::kCompleted && queue.release(b),
      "peer completion was affected by another request's cancellation");
  require(queue.start(other) && queue.try_write(other, 13) &&
      queue.read(a).state == i::QueuedTokenConsumer::ReadState::kStale &&
      !queue.request_cancel(a) && !queue.release(a), "stale key affected replacement request");
  queue.shutdown();
  require(queue.cancel_requested(other) && queue.read(other).token == 13 &&
      queue.read(other).state == i::QueuedTokenConsumer::ReadState::kClosed && !queue.release(other),
      "shutdown fabricated retirement or discarded accepted output");
  queue.finished(other, true);
  require(queue.read(other).state == i::QueuedTokenConsumer::ReadState::kCancelled && queue.release(other) &&
      !queue.wait_for_start(1), "actual retirement did not survive shutdown");
}

void transport_slot_waits_for_terminal_observation_before_reuse() {
  i::RequestWindow owner; owner.prepare(1, 1);
  const auto key = owner.enqueue(0);
  i::QueuedTokenConsumer queue(1, 1);
  require(queue.can_start(0) && !queue.can_start(1) && queue.start(key) && !queue.can_start(0),
      "transport slot availability did not follow its original owner");
  queue.finished(key, false);
  require(!queue.release(key) && !queue.can_start(0), "unobserved terminal was discarded for reuse");
  require(queue.read(key).state == i::QueuedTokenConsumer::ReadState::kCompleted && !queue.can_start(0),
      "terminal observation alone released the original slot");
  require(queue.release(key) && queue.can_start(0), "observed terminal did not release transport capacity");
  queue.wait(); // Release notified the producer before this wait; no wake is lost.
  owner.admitted(key); owner.retire_active(key); owner.release(key);
  const auto next = owner.enqueue(0);
  require(queue.start(next) && !queue.release(key) && !queue.try_write(key, 9),
      "stale terminal identity affected the replacement transport slot");
}

void abandoned_transport_waits_for_actual_retirement() {
  i::RequestWindow owner; owner.prepare(1, 1);
  i::QueuedTokenConsumer queue(1, 1);
  const auto first = owner.enqueue(0);
  require(queue.start(first) && queue.try_write(first, 7) && queue.abandon(first) &&
      queue.cancel_requested(first) && !queue.can_start(0),
      "abandonment reused an unretired model slot");
  queue.finished(first, true);
  require(queue.can_start(0) && queue.read(first).state == i::QueuedTokenConsumer::ReadState::kStale,
      "retired abandoned transport retained its unread output");
  owner.admitted(first); owner.retire_active(first); owner.release(first);
  const auto second = owner.enqueue(0);
  require(queue.start(second) && queue.try_write(second, 8), "replacement transport did not start");
  queue.finished(second, false);
  require(!queue.can_start(0) && queue.abandon(second) && queue.can_start(0),
      "abandonment after actual retirement did not release unread output");
  require(!queue.abandon(first) && !queue.try_write(first, 9),
      "abandoned stale key affected a later request");
}

void new_admission_skips_unreleased_transport_after_model_retirement() {
  i::RequestLoop owner; owner.prepare(2, 2); Adapter adapter{1, 1};
  i::QueuedTokenConsumer queue(2, 1); i::TokenDelivery first_delivery(1), peer_delivery(1);
  const auto first = owner.enqueue(0, &first_delivery.window()); first_delivery.bind(owner, first);
  require(queue.start(first), "first transport slot was not attached");
  (void)owner.pump_once(adapter); (void)first_delivery.service(owner, adapter, queue);
  (void)owner.pump_once(adapter); (void)first_delivery.service(owner, adapter, queue);
  require(!owner.matches(first) && owner.free_slot() == 0 && !queue.can_start(0),
      "model retirement incorrectly implied downstream terminal observation");
  const auto available = owner.free_slot([&](std::size_t slot) { return queue.can_start(slot); });
  require(available == 1, "admission did not skip the downstream-owned first slot");
  const auto peer = owner.enqueue(*available, &peer_delivery.window()); peer_delivery.bind(owner, peer);
  require(queue.start(peer), "peer could not use the independently available slot");
  (void)owner.pump_once(adapter); (void)peer_delivery.service(owner, adapter, queue);
  (void)owner.pump_once(adapter); (void)peer_delivery.service(owner, adapter, queue);
  require(queue.read(first).token == 1 &&
      queue.read(first).state == i::QueuedTokenConsumer::ReadState::kCompleted && queue.release(first) &&
      owner.free_slot([&](std::size_t slot) { return queue.can_start(slot); }) == 0,
      "observing the original terminal did not restore its admission capacity");
  require(queue.read(peer).token == 11 && queue.read(peer).state == i::QueuedTokenConsumer::ReadState::kCompleted &&
      queue.release(peer), "new peer lost output or its own terminal");
}

struct IngressTestControl final {
  std::mutex mutex;
  std::condition_variable changed;
  bool started = false, proceed = false, fail = false, wrong_terminal = false, early_stop_first = false;
  std::vector<std::size_t> batches;
  std::vector<std::optional<std::uint64_t>> seeds;
};

class IngressTestExecutor final : public i::ModelExecutor {
public:
  explicit IngressTestExecutor(std::shared_ptr<IngressTestControl> control) : control_(std::move(control)) {}
  i::ModelDescription describe() const override { return {256, 128, {3, 4}}; }
  void validate(i::GenerationBatch batch) override {
    for (const auto &request : batch.sessions)
      if (request.turns.front().prompt.front() < 0) throw i::RequestRejected("invalid test token");
  }
  void execute(i::GenerationBatch batch, i::TokenConsumerRef &consumer) override {
    std::vector<i::RequestKey> keys, foreign_keys;
    i::RequestWindow foreign;
    foreign.prepare(batch.sessions.size(), batch.sessions.size());
    window_.prepare(batch.sessions.size(), batch.sessions.size());
    for (std::size_t index = 0; index < batch.sessions.size(); ++index) {
      const auto key = window_.enqueue(index); window_.admitted(key);
      if (!consumer.start(key)) throw std::runtime_error("test consumer refused its key");
      keys.push_back(key);
      const auto foreign_key = foreign.enqueue(index); foreign.admitted(foreign_key); foreign_keys.push_back(foreign_key);
    }
    {
      std::unique_lock lock(control_->mutex);
      control_->batches.push_back(batch.sessions.size());
      for (const auto &schedule : batch.sessions) control_->seeds.push_back(schedule.sampling_seed);
      if (control_->batches.size() == 1) {
        control_->started = true; control_->changed.notify_all();
        control_->changed.wait(lock, [&] { return control_->proceed; });
        if (control_->fail) throw std::runtime_error("scripted model failure");
      }
    }
    for (std::size_t slot = 0; slot < keys.size(); ++slot) {
      const auto key = keys[slot];
      const auto emit_turn = [&](const i::GenerationTurn &turn, std::size_t turn_index) {
        const auto first = turn.prompt.empty() ? 0 : turn.prompt.front();
        const auto outputs = control_->early_stop_first && !turn_index ? std::size_t{1} : turn.max_outputs;
        for (std::size_t n = 0; n < outputs && !consumer.cancel_requested(key);) {
          if (consumer.try_write(key, static_cast<std::int32_t>(first + n))) ++n;
          else if (!consumer.cancel_requested(key)) consumer.wait();
        }
        if (batch.sessions[slot].reports_turn_ends())
          while (!consumer.cancel_requested(key) && !consumer.try_turn_end(key, turn_index)) consumer.wait();
      };
      for (std::size_t turn_index = 0; turn_index < batch.sessions[slot].turns.size(); ++turn_index)
        emit_turn(batch.sessions[slot].turns[turn_index], turn_index);
      if (batch.sessions[slot].accepts_live_input()) {
        if (!batch.input) throw std::logic_error("live test request lacks its input source");
        auto turn_index = batch.sessions[slot].turns.size();
        while (!consumer.cancel_requested(key)) {
          auto input = batch.input->poll_turn(key);
          if (input.state == i::TurnInputState::kClosed) break;
          if (input.state == i::TurnInputState::kWaiting) { consumer.wait(); continue; }
          emit_turn(input.turn, turn_index++);
        }
      }
      const auto cancelled = consumer.cancel_requested(key);
      window_.retire_active(key); window_.release(key);
      consumer.finished(control_->wrong_terminal ? foreign_keys[slot] : key, cancelled);
    }
  }
private:
  i::RequestWindow window_;
  std::shared_ptr<IngressTestControl> control_;
};

// A host-only request owner exercises the real dispatcher/input handoff while
// retaining the existing RequestLoop as the sole slot and retirement owner.
class DynamicIngressTestExecutor final : public i::ModelExecutor {
public:
  i::ModelDescription describe() const override { return {256, 128, {}}; }
  bool supports_dynamic_admission() const noexcept override { return true; }
  void validate(i::GenerationBatch batch) override {
    for (const auto &schedule : batch.sessions)
      if (schedule.turns.front().prompt.front() < 0) throw i::RequestRejected("invalid test token");
  }
  void execute(i::GenerationBatch batch, i::TokenConsumerRef &consumer) override {
    if (!batch.input || batch.sessions.size() != 1) throw std::logic_error("dynamic fixture needs one initial request");
    i::RequestLoop owner; owner.prepare(2, 2); Adapter adapter{128, 128};
    std::array<i::GenerationSessionSchedule, 2> schedules;
    std::array<std::optional<i::TokenDelivery>, 2> deliveries;
    std::array<i::RequestKey, 2> keys{};
    const auto admit = [&](std::size_t slot, i::GenerationSessionSchedule schedule) {
      schedules[slot] = std::move(schedule);
      auto &stored = schedules[slot];
      if (stored.accepts_live_input()) stored.turns.reserve(128);
      adapter.progress_values[slot].emplace(128);
      auto &progress = *adapter.progress_values[slot];
      if (stored.accepts_live_input()) progress.bind_open_turns(stored.turns, 128, 128);
      else progress.bind_turns(stored.turns);
      adapter.stopped[slot] = false;
      deliveries[slot].emplace(consumer.max_pending(), stored.reports_turn_ends() ?
          (stored.accepts_live_input() ? 128 : stored.turns.size()) : 0);
      keys[slot] = owner.enqueue(slot, &deliveries[slot]->window());
      deliveries[slot]->bind(owner, keys[slot]);
      if (!consumer.start(keys[slot])) throw std::logic_error("dynamic fixture consumer refused slot");
    };
    admit(0, batch.sessions.front());
    for (;;) {
      bool progress = false;
      for (auto &delivery : deliveries) if (delivery) progress = delivery->service(owner, adapter, consumer) || progress;
      const auto pumped = owner.pump_once(adapter);
      for (auto &delivery : deliveries) if (delivery) progress = delivery->service(owner, adapter, consumer) || progress;
      for (std::size_t slot = 0; slot < keys.size(); ++slot) {
        if (!owner.matches(keys[slot]) || !adapter.progress(keys[slot]).waiting_for_input()) continue;
        auto input = batch.input->poll_turn(keys[slot]);
        if (input.state == i::TurnInputState::kWaiting) continue;
        auto &request = adapter.progress(keys[slot]);
        if (input.state == i::TurnInputState::kClosed) request.close_input();
        else {
          if (input.turn.prompt == std::vector<std::int32_t>{250}) {
            if (!consumer.fail(keys[slot], std::make_exception_ptr(i::RequestRejected("turn exceeds context"))))
              throw std::logic_error("dynamic fixture could not reject one turn");
            progress = true;
            continue;
          }
          schedules[slot].turns.push_back(std::move(input.turn));
          request.extend_turns(schedules[slot].turns);
        }
        owner.wake(keys[slot]); progress = true;
      }
      while (const auto slot = owner.free_slot([&](auto index) { return batch.input->can_admit(index); })) {
        auto incoming = batch.input->poll_admission(*slot);
        if (!incoming) break;
        if (incoming->schedule.turns.front().prompt.front() < 0) {
          batch.input->reject_admission(*slot, std::make_exception_ptr(i::RequestRejected("invalid test token")));
        } else admit(*slot, std::move(incoming->schedule));
        progress = true;
      }
      if (std::ranges::none_of(keys, [&](auto key) { return owner.matches(key); })) break;
      if (!pumped.executed && !pumped.waiting_retirement && !progress) consumer.wait();
    }
  }
};

void schedule_admission_is_model_independent_and_overflow_safe() {
  using Error = i::RequestAdmissionError;
  i::GenerationSessionSchedule schedule{"finite", {{{1}, 1}}};
  require(i::initial_sampling_cursor(schedule, 12, 3) == i::SamplingCursor{12, 3, 0},
      "request without a seed lost its configured default");
  schedule.sampling_seed = 0;
  require(i::initial_sampling_cursor(schedule, 12, 4) == i::SamplingCursor{0, 4, 0},
      "explicit zero request seed was replaced by the configured default");
  schedule.sampling_seed = 98;
  require(i::initial_sampling_cursor(schedule, 12, 5) == i::SamplingCursor{98, 5, 0},
      "request seed did not initialize its independent cursor");
  require(!i::check_request_schedule(schedule, 1, 2), "exact one-token context was rejected");
  require(i::check_request_schedule(schedule, 0, 2) == Error::kInvalidLimits,
      "missing context was accepted");
  schedule.turns.clear();
  require(i::check_request_schedule(schedule, 4, 2) == Error::kNoTurns,
      "empty schedule was accepted");
  schedule.turns = {{{1}, 1}, {{}, 1}};
  schedule.mode = i::GenerationSessionMode::kLive;
  require(i::check_request_schedule(schedule, 4, 2) == Error::kInvalidLiveTurns,
      "live input accepted pre-supplied future turns");
  schedule.mode = i::GenerationSessionMode::kTurns;
  require(!i::check_request_schedule(schedule, 4, 2), "empty later prompt was rejected");
  schedule.turns[1].prompt = {2};
  require(i::check_request_schedule(schedule, 4, 2) == Error::kToken,
      "out-of-vocabulary token was accepted");
  schedule.turns[1].prompt = {-1};
  require(i::check_request_schedule(schedule, 4, 2) == Error::kToken,
      "negative token was accepted");
  schedule.turns = {{{1}, (std::numeric_limits<std::size_t>::max)()}};
  require(i::check_request_schedule(schedule, 4, 2) == Error::kCapacity,
      "oversized quota overflowed the context check");
  schedule.turns = {{std::vector<std::int32_t>(25, 1), 12},
                    {std::vector<std::int32_t>(20, 1), 12}};
  require(!i::check_request_schedule(schedule, 64, 2),
      "future quotas were incorrectly combined before an early stop could release context");
}

void model_provider_selects_one_owner_and_rejects_ambiguity() {
  const auto a = +[](const std::filesystem::path &source) { return source.filename() == "model-a"; };
  const auto b = +[](const std::filesystem::path &source) { return source.filename() == "model-b"; };
  const auto load = +[](const i::ModelLoadConfig &config) -> std::unique_ptr<i::ModelExecutor> {
    require(config.assets.inputs.filename() == "model-b" && config.execution.context == 128 &&
        config.sampling.random && config.sampling.top_k == 40,
        "provider did not receive independent source, execution and sampling preparation facts");
    return std::make_unique<IngressTestExecutor>(std::make_shared<IngressTestControl>());
  };
  const std::array providers{i::ModelProvider{"a", a, load}, i::ModelProvider{"b", b, load}};
  require(i::select_model_provider(providers, "model-b").name == "b", "model provider selection ignored its input");
  i::ModelLoadConfig config;
  config.assets.inputs = "model-b"; config.execution.context = 128;
  config.sampling.random = true; config.sampling.top_k = 40;
  require(bool(i::load_model(providers, config)), "provider did not load its selected source");
  rejects<std::runtime_error>([&] { (void)i::select_model_provider(providers, "unknown"); });
  const std::array ambiguous{i::ModelProvider{"a", a, load}, i::ModelProvider{"other", a, load}};
  rejects<std::runtime_error>([&] { (void)i::select_model_provider(ambiguous, "model-a"); });
  const std::array invalid{i::ModelProvider{"missing", a, nullptr}};
  rejects<std::invalid_argument>([&] { (void)i::select_model_provider(invalid, "model-a"); });
}

void token_worker_rejects_malformed_frames_without_starting_model() {
  auto control = std::make_shared<IngressTestControl>();
  std::istringstream input("G\t0\t1\t1\nG\t7\t0\t1\nG\t8\t1\t-1\n"
      "G\t10\t1\t1\t0\t2\nG\t11\t1\t\t1\t2\nG\t12\t1\t1\t1\n"
      "g\t13\t256\t1\t1\ng\t14\t\t1\t1\ng\t15\t-1\t1\t1\nC\t9\nQ\n");
  std::ostringstream output;
  i::TokenWorker worker(std::make_unique<IngressTestExecutor>(control), 2, 2, 1, 16, input, output);
  require(worker.run() == 0, "worker protocol did not close cleanly");
  require(output.str() == "READY\t2\t256\t128\t3,4\nE\t0\tprotocol\tinvalid request identity\n"
      "E\t7\tinvalid\tinvalid output limit or token list\n"
      "E\t8\tinvalid\tinvalid output limit or token list\n"
      "E\t10\tinvalid\tinvalid output limit or token list\n"
      "E\t11\tinvalid\tinvalid output limit or token list\n"
      "E\t12\tprotocol\tunknown or malformed command\n"
      "E\t13\tinvalid\tinvalid output limit or token list\n"
      "E\t14\tinvalid\tinvalid output limit or token list\n"
      "E\t15\tinvalid\tinvalid output limit or token list\nC\t9\tstale\nBYE\n",
      "worker protocol accepted malformed input or changed response order");
  std::lock_guard lock(control->mutex);
  require(control->batches.empty(), "malformed worker frame reached the model");
}

class WorkerTestInput final : public std::streambuf {
public:
  void send(std::string_view value) {
    std::lock_guard lock(mutex_);
    for (const auto ch : value) bytes_.push_back(ch);
    changed_.notify_all();
  }
  void close() {
    std::lock_guard lock(mutex_); closed_ = true; changed_.notify_all();
  }
protected:
  int_type underflow() override {
    std::unique_lock lock(mutex_);
    changed_.wait(lock, [&] { return closed_ || !bytes_.empty(); });
    if (bytes_.empty()) return traits_type::eof();
    current_ = bytes_.front(); bytes_.pop_front();
    setg(&current_, &current_, &current_ + 1);
    return traits_type::to_int_type(current_);
  }
private:
  std::mutex mutex_;
  std::condition_variable changed_;
  std::deque<char> bytes_;
  char current_ = 0;
  bool closed_ = false;
};

class WorkerTestOutput final : public std::streambuf {
public:
  bool line(std::string &value, std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex_);
    if (!changed_.wait_for(lock, timeout, [&] { return bytes_.find('\n') != std::string::npos; })) return false;
    const auto end = bytes_.find('\n'); value = bytes_.substr(0, end); bytes_.erase(0, end + 1); return true;
  }
protected:
  int_type overflow(int_type value) override {
    if (traits_type::eq_int_type(value, traits_type::eof())) return traits_type::not_eof(value);
    const auto ch = traits_type::to_char_type(value); append(&ch, 1); return value;
  }
  std::streamsize xsputn(const char *data, std::streamsize size) override {
    append(data, static_cast<std::size_t>(size)); return size;
  }
  int sync() override { return 0; }
private:
  void append(const char *data, std::size_t size) {
    std::lock_guard lock(mutex_); bytes_.append(data, size); changed_.notify_all();
  }
  std::mutex mutex_;
  std::condition_variable changed_;
  std::string bytes_;
};

void token_worker_credit_and_cancellation_do_not_hang() {
  auto control = std::make_shared<IngressTestControl>(); control->proceed = true;
  WorkerTestInput input_buffer; WorkerTestOutput output_buffer;
  std::istream input(&input_buffer); std::ostream output(&output_buffer);
  i::TokenWorker worker(std::make_unique<IngressTestExecutor>(control), 1, 4, 1, 16, input, output, 987);
  int result = -1;
  std::thread runner([&] { result = worker.run(); });
  try {
    auto next = [&] {
      std::string line;
      if (!output_buffer.line(line, std::chrono::seconds(3))) throw std::runtime_error("worker test timed out");
      return line;
    };
    require(next() == "READY\t2\t256\t128\t3,4", "worker did not become ready");
    input_buffer.send("G\t1\t2\t10\n");
    require(next() == "A\t1" && next() == "T\t1\t10", "worker lost initial accepted token");
    std::string premature;
    require(!output_buffer.line(premature, std::chrono::milliseconds(30)), "worker bypassed output credit");
    input_buffer.send("K\t1\t2\n");
    require(next() == "E\t1\tprotocol\tinvalid output acknowledgement", "worker accepted excess credit");
    input_buffer.send("K\t1\t1\n");
    require(next() == "T\t1\t11", "worker did not resume after acknowledgement");
    input_buffer.send("K\t1\t1\n");
    require(next() == "F\t1\tcompleted", "worker lost normal terminal after acknowledgement");
    input_buffer.send("G\t4\t2\t40\t1\t\t2\t50\t1\t\n");
    require(next() == "A\t4", "worker did not admit a finite multi-turn session");
    for (const auto token : {40, 41, 0, 50, 51, 0}) {
      require(next() == "T\t4\t" + std::to_string(token), "worker reordered a session's turn outputs");
      input_buffer.send("K\t4\t1\n");
    }
    require(next() == "F\t4\tcompleted", "worker lost the multi-turn terminal");
    input_buffer.send("S\t5\t1\t60\t1\t\n");
    require(next() == "A\t5" && next() == "T\t5\t60", "worker lost the first session token");
    input_buffer.send("K\t5\t1\n");
    require(next() == "B\t5\t0", "worker lost the first turn boundary");
    input_buffer.send("K\t5\t1\n");
    require(next() == "T\t5\t0", "worker reordered the next turn token");
    input_buffer.send("K\t5\t1\n");
    require(next() == "B\t5\t1", "worker lost the final turn boundary");
    input_buffer.send("K\t5\t1\n");
    require(next() == "F\t5\tcompleted", "worker ended before the final boundary");
    input_buffer.send("O\t6\t1\t70\n");
    require(next() == "A\t6" && next() == "T\t6\t70", "worker lost live initial token");
    input_buffer.send("K\t6\t1\n");
    require(next() == "B\t6\t0", "worker lost live turn boundary");
    input_buffer.send("I\t6\t1\t1\t999\n");
    require(next() == "I\t6\t1\tinvalid", "worker accepted a live token outside the vocabulary");
    input_buffer.send("I\t6\t2\t1\t80\n");
    require(next() == "I\t6\t2\taccepted", "worker did not acknowledge the appended turn");
    input_buffer.send("Z\t6\t3\n");
    require(next() == "Z\t6\t3\taccepted", "worker did not acknowledge input closure");
    input_buffer.send("K\t6\t1\n");
    require(next() == "T\t6\t80", "live input overtook its boundary");
    input_buffer.send("K\t6\t1\n");
    require(next() == "B\t6\t1", "worker lost the appended turn boundary");
    input_buffer.send("K\t6\t1\n");
    require(next() == "F\t6\tcompleted", "closed live session lost its terminal");
    input_buffer.send("G\t2\t2\t20\n");
    require(next() == "A\t2" && next() == "T\t2\t20", "worker lost second request");
    input_buffer.send("C\t2\n");
    bool acknowledged = false, terminal = false;
    while (!acknowledged || !terminal) {
      const auto line = next();
      if (line == "C\t2\taccepted") acknowledged = true;
      else if (line == "F\t2\tcancelled" || line == "F\t2\tcompleted") terminal = true;
      else throw std::runtime_error("unexpected worker cancellation event");
    }
    input_buffer.send("G\t3\t2\t30\n");
    require(next() == "A\t3" && next() == "T\t3\t30", "worker lost shutdown fixture");
    input_buffer.send("Q\n");
    bool saw_terminal = false;
    for (;;) {
      const auto line = next();
      if (line == "BYE") break;
      if (line == "F\t3\tcancelled" || line == "F\t3\tcompleted" || line == "F\t3\tclosed")
        saw_terminal = true;
      else throw std::runtime_error("unexpected worker shutdown event");
    }
    require(saw_terminal, "worker shutdown dropped its active request");
    runner.join();
    require(result == 0, "worker exited with a failure");
    { std::lock_guard lock(control->mutex);
      require(!control->seeds.empty() && std::ranges::all_of(control->seeds,
          [](const auto seed) { return seed == 987; }), "worker did not attach its default seed to each request"); }
  } catch (...) {
    input_buffer.close(); runner.join(); throw;
  }
}

void ingress_accepts_requests_during_execution_and_preserves_identity() {
  auto control = std::make_shared<IngressTestControl>();
  i::GenerationDispatcher dispatcher(std::make_unique<IngressTestExecutor>(control), 2, 2, 1);
  auto first = dispatcher.submit({"first", {{{10}, 2}}});
  {
    std::unique_lock lock(control->mutex);
    control->changed.wait(lock, [&] { return control->started; });
  }
  auto second = dispatcher.submit({"second", {{{20}, 2}}});
  auto discarded = dispatcher.submit({"discarded", {{{30}, 1}}});
  rejects<std::runtime_error>([&] { (void)dispatcher.submit({"overflow", {{{50}, 1}}}); });
  require(discarded.cancel() && discarded.read().kind == i::GenerationRequestHandle::EventKind::kCancelled,
      "queued cancellation did not finish without model admission");
  auto third = dispatcher.submit({"third", {{{40}, 1}}});
  {
    std::lock_guard lock(control->mutex); control->proceed = true; control->changed.notify_all();
  }
  using Kind = i::GenerationRequestHandle::EventKind;
  auto read_token = [](auto &request, std::int32_t expected) {
    const auto event = request.read();
    require(event.kind == Kind::kToken && event.token == expected, "request received another request's token");
  };
  read_token(first, 10); read_token(first, 11);
  require(first.read().kind == Kind::kCompleted, "first request lost completion");
  read_token(second, 20); read_token(second, 21);
  require(second.read().kind == Kind::kCompleted, "second request lost completion");
  read_token(third, 40);
  require(third.read().kind == Kind::kCompleted && discarded.read().kind == Kind::kCancelled,
      "peer or cancelled request lost its terminal identity");
  {
    std::lock_guard lock(control->mutex);
    require(control->batches == std::vector<std::size_t>{1, 2},
        "requests arriving during execution were not admitted as the next cohort");
  }
  dispatcher.shutdown();
  rejects<std::runtime_error>([&] { (void)dispatcher.submit({"closed", {{{1}, 1}}}); });
}

void ingress_cancels_active_request_without_stopping_peer() {
  auto control = std::make_shared<IngressTestControl>();
  i::GenerationDispatcher dispatcher(std::make_unique<IngressTestExecutor>(control), 2, 2, 1);
  auto first = dispatcher.submit({"first", {{{10}, 2}}});
  {
    std::unique_lock lock(control->mutex);
    control->changed.wait(lock, [&] { return control->started; });
  }
  auto peer = dispatcher.submit({"peer", {{{20}, 1}}});
  require(first.cancel(), "active request cancellation was not accepted");
  {
    std::lock_guard lock(control->mutex); control->proceed = true; control->changed.notify_all();
  }
  using Kind = i::GenerationRequestHandle::EventKind;
  require(first.read().kind == Kind::kCancelled, "active cancellation became completion");
  const auto item = peer.read();
  require(item.kind == Kind::kToken && item.token == 20 && peer.read().kind == Kind::kCompleted,
      "active cancellation prevented a later peer from finishing");
  dispatcher.shutdown();
}

void live_ingress_appends_and_closes_without_losing_wakes() {
  auto control = std::make_shared<IngressTestControl>(); control->early_stop_first = true;
  i::GenerationDispatcher dispatcher(std::make_unique<IngressTestExecutor>(control), 1, 2, 1);
  auto session = dispatcher.submit({"live", {{{10}, 128}}, i::GenerationSessionMode::kLive});
  using Input = i::GenerationRequestHandle::InputResult;
  using Event = i::GenerationRequestHandle::EventKind;
  require(session.append({{}, 1}) == Input::kCapacity && session.append({{-1}, 1}) == Input::kInvalid &&
      session.append({{1}, 0}) == Input::kInvalid, "live ingress accepted an invalid or over-capacity turn");
  { std::lock_guard lock(control->mutex); control->proceed = true; control->changed.notify_all(); }
  require(session.read().kind == Event::kToken, "live session lost its first token");
  const auto first = session.read();
  require(first.kind == Event::kTurnEnd && first.turn == 0, "live session lost its first boundary");
  require(session.append({{20}, 2}) == Input::kAccepted && session.append({{}, 1}) == Input::kAccepted &&
      session.close_input() && !session.close_input() && session.append({{1}, 1}) == Input::kClosed,
      "live input did not release unused quota, queue turns or close exactly once");
  const auto a = session.read(), b = session.read(), boundary = session.read();
  require(a.kind == Event::kToken && a.token == 20 && b.kind == Event::kToken && b.token == 21 &&
      boundary.kind == Event::kTurnEnd && boundary.turn == 1, "queued live turn lost its order");
  const auto c = session.read(), last = session.read();
  require(c.kind == Event::kToken && c.token == 0 && last.kind == Event::kTurnEnd && last.turn == 2 &&
      session.read().kind == Event::kCompleted, "closed live input lost its queued turn or terminal");

  auto cancelled = dispatcher.submit({"cancel", {{{4}, 1}}, i::GenerationSessionMode::kLive});
  require(cancelled.read().kind == Event::kToken && cancelled.read().kind == Event::kTurnEnd &&
      cancelled.cancel() && cancelled.read().kind == Event::kCancelled &&
      cancelled.append({{1}, 1}) == Input::kClosed, "idle live cancellation did not wake and retire");
  auto later = dispatcher.submit({"later", {{{30}, 1}}});
  require(later.read().token == 30 && later.read().kind == Event::kCompleted,
      "closed live session prevented a later request");
  dispatcher.shutdown();
}

void live_ingress_shutdown_wakes_idle_owner() {
  auto control = std::make_shared<IngressTestControl>(); control->proceed = true;
  i::GenerationDispatcher dispatcher(std::make_unique<IngressTestExecutor>(control), 1, 2, 1);
  auto session = dispatcher.submit({"live", {{{10}, 1}}, i::GenerationSessionMode::kLive});
  using Event = i::GenerationRequestHandle::EventKind;
  require(session.read().kind == Event::kToken && session.read().kind == Event::kTurnEnd,
      "idle shutdown fixture did not finish its first turn");
  dispatcher.shutdown();
  const auto terminal = session.read();
  require(terminal.kind == Event::kCancelled || terminal.kind == Event::kClosed,
      "shutdown reported an open live session as completed");
}

void dynamic_ingress_runs_new_requests_beside_paused_session() {
  i::GenerationDispatcher dispatcher(std::make_unique<DynamicIngressTestExecutor>(), 2, 4, 1);
  using Event = i::GenerationRequestHandle::EventKind;
  using Input = i::GenerationRequestHandle::InputResult;
  auto live = dispatcher.submit({"live", {{{1}, 1}}, i::GenerationSessionMode::kLive});
  require(live.read().kind == Event::kToken && live.read().kind == Event::kTurnEnd,
      "dynamic fixture did not pause its original live session");
  auto invalid = dispatcher.submit({"invalid", {{{-1}, 1}}});
  const auto rejection = invalid.read();
  require(rejection.kind == Event::kFailed && rejection.failure, "invalid new arrival was not isolated");
  bool is_rejection = false;
  try { std::rethrow_exception(rejection.failure); } catch (const i::RequestRejected &) { is_rejection = true; }
  require(is_rejection, "new admission lost its typed rejection");
  auto incompatible = dispatcher.submit({"different-sampling", {{{30}, 1}}}, {}, 1);
  std::array peers{dispatcher.submit({"first-peer", {{{20}, 1}}}),
                   dispatcher.submit({"second-peer", {{{20}, 1}}})};
  for (auto &peer : peers) {
    const auto token = peer.read();
    require(token.kind == Event::kToken && token.token == 11 && peer.read().kind == Event::kCompleted,
        "compatible requests did not pass a differently sampled waiter in arrival order");
  }
  require(incompatible.cancel() && incompatible.read().kind == Event::kCancelled,
      "skipping an incompatible waiter lost its cancellation identity");
  require(live.append({{2}, 1}) == Input::kAccepted && live.close_input(), "original session could not resume");
  const auto token = live.read(), boundary = live.read();
  require(token.kind == Event::kToken && token.token == 2 && boundary.kind == Event::kTurnEnd && boundary.turn == 1 &&
      live.read().kind == Event::kCompleted, "new arrivals changed the original session's ordered continuation");
  dispatcher.shutdown();
}

void abandoned_dynamic_request_releases_slot_beside_live_session() {
  i::GenerationDispatcher dispatcher(std::make_unique<DynamicIngressTestExecutor>(), 2, 4, 1);
  using Event = i::GenerationRequestHandle::EventKind;
  auto live = dispatcher.submit({"live", {{{1}, 1}}, i::GenerationSessionMode::kLive});
  require(live.read().kind == Event::kToken && live.read().kind == Event::kTurnEnd,
      "abandonment fixture did not pause its live session");
  {
    auto abandoned = dispatcher.submit({"abandoned", {{{20}, 2}}});
    require(abandoned.read().kind == Event::kToken, "abandonment fixture did not start its peer");
  }
  auto replacement = dispatcher.submit({"replacement", {{{30}, 1}}});
  require(replacement.read().kind == Event::kToken && replacement.read().kind == Event::kCompleted,
      "abandoned request retained the only available slot");
  require(live.close_input() && live.read().kind == Event::kCompleted,
      "abandoned peer changed the original live session");
  dispatcher.shutdown();
}

void late_turn_rejection_preserves_other_requests() {
  i::GenerationDispatcher dispatcher(std::make_unique<DynamicIngressTestExecutor>(), 2, 4, 1);
  using Event = i::GenerationRequestHandle::EventKind;
  using Input = i::GenerationRequestHandle::InputResult;
  auto anchor = dispatcher.submit({"anchor", {{{1}, 1}}, i::GenerationSessionMode::kLive});
  require(anchor.read().kind == Event::kToken && anchor.read().kind == Event::kTurnEnd,
      "late rejection fixture did not pause its anchor");
  auto rejected = dispatcher.submit({"rejected", {{{2}, 1}}, i::GenerationSessionMode::kLive});
  require(rejected.read().kind == Event::kToken && rejected.read().kind == Event::kTurnEnd &&
      rejected.append({{250}, 1}) == Input::kAccepted,
      "late rejection fixture did not submit its next turn");
  const auto failed = rejected.read();
  require(failed.kind == Event::kFailed && failed.failure, "late rejection was not delivered after retirement");
  bool typed = false;
  try { std::rethrow_exception(failed.failure); } catch (const i::RequestRejected &) { typed = true; }
  require(typed, "late rejection lost its request-specific error");
  auto peer = dispatcher.submit({"peer", {{{3}, 1}}});
  require(peer.read().kind == Event::kToken && peer.read().kind == Event::kCompleted,
      "late rejection blocked or poisoned an unrelated request");
  require(anchor.close_input() && anchor.read().kind == Event::kCompleted,
      "late rejection interrupted the other live session");
  dispatcher.shutdown();
}

void ingress_abandoned_handle_releases_its_request() {
  auto control = std::make_shared<IngressTestControl>();
  i::GenerationDispatcher dispatcher(std::make_unique<IngressTestExecutor>(control), 1, 2, 1);
  {
    auto abandoned = dispatcher.submit({"abandoned", {{{10}, 2}}});
    std::unique_lock lock(control->mutex);
    control->changed.wait(lock, [&] { return control->started; });
  } // The last client handle requests cancellation without fabricating retirement.
  auto peer = dispatcher.submit({"peer", {{{20}, 1}}});
  {
    std::lock_guard lock(control->mutex); control->proceed = true; control->changed.notify_all();
  }
  using Kind = i::GenerationRequestHandle::EventKind;
  const auto token = peer.read();
  require(token.kind == Kind::kToken && token.token == 20 && peer.read().kind == Kind::kCompleted,
      "abandoned request permanently blocked its peer");
  dispatcher.shutdown();
}

void ingress_shutdown_does_not_fabricate_active_retirement() {
  auto control = std::make_shared<IngressTestControl>();
  i::GenerationDispatcher dispatcher(std::make_unique<IngressTestExecutor>(control), 1, 2, 1);
  auto active = dispatcher.submit({"active", {{{10}, 2}}});
  {
    std::unique_lock lock(control->mutex);
    control->changed.wait(lock, [&] { return control->started; });
  }
  auto pending = dispatcher.submit({"pending", {{{20}, 1}}});
  std::thread stopper([&] { dispatcher.shutdown(); });
  const auto pending_event = pending.read();
  const auto active_event = active.read();
  {
    std::lock_guard lock(control->mutex); control->proceed = true; control->changed.notify_all();
  }
  stopper.join();
  using Kind = i::GenerationRequestHandle::EventKind;
  require(pending_event.kind == Kind::kClosed && active_event.kind == Kind::kClosed,
      "shutdown reported an unretired or unstarted request as completed");
}

void ingress_rejects_bad_request_without_poisoning_model() {
  auto control = std::make_shared<IngressTestControl>();
  i::GenerationDispatcher dispatcher(std::make_unique<IngressTestExecutor>(control), 2, 2, 1);
  auto invalid = dispatcher.submit({"invalid", {{{-1}, 1}}});
  const auto rejected = invalid.read();
  require(rejected.kind == i::GenerationRequestHandle::EventKind::kFailed && rejected.failure,
      "invalid request did not report its admission rejection");
  bool exact = false;
  try { std::rethrow_exception(rejected.failure); } catch (const i::RequestRejected &) { exact = true; }
  require(exact, "invalid request lost its admission-rejection identity");
  auto valid = dispatcher.submit({"valid", {{{20}, 1}}});
  {
    std::unique_lock lock(control->mutex);
    control->changed.wait(lock, [&] { return control->started; });
    control->proceed = true; control->changed.notify_all();
  }
  const auto token = valid.read();
  require(token.kind == i::GenerationRequestHandle::EventKind::kToken && token.token == 20 &&
      valid.read().kind == i::GenerationRequestHandle::EventKind::kCompleted,
      "an admission rejection poisoned a later valid request");
  dispatcher.shutdown();
}

void ingress_rejection_preserves_later_peer_slot() {
  auto control = std::make_shared<IngressTestControl>();
  i::GenerationDispatcher dispatcher(std::make_unique<IngressTestExecutor>(control), 2, 3, 1);
  auto first = dispatcher.submit({"first", {{{10}, 1}}});
  {
    std::unique_lock lock(control->mutex);
    control->changed.wait(lock, [&] { return control->started; });
  }
  auto invalid = dispatcher.submit({"invalid", {{{-1}, 1}}});
  auto valid = dispatcher.submit({"valid", {{{20}, 1}}});
  {
    std::lock_guard lock(control->mutex); control->proceed = true; control->changed.notify_all();
  }
  using Kind = i::GenerationRequestHandle::EventKind;
  require(first.read().kind == Kind::kToken && first.read().kind == Kind::kCompleted,
      "first request did not finish");
  const auto rejected = invalid.read();
  const auto token = valid.read();
  require(rejected.kind == Kind::kFailed && rejected.failure && token.kind == Kind::kToken && token.token == 20 &&
      valid.read().kind == Kind::kCompleted, "rejected cohort member displaced a valid peer");
  {
    std::lock_guard lock(control->mutex);
    require(control->batches == std::vector<std::size_t>{1, 1},
        "rejected member was submitted to the model");
  }
  dispatcher.shutdown();
}

void ingress_rejects_foreign_terminal_without_hanging() {
  auto control = std::make_shared<IngressTestControl>(); control->wrong_terminal = true;
  i::GenerationDispatcher dispatcher(std::make_unique<IngressTestExecutor>(control), 1, 2, 1);
  auto request = dispatcher.submit({"wrong-terminal", {{{10}, 1}}});
  {
    std::unique_lock lock(control->mutex);
    control->changed.wait(lock, [&] { return control->started; });
    control->proceed = true; control->changed.notify_all();
  }
  using Kind = i::GenerationRequestHandle::EventKind;
  const auto token = request.read();
  const auto terminal = request.read();
  require(token.kind == Kind::kToken && token.token == 10 && terminal.kind == Kind::kFailed && terminal.failure,
      "foreign terminal was accepted or left the reader blocked");
  dispatcher.shutdown();
}

void ingress_propagates_model_failure_without_inventing_completion() {
  auto control = std::make_shared<IngressTestControl>(); control->fail = true;
  i::GenerationDispatcher dispatcher(std::make_unique<IngressTestExecutor>(control), 1, 2, 1);
  auto first = dispatcher.submit({"first", {{{10}, 1}}});
  {
    std::unique_lock lock(control->mutex);
    control->changed.wait(lock, [&] { return control->started; });
  }
  auto waiting = dispatcher.submit({"waiting", {{{20}, 1}}});
  {
    std::lock_guard lock(control->mutex); control->proceed = true; control->changed.notify_all();
  }
  using Kind = i::GenerationRequestHandle::EventKind;
  const auto a = first.read(), b = waiting.read();
  require(a.kind == Kind::kFailed && b.kind == Kind::kFailed && a.failure && b.failure,
      "model failure was treated as a completed request");
  bool exact = false;
  try { std::rethrow_exception(a.failure); } catch (const std::runtime_error &error) {
    exact = std::string_view(error.what()) == "scripted model failure";
  }
  require(exact, "ingress lost the original model failure");
  rejects<std::runtime_error>([&] { (void)dispatcher.submit({"after", {{{1}, 1}}}); });
  dispatcher.shutdown();
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

// These are scripted facts of the original owner, not another executor. The
// common driver must neither execute during observation nor discard metadata
// merely because an error was returned without actual retirement.
struct ExecutionStage final {
  std::array<i::RequestWavePendingMember, 2> members{};
  std::size_t count = 0, next_count = 0;
  std::size_t advances = 0, starts = 0, resumes = 0, drains = 0, observations = 0, aborts = 0;
  i::RequestWaveAdvance next{true, {true, {}}};
  i::RetirementObservation retirement{true, {}};
  std::exception_ptr failure{};
  bool saw_drain = false;

  std::size_t pending_count() const noexcept { return count; }
  i::RequestWavePendingMember pending_member(std::size_t at) const noexcept { return members[at]; }
  i::RequestWaveAdvance advance(i::RequestWaveAction action, i::RequestWave wave) {
    ++advances; saw_drain = wave.drain_requested;
    if (action == i::RequestWaveAction::kStart) ++starts;
    else if (action == i::RequestWaveAction::kResume) ++resumes;
    else ++drains;
    if (failure) std::rethrow_exception(failure);
    count = next_count;
    for (std::size_t at = 0; at < count; ++at) members[at] = {wave.keys[at], 1};
    return next;
  }
  i::RetirementObservation retire(i::RequestWaveRetirement action) noexcept {
    if (action == i::RequestWaveRetirement::kObserve) ++observations;
    else ++aborts;
    if (retirement.retired) count = 0;
    return retirement;
  }
};

void common_wave_synchronous_uses_no_extra_owner_or_allocation() {
  i::RequestWindow window; window.prepare(1, 1);
  const std::array keys{window.enqueue(0)};
  const std::array<i::RequestCreditGrant, 1> credits{{{0, 1, false}}};
  std::array<std::size_t, 1> remaining{99}; ExecutionStage stage;
  allocation_count = 0; track_allocations = true;
  const auto result = i::advance_request_wave({keys, credits}, remaining, stage);
  track_allocations = false;
  require(result.advanced && result.remaining.retired && result.remaining.status.ok() && !result.failure &&
      !remaining[0] && stage.starts == 1 && !stage.resumes && !stage.observations && !stage.aborts &&
      !allocation_count, "common synchronous wave allocated or added a retirement observer");
}

void common_wave_resumes_same_cohort_and_original_reservations() {
  i::RequestWindow window; window.prepare(2, 2);
  const std::array keys{window.enqueue(0), window.enqueue(1)};
  const std::array<i::RequestCreditGrant, 2> initial{{{0, 2, false}, {0, 2, false}}};
  std::array<std::size_t, 2> remaining{}; ExecutionStage stage;
  stage.next_count = 2; stage.next = {true, {false, {}}};
  const auto submitted = i::advance_request_wave({keys, initial}, remaining, stage);
  require(submitted.advanced && !submitted.remaining.retired && !submitted.failure &&
      remaining == std::array<std::size_t, 2>{1, 1} && stage.count == 2,
      "common wave did not report the original pending member reservations");
  const std::array<i::RequestCreditGrant, 2> held{{{1, 0, false}, {1, 0, false}}};
  stage.next_count = 0; stage.next = {true, {true, {}}};
  const auto completed = i::advance_request_wave({keys, held}, remaining, stage);
  require(completed.advanced && completed.remaining.retired && !completed.failure &&
      remaining == std::array<std::size_t, 2>{0, 0} && stage.starts == 1 && stage.resumes == 1 &&
      !stage.aborts && !stage.observations, "common wave restarted pending work or lost its held credit");
}

void common_wave_drain_and_observation_never_start_new_work() {
  i::RequestWindow window; window.prepare(1, 1);
  const std::array keys{window.enqueue(0)};
  const std::array<i::RequestCreditGrant, 1> held{{{1, 0, true}}}, empty{{{0, 0, true}}};
  std::array<std::size_t, 1> remaining{}; ExecutionStage stage;
  stage.count = 1; stage.members[0] = {keys[0], 1};
  const auto drained = i::advance_request_wave({keys, held, false, true}, remaining, stage);
  require(drained.advanced && drained.remaining.retired && !drained.failure && stage.saw_drain &&
      !stage.starts && stage.resumes == 1 && !remaining[0], "drain restarted work instead of resuming its original call");
  stage.next = {false, {true, {}}};
  const auto idle = i::advance_request_wave({keys, empty, false, true}, remaining, stage);
  require(!idle.advanced && idle.remaining.retired && !idle.failure && stage.drains == 1 &&
      !stage.starts, "an idle drain became a new submission");
  stage.retirement = {false, {}};
  const auto observing = i::advance_request_wave({keys, empty, true, true}, remaining, stage);
  require(!observing.advanced && !observing.remaining.retired && !observing.failure &&
      stage.advances == 2 && stage.observations == 1 && !stage.aborts,
      "observation-only execution advanced work or pretended pending work retired");
}

void common_wave_failure_keeps_exception_and_actual_retirement_independent() {
  i::RequestWindow window; window.prepare(1, 1);
  const std::array keys{window.enqueue(0)};
  const std::array<i::RequestCreditGrant, 1> held{{{1, 0, false}}};
  std::array<std::size_t, 1> remaining{1}; ExecutionStage stage;
  stage.count = 1; stage.members[0] = {keys[0], 1};
  stage.failure = std::make_exception_ptr(WaveFailure{});
  const auto native = hyle::Status::FromNative(hyle::StatusCode::Internal,
      {hyle::NativeErrorDomain::CudaDriver, 999}, "original retirement failure");
  stage.retirement = {false, native};
  const auto failed = i::advance_request_wave({keys, held}, remaining, stage);
  // MSVC copies the exception while rethrowing and capturing it again. Check
  // its semantic identity rather than requiring the same exception_ptr owner.
  bool original_failure = false;
  if (failed.failure) try { std::rethrow_exception(failed.failure); }
  catch (const WaveFailure &error) {
    original_failure = error.payload == WaveFailure::expected_payload &&
        std::string_view(error.what()) == "scripted failure after committed A";
  } catch (...) {}
  require(failed.advanced && original_failure && !failed.remaining.retired &&
      failed.remaining.status == native && stage.count == 1 && stage.aborts == 1 && !remaining[0],
      "common wave replaced the original exception/status or cleared unretired metadata");
  const auto unretired = i::advance_request_wave({keys, held, true}, remaining, stage);
  require(!unretired.advanced && !unretired.failure && !unretired.remaining.retired &&
      unretired.remaining.status == native && stage.count == 1 && stage.advances == 1,
      "failure observation reexecuted a call or converted error into release permission");
  stage.retirement.retired = true;
  const auto retired = i::advance_request_wave({keys, held, true}, remaining, stage);
  require(!retired.advanced && retired.remaining.retired && retired.remaining.status == native &&
      !retired.failure && !stage.count && stage.advances == 1 && stage.observations == 2,
      "actual retired failure lost its status or failed to release the original metadata");
}

void common_wave_rejects_foreign_reordered_or_unbacked_pending_credit() {
  i::RequestWindow window, foreign; window.prepare(2, 2); foreign.prepare(2, 2);
  const std::array keys{window.enqueue(0), window.enqueue(1)};
  const std::array foreign_keys{foreign.enqueue(0), foreign.enqueue(1)};
  const std::array<i::RequestCreditGrant, 2> held{{{1, 0, false}, {1, 0, false}}};
  std::array<std::size_t, 2> remaining{}; ExecutionStage stage;
  stage.count = 2; stage.members = {{{keys[0], 1}, {keys[1], 1}}}; stage.retirement = {false, {}};
  const auto rejected = [&](std::span<const i::RequestKey> incoming,
      std::span<const i::RequestCreditGrant> grants) {
    const auto result = i::advance_request_wave({incoming, grants}, remaining, stage);
    require(result.failure && !result.remaining.retired && !stage.advances && stage.count == 2 &&
        remaining == std::array<std::size_t, 2>{0, 0}, "invalid cohort advanced work or discarded original metadata");
    bool exact = false;
    try { std::rethrow_exception(result.failure); } catch (const std::logic_error &) { exact = true; }
    require(exact, "invalid cohort lost its control error identity");
  };
  rejected(foreign_keys, held);
  const std::array reordered{keys[1], keys[0]}; rejected(reordered, held);
  auto wrong = held; wrong[1].reserved = 0; rejected(keys, wrong);
  require(stage.aborts == 3, "invalid cohort did not use the original abort owner");
  rejects<std::logic_error>([&] { (void)i::advance_request_wave({keys, held}, std::span(remaining).first(1), stage); });
  require(stage.aborts == 3 && !stage.advances, "malformed extent mutated an original execution owner");
}
}

int main() {
  try {
    fifo_and_headless_reservations();
    preflight_and_actual_eviction_facts();
    deferred_terminal_keeps_original_identity_without_execution();
    errors_preserve_proof_and_committed_boundaries();
    open_input_and_midturn_pause_keep_progress();
    paused_session_allows_reuse_of_actually_retired_peer_slot();
    generic_token_delivery_preserves_peer_and_retirement();
    generic_token_delivery_catches_cancel_during_write();
    delivery_orders_turn_boundaries_and_retirement();
    turn_boundary_queue_is_bounded_and_keyed();
    queued_delivery_preserves_bounds_and_terminal_identity();
    transport_slot_waits_for_terminal_observation_before_reuse();
    abandoned_transport_waits_for_actual_retirement();
    new_admission_skips_unreleased_transport_after_model_retirement();
    schedule_admission_is_model_independent_and_overflow_safe();
    model_provider_selects_one_owner_and_rejects_ambiguity();
    token_worker_rejects_malformed_frames_without_starting_model();
    token_worker_credit_and_cancellation_do_not_hang();
    ingress_accepts_requests_during_execution_and_preserves_identity();
    ingress_cancels_active_request_without_stopping_peer();
    live_ingress_appends_and_closes_without_losing_wakes();
    live_ingress_shutdown_wakes_idle_owner();
    dynamic_ingress_runs_new_requests_beside_paused_session();
    abandoned_dynamic_request_releases_slot_beside_live_session();
    late_turn_rejection_preserves_other_requests();
    ingress_abandoned_handle_releases_its_request();
    ingress_shutdown_does_not_fabricate_active_retirement();
    ingress_rejects_bad_request_without_poisoning_model();
    ingress_rejection_preserves_later_peer_slot();
    ingress_rejects_foreign_terminal_without_hanging();
    ingress_propagates_model_failure_without_inventing_completion();
    prepared_hot_owner_allocates_nothing();
    in_flight_waves_keep_exact_credit_and_allow_continuation();
    pending_cohort_defers_cancel_eos_and_admission();
    arrival_during_native_work_waits_for_original_retirement();
    failure_observation_is_once_and_cleanup_cannot_resubmit();
    invalid_wave_reservation_retains_real_retirement_gate();
    explicit_yield_drains_then_preserves_original_fifo_and_turn();
    common_wave_synchronous_uses_no_extra_owner_or_allocation();
    common_wave_resumes_same_cohort_and_original_reservations();
    common_wave_drain_and_observation_never_start_new_work();
    common_wave_failure_keeps_exception_and_actual_retirement_independent();
    common_wave_rejects_foreign_reordered_or_unbacked_pending_credit();
    std::puts("request loop: 30 host contract groups passed; no model/device loaded");
    return 0;
  } catch (const std::exception &error) {
    track_allocations = false;
    std::fprintf(stderr, "request loop: %s\n", error.what()); return 1;
  }
}
