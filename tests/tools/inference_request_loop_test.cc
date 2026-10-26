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
  std::array<std::array<std::int32_t, 4>, 4> inputs{};
  std::array<std::size_t, 4> executions{}, publications{}, retirement_polls{}, eviction_polls{};
  std::array<i::RequestKey, 32> admission_order{};
  std::array<std::size_t, 32> active_at_admission{};
  std::size_t admissions = 0, waves = 0, resident_queries = 0;
  std::size_t deferred_terminal = 0, deferred_eviction = 0, physical_bytes = 4096;
  std::optional<std::size_t> reject_preflight, throw_after_commit, retirement_error;
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
    for (auto key : active) {
      auto &p = progress(key); residents[key.slot] = true; ++executions[key.slot];
      const std::array output{static_cast<std::int32_t>(key.slot * 10 + p.generated.size() + 1)};
      require(p.observe_commit(p.observed_commits(), output, true, true, false), "host output was replayed");
      if (throw_after_commit == key.slot) throw std::runtime_error("test observer failed after commit");
    }
  }
  void publish_turn(i::RequestKey key) {
    require(progress(key).turn_done, "owner published an unfinished turn");
    ++publications[key.slot];
  }
  i::RetirementObservation retire(i::RequestKey key, i::RequestRetirement reason) {
    auto &polls = reason == i::RequestRetirement::kCompleted ? retirement_polls[key.slot] : eviction_polls[key.slot];
    ++polls;
    if (retirement_error == key.slot) {
      if (error_is_retired) residents[key.slot] = false;
      return {error_is_retired, error};
    }
    auto &remaining = reason == i::RequestRetirement::kCompleted ? deferred_terminal : deferred_eviction;
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
  require(!owner.matches(other), "foreign window identity matched by slot and generation");
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
}

void open_input_and_midturn_pause_keep_progress() {
  i::RequestLoop owner; owner.prepare(1, 1); Adapter adapter{3};
  const std::array turns{i::GenerationTurn{{1}, 2}, i::GenerationTurn{{}, 1}};
  adapter.progress_values[0]->bind_open_turns(std::span(turns).first(1), 2, 1);
  const auto key = owner.enqueue(0); (void)owner.pump_once(adapter);
  const auto generated = adapter.progress(key).generated;
  const auto commits = adapter.progress(key).observed_commits();
  (void)owner.pause(key);
  require(owner.pump_once(adapter).executed == false && adapter.progress(key).generated == generated &&
      adapter.progress(key).observed_commits() == commits && adapter.progress(key).turn_index == 0,
      "midturn pause changed observed outputs or turn position");
  owner.wake(key); (void)owner.pump_once(adapter);
  require(adapter.progress(key).waiting_for_input() && owner.matches(key) && owner.active().empty() &&
      adapter.publications[0] == 1, "open input was completed or its boundary was not paused");
  adapter.progress(key).extend_turns(turns); owner.wake(key);
  require(adapter.progress(key).turn_index == 0, "wake advanced model-facing progress before admission");
  (void)owner.pump_once(adapter);
  require(adapter.progress(key).turn_index == 1 && adapter.progress(key).generated.size() == 3 &&
      adapter.publications[0] == 2 && owner.matches(key), "actual later input did not resume the same request");
  adapter.progress(key).close_input(); owner.wake(key);
  const auto closed = owner.pump_once(adapter);
  require(closed.completed == 1 && !closed.executed && !owner.matches(key) && adapter.executions[0] == 3 &&
      adapter.publications[0] == 2, "closing paused input restored/reexecuted or republished an old turn");
}

void prepared_hot_owner_allocates_nothing() {
  i::RequestLoop owner; owner.prepare(2, 2); Adapter adapter{3, 3};
  (void)owner.enqueue(0); (void)owner.enqueue(1);
  allocation_count = 0; track_allocations = true;
  for (std::size_t step = 0; step < 3; ++step) (void)owner.pump_once(adapter);
  track_allocations = false;
  require(!allocation_count && adapter.executions[0] == 3 && adapter.executions[1] == 3 &&
      owner.active().empty() && !owner.waiting_front(), "valid prepared owner loop allocated host storage");
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
