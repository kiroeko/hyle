// Host request semantics only. No model, native route, or test framework.
#include "../../engine/inference/request_progress.h"
#include "../../engine/inference/request_window.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <type_traits>

namespace {
bool track_allocations = false;
std::size_t allocation_count = 0;
}
void *operator new(std::size_t bytes) {
  if (track_allocations) ++allocation_count;
  if (auto *memory = std::malloc(bytes ? bytes : 1)) return memory;
  throw std::bad_alloc{};
}
void *operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void *memory) noexcept { std::free(memory); }
void operator delete[](void *memory) noexcept { std::free(memory); }
void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void *memory, std::size_t) noexcept { std::free(memory); }

namespace {
using hyle::inference::GenerationTurn;
using hyle::inference::RequestProgress;
using hyle::inference::RequestWindow;
static_assert(!std::is_copy_constructible_v<RequestProgress> && !std::is_copy_assignable_v<RequestProgress>);
static_assert(std::is_nothrow_move_constructible_v<RequestProgress> && std::is_nothrow_move_assignable_v<RequestProgress>);
void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
template<class Operation> void rejects(Operation operation) {
  try { operation(); }
  catch (const std::exception &) { return; }
  throw std::runtime_error("invalid request progress transition was accepted");
}

std::vector<std::int32_t> delivered_journey(bool batched) {
  const std::array turns{GenerationTurn{{1, 2}, 3}, GenerationTurn{{3}, 2}, GenerationTurn{{}, 2}};
  RequestProgress progress(7); progress.bind_turns(turns);
  std::array<std::int32_t, 2> first{1, 2};
  require(progress.begin_turn(first).size() == 2, "first turn added an unconsumed pending slot");
  std::size_t ordinal = 0;
  const auto observe = [&](std::span<const std::int32_t> emitted, bool prefill, bool stopped) {
    const auto current = ordinal++;
    require(progress.observe_commit(current, emitted, true, prefill, stopped), "new committed output was not observed");
    const auto size = progress.generated.size();
    require(!progress.observe_commit(current, emitted, true, prefill, stopped) &&
        progress.generated.size() == size, "re-observation duplicated delivered tokens");
  };
  const std::array<std::int32_t, 1> seven{7}, eight{8}, nine{9}, seventeen{17}, eighteen{18}, end{19};
  const std::array<std::int32_t, 2> pair{8, 9};
  observe(seven, true, false);
  if (batched) observe(pair, false, false);
  else { observe(eight, false, false); observe(nine, false, false); }
  require(progress.turn_done && progress.remaining_outputs() == 0 && progress.more_turns(),
      "quota did not finish exactly the first turn");
  progress.advance_turn();
  std::array<std::int32_t, 2> second{-1, 3};
  const auto second_input = progress.begin_turn(second, 9);
  require(second_input.size() == 2 && second_input[0] == 9 && second_input[1] == 3,
      "later turn did not consume the previous pending once before its user suffix");
  observe(seventeen, true, false); observe(eighteen, false, false);
  progress.advance_turn();
  std::array<std::int32_t, 1> third{-1};
  const auto third_input = progress.begin_turn(third, 18);
  require(third_input.size() == 1 && third_input[0] == 18, "empty user suffix lost pending-only continuation");
  observe(end, true, true);
  require(progress.turn_done && progress.remaining_outputs() == 0 &&
      progress.output_limit() - progress.generated.size() == 1 && !progress.more_turns(),
      "explicit stop was incorrectly reduced to quota exhaustion");
  require(progress.generated == std::vector<std::int32_t>({7, 8, 9, 17, 18, 19}),
      "delivered journey changed with batching");
  return progress.generated;
}

void invalid_transitions_are_atomic() {
  const std::array turns{GenerationTurn{{1}, 2}, GenerationTurn{{}, 1}};
  RequestProgress progress(3); progress.bind_turns(turns); progress.generated.reserve(128);
  std::array<std::int32_t, 1> input{1}; progress.begin_turn(input);
  const std::array<std::int32_t, 1> seven{7}, eight{8}, nine{9};
  const std::array<std::int32_t, 2> excess{8, 9};
  rejects([&] { progress.observe_commit(1, seven, true, true, false); });
  require(progress.observed_commits() == 0 && progress.generated.empty() && !progress.prefilled,
      "gap changed unobserved request state");
  rejects([&] { progress.begin_turn(input); });
  require(input[0] == 1, "repeated begin changed the original input");
  progress.observe_commit(0, seven, true, true, false);
  rejects([&] { progress.observe_commit(1, excess, true, false, false); });
  rejects([&] { progress.advance_turn(); });
  require(progress.generated.size() == 1 && progress.generated.back() == 7 &&
      progress.observed_commits() == 1 && !progress.turn_done,
      "quota failure partially advanced progress or used vector capacity as user quota");
  require(!progress.observe_commit(0, excess, true, true, true) && progress.generated.size() == 1,
      "duplicate observation reinterpreted obsolete output facts");
  progress.observe_commit(1, eight, true, false, false); progress.advance_turn();
  std::array<std::int32_t, 1> pending_slot{-1};
  rejects([&] { progress.begin_turn(pending_slot, 77); });
  require(pending_slot[0] == -1 && !progress.turn_started(), "bad pending overwrote the borrowed input before rejection");
  progress.begin_turn(pending_slot, 8);
  rejects([&] { progress.apply_outputs(excess, true, true, false); });
  require(pending_slot[0] == 8 && progress.generated.size() == 2 && progress.observed_commits() == 2,
      "failed final turn output changed original progress");
  progress.observe_commit(2, nine, true, true, false);
  require(progress.turn_done && progress.generated.size() == progress.maximum_outputs(),
      "valid retry did not reach the actual finite output bound");
}

void prefix_output_is_not_another_model_commit() {
  RequestProgress progress(3); progress.generated.reserve(128); progress.begin_turn();
  progress.adopt_committed_history(4);
  const std::array<std::int32_t, 1> pending{11};
  progress.apply_outputs(pending, true, true, false);
  require(progress.generated.size() == 1 && progress.prefilled && progress.observed_commits() == 4,
      "cache-hit output fabricated a graph commit or re-emitted history");
  const std::array<std::int32_t, 2> tail{12, 13};
  require(!progress.observe_commit(3, tail, true, false, false), "imported history was re-observed");
  progress.observe_commit(4, tail, true, false, false);
  require(progress.turn_done && progress.observed_commits() == 5 && progress.generated.size() == 3,
      "ordinary continuation did not follow the imported authoritative history");
  rejects([&] { progress.adopt_committed_history(8); });
  require(progress.observed_commits() == 5 && progress.generated.back() == 13,
      "history adoption overwrote an active output stream");
}

void finite_fifo_hot_transitions_allocate_nothing() {
  const std::array turns{GenerationTurn{{2}, 1}, GenerationTurn{{}, 1}};
  std::array progress{RequestProgress(2), RequestProgress(2)};
  for (auto &request : progress) request.bind_turns(turns);
  RequestWindow window; window.prepare(2, 1);
  const std::array keys{window.enqueue(0), window.enqueue(1)};
  std::array<std::array<std::array<std::int32_t, 1>, 2>, 2> inputs{};
  for (auto &request : inputs) request[0][0] = 2;
  const std::array<std::size_t, 4> order{0, 1, 0, 1};
  allocation_count = 0; track_allocations = true;
  for (const auto index : order) {
    auto &request = progress[index];
    require(window.waiting_front() == keys[index], "finite turn FIFO changed");
    const auto turn = request.turn_index;
    request.begin_turn(inputs[index][turn], turn ? std::optional(request.generated.back()) : std::nullopt);
    window.admitted(keys[index]);
    const std::array emitted{static_cast<std::int32_t>(10 + 100 * index + turn)};
    if (index == 1 && turn == 0) {
      request.adopt_committed_history(7);
      request.apply_outputs(emitted, true, true, false);
    } else request.observe_commit(request.observed_commits(), emitted, true, true, false);
    const auto more = request.more_turns();
    require(request.turn_done, "bounded turn was not complete at its actual output quota");
    window.release_turn(keys[index], more);
    if (!more) window.release(keys[index]);
    if (more) request.advance_turn();
  }
  track_allocations = false;
  require(allocation_count == 0, "valid cold-prepared request progress allocated on the hot path");
  require(window.active().empty() && !window.waiting_front() &&
      progress[0].observed_commits() == 2 && progress[1].observed_commits() == 8 &&
      inputs[0][1][0] == 10 && inputs[1][1][0] == 110,
      "mixed cache/ordinary turns lost actual pending or commit counts");
}
} // namespace

int main() {
  try {
    require(delivered_journey(false) == delivered_journey(true), "batching changed host delivery semantics");
    std::puts("PASS ordinary/batched outputs, quota/stop and pending-only turn continuation");
    invalid_transitions_are_atomic();
    std::puts("PASS gap/duplicate/quota/begin failures and side-effect-free retry");
    prefix_output_is_not_another_model_commit();
    std::puts("PASS imported history, no-Graph output and authoritative commit counts");
    finite_fifo_hot_transitions_allocate_nothing();
    std::puts("PASS mixed two-user FIFO turns and allocation-free prepared transitions");
    return 0;
  } catch (const std::exception &error) {
    track_allocations = false; std::fprintf(stderr, "%s\n", error.what()); return 1;
  }
}
