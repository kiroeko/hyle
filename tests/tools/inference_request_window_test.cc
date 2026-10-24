// Standard-library-only host window test. No model or device is loaded.
#include "../../engine/inference/request_window.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <new>
#include <stdexcept>
#include <type_traits>

namespace {
bool track_allocations = false;
std::size_t allocation_count = 0;
}

void *operator new(std::size_t bytes) {
  if (track_allocations) ++allocation_count;
  if (auto *result = std::malloc(bytes ? bytes : 1)) return result;
  throw std::bad_alloc{};
}
void *operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void *memory) noexcept { std::free(memory); }
void operator delete[](void *memory) noexcept { std::free(memory); }
void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void *memory, std::size_t) noexcept { std::free(memory); }

namespace {
using hyle::inference::RequestWindow;
static_assert(!std::is_copy_constructible_v<RequestWindow> && !std::is_copy_assignable_v<RequestWindow>);
static_assert(!std::is_move_constructible_v<RequestWindow> && !std::is_move_assignable_v<RequestWindow>);

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
template<class Error, class Operation>
void rejects(Operation operation) {
  try { operation(); }
  catch (const Error &) { return; }
  throw std::runtime_error("invalid request-window transition was accepted");
}
void active_is(const RequestWindow &window, std::initializer_list<std::size_t> expected) {
  const auto active = window.active();
  require(active.size() == expected.size() && std::equal(active.begin(), active.end(), expected.begin()),
      "request-window active order changed");
}

void one_resident_yields_whole_turns_fifo() {
  RequestWindow window;
  require(!window.waiting_front(), "unprepared request window offered admission");
  window.prepare(2, 4, 1);
  const std::array expected{std::size_t{0}, std::size_t{1}, std::size_t{0}, std::size_t{1}};
  for (std::size_t turn = 0; turn < expected.size(); ++turn) {
    require(window.waiting_front() == expected[turn], "N1 did not preserve whole-turn FIFO order");
    window.admitted(expected[turn]);
    require(!window.waiting_front(), "N1 admitted a peer while its only slot was reserved");
    const auto order = window.release_turn(expected[turn], turn < 2);
    require(order == (turn < 2 ? turn + 1 : 0), "whole-turn idle ordering changed");
  }
  require(window.active().empty() && !window.waiting_front(), "completed N1 requests remained runnable");
  // N is a limit, not a requirement that this many distinct requests exist.
  window.prepare(1, 1, 3);
  window.admitted(0);
  window.release_turn(0, false);
  require(!window.waiting_front(), "cold prepare did not start a fresh finite window");
}

void idle_query_uses_actual_residents_and_reserved_slots() {
  RequestWindow window;
  window.prepare(4, 6, 2);
  std::array resident{false, false, true, true};
  std::array<std::size_t, 4> order{0, 0, 7, 5};
  const auto victim = [&] {
    return window.idle_victim([&](auto index) { return resident[index]; },
        [&](auto index) { return order[index]; });
  };
  window.admitted(0); // No Session/head yet, but its active slot is reserved.
  require(victim() == 3, "idle query did not choose the oldest actual resident");
  resident[3] = false;
  window.admitted(1);
  require(!window.waiting_front(), "two headless active reservations exceeded N2");
  resident[0] = true;
  require(victim() == 2, "an active owner was selected for idle eviction");
  resident[2] = false;
  require(!victim(), "idle query invented a resident or selected an active request");
  resident[2] = resident[3] = true;
  order[2] = order[3] = 9;
  require(victim() == 2, "equal idle orders lost their first-index tie break");
  active_is(window, {0, 1});
  require(window.release_turn(0, true) == 1 && window.release_turn(1, true) == 2,
      "completed turns were not assigned increasing idle order");
  const std::array expected{std::size_t{2}, std::size_t{3}, std::size_t{0}, std::size_t{1}};
  for (const auto index : expected) {
    require(window.waiting_front() == index, "requeued turns passed earlier waiting requests");
    window.admitted(index); window.release_turn(index, false);
  }
}

void invalid_transitions_leave_queue_and_active_unchanged() {
  RequestWindow window;
  window.prepare(3, 4, 2);
  rejects<std::invalid_argument>([&] { window.prepare(0, 4, 2); });
  rejects<std::invalid_argument>([&] { window.prepare(3, 2, 2); });
  rejects<std::invalid_argument>([&] { window.prepare(3, 4, 0); });
  require(window.waiting_front() == 0, "invalid cold bounds modified the FIFO head");
  rejects<std::logic_error>([&] { window.admitted(1); });
  rejects<std::out_of_range>([&] { window.admitted(3); });
  rejects<std::logic_error>([&] { window.release_turn(0, true); });
  active_is(window, {});
  require(window.waiting_front() == 0, "invalid admission/release advanced the FIFO");
  window.admitted(0); window.admitted(1);
  rejects<std::logic_error>([&] { window.admitted(2); });
  rejects<std::logic_error>([&] { window.admitted(0); });
  active_is(window, {0, 1});
  window.release_turn(0, true); // The one spare turn is now the FIFO tail.
  require(window.waiting_front() == 2, "failed full admission consumed a waiting request");
  rejects<std::out_of_range>([&] { window.release_turn(1, true); });
  active_is(window, {1});
  require(window.waiting_front() == 2, "overflowing requeue changed the waiting order");
  window.release_turn(1, false);
  for (const auto index : {std::size_t{2}, std::size_t{0}}) {
    require(window.waiting_front() == index, "failed release lost a previously queued turn");
    window.admitted(index); window.release_turn(index, false);
  }
  require(!window.waiting_front(), "rejected transitions appended hidden FIFO entries");
}

void cold_spill_capacity_covers_all_hot_transitions() {
  RequestWindow window;
  window.prepare(7, 21, 5); // Both vectors exceed the prior small N/R examples.
  std::array<std::size_t, 7> turns{};
  std::array<bool, 7> resident{};
  std::array<std::size_t, 7> idle{};
  std::size_t completed = 0, admissions = 0;
  const std::size_t *active_data = nullptr;
  const auto before = allocation_count;
  track_allocations = true;
  try {
    while (completed != turns.size()) {
      while (const auto index = window.waiting_front()) {
        window.admitted(*index); ++admissions;
        if (!active_data) active_data = window.active().data();
        require(window.active().data() == active_data, "prepared active storage moved during admission");
      }
      while (!window.active().empty()) {
        const auto index = window.active().front();
        resident[index] = true;
        const auto more = ++turns[index] != 3;
        idle[index] = window.release_turn(index, more);
        if (!more) { resident[index] = false; ++completed; }
        if (!window.active().empty())
          require(window.active().data() == active_data, "prepared active storage moved during release");
        (void)window.idle_victim([&](auto member) { return resident[member]; },
            [&](auto member) { return idle[member]; });
      }
    }
  } catch (...) { track_allocations = false; throw; }
  track_allocations = false;
  require(admissions == 21 && !window.waiting_front(), "finite spill-capacity turns were lost or duplicated");
  require(allocation_count == before, "valid request-window operations allocated after cold prepare");
}
} // namespace

int main() {
  try {
    one_resident_yields_whole_turns_fifo();
    idle_query_uses_actual_residents_and_reserved_slots();
    invalid_transitions_leave_queue_and_active_unchanged();
    cold_spill_capacity_covers_all_hot_transitions();
    std::puts("request window: 4 checks passed");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "request window: %s\n", error.what());
    return 1;
  }
}
