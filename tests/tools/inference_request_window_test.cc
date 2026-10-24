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
bool track_allocations = false, fail_next_allocation = false;
std::size_t allocation_count = 0;
}

void *operator new(std::size_t bytes) {
  if (fail_next_allocation) { fail_next_allocation = false; throw std::bad_alloc{}; }
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
using hyle::inference::RequestKey;
using hyle::inference::RequestWindow;
static_assert(!std::is_copy_constructible_v<RequestWindow> && !std::is_copy_assignable_v<RequestWindow>);
static_assert(!std::is_move_constructible_v<RequestWindow> && !std::is_move_assignable_v<RequestWindow>);
static_assert(std::is_nothrow_copy_constructible_v<RequestKey>);

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
template<class Error, class Operation> void rejects(Operation operation) {
  try { operation(); }
  catch (const Error &) { return; }
  throw std::runtime_error("invalid request-window transition was accepted");
}
void active_is(const RequestWindow &window, std::initializer_list<std::size_t> expected) {
  const auto active = window.active();
  require(active.size() == expected.size(), "request-window active width changed");
  auto index = expected.begin();
  for (auto key : active) require(key.slot == *index++, "request-window active order changed");
}

void finite_and_later_arrival_share_one_fifo() {
  RequestWindow window;
  require(!window.waiting_front(), "unprepared request window offered admission");
  window.prepare(2, 1);
  const std::array keys{window.enqueue(0), window.enqueue(1)};
  const std::array expected{std::size_t{0}, std::size_t{1}, std::size_t{0}, std::size_t{1}};
  for (std::size_t turn = 0; turn < expected.size(); ++turn) {
    const auto key = keys[expected[turn]];
    require(window.waiting_front() == key, "N1 did not preserve whole-turn FIFO order");
    window.admitted(key);
    require(!window.waiting_front(), "N1 admitted a peer while its only slot was reserved");
    const auto order = window.release_turn(key, turn < 2);
    require(order == (turn < 2 ? turn + 1 : 0), "whole-turn idle ordering changed");
    if (turn >= 2) window.release(key); // Owner's native retirement is external.
  }
  require(window.active().empty() && !window.waiting_front(), "finite FIFO left a phantom request");
  const auto later = window.enqueue(1);
  require(later != keys[1] && later.generation > keys[1].generation && window.waiting_front() == later,
      "later arrival did not use the original FIFO with a fresh identity");
  require(!window.cancel_queued(keys[1]) && window.waiting_front() == later,
      "stale cancellation withdrew a later occupant");
  require(window.cancel_queued(later), "queued later arrival was not withdrawn");
  window.release(later);
}

void queued_cancel_preserves_order_and_requires_retirement() {
  RequestWindow window; window.prepare(4, 1);
  const std::array keys{window.enqueue(0), window.enqueue(1), window.enqueue(2), window.enqueue(3)};
  window.admitted(keys[0]);
  require(!window.cancel_queued(keys[0]), "queued cancel released an active native participant");
  active_is(window, {0});
  require(window.cancel_queued(keys[2]) && window.matches(keys[2]), "queued cancellation did not retain its retiring identity");
  rejects<std::logic_error>([&] { (void)window.enqueue(2); });
  rejects<std::logic_error>([&] { window.prepare(4, 1); });
  rejects<std::logic_error>([&] { window.release(keys[0]); });
  window.release(keys[2]);
  const auto replacement = window.enqueue(2);
  require(!window.matches(keys[2]) && !window.cancel_queued(keys[2]), "retired generation was revived");
  window.release_turn(keys[0], false); window.release(keys[0]);
  const std::array remainder{keys[1], keys[3], replacement};
  for (auto key : remainder) {
    require(window.waiting_front() == key, "middle queued cancel reordered surviving arrivals");
    window.admitted(key); window.release_turn(key, false); window.release(key);
  }
  require(!window.waiting_front() && window.active().empty(), "cancelled FIFO did not drain");
}

void idle_query_excludes_active_retiring_and_free_slots() {
  RequestWindow window; window.prepare(3, 2);
  const std::array keys{window.enqueue(0), window.enqueue(1), window.enqueue(2)};
  window.admitted(keys[0]); window.admitted(keys[1]);
  const auto resident = [](std::size_t) { return true; };
  const auto idle = [](std::size_t index) { return index; };
  require(window.idle_victim(resident, idle) == 2, "reserved active slots were treated as idle");
  require(window.cancel_queued(keys[2]) && !window.idle_victim(resident, idle), "retiring queued cancellation became an idle victim");
  window.release(keys[2]);
  require(!window.idle_victim(resident, idle), "free payload slot became an idle victim");
  window.release_turn(keys[0], true);
  window.release_turn(keys[1], false);
  require(window.idle_victim(resident, idle) == 0, "queued future turn was not the only eligible resident");
  require(!window.idle_victim([](std::size_t) { return false; }, idle), "host index query fabricated a resident head");
  window.release(keys[1]); window.cancel_queued(keys[0]); window.release(keys[0]);
}

void invalid_reset_and_stale_events_are_atomic() {
  RequestWindow window; window.prepare(3, 2);
  const auto old = window.enqueue(0), queued = window.enqueue(1);
  rejects<std::out_of_range>([&] { (void)window.enqueue(3); });
  rejects<std::logic_error>([&] { (void)window.enqueue(0); });
  rejects<std::logic_error>([&] { window.admitted(queued); });
  rejects<std::logic_error>([&] { window.prepare(8, 4); });
  require(window.waiting_front() == old && window.active().empty(), "failed admission/reset changed the prepared FIFO");
  require(window.cancel_queued(old) && window.cancel_queued(queued), "valid withdrawal failed");
  window.release(old); window.release(queued);
  fail_next_allocation = true;
  rejects<std::bad_alloc>([&] { window.prepare(9, 4); });
  const auto last_old_capacity = window.enqueue(2);
  require(last_old_capacity.generation > queued.generation, "failed cold prepare reset generation or lost old capacity");
  window.cancel_queued(last_old_capacity); window.release(last_old_capacity);
  window.prepare(1, 1);
  const auto current = window.enqueue(0);
  require(current.generation > last_old_capacity.generation && !window.matches(old), "cold reset revived an old identity");
  rejects<std::out_of_range>([&] { window.admitted(old); });
  rejects<std::out_of_range>([&] { window.release(old); });
  require(!window.cancel_queued(old) && window.waiting_front() == current, "stale event changed the reset window");
  RequestWindow other; other.prepare(1, 1); const auto foreign = other.enqueue(0);
  require(!window.matches(foreign) && !window.cancel_queued(foreign), "foreign window key was accepted");
  require(!window.matches(RequestKey{}), "default key became a request");
  window.admitted(current); window.release_turn(current, false); window.release(current);
  other.cancel_queued(foreign); other.release(foreign);
}

void bounded_ring_reuses_slots_without_hot_allocation() {
  RequestWindow window; window.prepare(7, 5);
  std::array<RequestKey, 7> keys;
  for (std::size_t slot = 0; slot < keys.size(); ++slot) keys[slot] = window.enqueue(slot);
  std::array<std::size_t, 7> turns{}, lifetimes{};
  std::size_t completed = 0, admissions = 0;
  const RequestKey *active_data = nullptr;
  const auto before = allocation_count;
  track_allocations = true;
  try {
    while (completed != keys.size()) {
      while (const auto key = window.waiting_front()) {
        require(key->slot == admissions % keys.size(), "ring wrap/reoccupation changed FIFO order");
        window.admitted(*key); ++admissions;
        if (!active_data) active_data = window.active().data();
        require(window.active().data() == active_data, "prepared active storage moved during admission");
      }
      while (!window.active().empty()) {
        const auto key = window.active().front(); // Copy before mutating active span.
        const bool more = ++turns[key.slot] != 3;
        window.release_turn(key, more);
        if (!more) {
          window.release(key);
          if (++lifetimes[key.slot] == 32) ++completed;
          else {
            turns[key.slot] = 0;
            keys[key.slot] = window.enqueue(key.slot);
            require(!window.matches(key) && !window.cancel_queued(key), "old event affected reoccupied slot");
          }
        }
        if (!window.active().empty()) require(window.active().data() == active_data, "active storage moved on release");
      }
    }
  } catch (...) { track_allocations = false; throw; }
  track_allocations = false;
  require(admissions == 7 * 32 * 3 && window.active().empty() && !window.waiting_front(),
      "bounded live slots lost or duplicated logical turns");
  require(allocation_count == before, "valid prepared request-window transitions allocated");
}
} // namespace

int main() {
  try {
    finite_and_later_arrival_share_one_fifo();
    queued_cancel_preserves_order_and_requires_retirement();
    idle_query_excludes_active_retiring_and_free_slots();
    invalid_reset_and_stale_events_are_atomic();
    bounded_ring_reuses_slots_without_hot_allocation();
    std::puts("request window: 5 checks passed");
    return 0;
  } catch (const std::exception &error) {
    track_allocations = false;
    std::fprintf(stderr, "request window: %s\n", error.what());
    return 1;
  }
}
