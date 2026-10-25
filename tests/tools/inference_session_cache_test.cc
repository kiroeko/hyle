// Host metadata only: callback facts do not simulate native work or own a model.
#include "../../engine/inference/session_cache.h"

#include <cstddef>
#include <cstdio>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>

namespace {
using hyle::inference::CacheAdmissionStatus;
using hyle::inference::RetirementObservation;
using hyle::inference::SessionCache;
static_assert(!std::is_convertible_v<RetirementObservation, bool>);

void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}

void pending_error_keeps_charge_until_observed_retirement() {
  SessionCache cache({1, 64});
  std::size_t callbacks = 0;
  const auto admitted = cache.admit(10, 64, [&](std::size_t) {
    ++callbacks;
    return RetirementObservation{true, {}};
  });
  require(admitted.accepted() && callbacks == 0, "initial cache admission unexpectedly retired an owner");
  const auto error = hyle::Status::Internal("original owner retirement failure");
  const auto pending = cache.admit(20, 64, [&](std::size_t owner) {
    require(owner == 10, "wrong owner selected for pending retirement");
    ++callbacks;
    return RetirementObservation{false, error};
  });
  require(pending.status == CacheAdmissionStatus::kRetirementUnproven &&
      pending.retirement_status == error && !pending.accepted() &&
      !pending.evicted_sessions && !pending.evicted_bytes && callbacks == 1,
      "pending owner error was lost or interpreted as released capacity");
  require(!cache.contains(10) && cache.retiring(10) && !cache.pin(10) &&
      !cache.contains(20) && cache.retained_sessions() == 1 && cache.retained_bytes() == 64,
      "unretired error made an old owner visible, returned its charge or inserted the new owner");

  const auto retired = cache.collect_retired(10, [&](std::size_t owner) {
    require(owner == 10, "collection changed retiring owner identity");
    ++callbacks;
    return RetirementObservation{true, error};
  });
  require(retired.retired && retired.status == error && callbacks == 2 &&
      !cache.contains(10) && !cache.retiring(10) && !cache.contains(20) &&
      cache.retained_sessions() == 0 && cache.retained_bytes() == 0,
      "retired owner error was erased or its metadata charge was not returned exactly once");
  const auto absent = cache.collect_retired(10, [&](std::size_t) {
    ++callbacks;
    return RetirementObservation{true, error};
  });
  require(!absent.retired && absent.status.ok() && callbacks == 2 &&
      cache.retained_sessions() == 0 && cache.retained_bytes() == 0,
      "repeated collection called an absent owner or returned its charge twice");
}

void completed_error_releases_old_entry_without_admitting_new_entry() {
  SessionCache cache({1, 64});
  const auto first = cache.admit(10, 64, [](std::size_t) { return RetirementObservation{true, {}}; });
  require(first.accepted(), "first cache entry was not admitted");
  const auto error = hyle::Status::Internal("completed owner failure");
  std::size_t callbacks = 0;
  const auto failed = cache.admit(20, 64, [&](std::size_t owner) {
    require(owner == 10, "completed retirement selected the wrong owner");
    ++callbacks;
    return RetirementObservation{true, error};
  });
  require(failed.status == CacheAdmissionStatus::kRetirementFailed && !failed.accepted() &&
      failed.retirement_status == error && failed.evicted_sessions == 1 && failed.evicted_bytes == 64 &&
      callbacks == 1 && !cache.contains(10) && !cache.retiring(10) && !cache.contains(20) &&
      cache.retained_sessions() == 0 && cache.retained_bytes() == 0,
      "successful cleanup hid the owner failure or continued failed admission into a new entry");
  // The directory reports this failed admission; it does not invent a global
  // fatal state or implicitly retry the owner. A later explicit admission is distinct.
  const auto later = cache.admit(30, 64, [](std::size_t) { return RetirementObservation{true, {}}; });
  require(later.accepted() && cache.contains(30) && !cache.contains(20) && failed.retirement_status == error,
      "later explicit admission changed the original failed outcome or inserted its rejected owner");
}

void pin_blocks_collection_and_success_preserves_idle_lru() {
  SessionCache cache({2, 128});
  for (auto owner : {std::size_t{10}, std::size_t{20}}) {
    const auto admitted = cache.admit(owner, 64, [](std::size_t) { return RetirementObservation{true, {}}; });
    require(admitted.accepted(), "normal initial entries were not admitted");
  }
  auto pin = cache.pin(10);
  require(static_cast<bool>(pin) && cache.begin_retire(10), "active owner could not be withdrawn");
  std::size_t callbacks = 0;
  const auto pinned = cache.collect_retired(10, [&](std::size_t) {
    ++callbacks;
    return RetirementObservation{true, {}};
  });
  require(!pinned.retired && pinned.status.ok() && callbacks == 0 &&
      cache.retained_sessions() == 2 && cache.retained_bytes() == 128,
      "active pin did not block collector and metadata charge reuse");
  pin.reset();
  require(!cache.contains(10) && cache.retiring(10) && !cache.pin(10),
      "returning a pin made a retiring entry visible again");
  const auto retired = cache.collect_retired(10, [&](std::size_t owner) {
    require(owner == 10, "pin release changed owner identity");
    ++callbacks;
    return RetirementObservation{true, {}};
  });
  require(retired.retired && retired.status.ok() && callbacks == 1 &&
      cache.retained_sessions() == 1 && cache.retained_bytes() == 64,
      "successful original-owner release did not return its charge");
  const auto fill = cache.admit(30, 64, [](std::size_t) { return RetirementObservation{true, {}}; });
  require(fill.accepted() && !fill.evicted_sessions, "free slot was not reused without another eviction");
  const auto replacement = cache.admit(40, 64, [&](std::size_t owner) {
    require(owner == 20, "ordinary successful eviction did not retain idle LRU order");
    ++callbacks;
    return RetirementObservation{true, {}};
  });
  require(replacement.accepted() && replacement.retirement_status.ok() &&
      replacement.evicted_sessions == 1 && replacement.evicted_bytes == 64 && callbacks == 2 &&
      !cache.contains(20) && cache.contains(30) && cache.contains(40) &&
      cache.retained_sessions() == 2 && cache.retained_bytes() == 128,
      "normal LRU admission changed status, owner visibility or bounded charges");
}
} // namespace

int main() {
  try {
    pending_error_keeps_charge_until_observed_retirement();
    completed_error_releases_old_entry_without_admitting_new_entry();
    pin_blocks_collection_and_success_preserves_idle_lru();
    std::puts("session cache: 3 host retirement/charge contracts passed; no native work simulated");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "session cache failure: %s\n", error.what());
    return 1;
  }
}
