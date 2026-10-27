#pragma once

#include <algorithm>
#include <cstddef>
#include <exception>
#include <span>
#include <stdexcept>

#include "request_loop.h"

namespace hyle::inference {

enum class RequestWaveAction { kStart, kResume, kDrain };
enum class RequestWaveRetirement { kObserve, kAbort };

// Borrowed facts about an original pending call, not another credit ledger.
struct RequestWavePendingMember final {
  RequestKey key;
  std::size_t reserved = 0;
};

struct RequestWaveAdvance final {
  bool advanced = false;
  RetirementObservation remaining{true, {}};
};

// Stateless serialized wave control. Stage is a statically bound original
// execution owner, with pending_count(), pending_member(j), advance(action,
// wave), and noexcept retire(observe|abort). It alone submits/observes native
// work and validates that its pending metadata still names that work.
//
// kDrain starts no work; kResume honors wave.drain_requested without extending
// the accepted work. retire() clears pending metadata only when actually
// retired, independently of status. This function owns no call, completion,
// queue, cohort storage or persistent reservation. RequestLoop still validates
// observed progress against its original granted output credit.
template<class Stage>
RequestWaveResult advance_request_wave(RequestWave wave,
    std::span<std::size_t> remaining_reserved, Stage &stage) {
  static_assert(noexcept(stage.retire(RequestWaveRetirement::kObserve)));
  if (wave.keys.size() != wave.credits.size() || wave.keys.size() != remaining_reserved.size())
    throw std::logic_error("request wave lost its original credit extent");
  std::ranges::fill(remaining_reserved, std::size_t{0});
  if (wave.observe_only) return {.remaining = stage.retire(RequestWaveRetirement::kObserve)};
  try {
    const auto pending = stage.pending_count();
    if (pending && pending != wave.keys.size())
      throw std::logic_error("pending request wave changed its cohort width");
    for (std::size_t j = 0; j < wave.keys.size(); ++j) {
      if (pending) {
        const auto member = stage.pending_member(j);
        if (member.key != wave.keys[j] || member.reserved != wave.credits[j].reserved)
          throw std::logic_error("pending request wave changed its identity or reserved output");
      } else if (wave.credits[j].reserved)
        throw std::logic_error("request wave retained credit without its original call");
    }
    const auto action = pending ? RequestWaveAction::kResume :
        wave.drain_requested ? RequestWaveAction::kDrain : RequestWaveAction::kStart;
    const auto observed = stage.advance(action, wave);
    const auto remaining = stage.pending_count();
    if (remaining) {
      if (remaining != wave.keys.size() || observed.remaining.retired)
        throw std::logic_error("request wave pending metadata disagrees with its retirement");
      for (std::size_t j = 0; j < remaining; ++j) {
        const auto member = stage.pending_member(j);
        if (member.key != wave.keys[j])
          throw std::logic_error("request wave changed its pending cohort identity");
        remaining_reserved[j] = member.reserved;
      }
    }
    return {.advanced = observed.advanced, .remaining = observed.remaining};
  } catch (...) {
    const auto failure = std::current_exception();
    std::ranges::fill(remaining_reserved, std::size_t{0});
    return {.advanced = true, .remaining = stage.retire(RequestWaveRetirement::kAbort), .failure = failure};
  }
}

} // namespace hyle::inference
