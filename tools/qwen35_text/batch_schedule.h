// Private consumer planning only. No Model, cache, native work or state owner.
#pragma once

#include "session_schedule.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qwen35::benchmark {

struct ScheduledCall {
  std::size_t session, call, phase, local, ordinal;
  PhaseKind kind;
  std::size_t query_rows;
};
struct ScheduleWave {
  std::size_t turn, cohort; // Turn is 1-based; cohort is 0-based within turn.
  std::vector<ScheduledCall> calls;
};

// A complete turn owns one active cohort. Every wave offers at most one next
// call per Session; no call from its next turn can cross that dependency.
// Input phases are the already validated immutable session manifests.
inline std::vector<ScheduleWave> make_schedule_waves(
    std::span<const SessionSchedule> sessions, std::size_t active_limit) {
  if (sessions.empty() || !active_limit)
    throw std::runtime_error("batch schedule requires sessions and a positive active limit");
  std::size_t turns = 0;
  for (const auto &session : sessions) {
    if (session.phases.empty()) throw std::runtime_error("batch schedule has an empty session");
    for (std::size_t i = 0; i < session.phases.size(); ++i) {
      const auto &phase = session.phases[i];
      if (phase.ids.empty() || phase.turn != i / 2 + 1 ||
          phase.kind != (i % 2 ? PhaseKind::kDecode : PhaseKind::kPrompt))
        throw std::runtime_error("batch schedule requires ordered nonempty prompt/decode turns");
    }
    turns = (std::max)(turns, session.phases.back().turn);
  }
  struct Cursor { std::size_t call = 0, phase = 0, local = 0; };
  std::vector<Cursor> cursors(sessions.size());
  std::vector<ScheduleWave> waves;
  std::size_t ordinal = 0;
  for (std::size_t turn = 1; turn <= turns; ++turn) {
    std::vector<std::size_t> eligible;
    eligible.reserve(sessions.size());
    for (std::size_t s = 0; s < sessions.size(); ++s)
      if (sessions[s].phases.back().turn >= turn) eligible.push_back(s);
    std::size_t cohort = 0;
    for (std::size_t first = 0; first < eligible.size(); ++cohort) {
      const auto end = first + (std::min)(active_limit, eligible.size() - first);
      for (;;) {
        ScheduleWave wave{turn, cohort, {}};
        wave.calls.reserve(end - first);
        for (auto at = first; at < end; ++at) {
          const auto s = eligible[at]; auto &cursor = cursors[s];
          if (cursor.phase == sessions[s].phases.size()) continue;
          const auto &phase = sessions[s].phases[cursor.phase];
          if (phase.turn != turn) continue;
          if (ordinal == (std::numeric_limits<std::size_t>::max)())
            throw std::runtime_error("batch schedule call count overflows");
          const auto prompt = phase.kind == PhaseKind::kPrompt;
          wave.calls.push_back({s, cursor.call++, cursor.phase, cursor.local,
              ordinal++, phase.kind, prompt ? phase.ids.size() : std::size_t{1}});
          if (prompt || ++cursor.local == phase.ids.size()) {
            ++cursor.phase; cursor.local = 0;
          }
        }
        if (wave.calls.empty()) break;
        waves.push_back(std::move(wave));
      }
      first = end;
    }
  }
  return waves;
}

struct ScheduleBucket {
  PhaseKind kind;
  std::size_t query_rows;
  std::vector<std::size_t> calls; // Indices into this immutable wave.calls.
};

// Resolve prefix lookup/restoration before calling this function. Zero means
// a caller-proven full hit, not permission to skip a token. Such a call remains
// in wave.calls for its host result, Session pin and external version charge.
// Bucket execution must finish before admitting the next wave. No padding.
inline std::vector<ScheduleBucket> bucket_schedule_wave(
    const ScheduleWave &wave, std::span<const std::size_t> actual_rows) {
  if (actual_rows.size() != wave.calls.size())
    throw std::runtime_error("batch wave query lengths do not match its calls");
  std::vector<ScheduleBucket> buckets;
  for (std::size_t i = 0; i < wave.calls.size(); ++i) {
    const auto &call = wave.calls[i]; const auto rows = actual_rows[i];
    if (rows > call.query_rows)
      throw std::runtime_error("batch wave actual query exceeds its logical input");
    if (!rows) continue;
    auto found = std::find_if(buckets.begin(), buckets.end(), [&](const auto &bucket) {
      return bucket.kind == call.kind && bucket.query_rows == rows;
    });
    if (found == buckets.end()) buckets.push_back({call.kind, rows, {i}});
    else found->calls.push_back(i);
  }
  return buckets;
}

struct BatchShapeUse {
  std::size_t requests, query_rows, max_wave_calls;
};

// Options are indexed by ScheduledCall.ordinal. Nominal cache-miss rows are
// always retained; otherwise supply only finite reachable suffix alternatives
// (and zero on a full hit). Future/same-wave publications are not earlier
// checkpoints. An empty outer span or per-call list adds no alternatives.
// Restores use their original R1 chunks and require separate use counts. R1's
// nominal shape domain is retained even when every runtime call happens to hit.
// max_wave_calls bounds exact-shape batches, including resource-driven splits:
// K possible members allow floor(K/R) R-wide batches. Either replay slot may
// conservatively use that complete bound.
inline std::vector<BatchShapeUse> plan_batch_shape_uses(
    std::span<const ScheduleWave> waves,
    std::span<const std::vector<std::size_t>> possible_rows_by_ordinal = {}) {
  std::size_t count = 0;
  for (const auto &wave : waves) {
    if (wave.calls.size() > (std::numeric_limits<std::size_t>::max)() - count)
      throw std::runtime_error("batch shape call count overflows");
    count += wave.calls.size();
  }
  if (!possible_rows_by_ordinal.empty() && possible_rows_by_ordinal.size() != count)
    throw std::runtime_error("batch shape options do not match the schedule");
  std::map<std::pair<std::size_t, std::size_t>, std::size_t> uses;
  for (const auto &wave : waves) {
    struct Support { PhaseKind kind; std::size_t rows, requests; };
    std::vector<Support> supported;
    for (const auto &call : wave.calls) {
      if (call.ordinal >= count || !call.query_rows)
        throw std::runtime_error("batch shape plan requires original schedule calls");
      uses.try_emplace({1, call.query_rows}, 0); // Original R1 restore domain.
      std::vector<std::size_t> rows;
      if (!possible_rows_by_ordinal.empty()) rows = possible_rows_by_ordinal[call.ordinal];
      rows.push_back(call.query_rows);
      std::sort(rows.begin(), rows.end()); rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
      for (const auto t : rows) {
        if (t > call.query_rows)
          throw std::runtime_error("batch shape alternative exceeds its logical input");
        if (!t) continue;
        auto found = std::find_if(supported.begin(), supported.end(), [&](const auto &entry) {
          return entry.kind == call.kind && entry.rows == t;
        });
        if (found == supported.end()) supported.push_back({call.kind, t, 1});
        else ++found->requests;
      }
    }
    for (const auto &entry : supported) {
      // A bucket may shrink after full hits or shorter suffixes. Only Sessions
      // in THIS wave which can supply this kind/T contribute to its width.
      for (std::size_t r = 1; r <= entry.requests; ++r) {
        auto &value = uses[{r, entry.rows}];
        const auto batches = entry.requests / r;
        if (batches > (std::numeric_limits<std::size_t>::max)() - value)
          throw std::runtime_error("batch shape replay bound overflows");
        value += batches;
      }
    }
  }
  std::vector<BatchShapeUse> result;
  result.reserve(uses.size());
  for (const auto &[shape, calls] : uses) result.push_back({shape.first, shape.second, calls});
  return result;
}

} // namespace qwen35::benchmark
