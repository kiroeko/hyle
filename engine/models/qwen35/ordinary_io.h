#pragma once

#include <cstddef>
#include <cstdint>

#include "execution_mode.h"
#include "../../inference/backend.h"
#include "../../inference/route_reason.h"

namespace qwen35 {

// Cold model/Phase facts, not runtime permissions. One Family can serve both
// ordinary and imported Phases, so materialization and host-batch admission
// intentionally have separate conditions.
struct OrdinaryIoCapabilities final {
  hyle::inference::BackendKind backend = hyle::inference::BackendKind::kDefault;
  GraphRole role = GraphRole::kText;
  TextCompletion completion = TextCompletion::kEmitNext;
  RowPacking packing = RowPacking::kRectangular;
  bool cpu_assist = false, diagnostic = false, imported_sources = false;

  constexpr bool packet_inputs() const noexcept {
    return role == GraphRole::kText &&
        (packing == RowPacking::kRectangular || packs_prefill(packing)) &&
        !diagnostic && (!imported_sources || packs_prefill(packing)) &&
        (backend == hyle::inference::BackendKind::kCpu || backend == hyle::inference::BackendKind::kCuda);
  }
  constexpr bool caller_small_results() const noexcept {
    return role == GraphRole::kText && completion == TextCompletion::kEmitNext &&
        backend == hyle::inference::BackendKind::kCuda && !cpu_assist && !diagnostic;
  }
  constexpr bool batched_small_results(bool has_valid, bool has_accept) const noexcept {
    return caller_small_results() && !imported_sources && has_valid && !has_accept;
  }
};

enum class OrdinaryInputMode : std::uint8_t { kSeparate, kPacket };
enum class OrdinaryResultMode : std::uint8_t { kSeparate, kBatchBool, kBatchFloat };

// Selected only after this exact physical slot's cold allocations succeed or
// legally decline. Budget fallback may differ between slots. A hot failure
// never changes this selection or replaces its original fixed owners.
struct OrdinaryIoPlan final {
  OrdinaryInputMode inputs = OrdinaryInputMode::kSeparate;
  OrdinaryResultMode results = OrdinaryResultMode::kSeparate;
  bool packet_last_ids = false, packet_history_select = false, packet_sampling_identities = false;

  constexpr bool packet() const noexcept { return inputs == OrdinaryInputMode::kPacket; }
  constexpr bool batched_results() const noexcept { return results != OrdinaryResultMode::kSeparate; }
};

// A different capability from host input/result IO: device relay requires an
// unmodified ordinary sampler or deterministic MTP. Sampled MTP has a
// separate acceptance/publishing path. Keep actual cohort/Source/capture
// checks with their original runtime owners.
struct DeviceRelayCapabilities final {
  hyle::inference::BackendKind backend = hyle::inference::BackendKind::kDefault;
  bool cpu_assist = false, scheduler = false, sampled_mtp = false, logits_modifiers = false;

  constexpr hyle::inference::RouteReason refusal_reason() const noexcept {
    if (backend != hyle::inference::BackendKind::kCuda) return hyle::inference::RouteReason::kBackend;
    if (cpu_assist) return hyle::inference::RouteReason::kCpuAssist;
    if (scheduler) return hyle::inference::RouteReason::kScheduler;
    if (sampled_mtp) return hyle::inference::RouteReason::kSampling;
    if (logits_modifiers) return hyle::inference::RouteReason::kLogitsModifiers;
    return hyle::inference::RouteReason::kNone;
  }
  constexpr bool eligible() const noexcept { return refusal_reason() == hyle::inference::RouteReason::kNone; }
};

// Cold eligibility for this model's ordinary prefill relay. This is separate
// from decode relay: future implementations may support either one alone.
// Per-episode shape, tracing, Source and resource checks remain at execution.
struct PrefillRelayFacts final {
  hyle::inference::BackendKind backend = hyle::inference::BackendKind::kDefault;
  bool cpu_assist = false, scheduler = false, draft_cache = false;
  std::size_t in_flight_batches = 0;
  bool unmodified_greedy = false;

  constexpr bool eligible() const noexcept {
    return backend == hyle::inference::BackendKind::kCuda && !cpu_assist && !scheduler &&
        !draft_cache && in_flight_batches >= 2 && unmodified_greedy;
  }
};

} // namespace qwen35
