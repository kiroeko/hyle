#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

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

// Cold fixed metadata may be uploaded by the original Graph queue. Keep the
// token port in its device domain so an actual predecessor Source still binds
// without a host roundtrip. The concrete transfer/capture is qualified by Graph
// on these exact owners, not inferred from a backend name or model geometry.
inline std::optional<hyle::graph::StorageDomainId> generation_control_storage(
    const hyle::inference::Backend &execution, const OrdinaryIoCapabilities &facts) noexcept {
  if ((facts.role != GraphRole::kText && !has_mtp(facts.role)) || facts.completion != TextCompletion::kEmitNext ||
      facts.packing != RowPacking::kRectangular || facts.cpu_assist || facts.diagnostic ||
      execution.queue_bindings().size() != 1 || !execution.staging() || execution.staging() == execution.device() ||
      !execution.staging().properties().has(hyle::mem::MemoryProperty::kHostAccessible)) return {};
  const auto &primary = execution.storage_owner();
  if (!primary || !primary.context_identity()) return {};
  const auto domains = execution.storage_domains();
  for (std::size_t index = 0; index < domains.size(); ++index) {
    const auto &domain = domains[index];
    if (domain.resource == execution.staging() && domain.owner &&
        domain.owner.context_identity() == primary.context_identity())
      return hyle::graph::StorageDomainId{static_cast<std::uint32_t>(index)};
  }
  return {};
}

struct NativeOrdinaryIoDomains final {
  hyle::graph::StorageDomainId device;
  hyle::graph::StorageDomainId staging;
};

// A native program keeps its original token input/writer on the device. Host
// delivery owns a separate output, without changing the Source's origin.
// Qualify existing endpoint/resource owners; Graph still qualifies the actual
// transfers and external allocations. These facts grant no execution right.
inline std::optional<NativeOrdinaryIoDomains> native_ordinary_io_domains(
    const hyle::inference::Backend &execution, const OrdinaryIoCapabilities &facts) noexcept {
  if (facts.role != GraphRole::kText || facts.completion != TextCompletion::kEmitNext ||
      facts.cpu_assist || facts.diagnostic || !execution.queue_bindings().empty()) return {};
  const auto &primary = execution.storage_owner();
  if (!primary || primary.context_identity() || !primary.endpoint_identity() ||
      execution.staging() == execution.device()) return {};
  const auto domains = execution.storage_domains();
  std::optional<hyle::graph::StorageDomainId> device, staging;
  for (std::size_t index = 0; index < domains.size(); ++index) {
    const auto &domain = domains[index];
    if (!domain.owner || domain.owner.endpoint_identity() != primary.endpoint_identity()) continue;
    const auto id = hyle::graph::StorageDomainId{static_cast<std::uint32_t>(index)};
    if (domain.resource == execution.device()) device = id;
    if (domain.resource == execution.staging()) staging = id;
  }
  if (!device || !staging) return {};
  return NativeOrdinaryIoDomains{*device, *staging};
}

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

// The prepared one-proposal sampler publishes one real acceptance bit and
// reserves exactly three logical draws, independently of accepting the draft.
// Longer sampled chains retain their own separately qualified recipe.
constexpr bool device_mtp_sampling_supported(bool random, std::size_t proposals) noexcept {
  return !random || proposals == 1;
}

// A different capability from host input/result IO: device relay requires an
// unmodified ordinary sampler or a qualified MTP recipe. Sampled MTP retains
// its actual acceptance/publishing path. Keep actual cohort/Source/capture
// checks with their original runtime owners.
struct DeviceRelayCapabilities final {
  hyle::inference::BackendKind backend = hyle::inference::BackendKind::kDefault;
  bool cpu_assist = false, scheduler = false, sampled_mtp = false, logits_modifiers = false;
  // Provision the existing second candidate slot for an ordinary native route.
  // The actual prepared Graph must qualify before decode can select it; this
  // request is not a claim that every region/backend imports pending sources.
  bool native_source_candidate = false;
  bool sampled_mtp_successor = false;

  constexpr hyle::inference::RouteReason refusal_reason() const noexcept {
    if (backend != hyle::inference::BackendKind::kCuda && !native_source_candidate)
      return hyle::inference::RouteReason::kBackend;
    if (cpu_assist) return hyle::inference::RouteReason::kCpuAssist;
    if (scheduler) return hyle::inference::RouteReason::kScheduler;
    if (sampled_mtp && !sampled_mtp_successor) return hyle::inference::RouteReason::kSampling;
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
