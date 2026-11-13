#pragma once

#include "model_core.h"
#include "decode_route_policy.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

namespace qwen35 {

// The route policy decides whether a continuation is a candidate. Qwen owns
// the provisional cache geometry and asks the actual backend about every
// target cache storage. The returned endpoint is reused by the Model.
struct MtpContinuationInputs final {
  const DecodeRouteFacts &routes;
  const ModelGeometry &geometry;
  const PreparedWeightSource &source;
  PrecisionProfile precision;
  BackendSelection backend;
  std::size_t context, residents, active_sequences, query_rows, recurrent_bytes, draft_depth;
  bool overlap, verification, random;
  const Backend *shared_endpoint = nullptr;
};

struct MtpContinuationPreparation final {
  std::optional<Backend> endpoint;
  bool enabled = false;
  hyle::inference::RouteReason reason = hyle::inference::RouteReason::kNone;
};

inline MtpContinuationPreparation prepare_mtp_continuation(const MtpContinuationInputs &inputs) {
  namespace inference = hyle::inference;
  constexpr auto exact_token_domain = std::int64_t{1} << std::numeric_limits<float>::digits;
  const auto decision = qwen35::select_continuous_mtp(inputs.routes, inputs.overlap, inputs.residents,
      inputs.geometry.attention_layers() != 0, inputs.geometry.vocabulary <= exact_token_domain);
  if (!decision.enabled) return {{}, false, decision.reason};

  const auto provisional = cache::plan(inputs.geometry, inputs.source.cache_row_bytes, inputs.context,
      inputs.residents, inputs.query_rows, inputs.recurrent_bytes, inputs.active_sequences,
      cache::add(inputs.draft_depth, 1), true, inputs.random, 3,
      inputs.verification ? 2 : 0, inputs.precision, inputs.verification ? inputs.draft_depth : 0);
  auto endpoint = inputs.shared_endpoint ? take(inputs.shared_endpoint->share_endpoint()) : take(Backend::create(inputs.backend));
  const bool enabled = supports_mtp_attention(endpoint,
      attention_capacity(inputs.geometry, inputs.precision, provisional), inputs.source.cache_storage);
  const auto reason = enabled ? inference::RouteReason::kSelected : inference::RouteReason::kAttentionCapability;
  return {std::optional<Backend>{std::move(endpoint)}, enabled, reason};
}

} // namespace qwen35
