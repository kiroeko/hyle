#pragma once

// Qwen-private cold cache capacity and physical byte facts. No storage,
// lease, version or session owner is created or retained by these values.
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "model_geometry.h"

namespace qwen35::cache {

inline std::size_t add(std::size_t a, std::size_t b) {
  if (b > (std::numeric_limits<std::size_t>::max)() - a)
    throw std::runtime_error("cache backing byte count overflows");
  return a + b;
}
inline std::size_t multiply(std::size_t a, std::size_t b) {
  if (a && b > (std::numeric_limits<std::size_t>::max)() / a)
    throw std::runtime_error("cache backing byte count overflows");
  return a * b;
}

// State pools use this alignment explicitly at allocation sites. Capacity
// includes the gaps between simultaneously live rows, not only their payloads.
inline constexpr std::size_t state_pool_alignment = 256;
inline std::size_t state_pool_padding(std::size_t payload) {
  return (state_pool_alignment - payload % state_pool_alignment) % state_pool_alignment;
}
inline std::size_t state_pool_backing_bytes(std::size_t payload, std::size_t slots) {
  if (!payload || !slots)
    throw std::runtime_error("state pool requires positive payload and slot counts");
  return add(add(multiply(payload, slots), multiply(state_pool_padding(payload), slots - 1)), 1);
}

struct KvRowBytes final {
  std::size_t key = 0, value = 0;
  friend constexpr bool operator==(const KvRowBytes &, const KvRowBytes &) noexcept = default;
};

// Original physical source facts, including an available but unused draft.
// One bank's carrier never implies another bank's dtype, scale or row extent.
struct CacheRowBytes final {
  KvRowBytes target, draft;
  friend constexpr bool operator==(const CacheRowBytes &, const CacheRowBytes &) noexcept = default;
};

// Target and draft inventories use the same page geometry but independent
// typed arena blocks and byte extents; this policy is not a Compute owner.
struct Plan final {
  static constexpr std::uint32_t page_rows = 16;
  CacheRowBytes row_bytes;
  std::size_t resident_sessions = 0, active_sequences = 1, recurrent_payload_bytes = 0;
  std::uint32_t max_requests = 0, max_query_rows = 0, max_kv_rows = 0;
  std::uint32_t max_pages = 0, max_page_refs = 0, state_slots = 0;
  std::uint32_t snapshot_slots = 0, candidate_boundaries = 1, hidden_slots = 0;
  std::uint32_t in_flight_batches = 1;
  std::size_t key_bank_bytes = 0, value_bank_bytes = 0; // Per target layer.
  std::size_t draft_key_bank_bytes = 0, draft_value_bank_bytes = 0; // Zero without a draft plan.
  std::size_t hidden_row_bytes = 0, hidden_bytes = 0, backing_bytes = 0;
  std::size_t attention_layers = 0, recurrent_layers = 0;
  bool draft_cache = false;
};
inline Plan plan(const ModelGeometry &geometry, CacheRowBytes row_bytes,
                 std::size_t context, std::size_t residents,
                 std::size_t query_rows_per_request, std::size_t recurrent_payload,
                  std::size_t active_sequences = 1,
                   std::size_t candidate_boundaries = 1,
                   bool draft_cache = false, bool frontier_hidden = false,
                   std::size_t in_flight_batches = 1) {
  geometry.validate();
  if (!row_bytes.target.key || !row_bytes.target.value ||
      ((!row_bytes.draft.key) != (!row_bytes.draft.value)) ||
      (draft_cache && (!row_bytes.draft.key || !row_bytes.draft.value)))
    throw std::runtime_error("cache plan requires actual target K/V row extents and an independent complete draft pair when used");
  if (context > static_cast<std::size_t>(geometry.context_limit) || (draft_cache && geometry.mtp_layers != 1))
    throw std::runtime_error("cache context or NextN selection exceeds source geometry");
  if (!context || !residents || !query_rows_per_request || query_rows_per_request > context ||
      !active_sequences || active_sequences > residents || !in_flight_batches ||
      !candidate_boundaries || candidate_boundaries > context)
    throw std::runtime_error("cache plan requires valid context/query rows, one to N active sequences, positive in-flight batch depth and candidate boundaries within context");
  const auto refs = add(context, Plan::page_rows - 1) / Plan::page_rows;
  // Depth counts simultaneously retained batches, independently of each
  // batch's R sequences and K verifier boundaries. Every in-flight sequence
  // retains one candidate version and can require one extra partial COW tail
  // beside the N committed complete-context capacities. This is a capacity
  // bound, not permission to reuse an in-flight page or publish a candidate.
  const auto pending_sequences = multiply(in_flight_batches, active_sequences);
  const auto pages = add(multiply(refs, residents), pending_sequences);
  const auto snapshots = add(residents, pending_sequences);
  // Each verifier prefix has independent material state, while its episode
  // still creates one unpublished version. Committed heads may retain carry
  // and prior together; each candidate retains one target row per prefix.
  const auto candidate_states = multiply(candidate_boundaries, pending_sequences);
  const auto states = add(residents, candidate_states);
  const auto hidden_slots = draft_cache || candidate_boundaries > 1
      ? add(multiply(2, residents), candidate_states)
      : frontier_hidden ? snapshots : 0;
  const auto hidden_bytes = hidden_slots ? state_pool_backing_bytes(geometry.hidden_bytes(), hidden_slots) : 0;
  const auto page_refs = multiply(refs, active_sequences);
  const auto query_rows = multiply(query_rows_per_request, active_sequences);
  const auto kv_rows = multiply(context, active_sequences);
  constexpr auto max_index = std::size_t((std::numeric_limits<std::int32_t>::max)());
  if (active_sequences >= max_index || pages > max_index || page_refs > max_index ||
      query_rows > max_index || kv_rows > max_index ||
      states > std::size_t((std::numeric_limits<std::uint32_t>::max)()) ||
      hidden_slots > std::size_t((std::numeric_limits<std::uint32_t>::max)()))
    throw std::runtime_error("cache batch/page/state capacity exceeds the supported index range");
  Plan result;
  result.row_bytes = row_bytes;
  result.resident_sessions = residents;
  result.active_sequences = active_sequences;
  result.recurrent_payload_bytes = recurrent_payload;
  result.max_requests = static_cast<std::uint32_t>(active_sequences);
  result.max_query_rows = static_cast<std::uint32_t>(query_rows);
  result.max_kv_rows = static_cast<std::uint32_t>(kv_rows);
  result.max_pages = static_cast<std::uint32_t>(pages);
  result.max_page_refs = static_cast<std::uint32_t>(page_refs);
  result.state_slots = static_cast<std::uint32_t>(states);
  result.snapshot_slots = static_cast<std::uint32_t>(snapshots);
  result.candidate_boundaries = static_cast<std::uint32_t>(candidate_boundaries);
  result.in_flight_batches = static_cast<std::uint32_t>(in_flight_batches);
  result.hidden_slots = static_cast<std::uint32_t>(hidden_slots);
  result.hidden_bytes = hidden_bytes;
  result.draft_cache = draft_cache;
  result.hidden_row_bytes = geometry.hidden_bytes();
  result.attention_layers = geometry.attention_layers(); result.recurrent_layers = geometry.recurrent_layers();
  const auto bank_rows = multiply(pages, Plan::page_rows);
  result.key_bank_bytes = add(multiply(bank_rows, row_bytes.target.key), 1);
  result.value_bank_bytes = add(multiply(bank_rows, row_bytes.target.value), 1);
  if (draft_cache) {
    result.draft_key_bank_bytes = add(multiply(bank_rows, row_bytes.draft.key), 1);
    result.draft_value_bank_bytes = add(multiply(bank_rows, row_bytes.draft.value), 1);
  }
  // Each real arena contributes its own final byte. An available unused draft
  // remains in row_bytes but has no arena or backing charge in this plan.
  const auto arena_bytes = add(multiply(add(result.key_bank_bytes, result.value_bank_bytes),
      result.attention_layers), add(result.draft_key_bank_bytes, result.draft_value_bank_bytes));
  std::size_t recurrent_padding = 0;
  if (result.recurrent_layers) {
    const auto history_bytes = multiply(multiply(static_cast<std::size_t>(geometry.conv_kernel - 1),
        static_cast<std::size_t>(geometry.gdn_qkv_width())), 2);
    const auto recurrent_bytes = multiply(multiply(static_cast<std::size_t>(geometry.gdn_value_heads),
        static_cast<std::size_t>(geometry.gdn_key_depth)),
        multiply(static_cast<std::size_t>(geometry.gdn_value_depth), 4));
    recurrent_padding = multiply(multiply(add(state_pool_padding(history_bytes),
        state_pool_padding(recurrent_bytes)), states - 1), result.recurrent_layers);
  }
  result.backing_bytes = add(add(arena_bytes,
      add(add(multiply(recurrent_payload, result.state_slots), multiply(2, result.recurrent_layers)),
          recurrent_padding)), hidden_bytes);
  return result;
}

} // namespace qwen35::cache
