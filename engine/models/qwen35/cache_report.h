#pragma once

#include <array>
#include <cstddef>
#include <sstream>
#include <string>

#include "model_core.h"

namespace hyle::engine::reporting {

// Presentation belongs to the entry; the model retains the actual accounting facts.
inline std::string qwen35_cache_report(const qwen35::Model &model,
    std::array<std::size_t, 3> packed_prefill_offset = {}) {
  if (packed_prefill_offset[0] > model.packed_prefill_calls ||
      packed_prefill_offset[1] > model.packed_prefill_active_rows ||
      packed_prefill_offset[2] > model.packed_prefill_physical_rows)
    throw std::logic_error("packed prefill report offset exceeds actual completed work");
  const auto usage = model.cache_usage();
  std::ostringstream out;
  if (model.sampling_configuration_count() > 1 || !model.default_sampler.modifiers.history.empty()) out << ",\"sampling_scope\":\"default_configuration\"";
  out << ",\"sampling\":\"" << (model.default_sampler.enabled ? "random" : "greedy") << '"'
      << ",\"sampling_policy_order\":\"" << (model.default_sampler.modifiers.history.empty()
          ? "temperature_top_k_top_p" : "bias_repetition_frequency_presence_mask_temperature_top_k_top_p") << '\"'
      << ",\"history_repetition\":" << model.default_sampler.modifiers.history.repetition
      << ",\"history_presence\":" << model.default_sampler.modifiers.history.presence
      << ",\"history_frequency\":" << model.default_sampler.modifiers.history.frequency
      << ",\"history_counts_snapshot_blocks\":" << (model.resources_->history_counts_pool.valid() ? model.cache_plan.active_sequences : 0)
      << ",\"sampling_temperature\":" << model.default_sampler.policy.temperature
      << ",\"sampling_top_k\":" << model.default_sampler.policy.top_k
      << ",\"sampling_top_p\":" << model.default_sampler.policy.top_p
      << ",\"sampling_seed\":" << model.sampling_seed
      << ",\"mtp_acceptance_policy\":\"" << (!model.cache_plan.draft_cache ? "disabled" : model.default_sampler.enabled ? "min_one_p_over_q" : "equal_greedy_ids") << '"'
      << ",\"frontier_sampling_graph_calls\":" << model.frontier_sampling_graph_calls
      << ",\"mtp_max_drafts\":" << (model.cache_plan.draft_cache ? model.max_drafts() : 0)
      << ",\"draft_vocabulary_requested_rows\":" << (model.resources_->draft_vocabulary ? model.resources_->draft_vocabulary->size() : 0)
      << ",\"draft_vocabulary_rows\":" << (model.resources_->draft_vocabulary && model.resources_->draft_head ? model.resources_->draft_head->original_ids.description.dimensions()[0] : 0)
      << ",\"draft_vocabulary_identity\":\"" << (model.resources_->draft_vocabulary ? model.resources_->draft_vocabulary_identity : std::string{}) << '"'
      << ",\"draft_vocabulary_weight_bytes\":" << (model.resources_->draft_vocabulary && model.resources_->draft_head ? model.resources_->draft_head->extra_weight_bytes() : 0)
      << ",\"draft_vocabulary_indexed_rows\":" << (model.resources_->draft_head && model.resources_->draft_head->indexed_weight ? "true" : "false")
      << ",\"draft_vocabulary_map_bytes\":" << (model.resources_->draft_vocabulary && model.resources_->draft_head ? qwen35::take(model.resources_->draft_head->original_ids.description.required_bytes()) : 0)
      << ",\"draft_vocabulary_policy\":\"greedy_unmodified\""
      << ",\"draft_head_rows\":" << (model.resources_->draft_vocabulary ? model.resources_->draft_vocabulary->size() : model.resources_->draft_head ? model.resources_->draft_head->projection_weight().description.dimensions()[0] : 0)
      << ",\"draft_head_weight_bytes\":" << (model.resources_->draft_head ? model.resources_->draft_head->extra_weight_bytes() : 0)
      << ",\"draft_head_identity\":\"" << (model.resources_->draft_head ? model.resources_->draft_head->identity : std::string{}) << '"'
      << ",\"draft_head_storage\":\"" << (!model.resources_->draft_head ? "original" : model.resources_->draft_head->projection_weight().description.is_plain() ? "dense" :
          model.resources_->draft_head->projection_weight().description.storage_format().id()==qwen35::c::kGgufQ4K ? "Q4_K" : "gguf_encoded") << '"'
      << ",\"draft_head_prepare_seconds\":" << model.resources_->draft_head_prepare_seconds
      << ",\"draft_program_graph_calls\":" << model.draft_program_graph_calls
      << ",\"draft_chain_graph_calls\":" << model.draft_chain_graph_calls
      << ",\"draft_chain_attention_rows\":" << model.draft_chain_attention_rows
      << ",\"device_verification_handoffs\":" << model.device_verification_handoffs
      << ",\"device_verification_hidden_rows\":" << model.cache_plan.auxiliary_hidden_rows
      << ",\"prefix_decision_graph_calls\":" << model.prefix_decision_graph_calls
      << ",\"packed_prefill_calls\":" << model.packed_prefill_calls - packed_prefill_offset[0]
      << ",\"packed_prefill_active_rows\":" << model.packed_prefill_active_rows - packed_prefill_offset[1]
      << ",\"packed_prefill_physical_rows\":" << model.packed_prefill_physical_rows - packed_prefill_offset[2]
      << ",\"cold_exact_admissions\":" << model.shape_plan.admissions()
      << ",\"cold_exact_evictions\":" << model.shape_plan.evictions()
      << ",\"cpu_ahead_prepared\":" << model.cpu_ahead_prepared
      << ",\"cpu_ahead_hits\":" << model.cpu_ahead_hits
      << ",\"cpu_ahead_discarded\":" << model.cpu_ahead_discarded
      << ",\"cpu_ahead_prepare_seconds\":" << model.cpu_ahead_prepare_seconds
      << ",\"backend\":\"" << model.execution.name()
      << "\",\"cache_budget_scope\":\"shared_pools_retained_outputs_and_resource_metadata\""
      << ",\"cpu_ffn_layers\":" << model.ffn_placement.cpu_layers()
      << ",\"original_host_weight_bytes\":" << model.host_weight_bytes
      << ",\"original_device_weight_bytes\":" << model.device_weight_bytes
      << ",\"cache_shared_backing_bytes\":" << model.known_cache_backing_bytes
      << ",\"cache_retained_output_backing_bytes\":" << usage.live_bytes -
          usage.live_by_category[static_cast<std::size_t>(qwen35::c::ResourceBudgetCategory::kMetadata)] - model.known_cache_backing_bytes
      << ",\"cache_metadata_requested_bytes\":" <<
          usage.live_by_category[static_cast<std::size_t>(qwen35::c::ResourceBudgetCategory::kMetadata)]
      << ",\"cache_live_requested_bytes\":" << usage.live_bytes
      << ",\"cache_peak_requested_bytes\":" << usage.peak_bytes
      << ",\"cache_requested_byte_limit\":";
  if (usage.limit_bytes == SIZE_MAX) out << "null";
  else out << usage.limit_bytes;
  out << ",\"native_execution_private_bytes_unknown\":" <<
      ((model.execution.native_generated_storage() || model.execution.host_only() || model.ffn_placement.cpu_layers()) ? "true" : "false")
      << ",\"physical_vram_limit_known\":false";
  if (model.execution.host_only() || model.ffn_placement.cpu_layers()) {
    out << ",\"execution_shared_requested_byte_limit\":null"
        << ",\"workspace_limit_scope\":\"known_per_invocation_graph_workspace\""
        << ",\"workspace_domains\":[";
    const auto domain_count = model.execution.host_only() ? 1u : model.known_workspace_peak.size();
    for (std::size_t i = 0; i < domain_count; ++i) {
      if (i) out << ',';
      out << "{\"domain\":" << i << ",\"known_peak_bytes\":" << model.known_workspace_peak[i]
          << ",\"limit_bytes\":" << model.known_workspace_limits[i] << '}';
    }
    out << ']';
  }
  return out.str();
}

} // namespace hyle::engine::reporting
