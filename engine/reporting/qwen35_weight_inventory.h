#pragma once

#include "../models/qwen35/weight_inventory.h"
#include "../inference/route_reason.h"

#include <string>

namespace hyle::engine::reporting {

// Reasons describe the outer driver's decisions; they are not model state.
struct RouteDecisionFacts final {
  inference::RouteReason mtp_selection_reason = inference::RouteReason::kNotConfigured;
  inference::RouteReason mtp_depth_source = inference::RouteReason::kNotConfigured;
  inference::RouteReason draft_vocabulary_selection_reason = inference::RouteReason::kNotConfigured;
  inference::RouteReason bounded_generation_plan_reason = inference::RouteReason::kNotConfigured;
  inference::RouteReason decode_overlap_plan_reason = inference::RouteReason::kNotConfigured;
  inference::RouteReason device_verification_plan_reason = inference::RouteReason::kNotConfigured;
  inference::RouteReason overlap_depth_plan_reason = inference::RouteReason::kNotConfigured;
};

// JSON fragment for the existing CLI report. The model supplies cold facts;
// this formatting does not select a route or own model state.
inline std::string qwen35_weight_inventory(const qwen35::WeightInventoryFacts &facts,
    const RouteDecisionFacts &routes = {}) {
  using hyle::inference::route_reason_name;
  auto result = ",\"weight_count_scope\":\"selected_physical_roles\",\"text_tensors\":" +
      std::to_string(facts.text_tensors) + ",\"mtp_tensors\":" + std::to_string(facts.mtp_tensors);
  result += ",\"mtp_selection_reason\":\"" + std::string(route_reason_name(routes.mtp_selection_reason)) +
      "\",\"mtp_depth_source\":\"" + std::string(route_reason_name(routes.mtp_depth_source)) +
      "\",\"draft_vocabulary_selection_reason\":\"" + std::string(route_reason_name(routes.draft_vocabulary_selection_reason)) +
      "\",\"bounded_generation_plan_reason\":\"" + std::string(route_reason_name(routes.bounded_generation_plan_reason)) +
      "\",\"decode_overlap_plan_reason\":\"" + std::string(route_reason_name(routes.decode_overlap_plan_reason)) +
      "\",\"device_verification_plan_reason\":\"" + std::string(route_reason_name(routes.device_verification_plan_reason)) +
      "\",\"overlap_depth_plan_reason\":\"" + std::string(route_reason_name(routes.overlap_depth_plan_reason)) + "\"";
  if (facts.float8_weight_policy) result += ",\"canonical_weight_tensors\":" + std::to_string(facts.canonical_tensors) +
      ",\"canonical_weight_bytes\":" + std::to_string(facts.canonical_bytes) +
      ",\"fp8_block128_groups\":" + std::to_string(facts.fp8_groups) +
      ",\"fp8_import_temporary_group_peak_bytes\":" + std::to_string(facts.fp8_import_peak_bytes) +
      ",\"fp8_block128_policy\":{\"activation_scale\":\"divide\",\"activation_floor_f32_bits\":\"2edbe6ff\","
      "\"input_dtype\":\"BF16\",\"accumulation_dtype\":\"F32\",\"output_dtype\":\"BF16\","
      "\"weight_scale_conversion\":\"bf16_to_f32_exact\",\"reader\":\"" + std::string(facts.fp8_reader) + "\"}";
  if (facts.ct_channel_groups || facts.ct_local_groups)
    result += ",\"ct_channel_token_groups\":" + std::to_string(facts.ct_channel_groups) +
        ",\"ct_local_nvfp4_groups\":" + std::to_string(facts.ct_local_groups);
  if (facts.ct_block_groups) result += ",\"ct_block128_bf16_groups\":" + std::to_string(facts.ct_block_groups);
  if (facts.omitted_cache_roles) result += ",\"kv_cache_dtype\":\"BF16\",\"omitted_original_cache_calibration_roles\":" +
      std::to_string(facts.omitted_cache_roles);
  if (facts.cache_float8) result += ",\"kv_cache_dtype\":\"E4M3FN\",\"original_cache_calibration_roles\":" +
      std::to_string(facts.cache_calibration_roles);
  if (facts.draft_cache) {
    const auto plain_dtype = facts.plain_cache_f16 ? "F16" : "BF16";
    result += ",\"target_kv_cache_dtype\":\"" + std::string(facts.cache_float8 ? "E4M3FN" : plain_dtype) +
        "\",\"draft_kv_cache_dtype\":\"" + plain_dtype + "\"";
  }
  return result;
}

} // namespace hyle::engine::reporting
