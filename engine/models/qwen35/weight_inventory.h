#pragma once

#include <cstddef>
#include <string_view>

#include "../../inference/route_reason.h"

namespace qwen35 {

// Cold model and route facts. Formatting belongs to the consumer, and this
// value owns no model, execution resource or state-publication authority.
struct WeightInventoryFacts final {
  std::size_t text_tensors = 0, mtp_tensors = 0;
  hyle::inference::RouteReason mtp_selection_reason = hyle::inference::RouteReason::kNotConfigured;
  hyle::inference::RouteReason mtp_depth_source = hyle::inference::RouteReason::kNotConfigured;
  hyle::inference::RouteReason draft_vocabulary_selection_reason = hyle::inference::RouteReason::kNotConfigured;
  hyle::inference::RouteReason bounded_generation_plan_reason = hyle::inference::RouteReason::kNotConfigured;
  hyle::inference::RouteReason decode_overlap_plan_reason = hyle::inference::RouteReason::kNotConfigured;
  hyle::inference::RouteReason device_verification_plan_reason = hyle::inference::RouteReason::kNotConfigured;
  hyle::inference::RouteReason overlap_depth_plan_reason = hyle::inference::RouteReason::kNotConfigured;
  bool float8_weight_policy = false;
  std::size_t canonical_tensors = 0, canonical_bytes = 0, fp8_groups = 0, fp8_import_peak_bytes = 0;
  std::string_view fp8_reader;
  std::size_t ct_channel_groups = 0, ct_local_groups = 0, ct_block_groups = 0, omitted_cache_roles = 0;
  bool cache_float8 = false;
  std::size_t cache_calibration_roles = 0;
  bool draft_cache = false, plain_cache_f16 = false;
};

} // namespace qwen35
