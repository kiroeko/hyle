#pragma once

#include <cstdio>

#include "d5_deepseek_official_fragments.h"
#include "d5_grouped_fp8_fragments.h"

namespace d5_vulkan_official_boundary {
namespace c = hyle::compute;
namespace sem = d5_qwen38_24t_semantic_fragments;

template <class Check> void for_each(Check check) {
  for (bool flash : {false, true}) {
    d5_moe_execution_fragments::MoeReference unused;
    check(d5_grouped_fp8_fragments::glm_moe(flash, &unused));
  }
  for (auto index : {3u, 4u}) {
    d5_moe_execution_fragments::MoeReference unused;
    check(d5_deepseek_official_fragments::moe_chain(
        d5_moe_execution_cases::kCases[index], false, &unused, false));
  }
}

inline const char *missing_operation(const c::Operation &operation) {
  if (operation.custom_as<c::model_ops::GroupedFloat8Linear>())
    return "GroupedFloat8Linear";
  if (operation.custom_as<c::model_ops::GroupedFloat4Linear>())
    return "GroupedFloat4Linear";
  if (operation.custom_as<c::model_ops::Float8BlockLinear>())
    return "Float8BlockLinear";
  return nullptr;
}

inline void record(const char *entry, const sem::Fragment &fragment,
                   std::size_t prepared, const char *operation,
                   const hyle::Status &status) {
  const auto name = fragment.profile->model_name;
  const auto message = status.message();
  std::printf("OFFICIAL_VULKAN_REJECT entry=%s model=%.*s region=%llu "
              "nodes=%zu prepared=%zu operation=%s code=%d message=%.*s\n",
              entry, int(name.size()), name.data(),
              static_cast<unsigned long long>(fragment.region), fragment.nodes.size(),
              prepared, operation, int(status.code()), int(message.size()), message.data());
}
} // namespace d5_vulkan_official_boundary
