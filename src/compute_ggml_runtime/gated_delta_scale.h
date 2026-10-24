#pragma once

// Private C++17 metadata contract shared by current Hyle hosts and locked
// native programs. Slot 0 is K; slot 1 belongs to CUDA state-output planning.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include "ggml.h"

namespace hyle::compute::ggml_runtime {
inline constexpr std::int32_t kGdnOutputScaleContract = 0x48594732;
inline constexpr int kGdnOutputScaleSlot = 2;
inline constexpr int kGdnOutputScalePresenceSlot = 3;

inline bool valid_gdn_output_scale(std::optional<float> scale) noexcept {
  return !scale || std::isfinite(*scale);
}
inline void set_gdn_output_scale(ggml_tensor *node,
                                 std::optional<float> scale) noexcept {
  node->op_params[kGdnOutputScalePresenceSlot] =
      scale ? kGdnOutputScaleContract : 0;
  // Never hash/compare optional padding or use float equality. Empty has a
  // canonical zero payload; explicit signed zero retains its original bits.
  const float value = scale.value_or(0.0F);
  std::memcpy(&node->op_params[kGdnOutputScaleSlot], &value, sizeof(value));
}
inline bool has_gdn_output_scale(const ggml_tensor *node) noexcept {
  return node->op_params[kGdnOutputScalePresenceSlot] == kGdnOutputScaleContract;
}
inline float gdn_output_scale(const ggml_tensor *node,
                              float default_scale) noexcept {
  if (!has_gdn_output_scale(node)) return default_scale;
  float value;
  std::memcpy(&value, &node->op_params[kGdnOutputScaleSlot], sizeof(value));
  return value;
}
} // namespace hyle::compute::ggml_runtime

// Link each host against its real native implementation, not a header-only
// capability claim. An old native cache cannot silently ignore the attribute.
extern "C" std::int32_t ggml_hyle_cpu_gdn_output_scale_contract();
extern "C" std::int32_t ggml_hyle_cuda_gdn_output_scale_contract();
extern "C" std::int32_t ggml_hyle_metal_gdn_output_scale_contract();
extern "C" std::int32_t ggml_hyle_vulkan_gdn_output_scale_contract();
