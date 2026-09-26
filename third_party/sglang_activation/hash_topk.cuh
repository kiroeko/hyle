// SPDX-License-Identifier: Apache-2.0
// SGLang 06992c92: complete hash routing kernel; TVM host wrapper omitted.
// HYLE adds checked token/expert bounds using its existing device status.
// Arithmetic, warp SUM and output order are unchanged; see provenance.json.
#pragma once
#include <cstdint>

namespace hyle::compute::cuda_native::sglang_hash_vendor {
namespace device = sglang_vendor::device;
// Hash routing ignores scores, so an all-underflowed row would renorm 0/0 to NaN.
constexpr float kRenormalizeSumEpsilon = 1e-20f;

[[maybe_unused]]
HYLE_SGL_DEVICE float act_sqrt_softplus(float x) {
  const float softplus = fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x)));
  return sqrtf(softplus);
}

struct MoEHashTopKParams {
  const float* __restrict__ router_logits;
  const int64_t* __restrict__ input_id;
  const int32_t* __restrict__ tid2eid;
  int32_t* __restrict__ topk_ids;
  float* __restrict__ topk_weights;
  uint32_t num_tokens;
  uint32_t topk;
  uint32_t num_routed_experts;
  uint32_t num_shared_experts;
  float routed_scaling_factor;
  uint32_t vocabulary_size;
  unsigned* status;
};

template <auto Fn, bool kUsePDL>
__global__ void moe_hash_topk_fused(const MoEHashTopKParams __grid_constant__ params) {
  using namespace device;
  const auto& [
    router_logits, input_id, tid2eid, topk_ids, topk_weights, // pointers
    num_tokens, topk, num_routed_experts, num_shared_experts, routed_scaling_factor, vocabulary_size, status] =
      params;

  const uint32_t topk_fused = topk + num_shared_experts;
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t warp_id = tid / kWarpThreads;
  const uint32_t lane_id = tid % kWarpThreads;
  if (warp_id >= num_tokens) return;
  // we can safely prefetch the token id
  const auto token_id = input_id[warp_id];

  PDLWaitPrimary<kUsePDL>();

  // HYLE safety adapter: reject before reading table payload.
  if (token_id < 0 || static_cast<uint64_t>(token_id) >= vocabulary_size) {
    if (lane_id < topk) {
      topk_ids[warp_id * topk_fused + lane_id] = 0;
      topk_weights[warp_id * topk_fused + lane_id] = 0.0f;
    }
    if (lane_id == 0)
      atomicCAS(status, 0u, static_cast<unsigned>(cuda::detail::DeviceStatusCode::kIndexOutOfRange));
    return;
  }

  float routed_weight = 0.0f;
  int32_t expert_id = 0;
  bool invalid_expert = false;
  if (lane_id < topk) {
    expert_id = tid2eid[token_id * topk + lane_id];
    invalid_expert = expert_id < 0 || static_cast<uint32_t>(expert_id) >= num_routed_experts;
  }
  // All lanes in a token warp take the same failure path before score reads.
  if (__any_sync(0xffffffffU, invalid_expert)) {
    if (lane_id < topk) {
      topk_ids[warp_id * topk_fused + lane_id] = 0;
      topk_weights[warp_id * topk_fused + lane_id] = 0.0f;
    }
    if (lane_id == 0)
      atomicCAS(status, 0u, static_cast<unsigned>(cuda::detail::DeviceStatusCode::kIndexOutOfRange));
    return;
  }
  if (lane_id < topk)
    routed_weight = Fn(router_logits[warp_id * num_routed_experts + expert_id]);

  const auto routed_sum = device::warp::reduce_sum(routed_weight);
  if (lane_id < topk_fused) {
    const bool is_shared = lane_id >= topk;
    const auto output_offset = warp_id * topk_fused + lane_id;
    topk_ids[output_offset] = is_shared ? num_routed_experts + lane_id - topk : expert_id;
    topk_weights[output_offset] =
        is_shared ? 1.0f / routed_scaling_factor : routed_weight / (routed_sum + kRenormalizeSumEpsilon);
  }

  PDLTriggerSecondary<kUsePDL>();
}

} // namespace hyle::compute::cuda_native::sglang_hash_vendor
