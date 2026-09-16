// SGLang Apache-2.0 device fragment at 06992c92ad4494fe9f9636e75be3e35040512447.
// Arithmetic/indexing retained; namespace and support macro isolated.
// See LICENSE, NOTICE and provenance.json.
#pragma once

#include "sglang_device_compat.cuh"

namespace hyle::compute::cuda_native::sglang_vendor {

enum class ActivationKind : uint32_t {
  kSiLU,
  kGELU,
  kGELUTanh,
  kReLU2,
};

template <ActivationKind kAct>
HYLE_SGL_DEVICE float apply_activation_f32(float x_f32) {
  if constexpr (kAct == ActivationKind::kSiLU) {
    return x_f32 / (1.0f + expf(-x_f32));
  } else if constexpr (kAct == ActivationKind::kGELU) {
    constexpr auto kSqrt1Over2 = 0.7071067811865475f;
    return x_f32 * (0.5f * (1.0f + erff(x_f32 * kSqrt1Over2)));
  } else if constexpr (kAct == ActivationKind::kGELUTanh) {
    constexpr auto kGeluTanhAlpha = 0.044715f;
    constexpr auto kGeluTanhBeta = 0.7978845608028654f;
    const float cdf = 0.5f * (1.0f + tanhf(kGeluTanhBeta * (x_f32 + kGeluTanhAlpha * x_f32 * x_f32 * x_f32)));
    return x_f32 * cdf;
  } else if constexpr (kAct == ActivationKind::kReLU2) {
    const float relu = x_f32 > 0.0f ? x_f32 : 0.0f;
    return relu * relu;
  } else {
    static_assert(host::dependent_false_v<decltype(kAct)>, "unsupported activation kind");
    return 0.0f;
  }
}

struct ActivationParams {
  const void* __restrict__ input;
  void* __restrict__ out;
  uint32_t hidden_dim;
  uint32_t num_tokens;
  // Optional MoE expert filtering: when expert_ids != nullptr, a token is
  // skipped if expert_ids[token_id / expert_step] == -1. expert_step is 1
  // for per-token routing and BLOCK_SIZE_M for sorted/TMA routing.
  const int32_t* __restrict__ expert_ids;
  uint32_t expert_step;
};

template <
    typename T,
    ActivationKind kAct,
    bool kUsePDL,
    bool kFilterExpert,
    bool kRoundActivation = false,
    bool kReuseInput = false>
__global__ void act_and_mul_kernel(const __grid_constant__ ActivationParams params) {
  using namespace device;
  constexpr auto kVecSize = kMaxVecBytes / sizeof(T);
  using vec_t = AlignedVector<T, kMaxVecBytes / sizeof(T)>;
  const auto num_vecs = params.hidden_dim / kVecSize;  // per token
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto token_id = tid / num_vecs;

  if (token_id >= params.num_tokens) return;
  if constexpr (kFilterExpert) {
    if (params.expert_ids[token_id / params.expert_step] == -1) return;
  }
  const auto offset = tid % num_vecs;
  const auto input_offset = token_id * (num_vecs * 2) + offset;
  const auto output_offset = kReuseInput ? input_offset : tid;
  PDLWaitPrimary<kUsePDL>();
  const auto gate = device::load_as<vec_t>(params.input, input_offset);
  const auto up = device::load_as<vec_t>(params.input, input_offset + num_vecs);
  vec_t out;
#pragma unroll
  for (int i = 0; i < kVecSize; ++i) {
    const float gate_f32 = device::cast<fp32_t>(gate[i]);
    const float up_f32 = device::cast<fp32_t>(up[i]);
    if constexpr (kRoundActivation) {
      const T activated = device::cast<T>(apply_activation_f32<kAct>(gate_f32));
      out[i] = device::cast<T>(device::cast<fp32_t>(activated) * up_f32);
    } else {
      out[i] = device::cast<T>(apply_activation_f32<kAct>(gate_f32) * up_f32);
    }
  }
  if constexpr (kReuseInput) {
    device::store_as<vec_t>(const_cast<void*>(params.input), out, output_offset);
  } else {
    device::store_as<vec_t>(params.out, out, output_offset);
  }
  PDLTriggerSecondary<kUsePDL>();
}


} // namespace hyle::compute::cuda_native::sglang_vendor

#undef HYLE_SGL_DEVICE
