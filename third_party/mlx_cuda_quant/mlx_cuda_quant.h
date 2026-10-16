// Original MLX row equations/CUDA warp reduction adapted to HYLE typed ports.
// Copyright © 2023-2026 Apple Inc. MIT; see LICENSE and SOURCE.json.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace hyle::borrowed::mlx_cuda_quant {
template<class Type>
__device__ __forceinline__ float load(const void *pointer, unsigned long long index,
                                     Type type) {
  if (type == Type::kFloat16)
    return __half2float(static_cast<const __half *>(pointer)[index]);
  if (type == Type::kBFloat16)
    return __bfloat162float(static_cast<const __nv_bfloat16 *>(pointer)[index]);
  return static_cast<const float *>(pointer)[index];
}
template<class Type>
__device__ __forceinline__ float rounded(float value, Type type) {
  if (type == Type::kFloat16) return __half2float(__float2half_rn(value));
  if (type == Type::kBFloat16)
    return __bfloat162float(__float2bfloat16_rn(value));
  return value;
}
__device__ __forceinline__ float fp4(unsigned code) {
  const float values[8]{0.F,.5F,1.F,1.5F,2.F,3.F,4.F,6.F};
  return (code & 8) ? -values[code & 7] : values[code & 7];
}
__device__ __forceinline__ float fp8(unsigned code) {
  __nv_fp8_e4m3 value;
  value.__x = static_cast<__nv_fp8_storage_t>(code);
  return static_cast<float>(value);
}
__device__ __forceinline__ float warp_sum(float value) {
  for (int distance = 16; distance != 0; distance /= 2)
    value = __fadd_rn(value, __shfl_down_sync(0xffffffffU, value, distance));
  return value;
}
template<class T>
__device__ __forceinline__ void store(void *output, unsigned long long index,
                                     float value) {
  static_cast<T *>(output)[index] = static_cast<T>(value);
}
template<>
__device__ __forceinline__ void store<__half>(void *output,
    unsigned long long index, float value) {
  static_cast<__half *>(output)[index] = __float2half_rn(value);
}
template<>
__device__ __forceinline__ void store<__nv_bfloat16>(void *output,
    unsigned long long index, float value) {
  static_cast<__nv_bfloat16 *>(output)[index] = __float2bfloat16_rn(value);
}
} // namespace hyle::borrowed::mlx_cuda_quant
