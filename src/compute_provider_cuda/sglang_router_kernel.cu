#include "sglang_router_kernel.h"
#include "sglang_device_compat.cuh"
#include "device_status_kernel.h"
#include "moe_gate.cuh"
#include "hash_topk.cuh"

namespace hyle::compute::cuda_native::native {
namespace v = sglang_routing_vendor;

cudaError_t sglang_router_prepare(unsigned rows, unsigned experts, float scaling,
                                  SglangRouterProgram *output) noexcept {
  if (!output || rows == 0 || (experts != 256 && experts != 384))
    return cudaErrorInvalidValue;
  const bool small = rows <= v::kSmallTokenThreshold;
  const void *kernel = small
      ? (experts == 256
             ? reinterpret_cast<const void *>(v::moe_fused_gate_kernel_small_token<8, v::ScoringFunc::kSqrtSoftplus>)
             : reinterpret_cast<const void *>(v::moe_fused_gate_kernel_small_token<12, v::ScoringFunc::kSqrtSoftplus>))
      : reinterpret_cast<const void *>(v::moe_fused_gate_kernel<v::ScoringFunc::kSqrtSoftplus>);
  const unsigned threads_x = small ? experts : v::kWarpSize;
  const unsigned threads_y = small ? 1U : v::kWarpsPerCTA;
  const unsigned blocks = small ? rows : (rows + v::kWarpsPerCTA - 1) / v::kWarpsPerCTA;
  const std::size_t shared = 2U * experts * sizeof(float) * threads_y;
  cudaFuncAttributes attributes{};
  int device = 0, shared_limit = 0, grid_limit = 0;
  auto status = cudaFuncGetAttributes(&attributes, kernel);
  if (status == cudaSuccess) status = cudaGetDevice(&device);
  if (status == cudaSuccess)
    status = cudaDeviceGetAttribute(&shared_limit, cudaDevAttrMaxSharedMemoryPerBlock, device);
  if (status == cudaSuccess)
    status = cudaDeviceGetAttribute(&grid_limit, cudaDevAttrMaxGridDimX, device);
  if (status != cudaSuccess) return status;
  if (attributes.maxThreadsPerBlock < static_cast<int>(threads_x * threads_y) ||
      shared + attributes.sharedSizeBytes > static_cast<std::size_t>(shared_limit) ||
      blocks > static_cast<unsigned>(grid_limit))
    return cudaErrorNotSupported;
  *output = {kernel, rows, experts, blocks, threads_x, threads_y, shared, scaling};
  return cudaSuccess;
}

cudaError_t sglang_router_launch(const SglangRouterProgram &program,
                                 const void *logits, const void *bias,
                                 void *weights, void *ids,
                                 cudaStream_t stream) noexcept {
  v::MoEFusedGateParams params{static_cast<const float *>(logits),
      static_cast<const float *>(bias), static_cast<float *>(weights),
      static_cast<std::int32_t *>(ids), program.rows, program.experts,
      6, 0, true, program.scaling, true};
  void *arguments[]{&params};
  return cudaLaunchKernel(program.kernel, dim3(program.blocks),
                          dim3(program.threads_x, program.threads_y), arguments,
                          program.shared_bytes, stream);
}

cudaError_t sglang_hash_router_prepare(unsigned rows, unsigned experts,
                                       unsigned vocabulary,
                                       SglangHashRouterProgram *output) noexcept {
  if (!output || rows == 0 || vocabulary == 0 || (experts != 256 && experts != 384))
    return cudaErrorInvalidValue;
  namespace hash = sglang_hash_vendor;
  const auto *kernel = reinterpret_cast<const void *>(
      hash::moe_hash_topk_fused<hash::act_sqrt_softplus, false>);
  const unsigned blocks = (rows + 3U) / 4U;
  cudaFuncAttributes attributes{};
  int device = 0, grid_limit = 0;
  auto status = cudaFuncGetAttributes(&attributes, kernel);
  if (status == cudaSuccess) status = cudaGetDevice(&device);
  if (status == cudaSuccess)
    status = cudaDeviceGetAttribute(&grid_limit, cudaDevAttrMaxGridDimX, device);
  if (status != cudaSuccess) return status;
  if (attributes.maxThreadsPerBlock < 128 || blocks > static_cast<unsigned>(grid_limit))
    return cudaErrorNotSupported;
  *output = {kernel, rows, experts, vocabulary, blocks};
  return cudaSuccess;
}

cudaError_t sglang_hash_router_launch(const SglangHashRouterProgram &program,
                                      const void *logits, const void *tokens,
                                      const void *table, void *weights, void *ids,
                                      unsigned *status, cudaStream_t stream) noexcept {
  sglang_hash_vendor::MoEHashTopKParams params{
      static_cast<const float *>(logits), static_cast<const std::int64_t *>(tokens),
      static_cast<const std::int32_t *>(table), static_cast<std::int32_t *>(ids),
      static_cast<float *>(weights), program.rows, 6, program.experts, 0, 1.0F,
      program.vocabulary, status};
  void *arguments[]{&params};
  return cudaLaunchKernel(program.kernel, dim3(program.blocks), dim3(128),
                          arguments, 0, stream);
}
} // namespace hyle::compute::cuda_native::native
