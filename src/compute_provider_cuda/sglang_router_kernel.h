#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

namespace hyle::compute::cuda_native::native {
struct SglangRouterProgram final {
  const void *kernel = nullptr;
  unsigned rows = 0, experts = 0, blocks = 0, threads_x = 0, threads_y = 1;
  std::size_t shared_bytes = 0;
  float scaling = 1;
};
cudaError_t sglang_router_prepare(unsigned rows, unsigned experts, float scaling,
                                  SglangRouterProgram *output) noexcept;
cudaError_t sglang_router_launch(const SglangRouterProgram &program,
                                 const void *logits, const void *bias,
                                 void *weights, void *ids,
                                 cudaStream_t stream) noexcept;
struct SglangHashRouterProgram final {
  const void *kernel = nullptr;
  unsigned rows = 0, experts = 0, vocabulary = 0, blocks = 0;
};
cudaError_t sglang_hash_router_prepare(unsigned rows, unsigned experts,
                                       unsigned vocabulary,
                                       SglangHashRouterProgram *output) noexcept;
cudaError_t sglang_hash_router_launch(const SglangHashRouterProgram &program,
                                      const void *logits, const void *tokens,
                                      const void *table, void *weights, void *ids,
                                      unsigned *status, cudaStream_t stream) noexcept;
} // namespace hyle::compute::cuda_native::native
