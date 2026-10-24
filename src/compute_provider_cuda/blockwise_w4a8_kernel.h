#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace hyle::compute::cuda_native::blockwise_w4a8 {
struct Program final {
  std::uint64_t tokens=0, slots=1, experts=1, columns=0, depth=0;
  std::uint64_t quant_rows=0, activation_token_stride=0, activation_slot_stride=0;
  bool grouped=false, shared=false;
  const void *quantize_kernel=nullptr, *linear_kernel=nullptr;
};
cudaError_t prepare_kernel(Program &) noexcept;
cudaError_t launch(const Program &,const void *const *inputs,void *workspace,
                   void *output,unsigned *status,cudaStream_t) noexcept;
} // namespace hyle::compute::cuda_native::blockwise_w4a8
