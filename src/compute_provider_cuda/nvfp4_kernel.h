#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace hyle::compute::cuda_native::nvfp4 {
enum class Contract : std::uint8_t {
  kModelOpt, kLocalFloat32, kLocalBFloat16, kMxfp4, kMxfp4BFloat16
};
struct Program final {
  std::uint64_t rows = 0, columns = 0, depth = 0;
  // Selected output rows and activation QDQ/validation rows differ for shared A.
  std::uint64_t quant_rows = 0;
  std::uint64_t slots = 1, experts = 1, activation_slots = 1;
  // BF16 element strides of the original grouped activation, not an expanded A.
  std::uint64_t activation_token_stride = 0, activation_slot_stride = 0;
  bool grouped = false;
  Contract contract = Contract::kModelOpt;
  // Resolved during prepare, including driver function materialization.
  // BF16-only MXFP4 uses an input validator in place of the quantizer.
  const void *quantize_kernel = nullptr, *linear_kernel = nullptr;
};
cudaError_t prepare_kernel(Program &) noexcept;
cudaError_t launch(const Program &, const void *const *inputs, void *activation,
                   void *output, unsigned *status, cudaStream_t) noexcept;
} // namespace hyle::compute::cuda_native::nvfp4
