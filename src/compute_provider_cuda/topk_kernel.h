#pragma once

#include <cstddef>

#include "native_kernel.h"

namespace hyle::compute::cuda_native::native {

struct TopKLayout final {
  unsigned rows = 0;
  unsigned width = 0;
  unsigned count = 0;
  unsigned inner = 1;
  unsigned rank = 0;
  unsigned axis = 0;
  unsigned dimensions[8]{};
  std::size_t strides[8]{};
};
struct TopKProgram final {
  const void *kernel = nullptr;
  TopKLayout layout{};
};
cudaError_t topk_prepare(ElementType type, bool largest, bool dense,
                         const TopKLayout &layout,
                         TopKProgram *output) noexcept;
cudaError_t topk_launch(const TopKProgram &program, const void *input,
                        void *values, void *indices,
                        cudaStream_t stream) noexcept;

} // namespace hyle::compute::cuda_native::native
