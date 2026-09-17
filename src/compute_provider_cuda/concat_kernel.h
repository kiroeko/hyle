#pragma once

#include "native_kernel.h"
#include <cstddef>

namespace hyle::compute::cuda_native::native {
inline constexpr unsigned kConcatInputs = 4;
struct ConcatLayout final {
  unsigned elements = 0;
  unsigned inputs = 0;
  unsigned inner = 1;
  unsigned output_row = 0;
  unsigned output_offset = 0;
  unsigned rank = 0;
  unsigned axis = 0;
  unsigned chunks[kConcatInputs]{};
  unsigned dimensions[8]{};
  std::size_t strides[kConcatInputs][8]{};
};
struct ConcatPointers final {
  const void *input[kConcatInputs]{};
};
struct ConcatProgram final {
  const void *kernel = nullptr;
  ConcatLayout layout{};
};
cudaError_t concat_prepare(unsigned scalar_bytes, bool dense,
                           const ConcatLayout &layout,
                           ConcatProgram *output) noexcept;
cudaError_t concat_launch(const ConcatProgram &program,
                          const ConcatPointers &inputs, void *output,
                          cudaStream_t stream) noexcept;
} // namespace hyle::compute::cuda_native::native
