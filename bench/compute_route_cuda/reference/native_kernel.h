#pragma once

#include <cuda_runtime_api.h>

namespace hyle::compute::sglang_reference::native {

// Private CUDA translation-unit seam. Public consumers only need C++23.
struct Program final {
  const void *kernel = nullptr;
  unsigned elements = 0;
  unsigned hidden = 0;
  unsigned blocks = 0;
};

cudaError_t prepare(bool half, bool packed, unsigned elements, unsigned hidden,
                    Program *output) noexcept;
cudaError_t launch(const Program &program, const void *first,
                   const void *second, void *output,
                   cudaStream_t stream) noexcept;

} // namespace hyle::compute::sglang_reference::native
