#pragma once

#include <cuda_runtime_api.h>

namespace hyle::compute::cuda_native::native {

// Private CUDA translation-unit seam. Public consumers only need C++23.
enum class ElementType { kFloat32, kFloat16, kBFloat16 };
enum class Activation { kSilu, kGelu, kGeluTanh };

struct Program final {
  const void *kernel = nullptr;
  unsigned elements = 0;
  unsigned hidden = 0;
  unsigned blocks = 0;
};

cudaError_t prepare(ElementType type, Activation activation, bool packed,
                    unsigned elements, unsigned hidden,
                    Program *output) noexcept;
cudaError_t launch(const Program &program, const void *first,
                   const void *second, void *output,
                   cudaStream_t stream) noexcept;

cudaError_t sglang_prepare(ElementType type, Activation activation, bool packed,
                           unsigned elements, unsigned hidden,
                           Program *output) noexcept;
cudaError_t sglang_launch(const Program &program, const void *first,
                          const void *second, void *output,
                          cudaStream_t stream) noexcept;

} // namespace hyle::compute::cuda_native::native
