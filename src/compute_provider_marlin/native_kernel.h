#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

namespace hyle::compute::marlin::native {

// Private C++20 CUDA translation-unit boundary. No HYLE C++23 runtime types
// or third-party template definitions leak into the public C++ interface.
struct Program final {
  const void *kernel = nullptr;
  int rows = 0;
  int columns = 0;
  int inner = 0;
  int blocks = 0;
  int threads = 0;
  int shared_bytes = 0;
  std::size_t locks_offset = 0;
  std::size_t workspace_bytes = 0;
};

cudaError_t prepare(int rows, int columns, int inner, int multiprocessors,
                    int maximum_shared_bytes, Program *output) noexcept;
cudaError_t launch(const Program &program, const void *activation,
                   const void *weights, const void *scales, void *output,
                   void *workspace, cudaStream_t stream) noexcept;
cudaError_t prepare_repack(int maximum_shared_bytes) noexcept;
cudaError_t repack(const std::uint32_t *source, void *destination, int inner,
                   int columns, int blocks, int shared_bytes,
                   cudaStream_t stream) noexcept;

} // namespace hyle::compute::marlin::native
