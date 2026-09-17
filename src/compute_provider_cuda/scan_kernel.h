#pragma once

#include <cstddef>

#include "native_kernel.h"

namespace hyle::compute::cuda_native::native {
struct ScanLayout final {
  unsigned rows = 0;
  unsigned width = 0;
  unsigned inner = 1;
  unsigned rank = 0;
  unsigned dimensions[8]{};
  std::size_t strides[8]{};
};
struct ScanProgram final {
  const void *validate = nullptr;
  const void *write = nullptr;
  ScanLayout layout{};
  unsigned threads = 32;
};
cudaError_t scan_prepare(bool integer, bool exclusive, bool dense,
                         const ScanLayout &layout,
                         ScanProgram *output) noexcept;
cudaError_t scan_launch(const ScanProgram &program, const void *input,
                        void *output, unsigned *status,
                        cudaStream_t stream) noexcept;
} // namespace hyle::compute::cuda_native::native
