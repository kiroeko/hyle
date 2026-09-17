#pragma once

#include <cstddef>

#include "native_kernel.h"

namespace hyle::compute::cuda_native::native {
struct GatherLayout final {
  unsigned elements = 0;
  unsigned indices = 0;
  unsigned inner = 1;
  unsigned copy_bytes = 0;
  unsigned axis = 0;
  unsigned source_rank = 0;
  unsigned index_rank = 0;
  std::size_t axis_extent = 0;
  std::size_t dimensions[8]{};
  std::size_t strides[8]{};
  unsigned index_dimensions[8]{};
  std::size_t index_strides[8]{};
};
struct GatherProgram final {
  const void *validate = nullptr;
  const void *gather = nullptr;
  GatherLayout layout{};
};
cudaError_t gather_prepare(bool index64, bool source_dense, bool indices_dense,
                           bool checked, const GatherLayout &layout,
                           GatherProgram *output) noexcept;
cudaError_t gather_launch(const GatherProgram &program, const void *source,
                          const void *indices, void *output, unsigned *status,
                          cudaStream_t stream) noexcept;
} // namespace hyle::compute::cuda_native::native
