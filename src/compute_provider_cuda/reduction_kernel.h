#pragma once

#include <cstddef>

#include "native_kernel.h"

namespace hyle::compute::cuda_native::native {

enum class ReductionKind {
  kSum,
  kMean,
  kMinimum,
  kMaximum,
  kSoftmax,
  kLogSoftmax
};
enum class ReductionMask { kNone, kBoolean, kAdditive };

// Row coordinates enumerate retained axes; column coordinates enumerate reduced
// axes. All dimensions and bindings are qualified before a kernel is selected.
struct ReductionLayout final {
  unsigned rows = 0;
  unsigned width = 0;
  unsigned inner = 1;
  unsigned rank = 0;
  unsigned axes = 0;
  unsigned dimensions[8]{};
  std::size_t input_strides[8]{};
  std::size_t mask_strides[8]{};
  float scale = 1;
};
struct ReductionProgram final {
  const void *kernel = nullptr;
  ReductionLayout layout{};
  unsigned shared_bytes = 0;
};

cudaError_t reduction_prepare(ElementType type, ReductionKind kind,
                              ReductionMask mask, bool dense,
                              const ReductionLayout &layout,
                              ReductionProgram *output) noexcept;
cudaError_t reduction_launch(const ReductionProgram &program, const void *input,
                             const void *mask, void *output,
                             cudaStream_t stream) noexcept;

} // namespace hyle::compute::cuda_native::native
