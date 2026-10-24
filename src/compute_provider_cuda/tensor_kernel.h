#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

namespace hyle::compute::cuda_native::native {

enum class TensorKind {
  kCopy,
  kFill,
  kCast,
  kSelect,
  kAbs,
  kNeg,
  kExp,
  kLog,
  kSqrt,
  kRsqrt,
  kTanh,
  kSigmoid,
  kRelu,
  kGelu,
  kGeluTanh,
  kSilu,
  kSoftplus,
  kAdd,
  kSubtract,
  kMultiply,
  kDivide,
  kMinimum,
  kMaximum,
};

enum class TensorElementType {
  kFloat32, kFloat16, kBFloat16, kFloat8E4M3FN, kFloat8E5M2, kBool,
  kInt32,
};

// CUDA-only launch metadata. Strides are bytes, including zero broadcasts;
// generic kernels load bytes safely even for odd strides. Dense instances
// compile away coordinate mapping. No public descriptor crosses this seam.
struct TensorLayout final {
  std::uint64_t elements = 0;
  unsigned rank = 0;
  std::uint64_t dimensions[8]{};
  std::size_t strides[3][8]{};
  std::uint64_t scalar_bits = 0;
};

struct TensorProgram final {
  const void *kernel = nullptr;
  unsigned blocks = 0;
  TensorLayout layout{};
};

cudaError_t tensor_prepare(TensorKind kind, TensorElementType output_type,
                           TensorElementType input_type, unsigned storage_width,
                           bool dense, const TensorLayout &layout,
                           TensorProgram *output) noexcept;
cudaError_t tensor_launch(const TensorProgram &program, const void *first,
                          const void *second, const void *third, void *output,
                          cudaStream_t stream) noexcept;

} // namespace hyle::compute::cuda_native::native
