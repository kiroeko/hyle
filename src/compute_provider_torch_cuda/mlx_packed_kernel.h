#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace hyle::compute::torch_cuda::native {
enum class MlxFormat { kAffine4, kMxfp4, kMxfp8 };
enum class MlxKind { kLinear, kGrouped, kEmbedding };
enum class Scalar { kFloat16, kBFloat16, kFloat32 };
struct MlxPacked final {
  MlxFormat format;
  MlxKind kind;
  Scalar activation_type, scale_type, output_type;
  std::uint32_t group;
  std::uint64_t rows, columns, depth, experts;
};
cudaError_t mlx_packed_launch(const MlxPacked &program,
    const void *const *inputs, void *output, unsigned *status,
    cudaStream_t stream) noexcept;
} // namespace hyle::compute::torch_cuda::native
