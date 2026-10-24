#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace hyle::compute::torch_cuda::native {
enum class MlxFormat { kAffine, kMxfp4, kMxfp8 };
enum class MlxKind { kLinear, kGrouped, kEmbedding };
enum class MlxOrientation { kTransposeWeight, kWeightAsStored };
enum class Scalar { kFloat16, kBFloat16, kFloat32 };
struct MlxPacked final {
  MlxFormat format;
  MlxKind kind;
  MlxOrientation orientation;
  Scalar activation_type, scale_type, output_type;
  std::uint32_t bits, group;
  std::uint64_t rows, columns, depth, experts;
  // Private host/native geometry, in activation elements. IDs/output retain
  // rows=T*S; shared activation slots use a real zero stride.
  std::uint64_t slots, activation_token_stride, activation_slot_stride;
};
enum class ArtifactFormat { kCtSymmetric, kCtAsymmetric, kGptqV1Int4, kNf4DoubleQuant };
struct ArtifactPacked final {
  ArtifactFormat format;
  Scalar scale_type, decode_type, operand_type;
  std::uint32_t group;
  std::uint64_t rows, columns, depth;
  // Original CT decoder banks stay flat in expert/output-row order. Nonzero
  // experts selects GroupedMatMul's F32 A/output, with real byte strides.
  std::uint64_t experts = 0, slots = 1;
  std::uint64_t activation_token_stride = 0, activation_slot_stride = 0,
                activation_depth_stride = sizeof(float);
};
struct StaticFloat8 final {
  Scalar activation_type, output_type;
  std::uint64_t rows, columns, depth;
  std::uint32_t activation_scale_stride = 0, weight_scale_stride = 0;
  std::uint32_t weight_scale_first = 0;
};
cudaError_t static_fp8_prepare(const StaticFloat8 &program) noexcept;
cudaError_t static_fp8_launch(const StaticFloat8 &program,
    const void *const *inputs, void *encoded_activation, void *output,
    cudaStream_t stream) noexcept;
cudaError_t mlx_packed_launch(const MlxPacked &program,
    const void *const *inputs, void *output, unsigned *status,
    cudaStream_t stream) noexcept;
cudaError_t artifact_packed_prepare(const ArtifactPacked &program,
    bool decode_only) noexcept;
cudaError_t artifact_packed_linear_launch(const ArtifactPacked &program,
    const void *const *inputs, void *output, unsigned *status,
    cudaStream_t stream) noexcept;
cudaError_t gptq_decode_launch(const ArtifactPacked &program,
    const void *const *inputs, void *output, unsigned *status,
    cudaStream_t stream) noexcept;
} // namespace hyle::compute::torch_cuda::native
