#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

namespace hyle::compute::cuda_native::sampling {
enum class Type : std::uint8_t { kFloat32, kFloat16, kBFloat16 };
struct Program final {
  bool random_bits = false;
  bool log_probability = false;
  Type type = Type::kFloat32;
  std::uint32_t rows = 0, width = 0, chunks = 0;
  std::size_t row_max_offset = 0, partial_offset = 0, workspace_bytes = 0;
};
cudaError_t launch(const Program &, const void *logits,
                    const std::uint32_t *identities, void *indices,
                    float *log_probabilities, void *workspace, unsigned *status,
                    cudaStream_t stream) noexcept;
} // namespace hyle::compute::cuda_native::sampling
