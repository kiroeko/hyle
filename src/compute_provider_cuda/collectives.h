#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

namespace hyle::compute::cuda_native::collectives {

enum class Kind : std::uint8_t { kScan, kTopK, kGather };
enum class Type : std::uint8_t { kFloat32, kInt32, kFloat16, kBFloat16 };

// Private cold description. CUDA templates never include the public C++23 API.
struct Program final {
  Kind kind = Kind::kScan;
  Type type = Type::kFloat32;
  std::uint32_t rank = 0;
  std::uint64_t dimensions[8]{};
  std::uint64_t strides[8]{};
  std::size_t count = 0;
  std::size_t width = 0;
  std::size_t inner = 1;
  std::size_t k = 0;
  bool dense = false;
  bool exclusive = false;
  bool largest = true;
  std::uint32_t index_rank = 0;
  std::uint64_t index_dimensions[8]{};
  std::uint64_t index_strides[8]{};
  std::size_t index_count = 0;
  std::size_t element_bytes = 0;
  bool index_i64 = false;
  bool checked = true;
  std::size_t workspace_bytes = 0;
  std::size_t temporary_offset = 0;
  std::size_t temporary_bytes = 0;
  std::size_t prefix_offset = 0;
  std::size_t bounds_offset = 0;
  std::size_t keys_in_offset = 0;
  std::size_t keys_out_offset = 0;
  std::size_t indices_in_offset = 0;
  std::size_t indices_out_offset = 0;
};

const char *version() noexcept;
cudaError_t configure(Program *program) noexcept;
cudaError_t submit(const Program &program, const void *input,
                   const void *index_input, void *values, void *indices,
                   void *workspace, cudaStream_t stream,
                   bool *validation_failed) noexcept;

} // namespace hyle::compute::cuda_native::collectives
