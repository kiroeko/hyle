#include "cuda_paged_append.h"

#include <cuda_runtime.h>

namespace hyle::compute::cuda::detail {
namespace {

// Each block owns one complete, disjoint destination range. All offsets are
// row coordinates in the common wire format; F16/BF16 payloads remain bits.
__global__ void append_rows(
    void *arena_bytes, const void *input_bytes, const void *metadata_bytes,
    compute::detail::PagedAppendNativeSpec spec) {
  const auto *metadata =
      static_cast<const compute::detail::PagedAppendMetadata *>(metadata_bytes);
  const auto active = metadata->active_count;
  if (active > spec.instruction_capacity || metadata->reserved0 != 0 ||
      metadata->reserved1 != 0 || metadata->reserved2 != 0)
    return;
  const auto *instructions = reinterpret_cast<
      const compute::detail::PagedAppendInstruction *>(metadata + 1);
  auto *arena = static_cast<std::uint16_t *>(arena_bytes);
  const auto *input = static_cast<const std::uint16_t *>(input_bytes);
  const auto row_elements = spec.row_bytes / sizeof(std::uint16_t);
  for (std::uint64_t index = blockIdx.x; index < active; index += gridDim.x) {
    const auto instruction = instructions[index];
    // Common validated the whole host episode before upload. These guards
    // also prevent a malformed device record from reading/writing payload.
    if (instruction.reserved0 != 0 || instruction.reserved1 != 0 ||
        instruction.rows == 0 || instruction.source_kind > 1 ||
        instruction.destination_page >= spec.physical_pages ||
        instruction.destination_row > spec.page_rows ||
        instruction.rows > spec.page_rows - instruction.destination_row)
      continue;
    const std::uint16_t *source = nullptr;
    if (instruction.source_kind == 0) {
      if (instruction.source_row > spec.input_rows ||
          instruction.rows > spec.input_rows - instruction.source_row)
        continue;
      source = input + std::size_t(instruction.source_row) * row_elements;
    } else {
      if (instruction.source_page >= spec.physical_pages ||
          instruction.source_row > spec.page_rows ||
          instruction.rows > spec.page_rows - instruction.source_row)
        continue;
      source = arena +
          (std::size_t(instruction.source_page) * spec.page_rows +
           instruction.source_row) * row_elements;
    }
    auto *destination = arena +
        (std::size_t(instruction.destination_page) * spec.page_rows +
         instruction.destination_row) * row_elements;
    const auto elements = std::size_t(instruction.rows) * row_elements;
    for (std::size_t element = threadIdx.x; element < elements;
         element += blockDim.x)
      destination[element] = source[element];
  }
}

} // namespace

const void *paged_append_kernel() noexcept {
  return reinterpret_cast<const void *>(append_rows);
}

cudaError_t launch_paged_append(
    const void *kernel, void *arena, const void *rows, const void *metadata,
    compute::detail::PagedAppendNativeSpec spec, cudaStream_t stream) noexcept {
  // The bound depends only on cold query/request capacity, never KV history.
  // A grid-stride instruction loop also keeps very large capacities within
  // CUDA's portable one-dimensional grid limit without another native call.
  const auto blocks = spec.instruction_capacity < 65535U
      ? spec.instruction_capacity : 65535U;
  void *arguments[]{&arena, &rows, &metadata, &spec};
  return cudaLaunchKernel(kernel, dim3(blocks), dim3(256), arguments, 0, stream);
}

} // namespace hyle::compute::cuda::detail
