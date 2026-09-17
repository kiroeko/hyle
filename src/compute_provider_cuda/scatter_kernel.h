#pragma once

#include <cstddef>

#include "native_kernel.h"

namespace hyle::compute::cuda_native::native {
struct ScatterLayout final {
  unsigned elements = 0;
  unsigned updates = 0;
  unsigned indices = 0;
  unsigned inner = 1;
  unsigned bytes = 0;
  unsigned axis = 0;
  unsigned extent = 0;
  unsigned source_rank = 0;
  unsigned index_rank = 0;
  unsigned update_rank = 0;
  unsigned dimensions[8]{};
  std::size_t source_strides[8]{};
  unsigned index_dimensions[8]{};
  std::size_t index_strides[8]{};
  unsigned update_dimensions[8]{};
  std::size_t update_strides[8]{};
};
struct ScatterProgram final {
  const void *pack = nullptr;
  const void *copy = nullptr;
  const void *update = nullptr;
  const void *single = nullptr;
  ScatterLayout layout{};
  std::size_t array_bytes = 0;
  std::size_t sort_bytes = 0;
  std::size_t workspace_bytes = 0;
  int sort_bits = 1;
  bool checked = false;
};
struct ScatterSort final {
  void *workspace = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
};
cudaError_t scatter_prepare(ElementType type, bool add, bool index64,
                            bool source_dense, bool indices_dense,
                            bool updates_dense, bool checked,
                            const ScatterLayout &layout,
                            ScatterProgram *output) noexcept;
cudaError_t scatter_sort_create(const ScatterProgram &program, void *workspace,
                                ScatterSort *output) noexcept;
void scatter_sort_destroy(ScatterSort &sort) noexcept;
cudaError_t scatter_launch(const ScatterProgram &program,
                           const ScatterSort *sort, const void *source,
                           const void *indices, const void *updates,
                           void *output, unsigned *status,
                           cudaStream_t stream) noexcept;
} // namespace hyle::compute::cuda_native::native
