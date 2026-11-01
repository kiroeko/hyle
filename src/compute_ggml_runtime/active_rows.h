#pragma once

#include <cstdint>

#include <ggml.h>

// Same-build Commands-only matrix contract. Native headers retain physical C;
// a trusted cold configuration supplies L from the original checked offsets.
// This metadata grants no storage permission and never changes a shared recipe.
namespace hyle::compute::ggml_cuda::internal {
inline constexpr std::int32_t kActiveRowsMatMulContract = 0x41524d31;
inline constexpr int kActiveRowsMatMulParameter = 5;
inline constexpr int kActiveRowsMatMulRowsParameter = 6;

inline bool is_active_rows_matmul(const ggml_tensor *tensor) noexcept {
  return tensor && tensor->op == GGML_OP_MUL_MAT &&
      tensor->op_params[kActiveRowsMatMulParameter] == kActiveRowsMatMulContract;
}
inline void set_matmul_active_rows(ggml_tensor *tensor, std::int32_t rows) noexcept {
  tensor->op_params[kActiveRowsMatMulRowsParameter] = rows;
}
} // namespace hyle::compute::ggml_cuda::internal

// Verify the actual linked native implementation before opting into ARM1.
extern "C" std::int32_t ggml_hyle_cuda_matmul_active_rows_contract();
