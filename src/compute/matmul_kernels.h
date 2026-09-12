#pragma once

#include "native_route.h"

namespace hyle::compute::detail {

#define HYLE_COMPUTE_DECLARE_ROW_MAJOR_MATMUL(suffix)                          \
  void run_matmul_row_major_##suffix(                                          \
      const float *left, const float *right, float *output,                    \
      const NativeParameters &parameters, float unused) noexcept

HYLE_COMPUTE_DECLARE_ROW_MAJOR_MATMUL(scalar);
HYLE_COMPUTE_DECLARE_ROW_MAJOR_MATMUL(sse4);
HYLE_COMPUTE_DECLARE_ROW_MAJOR_MATMUL(avx2);
HYLE_COMPUTE_DECLARE_ROW_MAJOR_MATMUL(avx512);
HYLE_COMPUTE_DECLARE_ROW_MAJOR_MATMUL(neon);
HYLE_COMPUTE_DECLARE_ROW_MAJOR_MATMUL(sve);
HYLE_COMPUTE_DECLARE_ROW_MAJOR_MATMUL(rvv);

#undef HYLE_COMPUTE_DECLARE_ROW_MAJOR_MATMUL

} // namespace hyle::compute::detail
