#pragma once

#include <algorithm>
#include <cstddef>

#include "hwy/highway.h"

#include "matmul_kernels.h"

#define HYLE_COMPUTE_CAT_I(left, right) left##right
#define HYLE_COMPUTE_CAT(left, right) HYLE_COMPUTE_CAT_I(left, right)

namespace hyle::compute::detail {
namespace {

// The 16-register AVX2/SSE floor favors the same 3-row by 4-vector balance as
// oneDNN's ordinary f32 BRGEMM heuristic: 12 accumulators, three A broadcasts
// and one rotating B load. Keeping four B vectors in flight would spill, so B
// is deliberately loaded one vector at a time while each A broadcast is reused
// four times. The caller bounds the column range so every row tile consumes a
// B slab while it remains in private cache. Unlike oneDNN's register-controlled
// JIT, this portable C++ body does not force a K unroll that makes MSVC spill
// the accumulators.
HWY_INLINE void matmul_rows3(const float *left, const float *right,
                             float *output, std::size_t inner,
                             std::size_t columns, std::size_t first_column,
                             std::size_t last_column) noexcept {
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<float> d;
  using V = decltype(hn::Zero(d));
  const std::size_t lanes = hn::Lanes(d);
  std::size_t column = first_column;

  for (; column + lanes * 4 <= last_column; column += lanes * 4) {
    V c00 = hn::Zero(d);
    V c01 = hn::Zero(d);
    V c02 = hn::Zero(d);
    V c03 = hn::Zero(d);
    V c10 = hn::Zero(d);
    V c11 = hn::Zero(d);
    V c12 = hn::Zero(d);
    V c13 = hn::Zero(d);
    V c20 = hn::Zero(d);
    V c21 = hn::Zero(d);
    V c22 = hn::Zero(d);
    V c23 = hn::Zero(d);
    const float *right_row = right + column;
    for (std::size_t k = 0; k < inner; ++k, right_row += columns) {
      const V a0 = hn::Set(d, left[k]);
      const V a1 = hn::Set(d, left[inner + k]);
      const V a2 = hn::Set(d, left[inner * 2 + k]);
      V b = hn::LoadU(d, right_row);
      c00 = hn::MulAdd(a0, b, c00);
      c10 = hn::MulAdd(a1, b, c10);
      c20 = hn::MulAdd(a2, b, c20);
      b = hn::LoadU(d, right_row + lanes);
      c01 = hn::MulAdd(a0, b, c01);
      c11 = hn::MulAdd(a1, b, c11);
      c21 = hn::MulAdd(a2, b, c21);
      b = hn::LoadU(d, right_row + lanes * 2);
      c02 = hn::MulAdd(a0, b, c02);
      c12 = hn::MulAdd(a1, b, c12);
      c22 = hn::MulAdd(a2, b, c22);
      b = hn::LoadU(d, right_row + lanes * 3);
      c03 = hn::MulAdd(a0, b, c03);
      c13 = hn::MulAdd(a1, b, c13);
      c23 = hn::MulAdd(a2, b, c23);
    }
    hn::StoreU(c00, d, output + column);
    hn::StoreU(c01, d, output + column + lanes);
    hn::StoreU(c02, d, output + column + lanes * 2);
    hn::StoreU(c03, d, output + column + lanes * 3);
    hn::StoreU(c10, d, output + columns + column);
    hn::StoreU(c11, d, output + columns + column + lanes);
    hn::StoreU(c12, d, output + columns + column + lanes * 2);
    hn::StoreU(c13, d, output + columns + column + lanes * 3);
    hn::StoreU(c20, d, output + columns * 2 + column);
    hn::StoreU(c21, d, output + columns * 2 + column + lanes);
    hn::StoreU(c22, d, output + columns * 2 + column + lanes * 2);
    hn::StoreU(c23, d, output + columns * 2 + column + lanes * 3);
  }

  for (; column + lanes <= last_column; column += lanes) {
    V c0 = hn::Zero(d);
    V c1 = hn::Zero(d);
    V c2 = hn::Zero(d);
    const float *right_row = right + column;
    for (std::size_t k = 0; k < inner; ++k, right_row += columns) {
      const V b = hn::LoadU(d, right_row);
      c0 = hn::MulAdd(hn::Set(d, left[k]), b, c0);
      c1 = hn::MulAdd(hn::Set(d, left[inner + k]), b, c1);
      c2 = hn::MulAdd(hn::Set(d, left[inner * 2 + k]), b, c2);
    }
    hn::StoreU(c0, d, output + column);
    hn::StoreU(c1, d, output + columns + column);
    hn::StoreU(c2, d, output + columns * 2 + column);
  }

  for (; column < last_column; ++column) {
    float c0 = 0.0F;
    float c1 = 0.0F;
    float c2 = 0.0F;
    const float *right_value = right + column;
    for (std::size_t k = 0; k < inner; ++k, right_value += columns) {
      const float b = *right_value;
      c0 += left[k] * b;
      c1 += left[inner + k] * b;
      c2 += left[inner * 2 + k] * b;
    }
    output[column] = c0;
    output[columns + column] = c1;
    output[columns * 2 + column] = c2;
  }
}

// Four rows by two SIMD vectors fits the 16-register x86-64 floor while
// reusing every B load across four output rows. The single-row path keeps four
// adjacent B vectors live, which avoids a page-stride access pattern for the
// decode case. Both kernels preserve K accumulation order and require neither
// packing nor scratch storage.
HWY_INLINE void matmul_rows4(const float *left, const float *right,
                             float *output, std::size_t inner,
                             std::size_t columns, std::size_t first_column,
                             std::size_t last_column) noexcept {
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<float> d;
  using V = decltype(hn::Zero(d));
  const std::size_t lanes = hn::Lanes(d);
  std::size_t column = first_column;

  for (; column + lanes * 2 <= last_column; column += lanes * 2) {
    V c00 = hn::Zero(d);
    V c01 = hn::Zero(d);
    V c10 = hn::Zero(d);
    V c11 = hn::Zero(d);
    V c20 = hn::Zero(d);
    V c21 = hn::Zero(d);
    V c30 = hn::Zero(d);
    V c31 = hn::Zero(d);
    const float *right_row = right + column;
    for (std::size_t k = 0; k < inner; ++k, right_row += columns) {
      const V b0 = hn::LoadU(d, right_row);
      const V b1 = hn::LoadU(d, right_row + lanes);
      const V a0 = hn::Set(d, left[k]);
      const V a1 = hn::Set(d, left[inner + k]);
      const V a2 = hn::Set(d, left[inner * 2 + k]);
      const V a3 = hn::Set(d, left[inner * 3 + k]);
      c00 = hn::MulAdd(a0, b0, c00);
      c01 = hn::MulAdd(a0, b1, c01);
      c10 = hn::MulAdd(a1, b0, c10);
      c11 = hn::MulAdd(a1, b1, c11);
      c20 = hn::MulAdd(a2, b0, c20);
      c21 = hn::MulAdd(a2, b1, c21);
      c30 = hn::MulAdd(a3, b0, c30);
      c31 = hn::MulAdd(a3, b1, c31);
    }
    hn::StoreU(c00, d, output + column);
    hn::StoreU(c01, d, output + column + lanes);
    hn::StoreU(c10, d, output + columns + column);
    hn::StoreU(c11, d, output + columns + column + lanes);
    hn::StoreU(c20, d, output + columns * 2 + column);
    hn::StoreU(c21, d, output + columns * 2 + column + lanes);
    hn::StoreU(c30, d, output + columns * 3 + column);
    hn::StoreU(c31, d, output + columns * 3 + column + lanes);
  }

  for (; column + lanes <= last_column; column += lanes) {
    V c0 = hn::Zero(d);
    V c1 = hn::Zero(d);
    V c2 = hn::Zero(d);
    V c3 = hn::Zero(d);
    const float *right_row = right + column;
    for (std::size_t k = 0; k < inner; ++k, right_row += columns) {
      const V b = hn::LoadU(d, right_row);
      c0 = hn::MulAdd(hn::Set(d, left[k]), b, c0);
      c1 = hn::MulAdd(hn::Set(d, left[inner + k]), b, c1);
      c2 = hn::MulAdd(hn::Set(d, left[inner * 2 + k]), b, c2);
      c3 = hn::MulAdd(hn::Set(d, left[inner * 3 + k]), b, c3);
    }
    hn::StoreU(c0, d, output + column);
    hn::StoreU(c1, d, output + columns + column);
    hn::StoreU(c2, d, output + columns * 2 + column);
    hn::StoreU(c3, d, output + columns * 3 + column);
  }

  for (; column < last_column; ++column) {
    float c0 = 0.0F;
    float c1 = 0.0F;
    float c2 = 0.0F;
    float c3 = 0.0F;
    const float *right_value = right + column;
    for (std::size_t k = 0; k < inner; ++k, right_value += columns) {
      const float b = *right_value;
      c0 += left[k] * b;
      c1 += left[inner + k] * b;
      c2 += left[inner * 2 + k] * b;
      c3 += left[inner * 3 + k] * b;
    }
    output[column] = c0;
    output[columns + column] = c1;
    output[columns * 2 + column] = c2;
    output[columns * 3 + column] = c3;
  }
}

// A two-row tail keeps the same four-vector N width as the prefill kernel.
// Sharing each B load across both remaining rows avoids falling back to two
// independent GEMV passes for common blocks such as M=32 (10*3+2).
HWY_INLINE void matmul_rows2(const float *left, const float *right,
                             float *output, std::size_t inner,
                             std::size_t columns, std::size_t first_column,
                             std::size_t last_column) noexcept {
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<float> d;
  using V = decltype(hn::Zero(d));
  const std::size_t lanes = hn::Lanes(d);
  std::size_t column = first_column;

  for (; column + lanes * 4 <= last_column; column += lanes * 4) {
    V c00 = hn::Zero(d);
    V c01 = hn::Zero(d);
    V c02 = hn::Zero(d);
    V c03 = hn::Zero(d);
    V c10 = hn::Zero(d);
    V c11 = hn::Zero(d);
    V c12 = hn::Zero(d);
    V c13 = hn::Zero(d);
    const float *right_row = right + column;
    for (std::size_t k = 0; k < inner; ++k, right_row += columns) {
      const V a0 = hn::Set(d, left[k]);
      const V a1 = hn::Set(d, left[inner + k]);
      V b = hn::LoadU(d, right_row);
      c00 = hn::MulAdd(a0, b, c00);
      c10 = hn::MulAdd(a1, b, c10);
      b = hn::LoadU(d, right_row + lanes);
      c01 = hn::MulAdd(a0, b, c01);
      c11 = hn::MulAdd(a1, b, c11);
      b = hn::LoadU(d, right_row + lanes * 2);
      c02 = hn::MulAdd(a0, b, c02);
      c12 = hn::MulAdd(a1, b, c12);
      b = hn::LoadU(d, right_row + lanes * 3);
      c03 = hn::MulAdd(a0, b, c03);
      c13 = hn::MulAdd(a1, b, c13);
    }
    hn::StoreU(c00, d, output + column);
    hn::StoreU(c01, d, output + column + lanes);
    hn::StoreU(c02, d, output + column + lanes * 2);
    hn::StoreU(c03, d, output + column + lanes * 3);
    hn::StoreU(c10, d, output + columns + column);
    hn::StoreU(c11, d, output + columns + column + lanes);
    hn::StoreU(c12, d, output + columns + column + lanes * 2);
    hn::StoreU(c13, d, output + columns + column + lanes * 3);
  }

  for (; column + lanes <= last_column; column += lanes) {
    V c0 = hn::Zero(d);
    V c1 = hn::Zero(d);
    const float *right_row = right + column;
    for (std::size_t k = 0; k < inner; ++k, right_row += columns) {
      const V b = hn::LoadU(d, right_row);
      c0 = hn::MulAdd(hn::Set(d, left[k]), b, c0);
      c1 = hn::MulAdd(hn::Set(d, left[inner + k]), b, c1);
    }
    hn::StoreU(c0, d, output + column);
    hn::StoreU(c1, d, output + columns + column);
  }

  for (; column < last_column; ++column) {
    float c0 = 0.0F;
    float c1 = 0.0F;
    const float *right_value = right + column;
    for (std::size_t k = 0; k < inner; ++k, right_value += columns) {
      const float b = *right_value;
      c0 += left[k] * b;
      c1 += left[inner + k] * b;
    }
    output[column] = c0;
    output[columns + column] = c1;
  }
}

HWY_INLINE void matmul_row1(const float *left, const float *right,
                            float *output, std::size_t inner,
                            std::size_t columns, std::size_t first_column,
                            std::size_t last_column) noexcept {
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<float> d;
  using V = decltype(hn::Zero(d));
  const std::size_t lanes = hn::Lanes(d);
  std::size_t column = first_column;

  for (; column + lanes * 4 <= last_column; column += lanes * 4) {
    V c0 = hn::Zero(d);
    V c1 = hn::Zero(d);
    V c2 = hn::Zero(d);
    V c3 = hn::Zero(d);
    const float *right_row = right + column;
    for (std::size_t k = 0; k < inner; ++k, right_row += columns) {
      const V a = hn::Set(d, left[k]);
      c0 = hn::MulAdd(a, hn::LoadU(d, right_row), c0);
      c1 = hn::MulAdd(a, hn::LoadU(d, right_row + lanes), c1);
      c2 = hn::MulAdd(a, hn::LoadU(d, right_row + lanes * 2), c2);
      c3 = hn::MulAdd(a, hn::LoadU(d, right_row + lanes * 3), c3);
    }
    hn::StoreU(c0, d, output + column);
    hn::StoreU(c1, d, output + column + lanes);
    hn::StoreU(c2, d, output + column + lanes * 2);
    hn::StoreU(c3, d, output + column + lanes * 3);
  }

  for (; column + lanes <= last_column; column += lanes) {
    V accumulator = hn::Zero(d);
    const float *right_row = right + column;
    for (std::size_t k = 0; k < inner; ++k, right_row += columns) {
      accumulator =
          hn::MulAdd(hn::Set(d, left[k]), hn::LoadU(d, right_row), accumulator);
    }
    hn::StoreU(accumulator, d, output + column);
  }

  for (; column < last_column; ++column) {
    float accumulator = 0.0F;
    const float *right_value = right + column;
    for (std::size_t k = 0; k < inner; ++k, right_value += columns)
      accumulator += left[k] * *right_value;
    output[column] = accumulator;
  }
}

} // namespace

void HYLE_COMPUTE_CAT(run_matmul_row_major_, HYLE_SIMD_VARIANT_SUFFIX)(
    const float *left, const float *right, float *output,
    const NativeParameters &parameters, float) noexcept {
  const std::size_t rows = parameters[0];
  const std::size_t inner = parameters[1];
  const std::size_t columns = parameters[2];
  if (rows == 0 || columns == 0)
    return;
  if (inner == 0) {
    std::fill_n(output, rows * columns, 0.0F);
    return;
  }

  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<float> d;
  const std::size_t lanes = hn::Lanes(d);

  // For a single/few row decode, wider adjacent-column tiles minimize loop and
  // broadcast overhead. Once several row tiles consume the same B matrix,
  // bound N so each slab fits comfortably below the private-cache capacity and
  // finish all rows before advancing. This is cache blocking, not packing: the
  // public dense layout remains untouched and submit still needs no workspace.
  // Eight AVX2 vectors are 64 f32 columns, matching oneDNN's ordinary BRGEMM
  // default N block and keeping the K=512 slab at 128 KiB, well below private
  // L2. Helpers are force-inlined, so this cache boundary does not multiply
  // Win64 register-save prologues.
  const std::size_t column_chunk = rows >= 16 ? lanes * 8 : columns;
  for (std::size_t first = 0; first < columns; first += column_chunk) {
    const std::size_t last = (std::min)(columns, first + column_chunk);
    std::size_t row = 0;
    if (rows >= 16) {
      for (; row + 3 <= rows; row += 3) {
        matmul_rows3(left + row * inner, right, output + row * columns, inner,
                     columns, first, last);
      }
    } else {
      for (; row + 4 <= rows; row += 4) {
        matmul_rows4(left + row * inner, right, output + row * columns, inner,
                     columns, first, last);
      }
    }
    for (; row + 2 <= rows; row += 2) {
      matmul_rows2(left + row * inner, right, output + row * columns, inner,
                   columns, first, last);
    }
    for (; row < rows; ++row) {
      matmul_row1(left + row * inner, right, output + row * columns, inner,
                  columns, first, last);
    }
  }
}

} // namespace hyle::compute::detail

#undef HYLE_COMPUTE_CAT
#undef HYLE_COMPUTE_CAT_I
