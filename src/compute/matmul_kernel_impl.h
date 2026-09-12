#pragma once

#include <algorithm>
#include <cstddef>

#include "hwy/highway.h"

#include "matmul_kernels.h"

#define HYLE_COMPUTE_CAT_I(left, right) left##right
#define HYLE_COMPUTE_CAT(left, right) HYLE_COMPUTE_CAT_I(left, right)

namespace hyle::compute::detail {
namespace {

// Six rows by two SIMD vectors is the largest ordinary f32 outer-product tile
// that fits the 16-register AVX2/SSE floor without spilling: 12 accumulators,
// two B vectors and one reusable A broadcast. It is also the mature BRGEMM
// register-balance used by oneDNN's AVX2 blocking calculation. The caller may
// bound the column range so all row tiles consume a B slab while it remains in
// private cache.
HWY_INLINE void matmul_rows6(const float *left, const float *right,
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
    V c40 = hn::Zero(d);
    V c41 = hn::Zero(d);
    V c50 = hn::Zero(d);
    V c51 = hn::Zero(d);
    const float *right_row = right + column;
    for (std::size_t k = 0; k < inner; ++k, right_row += columns) {
      const V b0 = hn::LoadU(d, right_row);
      const V b1 = hn::LoadU(d, right_row + lanes);
      V a = hn::Set(d, left[k]);
      c00 = hn::MulAdd(a, b0, c00);
      c01 = hn::MulAdd(a, b1, c01);
      a = hn::Set(d, left[inner + k]);
      c10 = hn::MulAdd(a, b0, c10);
      c11 = hn::MulAdd(a, b1, c11);
      a = hn::Set(d, left[inner * 2 + k]);
      c20 = hn::MulAdd(a, b0, c20);
      c21 = hn::MulAdd(a, b1, c21);
      a = hn::Set(d, left[inner * 3 + k]);
      c30 = hn::MulAdd(a, b0, c30);
      c31 = hn::MulAdd(a, b1, c31);
      a = hn::Set(d, left[inner * 4 + k]);
      c40 = hn::MulAdd(a, b0, c40);
      c41 = hn::MulAdd(a, b1, c41);
      a = hn::Set(d, left[inner * 5 + k]);
      c50 = hn::MulAdd(a, b0, c50);
      c51 = hn::MulAdd(a, b1, c51);
    }
    hn::StoreU(c00, d, output + column);
    hn::StoreU(c01, d, output + column + lanes);
    hn::StoreU(c10, d, output + columns + column);
    hn::StoreU(c11, d, output + columns + column + lanes);
    hn::StoreU(c20, d, output + columns * 2 + column);
    hn::StoreU(c21, d, output + columns * 2 + column + lanes);
    hn::StoreU(c30, d, output + columns * 3 + column);
    hn::StoreU(c31, d, output + columns * 3 + column + lanes);
    hn::StoreU(c40, d, output + columns * 4 + column);
    hn::StoreU(c41, d, output + columns * 4 + column + lanes);
    hn::StoreU(c50, d, output + columns * 5 + column);
    hn::StoreU(c51, d, output + columns * 5 + column + lanes);
  }

  for (; column + lanes <= last_column; column += lanes) {
    V c0 = hn::Zero(d);
    V c1 = hn::Zero(d);
    V c2 = hn::Zero(d);
    V c3 = hn::Zero(d);
    V c4 = hn::Zero(d);
    V c5 = hn::Zero(d);
    const float *right_row = right + column;
    for (std::size_t k = 0; k < inner; ++k, right_row += columns) {
      const V b = hn::LoadU(d, right_row);
      c0 = hn::MulAdd(hn::Set(d, left[k]), b, c0);
      c1 = hn::MulAdd(hn::Set(d, left[inner + k]), b, c1);
      c2 = hn::MulAdd(hn::Set(d, left[inner * 2 + k]), b, c2);
      c3 = hn::MulAdd(hn::Set(d, left[inner * 3 + k]), b, c3);
      c4 = hn::MulAdd(hn::Set(d, left[inner * 4 + k]), b, c4);
      c5 = hn::MulAdd(hn::Set(d, left[inner * 5 + k]), b, c5);
    }
    hn::StoreU(c0, d, output + column);
    hn::StoreU(c1, d, output + columns + column);
    hn::StoreU(c2, d, output + columns * 2 + column);
    hn::StoreU(c3, d, output + columns * 3 + column);
    hn::StoreU(c4, d, output + columns * 4 + column);
    hn::StoreU(c5, d, output + columns * 5 + column);
  }

  for (; column < last_column; ++column) {
    float c0 = 0.0F;
    float c1 = 0.0F;
    float c2 = 0.0F;
    float c3 = 0.0F;
    float c4 = 0.0F;
    float c5 = 0.0F;
    const float *right_value = right + column;
    for (std::size_t k = 0; k < inner; ++k, right_value += columns) {
      const float b = *right_value;
      c0 += left[k] * b;
      c1 += left[inner + k] * b;
      c2 += left[inner * 2 + k] * b;
      c3 += left[inner * 3 + k] * b;
      c4 += left[inner * 4 + k] * b;
      c5 += left[inner * 5 + k] * b;
    }
    output[column] = c0;
    output[columns + column] = c1;
    output[columns * 2 + column] = c2;
    output[columns * 3 + column] = c3;
    output[columns * 4 + column] = c4;
    output[columns * 5 + column] = c5;
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
    for (; row + 6 <= rows; row += 6) {
      matmul_rows6(left + row * inner, right, output + row * columns, inner,
                   columns, first, last);
    }
    for (; row + 4 <= rows; row += 4) {
      matmul_rows4(left + row * inner, right, output + row * columns, inner,
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
