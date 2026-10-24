// Original MLX row mapping/reduction, shared by typed packed codecs.
// Copyright © 2023-2026 Apple Inc. MIT; see third_party/mlx_cuda_quant/LICENSE
// and SOURCE.json for the pinned source and original equations.
#pragma once

#include "mlx_cuda_quant.h"
#include <type_traits>
#include <utility>

namespace hyle::compute::cuda_native::packed {
using Index = unsigned long long;

template<class Loader, class = void>
struct HasActivationValue : std::false_type {};
template<class Loader>
struct HasActivationValue<Loader, std::void_t<decltype(std::declval<Loader>().activation_value(
    static_cast<const void *>(nullptr), Index{}, Index{}))>> : std::true_type {};

// Original MLX qmv warp-to-row mapping/F32 FMA, warp reduction and group tree.
// Rows=4 shares only decoded weights; each activation keeps its own arithmetic.
template<class Output, unsigned Rows, class Loader, class Scalar>
__device__ __forceinline__ void packed_rows(Index rows, Index columns,
    Index depth, unsigned group_size, Scalar activation_type,
    const void *activation, void *output, Loader loader) {
  const Index groups = depth / group_size;
  const Index first = (Index(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  const Index stride = Index(gridDim.x) * blockDim.x / 32;
  const unsigned lane = threadIdx.x % 32;
  const Index row_tiles = rows / Rows + (rows % Rows != 0);
  for (Index i = first; i < row_tiles * columns; i += stride) {
    const Index row = (i / columns) * Rows, column = i % columns;
    const Index weight_row = loader.weight_row(row, column);
    float result[Rows] = {};
    for (Index group = 0; group < groups; ++group) {
      const auto decoded = loader.group(weight_row, group);
      float dot[Rows] = {};
      for (typename Loader::GroupIndex j = lane; j < group_size; j += 32) {
        const Index k = group * group_size + j;
        const float value = decoded.value(k);
        #pragma unroll
        for (unsigned m = 0; m < Rows; ++m)
          if (row + m < rows) {
            float a;
            if constexpr (HasActivationValue<Loader>::value)
              a = loader.activation_value(activation, row + m, k);
            else
              a = borrowed::mlx_cuda_quant::load(activation,
                  loader.activation_offset(row + m) + k, activation_type);
            dot[m] = __fmaf_rn(a, value, dot[m]);
          }
      }
      #pragma unroll
      for (unsigned m = 0; m < Rows; ++m) {
        dot[m] = borrowed::mlx_cuda_quant::warp_sum(dot[m]);
        result[m] = __fadd_rn(result[m], decoded.finish(dot[m]));
      }
    }
    if (lane == 0) {
      #pragma unroll
      for (unsigned m = 0; m < Rows; ++m)
        if (row + m < rows)
          borrowed::mlx_cuda_quant::store<Output>(output, (row + m) * columns + column, result[m]);
    }
  }
}

} // namespace hyle::compute::cuda_native::packed
