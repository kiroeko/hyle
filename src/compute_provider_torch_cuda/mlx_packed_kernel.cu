#include "mlx_packed_kernel.h"

#include "device_status_kernel.h"
#include "mlx_cuda_quant.h"

#include <cmath>

namespace hyle::compute::torch_cuda::native {
namespace {
namespace q = borrowed::mlx_cuda_quant;
using Format = MlxFormat;
using Kind = MlxKind;
using Index = unsigned long long;

__global__ void check_ids(const std::int32_t *ids, Index rows, Index limit,
                          unsigned *status) {
  for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < rows;
       i += Index(gridDim.x) * blockDim.x) {
    const auto id = ids[i];
    if (id < 0 || Index(id) >= limit)
      atomicCAS(status, 0U,
          unsigned(cuda::detail::DeviceStatusCode::kIndexOutOfRange));
  }
}

// Original MLX qmv warp-to-row mapping with the widened CPU equations:
// no global unpacked bank, narrowed affine weights, or MX scale before dot.
template<class Output, bool Embedding>
__global__ void packed(MlxPacked p, const void *activation,
    const std::uint32_t *weights, const void *scales, const void *biases,
    const std::int32_t *ids, void *output, const unsigned *status) {
  if (status && *status != 0) return;
  const unsigned bits = p.format == Format::kMxfp8 ? 8 : 4;
  const unsigned pack = 32 / bits;
  const Index groups = p.depth / p.group;
  if constexpr (Embedding) {
    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x;
         i < p.rows * p.depth; i += Index(gridDim.x) * blockDim.x) {
      const Index row = i / p.depth, k = i % p.depth;
      const Index source = Index(ids[row]);
      const auto word = weights[source * (p.depth / pack) + k / pack];
      const float code = float((word >> ((k % pack) * bits)) & 15U);
      const Index group = source * groups + k / p.group;
      // affine_dequantize rounds its pointwise multiply and add separately.
      const float value = q::rounded(__fmul_rn(code,
          q::load(scales, group, p.scale_type)), p.output_type);
      q::store<Output>(output, i, __fadd_rn(value,
          q::load(biases, group, p.scale_type)));
    }
  } else {
    const Index first = (Index(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const Index stride = Index(gridDim.x) * blockDim.x / 32;
    const unsigned lane = threadIdx.x % 32;
    for (Index i = first; i < p.rows * p.columns; i += stride) {
      const Index row = i / p.columns, column = i % p.columns;
      const Index expert = p.kind == Kind::kGrouped ? Index(ids[row]) : 0;
      const Index weight_row = expert * p.columns + column;
      float result = 0;
      for (Index group = 0; group < groups; ++group) {
        const Index scale_index = weight_row * groups + group;
        const bool affine = p.format == Format::kAffine4;
        const float scale = affine ? q::load(scales, scale_index, p.scale_type)
            : q::rounded(ldexpf(1.F, int(static_cast<const std::uint8_t *>(scales)[scale_index]) - 127), p.output_type);
        const float bias = affine ? q::load(biases, scale_index, p.scale_type) : 0.F;
        float dot = 0;
        for (unsigned j = lane; j < p.group; j += 32) {
          const Index k = group * p.group + j;
          const auto word = weights[weight_row * (p.depth / pack) + k / pack];
          const auto code = (word >> ((k % pack) * bits)) & ((1U << bits) - 1);
          const float value = affine ? __fadd_rn(__fmul_rn(float(code), scale), bias)
              : p.format == Format::kMxfp4 ? q::fp4(code) : q::fp8(code);
          dot = __fmaf_rn(q::load(activation, row * p.depth + k, p.activation_type), value, dot);
        }
        dot = q::warp_sum(dot);
        result = __fadd_rn(result, affine ? dot : __fmul_rn(dot, scale));
      }
      if (lane == 0) q::store<Output>(output, i, result);
    }
  }
}

unsigned blocks(Index items, unsigned per_block) {
  const auto count = (items + per_block - 1) / per_block;
  return unsigned(count < 65535 ? count : 65535);
}

template<class Output>
cudaError_t launch(const MlxPacked &p, const void *const *inputs, void *output,
                   unsigned *status, cudaStream_t stream) {
  const bool embedding = p.kind == Kind::kEmbedding;
  const bool checked = p.kind != Kind::kLinear;
  const auto w = embedding ? 0 : 1;
  const void *a = embedding ? nullptr : inputs[0];
  const auto *weights = static_cast<const std::uint32_t *>(inputs[w]);
  const void *scales = inputs[w + 1];
  const void *biases = p.format == Format::kAffine4 ? inputs[w + 2] : nullptr;
  const auto *ids = checked ? static_cast<const std::int32_t *>(inputs[embedding ? 3 : p.format == Format::kAffine4 ? 4 : 3]) : nullptr;
  if (checked) {
    if (!status) return cudaErrorInvalidValue;
    Index limit = embedding ? p.columns : p.experts;
    Index rows = p.rows;
    void *args[]{&ids, &rows, &limit, &status};
    const auto result = cudaLaunchKernel(reinterpret_cast<const void *>(check_ids),
        dim3(blocks(rows, 256)), dim3(256), args, 0, stream);
    if (result != cudaSuccess) return result;
  }
  MlxPacked spec = p;
  void *args[]{&spec, &a, &weights, &scales, &biases, &ids, &output, &status};
  const void *kernel = embedding ? reinterpret_cast<const void *>(packed<Output, true>)
                                 : reinterpret_cast<const void *>(packed<Output, false>);
  return cudaLaunchKernel(kernel,
      dim3(blocks(p.rows * (embedding ? p.depth : p.columns), embedding ? 256 : 8)),
      dim3(256), args, 0, stream);
}
} // namespace

cudaError_t mlx_packed_launch(const MlxPacked &p, const void *const *inputs,
    void *output, unsigned *status, cudaStream_t stream) noexcept {
  if (p.rows == 0) return cudaSuccess;
  switch (p.output_type) {
  case Scalar::kFloat16: return launch<__half>(p, inputs, output, status, stream);
  case Scalar::kBFloat16: return launch<__nv_bfloat16>(p, inputs, output, status, stream);
  case Scalar::kFloat32: return launch<float>(p, inputs, output, status, stream);
  default: return cudaErrorInvalidValue;
  }
}
} // namespace hyle::compute::torch_cuda::native
