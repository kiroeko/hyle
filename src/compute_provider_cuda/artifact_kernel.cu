#include "artifact_kernel.h"

#include "device_status_kernel.h"
#include "mlx_cuda_quant.h"
#include "packed_rows.cuh"

#include <cmath>
#include <type_traits>

namespace hyle::compute::cuda_native::artifact::native {
namespace {
namespace q = borrowed::mlx_cuda_quant;
using Format = MlxFormat;
using Kind = MlxKind;
using Index = unsigned long long;

// MLX affine codes form one continuous little-endian bitstream per K row.
// In particular five/six-bit codes may straddle adjacent U32 words.
__device__ __forceinline__ unsigned unpack(const std::uint32_t *weights,
    Index row, Index depth, Index k, unsigned bits) {
  const Index row_words = (depth / 32) * bits;
  const Index bit = k * bits;
  const Index word = row * row_words + bit / 32;
  const unsigned shift = unsigned(bit % 32);
  unsigned code = weights[word] >> shift;
  if (shift + bits > 32) code |= weights[word + 1] << (32 - shift);
  return code & ((1U << bits) - 1U);
}

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

// The row carrier below is shared, while each codec retains its own load and
// rounding contract. MLX group-scale-after-dot is deliberately not CT math.
struct MlxGroup final {
  MlxPacked p;
  const std::uint32_t *weights;
  Index weight_row;
  float scale, bias;
  bool affine;
  __device__ __forceinline__ float value(Index k) const {
    const auto code = unpack(weights, weight_row, p.depth, k, p.bits);
    return affine ? __fadd_rn(__fmul_rn(float(code), scale), bias)
        : p.format == Format::kMxfp4 ? q::fp4(code) : q::fp8(code);
  }
  __device__ __forceinline__ float finish(float dot) const {
    return affine ? dot : __fmul_rn(dot, scale);
  }
};
struct MlxRows final {
  using GroupIndex = unsigned;
  MlxPacked p;
  const std::uint32_t *weights;
  const void *scales, *biases;
  const std::int32_t *ids;
  __device__ __forceinline__ Index activation_offset(Index row) const {
    return p.kind == Kind::kGrouped
        ? (row / p.slots) * p.activation_token_stride +
          (row % p.slots) * p.activation_slot_stride
        : row * p.depth;
  }
  __device__ __forceinline__ Index weight_row(Index row, Index column) const {
    const Index expert = p.kind == Kind::kGrouped ? Index(ids[row]) : 0;
    return expert * p.columns + column;
  }
  __device__ __forceinline__ MlxGroup group(Index weight_row, Index group) const {
    const Index index = weight_row * (p.depth / p.group) + group;
    const bool affine = p.format == Format::kAffine;
    const float scale = affine ? q::load(scales, index, p.scale_type)
        : q::rounded(ldexpf(1.F, int(static_cast<const std::uint8_t *>(scales)[index]) - 127), p.output_type);
    const float bias = affine ? q::load(biases, index, p.scale_type) : 0.F;
    return {p, weights, weight_row, scale, bias, affine};
  }
};
template<bool Gptq>
struct ArtifactGroup final {
  ArtifactPacked p;
  const std::uint32_t *codes;
  const std::uint32_t *zeros;
  const void *scales;
  const std::int32_t *indices;
  Index weight_row;
  float scale;
  unsigned zero;
  __device__ __forceinline__ float value(Index k) const {
    const auto word = codes[Gptq ? (k / 8) * p.columns + weight_row
                                 : weight_row * (p.depth / 8) + k / 8];
    const auto code = (word >> (4 * unsigned(k % 8))) & 15U;
    unsigned point_zero = zero;
    float point_scale = scale;
    if constexpr (Gptq) {
      // The preceding ID kernel has checked the entire mutable K-vector.
      // g_idx selects an SF/Z group, not an activation/code permutation.
      const Index group = Index(indices[k]);
      point_zero = ((zeros[group * (p.columns / 8) + weight_row / 8] >>
          (4 * unsigned(weight_row % 8))) & 15U) + 1U;
      point_scale = q::load(scales, group * p.columns + weight_row, p.scale_type);
    }
    // Centered integers, including GPTQ's -16, are exactly representable.
    // Preserve its pointwise product RNE, then the actual Decode and optional
    // MatMul operand casts. None of these boundaries may move after the dot.
    const float product = q::rounded(__fmul_rn(float(int(code) - int(point_zero)),
                                              point_scale), p.scale_type);
    const float decoded = q::rounded(product, p.decode_type);
    return q::rounded(decoded, p.operand_type);
  }
  __device__ __forceinline__ float finish(float dot) const { return dot; }
};
template<bool Gptq, bool Grouped = false>
struct ArtifactRows final {
  using GroupIndex = Index;
  ArtifactPacked p;
  const std::uint32_t *codes, *zeros;
  const void *scales;
  const std::int32_t *indices;
  __device__ __forceinline__ Index activation_offset(Index row) const {
    return row * p.depth;
  }
  template<bool Enabled = Grouped, std::enable_if_t<Enabled, int> = 0>
  __device__ __forceinline__ float activation_value(const void *activation,
      Index row, Index k) const {
    const auto *source = static_cast<const std::uint8_t *>(activation) +
        (row / p.slots) * p.activation_token_stride +
        (row % p.slots) * p.activation_slot_stride + k * p.activation_depth_stride;
    // Public plain A has byte strides. Four byte loads also support layouts
    // whose element origins are not naturally aligned, without an A copy.
    return __uint_as_float(unsigned(source[0]) | (unsigned(source[1]) << 8) |
        (unsigned(source[2]) << 16) | (unsigned(source[3]) << 24));
  }
  __device__ __forceinline__ Index weight_row(Index row, Index column) const {
    return Grouped ? Index(indices[row]) * p.columns + column : column;
  }
  __device__ __forceinline__ ArtifactGroup<Gptq> group(Index row, Index group) const {
    if constexpr (Gptq)
      return {p, codes, zeros, scales, indices, row, 0.F, 0U};
    const Index groups = p.depth / p.group;
    const auto zero = p.format == ArtifactFormat::kCtAsymmetric
        ? (zeros[(row / 8) * groups + group] >> (4 * unsigned(row % 8))) & 15U
        : 8U;
    return {p, codes, zeros, scales, nullptr, row,
        q::load(scales, row * groups + group, p.scale_type), zero};
  }
};

// NF4 double-quant uses the same original packed row reduction. Only its
// point reader differs: U8 high-first codes and the six public Decode inputs.
struct Nf4Group final {
  ArtifactPacked p;
  const std::uint8_t *codes;
  const float *codebook;
  Index row;
  float scale;
  __device__ __forceinline__ float value(Index k) const {
    const Index original = row * p.depth + k;
    const auto code = (codes[original / 2] >> (original % 2 ? 0 : 4)) & 15U;
    const float decoded = q::rounded(__fmul_rn(codebook[code], scale), p.decode_type);
    return q::rounded(decoded, p.operand_type);
  }
  __device__ __forceinline__ float finish(float dot) const { return dot; }
};
struct Nf4Rows final {
  using GroupIndex = Index;
  ArtifactPacked p;
  const std::uint8_t *codes, *absmax;
  const float *nested_scale, *nested_map, *offset, *codebook;
  __device__ __forceinline__ Index activation_offset(Index row) const {
    return row * p.depth;
  }
  __device__ __forceinline__ Index weight_row(Index, Index column) const { return column; }
  __device__ __forceinline__ Nf4Group group(Index row, Index group) const {
    const Index original_group = row * (p.depth / p.group) + group;
    const float product = __fmul_rn(nested_map[absmax[original_group]],
                                    nested_scale[original_group / 256]);
    // Original BNB rounds nested multiplication, offset addition, and NF4
    // multiplication separately in F32; never contract these into an FMA.
    const float scale = __fadd_rn(product, *offset);
    return {p, codes, codebook, row, scale};
  }
};

using cuda_native::packed::packed_rows;

// Original MLX qmv warp-to-row mapping with the widened CPU equations:
// no global unpacked bank, narrowed affine weights, or MX scale before dot.
template<class Output, bool Embedding, unsigned Rows = 1, bool AsStored = false>
__global__ void packed(MlxPacked p, const void *activation,
    const std::uint32_t *weights, const void *scales, const void *biases,
    const std::int32_t *ids, void *output, const unsigned *status) {
  if (status && *status != 0) return;
  if constexpr (Embedding) {
    const unsigned bits = p.bits;
    const Index groups = p.depth / p.group;
    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x;
         i < p.rows * p.depth; i += Index(gridDim.x) * blockDim.x) {
      const Index row = i / p.depth, k = i % p.depth;
      const Index source = Index(ids[row]);
      const float code = float(unpack(weights, source, p.depth, k, bits));
      const Index group = source * groups + k / p.group;
      // affine_dequantize rounds its pointwise multiply and add separately.
      const float value = q::rounded(__fmul_rn(code,
          q::load(scales, group, p.scale_type)), p.output_type);
      q::store<Output>(output, i, __fadd_rn(value,
          q::load(biases, group, p.scale_type)));
    }
  } else if constexpr (AsStored) {
    const Index first = (Index(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const Index stride = Index(gridDim.x) * blockDim.x / 32;
    const unsigned lane = threadIdx.x % 32;
    const Index groups = p.depth / p.group;
    for (Index i = first; i < p.rows * p.depth; i += stride) {
      const Index row = i / p.depth, k = i % p.depth;
      const Index expert = p.kind == Kind::kGrouped ? Index(ids[row]) : 0;
      const Index a = p.kind == Kind::kGrouped
          ? (row / p.slots) * p.activation_token_stride +
            (row % p.slots) * p.activation_slot_stride
          : row * p.columns;
      float dot = 0.F;
      for (Index n = lane; n < p.columns; n += 32) {
        const Index weight_row = expert * p.columns + n;
        const Index group = weight_row * groups + k / p.group;
        const float code = float(unpack(weights, weight_row, p.depth, k, p.bits));
        const float value = __fadd_rn(__fmul_rn(code,
            q::load(scales, group, p.scale_type)), q::load(biases, group, p.scale_type));
        dot = __fmaf_rn(q::load(activation, a + n, p.activation_type), value, dot);
      }
      dot = q::warp_sum(dot);
      if (lane == 0) q::store<Output>(output, i, dot);
    }
  } else {
    packed_rows<Output, Rows>(p.rows, p.columns, p.depth, p.group,
        p.activation_type, activation, output,
        MlxRows{p, weights, scales, biases, ids});
  }
}

template<class Output, unsigned Rows, bool Gptq, bool Grouped = false>
__global__ void artifact_packed(ArtifactPacked p, const void *activation,
    const std::uint32_t *codes, const std::uint32_t *zeros,
    const void *scales, const std::int32_t *indices, void *output,
    const unsigned *status) {
  // Only check_ids writes status, and completes on this stream before us.
  if (status && *status != 0) return;
  packed_rows<Output, Rows>(p.rows, p.columns, p.depth, p.group,
      Grouped ? Scalar::kFloat32 : p.operand_type, activation, output,
      ArtifactRows<Gptq, Grouped>{p, codes, zeros, scales, indices});
}

template<class Output, unsigned Rows>
__global__ void nf4_packed(ArtifactPacked p, const void *activation,
    const std::uint8_t *codes, const std::uint8_t *absmax,
    const float *nested_scale, const float *nested_map, const float *offset,
    const float *codebook, void *output) {
  packed_rows<Output, Rows>(p.rows, p.columns, p.depth, p.group,
      p.operand_type, activation, output,
      Nf4Rows{p, codes, absmax, nested_scale, nested_map, offset, codebook});
}

template<class Output>
__global__ void gptq_decode(ArtifactPacked p, const std::uint32_t *codes,
    const std::uint32_t *zeros, const void *scales, const std::int32_t *indices,
    void *output, const unsigned *status) {
  if (*status != 0) return;
  for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x;
       i < p.columns * p.depth; i += Index(gridDim.x) * blockDim.x) {
    const ArtifactGroup<true> point{p, codes, zeros, scales, indices, i / p.depth, 0.F, 0U};
    q::store<Output>(output, i, point.value(i % p.depth));
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
  const void *biases = p.format == Format::kAffine ? inputs[w + 2] : nullptr;
  const auto *ids = checked ? static_cast<const std::int32_t *>(inputs[embedding ? 3 : p.format == Format::kAffine ? 4 : 3]) : nullptr;
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
  const bool as_stored = p.orientation == MlxOrientation::kWeightAsStored;
  const bool row_batch = !as_stored && p.kind == Kind::kLinear && p.rows > 1;
  const Index work_rows = row_batch ? 1 + (p.rows - 1) / 4 : p.rows;
  const void *kernel = embedding ? reinterpret_cast<const void *>(packed<Output, true>)
      : as_stored ? reinterpret_cast<const void *>(packed<Output, false, 1, true>)
      : row_batch ? reinterpret_cast<const void *>(packed<Output, false, 4>)
                  : reinterpret_cast<const void *>(packed<Output, false>);
  return cudaLaunchKernel(kernel,
      dim3(blocks(work_rows * (embedding || as_stored ? p.depth : p.columns), embedding ? 256 : 8)),
      dim3(256), args, 0, stream);
}
cudaError_t check_artifact_indices(const ArtifactPacked &p, const std::int32_t *indices,
    unsigned *status, cudaStream_t stream) {
  if (!status || !indices) return cudaErrorInvalidValue;
  Index count = p.experts ? p.rows : p.depth;
  Index limit = p.experts ? p.experts : p.depth / p.group;
  void *args[]{&indices, &count, &limit, &status};
  return cudaLaunchKernel(reinterpret_cast<const void *>(check_ids),
      dim3(blocks(count, 256)), dim3(256), args, 0, stream);
}

template<class Output, bool DecodeOnly = false>
cudaError_t launch_artifact(const ArtifactPacked &p, const void *const *inputs,
                      void *output, unsigned *status, cudaStream_t stream) {
  ArtifactPacked spec = p;
  constexpr auto first = DecodeOnly ? 0 : 1;
  if (p.format == ArtifactFormat::kNf4DoubleQuant) {
    if constexpr (DecodeOnly) return cudaErrorInvalidValue;
    const void *activation = inputs[0];
    const auto *codes = static_cast<const std::uint8_t *>(inputs[1]);
    const auto *absmax = static_cast<const std::uint8_t *>(inputs[2]);
    const auto *nested_scale = static_cast<const float *>(inputs[3]);
    const auto *nested_map = static_cast<const float *>(inputs[4]);
    const auto *offset = static_cast<const float *>(inputs[5]);
    const auto *codebook = static_cast<const float *>(inputs[6]);
    void *args[]{&spec, &activation, &codes, &absmax, &nested_scale,
                 &nested_map, &offset, &codebook, &output};
    const bool row_batch = p.rows > 1;
    const Index work_rows = row_batch ? 1 + (p.rows - 1) / 4 : p.rows;
    const auto *kernel = row_batch ? reinterpret_cast<const void *>(nf4_packed<Output, 4>)
                                   : reinterpret_cast<const void *>(nf4_packed<Output, 1>);
    return cudaLaunchKernel(kernel, dim3(blocks(work_rows * p.columns, 8)),
                            dim3(256), args, 0, stream);
  }
  const bool gptq = p.format == ArtifactFormat::kGptqV1Int4;
  const bool zeros_present = p.format != ArtifactFormat::kCtSymmetric;
  const auto *codes = static_cast<const std::uint32_t *>(inputs[first]);
  const auto *zeros = zeros_present ? static_cast<const std::uint32_t *>(inputs[first + 1]) : nullptr;
  const void *scales = inputs[first + (zeros_present ? 2 : 1)];
  const auto *indices = gptq || p.experts
      ? static_cast<const std::int32_t *>(inputs[first + (zeros_present ? 3 : 2)]) : nullptr;
  if (gptq || p.experts) {
    const auto result = check_artifact_indices(p, indices, status, stream);
    if (result != cudaSuccess) return result;
  }
  if constexpr (DecodeOnly) {
    void *args[]{&spec, &codes, &zeros, &scales, &indices, &output, &status};
    return cudaLaunchKernel(reinterpret_cast<const void *>(gptq_decode<Output>),
        dim3(blocks(p.columns * p.depth, 256)), dim3(256), args, 0, stream);
  }
  const void *activation = inputs[0];
  void *args[]{&spec, &activation, &codes, &zeros, &scales, &indices, &output, &status};
  const bool row_batch = p.experts == 0 && p.rows > 1;
  const Index work_rows = row_batch ? 1 + (p.rows - 1) / 4 : p.rows;
  const auto *kernel = p.experts ? reinterpret_cast<const void *>(artifact_packed<Output, 1, false, true>) : gptq
      ? (row_batch ? reinterpret_cast<const void *>(artifact_packed<Output, 4, true>)
                   : reinterpret_cast<const void *>(artifact_packed<Output, 1, true>))
      : (row_batch ? reinterpret_cast<const void *>(artifact_packed<Output, 4, false>)
                   : reinterpret_cast<const void *>(artifact_packed<Output, 1, false>));
  return cudaLaunchKernel(kernel,
      dim3(blocks(work_rows * p.columns, 8)), dim3(256), args, 0, stream);
}

template<class Output>
cudaError_t prepare_artifact(const ArtifactPacked &p, bool decode_only) {
  cudaFuncAttributes attributes{};
  if (p.format == ArtifactFormat::kNf4DoubleQuant) {
    if (decode_only || p.experts) return cudaErrorInvalidValue;
    const auto *kernel = p.rows > 1 ? reinterpret_cast<const void *>(nf4_packed<Output, 4>)
                                   : reinterpret_cast<const void *>(nf4_packed<Output, 1>);
    return cudaFuncGetAttributes(&attributes, kernel);
  }
  if (p.format == ArtifactFormat::kGptqV1Int4 || p.experts) {
    const auto result = cudaFuncGetAttributes(&attributes,
        reinterpret_cast<const void *>(check_ids));
    if (result != cudaSuccess) return result;
  }
  const bool gptq = p.format == ArtifactFormat::kGptqV1Int4;
  const auto *kernel = decode_only ? reinterpret_cast<const void *>(gptq_decode<Output>)
      : p.experts ? reinterpret_cast<const void *>(artifact_packed<Output, 1, false, true>)
      : gptq ? (p.rows > 1 ? reinterpret_cast<const void *>(artifact_packed<Output, 4, true>)
                          : reinterpret_cast<const void *>(artifact_packed<Output, 1, true>))
             : (p.rows > 1 ? reinterpret_cast<const void *>(artifact_packed<Output, 4, false>)
                          : reinterpret_cast<const void *>(artifact_packed<Output, 1, false>));
  return cudaFuncGetAttributes(&attributes, kernel);
}

__global__ void static_quantize(StaticFloat8 p, const void *input,
    const float *scale, std::uint8_t *encoded) {
  const float scalar_reciprocal = p.activation_scale_stride ? 0.F : __fdiv_rn(1.F, scale[0]);
  for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < p.rows * p.depth;
       i += Index(gridDim.x) * blockDim.x) {
    const float reciprocal = p.activation_scale_stride
        ? __fdiv_rn(1.F, scale[i / p.depth]) : scalar_reciprocal;
    const float value = __fmul_rn(q::load(input, i, p.activation_type), reciprocal);
    const auto bits = __float_as_uint(value), sign = (bits >> 24) & 0x80U;
    unsigned code = (bits & 0x7fffffffU) > 0x7f800000U ? sign | 0x7fU
        : __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
    if ((code & 0x7fU) == 0) code = sign;
    encoded[i] = static_cast<std::uint8_t>(code);
  }
}
template<class Output>
__global__ void static_code_dot(StaticFloat8 p, const std::uint8_t *a,
    const std::uint8_t *w, const float *sa, const float *sw, void *output) {
  const Index first = (Index(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  const Index stride = Index(gridDim.x) * blockDim.x / 32;
  const unsigned lane = threadIdx.x % 32;
  for (Index i = first; i < p.rows * p.columns; i += stride) {
    const Index row = i / p.columns, column = i % p.columns;
    float dot = 0.F;
    for (Index k = lane; k < p.depth; k += 32)
      dot = __fmaf_rn(q::fp8(a[row * p.depth + k]), q::fp8(w[column * p.depth + k]), dot);
    dot = q::warp_sum(dot);
    if (lane == 0) {
      const float activation_scale = sa[row * p.activation_scale_stride];
      const float weight_scale = sw[column * p.weight_scale_stride];
      const float first_scale = p.weight_scale_first ? weight_scale : activation_scale;
      const float second_scale = p.weight_scale_first ? activation_scale : weight_scale;
      q::store<Output>(output, i, __fmul_rn(__fmul_rn(dot, first_scale), second_scale));
    }
  }
}
template<class Output>
cudaError_t static_launch(const StaticFloat8 &p, const void *const *inputs,
    void *encoded, void *output, cudaStream_t stream) {
  auto spec = p;
  const void *a = inputs[0], *w = inputs[1], *sa = inputs[2], *sw = inputs[3];
  void *quant_args[]{&spec, &a, &sa, &encoded};
  auto result = cudaLaunchKernel(reinterpret_cast<const void *>(static_quantize),
      dim3(blocks(p.rows * p.depth, 256)), dim3(256), quant_args, 0, stream);
  if (result != cudaSuccess) return result;
  void *dot_args[]{&spec, &encoded, &w, &sa, &sw, &output};
  return cudaLaunchKernel(reinterpret_cast<const void *>(static_code_dot<Output>),
      dim3(blocks(p.rows * p.columns, 8)), dim3(256), dot_args, 0, stream);
}

} // namespace

cudaError_t static_fp8_prepare(const StaticFloat8 &p) noexcept {
  cudaFuncAttributes attributes{};
  auto result = cudaFuncGetAttributes(&attributes, reinterpret_cast<const void *>(static_quantize));
  if (result != cudaSuccess) return result;
  const auto *kernel = p.output_type == Scalar::kFloat16 ? reinterpret_cast<const void *>(static_code_dot<__half>)
      : p.output_type == Scalar::kBFloat16 ? reinterpret_cast<const void *>(static_code_dot<__nv_bfloat16>)
      : reinterpret_cast<const void *>(static_code_dot<float>);
  return cudaFuncGetAttributes(&attributes, kernel);
}
cudaError_t static_fp8_launch(const StaticFloat8 &p, const void *const *inputs,
    void *encoded, void *output, cudaStream_t stream) noexcept {
  if (p.output_type == Scalar::kFloat16) return static_launch<__half>(p, inputs, encoded, output, stream);
  if (p.output_type == Scalar::kBFloat16) return static_launch<__nv_bfloat16>(p, inputs, encoded, output, stream);
  return static_launch<float>(p, inputs, encoded, output, stream);
}

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
cudaError_t artifact_packed_linear_launch(const ArtifactPacked &p,
    const void *const *inputs, void *output, unsigned *status,
    cudaStream_t stream) noexcept {
  if (p.rows == 0) return cudaSuccess;
  if (p.experts) return launch_artifact<float>(p, inputs, output, status, stream);
  switch (p.operand_type) {
  case Scalar::kFloat16: return launch_artifact<__half>(p, inputs, output, status, stream);
  case Scalar::kBFloat16: return launch_artifact<__nv_bfloat16>(p, inputs, output, status, stream);
  case Scalar::kFloat32: return launch_artifact<float>(p, inputs, output, status, stream);
  default: return cudaErrorInvalidValue;
  }
}

cudaError_t gptq_decode_launch(const ArtifactPacked &p, const void *const *inputs,
    void *output, unsigned *status, cudaStream_t stream) noexcept {
  switch (p.decode_type) {
  case Scalar::kFloat16: return launch_artifact<__half, true>(p, inputs, output, status, stream);
  case Scalar::kBFloat16: return launch_artifact<__nv_bfloat16, true>(p, inputs, output, status, stream);
  case Scalar::kFloat32: return launch_artifact<float, true>(p, inputs, output, status, stream);
  default: return cudaErrorInvalidValue;
  }
}

cudaError_t artifact_packed_prepare(const ArtifactPacked &p, bool decode_only) noexcept {
  if (p.experts) return prepare_artifact<float>(p, false);
  const auto type = decode_only ? p.decode_type : p.operand_type;
  switch (type) {
  case Scalar::kFloat16: return prepare_artifact<__half>(p, decode_only);
  case Scalar::kBFloat16: return prepare_artifact<__nv_bfloat16>(p, decode_only);
  case Scalar::kFloat32: return prepare_artifact<float>(p, decode_only);
  default: return cudaErrorInvalidValue;
  }
}

} // namespace hyle::compute::cuda_native::artifact::native
