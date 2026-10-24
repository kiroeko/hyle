#include "tensor_kernel.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <type_traits>

#include "activation.cuh"

namespace hyle::compute::cuda_native::native {
namespace {
constexpr unsigned kThreads = 256;

template <class T> __device__ float decode(T value) { return value; }
template <> __device__ float decode(__half value) {
  return __half2float(value);
}
template <> __device__ float decode(__nv_bfloat16 value) {
  return __bfloat162float(value);
}
template <class T> __device__ T encode(float value) { return value; }
template <> __device__ __half encode(float value) {
  return __float2half_rn(value);
}
template <> __device__ __nv_bfloat16 encode(float value) {
  return __float2bfloat16_rn(value);
}

template <class T, bool Dense, class Index>
__device__ T read(const void *source, Index index, std::size_t offset) {
  if constexpr (Dense) {
    return static_cast<const T *>(source)[index];
  } else {
    T value;
    auto *bytes = reinterpret_cast<unsigned char *>(&value);
    const auto *input = static_cast<const unsigned char *>(source) + offset;
#pragma unroll
    for (unsigned byte = 0; byte < sizeof(T); ++byte)
      bytes[byte] = input[byte];
    return value;
  }
}

template<class T> constexpr bool fp8_storage =
    std::is_same_v<T, __nv_fp8_e4m3> || std::is_same_v<T, __nv_fp8_e5m2>;

// Storage widening preserves NaN sign/payload, unlike an arithmetic conversion.
template<class T> __device__ std::uint32_t cast_bits(T value) {
  if constexpr (std::is_same_v<T, float>) {
    return __float_as_uint(value);
  } else if constexpr (std::is_same_v<T, __half>) {
    const unsigned bits = __half_as_ushort(value);
    if ((bits & 0x7fffU) > 0x7c00U)
      return ((bits & 0x8000U) << 16) | 0x7f800000U | ((bits & 0x3ffU) << 13);
    return __float_as_uint(__half2float(value));
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return std::uint32_t(__bfloat16_as_ushort(value)) << 16;
  } else {
    static_assert(fp8_storage<T>);
    constexpr bool e4 = std::is_same_v<T, __nv_fp8_e4m3>;
    const unsigned bits = value.__x, magnitude = bits & 0x7fU;
    if (e4 ? magnitude == 0x7fU : magnitude > 0x7cU)
      return ((bits & 0x80U) << 24) | 0x7f800000U |
          (e4 ? 0x00400000U : ((bits & 3U) << 21));
    return __float_as_uint(__half2float(__half(__nv_cvt_fp8_to_halfraw(
        value.__x, e4 ? __NV_E4M3 : __NV_E5M2))));
  }
}

template<class Out, class In> __device__ Out storage_cast(In value) {
  if constexpr (std::is_same_v<In, std::int32_t>) {
    static_assert(std::is_same_v<Out, float>);
    return __int2float_rn(value);
  } else if constexpr (std::is_same_v<In, std::uint8_t>) {
    return encode<Out>(value != 0 ? 1.0F : 0.0F);
  } else {
    const auto bits = cast_bits(value), magnitude = bits & 0x7fffffffU;
    if constexpr (std::is_same_v<Out, bool>) {
      return magnitude != 0; // Includes subnormals even under FTZ.
    } else if constexpr (std::is_same_v<Out, float>) {
      return __uint_as_float(bits);
    } else if constexpr (fp8_storage<Out>) {
      constexpr bool e4 = std::is_same_v<Out, __nv_fp8_e4m3>;
      const auto sign = static_cast<std::uint8_t>(bits >> 24) & 0x80U;
      Out result;
      if (magnitude > 0x7f800000U)
        result.__x = static_cast<std::uint8_t>(sign | (e4 ? 0x7fU : 0x7eU));
      else if (!e4 && magnitude == 0x7f800000U)
        result.__x = static_cast<std::uint8_t>(sign | 0x7cU); // True E5M2 infinities survive.
      else {
        result.__x = __nv_cvt_float_to_fp8(__uint_as_float(bits), __NV_SATFINITE,
                                          e4 ? __NV_E4M3 : __NV_E5M2);
        if ((result.__x & 0x7fU) == 0)
          result.__x = static_cast<std::uint8_t>(sign);
      }
      return result;
    } else if constexpr (std::is_same_v<Out, __half>) {
      if (magnitude > 0x7f800000U)
        return __ushort_as_half(static_cast<unsigned short>(((bits >> 16) & 0x8000U) | 0x7fffU));
      return __float2half_rn(__uint_as_float(bits));
    } else {
      static_assert(std::is_same_v<Out, __nv_bfloat16>);
      if (magnitude > 0x7f800000U)
        return __ushort_as_bfloat16(static_cast<unsigned short>((bits >> 16) | 0x40U));
      return __float2bfloat16_rn(__uint_as_float(bits));
    }
  }
}

__device__ float sigmoid(float x) {
  const float e = expf(-fabsf(x));
  return x < 0 ? e / (1 + e) : 1 / (1 + e);
}

template <TensorKind Kind> __device__ float unary(float x) {
  if constexpr (Kind == TensorKind::kAbs)
    return fabsf(x);
  else if constexpr (Kind == TensorKind::kNeg)
    return -x;
  else if constexpr (Kind == TensorKind::kExp)
    return expf(x);
  else if constexpr (Kind == TensorKind::kLog)
    return logf(x);
  else if constexpr (Kind == TensorKind::kSqrt)
    return sqrtf(x);
  else if constexpr (Kind == TensorKind::kRsqrt)
    return 1 / sqrtf(x);
  else if constexpr (Kind == TensorKind::kTanh)
    return tanhf(x);
  else if constexpr (Kind == TensorKind::kSigmoid)
    return sigmoid(x);
  else if constexpr (Kind == TensorKind::kRelu)
    return isnan(x) ? x : (x > 0 ? x : 0.0F);
  else if constexpr (Kind == TensorKind::kSoftplus)
    return fmaxf(x, 0.0F) + log1pf(expf(-fabsf(x)));
  else {
    static_assert(Kind == TensorKind::kSilu || Kind == TensorKind::kGelu ||
                  Kind == TensorKind::kGeluTanh);
    constexpr auto borrowed = Kind == TensorKind::kSilu
        ? sglang_vendor::ActivationKind::kSiLU
        : Kind == TensorKind::kGelu ? sglang_vendor::ActivationKind::kGELU
                                   : sglang_vendor::ActivationKind::kGELUTanh;
    // Borrow the same stable-tail/-Inf math as the mature vector program;
    // this kernel only supplies the otherwise unsupported storage indexing.
    return sglang_vendor::apply_unary_activation_f32<borrowed>(x);
  }
}

template <TensorKind Kind> __device__ float binary(float x, float y) {
  if constexpr (Kind == TensorKind::kAdd)
    return x + y;
  else if constexpr (Kind == TensorKind::kSubtract)
    return x - y;
  else if constexpr (Kind == TensorKind::kMultiply)
    return x * y;
  else if constexpr (Kind == TensorKind::kDivide)
    return x / y;
  else if constexpr (Kind == TensorKind::kMinimum)
    return fminf(x, y);
  else {
    static_assert(Kind == TensorKind::kMaximum);
    return fmaxf(x, y);
  }
}

template <TensorKind Kind, class Out, class In, bool Dense, class Index>
__device__ void tensor_element(const void *first, const void *second,
                               const void *third, Out *output,
                               const TensorLayout &layout, Index index) {
  constexpr bool is_binary = Kind >= TensorKind::kAdd &&
                             Kind <= TensorKind::kMaximum;
  constexpr unsigned arity = Kind == TensorKind::kFill     ? 0
                             : Kind == TensorKind::kSelect ? 3
                             : is_binary                   ? 2
                                                           : 1;
  std::size_t offsets[3]{};
  if constexpr (!Dense && arity != 0) {
    Index remaining = index;
    for (unsigned axis = layout.rank; axis-- > 0;) {
      const auto dimension = static_cast<Index>(layout.dimensions[axis]);
      const Index coordinate = remaining % dimension;
      remaining /= dimension;
#pragma unroll
      for (unsigned input = 0; input < arity; ++input)
        offsets[input] += coordinate * layout.strides[input][axis];
    }
  }
  if constexpr (Kind == TensorKind::kFill) {
    output[index] = static_cast<Out>(layout.scalar_bits);
  } else if constexpr (Kind == TensorKind::kCopy) {
    output[index] = read<Out, Dense>(first, index, offsets[0]);
  } else if constexpr (Kind == TensorKind::kSelect) {
    const bool mask = read<unsigned char, Dense>(first, index, offsets[0]) != 0;
    output[index] = mask ? read<Out, Dense>(second, index, offsets[1])
                         : read<Out, Dense>(third, index, offsets[2]);
  } else if constexpr (Kind == TensorKind::kCast) {
    output[index] = storage_cast<Out>(read<In, Dense>(first, index, offsets[0]));
  } else {
    const float x = decode(read<In, Dense>(first, index, offsets[0]));
    if constexpr (is_binary)
      output[index] = encode<Out>(
          binary<Kind>(x, decode(read<In, Dense>(second, index, offsets[1]))));
    else
      output[index] = encode<Out>(unary<Kind>(x));
  }
}

template <TensorKind Kind, class Out, class In, bool Dense>
__global__ void tensor(const void *first, const void *second, const void *third,
                       Out *output, TensorLayout layout) {
  if constexpr (Kind == TensorKind::kCast || Kind == TensorKind::kCopy) {
    // Cast keeps the former managed route's wide descriptor domain. Bounded
    // grid-stride execution avoids a giant launch and integer wrap at the tail.
    const auto stride = std::uint64_t(gridDim.x) * blockDim.x;
    auto index = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    while (index < layout.elements) {
      tensor_element<Kind, Out, In, Dense>(first, second, third, output, layout, index);
      if (layout.elements - index <= stride)
        break;
      index += stride;
    }
  } else {
    // Preserve the original 32-bit arithmetic/indexing domain of math kernels.
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < static_cast<unsigned>(layout.elements))
      tensor_element<Kind, Out, In, Dense>(first, second, third, output, layout, index);
  }
}

template <TensorKind Kind, class Out, class In = Out>
const void *bind(bool dense) noexcept {
  // Dense storage passes omit coordinate division and byte-wise loads.
  // Arithmetic retains its existing specialized domains below.
  constexpr bool unique = Kind == TensorKind::kCast || Kind == TensorKind::kCopy ||
      Kind == TensorKind::kFill || Kind == TensorKind::kSelect ||
      Kind == TensorKind::kRelu || Kind == TensorKind::kMinimum || Kind == TensorKind::kMaximum;
  constexpr bool f16 = std::is_same_v<Out, __half> &&
      (Kind == TensorKind::kAbs || Kind == TensorKind::kNeg || Kind == TensorKind::kLog ||
       Kind == TensorKind::kSoftplus || Kind == TensorKind::kAdd ||
       Kind == TensorKind::kSubtract || Kind == TensorKind::kMultiply);
  constexpr bool bf16 = std::is_same_v<Out, __nv_bfloat16> &&
      (Kind == TensorKind::kNeg || Kind == TensorKind::kExp || Kind == TensorKind::kLog ||
       Kind == TensorKind::kSqrt || Kind == TensorKind::kRsqrt || Kind == TensorKind::kSoftplus ||
       Kind == TensorKind::kSubtract || Kind == TensorKind::kMultiply || Kind == TensorKind::kDivide);
  if constexpr (unique || f16 || bf16)
    return dense ? reinterpret_cast<const void *>(tensor<Kind, Out, In, true>)
                 : reinterpret_cast<const void *>(tensor<Kind, Out, In, false>);
  else
    return reinterpret_cast<const void *>(tensor<Kind, Out, In, false>);
}

template <class T>
const void *bind_cast(TensorElementType input, bool dense) noexcept {
  switch (input) {
  case TensorElementType::kFloat32:
    return bind<TensorKind::kCast, T, float>(dense);
  case TensorElementType::kFloat16:
    return bind<TensorKind::kCast, T, __half>(dense);
  case TensorElementType::kBFloat16:
    return bind<TensorKind::kCast, T, __nv_bfloat16>(dense);
  case TensorElementType::kFloat8E4M3FN:
    if constexpr (!std::is_same_v<T, bool>)
      return bind<TensorKind::kCast, T, __nv_fp8_e4m3>(dense);
    return nullptr;
  case TensorElementType::kFloat8E5M2:
    if constexpr (!std::is_same_v<T, bool>)
      return bind<TensorKind::kCast, T, __nv_fp8_e5m2>(dense);
    return nullptr;
  case TensorElementType::kBool:
    if constexpr (!fp8_storage<T> && !std::is_same_v<T, bool>)
      return bind<TensorKind::kCast, T, std::uint8_t>(dense);
    return nullptr;
  case TensorElementType::kInt32:
    if constexpr (std::is_same_v<T, float>)
      return bind<TensorKind::kCast, T, std::int32_t>(dense);
    return nullptr;
  }
  return nullptr;
}

template <class T>
const void *bind_numeric(TensorKind kind, bool dense) noexcept {
  switch (kind) {
#define HYLE_CUDA_TENSOR_BIND(name)                                            \
  case TensorKind::name:                                                       \
    return bind<TensorKind::name, T>(dense)
    HYLE_CUDA_TENSOR_BIND(kAbs);
    HYLE_CUDA_TENSOR_BIND(kNeg);
    HYLE_CUDA_TENSOR_BIND(kExp);
    HYLE_CUDA_TENSOR_BIND(kLog);
    HYLE_CUDA_TENSOR_BIND(kSqrt);
    HYLE_CUDA_TENSOR_BIND(kRsqrt);
    HYLE_CUDA_TENSOR_BIND(kTanh);
    HYLE_CUDA_TENSOR_BIND(kSigmoid);
    HYLE_CUDA_TENSOR_BIND(kRelu);
    HYLE_CUDA_TENSOR_BIND(kGelu);
    HYLE_CUDA_TENSOR_BIND(kGeluTanh);
    HYLE_CUDA_TENSOR_BIND(kSilu);
    HYLE_CUDA_TENSOR_BIND(kSoftplus);
    HYLE_CUDA_TENSOR_BIND(kAdd);
    HYLE_CUDA_TENSOR_BIND(kSubtract);
    HYLE_CUDA_TENSOR_BIND(kMultiply);
    HYLE_CUDA_TENSOR_BIND(kDivide);
    HYLE_CUDA_TENSOR_BIND(kMinimum);
    HYLE_CUDA_TENSOR_BIND(kMaximum);
#undef HYLE_CUDA_TENSOR_BIND
  default:
    return nullptr;
  }
}

template <class T>
const void *bind_storage(TensorKind kind, bool dense) noexcept {
  switch (kind) {
  case TensorKind::kCopy:
    return bind<TensorKind::kCopy, T>(dense);
  case TensorKind::kFill:
    return bind<TensorKind::kFill, T>(true);
  case TensorKind::kSelect:
    return bind<TensorKind::kSelect, T>(dense);
  default:
    return nullptr;
  }
}
} // namespace

cudaError_t tensor_prepare(TensorKind kind, TensorElementType output_type,
                           TensorElementType input_type, unsigned storage_width,
                           bool dense, const TensorLayout &layout,
                           TensorProgram *output) noexcept {
  if (output == nullptr || layout.rank > 8)
    return cudaErrorInvalidValue;
  const void *kernel = nullptr;
  if (kind == TensorKind::kCopy || kind == TensorKind::kFill ||
      kind == TensorKind::kSelect) {
    switch (storage_width) {
    case 1:
      kernel = bind_storage<std::uint8_t>(kind, dense);
      break;
    case 2:
      kernel = bind_storage<std::uint16_t>(kind, dense);
      break;
    case 4:
      kernel = bind_storage<std::uint32_t>(kind, dense);
      break;
    case 8:
      kernel = bind_storage<std::uint64_t>(kind, dense);
      break;
    default:
      return cudaErrorInvalidValue;
    }
  } else if (kind == TensorKind::kCast) {
    switch (output_type) {
    case TensorElementType::kFloat32: kernel = bind_cast<float>(input_type, dense); break;
    case TensorElementType::kFloat16: kernel = bind_cast<__half>(input_type, dense); break;
    case TensorElementType::kBFloat16: kernel = bind_cast<__nv_bfloat16>(input_type, dense); break;
    case TensorElementType::kFloat8E4M3FN: kernel = bind_cast<__nv_fp8_e4m3>(input_type, dense); break;
    case TensorElementType::kFloat8E5M2: kernel = bind_cast<__nv_fp8_e5m2>(input_type, dense); break;
    case TensorElementType::kBool: kernel = bind_cast<bool>(input_type, dense); break;
    case TensorElementType::kInt32: return cudaErrorInvalidValue;
    }
  } else {
    switch (output_type) {
    case TensorElementType::kFloat32:
      kernel = bind_numeric<float>(kind, dense);
      break;
    case TensorElementType::kFloat16:
      kernel = bind_numeric<__half>(kind, dense);
      break;
    case TensorElementType::kBFloat16:
      kernel = bind_numeric<__nv_bfloat16>(kind, dense);
      break;
    default: return cudaErrorInvalidValue;
    }
  }
  if (kernel == nullptr)
    return cudaErrorInvalidValue;
  if (layout.elements != 0) {
    cudaFuncAttributes attributes{};
    const auto result = cudaFuncGetAttributes(&attributes, kernel);
    if (result != cudaSuccess)
      return result;
    if (attributes.maxThreadsPerBlock < static_cast<int>(kThreads))
      return cudaErrorNotSupported;
  }
  const auto blocks = layout.elements / kThreads + (layout.elements % kThreads != 0);
  const auto launch_blocks = (kind == TensorKind::kCast || kind == TensorKind::kCopy) && blocks > 65535
                                ? 65535U : static_cast<unsigned>(blocks);
  *output = {kernel, launch_blocks, layout};
  return cudaSuccess;
}

cudaError_t tensor_launch(const TensorProgram &program, const void *first,
                          const void *second, const void *third, void *output,
                          cudaStream_t stream) noexcept {
  if (program.layout.elements == 0)
    return cudaSuccess;
  void *arguments[]{&first, &second, &third, &output,
                    const_cast<TensorLayout *>(&program.layout)};
  return cudaLaunchKernel(program.kernel, dim3(program.blocks), dim3(kThreads),
                          arguments, 0, stream);
}
} // namespace hyle::compute::cuda_native::native
