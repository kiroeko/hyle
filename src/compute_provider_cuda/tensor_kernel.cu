#include "tensor_kernel.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
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

template <class T, bool Dense>
__device__ T read(const void *source, unsigned index, std::size_t offset) {
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

template <TensorKind Kind, class Out, class In, bool Dense>
__global__ void tensor(const void *first, const void *second, const void *third,
                       Out *output, TensorLayout layout) {
  const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= layout.elements)
    return;
  constexpr unsigned arity = Kind == TensorKind::kFill     ? 0
                             : Kind == TensorKind::kSelect ? 3
                             : Kind >= TensorKind::kAdd    ? 2
                                                           : 1;
  std::size_t offsets[3]{};
  if constexpr (!Dense && arity != 0) {
    unsigned remaining = index;
    for (unsigned axis = layout.rank; axis-- > 0;) {
      const unsigned coordinate = remaining % layout.dimensions[axis];
      remaining /= layout.dimensions[axis];
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
  } else {
    const float x = decode(read<In, Dense>(first, index, offsets[0]));
    if constexpr (Kind == TensorKind::kCast)
      output[index] = encode<Out>(x);
    else if constexpr (Kind >= TensorKind::kAdd)
      output[index] = encode<Out>(
          binary<Kind>(x, decode(read<In, Dense>(second, index, offsets[1]))));
    else
      output[index] = encode<Out>(unary<Kind>(x));
  }
}

template <TensorKind Kind, class Out, class In = Out>
const void *bind(bool dense) noexcept {
  // Keep dedicated dense code only for a complete C2 winner or a genuinely
  // single-carrier semantic. Other ordinary domains use the mature default;
  // their unique stride/resource domains share the existing generic program.
  constexpr bool unique = Kind == TensorKind::kFill || Kind == TensorKind::kSelect ||
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
const void *bind_numeric(TensorKind kind, ElementType input,
                         bool dense) noexcept {
  switch (kind) {
  case TensorKind::kCast:
    switch (input) {
    case ElementType::kFloat32:
      return bind<TensorKind::kCast, T, float>(dense);
    case ElementType::kFloat16:
      return bind<TensorKind::kCast, T, __half>(dense);
    case ElementType::kBFloat16:
      return bind<TensorKind::kCast, T, __nv_bfloat16>(dense);
    }
    return nullptr;
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

cudaError_t tensor_prepare(TensorKind kind, ElementType output_type,
                           ElementType input_type, unsigned storage_width,
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
  } else {
    switch (output_type) {
    case ElementType::kFloat32:
      kernel = bind_numeric<float>(kind, input_type, dense);
      break;
    case ElementType::kFloat16:
      kernel = bind_numeric<__half>(kind, input_type, dense);
      break;
    case ElementType::kBFloat16:
      kernel = bind_numeric<__nv_bfloat16>(kind, input_type, dense);
      break;
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
  *output = {kernel, (layout.elements + kThreads - 1) / kThreads, layout};
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
