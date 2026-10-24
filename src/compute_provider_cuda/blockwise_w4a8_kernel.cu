#include "blockwise_w4a8_kernel.h"

#include <cuda_bf16.h>
#include <cuda/atomic>
#include "device_status_kernel.h"

namespace hyle::compute::cuda_native::blockwise_w4a8 {
namespace {
using Index=unsigned long long;
constexpr unsigned kThreads=128;
constexpr Index kGroup=512;
__device__ __forceinline__ bool failed(unsigned *status) {
  return ::cuda::atomic_ref<unsigned,::cuda::thread_scope_device>(*status)
      .load(::cuda::std::memory_order_relaxed)!=0;
}
__device__ __forceinline__ void invalid(unsigned *status) {
  atomicCAS(status,0U,unsigned(cuda::detail::DeviceStatusCode::kInvalidNumericInput));
}
__device__ __forceinline__ __nv_bfloat16 rounded(float value) {
  return __float2bfloat16_rn(value);
}
__global__ void check_ids(Program p,const int *ids,unsigned *status) {
  const Index count=p.tokens*p.slots;
  for (Index i=Index(blockIdx.x)*blockDim.x+threadIdx.x;i<count;
       i+=Index(gridDim.x)*blockDim.x) {
    const int id=ids[i];
    if (id<0 || Index(id)>=p.experts)
      atomicCAS(status,0U,unsigned(cuda::detail::DeviceStatusCode::kIndexOutOfRange));
  }
}
template<bool Grouped>
__global__ void quantize(Program p,const __nv_bfloat16 *input,signed char *codes,
                         __nv_bfloat16 *scales,unsigned *status) {
  if (failed(status)) return;
  const Index groups=p.depth/kGroup,count=p.quant_rows*groups;
  for (Index i=Index(blockIdx.x)*blockDim.x+threadIdx.x;i<count;
       i+=Index(gridDim.x)*blockDim.x) {
    const Index row=i/groups,group=i%groups;
    Index source=row*p.depth;
    if constexpr (Grouped) {
      const Index token=p.shared?row:row/p.slots;
      const Index slot=p.shared?0:row%p.slots;
      source=token*p.activation_token_stride+slot*p.activation_slot_stride;
    }
    source+=group*kGroup;
    // Positive finite BF16 bit patterns sort exactly like their magnitudes.
    // Two bounded passes avoid a per-thread512-element float local array.
    unsigned maximum=0;
    bool valid=true;
    for (unsigned lane=0;lane<kGroup;++lane) {
      const unsigned magnitude=__bfloat16_as_ushort(input[source+lane])&0x7fffU;
      valid &= magnitude<0x7f80U;
      maximum=maximum>magnitude?maximum:magnitude;
    }
    if (!valid) {invalid(status);continue;}
    const float max_value=__uint_as_float(maximum<<16);
    const auto scale=rounded(__fmul_rn(max_value,0.00787353515625F));
    const auto inverse=(__bfloat16_as_ushort(scale)&0x7fffU)==0
        ?rounded(0.F):rounded(__fdiv_rn(1.F,__bfloat162float(scale)));
    scales[i]=scale;
    for (unsigned lane=0;lane<kGroup;++lane) {
      const auto value=rounded(__fmul_rn(__bfloat162float(input[source+lane]),
                                       __bfloat162float(inverse)));
      const auto magnitude=__bfloat16_as_ushort(value)&0x7fffU;
      const float v=__bfloat162float(value);
      // Tensor.to(I8) is portable only when the truncated value fits.127.5
      // is valid; +128 is not. Never invoke an out-of-domain integer cast.
      if (magnitude>=0x7f80U || v>=128.F || v<=-129.F) {
        invalid(status);valid=false;break;
      }
      codes[row*p.depth+group*kGroup+lane]=static_cast<signed char>(__float2int_rz(v));
    }
    if (!valid) continue;
  }
}
template<bool Grouped>
__global__ void linear(Program p,const std::uint32_t *weight,const float *weight_scales,
    const int *ids,const signed char *activation,const __nv_bfloat16 *activation_scales,
    __nv_bfloat16 *output,unsigned *status) {
  if (failed(status)) return;
  const Index rows=p.tokens*p.slots,count=rows*p.columns,groups=p.depth/kGroup;
  for (Index i=Index(blockIdx.x)*blockDim.x+threadIdx.x;i<count;
       i+=Index(gridDim.x)*blockDim.x) {
    const Index row=i/p.columns,column=i%p.columns;
    Index weight_row=column,activation_row=row;
    if constexpr (Grouped) {
      // Complete IDs were checked by a preceding kernel on this same stream.
      weight_row=Index(ids[row])*p.columns+column;
      if (p.shared) activation_row=row/p.slots;
    }
    auto accumulated=rounded(0.F);
    bool valid=true;
    for (Index group=0;group<groups;++group) {
      const float raw_scale=weight_scales[weight_row*groups+group];
      const unsigned scale_bits=__float_as_uint(raw_scale);
      if (scale_bits==0 || scale_bits>=0x7f800000U) {
        invalid(status);valid=false;break;
      }
      int dot=0;
      const Index first=group*kGroup;
      for (unsigned lane=0;lane<kGroup;++lane) {
        const Index depth=first+lane;
        const auto word=weight[weight_row*(p.depth/8)+depth/8];
        const int code=int((word>>(4*(depth%8)))&15U)-8;
        dot+=int(activation[activation_row*p.depth+depth])*code;
      }
      // |dot|<=512*128*8, so signed I32 accumulation is exact and safe.
      // Every following BF16 boundary is separate and in increasing K order.
      const auto dot_b=rounded(static_cast<float>(dot));
      const auto sa=activation_scales[activation_row*groups+group];
      const auto partial=rounded(__fmul_rn(__bfloat162float(dot_b),__bfloat162float(sa)));
      const auto sf=rounded(raw_scale);
      const auto scaled=rounded(__fmul_rn(__bfloat162float(partial),__bfloat162float(sf)));
      accumulated=rounded(__fadd_rn(__bfloat162float(accumulated),__bfloat162float(scaled)));
    }
    if (valid) output[i]=accumulated;
  }
}
unsigned grid(Index count) noexcept {
  const Index blocks=(count+kThreads-1)/kThreads;
  return unsigned(blocks<65535?blocks:65535);
}
} // namespace
cudaError_t prepare_kernel(Program &p) noexcept {
  if (p.grouped) {
    p.quantize_kernel=reinterpret_cast<const void *>(quantize<true>);
    p.linear_kernel=reinterpret_cast<const void *>(linear<true>);
  } else {
    p.quantize_kernel=reinterpret_cast<const void *>(quantize<false>);
    p.linear_kernel=reinterpret_cast<const void *>(linear<false>);
  }
  cudaFuncAttributes attributes{};
  if (p.grouped) {
    const auto status=cudaFuncGetAttributes(&attributes,check_ids);
    if (status!=cudaSuccess) return status;
  }
  auto status=cudaFuncGetAttributes(&attributes,p.quantize_kernel);
  if (status!=cudaSuccess) return status;
  return cudaFuncGetAttributes(&attributes,p.linear_kernel);
}
cudaError_t launch(const Program &p,const void *const *inputs,void *workspace,
                   void *destination,unsigned *status,cudaStream_t stream) noexcept {
  auto program=p;
  if (p.grouped) {
    const auto *ids=static_cast<const int *>(inputs[3]);
    void *args[]{&program,&ids,&status};
    const auto result=cudaLaunchKernel(reinterpret_cast<const void *>(check_ids),
        dim3(grid(p.tokens*p.slots)),dim3(kThreads),args,0,stream);
    if (result!=cudaSuccess) return result;
  }
  const auto *input=static_cast<const __nv_bfloat16 *>(inputs[0]);
  auto *codes=static_cast<signed char *>(workspace);
  auto *scales=reinterpret_cast<__nv_bfloat16 *>(codes+p.quant_rows*p.depth);
  void *quantize_args[]{&program,&input,&codes,&scales,&status};
  auto result=cudaLaunchKernel(p.quantize_kernel,dim3(grid(p.quant_rows*(p.depth/kGroup))),
      dim3(kThreads),quantize_args,0,stream);
  if (result!=cudaSuccess) return result;
  const auto *weight=static_cast<const std::uint32_t *>(inputs[1]);
  const auto *sf=static_cast<const float *>(inputs[2]);
  const auto *ids=p.grouped?static_cast<const int *>(inputs[3]):nullptr;
  const auto *quantized=codes;
  const auto *activation_scales=scales;
  auto *output=static_cast<__nv_bfloat16 *>(destination);
  void *linear_args[]{&program,&weight,&sf,&ids,&quantized,&activation_scales,&output,&status};
  return cudaLaunchKernel(p.linear_kernel,dim3(grid(p.tokens*p.slots*p.columns)),
      dim3(kThreads),linear_args,0,stream);
}
} // namespace hyle::compute::cuda_native::blockwise_w4a8
