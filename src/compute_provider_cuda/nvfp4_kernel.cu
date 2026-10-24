#include "nvfp4_kernel.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda/atomic>
#include "device_status_kernel.h"
#include "packed_rows.cuh"

namespace hyle::compute::cuda_native::nvfp4 {
namespace {
using Index=unsigned long long;
enum class Scalar { kFloat16, kBFloat16, kFloat32 };
constexpr unsigned kThreads=256;
__device__ void invalid(unsigned *status) {
  atomicCAS(status,0U,unsigned(cuda::detail::DeviceStatusCode::kInvalidNumericInput));
}
__device__ __forceinline__ bool failed(unsigned *status) {
  return ::cuda::atomic_ref<unsigned,::cuda::thread_scope_device>(*status)
      .load(::cuda::std::memory_order_relaxed)!=0;
}
__device__ float fp8(unsigned char code) {
  return __half2float(__half(__nv_cvt_fp8_to_halfraw(code,__NV_E4M3)));
}
__device__ float e2m1(float x) {
  if (x<=.25F) return 0.F;
  if (x<.75F) return .5F;
  if (x<=1.25F) return 1.F;
  if (x<1.75F) return 1.5F;
  if (x<=2.5F) return 2.F;
  if (x<3.5F) return 3.F;
  return x<=5.F ? 4.F : 6.F;
}
// E8M0 code zero is 2^-127, a F32 subnormal, rather than zero.
// Code 255 is rejected by the weight loader before this conversion.
__device__ __forceinline__ float e8m0(unsigned code) {
  return __uint_as_float(code==0?0x00400000U:code<<23);
}
__device__ __forceinline__ void quantize_mxfp4_group(
    const __nv_bfloat16 *input,__nv_bfloat16 *activation,unsigned *status) {
  unsigned maximum=0;
  float values[32];
  bool valid=true;
  #pragma unroll
  for (unsigned j=0;j<32;++j) {
    const unsigned bits=__bfloat16_as_ushort(input[j])&0x7fffU;
    valid &= bits<0x7f80U;
    maximum=maximum>bits?maximum:bits;
    values[j]=__bfloat162float(input[j]);
  }
  if (!valid) {invalid(status);return;}
  const unsigned rounded=(maximum+0x20U)&0x7f80U;
  // Portable finite-BF16 reference: handle zero and finite-to-Inf carry
  // explicitly; the producer's undefined log2(0) cast is not imported.
  int exponent=rounded==0? -127 : rounded==0x7f80U?127:
      int(rounded>>7)-127-2;
  exponent=exponent< -127? -127:exponent>127?127:exponent;
  const float scale=e8m0(unsigned(exponent+127));
  #pragma unroll
  for (unsigned j=0;j<32;++j) {
    const float divided=__bfloat162float(
        __float2bfloat16_rn(__fdiv_rn(values[j],scale)));
    const float quantized=copysignf(e2m1(fabsf(divided)),divided);
    activation[j]=__float2bfloat16_rn(__fmul_rn(quantized,scale));
  }
}
__global__ void quantize_mxfp4(Program p,const __nv_bfloat16 *input,
    __nv_bfloat16 *activation,unsigned *status) {
  if (failed(status)) return;
  const bool shared=p.grouped &&
      (p.activation_slots==1 || p.activation_slot_stride==0);
  const Index groups_per_row=p.depth/32,groups=p.quant_rows*groups_per_row;
  for (Index group=Index(blockIdx.x)*blockDim.x+threadIdx.x;group<groups;
       group+=Index(gridDim.x)*blockDim.x) {
    const Index row=group/groups_per_row;
    const Index token=p.grouped?(shared?row:row/p.slots):row;
    const Index slot=p.grouped && !shared?row%p.slots:0;
    const Index source=p.grouped?
        token*p.activation_token_stride+slot*p.activation_slot_stride:
        row*p.depth;
    quantize_mxfp4_group(input+source+(group%groups_per_row)*32,
                        activation+group*32,status);
  }
}
__global__ void validate_mxfp4(Program p,const __nv_bfloat16 *input,
    __nv_bfloat16 *,unsigned *status) {
  if (failed(status)) return;
  const bool shared=p.grouped &&
      (p.activation_slots==1 || p.activation_slot_stride==0);
  const Index elements=p.quant_rows*p.depth;
  for (Index element=Index(blockIdx.x)*blockDim.x+threadIdx.x;element<elements;
       element+=Index(gridDim.x)*blockDim.x) {
    const Index row=element/p.depth,k=element%p.depth;
    const Index token=p.grouped?(shared?row:row/p.slots):row;
    const Index slot=p.grouped && !shared?row%p.slots:0;
    const Index source=p.grouped?
        token*p.activation_token_stride+slot*p.activation_slot_stride+k:
        element;
    if ((__bfloat16_as_ushort(input[source])&0x7fffU)>=0x7f80U)
      invalid(status);
  }
}
template<Contract C>
__device__ __forceinline__ void quantize_group(const __nv_bfloat16 *input,
    __nv_bfloat16 *activation, float g, unsigned *status) {
    float values[16],maximum=0.F;
    bool valid=true;
    #pragma unroll
    for (unsigned j=0;j<16;++j) {
      values[j]=__bfloat162float(input[j]);
      valid &= isfinite(values[j]);
      maximum=fmaxf(maximum,fabsf(values[j]));
    }
    if (!valid) {invalid(status);return;}
    float scale;
    if constexpr (C==Contract::kModelOpt) {
      const float ratio=fminf(__fdiv_rn(maximum,__fmul_rn(6.F,g)),448.F);
      const auto code=__nv_cvt_float_to_fp8(ratio,__NV_SATFINITE,__NV_E4M3);
      scale=__fmul_rn(fp8(code),g);
      if (!(scale>=1.e-5F)) scale=1.F;
    } else {
      const float divided=__bfloat162float(
          __float2bfloat16_rn(__fdiv_rn(maximum,6.F)));
      const float ratio=fminf(__fmul_rn(g,divided),448.F);
      const auto code=__nv_cvt_float_to_fp8(ratio,__NV_SATFINITE,__NV_E4M3);
      const float block=fp8(code);
      scale=__fdiv_rn(block==0.F?.125F:block,g);
    }
    #pragma unroll
    for (unsigned j=0;j<16;++j) {
      if constexpr (C==Contract::kModelOpt) {
        const float rounded=e2m1(__fdiv_rn(fabsf(values[j]),scale));
        const float value=__fmul_rn(rounded,scale);
        activation[j]=__float2bfloat16_rn(values[j]>=0.F?value:-value);
      } else {
        float value=fmaxf(-6.F,fminf(__fdiv_rn(values[j],scale),6.F));
        if constexpr (C==Contract::kLocalBFloat16)
          value=__bfloat162float(__float2bfloat16_rn(value));
        const float magnitude=e2m1(fabsf(value));
        // Portable Torch sign convention: sign(-0)=0, sign(negative)=-1.
        const float quantized=value==0.F?0.F:copysignf(magnitude,value);
        activation[j]=__float2bfloat16_rn(__fmul_rn(quantized,scale));
      }
    }
}
template<Contract C>
__global__ void quantize(Program p, const __nv_bfloat16 *input,
    const float *weight_global, const float *input_global,
    __nv_bfloat16 *activation, unsigned *status) {
  const float g=*input_global,wg=*weight_global;
  if (!(g>0.F) || !isfinite(g) || !(wg>0.F) || !isfinite(wg)) {
    invalid(status);return;
  }
  const Index groups=p.rows*(p.depth/16);
  for (Index group=Index(blockIdx.x)*blockDim.x+threadIdx.x;group<groups;
       group+=Index(gridDim.x)*blockDim.x)
    quantize_group<C>(input+group*16,activation+group*16,g,status);
}
__global__ void check_ids(Program p, const std::int32_t *ids, unsigned *status) {
  if (failed(status)) return;
  for (Index row=Index(blockIdx.x)*blockDim.x+threadIdx.x;row<p.rows;
       row+=Index(gridDim.x)*blockDim.x) {
    const auto expert=ids[row];
    if (expert<0 || Index(expert)>=p.experts)
      atomicCAS(status,0U,unsigned(cuda::detail::DeviceStatusCode::kIndexOutOfRange));
  }
}
template<Contract C>
__global__ void quantize_grouped(Program p, const __nv_bfloat16 *input,
    const float *weight_global, const float *input_global,
    const std::int32_t *ids, __nv_bfloat16 *activation, unsigned *status) {
  // The preceding kernel has checked every ID before any expert addressing.
  if (failed(status)) return;
  const Index groups_per_row=p.depth/16,groups=p.rows*groups_per_row;
  for (Index group=Index(blockIdx.x)*blockDim.x+threadIdx.x;group<groups;
       group+=Index(gridDim.x)*blockDim.x) {
    const Index row=group/groups_per_row,expert=Index(ids[row]);
    const float g=input_global[expert],wg=weight_global[expert];
    if (!(g>0.F) || !isfinite(g) || !(wg>0.F) || !isfinite(wg)) {
      invalid(status);continue;
    }
    const Index token=row/p.slots,slot=p.activation_slots==1?0:row%p.slots;
    const Index source=token*p.activation_token_stride+slot*p.activation_slot_stride+
                       (group%groups_per_row)*16;
    quantize_group<C>(input+source,activation+group*16,g,status);
  }
}
template<bool Local>
struct Group final {
  const unsigned char *weights;
  Index row,depth;
  float scale;
  __device__ __forceinline__ float value(Index k) const {
    const auto byte=weights[row*(depth/2)+k/2];
    const unsigned code=(byte>>(4*unsigned(k%2)))&15;
    // ModelOpt's original table maps code8 to+0; local storage retains-0.
    const float values[16]{0.F,.5F,1.F,1.5F,2.F,3.F,4.F,6.F,
                          0.F,-.5F,-1.F,-1.5F,-2.F,-3.F,-4.F,-6.F};
    const float decoded=Local && code==8?-0.F:values[code];
    return __bfloat162float(__float2bfloat16_rn(__fmul_rn(decoded,scale)));
  }
  __device__ __forceinline__ float finish(float dot) const {return dot;}
};
template<bool Grouped,bool Local>
struct Rows final {
  using GroupIndex=Index;
  Program p;
  const unsigned char *weights,*scales;
  const float *global;
  const std::int32_t *ids;
  unsigned *status;
  __device__ __forceinline__ Index activation_offset(Index row) const {return row*p.depth;}
  __device__ __forceinline__ Index weight_row(Index row,Index column) const {
    if constexpr (Grouped) return Index(ids[row])*p.columns+column;
    return column;
  }
  __device__ __forceinline__ Group<Local> group(Index row,Index group) const {
    const auto code=scales[row*(p.depth/16)+group];
    const float scale=fp8(code);
    if (!(scale>=0.F) || !isfinite(scale)) invalid(status);
    const Index expert=Grouped?row/p.columns:0;
    const float effective=Local?__fdiv_rn(scale,global[expert]):
                                __fmul_rn(scale,global[expert]);
    return {weights,row,p.depth,effective};
  }
};
template<bool Grouped,bool OriginalActivation>
struct Mxfp4Rows final {
  using GroupIndex=Index;
  Program p;
  const unsigned char *weights,*scales;
  const std::int32_t *ids;
  unsigned *status;
  __device__ __forceinline__ Index activation_offset(Index row) const {
    if constexpr (Grouped && OriginalActivation)
      return (row/p.slots)*p.activation_token_stride+
          (p.activation_slots==1?0:row%p.slots)*p.activation_slot_stride;
    if constexpr (Grouped)
      if (p.activation_slots==1 || p.activation_slot_stride==0)
        return (row/p.slots)*p.depth;
    return row*p.depth;
  }
  __device__ __forceinline__ Index weight_row(Index row,Index column) const {
    if constexpr (Grouped) return Index(ids[row])*p.columns+column;
    return column;
  }
  __device__ __forceinline__ Group<true> group(Index row,Index group) const {
    const auto code=scales[row*(p.depth/32)+group];
    if (code==255) invalid(status);
    // Reuse the signed adjacent-nibble decoder and its BF16 product boundary.
    // Only selected expert rows are visited; unselected scale bytes are inert.
    float scale=code==255?0.F:e8m0(code);
    if constexpr (OriginalActivation)
      scale=__bfloat162float(__float2bfloat16_rn(scale));
    return {weights,row,p.depth,scale};
  }
};
template<unsigned Count,bool Grouped,bool OriginalActivation>
__global__ void linear_mxfp4(Program p,const __nv_bfloat16 *activation,
    const unsigned char *weights,const unsigned char *scales,
    const std::int32_t *ids,__nv_bfloat16 *output,unsigned *status) {
  if (failed(status)) return;
  packed::packed_rows<__nv_bfloat16,Count>(p.rows,p.columns,p.depth,32,
      Scalar::kBFloat16,activation,output,
      Mxfp4Rows<Grouped,OriginalActivation>{p,weights,scales,ids,status});
}
template<unsigned Count,bool Grouped,bool Local>
__global__ void linear(Program p, const __nv_bfloat16 *activation,
    const unsigned char *weights,const unsigned char *scales,const float *global,
    const std::int32_t *ids,__nv_bfloat16 *output,unsigned *status) {
  if (failed(status)) return;
  packed::packed_rows<__nv_bfloat16,Count>(p.rows,p.columns,p.depth,16,
      Scalar::kBFloat16,activation,output,Rows<Grouped,Local>{p,weights,scales,global,ids,status});
}
unsigned blocks(Index items, unsigned per_block) {
  const auto count=items/per_block+(items%per_block!=0);
  return unsigned(count<65535?count:65535);
}
template<Contract C>
cudaError_t configure(Program &p) noexcept {
  constexpr bool local=C!=Contract::kModelOpt;
  p.quantize_kernel=p.grouped?reinterpret_cast<const void *>(quantize_grouped<C>):
                               reinterpret_cast<const void *>(quantize<C>);
  p.linear_kernel=p.grouped?reinterpret_cast<const void *>(linear<1,true,local>):
      (p.rows>1?reinterpret_cast<const void *>(linear<4,false,local>):
                reinterpret_cast<const void *>(linear<1,false,local>));
  cudaFuncAttributes attributes;
  auto status=cudaFuncGetAttributes(&attributes,p.quantize_kernel);
  if (status==cudaSuccess && p.grouped)
    status=cudaFuncGetAttributes(&attributes,check_ids);
  if (status==cudaSuccess)
    status=cudaFuncGetAttributes(&attributes,p.linear_kernel);
  return status;
}
template<bool OriginalActivation>
cudaError_t configure_mxfp4(Program &p) noexcept {
  p.quantize_kernel=OriginalActivation?reinterpret_cast<const void *>(validate_mxfp4):
      reinterpret_cast<const void *>(quantize_mxfp4);
  p.linear_kernel=p.grouped?
      reinterpret_cast<const void *>(linear_mxfp4<1,true,OriginalActivation>):
      (p.rows>1?reinterpret_cast<const void *>(linear_mxfp4<4,false,OriginalActivation>):
                reinterpret_cast<const void *>(linear_mxfp4<1,false,OriginalActivation>));
  cudaFuncAttributes attributes;
  auto status=cudaFuncGetAttributes(&attributes,p.quantize_kernel);
  if (status==cudaSuccess && p.grouped)
    status=cudaFuncGetAttributes(&attributes,check_ids);
  if (status==cudaSuccess)
    status=cudaFuncGetAttributes(&attributes,p.linear_kernel);
  return status;
}
cudaError_t launch_mxfp4(const Program &p,const void *const *inputs,void *activation,
    void *output,unsigned *status,cudaStream_t stream) noexcept {
  Program spec=p;
  const auto *input=static_cast<const __nv_bfloat16 *>(inputs[0]);
  const auto *ids=p.grouped?static_cast<const std::int32_t *>(inputs[3]):nullptr;
  auto *a=static_cast<__nv_bfloat16 *>(activation);
  auto result=cudaSuccess;
  if (p.grouped) {
    // The entire ID tensor is checked in a preceding kernel before the
    // selected-bank loader can perform any expert-derived addressing.
    void *id_args[]{&spec,&ids,&status};
    result=cudaLaunchKernel(reinterpret_cast<const void *>(check_ids),
        dim3(blocks(p.rows,kThreads)),dim3(kThreads),id_args,0,stream);
    if (result!=cudaSuccess) return result;
  }
  void *quant_args[]{&spec,&input,&a,&status};
  const bool original=p.contract==Contract::kMxfp4BFloat16;
  result=cudaLaunchKernel(p.quantize_kernel,
      dim3(blocks(p.quant_rows*(original?p.depth:p.depth/32),kThreads)),
      dim3(kThreads),quant_args,0,stream);
  if (result!=cudaSuccess) return result;
  const __nv_bfloat16 *linear_a=original?input:a;
  const auto *w=static_cast<const unsigned char *>(inputs[1]);
  const auto *s=static_cast<const unsigned char *>(inputs[2]);
  auto *y=static_cast<__nv_bfloat16 *>(output);
  void *args[]{&spec,&linear_a,&w,&s,&ids,&y,&status};
  const bool share_rows=!p.grouped && p.rows>1;
  const auto rows=share_rows?p.rows/4+(p.rows%4!=0):p.rows;
  return cudaLaunchKernel(p.linear_kernel,dim3(blocks(rows*p.columns,8)),dim3(kThreads),args,0,stream);
}
} // namespace
cudaError_t prepare_kernel(Program &p) noexcept {
  switch (p.contract) {
  case Contract::kModelOpt: return configure<Contract::kModelOpt>(p);
  case Contract::kLocalFloat32: return configure<Contract::kLocalFloat32>(p);
  case Contract::kLocalBFloat16: return configure<Contract::kLocalBFloat16>(p);
  case Contract::kMxfp4: return configure_mxfp4<false>(p);
  case Contract::kMxfp4BFloat16: return configure_mxfp4<true>(p);
  }
  return cudaErrorInvalidValue;
}
cudaError_t launch(const Program &p,const void *const *inputs,void *activation,
    void *output,unsigned *status,cudaStream_t stream) noexcept {
  if (!p.quantize_kernel || !p.linear_kernel) return cudaErrorInvalidValue;
  if (p.contract==Contract::kMxfp4 || p.contract==Contract::kMxfp4BFloat16)
    return launch_mxfp4(p,inputs,activation,output,status,stream);
  Program spec=p;
  const auto *input=static_cast<const __nv_bfloat16 *>(inputs[0]);
  const auto *wg=static_cast<const float *>(inputs[3]);
  const auto *ag=static_cast<const float *>(inputs[4]);
  const auto *ids=p.grouped?static_cast<const std::int32_t *>(inputs[5]):nullptr;
  auto *a=static_cast<__nv_bfloat16 *>(activation);
  auto result=cudaSuccess;
  if (p.grouped) {
    void *id_args[]{&spec,&ids,&status};
    result=cudaLaunchKernel(reinterpret_cast<const void *>(check_ids),
        dim3(blocks(p.rows,kThreads)),dim3(kThreads),id_args,0,stream);
    if (result!=cudaSuccess) return result;
    void *quant_args[]{&spec,&input,&wg,&ag,&ids,&a,&status};
    result=cudaLaunchKernel(p.quantize_kernel,
        dim3(blocks(p.rows*(p.depth/16),kThreads)),dim3(kThreads),quant_args,0,stream);
  } else {
    void *quant_args[]{&spec,&input,&wg,&ag,&a,&status};
    result=cudaLaunchKernel(p.quantize_kernel,
        dim3(blocks(p.rows*(p.depth/16),kThreads)),dim3(kThreads),quant_args,0,stream);
  }
  if (result!=cudaSuccess) return result;
  const auto *w=static_cast<const unsigned char *>(inputs[1]);
  const auto *s=static_cast<const unsigned char *>(inputs[2]);
  auto *y=static_cast<__nv_bfloat16 *>(output);
  void *args[]{&spec,&a,&w,&s,&wg,&ids,&y,&status};
  // Rows=4 shares decoded weights, which is valid only for the dense bank.
  const bool share_rows=!p.grouped && p.rows>1;
  const auto rows=share_rows?p.rows/4+(p.rows%4!=0):p.rows;
  return cudaLaunchKernel(p.linear_kernel,dim3(blocks(rows*p.columns,8)),dim3(kThreads),args,0,stream);
}
} // namespace hyle::compute::cuda_native::nvfp4
