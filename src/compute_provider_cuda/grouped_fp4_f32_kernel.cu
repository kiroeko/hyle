// HYLE-owned original-FP4 expert GEMM. An ordinary same-build CUDA kernel;
// the independent original A128 E4 quantizer remains unchanged.
//
// E4 and E2M1 codes widen EXACTLY to normal BF16 (or signed zero). Each
// original K32 code dot uses BF16-input/F32-accumulator MMA, followed by the
// original two F32 scale multiplies and F32 outer accumulation. In particular
// no FP8 QMMA partial is assumed to have IEEE F32 numerical precision.
#include "grouped_fp4_f32_kernel.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "device_status_kernel.h"

namespace {
using Index=unsigned long long;
constexpr unsigned kThreads=128,kColumns=64,kBlock=32;

__device__ __forceinline__ void invalid(unsigned *status) {
  atomicCAS(status,0U,static_cast<unsigned>(hyle::compute::cuda::detail::DeviceStatusCode::kInvalidNumericInput));
}
__device__ __forceinline__ __nv_bfloat16 e4(unsigned char encoded,unsigned *status) {
  const unsigned magnitude=encoded&127U,exponent=magnitude>>3U,mantissa=magnitude&7U;
  if(magnitude==127U) {invalid(status);return __ushort_as_bfloat16(0);}
  const unsigned sign=unsigned(encoded&128U)<<8U;
  if(exponent==0U) {
    const auto positive=__float2bfloat16_rn(float(mantissa)*0x1p-9F);
    return __ushort_as_bfloat16(static_cast<unsigned short>(__bfloat16_as_ushort(positive)|sign));
  }
  return __ushort_as_bfloat16(static_cast<unsigned short>(sign|((exponent+120U)<<7U)|(mantissa<<4U)));
}
__device__ __forceinline__ __nv_bfloat16 e2m1(unsigned code) {
  const unsigned magnitude=code&7U,sign=(code&8U)<<12U;
  if(magnitude<2U) {
    const unsigned value=magnitude==0U?0U:0x3f00U;
    return __ushort_as_bfloat16(static_cast<unsigned short>(sign|value));
  }
  const unsigned value=(((magnitude>>1U)+126U)<<7U)|((magnitude&1U)<<6U);
  return __ushort_as_bfloat16(static_cast<unsigned short>(sign|value));
}
} // namespace

// The first ten arguments retain the original indexed-FP4 ABI. Runtime N/K
// are the final two arguments, so all four supported geometries share this
// one kernel. This same external symbol can be compiled separately as a
// diagnostic CUBIN; product execution uses the normal CUDA TU/host adapter.
// Only code tiles are decoded; the full W stays packed.
extern "C" __global__ __launch_bounds__(kThreads,1) void hyle_grouped_fp4_f32(
    const unsigned char *activation,const unsigned char *weight,__nv_bfloat16 *output,
    const float *activation_scales,const float *weight_scales,const int *ids,
    unsigned *status,int experts,int slots_per_token,int activation_slots,int columns,int depth) {
  using namespace nvcuda;
  __shared__ __align__(32) __nv_bfloat16 a_tile[16*kBlock];
  __shared__ __align__(32) __nv_bfloat16 b_tile[kColumns*kBlock];
  __shared__ __align__(32) float partial[4*16*16];
  __shared__ int selected,proceed;
  const unsigned thread=threadIdx.x,warp=thread/32U,lane=thread%32U;
  const Index slot=blockIdx.z;
  if(thread==0U) {
    selected=ids[slot];
    const bool valid=selected>=0&&selected<experts;
    if(!valid)atomicCAS(status,0U,static_cast<unsigned>(hyle::compute::cuda::detail::DeviceStatusCode::kIndexOutOfRange));
    proceed=valid&&atomicAdd(status,0U)==0U;
  }
  __syncthreads();
  if(!proceed)return; // CTA-uniform, before any collective WMMA operation.
  const Index n=static_cast<unsigned>(columns),k=static_cast<unsigned>(depth);
  const Index row=activation_slots==1?slot/static_cast<unsigned>(slots_per_token):slot;
  const Index first=Index(blockIdx.x)*kColumns;
  const Index column=first+warp*16U+lane;
  const Index blocks=k/kBlock;
  for(unsigned i=thread;i<16*kBlock;i+=kThreads)a_tile[i]=__ushort_as_bfloat16(0);
  __syncthreads();
  float accumulated=0.F;
  for(Index block=0;block<blocks;++block) {
    if(thread<kBlock)a_tile[thread]=e4(activation[row*k+block*kBlock+thread],status);
    for(unsigned element=thread;element<kColumns*kBlock;element+=kThreads) {
      const Index weight_column=first+element/kBlock;
      const Index reduction=block*kBlock+element%kBlock;
      unsigned code=0;
      if(weight_column<n) {
        const auto packed=weight[(Index(selected)*n+weight_column)*(k/2)+reduction/2];
        code=(unsigned(packed)>>(4U*unsigned(reduction%2)))&15U;
      }
      b_tile[element]=e2m1(code);
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> lhs;
    wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> rhs;
    wmma::fragment<wmma::accumulator,16,16,16,float> dot;
    wmma::fill_fragment(dot,0.F);
    // B is stored [N64,K32]. This is the column-major RHS [K32,N64]
    // required by the ordinary A@W^T contract; no weight reorder is cached.
    wmma::load_matrix_sync(lhs,a_tile,kBlock);
    wmma::load_matrix_sync(rhs,b_tile+warp*16*kBlock,kBlock);
    wmma::mma_sync(dot,lhs,rhs,dot);
    wmma::load_matrix_sync(lhs,a_tile+16,kBlock);
    wmma::load_matrix_sync(rhs,b_tile+warp*16*kBlock+16,kBlock);
    wmma::mma_sync(dot,lhs,rhs,dot);
    wmma::store_matrix_sync(partial+warp*16*16,dot,16,wmma::mem_row_major);
    __syncthreads();
    if(lane<16U&&column<n) {
      const float a_scale=activation_scales[row*(k/128)+block/4];
      const float w_scale=weight_scales[(Index(selected)*n+column)*blocks+block];
      if(!(a_scale>0.F)||!isfinite(a_scale)||!(w_scale>0.F)||!isfinite(w_scale))invalid(status);
      const float scaled=__fmul_rn(__fmul_rn(partial[warp*16*16+lane],a_scale),w_scale);
      accumulated=__fadd_rn(accumulated,scaled);
    }
    __syncthreads(); // All consumers finish before the next code tile write.
  }
  if(lane<16U&&column<n) {
    if(!isfinite(accumulated))invalid(status);
    output[slot*n+column]=__float2bfloat16_rn(accumulated);
  }
}

namespace hyle::compute::cuda_native::deepseek_fp4::native {
cudaError_t prepare(Program *program) noexcept {
  if(!program||program->experts<=0||program->columns<=0||program->depth<=0||program->depth%128||
      program->slots<=0||(program->activation_slots!=1&&program->activation_slots!=program->slots)||
      program->rows<=0||program->rows%program->slots||program->rows>65535)
    return cudaErrorInvalidValue;
  cudaFuncAttributes attributes{};
  const auto result=cudaFuncGetAttributes(&attributes,hyle_grouped_fp4_f32);
  if(result!=cudaSuccess)return result;
  if(attributes.maxThreadsPerBlock<static_cast<int>(kThreads))return cudaErrorNotSupported;
  program->kernel=reinterpret_cast<const void *>(hyle_grouped_fp4_f32);
  return cudaSuccess;
}
cudaError_t launch(const Program &program,const unsigned char *activation,
    const unsigned char *weight,void *output,const float *activation_scales,
    const float *weight_scales,const int *ids,unsigned *status,cudaStream_t stream) noexcept {
  int experts=program.experts,slots=program.slots,activation_slots=program.activation_slots,
      columns=program.columns,depth=program.depth;
  void *arguments[]{&activation,&weight,&output,&activation_scales,&weight_scales,
      &ids,&status,&experts,&slots,&activation_slots,&columns,&depth};
  return cudaLaunchKernel(program.kernel,dim3((static_cast<unsigned>(columns)+kColumns-1)/kColumns,1,
      static_cast<unsigned>(program.rows)),dim3(kThreads),arguments,0,stream);
}
} // namespace hyle::compute::cuda_native::deepseek_fp4::native
