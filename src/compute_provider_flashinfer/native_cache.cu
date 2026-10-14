#include "native_batch.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/cascade.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variants.cuh>
#include <algorithm>
#include <stdexcept>
#include <type_traits>

namespace hyle::compute::flashinfer::native {
namespace {
template<unsigned Depth>cudaError_t merge_f32(float *values,float *scores,std::int32_t *indptr,
    float *output,float *lse,unsigned rows,unsigned *actual_rows,unsigned heads,cudaStream_t stream){
  // Bind the original persistent merge kernel only for this card's finite
  // depths. The generic upstream host dispatcher also instantiates D512,
  // whose 512-bit async copy is invalid for F32 inputs at the locked revision.
  constexpr unsigned vector=Depth/32>4?Depth/32:4;
  constexpr unsigned x=Depth/vector,y=128/x;
  constexpr unsigned shared=4*y*Depth*sizeof(float)+128*sizeof(float);
  auto kernel=::flashinfer::PersistentVariableLengthMergeStatesKernel<vector,x,y,4,float,float,std::int32_t>;
  int device=0,sms=0,occupancy=0;
  auto status=cudaGetDevice(&device);if(status!=cudaSuccess)return status;
  status=cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device);if(status!=cudaSuccess)return status;
  status=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,kernel,128,shared);if(status!=cudaSuccess)return status;
  const auto needed=(std::uint64_t(rows)*heads+unsigned(sms)-1)/unsigned(sms);
  const auto blocks=unsigned(sms)*unsigned((std::min)(std::uint64_t(occupancy),needed));
  status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,shared);if(status!=cudaSuccess)return status;
  void *arguments[]{&values,&scores,&indptr,&output,&lse,&rows,&actual_rows,&heads};
  return cudaLaunchKernel(reinterpret_cast<const void*>(kernel),dim3(blocks),dim3(x,y),arguments,shared,stream);
}
}
}
namespace flashinfer {
template<> inline cudaError_t VariableLengthMergeStates<float,float,std::int32_t>(
    float *values,float *scores,std::int32_t *indptr,float *output,float *lse,
    std::uint32_t rows,std::uint32_t *actual_rows,std::uint32_t heads,std::uint32_t depth,
    bool enable_pdl,cudaStream_t stream){
  if(enable_pdl)return cudaErrorNotSupported; // this SM89 domain always passes false
  if(!rows || !heads)return cudaSuccess;
  switch(depth){
  case 64:return hyle::compute::flashinfer::native::merge_f32<64>(values,scores,indptr,output,lse,rows,actual_rows,heads,stream);
  case 128:return hyle::compute::flashinfer::native::merge_f32<128>(values,scores,indptr,output,lse,rows,actual_rows,heads,stream);
  case 256:return hyle::compute::flashinfer::native::merge_f32<256>(values,scores,indptr,output,lse,rows,actual_rows,heads,stream);
  }
  return cudaErrorInvalidValue;
}
}
#include <flashinfer/attention/prefill.cuh>

namespace hyle::compute::flashinfer::native {
namespace fi=::flashinfer;

std::size_t batch_output_staging_bytes(BatchSpec spec) noexcept {
  if(!spec.fp8)return 0;
  const auto bytes=std::size_t(spec.query_rows)*spec.query_heads*spec.value_head_dim*sizeof(float);
  return (bytes+15)&~std::size_t(15);
}
namespace {
template<class T> __global__ void scale_cast(const float *input,T *output,
                                            std::uint64_t count,float scale){
  for(auto index=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;index<count;
      index+=std::uint64_t(gridDim.x)*blockDim.x)
    output[index]=static_cast<T>(input[index]*scale);
}
template<class Params>void bind_plan(Params &params,const fi::PrefillPlanInfo &plan,void *integer){
  const auto pointer=[&](std::int64_t offset){return reinterpret_cast<std::int32_t*>(static_cast<char*>(integer)+offset);};
  params.request_indices=pointer(plan.request_indices_offset);
  params.qo_tile_indices=pointer(plan.qo_tile_indices_offset);
  params.kv_tile_indices=pointer(plan.kv_tile_indices_offset);
  params.o_indptr=pointer(plan.o_indptr_offset);
  params.kv_chunk_size_ptr=pointer(plan.kv_chunk_size_ptr_offset);
  params.padded_batch_size=std::uint32_t(plan.padded_batch_size);
  params.max_total_num_rows=std::uint32_t(plan.total_num_rows);
  if(plan.split_kv)params.merge_indptr=pointer(plan.merge_indptr_offset);
}
template<unsigned QK,unsigned VO,bool Paged,fi::MaskMode Mask,class Params>
cudaError_t launch(Params params,const fi::PrefillPlanInfo &plan,void *floating,cudaStream_t stream){
  using Variant=fi::DefaultAttention<false,true,false,false>;
  auto *temporary=plan.split_kv?reinterpret_cast<typename Params::DTypeO*>(static_cast<char*>(floating)+plan.v_offset):nullptr;
  auto *scores=plan.split_kv?reinterpret_cast<float*>(static_cast<char*>(floating)+plan.s_offset):nullptr;
  switch(plan.cta_tile_q){
#define HYLE_CACHE_TILE(tile) case tile: if constexpr(Paged) return fi::BatchPrefillWithPagedKVCacheDispatched<tile,QK,VO,fi::PosEncodingMode::kNone,false,Mask,Variant>(params,temporary,scores,false,stream); else return fi::BatchPrefillWithRaggedKVCacheDispatched<tile,QK,VO,fi::PosEncodingMode::kNone,false,Mask,Variant>(params,temporary,scores,false,stream)
    HYLE_CACHE_TILE(16);HYLE_CACHE_TILE(64);HYLE_CACHE_TILE(128);
#undef HYLE_CACHE_TILE
  }
  throw std::runtime_error("FlashInfer cache CTA outside qualified dispatch");
}
template<unsigned QK,unsigned VO,bool Paged,class Params>
cudaError_t launch_mask(Params params,const fi::PrefillPlanInfo &plan,BatchSpec spec,void *floating,cudaStream_t stream){
  return spec.causal?launch<QK,VO,Paged,fi::MaskMode::kCausal>(params,plan,floating,stream):
                     launch<QK,VO,Paged,fi::MaskMode::kNone>(params,plan,floating,stream);
}
template<class Q,class KV,unsigned QK,unsigned VO>
cudaError_t run_typed(const fi::PrefillPlanInfo &plan,BatchSpec spec,const void *q,const void *k,
                     const void *v,const void *metadata,void *output,void *floating,void *integer,cudaStream_t stream){
  constexpr bool fp8=sizeof(KV)==1;
  using O=std::conditional_t<fp8,float,Q>;
  auto *native_output=fp8?reinterpret_cast<O*>(floating):static_cast<O*>(output);
  auto *plan_floating=static_cast<char*>(floating)+batch_output_staging_bytes(spec);
  const auto pointer=[&](std::size_t offset){return reinterpret_cast<std::int32_t*>(const_cast<char*>(static_cast<const char*>(metadata))+offset);};
  cudaError_t status;
  if(spec.paged){
    fi::BatchPrefillPagedParams<Q,KV,O,std::int32_t> params;
    params.q=const_cast<Q*>(static_cast<const Q*>(q));params.o=native_output;
    params.q_indptr=pointer(spec.query_indptr_offset);
    const std::int64_t ks[]{std::int64_t(spec.page_size)*spec.kv_heads*QK,std::int64_t(spec.kv_heads)*QK,QK,1};
    const std::int64_t vs[]{std::int64_t(spec.page_size)*spec.kv_heads*VO,std::int64_t(spec.kv_heads)*VO,VO,1};
    params.paged_kv=fi::paged_kv_t<KV,std::int32_t>(spec.kv_heads,spec.page_size,QK,spec.requests,fi::QKVLayout::kNHD,
      const_cast<KV*>(static_cast<const KV*>(k)),const_cast<KV*>(static_cast<const KV*>(v)),ks,vs,
      pointer(spec.page_indices_offset),pointer(spec.kv_indptr_offset),pointer(spec.tail_offset));
    params.num_qo_heads=spec.query_heads;params.group_size=fi::uint_fastdiv(spec.query_heads/spec.kv_heads);
    params.q_stride_n=spec.query_heads*QK;params.q_stride_h=QK;
    params.window_left=spec.window_left;params.sm_scale=spec.scale*spec.key_scale;bind_plan(params,plan,integer);
    status=launch_mask<QK,VO,true>(params,plan,spec,plan_floating,stream);
  }else{
    fi::BatchPrefillRaggedParams<Q,KV,O,std::int32_t> params;
    params.q=const_cast<Q*>(static_cast<const Q*>(q));params.k=const_cast<KV*>(static_cast<const KV*>(k));
    params.v=const_cast<KV*>(static_cast<const KV*>(v));params.o=native_output;
    params.q_indptr=pointer(spec.query_indptr_offset);params.kv_indptr=pointer(spec.kv_indptr_offset);
    params.num_qo_heads=spec.query_heads;params.num_kv_heads=spec.kv_heads;
    params.group_size=fi::uint_fastdiv(spec.query_heads/spec.kv_heads);
    params.q_stride_n=spec.query_heads*QK;params.q_stride_h=QK;
    params.k_stride_n=spec.kv_heads*QK;params.k_stride_h=QK;
    params.v_stride_n=spec.kv_heads*VO;params.v_stride_h=VO;
    params.window_left=spec.window_left;params.sm_scale=spec.scale*spec.key_scale;bind_plan(params,plan,integer);
    status=launch_mask<QK,VO,false>(params,plan,spec,plan_floating,stream);
  }
  if constexpr(fp8){
    if(status!=cudaSuccess)return status;
    const auto count=std::uint64_t(spec.query_rows)*spec.query_heads*VO;
    const auto blocks=static_cast<unsigned>((std::min)(std::uint64_t(65535),(count+255)/256));
    scale_cast<<<blocks,256,0,stream>>>(static_cast<const float*>(floating),static_cast<Q*>(output),count,spec.value_scale);
    return cudaGetLastError();
  }
  return status;
}
template<class Q,class KV>cudaError_t dispatch(const fi::PrefillPlanInfo &plan,BatchSpec spec,
    const void *q,const void *k,const void *v,const void *metadata,void *output,void *floating,void *integer,cudaStream_t stream){
  if(spec.qk_head_dim==64 && spec.value_head_dim==128)
    return run_typed<Q,KV,64,128>(plan,spec,q,k,v,metadata,output,floating,integer,stream);
  switch(spec.qk_head_dim){
  case 64:return run_typed<Q,KV,64,64>(plan,spec,q,k,v,metadata,output,floating,integer,stream);
  case 128:return run_typed<Q,KV,128,128>(plan,spec,q,k,v,metadata,output,floating,integer,stream);
  case 256:return run_typed<Q,KV,256,256>(plan,spec,q,k,v,metadata,output,floating,integer,stream);
  }
  throw std::runtime_error("FlashInfer cache dimension outside admitted domain");
}
template<class Q>cudaError_t dispatch_kv(const fi::PrefillPlanInfo &plan,BatchSpec spec,
    const void *q,const void *k,const void *v,const void *metadata,void *output,void *floating,void *integer,cudaStream_t stream){
  return spec.fp8?dispatch<Q,__nv_fp8_e4m3>(plan,spec,q,k,v,metadata,output,floating,integer,stream):
                  dispatch<Q,Q>(plan,spec,q,k,v,metadata,output,floating,integer,stream);
}
}
cudaError_t run_cache_batch(const void *state,BatchSpec spec,const void *q,const void *k,const void *v,
    const void *metadata,void *output,void *floating,void *integer,cudaStream_t stream){
  if(!spec.query_rows)return cudaSuccess;
  const auto &plan=*static_cast<const fi::PrefillPlanInfo*>(state);
  return spec.bfloat16?dispatch_kv<__nv_bfloat16>(plan,spec,q,k,v,metadata,output,floating,integer,stream):
                      dispatch_kv<half>(plan,spec,q,k,v,metadata,output,floating,integer,stream);
}
}
