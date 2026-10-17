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
template<class Q,unsigned Depth>cudaError_t merge_f32(float *values,float *scores,std::int32_t *indptr,
    Q *output,float *lse,unsigned rows,unsigned *actual_rows,unsigned heads,float scale,cudaStream_t stream){
  // Bind the original persistent merge kernel only for this card's finite
  // depths. The generic upstream host dispatcher also instantiates D512,
  // whose 512-bit async copy is invalid for F32 inputs at the locked revision.
  constexpr unsigned vector=Depth/32>4?Depth/32:4;
  constexpr unsigned x=Depth/vector,y=128/x;
  // Two F32 stages keep D256's byte budget equal to four BF16 stages.
  constexpr unsigned stages=2,shared=stages*y*Depth*sizeof(float)+128*sizeof(float);
  auto kernel=::flashinfer::PersistentVariableLengthMergeStatesKernel<vector,x,y,stages,float,Q,std::int32_t,true>;
  if(!rows || !heads)return cudaSuccess;
  int device=0,sms=0,occupancy=0;
  auto status=cudaGetDevice(&device);if(status!=cudaSuccess)return status;
  status=cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device);if(status!=cudaSuccess)return status;
  status=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,kernel,128,shared);if(status!=cudaSuccess)return status;
  const auto needed=(std::uint64_t(rows)*heads+unsigned(sms)-1)/unsigned(sms);
  const auto blocks=unsigned(sms)*unsigned((std::min)(std::uint64_t(occupancy),needed));
  status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,shared);if(status!=cudaSuccess)return status;
  void *arguments[]{&values,&scores,&indptr,&output,&lse,&rows,&actual_rows,&heads,&scale};
  return cudaLaunchKernel(reinterpret_cast<const void*>(kernel),dim3(blocks),dim3(x,y),arguments,shared,stream);
}
template<class Q,unsigned Depth>cudaError_t merge_bf16(Q *values,float *scores,std::int32_t *indptr,
    Q *output,float *lse,unsigned rows,unsigned *actual_rows,unsigned heads,float scale,cudaStream_t stream){
  static_assert(std::is_same_v<Q,__nv_bfloat16> && Depth==256);
  constexpr unsigned vector=Depth/32,x=Depth/vector,y=128/x;
  // Four BF16 stages use the original two-stage F32 shared-memory budget.
  constexpr unsigned stages=4,shared=stages*y*Depth*sizeof(Q)+128*sizeof(float);
  auto kernel=::flashinfer::PersistentVariableLengthMergeStatesKernel<vector,x,y,stages,Q,Q,std::int32_t,true>;
  if(!rows || !heads)return cudaSuccess;
  int device=0,sms=0,occupancy=0;
  auto status=cudaGetDevice(&device);if(status!=cudaSuccess)return status;
  status=cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device);if(status!=cudaSuccess)return status;
  status=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,kernel,128,shared);if(status!=cudaSuccess)return status;
  const auto needed=(std::uint64_t(rows)*heads+unsigned(sms)-1)/unsigned(sms);
  const auto blocks=unsigned(sms)*unsigned((std::min)(std::uint64_t(occupancy),needed));
  status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,shared);if(status!=cudaSuccess)return status;
  void *arguments[]{&values,&scores,&indptr,&output,&lse,&rows,&actual_rows,&heads,&scale};
  return cudaLaunchKernel(reinterpret_cast<const void*>(kernel),dim3(blocks),dim3(x,y),arguments,shared,stream);
}
}
}
#include <flashinfer/attention/prefill.cuh>

namespace hyle::compute::flashinfer::native {
namespace fi=::flashinfer;

namespace {
struct Bf16PartialAttention:fi::DefaultAttention<false,false,false,false> {
  using Base=fi::DefaultAttention<false,false,false,false>;
  using Base::Base;
  static constexpr bool hyle_bf16_partials=true;
};
template<class Base>struct OutputParams:Base {float v_scale=1.F;};

template<class Base,class Q,unsigned Depth>struct MergeParams:Base {
  Q *final_output=nullptr;
  float final_scale=1.F;
  cudaError_t merge_states(float *values,float *scores,float *final_lse,cudaStream_t stream) const {
    return merge_f32<Q,Depth>(values,scores,this->merge_indptr,final_output,final_lse,
        this->max_total_num_rows,this->total_num_rows,this->num_qo_heads,final_scale,stream);
  }
};

template<class Base,class Q,unsigned Depth>struct Bf16MergeParams:Base {
  Q *final_output=nullptr;
  float final_scale=1.F;
  cudaError_t merge_states(Q *values,float *scores,float *final_lse,cudaStream_t stream) const {
    return merge_bf16<Q,Depth>(values,scores,this->merge_indptr,final_output,final_lse,
        this->max_total_num_rows,this->total_num_rows,this->num_qo_heads,final_scale,stream);
  }
};

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
  using Variant=std::conditional_t<fi::HyleHasMergeStates<Params>::value &&
      std::is_same_v<typename Params::DTypeO,__nv_bfloat16>,Bf16PartialAttention,
      fi::DefaultAttention<false,true,false,false>>;
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
template<class Q,class KV,unsigned QK,unsigned VO,bool Split,bool Bf16Partials=false>
cudaError_t run_output(const fi::PrefillPlanInfo &plan,BatchSpec spec,const void *q,const void *k,
                     const void *v,const void *metadata,void *output,void *floating,void *integer,cudaStream_t stream){
  constexpr bool fp8=sizeof(KV)==1;
  using O=std::conditional_t<Split && !Bf16Partials,float,Q>;
  // The explicit BF16 option narrows partials; both paths scale after merging.
  auto *native_output=static_cast<O*>(output);
  const auto bind_output=[&](auto &params){
    if constexpr(Split){params.final_output=static_cast<Q*>(output);params.final_scale=spec.value_scale;}
    else params.v_scale=fp8?spec.value_scale:1.F;
  };
  const auto pointer=[&](std::size_t offset){return reinterpret_cast<std::int32_t*>(const_cast<char*>(static_cast<const char*>(metadata))+offset);};
  cudaError_t status;
  if(spec.paged){
    using Base=fi::BatchPrefillPagedParams<Q,KV,O,std::int32_t>;
    std::conditional_t<Split,std::conditional_t<Bf16Partials,Bf16MergeParams<Base,Q,VO>,MergeParams<Base,Q,VO>>,OutputParams<Base>> params;
    params.q=const_cast<Q*>(static_cast<const Q*>(q));params.o=native_output;
    params.q_indptr=pointer(spec.query_indptr_offset);
    const std::int64_t ks[]{std::int64_t(spec.page_size)*spec.kv_heads*QK,std::int64_t(spec.kv_heads)*QK,QK,1};
    const std::int64_t vs[]{std::int64_t(spec.page_size)*spec.kv_heads*VO,std::int64_t(spec.kv_heads)*VO,VO,1};
    params.paged_kv=fi::paged_kv_t<KV,std::int32_t>(spec.kv_heads,spec.page_size,QK,spec.requests,fi::QKVLayout::kNHD,
      const_cast<KV*>(static_cast<const KV*>(k)),const_cast<KV*>(static_cast<const KV*>(v)),ks,vs,
      pointer(spec.page_indices_offset),pointer(spec.kv_indptr_offset),pointer(spec.tail_offset));
    params.num_qo_heads=spec.query_heads;params.group_size=fi::uint_fastdiv(spec.query_heads/spec.kv_heads);
    params.q_stride_n=spec.query_heads*QK;params.q_stride_h=QK;
    params.window_left=spec.window_left;params.sm_scale=spec.scale*spec.key_scale;bind_plan(params,plan,integer);bind_output(params);
    status=launch_mask<QK,VO,true>(params,plan,spec,floating,stream);
  }else{
    using Base=fi::BatchPrefillRaggedParams<Q,KV,O,std::int32_t>;
    std::conditional_t<Split,std::conditional_t<Bf16Partials,Bf16MergeParams<Base,Q,VO>,MergeParams<Base,Q,VO>>,OutputParams<Base>> params;
    params.q=const_cast<Q*>(static_cast<const Q*>(q));params.k=const_cast<KV*>(static_cast<const KV*>(k));
    params.v=const_cast<KV*>(static_cast<const KV*>(v));params.o=native_output;
    params.q_indptr=pointer(spec.query_indptr_offset);params.kv_indptr=pointer(spec.kv_indptr_offset);
    params.num_qo_heads=spec.query_heads;params.num_kv_heads=spec.kv_heads;
    params.group_size=fi::uint_fastdiv(spec.query_heads/spec.kv_heads);
    params.q_stride_n=spec.query_heads*QK;params.q_stride_h=QK;
    params.k_stride_n=spec.kv_heads*QK;params.k_stride_h=QK;
    params.v_stride_n=spec.kv_heads*VO;params.v_stride_h=VO;
    params.window_left=spec.window_left;params.sm_scale=spec.scale*spec.key_scale;bind_plan(params,plan,integer);bind_output(params);
    status=launch_mask<QK,VO,false>(params,plan,spec,floating,stream);
  }
  return status;
}
template<class Q,class KV,unsigned QK,unsigned VO>
cudaError_t run_typed(const fi::PrefillPlanInfo &plan,BatchSpec spec,const void *q,const void *k,
                     const void *v,const void *metadata,void *output,void *floating,void *integer,cudaStream_t stream){
  if constexpr(sizeof(KV)==1){
    if(plan.split_kv)return run_output<Q,KV,QK,VO,true>(plan,spec,q,k,v,metadata,output,floating,integer,stream);
  }
  return run_output<Q,KV,QK,VO,false>(plan,spec,q,k,v,metadata,output,floating,integer,stream);
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
  if(bf16_partials_active(spec,std::uint32_t(plan.cta_tile_q),plan.split_kv))
    return run_output<__nv_bfloat16,__nv_fp8_e4m3,256,256,true,true>(plan,spec,q,k,v,metadata,output,floating,integer,stream);
  return spec.bfloat16?dispatch_kv<__nv_bfloat16>(plan,spec,q,k,v,metadata,output,floating,integer,stream):
                      dispatch_kv<half>(plan,spec,q,k,v,metadata,output,floating,integer,stream);
}
}
