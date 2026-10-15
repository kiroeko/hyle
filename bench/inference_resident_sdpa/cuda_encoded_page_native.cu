#include "cuda_encoded_page_native.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <stdexcept>

// Namespace-only isolation prevents weak template/kernel coalescing with the
// candidate's separately linked, patched FlashInfer implementation.
#define flashinfer hyle_benchmark_original_flashinfer_69ff
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variants.cuh>
#undef flashinfer

namespace hyle::bench::resident::cuda_original {
namespace fi=::hyle_benchmark_original_flashinfer_69ff;
void *create_plan(){return new fi::PrefillPlanInfo;}
void destroy_plan(void *p) noexcept {delete static_cast<fi::PrefillPlanInfo *>(p);}
cudaError_t workspace(Spec s,const std::int32_t *q,const std::int32_t *k,cudaStream_t stream,Facts &f){
  return fi::PrefillPlanWorkspaceSize(f.floating,f.integer,const_cast<std::int32_t *>(q),
      const_cast<std::int32_t *>(k),s.query_rows,s.requests,s.query_heads,s.kv_heads,
      s.key_depth,s.value_depth,s.page_size,false,2,s.window,-1,false,0,0,stream,2);
}
cudaError_t plan(void *state,Spec s,const std::int32_t *q,const std::int32_t *k,
    void *floating,std::size_t fb,void *integer,void *host,std::size_t ib,cudaStream_t stream,Facts &f){
  auto &p=*static_cast<fi::PrefillPlanInfo *>(state);
  const auto status=fi::PrefillPlan(floating,fb,integer,host,ib,p,
      const_cast<std::int32_t *>(q),const_cast<std::int32_t *>(k),s.query_rows,s.requests,
      s.query_heads,s.kv_heads,s.key_depth,s.value_depth,s.page_size,false,2,
      s.window,-1,false,0,0,stream,2);
  f.tile=unsigned(p.cta_tile_q);f.padded=unsigned(p.padded_batch_size);f.split=p.split_kv;
  return status;
}
namespace {
template<unsigned K,unsigned V,fi::MaskMode Mask>
cudaError_t execute(const fi::PrefillPlanInfo &plan,Spec s,const void *q,const void *k,
    const void *v,const void *metadata,void *o,void *floating,void *integer,cudaStream_t stream){
  using T=__nv_bfloat16;
  fi::BatchPrefillPagedParams<T,T,T,std::int32_t> p;
  const auto mp=[&](std::size_t n){return reinterpret_cast<std::int32_t *>(
      const_cast<char *>(static_cast<const char *>(metadata))+n);};
  const auto ip=[&](std::int64_t n){return reinterpret_cast<std::int32_t *>(static_cast<char *>(integer)+n);};
  p.q=const_cast<T *>(static_cast<const T *>(q));p.o=static_cast<T *>(o);
  p.q_indptr=mp(s.query_offset);
  const std::int64_t ks[]{std::int64_t(s.page_size)*s.kv_heads*K,std::int64_t(s.kv_heads)*K,K,1};
  const std::int64_t vs[]{std::int64_t(s.page_size)*s.kv_heads*V,std::int64_t(s.kv_heads)*V,V,1};
  p.paged_kv=fi::paged_kv_t<T,std::int32_t>(s.kv_heads,s.page_size,K,s.requests,fi::QKVLayout::kNHD,
      const_cast<T *>(static_cast<const T *>(k)),const_cast<T *>(static_cast<const T *>(v)),
      ks,vs,mp(s.indices_offset),mp(s.kv_offset),mp(s.tails_offset));
  p.num_qo_heads=s.query_heads;p.group_size=fi::uint_fastdiv(s.query_heads/s.kv_heads);
  p.q_stride_n=s.query_heads*K;p.q_stride_h=K;p.window_left=s.window;p.sm_scale=s.scale;
  p.request_indices=ip(plan.request_indices_offset);p.qo_tile_indices=ip(plan.qo_tile_indices_offset);
  p.kv_tile_indices=ip(plan.kv_tile_indices_offset);p.o_indptr=ip(plan.o_indptr_offset);
  p.kv_chunk_size_ptr=ip(plan.kv_chunk_size_ptr_offset);p.padded_batch_size=unsigned(plan.padded_batch_size);
  p.max_total_num_rows=unsigned(plan.total_num_rows);
  if(plan.split_kv)p.merge_indptr=ip(plan.merge_indptr_offset);
  auto *partial=plan.split_kv?reinterpret_cast<T *>(static_cast<char *>(floating)+plan.v_offset):nullptr;
  auto *scores=plan.split_kv?reinterpret_cast<float *>(static_cast<char *>(floating)+plan.s_offset):nullptr;
  using Variant=fi::DefaultAttention<false,true,false,false>;
  switch(plan.cta_tile_q){
#define HYLE_ORIGINAL_TILE(N) case N:return fi::BatchPrefillWithPagedKVCacheDispatched<N,K,V,fi::PosEncodingMode::kNone,false,Mask,Variant>(p,partial,scores,false,stream)
    HYLE_ORIGINAL_TILE(16);HYLE_ORIGINAL_TILE(64);HYLE_ORIGINAL_TILE(128);
#undef HYLE_ORIGINAL_TILE
  }
  throw std::runtime_error("original BF16 peer CTA outside finite dispatch");
}
__global__ void decode_rows(const __nv_fp8_e4m3 *input,__nv_bfloat16 *output,
    const std::uint32_t *valid,unsigned pages,unsigned page_size,unsigned row_width,float scale){
  const auto total=std::uint64_t(pages)*page_size*row_width;
  for(auto i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<total;i+=std::uint64_t(gridDim.x)*blockDim.x){
    const auto row=i/row_width,page=row/page_size;
    // No read from an unborrowed page or a logically unused tail row.
    if(row%page_size<valid[page])output[i]=__float2bfloat16(float(input[i])*scale);
  }
}
}
cudaError_t run(const void *state,Spec s,const void *q,const void *k,const void *v,
    const void *m,void *o,void *f,void *i,cudaStream_t stream){
  const auto &p=*static_cast<const fi::PrefillPlanInfo *>(state);
  if(s.key_depth==256&&s.value_depth==256&&s.causal)
    return execute<256,256,fi::MaskMode::kCausal>(p,s,q,k,v,m,o,f,i,stream);
  if(s.key_depth==64&&s.value_depth==128&&!s.causal&&s.window==-1)
    return execute<64,128,fi::MaskMode::kNone>(p,s,q,k,v,m,o,f,i,stream);
  return cudaErrorNotSupported;
}
cudaError_t decode(const void *input,void *output,const std::uint32_t *valid,unsigned pages,
    unsigned page_size,unsigned heads,unsigned depth,float scale,cudaStream_t stream){
  const auto count=std::uint64_t(pages)*page_size*heads*depth;
  const auto blocks=unsigned((std::min)(std::uint64_t(65535),(count+255)/256));
  if(!blocks)return cudaSuccess;
  decode_rows<<<blocks,256,0,stream>>>(static_cast<const __nv_fp8_e4m3 *>(input),
      static_cast<__nv_bfloat16 *>(output),valid,pages,page_size,heads*depth,scale);
  return cudaGetLastError();
}
}
