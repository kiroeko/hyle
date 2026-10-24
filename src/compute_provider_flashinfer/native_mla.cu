#include "native_batch.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <flashinfer/attention/mla.cuh>
#include <flashinfer/attention/mla_params.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <new>

namespace hyle::compute::flashinfer::native {
namespace fi=::flashinfer;
std::size_t mla_state_bytes() noexcept{return sizeof(fi::MLAPlanInfo);}
void initialize_mla_state(void*p) noexcept{::new(p)fi::MLAPlanInfo{};}
void destroy_mla_state(void*p) noexcept{static_cast<fi::MLAPlanInfo*>(p)->~MLAPlanInfo();}
cudaError_t query_mla_workspace(BatchSpec s,const std::int32_t*q,const std::int32_t*k,
    const std::int32_t*lengths,cudaStream_t stream,BatchFacts&facts){
  facts={};if(!s.query_rows)return cudaSuccess;
  fi::MLAPlanInfo plan{};
  return fi::MLAPlan(nullptr,0,nullptr,nullptr,0,plan,const_cast<std::int32_t*>(q),
      const_cast<std::int32_t*>(k),const_cast<std::int32_t*>(lengths),s.requests,
      s.query_heads,512,s.causal,stream,true,&facts.float_bytes,&facts.int_bytes,sizeof(float));
}
cudaError_t plan_mla(void*state,BatchSpec s,const std::int32_t*q,const std::int32_t*k,
    const std::int32_t*lengths,void*f,std::size_t fb,void*i,void*h,std::size_t ib,
    cudaStream_t stream,BatchFacts&facts){
  auto&plan=*static_cast<fi::MLAPlanInfo*>(state);
  if(!s.query_rows){plan={};facts={};return cudaSuccess;}
  const auto result=fi::MLAPlan(f,fb,i,h,ib,plan,const_cast<std::int32_t*>(q),
      const_cast<std::int32_t*>(k),const_cast<std::int32_t*>(lengths),s.requests,
      s.query_heads,512,s.causal,stream,false,&facts.float_bytes,&facts.int_bytes,sizeof(float));
  facts.cta_tile_q=64;facts.padded_batch_size=std::uint32_t(plan.num_blks_x*plan.num_blks_y);
  facts.split_kv=false;
  if(result==cudaSuccess){
    const auto*works=reinterpret_cast<const std::int32_t*>(static_cast<const char*>(h)+plan.work_indptr_offset);
    const auto*partials=reinterpret_cast<const std::int32_t*>(static_cast<const char*>(h)+plan.partial_indptr_offset);
    for(std::int32_t work=0;work<works[plan.num_blks_y];++work)facts.split_kv|=partials[work]>=0;
  }
  return result;
}
template<class Q,class KV,unsigned Rope>cudaError_t run_typed(const fi::MLAPlanInfo&plan,BatchSpec s,
    const void*q,const void*cache,const void*metadata,void*o,void*f,void*i,cudaStream_t stream){
  fi::MLAParams<Q,KV,Q,std::int32_t,float> p{};
  p.q_nope=const_cast<Q*>(static_cast<const Q*>(q));p.q_pe=p.q_nope+512;
  p.ckv=const_cast<KV*>(static_cast<const KV*>(cache));p.kpe=p.ckv+512;
  p.final_o=static_cast<Q*>(o);p.sm_scale=s.scale;p.return_lse_base_on_e=false;
  // One physical compact row: CKV is both K-prefix and V, with the same scale.
  p.ckv_scale=s.key_scale;p.kpe_scale=s.key_scale;
  p.block_size=fi::uint_fastdiv(s.page_size);p.num_heads=fi::uint_fastdiv(s.query_heads);
  p.q_nope_stride_n=p.q_pe_stride_n=s.query_heads*(512+Rope);
  p.q_nope_stride_h=p.q_pe_stride_h=512+Rope;
  p.ckv_stride_page=p.kpe_stride_page=s.page_size*(512+Rope);
  p.ckv_stride_n=p.kpe_stride_n=512+Rope;p.o_stride_n=s.query_heads*512;p.o_stride_h=512;
  // Ragged indptr already contains global row offsets. The same original MLA
  // kernel reads that compact row directly, without a synthetic page table.
  p.linear_kv=!s.paged;
  p.kv_indices=s.paged?reinterpret_cast<std::int32_t*>(const_cast<char*>(static_cast<const char*>(metadata))+s.page_indices_offset):nullptr;
  const auto ptr=[&](std::int64_t offset){return reinterpret_cast<std::int32_t*>(static_cast<char*>(i)+offset);};
  p.q_indptr=ptr(plan.q_indptr_offset);p.kv_indptr=ptr(plan.kv_indptr_offset);
  p.partial_indptr=ptr(plan.partial_indptr_offset);
  p.merge_packed_offset_start=ptr(plan.merge_packed_offset_start_offset);
  p.merge_packed_offset_end=ptr(plan.merge_packed_offset_end_offset);
  p.merge_partial_packed_offset_start=ptr(plan.merge_partial_packed_offset_start_offset);
  p.merge_partial_packed_offset_end=ptr(plan.merge_partial_packed_offset_end_offset);
  p.merge_partial_stride=ptr(plan.merge_partial_stride_offset);
  p.q_len=ptr(plan.q_len_offset);p.kv_len=ptr(plan.kv_len_offset);p.q_start=ptr(plan.q_start_offset);
  p.kv_start=ptr(plan.kv_start_offset);p.kv_end=ptr(plan.kv_end_offset);p.work_indptr=ptr(plan.work_indptr_offset);
  p.partial_o=f?reinterpret_cast<float*>(static_cast<char*>(f)+plan.partial_o_offset):nullptr;
  p.partial_lse=f?reinterpret_cast<float*>(static_cast<char*>(f)+plan.partial_lse_offset):nullptr;
  if(s.causal)return fi::mla::BatchMLAPagedAttention<fi::MaskMode::kCausal,512,Rope>(
      p,std::uint32_t(plan.num_blks_x),std::uint32_t(plan.num_blks_y),stream);
  return fi::mla::BatchMLAPagedAttention<fi::MaskMode::kNone,512,Rope>(
      p,std::uint32_t(plan.num_blks_x),std::uint32_t(plan.num_blks_y),stream);
}
cudaError_t run_mla(const void*state,BatchSpec s,const void*q,const void*cache,
    const void*m,void*o,void*f,void*i,cudaStream_t stream){
  if(!s.query_rows)return cudaSuccess;
  if((!s.paged && s.page_size!=1) || s.kv_heads!=1 || s.value_head_dim!=512 ||
      (s.qk_head_dim!=512 && s.qk_head_dim!=576) ||
      (s.fp8 && (!s.bfloat16 || s.key_scale!=s.value_scale)))return cudaErrorInvalidValue;
  const auto&plan=*static_cast<const fi::MLAPlanInfo*>(state);
  if(s.fp8)return s.qk_head_dim==512?
      run_typed<__nv_bfloat16,__nv_fp8_e4m3,0>(plan,s,q,cache,m,o,f,i,stream):
      run_typed<__nv_bfloat16,__nv_fp8_e4m3,64>(plan,s,q,cache,m,o,f,i,stream);
  if(s.qk_head_dim==512)return s.bfloat16?run_typed<__nv_bfloat16,__nv_bfloat16,0>(plan,s,q,cache,m,o,f,i,stream):
      run_typed<half,half,0>(plan,s,q,cache,m,o,f,i,stream);
  return s.bfloat16?run_typed<__nv_bfloat16,__nv_bfloat16,64>(plan,s,q,cache,m,o,f,i,stream):
      run_typed<half,half,64>(plan,s,q,cache,m,o,f,i,stream);
}
} // namespace hyle::compute::flashinfer::native
